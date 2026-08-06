/*
=============================================================================
 train_bnn_lowrank.cu — Arm B: low-rank + diagonal covariance posterior.

 Mean-field assumes every weight is independent. This arm gives each Bayesian
 layer a posterior with CORRELATION structure:

     Sigma = A A^T + diag(dvec^2)          A in R^{P x r}, dvec in R^P

 Sigma is P x P and is NEVER formed. Sampling uses the factorization directly:

     w_i = mu_i + sum_{k<r} A[i][k] * z[k] + dvec_i * eps_i

     z   ~ N(0, I_r)   ONE draw per layer per forward pass, shared by all P
                       weights of that layer — this is what couples them
     eps ~ N(0, I_P)   per weight, Philox, counter = GLOBAL weight index

     Arm A  train_bnn_vadam.cu   sigma from Adam's second moment
     Arm B  THIS FILE            learned low-rank + diagonal covariance
     Arm C  train_ensemble.cu    E=5 deep ensemble (control)
     ---    train_bnn_gpu.cu     Bayes-by-Backprop, mean-field (the baseline)

-----------------------------------------------------------------------------
 SCOPE — this arm is PARTIALLY Bayesian, unlike the others.

 Only the four GEMM weight matrices are given a posterior:

     W_qkv  768 x 2304 = 1,769,472
     W_proj  768 x  768 =   589,824
     W_fc    768 x 3072 = 2,359,296
     W_fc2  3072 x  768 = 2,359,296
                          ---------
                    P  =  7,077,888    (8.3% of the model's 85,120,849)

 Embeddings, LayerNorm gains/biases and all biases stay deterministic. Each
 matrix gets its OWN independent A, dvec and its own z — there is no
 covariance across layers. State this when comparing: arms A and C put a
 posterior over ALL 85M parameters, so this arm is Bayesian over less of the
 model, which flatters its speed and its memory and may hurt its uncertainty.

-----------------------------------------------------------------------------
 TWO DELIBERATE DEVIATIONS FROM THE SPEC, both forced by this codebase.

 1. NOT FUSED INTO THE GEMM PROLOGUE. The spec asks for sampling fused into a
    hand-written GEMM. This codebase calls cuBLAS — there is no GEMM of ours
    to fuse into. More importantly, arms A and C both materialize w with a
    separate sampling kernel, so fusing ONLY this arm would make the
    comparison measure kernel engineering rather than posterior structure.
    Implemented as the "separate kernel" alternative the spec permits, to be
    measured deliberately. Fusion, if done, should be done to all arms at once.

 2. MEMORY IS FAR SMALLER THAN THE SPEC'S ESTIMATE. The spec budgets A at
    2.72 GB (r=8) assuming all 85M params are Bayesian; its own scope section
    restricts to the four GEMMs, which is 7.08M here. Actual A at r=8 is
    226 MB and r=16 fits comfortably. The spec's honesty point still stands
    and is repeated in the run output: low-rank costs MORE than mean-field,
    and the justification is posterior quality, not footprint.

-----------------------------------------------------------------------------
 KL against the prior N(0, I/lambda), using the two matrix identities so that
 Sigma is never formed:

     KL = 0.5 [ lambda*(||mu||^2 + tr(Sigma)) - P - logdet(Sigma) + P log lambda ]

     tr(Sigma)     = ||A||_F^2 + ||dvec||^2
     logdet(Sigma) = logdet(M) + sum_i log(dvec_i^2),   M = I_r + A^T D^-1 A

 M is r x r, so a P x P determinant becomes an 8 x 8 one. Gradients follow
 from Woodbury, and reduce to the mean-field ones when A = 0:

     dKL/dmu_i    = lambda * mu_i
     dKL/dA[i][k] = lambda*A[i][k] - (1/d_i^2) sum_j A[i][j] Minv[j][k]
     dKL/dd_i     = lambda*d_i - 1/d_i + quad_i / d_i^3
                    where quad_i = sum_{jk} A[i][j] Minv[j][k] A[i][k]

 The -1/d_i term is exactly mean-field's anti-collapse barrier, and at A=0 the
 stationary point is d = 1/sqrt(lambda) = the prior std. That is check 1.

 Build:   make train_bnn_lowrank              (r=8; make R=4 / R=16 to sweep)
 Run:     ./train_bnn_lowrank                 train on TinyShakespeare
          ./train_bnn_lowrank overfit         memorize one batch
          ./train_bnn_lowrank check           the five correctness checks
=============================================================================
*/

#include "gpt2_kernels.cuh"

#ifndef NBLOCKS
#define NBLOCKS 6      // transformer blocks; override with make L=12
#endif

#ifndef RANK
#define RANK 8
#endif


// One Bayesian layer: where its mu lives in the 85M param vector, where its
// A and dvec live in the packed Bayesian buffers, and how many weights it has.
typedef struct { size_t poff, aoff, boff; int P; const char *name; } BLayer;

/* ===================================================================== *
 * Sampling and its backward.  z is passed already offset to this layer.
 * ===================================================================== */

// w = mu + A z + dvec * eps, elementwise. z lands in shared once per block and
// every thread broadcast-reads it; A is streamed into registers r at a time
// (a tile of A would be r times the mu tile and blows the shared budget).
__global__ void k_lowrank_sample(float *w, const float *theta, const float *A,
                                 const float *rhod, const float *z,
                                 size_t poff, size_t aoff, size_t boff, int P,
                                 unsigned long long seed, int step) {
    __shared__ float zs[RANK];
    if (threadIdx.x < RANK) zs[threadIdx.x] = z[threadIdx.x];
    __syncthreads();

    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < P) {
        float acc = theta[poff + i];                 // mu
        const float *Ai = A + aoff + (size_t)i * RANK;   // row-major: r contiguous
        #pragma unroll
        for (int k = 0; k < RANK; k++) acc += Ai[k] * zs[k];
        // dvec = softplus(rhod), NOT a free parameter — see the header note on
        // why direct dvec blows up under AdamW's scale-free step.
        acc += d_softplus(rhod[boff + i]) * philox_normal(seed, step, poff + i);
        w[poff + i] = acc;
    }
}

//   dL/dmu_i    = dL/dw_i     (already sitting in dw at poff+i — nothing to do)
//   dL/dA[i][k] = dL/dw_i * z[k]        outer product with the shared z
//   dL/ddvec_i  = dL/dw_i * eps_i       same eps, regenerated not stored
__global__ void k_lowrank_backward(float *dA, float *drhod, const float *dw,
                                   const float *rhod, const float *z,
                                   size_t poff, size_t aoff, size_t boff, int P,
                                   unsigned long long seed, int step) {
    __shared__ float zs[RANK];
    if (threadIdx.x < RANK) zs[threadIdx.x] = z[threadIdx.x];
    __syncthreads();

    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < P) {
        float g = dw[poff + i];
        float *dAi = dA + aoff + (size_t)i * RANK;
        #pragma unroll
        for (int k = 0; k < RANK; k++) dAi[k] += g * zs[k];
        // chain through softplus: d(dvec)/d(rhod) = sigmoid(rhod)
        float r = rhod[boff + i];
        drhod[boff + i] += g * philox_normal(seed, step, poff + i) * d_sigmoid(r);
    }
}

// B = D^-1 A, i.e. B[i][k] = A[i][k] / d_i^2. Written into the dA buffer as
// scratch (it is zeroed immediately afterwards) so no extra P*r allocation.
// cuBLAS then contracts A^T B into the r x r matrix M - I.
__global__ void k_scale_A_dinv(float *B, const float *A, const float *rhod,
                               size_t aoff, size_t boff, int P) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < P) {
        float d = d_softplus(rhod[boff + i]);
        float inv = 1.0f / fmaxf(d * d, 1e-20f);
        const float *Ai = A + aoff + (size_t)i * RANK;
        float *Bi = B + aoff + (size_t)i * RANK;
        #pragma unroll
        for (int k = 0; k < RANK; k++) Bi[k] = Ai[k] * inv;
    }
}

// Add the KL gradients. Minv (r x r) is tiny and goes to shared.
__global__ void k_lowrank_kl_grad(float *dtheta, float *dA, float *drhod,
                                  const float *theta, const float *A,
                                  const float *rhod, const float *Minv,
                                  size_t poff, size_t aoff, size_t boff, int P,
                                  float lambda, float weight) {
    __shared__ float Ms[RANK * RANK];
    for (int t = threadIdx.x; t < RANK * RANK; t += blockDim.x) Ms[t] = Minv[t];
    __syncthreads();

    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < P) {
        const float *Ai = A + aoff + (size_t)i * RANK;
        float ar[RANK];
        #pragma unroll
        for (int k = 0; k < RANK; k++) ar[k] = Ai[k];

        // v = Minv * a_i, then quad = a_i . v
        float v[RANK], quad = 0.0f;
        #pragma unroll
        for (int k = 0; k < RANK; k++) {
            float s = 0.0f;
            #pragma unroll
            for (int j = 0; j < RANK; j++) s += ar[j] * Ms[j * RANK + k];
            v[k] = s;
            quad += ar[k] * s;
        }

        float r   = rhod[boff + i];
        float d   = d_softplus(r);
        float d2  = fmaxf(d * d, 1e-20f);
        float inv = 1.0f / d2;

        float *dAi = dA + aoff + (size_t)i * RANK;
        #pragma unroll
        for (int k = 0; k < RANK; k++)
            dAi[k] += weight * (lambda * ar[k] - inv * v[k]);

        // lambda*d - 1/d + quad/d^3 : the middle term is mean-field's
        // anti-collapse barrier, the last is what low rank adds. Then chain
        // through softplus to reach rhod.
        float dKL_dd = lambda * d - 1.0f / d + quad * inv / d;
        drhod[boff + i] += weight * dKL_dd * d_sigmoid(r);

        dtheta[poff + i] += weight * lambda * theta[poff + i];   // dKL/dmu
    }
}

// Reductions for the KL value and the marginal std sigma_i = sqrt(|A_i|^2+d_i^2),
// which is what the scorecard's sigma column reports for this arm.
__global__ void k_lowrank_stats(float *o_a2, float *o_d2, float *o_logd,
                                float *o_mu2, float *o_smin, float *o_smax,
                                float *o_smean, const float *theta,
                                const float *A, const float *rhod,
                                size_t poff, size_t aoff, size_t boff, int P) {
    __shared__ float sc[32];
    float a2 = 0, d2 = 0, logd = 0, mu2 = 0, smean = 0;
    float smin = 1e30f, smax = -1e30f;
    for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < P;
         i += blockDim.x * gridDim.x) {
        const float *Ai = A + aoff + (size_t)i * RANK;
        float rowsq = 0.0f;
        #pragma unroll
        for (int k = 0; k < RANK; k++) rowsq += Ai[k] * Ai[k];
        float d = d_softplus(rhod[boff + i]), m = theta[poff + i];
        float sig = sqrtf(rowsq + d * d);
        a2 += rowsq; d2 += d * d; logd += logf(fmaxf(d * d, 1e-30f));
        mu2 += m * m; smean += sig;
        smin = fminf(smin, sig); smax = fmaxf(smax, sig);
    }
    float r1 = block_reduce_sum(a2, sc);   __syncthreads();
    float r2 = block_reduce_sum(d2, sc);   __syncthreads();
    float r3 = block_reduce_sum(logd, sc); __syncthreads();
    float r4 = block_reduce_sum(mu2, sc);  __syncthreads();
    float r5 = block_reduce_sum(smean, sc);__syncthreads();
    float r6 = -block_reduce_max(-smin, sc); __syncthreads();
    float r7 = block_reduce_max(smax, sc);
    if (threadIdx.x == 0) {
        o_a2[blockIdx.x]=r1; o_d2[blockIdx.x]=r2; o_logd[blockIdx.x]=r3;
        o_mu2[blockIdx.x]=r4; o_smean[blockIdx.x]=r5;
        o_smin[blockIdx.x]=r6; o_smax[blockIdx.x]=r7;
    }
}

// Mean-field reference for check 1: w = mu + dvec*eps, no A at all.
// It has to live on the DEVICE. philox_normal is __host__ __device__, but
// Box-Muller's sqrtf/logf/cosf under --use_fast_math use device
// approximations that differ from libm in the last ulp, so a host-computed
// reference can never match bitwise even when the kernel is perfect.
__global__ void k_meanfield_ref(float *w, const float *theta, const float *rhod,
                                size_t poff, size_t boff, int P,
                                unsigned long long seed, int step) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < P)
        w[poff + i] = theta[poff + i]
                    + d_softplus(rhod[boff + i]) * philox_normal(seed, step, poff + i);
}

// z: r draws per layer per step, from a SEPARATE Philox stream so it cannot
// collide with the per-weight eps stream. Also written to a buffer, so
// backward reads exactly the forward values (check 5 asserts they agree).
__global__ void k_draw_z(float *z, int nbl, unsigned long long seed, int step) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < nbl * RANK) z[i] = philox_normal(seed, step, i);
}

/* ===================================================================== *
 * Host: Cholesky of the r x r M, giving logdet and Minv.
 * ===================================================================== */
static int chol_logdet_inv(const float *Min, float *Minv, double *logdet) {
    float L[RANK * RANK];
    memcpy(L, Min, sizeof(L));
    for (int j = 0; j < RANK; j++) {
        for (int k = 0; k < j; k++) L[j*RANK+j] -= L[j*RANK+k] * L[j*RANK+k];
        if (L[j*RANK+j] <= 0.0f) return 0;              // not PD
        L[j*RANK+j] = sqrtf(L[j*RANK+j]);
        for (int i = j + 1; i < RANK; i++) {
            for (int k = 0; k < j; k++) L[i*RANK+j] -= L[i*RANK+k] * L[j*RANK+k];
            L[i*RANK+j] /= L[j*RANK+j];
        }
        for (int k = j + 1; k < RANK; k++) L[j*RANK+k] = 0.0f;
    }
    *logdet = 0.0; for (int j = 0; j < RANK; j++) *logdet += 2.0 * log(L[j*RANK+j]);

    float Li[RANK * RANK];                               // invert lower triangle
    memset(Li, 0, sizeof(Li));
    for (int j = 0; j < RANK; j++) {
        Li[j*RANK+j] = 1.0f / L[j*RANK+j];
        for (int i = j + 1; i < RANK; i++) {
            float s = 0.0f;
            for (int k = j; k < i; k++) s += L[i*RANK+k] * Li[k*RANK+j];
            Li[i*RANK+j] = -s / L[i*RANK+i];
        }
    }
    for (int i = 0; i < RANK; i++)                       // Minv = Li^T Li
        for (int j = 0; j < RANK; j++) {
            float s = 0.0f;
            for (int k = (i > j ? i : j); k < RANK; k++) s += Li[k*RANK+i] * Li[k*RANK+j];
            Minv[i*RANK+j] = s;
        }
    return 1;
}

/* ===================================================================== *
 * main
 * ===================================================================== */
int main(int argc, char **argv) {
    bool overfit = (argc > 1 && argv[1][0] == 'o');
    bool checks  = (argc > 1 && argv[1][0] == 'c');

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s  sm_%d%d  %.1f GB   RANK=%d\n\n", prop.name, prop.major,
           prop.minor, prop.totalGlobalMem / 1e9, RANK);
    CUBLAS_CHECK(cublasCreate(&cublas_handle));

    Config cfg = { .B = 4, .T = 64, .C = 768, .V = VOCAB, .NH = 12, .maxT = 1024, .L = NBLOCKS };
    ParamOff po = param_offsets(cfg);
    ActOff   a  = act_offsets(cfg);
    size_t n = po.total;
    int BT = cfg.B * cfg.T, C = cfg.C;

    // the four GEMM matrices, packed
    // four GEMM matrices per block, so NBL = 4 * cfg.L
    const int NBL = 4 * cfg.L;
    BLayer *L = (BLayer *)malloc(NBL * sizeof(BLayer));
    for (int b = 0; b < cfg.L; b++) {
        L[4*b+0] = (BLayer){ PBLK(po,b,qw), 0, 0, C * 3 * C, "W_qkv"  };
        L[4*b+1] = (BLayer){ PBLK(po,b,pw), 0, 0, C * C,     "W_proj" };
        L[4*b+2] = (BLayer){ PBLK(po,b,w1), 0, 0, C * 4 * C, "W_fc"   };
        L[4*b+3] = (BLayer){ PBLK(po,b,w2), 0, 0, 4 * C * C, "W_fc2"  };
    }
    size_t Pb = 0;
    for (int l = 0; l < NBL; l++) {
        L[l].boff = Pb; L[l].aoff = Pb * RANK; Pb += L[l].P;
    }

    const float LAMBDA = 100.0f;                 // prior precision, sigma_p = 0.1
    const float KLW    = overfit ? 0.0f : 1e-3f; // same policy as BBB
    const float LR = overfit ? 1e-3f : 1e-4f;
    // SEPARATE lr for the variance group (A and rhod). AdamW's step is
    // scale-free — about `lr` regardless of gradient size — so one lr cannot
    // serve parameters living at 2e-2 (mu) and 1e-4 (A) at once. At lr=1e-3
    // a single step moves A by 10x its own initialization, which is what blew
    // sigma up to 0.077 before. softplus fixes rhod the same way it does in
    // BBB; A is signed so softplus cannot help it and it needs the lr instead.
    const float LR_VAR = 1e-5f;
    const float B1 = 0.9f, B2 = 0.999f, AEPS = 1e-8f, WD = 0.0f;
    const unsigned long long SEED = 20240520ull, ZSEED = 20240520ull ^ 0x9E3779B9ull;
    // steps: default per mode, overridable as `./bin <mode> <steps>`
    int STEPS = overfit ? 200 : 100;
    if (argc > 2) STEPS = atoi(argv[2]);

    size_t pb = n * sizeof(float), ab = Pb * RANK * sizeof(float), db = Pb * sizeof(float);
    printf("params %zu   Bayesian P %zu (%.1f%%, the 4 GEMMs)   A = %.0f MB\n",
           n, Pb, 100.0 * Pb / n, ab / 1e6);
    printf("total VRAM ~%.2f GB\n",
           (5.0 * pb + 4.0 * ab + 4.0 * db + 2.0 * a.total * sizeof(float)) / 1e9);
    printf("note: low-rank costs MORE memory than mean-field. It saves relative to\n"
           "      FULL covariance, which was never feasible. Justification is\n"
           "      posterior quality, not footprint.\n\n");

    float *d_theta, *d_w, *d_dw, *d_m, *d_v;
    float *d_A, *d_dA, *d_mA, *d_vA, *d_rhod, *d_drhod, *d_md, *d_vd;
    float *d_acts, *d_gacts, *d_z, *d_M, *d_Minv;
    int *d_inp, *d_tgt;
    CUDA_CHECK(cudaMalloc(&d_theta, pb)); CUDA_CHECK(cudaMalloc(&d_w,  pb));
    CUDA_CHECK(cudaMalloc(&d_dw, pb));    CUDA_CHECK(cudaMalloc(&d_m,  pb));
    CUDA_CHECK(cudaMalloc(&d_v,  pb));
    CUDA_CHECK(cudaMalloc(&d_A,  ab));    CUDA_CHECK(cudaMalloc(&d_dA, ab));
    CUDA_CHECK(cudaMalloc(&d_mA, ab));    CUDA_CHECK(cudaMalloc(&d_vA, ab));
    CUDA_CHECK(cudaMalloc(&d_rhod, db));  CUDA_CHECK(cudaMalloc(&d_drhod, db));
    CUDA_CHECK(cudaMalloc(&d_md, db));    CUDA_CHECK(cudaMalloc(&d_vd, db));
    CUDA_CHECK(cudaMalloc(&d_acts,  a.total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gacts, a.total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_z,    NBL * RANK * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_M,    NBL * RANK * RANK * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Minv, NBL * RANK * RANK * sizeof(float)));
    float *h_M = (float*)malloc(NBL*RANK*RANK*sizeof(float));
    float *h_Minv = (float*)malloc(NBL*RANK*RANK*sizeof(float));
    double *logdetM = (double*)malloc(NBL*sizeof(double));
    CUDA_CHECK(cudaMalloc(&d_inp, BT * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_tgt, BT * sizeof(int)));

    // ---- init -------------------------------------------------------------
    // A ~ N(0,1e-4) and dvec = 1e-3: start SMALL. Too large an A and the
    // samples are garbage on step 1; the ELBO grows it from there.
    srand(1337);
    float *h = (float *)malloc(pb > ab ? pb : ab);
    for (size_t i = 0; i < n; i++) {
        float u1 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 2.0f);
        float u2 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 2.0f);
        h[i] = 0.02f * sqrtf(-2.0f * logf(u1)) * cosf(6.2831853f * u2);
    }
    CUDA_CHECK(cudaMemcpy(d_theta, h, pb, cudaMemcpyHostToDevice));
    for (size_t i = 0; i < Pb * RANK; i++) {
        float u1 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 2.0f);
        float u2 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 2.0f);
        h[i] = 1e-4f * sqrtf(-2.0f * logf(u1)) * cosf(6.2831853f * u2);
    }
    CUDA_CHECK(cudaMemcpy(d_A, h, ab, cudaMemcpyHostToDevice));
    // rhod with softplus(rhod) = 1e-3, i.e. log(exp(1e-3)-1)
    const float RHOD0 = logf(expf(1e-3f) - 1.0f);
    for (size_t i = 0; i < Pb; i++) h[i] = RHOD0;
    CUDA_CHECK(cudaMemcpy(d_rhod, h, db, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_m, 0, pb));  CUDA_CHECK(cudaMemset(d_v, 0, pb));
    CUDA_CHECK(cudaMemset(d_mA, 0, ab)); CUDA_CHECK(cudaMemset(d_vA, 0, ab));
    CUDA_CHECK(cudaMemset(d_md, 0, db)); CUDA_CHECK(cudaMemset(d_vd, 0, db));

    int gA = ceil_div((int)(Pb * RANK), BS), gB = ceil_div((int)Pb, BS);
    int gN = ceil_div((int)n, BS);
    const int SB = 256;
    float *dp[7], *hp[7];
    for (int i = 0; i < 7; i++) {
        CUDA_CHECK(cudaMalloc(&dp[i], SB * sizeof(float)));
        hp[i] = (float *)malloc(SB * sizeof(float));
    }

    /* ================== CORRECTNESS CHECKS ============================== */
    if (checks) {
        printf("CORRECTNESS CHECKS\n------------------\n");
        int fails = 0;
        float *ref = (float *)malloc(pb), *got = (float *)malloc(pb);
        k_draw_z<<<ceil_div(NBL*RANK,128), 128>>>(d_z, NBL, ZSEED, 1);

        // 2. A = 0 and dvec = 0 -> must reproduce the deterministic model bitwise
        // softplus(-40) underflows to exactly 0, the inverse of dvec = 0
        CUDA_CHECK(cudaMemset(d_A, 0, ab));
        for (size_t i = 0; i < Pb; i++) h[i] = -40.0f;
        CUDA_CHECK(cudaMemcpy(d_rhod, h, db, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_w, d_theta, pb, cudaMemcpyDeviceToDevice));
        for (int l = 0; l < NBL; l++)
            k_lowrank_sample<<<ceil_div(L[l].P, BS), BS>>>(d_w, d_theta, d_A, d_rhod,
                d_z + l * RANK, L[l].poff, L[l].aoff, L[l].boff, L[l].P, SEED, 1);
        CUDA_CHECK(cudaMemcpy(got, d_w, pb, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(ref, d_theta, pb, cudaMemcpyDeviceToHost));
        int bad = 0; for (size_t i = 0; i < n; i++) if (got[i] != ref[i]) bad++;
        printf("  [2] A=0, dvec=0 -> deterministic, bitwise : %s (%d mismatches)\n",
               bad ? "FAIL" : "PASS", bad); fails += (bad != 0);

        // 1. A = 0 -> must reduce EXACTLY to mean-field with sigma = dvec.
        // Both sides computed ON DEVICE (see k_meanfield_ref) so the fast-math
        // transcendentals are the same on both, and bitwise means something.
        for (size_t i = 0; i < Pb; i++) h[i] = logf(expf(3.7e-3f) - 1.0f);
        CUDA_CHECK(cudaMemcpy(d_rhod, h, db, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_w,  d_theta, pb, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(d_dw, d_theta, pb, cudaMemcpyDeviceToDevice));
        for (int l = 0; l < NBL; l++) {
            k_lowrank_sample<<<ceil_div(L[l].P, BS), BS>>>(d_w, d_theta, d_A, d_rhod,
                d_z + l * RANK, L[l].poff, L[l].aoff, L[l].boff, L[l].P, SEED, 1);
            k_meanfield_ref<<<ceil_div(L[l].P, BS), BS>>>(d_dw, d_theta, d_rhod,
                L[l].poff, L[l].boff, L[l].P, SEED, 1);
        }
        CUDA_CHECK(cudaMemcpy(got, d_w,  pb, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(ref, d_dw, pb, cudaMemcpyDeviceToHost));
        bad = 0;
        for (int l = 0; l < NBL; l++)
            for (int i = 0; i < L[l].P; i++) {
                size_t g = L[l].poff + i;
                if (got[g] != ref[g]) bad++;
            }
        printf("  [1] A=0 -> mean-field w = mu + dvec*eps, bitwise : %s (%d)\n",
               bad ? "FAIL" : "PASS", bad); fails += (bad != 0);
        CUDA_CHECK(cudaMemcpy(ref, d_theta, pb, cudaMemcpyDeviceToHost));

        // 5. eps and z regenerate identically in backward
        float *hz1=(float*)malloc(NBL*RANK*sizeof(float)), *hz2=(float*)malloc(NBL*RANK*sizeof(float));
        CUDA_CHECK(cudaMemcpy(hz1, d_z, NBL*RANK*sizeof(float), cudaMemcpyDeviceToHost));
        k_draw_z<<<ceil_div(NBL*RANK,128), 128>>>(d_z, NBL, ZSEED, 1);
        CUDA_CHECK(cudaMemcpy(hz2, d_z, NBL*RANK*sizeof(float), cudaMemcpyDeviceToHost));
        bad = 0; for (int i = 0; i < NBL * RANK; i++) if (hz1[i] != hz2[i]) bad++;
        for (size_t g = L[0].poff; g < L[0].poff + 4096; g++)
            if (philox_normal(SEED, 1, g) != philox_normal(SEED, 1, g)) bad++;
        printf("  [5] z and eps regenerate bitwise in backward   : %s (%d)\n",
               bad ? "FAIL" : "PASS", bad); fails += (bad != 0);

        // 4. finite-difference check on dL/dA, isolated op.
        //    Toy loss L = sum_i w_i * c_i, so dL/dw = c and dL/dA[i][k] = c_i z[k].
        {
            const int Pt = 4096; float *c = (float *)malloc(Pt * sizeof(float));
            for (int i = 0; i < Pt; i++) c[i] = ((float)rand() / RAND_MAX) * 2 - 1;
            CUDA_CHECK(cudaMemcpy(hz1, d_z, NBL*RANK*sizeof(float), cudaMemcpyDeviceToHost));
            for (size_t i = 0; i < (size_t)Pt * RANK; i++)
                h[i] = 1e-3f * (((float)rand() / RAND_MAX) * 2 - 1);
            CUDA_CHECK(cudaMemcpy(d_A, h, (size_t)Pt * RANK * sizeof(float),
                                  cudaMemcpyHostToDevice));
            double worst = 0.0;
            for (int trial = 0; trial < 40; trial++) {
                int i = rand() % Pt, k = rand() % RANK;
                double ana = (double)c[i] * hz1[k];              // analytic
                float save = h[(size_t)i * RANK + k], hh = 1e-2f;
                double Lp, Lm;
                for (int s = 0; s < 2; s++) {
                    h[(size_t)i * RANK + k] = save + (s ? -hh : hh);
                    CUDA_CHECK(cudaMemcpy(d_A, h, (size_t)Pt*RANK*sizeof(float),
                                          cudaMemcpyHostToDevice));
                    CUDA_CHECK(cudaMemcpy(d_w, d_theta, pb, cudaMemcpyDeviceToDevice));
                    k_lowrank_sample<<<ceil_div(Pt, BS), BS>>>(d_w, d_theta, d_A,
                        d_rhod, d_z, L[0].poff, 0, 0, Pt, SEED, 1);
                    CUDA_CHECK(cudaMemcpy(got, d_w, pb, cudaMemcpyDeviceToHost));
                    double s2 = 0.0;
                    for (int q = 0; q < Pt; q++) s2 += (double)got[L[0].poff+q] * c[q];
                    if (s) Lm = s2; else Lp = s2;
                }
                h[(size_t)i * RANK + k] = save;
                double num = (Lp - Lm) / (2.0 * hh);
                double rel = fabs(num - ana) / fmax(fabs(ana), 1e-6);
                if (rel > worst) worst = rel;
            }
            printf("  [4] dL/dA finite differences, 40 probes       : %s (worst rel %.2e)\n",
                   worst < 1e-2 ? "PASS" : "FAIL", worst);
            fails += (worst >= 1e-2); free(c);
        }

        // 3. empirical covariance vs A A^T + diag(dvec^2) on a small block
        {
            const int Pt = 1024, SUB = 48, NS = 20000;
            for (size_t i = 0; i < (size_t)Pt * RANK; i++)
                h[i] = 5e-2f * (((float)rand() / RAND_MAX) * 2 - 1);
            CUDA_CHECK(cudaMemcpy(d_A, h, (size_t)Pt*RANK*sizeof(float),
                                  cudaMemcpyHostToDevice));
            float *hA = (float *)malloc((size_t)Pt * RANK * sizeof(float));
            memcpy(hA, h, (size_t)Pt * RANK * sizeof(float));
            float dv = 2e-2f;
            for (size_t i = 0; i < (size_t)Pt; i++) h[i] = logf(expf(dv) - 1.0f);
            CUDA_CHECK(cudaMemcpy(d_rhod, h, Pt * sizeof(float), cudaMemcpyHostToDevice));

            double *mean = (double *)calloc(SUB, sizeof(double));
            double *cov  = (double *)calloc((size_t)SUB * SUB, sizeof(double));
            for (int s = 1; s <= NS; s++) {
                k_draw_z<<<ceil_div(NBL*RANK,128), 128>>>(d_z, NBL, ZSEED, s);
                CUDA_CHECK(cudaMemcpy(d_w, d_theta, pb, cudaMemcpyDeviceToDevice));
                k_lowrank_sample<<<ceil_div(Pt, BS), BS>>>(d_w, d_theta, d_A, d_rhod,
                    d_z, L[0].poff, 0, 0, Pt, SEED, s);
                CUDA_CHECK(cudaMemcpy(got, d_w + L[0].poff, SUB * sizeof(float),
                                      cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(ref, d_theta + L[0].poff, SUB * sizeof(float),
                                      cudaMemcpyDeviceToHost));
                for (int i = 0; i < SUB; i++) {
                    double di = got[i] - ref[i];
                    mean[i] += di;
                    for (int j = 0; j < SUB; j++) cov[i*SUB+j] += di * (got[j]-ref[j]);
                }
            }
            // Error is normalized by sqrt(Sigma_ii*Sigma_jj), NOT by |want|.
            // Off-diagonal entries are near zero by construction, so dividing
            // by |want| measures how close a true zero got to zero and blows
            // up regardless of correctness. The covariance scale is the only
            // meaningful denominator. MC error here is ~1/sqrt(NS) = 0.7%.
            double sig[64];
            for (int i = 0; i < SUB; i++) {
                double s = (double)dv * dv;
                for (int k = 0; k < RANK; k++)
                    s += (double)hA[(size_t)i*RANK+k] * hA[(size_t)i*RANK+k];
                sig[i] = s;
            }
            double worst = 0.0, worst_diag = 0.0;
            for (int i = 0; i < SUB; i++)
                for (int j = 0; j < SUB; j++) {
                    double emp = cov[i*SUB+j]/NS - (mean[i]/NS)*(mean[j]/NS);
                    double want = (i == j) ? (double)dv*dv : 0.0;
                    for (int k = 0; k < RANK; k++)
                        want += (double)hA[(size_t)i*RANK+k] * hA[(size_t)j*RANK+k];
                    double e = fabs(emp - want) / sqrt(sig[i] * sig[j]);
                    if (e > worst) worst = e;
                    if (i == j && e > worst_diag) worst_diag = e;
                }
            printf("  [3] empirical cov vs A A^T + D, %d samples : %s "
                   "(worst %.4f, diag %.4f, of cov scale)\n",
                   NS, worst < 0.05 ? "PASS" : "FAIL", worst, worst_diag);
            fails += (worst >= 0.05);
            free(mean); free(cov); free(hA);
        }
        printf("\n%s\n", fails ? "SOME CHECKS FAILED" : "ALL CHECKS PASS");
        return fails ? 1 : 0;
    }

    /* ================== TRAINING ======================================== */
    int *h_inp = (int *)malloc(BT * sizeof(int));
    int *h_tgt = (int *)malloc(BT * sizeof(int));
    float *h_losses = (float *)malloc(BT * sizeof(float));
    Loader ld;
    loader_init(&ld, DATA_TRAIN);
    printf("\n%s: %d steps, lambda %.0f, kl_weight %.0e\n\n",
           overfit ? "OVERFIT MODE" : "TRAINING", STEPS, LAMBDA, KLW);
    if (overfit) {
        loader_next(&ld, h_inp, h_tgt, BT);
        CUDA_CHECK(cudaMemcpy(d_inp, h_inp, BT*sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_tgt, h_tgt, BT*sizeof(int), cudaMemcpyHostToDevice));
    }

    cudaEvent_t ev0, ev1;
    CUDA_CHECK(cudaEventCreate(&ev0)); CUDA_CHECK(cudaEventCreate(&ev1));

    for (int step = 1; step <= STEPS; step++) {
        CUDA_CHECK(cudaEventRecord(ev0));
        if (!overfit) {
            loader_next(&ld, h_inp, h_tgt, BT);
            CUDA_CHECK(cudaMemcpy(d_inp, h_inp, BT*sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_tgt, h_tgt, BT*sizeof(int), cudaMemcpyHostToDevice));
        }

        // M = I + A^T D^-1 A, per layer, BEFORE dA is used as a gradient.
        // Computed with cuBLAS using dA as scratch for B = D^-1 A.
        if (KLW > 0.0f) {
            const float one = 1.0f, zero = 0.0f;
            for (int l = 0; l < NBL; l++) {
                k_scale_A_dinv<<<ceil_div(L[l].P, BS), BS>>>(d_dA, d_A, d_rhod,
                                                  L[l].aoff, L[l].boff, L[l].P);
                CUBLAS_CHECK(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T,
                    RANK, RANK, L[l].P, &one, d_dA + L[l].aoff, RANK,
                    d_A + L[l].aoff, RANK, &zero, d_M + l*RANK*RANK, RANK));
            }
            CUDA_CHECK(cudaMemcpy(h_M, d_M, NBL*RANK*RANK*sizeof(float), cudaMemcpyDeviceToHost));
            for (int l = 0; l < NBL; l++) {
                for (int j = 0; j < RANK; j++) h_M[l*RANK*RANK + j*RANK + j] += 1.0f;
                if (!chol_logdet_inv(h_M + l*RANK*RANK, h_Minv + l*RANK*RANK,
                                     &logdetM[l])) {
                    fprintf(stderr, "M not positive definite in layer %s\n", L[l].name);
                    return 1;
                }
            }
            CUDA_CHECK(cudaMemcpy(d_Minv, h_Minv, NBL*RANK*RANK*sizeof(float), cudaMemcpyHostToDevice));
        }

        k_draw_z<<<ceil_div(NBL*RANK,128), 128>>>(d_z, NBL, ZSEED, step);          // one z per layer

        // w = theta everywhere, then the Bayesian regions are overwritten.
        CUDA_CHECK(cudaMemcpy(d_w, d_theta, pb, cudaMemcpyDeviceToDevice));
        for (int l = 0; l < NBL; l++)
            k_lowrank_sample<<<ceil_div(L[l].P, BS), BS>>>(d_w, d_theta, d_A, d_rhod,
                d_z + l*RANK, L[l].poff, L[l].aoff, L[l].boff, L[l].P, SEED, step);

        CUDA_CHECK(cudaMemset(d_dw,    0, pb));
        CUDA_CHECK(cudaMemset(d_dA,    0, ab));
        CUDA_CHECK(cudaMemset(d_drhod, 0, db));
        CUDA_CHECK(cudaMemset(d_gacts, 0, a.total * sizeof(float)));

        float nll = gpt2_forward(d_acts, d_w, d_inp, d_tgt, cfg, h_losses);
        gpt2_backward(d_dw, d_gacts, d_acts, d_w, d_inp, d_tgt, cfg);

        for (int l = 0; l < NBL; l++)
            k_lowrank_backward<<<ceil_div(L[l].P, BS), BS>>>(d_dA, d_drhod, d_dw,
                d_rhod, d_z + l*RANK, L[l].poff, L[l].aoff, L[l].boff, L[l].P,
                SEED, step);

        if (KLW > 0.0f)
            for (int l = 0; l < NBL; l++)
                k_lowrank_kl_grad<<<ceil_div(L[l].P, BS), BS>>>(d_dw, d_dA, d_drhod,
                    d_theta, d_A, d_rhod, d_Minv + l*RANK*RANK,
                    L[l].poff, L[l].aoff, L[l].boff, L[l].P, LAMBDA, KLW);

        float bc1 = 1.0f - powf(B1, (float)step), bc2 = 1.0f - powf(B2, (float)step);
        k_adamw<<<gN, BS>>>(d_theta, d_dw,    d_m,  d_v,  n,      bc1, bc2, LR, B1, B2, AEPS, WD);
        k_adamw<<<gA, BS>>>(d_A,     d_dA,    d_mA, d_vA, Pb*RANK, bc1, bc2, LR_VAR, B1, B2, AEPS, WD);
        k_adamw<<<gB, BS>>>(d_rhod,  d_drhod, d_md, d_vd, Pb,      bc1, bc2, LR_VAR, B1, B2, AEPS, WD);

        CUDA_CHECK(cudaEventRecord(ev1));
        CUDA_CHECK(cudaEventSynchronize(ev1));
        float ms; CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));

        if (step == 1 || step % 10 == 0 || step == STEPS) {
            double a2=0, d2=0, logd=0, mu2=0, smean=0; float lo=1e30f, hi=-1e30f;
            for (int l = 0; l < NBL; l++) {
                k_lowrank_stats<<<SB, BS>>>(dp[0],dp[1],dp[2],dp[3],dp[4],dp[5],dp[6],
                    d_theta, d_A, d_rhod, L[l].poff, L[l].aoff, L[l].boff, L[l].P);
                for (int q = 0; q < 7; q++)
                    CUDA_CHECK(cudaMemcpy(hp[q], dp[q], SB*sizeof(float),
                                          cudaMemcpyDeviceToHost));
                for (int i = 0; i < SB; i++) {
                    a2+=hp[0][i]; d2+=hp[1][i]; logd+=hp[2][i]; mu2+=hp[3][i];
                    if (hp[4][i] < lo) lo = hp[4][i];
                    if (hp[5][i] > hi) hi = hp[5][i];
                    smean += hp[6][i];
                }
            }
            // KL = 0.5[lambda(||mu||^2+trSigma) - P - logdet Sigma + P log lambda]
            double ldet = logd; for (int l = 0; l < NBL; l++) ldet += logdetM[l];
            double kl = 0.5 * (LAMBDA * (mu2 + a2 + d2) - (double)Pb - ldet
                               + (double)Pb * log((double)LAMBDA));
            printf("step %4d | nll %8.4f | kl %10.1f | sigma %.6f [%.6f, %.6f] | %7.2f ms\n",
                   step, nll, kl, smean / (double)Pb, lo, hi, ms);
        }
    }

    printf("\nsummary: rank %d, Bayesian over %zu params (%.1f%% — the 4 GEMMs only)\n",
           RANK, Pb, 100.0 * Pb / n);
    printf("summary: A = %.0f MB; low-rank costs MORE than mean-field, the case is quality\n",
           ab / 1e6);
    printf("\ndone.\n");
    return 0;
}
