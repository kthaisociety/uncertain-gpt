/*
=============================================================================
 train_bnn_gpu.cu — Bayes-by-Backprop on the GPU.  (Blundell et al. 2015)

 The BASELINE Bayesian method: mu and rho are both free parameters, rho is
 learned by backprop through the reparameterization, and the prior enters
 through an explicit closed-form KL gradient.

 Everything that is not the optimizer — kernels, layout, forward, backward,
 dataloader — lives in gpt2_kernels.cuh and is shared with every other
 variant. What is in THIS file is exactly what makes the model Bayesian.

 Build:   make train_bnn_gpu
 Run:     ./train_bnn_gpu            train on TinyShakespeare
          ./train_bnn_gpu overfit    memorize one batch (the smoke test)
          ./train_bnn_gpu eval [steps] [S] [batches] [split]
                                     NLPD / ECE / aleatoric-epistemic-total
                                     from a SAVED checkpoint, on held-out
                                     val or test. Never trains.

-----------------------------------------------------------------------------
 WHAT THIS COSTS, AND WHY IT IS THE THING TO BEAT

 Ten parameter-sized buffers, against four for a deterministic GPT-2:

     mu, rho                        2 x 340 MB   the variational parameters
     m_mu, v_mu, m_rho, v_rho       4 x 340 MB   TWO AdamW states
     w, dw                          2 x 340 MB   sampled weights + their grad
     dmu, drho                      2 x 340 MB   accumulated parameter grads
                                   -----------
                                    10 x 340 MB = 3.40 GB

 and five kernel passes over them per step (sample, bayes_backward,
 kl_backward, AdamW on mu, AdamW on rho) plus three full-buffer memsets.

 Measured: ~45 ms of a ~66.7 ms step, none of which is the network learning.

 One optimization is already applied here and is NOT part of the baseline
 idea: eps is regenerated with Philox rather than stored, which removes a
 340 MB buffer and ~680 MB/step of traffic. Kept because it changes no math
 whatsoever — the overfit test reaches the same loss either way.

-----------------------------------------------------------------------------
 THE KL WEIGHT IS DERIVED, NOT TUNED.  (recorded because getting this wrong
 cost 1.0 nats and briefly made Bayesian methods look categorically worse
 than they are)

 The ELBO is

     loss = E_q[NLL]  +  KL(q || p) / N

 The 1/N is not a knob. It falls out of the KL being a per-DATASET quantity
 while the reported NLL is a per-TOKEN mean. N = the training token count, so
 KLW = 1/ld.n, computed from the loader.

 Measured on 6 blocks, char-level TinyShakespeare, 2000 steps:

     KLW = 1e-3  (hardcoded, stale)   nll 3.4328
     KLW = 1e-4                       nll 2.7034
     KLW = 1e-6  (= 1/N, correct)     nll 2.4389
     KLW = 0                          nll 2.4068
     deterministic single model       nll 2.3167

 At the correct weight the cost of being Bayesian is 0.12 nats. At the stale
 one it looked like 1.1. The value had been carried over from an earlier
 experiment with a different dataset and parameter count and never re-derived
 — which is exactly why it is now computed rather than typed.

 This is bottleneck E ("KL weight sensitivity") from the project taxonomy,
 measured: three orders of magnitude in one constant move the loss by a full
 nat. Worth reporting, since BNN papers commonly quote a tuned KL weight
 without deriving it.
=============================================================================
*/

#include "gpt2_kernels.cuh"

#ifndef NBLOCKS
#define NBLOCKS 6      // transformer blocks; override with make L=12
#endif

/* ===================================================================== *
 * The Bayesian layer.  eps is NEVER stored — every kernel that needs it
 * regenerates it from (seed, step, index).
 * ===================================================================== */

__global__ void k_bayes_sample(float *w, const float *mu, const float *rho,
                               size_t n, unsigned long long seed, int step) {
    size_t i = (size_t)blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) w[i] = mu[i] + d_softplus(rho[i]) * philox_normal(seed, step, i);
}

__global__ void k_bayes_backward(float *dmu, float *drho, const float *dw,
                                 const float *rho, size_t n,
                                 unsigned long long seed, int step) {
    size_t i = (size_t)blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        float g = dw[i];
        float eps = philox_normal(seed, step, i);   // the SAME eps the forward used
        dmu[i]  += g;
        drho[i] += g * eps * d_sigmoid(rho[i]);
    }
}

__global__ void k_kl_backward(float *dmu, float *drho, const float *mu,
                              const float *rho, size_t n, float sp, float weight) {
    size_t i = (size_t)blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        float sp2 = sp * sp, r = rho[i], sigma = d_softplus(r);
        dmu[i]  += weight * (mu[i] / sp2);
        drho[i] += weight * (sigma / sp2 - 1.0f / sigma) * d_sigmoid(r);
    }
}

// k_adamw moved to gpt2_kernels.cuh: the deep-ensemble baseline is its third
// user (BBB runs it twice, ensemble members once each), and kernels used by
// more than one trainer live in the header — one copy, no drift.

// Per-block partial sums of the KL and of sigma, for reporting only.
// Reduced on the host afterwards — a few thousand floats, not worth a kernel.
// Mean sigma alone is not enough: in a sparse-gradient model the mean is
// owned by the untouched majority (38.6M wte rows, <=256 touched per step),
// and it once masked a working mechanism as "pinned" in the Vadam run. So
// the extremes ride along: sig_min/sig_max bound the whole distribution.
__global__ void k_kl_stats(float *kl_part, float *sig_part,
                           float *smin_part, float *smax_part,
                           const float *mu, const float *rho, size_t n, float sp) {
    __shared__ float scratch[32];
    float sp2 = sp * sp;
    double kl = 0.0; float sig = 0.0f, smin = 1e30f, smax = -1e30f;
    for (size_t i = (size_t)blockDim.x * blockIdx.x + threadIdx.x;
         i < n; i += (size_t)blockDim.x * gridDim.x) {
        float sigma = d_softplus(rho[i]), mm = mu[i];
        kl  += logf(sp / sigma) + (sigma * sigma + mm * mm) / (2.0f * sp2) - 0.5f;
        sig += sigma;
        smin = fminf(smin, sigma);
        smax = fmaxf(smax, sigma);
    }
    float k = block_reduce_sum((float)kl, scratch);
    __syncthreads();
    float s = block_reduce_sum(sig, scratch);
    __syncthreads();
    float lo = -block_reduce_max(-smin, scratch);   // block-wide min via max
    __syncthreads();
    float hi = block_reduce_max(smax, scratch);
    if (threadIdx.x == 0) {
        kl_part[blockIdx.x] = k;  sig_part[blockIdx.x] = s;
        smin_part[blockIdx.x] = lo; smax_part[blockIdx.x] = hi;
    }
}


/* ===================================================================== *
 * eval — uncertainty from a SAVED checkpoint, no training.
 *
 *     ./train_bnn_gpu eval [steps] [S] [batches] [split]
 *
 * The arm's whole contribution to the shared harness is one callback: draw
 * s is one reparameterized sample w = mu + softplus(rho) * eps, i.e. the
 * training sampler with the step index replaced by the sample index. From
 * there gpt2_kernels.cuh does exactly what it does for the other three.
 * ===================================================================== */
typedef struct { float *w, *mu, *rho; size_t n; int grid; } BBBCtx;

static const float *bbb_draw(int s, void *vctx) {
    BBBCtx *c = (BBBCtx *)vctx;
    // step = s+1 on the eval seed: S independent draws, from a Philox stream
    // no training step ever consumed.
    k_bayes_sample<<<c->grid, BS>>>(c->w, c->mu, c->rho, c->n, UQ_EVAL_SEED, s + 1);
    CUDA_CHECK(cudaGetLastError());
    return c->w;
}

static int bbb_eval(int argc, char **argv) {
    if (uq_selftest()) return 1;
    UQArgs u = uq_parse_args(argc, argv);
    CUBLAS_CHECK(cublasCreate(&cublas_handle));

    Config cfg = { .B = 4, .T = 64, .C = 384, .V = VOCAB, .NH = 6, .maxT = 1024, .L = NBLOCKS };
    ParamOff p = param_offsets(cfg);
    ActOff   a = act_offsets(cfg);
    size_t n = p.total, pb = n * sizeof(float);
    int BT = cfg.B * cfg.T;

    BBBCtx c; c.n = n; c.grid = ceil_div((int)n, BS);
    CUDA_CHECK(cudaMalloc(&c.mu, pb));
    CUDA_CHECK(cudaMalloc(&c.rho, pb));
    CUDA_CHECK(cudaMalloc(&c.w, pb));
    if (ckpt_load("bbb", "mu",  c.mu,  n, cfg, u.steps) < 0) return 1;
    if (ckpt_load("bbb", "rho", c.rho, n, cfg, u.steps) < 0) return 1;

    float *d_acts; int *d_inp, *d_tgt;
    CUDA_CHECK(cudaMalloc(&d_acts, a.total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_inp, BT * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_tgt, BT * sizeof(int)));
    int   *h_inp    = (int *)malloc(BT * sizeof(int));
    int   *h_tgt    = (int *)malloc(BT * sizeof(int));
    float *h_losses = (float *)malloc(BT * sizeof(float));

    Loader ld; loader_init(&ld, uq_split_path(u.split));
    UQSpec spec = { .cfg = cfg, .S = u.S, .nbatches = u.nbatches,
                    .arm = "bbb", .split = u.split, .mix_label = "mixture",
                    .s_fixed_by_arm = 0, .fixed_batch = 0 };
    UQResult r;
    int bad = uq_evaluate(&spec, bbb_draw, &c, &ld, d_acts, d_inp, d_tgt,
                          h_inp, h_tgt, h_losses, &r);
    if (!bad) uq_report(&spec, &r);

    free(h_inp); free(h_tgt); free(h_losses); free(ld.tokens);
    cudaFree(c.mu); cudaFree(c.rho); cudaFree(c.w);
    cudaFree(d_acts); cudaFree(d_inp); cudaFree(d_tgt);
    cublasDestroy(cublas_handle);
    return bad;
}

/* ===================================================================== *
 * main
 * ===================================================================== */

int main(int argc, char **argv) {
    if (argc > 1 && argv[1][0] == 'e') return bbb_eval(argc, argv);
    bool overfit = (argc > 1 && argv[1][0] == 'o');

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s  sm_%d%d  %.1f GB\n\n", prop.name, prop.major, prop.minor,
           prop.totalGlobalMem / 1e9);
    CUBLAS_CHECK(cublasCreate(&cublas_handle));

    Config cfg = { .B = 4, .T = 64, .C = 384, .V = VOCAB, .NH = 6, .maxT = 1024, .L = NBLOCKS };
    if (cfg.C / cfg.NH != HS) { printf("HS must equal C/NH\n"); return 1; }

    ParamOff p = param_offsets(cfg);
    ActOff   a = act_offsets(cfg);
    size_t n = p.total;
    int BT = cfg.B * cfg.T;

    const float SP = 0.1f;                  // prior sigma
    // KLW is DERIVED from the dataset size once the loader is up (see below).
    // It is NOT a free knob: the ELBO is  E_q[NLL] + KL/N,  and that 1/N falls
    // out of the KL being a per-DATASET quantity while the reported NLL is a
    // per-TOKEN mean. See the header for what hardcoding it cost.
    float KLW = 0.0f;
    const bool klw_override = (argc > 3);   // explicit sweep value, if given
    // Overfit mode wants a big lr on purpose: the test is "can 256 tokens be
    // memorized at all", and at 1e-4 it only gets to nll 7.6 in 200 steps —
    // which proves the gradients point downhill but not that the chain is
    // complete. 1e-3 makes the answer unambiguous.
    const float LR = overfit ? 1e-3f : 1e-4f;
    const float B1 = 0.9f, B2 = 0.999f, AEPS = 1e-8f, WD = 0.0f;
    const unsigned long long SEED = 20240520ull;
    // steps: default per mode, overridable as `./bin <mode> <steps>`
    int STEPS = overfit ? 200 : 100;
    if (argc > 2) STEPS = atoi(argv[2]);

    printf("params %zu (%.0f MB/buffer)   acts %zu (%.0f MB)\n",
           n, n * 4 / 1e6, a.total, a.total * 4 / 1e6);
    printf("total VRAM ~%.2f GB\n\n", (10.0 * n + 2.0 * a.total) * 4 / 1e9);

    // ---- allocate ---------------------------------------------------------
    float *d_mu, *d_rho, *d_w, *d_dw, *d_dmu, *d_drho;
    float *d_m_mu, *d_v_mu, *d_m_rho, *d_v_rho;
    float *d_acts, *d_gacts;
    int *d_inp, *d_tgt;
    size_t pb = n * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_mu, pb));    CUDA_CHECK(cudaMalloc(&d_rho, pb));
    CUDA_CHECK(cudaMalloc(&d_w, pb));     CUDA_CHECK(cudaMalloc(&d_dw, pb));
    CUDA_CHECK(cudaMalloc(&d_dmu, pb));   CUDA_CHECK(cudaMalloc(&d_drho, pb));
    CUDA_CHECK(cudaMalloc(&d_m_mu, pb));  CUDA_CHECK(cudaMalloc(&d_v_mu, pb));
    CUDA_CHECK(cudaMalloc(&d_m_rho, pb)); CUDA_CHECK(cudaMalloc(&d_v_rho, pb));
    CUDA_CHECK(cudaMalloc(&d_acts,  a.total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gacts, a.total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_inp, BT * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_tgt, BT * sizeof(int)));

    // ---- init on the host, once -------------------------------------------
    // Same scheme as train_bnn.c: small normal for mu, rho = -5 so
    // softplus(-5) ~ 0.0067 (bottleneck F: this init is load-bearing).
    srand(1337);
    float *h_par = (float *)malloc(pb);
    for (size_t i = 0; i < n; i++) {
        float u1 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 2.0f);
        float u2 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 2.0f);
        h_par[i] = 0.02f * sqrtf(-2.0f * logf(u1)) * cosf(6.2831853f * u2);
    }
    CUDA_CHECK(cudaMemcpy(d_mu, h_par, pb, cudaMemcpyHostToDevice));
    for (size_t i = 0; i < n; i++) h_par[i] = -5.0f;
    CUDA_CHECK(cudaMemcpy(d_rho, h_par, pb, cudaMemcpyHostToDevice));
    free(h_par);
    CUDA_CHECK(cudaMemset(d_m_mu, 0, pb));  CUDA_CHECK(cudaMemset(d_v_mu, 0, pb));
    CUDA_CHECK(cudaMemset(d_m_rho, 0, pb)); CUDA_CHECK(cudaMemset(d_v_rho, 0, pb));

    int grid_n = ceil_div((int)n, BS);
    int stat_blocks = 1024;
    float *d_klp, *d_sigp;
    CUDA_CHECK(cudaMalloc(&d_klp,  stat_blocks * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sigp, stat_blocks * sizeof(float)));
    float *d_smin, *d_smax;
    CUDA_CHECK(cudaMalloc(&d_smin, stat_blocks * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_smax, stat_blocks * sizeof(float)));
    float *h_klp  = (float *)malloc(stat_blocks * sizeof(float));
    float *h_sigp = (float *)malloc(stat_blocks * sizeof(float));
    float *h_smin = (float *)malloc(stat_blocks * sizeof(float));
    float *h_smax = (float *)malloc(stat_blocks * sizeof(float));

    int *h_inp = (int *)malloc(BT * sizeof(int));
    int *h_tgt = (int *)malloc(BT * sizeof(int));
    float *h_losses = (float *)malloc(BT * sizeof(float));

    Loader ld;
    loader_init(&ld, DATA_TRAIN);

    // ---- the ELBO's KL weight, derived --------------------------------------
    if (!overfit) KLW = 1.0f / (float)ld.n;
    if (klw_override) KLW = (float)atof(argv[3]);

    if (overfit) {
        // One batch, reused forever. If the model cannot memorize 256 tokens,
        // something is wired wrong — this catches what a gradient check cannot:
        // sign errors, lr scaling, a forgotten zero-grad.
        loader_next(&ld, h_inp, h_tgt, BT);
        CUDA_CHECK(cudaMemcpy(d_inp, h_inp, BT * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_tgt, h_tgt, BT * sizeof(int), cudaMemcpyHostToDevice));
        printf("\nOVERFIT MODE: one batch, KL off, %d steps\n\n", STEPS);
    } else {
        printf("\nTRAINING: %d steps, kl_weight %.0e, sigma_prior %.2f\n\n", STEPS, KLW, SP);
    }

    cudaEvent_t ev0, ev1;
    CUDA_CHECK(cudaEventCreate(&ev0)); CUDA_CHECK(cudaEventCreate(&ev1));

    // ---- the training loop ------------------------------------------------
    for (int step = 1; step <= STEPS; step++) {
        CUDA_CHECK(cudaEventRecord(ev0));

        if (!overfit) {
            loader_next(&ld, h_inp, h_tgt, BT);
            CUDA_CHECK(cudaMemcpy(d_inp, h_inp, BT * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_tgt, h_tgt, BT * sizeof(int), cudaMemcpyHostToDevice));
        }

        // 1. draw one Monte Carlo sample of the weights. eps is not stored.
        k_bayes_sample<<<grid_n, BS>>>(d_w, d_mu, d_rho, n, SEED, step);

        // 2. zero every accumulator. All backward ops use +=.
        CUDA_CHECK(cudaMemset(d_dw,    0, pb));
        CUDA_CHECK(cudaMemset(d_dmu,   0, pb));
        CUDA_CHECK(cudaMemset(d_drho,  0, pb));
        CUDA_CHECK(cudaMemset(d_gacts, 0, a.total * sizeof(float)));

        // 3. forward and backward on the SAMPLED weights, unchanged from the
        //    deterministic model — this is the wrapper framing made literal.
        float nll = gpt2_forward(d_acts, d_w, d_inp, d_tgt, cfg, h_losses);
        gpt2_backward(d_dw, d_gacts, d_acts, d_w, d_inp, d_tgt, cfg);

        // 4. push dL/dw back onto mu and rho, regenerating the same eps.
        k_bayes_backward<<<grid_n, BS>>>(d_dmu, d_drho, d_dw, d_rho, n, SEED, step);

        // 5. the prior's pull.
        if (KLW > 0.0f)
            k_kl_backward<<<grid_n, BS>>>(d_dmu, d_drho, d_mu, d_rho, n, SP, KLW);

        // 6. two AdamW passes — mu and rho each carry their own optimizer state.
        float bc1 = 1.0f - powf(B1, (float)step);
        float bc2 = 1.0f - powf(B2, (float)step);
        k_adamw<<<grid_n, BS>>>(d_mu,  d_dmu,  d_m_mu,  d_v_mu,  n, bc1, bc2,
                                LR, B1, B2, AEPS, WD);
        k_adamw<<<grid_n, BS>>>(d_rho, d_drho, d_m_rho, d_v_rho, n, bc1, bc2,
                                LR, B1, B2, AEPS, WD);

        CUDA_CHECK(cudaEventRecord(ev1));
        CUDA_CHECK(cudaEventSynchronize(ev1));
        float ms; CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));

        if (step == 1 || step % 10 == 0 || step == STEPS) {
            k_kl_stats<<<stat_blocks, BS>>>(d_klp, d_sigp, d_smin, d_smax,
                                            d_mu, d_rho, n, SP);
            CUDA_CHECK(cudaMemcpy(h_klp,  d_klp,  stat_blocks * sizeof(float),
                                  cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_sigp, d_sigp, stat_blocks * sizeof(float),
                                  cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_smin, d_smin, stat_blocks * sizeof(float),
                                  cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_smax, d_smax, stat_blocks * sizeof(float),
                                  cudaMemcpyDeviceToHost));
            double kl = 0.0, sg = 0.0; float lo = 1e30f, hi = -1e30f;
            for (int i = 0; i < stat_blocks; i++) {
                kl += h_klp[i]; sg += h_sigp[i];
                if (h_smin[i] < lo) lo = h_smin[i];
                if (h_smax[i] > hi) hi = h_smax[i];
            }
            printf("step %4d | nll %8.4f | kl %10.1f | sigma %.6f [%.6f, %.6f] | %7.2f ms\n",
                   step, nll, kl, sg / (double)n, lo, hi, ms);
        }
    }

    if (!overfit) {
        ckpt_save("bbb", "mu",  d_mu,  n, cfg, STEPS);
        ckpt_save("bbb", "rho", d_rho, n, cfg, STEPS);
    }
    printf("\ndone.\n");
    free(h_inp); free(h_tgt); free(h_losses); free(h_klp); free(h_sigp); free(h_smin); free(h_smax);
    free(ld.tokens);
    cudaFree(d_mu); cudaFree(d_rho); cudaFree(d_w); cudaFree(d_dw);
    cudaFree(d_dmu); cudaFree(d_drho);
    cudaFree(d_m_mu); cudaFree(d_v_mu); cudaFree(d_m_rho); cudaFree(d_v_rho);
    cudaFree(d_acts); cudaFree(d_gacts); cudaFree(d_inp); cudaFree(d_tgt);
    cudaFree(d_klp); cudaFree(d_sigp); cudaFree(d_smin); cudaFree(d_smax);
    cublasDestroy(cublas_handle);
    return 0;
}
