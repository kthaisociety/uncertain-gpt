/*
=============================================================================
 train_bnn_vadam.cu — Vadam on the GPU.  (Khan et al. 2018, Fig. 1 right)

 Same network, same data, same forward/backward as train_bnn_gpu.cu — all of
 that comes from gpt2_kernels.cuh. What differs is ONLY the optimizer, and
 the comments below number every line against the algorithm:

     1.  eps ~ N(0, I)
     2.  sigma <- 1/sqrt(N*s + lambda)
     3.  theta <- mu + sigma o eps
     4.  sample minibatch
     5.  g <- -(1/M) sum grad log p(D_i | theta)      (backprop at theta)
     6.  m <- g1*m + (1-g1) * (g + lambda*mu/N)       (prior enters here)
     7.  s <- g2*s + (1-g2) * (g o g)                 (RAW g — no prior term)
     8.  m^ <- m/(1-g1^t),  s^ <- s/(1-g2^t)
     9.  mu <- mu - alpha * m^ / (sqrt(s^) + lambda/N)

 sigma is DERIVED from Adam's second moment, never stored or learned. What
 that deletes relative to Bayes-by-Backprop:

     buffers:  rho, drho, m_rho, v_rho, dmu     (5 x 340 MB gone; 10 -> 5)
     kernels:  k_bayes_backward                 (dmu == dw, nothing to do)
               k_kl_backward                    (the prior IS line 6's term)
               second k_adamw                   (one update pass, not two)
     eps:      regenerated only in the SAMPLE — backward never needs it,
               because nothing differentiates through sigma anymore.

 Buffers here:  mu, m, s, w, dw  = 5 x 340 MB = 1.70 GB   (BBB: 3.40 GB)
 Kernel passes over param-sized buffers per step: 2 + 1 memset  (BBB: 5 + 3)

 Build:   make train_bnn_vadam
 Run:     ./train_bnn_vadam            train on TinyShakespeare
          ./train_bnn_vadam overfit    memorize one batch (smoke test)
          ./train_bnn_vadam eval [steps] [S] [batches] [split] [lambda]
                                       NLPD / ECE / aleatoric-epistemic-
                                       total from a SAVED checkpoint, on
                                       held-out val or test. Never trains.
                                       sigma is derived, so eval needs the
                                       TRAINING N and lambda — see below.

-----------------------------------------------------------------------------
 THE TWO ALGORITHM INPUTS THAT ARE MODELING CHOICES (constants below):

 N — what counts as "one data example i" in the objective. The loss is a
     mean over TOKENS, so N = number of training tokens (305,260 for
     TinyShakespeare). Set from the loader at startup. If you decide a
     SEQUENCE is the example instead, N shrinks 64x and every lambda/N and
     N*s term moves with it — that is a paper-level decision, not a knob.

 s0, LAMBDA — at step 1 with s0 = 0, sigma = 1/sqrt(LAMBDA) exactly (the
     prior). LAMBDA = 100 makes that 0.1, matching the BBB run's
     sigma_prior for comparability. As s grows, sigma shrinks below the
     prior: certainty is EARNED from curvature, never learned by backprop.
=============================================================================
*/

#include "gpt2_kernels.cuh"

#ifndef NBLOCKS
#define NBLOCKS 6      // transformer blocks; override with make L=12
#endif

/* ===================================================================== *
 * The Vadam layer: two kernels. That is the entire optimizer.
 * ===================================================================== */

// Lines 1-3: sigma from s, one Philox draw, perturbed weights.
// Reads s BEFORE this step's update touches it (kernels are ordered).
__global__ void k_vadam_sample(float *w, const float *mu, const float *s,
                               size_t n, float N, float lambda,
                               unsigned long long seed, int step) {
    size_t i = (size_t)blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        float sigma = rsqrtf(N * s[i] + lambda);              // line 2
        float eps   = philox_normal(seed, step, i);           // line 1
        w[i] = mu[i] + sigma * eps;                           // line 3
    }
}

// Lines 6-9: the whole update, one pass. g arrives as dw straight from
// gpt2_backward — in BBB this needed k_bayes_backward to move dw onto dmu
// first; here dmu WOULD equal dw, so the copy (and the buffer) are gone.
__global__ void k_vadam_update(float *mu, float *m, float *s, const float *dw,
                               size_t n, float N, float lambda,
                               float bc1, float bc2, float alpha,
                               float g1, float g2) {
    size_t i = (size_t)blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        float g = dw[i];
        float mi = g1 * m[i] + (1.0f - g1) * (g + lambda * mu[i] / N);  // line 6
        float si = g2 * s[i] + (1.0f - g2) * g * g;                     // line 7
        m[i] = mi;
        s[i] = si;
        float m_hat = mi / bc1;                                         // line 8
        float s_hat = si / bc2;
        mu[i] -= alpha * m_hat / (sqrtf(s_hat) + lambda / N);           // line 9
    }
}

// Reporting only: mean sigma, and the closed-form KL of N(mu, sigma^2)
// against the prior N(0, 1/lambda) — same quantity the BBB run prints, so
// the two logs are directly comparable. sigma_p = 1/sqrt(lambda).
__global__ void k_vadam_stats(float *kl_part, float *sig_part,
                              float *smin_part, float *smax_part,
                              const float *mu, const float *s, size_t n,
                              float N, float lambda) {
    __shared__ float scratch[32];
    float sigma_p = rsqrtf(lambda);
    double kl = 0.0; float sig = 0.0f, sgmin = 1e30f, sgmax = -1e30f;
    for (size_t i = (size_t)blockDim.x * blockIdx.x + threadIdx.x;
         i < n; i += (size_t)blockDim.x * gridDim.x) {
        float sigma = rsqrtf(N * s[i] + lambda);
        float mm = mu[i];
        kl  += logf(sigma_p / sigma)
             + (sigma * sigma + mm * mm) * (0.5f * lambda) - 0.5f;
        sig += sigma;
        sgmin = fminf(sgmin, sigma);   // smallest sigma = most certain param
        sgmax = fmaxf(sgmax, sigma);   // largest = still at the prior
    }
    float k = block_reduce_sum((float)kl, scratch);
    __syncthreads();
    float sg = block_reduce_sum(sig, scratch);
    __syncthreads();
    float lo = -block_reduce_max(-sgmin, scratch);   // block min via max
    __syncthreads();
    float hi = block_reduce_max(sgmax, scratch);
    if (threadIdx.x == 0) {
        kl_part[blockIdx.x] = k;   sig_part[blockIdx.x] = sg;
        smin_part[blockIdx.x] = lo; smax_part[blockIdx.x] = hi;
    }
}

/* ===================================================================== *
 * eval — uncertainty from a SAVED checkpoint, no training.
 *
 *     ./train_bnn_vadam eval [steps] [S] [batches] [split] [lambda]
 *
 * The callback is k_vadam_sample with the step index replaced by the
 * sample index — the same sampler training used, so the posterior being
 * measured is the one that was fit.
 *
 * TWO INPUTS THIS ARM CANNOT READ OFF ITS CHECKPOINT, because sigma is
 * DERIVED rather than stored (only s is saved):
 *
 *   N        sigma = 1/sqrt(N*s + lambda) has the TRAINING token count
 *            inside it. Evaluating on val does not change N — using the
 *            val split's 5M here instead of the train split's 90M would
 *            inflate every sigma by ~4x and silently corrupt the
 *            comparison. So N is read from DATA_TRAIN's header (1 KB,
 *            no tokens loaded), exactly as the trainer computed it.
 *   lambda   defaults to the same DERIVED value the trainer uses,
 *            1/init_std^2. If a checkpoint was trained with an explicit
 *            lambda (argv[3] on the train path), pass the same value here
 *            or the sigmas will not match the fitted posterior.
 * ===================================================================== */
typedef struct { float *w, *mu, *s; size_t n; int grid; float N, lambda; } VadamCtx;

static const float *vadam_draw(int s, void *vctx) {
    VadamCtx *c = (VadamCtx *)vctx;
    k_vadam_sample<<<c->grid, BS>>>(c->w, c->mu, c->s, c->n, c->N, c->lambda,
                                    UQ_EVAL_SEED, s + 1);
    CUDA_CHECK(cudaGetLastError());
    return c->w;
}

static int vadam_eval(int argc, char **argv) {
    if (uq_selftest()) return 1;
    UQArgs u = uq_parse_args(argc, argv);
    CUBLAS_CHECK(cublasCreate(&cublas_handle));

    Config cfg = { .B = 4, .T = 64, .C = 384, .V = VOCAB, .NH = 6, .maxT = 1024, .L = NBLOCKS };
    ParamOff p = param_offsets(cfg);
    ActOff   a = act_offsets(cfg);
    size_t n = p.total, pb = n * sizeof(float);
    int BT = cfg.B * cfg.T;

    const float MU_INIT_STD = 0.02f;                       // same as training
    VadamCtx c;
    c.n = n; c.grid = ceil_div((int)n, BS);
    c.N      = (float)loader_count(DATA_TRAIN);            // the TRAINING N
    c.lambda = 1.0f / (MU_INIT_STD * MU_INIT_STD);
    if (argc > 6) c.lambda = (float)atof(argv[6]);
    printf("vadam eval: N = %.0f (from %s) | lambda = %.0f | prior sigma = %.4f\n",
           c.N, DATA_TRAIN, c.lambda, 1.0f / sqrtf(c.lambda));

    CUDA_CHECK(cudaMalloc(&c.mu, pb));
    CUDA_CHECK(cudaMalloc(&c.s,  pb));
    CUDA_CHECK(cudaMalloc(&c.w,  pb));
    if (ckpt_load("vadam", "mu", c.mu, n, cfg, u.steps) < 0) return 1;
    if (ckpt_load("vadam", "s",  c.s,  n, cfg, u.steps) < 0) return 1;

    float *d_acts; int *d_inp, *d_tgt;
    CUDA_CHECK(cudaMalloc(&d_acts, a.total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_inp, BT * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_tgt, BT * sizeof(int)));
    int   *h_inp    = (int *)malloc(BT * sizeof(int));
    int   *h_tgt    = (int *)malloc(BT * sizeof(int));
    float *h_losses = (float *)malloc(BT * sizeof(float));

    Loader ld; loader_init(&ld, uq_split_path(u.split));
    UQSpec spec = { .cfg = cfg, .S = u.S, .nbatches = u.nbatches,
                    .arm = "vadam", .split = u.split, .mix_label = "mixture",
                    .s_fixed_by_arm = 0, .fixed_batch = 0 };
    UQResult r;
    int bad = uq_evaluate(&spec, vadam_draw, &c, &ld, d_acts, d_inp, d_tgt,
                          h_inp, h_tgt, h_losses, &r);
    if (!bad) uq_report(&spec, &r);

    free(h_inp); free(h_tgt); free(h_losses); free(ld.tokens);
    cudaFree(c.mu); cudaFree(c.s); cudaFree(c.w);
    cudaFree(d_acts); cudaFree(d_inp); cudaFree(d_tgt);
    cublasDestroy(cublas_handle);
    return bad;
}

/* ===================================================================== *
 * main — same shape as the BBB main, minus everything Vadam deleted.
 *
 * Compiled out when this file is INCLUDED by the Python extension
 * (see the note over train_bnn_gpu.cu's main). vadam_draw above is shared
 * with uncertain::VadamTrainer unchanged. dev/vadam_{update,sample}.cu
 * include it the same way, so they test THESE kernels rather than a copy.
 * ===================================================================== */
#ifndef UNCERTAIN_AS_LIBRARY
int main(int argc, char **argv) {
    if (argc > 1 && argv[1][0] == 'e') return vadam_eval(argc, argv);
    bool overfit = (argc > 1 && argv[1][0] == 'o');

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s  sm_%d%d  %.1f GB\n\n", prop.name, prop.major, prop.minor,
           prop.totalGlobalMem / 1e9);
    CUBLAS_CHECK(cublasCreate(&cublas_handle));

    Config cfg = { .B = 4, .T = 64, .C = 384, .V = VOCAB, .NH = 6, .maxT = 1024, .L = NBLOCKS };
    ParamOff p = param_offsets(cfg);
    ActOff   a = act_offsets(cfg);
    size_t n = p.total;
    int BT = cfg.B * cfg.T;

    // ---- the algorithm's inputs -------------------------------------------
    // Vadam has no KL weight — the prior lives inside sigma = 1/sqrt(N*s+lambda),
    // so LAMBDA is where the same "stale prior scale" mistake shows up. It was
    // 100 (sigma_prior = 0.1), which is FIVE TIMES the scale the weights are
    // initialized at (0.02). That injects 500% relative noise and is why this
    // arm diverged on char-level data.
    //
    // Derived instead: the prior std should match the scale the weights live
    // at, so lambda = 1/init_std^2. At init_std = 0.02 that is 2500, giving a
    // step-1 sigma of exactly 0.02.
    const float MU_INIT_STD = 0.02f;
    float LAMBDA = 1.0f / (MU_INIT_STD * MU_INIT_STD);   // = 2500
    if (argc > 3) LAMBDA = (float)atof(argv[3]);         // explicit sweep value
    const float S0     = 0.0f;     // s init; 0 -> step-1 sigma is exactly the prior
    const float LR = overfit ? 1e-3f : 1e-4f;   // same lr policy as the BBB file
    const float G1 = 0.9f, G2 = 0.999f;
    const unsigned long long SEED = 20240520ull;
    // steps: default per mode, overridable as `./bin <mode> <steps>`
    int STEPS = overfit ? 200 : 100;
    if (argc > 2) STEPS = atoi(argv[2]);

    printf("params %zu (%.0f MB/buffer)   acts %zu (%.0f MB)\n",
           n, n * 4 / 1e6, a.total, a.total * 4 / 1e6);
    printf("total VRAM ~%.2f GB  (BBB needed ~%.2f)\n\n",
           (5.0 * n + 2.0 * a.total) * 4 / 1e9,
           (10.0 * n + 2.0 * a.total) * 4 / 1e9);

    // ---- allocate: FIVE param-sized buffers -------------------------------
    float *d_mu, *d_m, *d_s, *d_w, *d_dw, *d_acts, *d_gacts;
    int *d_inp, *d_tgt;
    size_t pb = n * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_mu, pb));
    CUDA_CHECK(cudaMalloc(&d_m,  pb));
    CUDA_CHECK(cudaMalloc(&d_s,  pb));
    CUDA_CHECK(cudaMalloc(&d_w,  pb));
    CUDA_CHECK(cudaMalloc(&d_dw, pb));
    CUDA_CHECK(cudaMalloc(&d_acts,  a.total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gacts, a.total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_inp, BT * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_tgt, BT * sizeof(int)));

    // ---- init: same mu scheme and seed as BBB, for comparability ----------
    srand(1337);
    float *h_par = (float *)malloc(pb);
    for (size_t i = 0; i < n; i++) {
        float u1 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 2.0f);
        float u2 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 2.0f);
        h_par[i] = 0.02f * sqrtf(-2.0f * logf(u1)) * cosf(6.2831853f * u2);
    }
    CUDA_CHECK(cudaMemcpy(d_mu, h_par, pb, cudaMemcpyHostToDevice));
    if (S0 == 0.0f) {
        CUDA_CHECK(cudaMemset(d_s, 0, pb));
    } else {
        for (size_t i = 0; i < n; i++) h_par[i] = S0;
        CUDA_CHECK(cudaMemcpy(d_s, h_par, pb, cudaMemcpyHostToDevice));
    }
    free(h_par);
    CUDA_CHECK(cudaMemset(d_m, 0, pb));

    int grid_n = ceil_div((int)n, BS);
    int stat_blocks = 1024;
    float *d_klp, *d_sigp, *d_smin, *d_smax;
    CUDA_CHECK(cudaMalloc(&d_klp,  stat_blocks * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sigp, stat_blocks * sizeof(float)));
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

    // N: one data example = one TOKEN, because the loss is a mean over
    // tokens. See the header for what changes if you decide otherwise.
    const float N = (float)ld.n;
    printf("N = %.0f (tokens)   lambda = %.0f   lambda/N = %.2e   step-1 sigma = %.3f\n",
           N, LAMBDA, LAMBDA / N, 1.0f / sqrtf(LAMBDA));

    if (overfit) {
        loader_next(&ld, h_inp, h_tgt, BT);
        CUDA_CHECK(cudaMemcpy(d_inp, h_inp, BT * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_tgt, h_tgt, BT * sizeof(int), cudaMemcpyHostToDevice));
        printf("\nOVERFIT MODE: one batch, %d steps\n\n", STEPS);
    } else {
        printf("\nTRAINING: %d steps\n\n", STEPS);
    }

    cudaEvent_t ev0, ev1;
    CUDA_CHECK(cudaEventCreate(&ev0)); CUDA_CHECK(cudaEventCreate(&ev1));

    // ---- the training loop ------------------------------------------------
    for (int step = 1; step <= STEPS; step++) {
        CUDA_CHECK(cudaEventRecord(ev0));

        if (!overfit) {
            loader_next(&ld, h_inp, h_tgt, BT);                        // line 4
            CUDA_CHECK(cudaMemcpy(d_inp, h_inp, BT * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_tgt, h_tgt, BT * sizeof(int), cudaMemcpyHostToDevice));
        }

        // lines 1-3: perturbed weights. sigma read off the CURRENT s.
        k_vadam_sample<<<grid_n, BS>>>(d_w, d_mu, d_s, n, N, LAMBDA, SEED, step);

        // gradient accumulators (backward ops all use +=)
        CUDA_CHECK(cudaMemset(d_dw,    0, pb));
        CUDA_CHECK(cudaMemset(d_gacts, 0, a.total * sizeof(float)));

        // line 5: backprop AT theta — the sampled w, not mu.
        float nll = gpt2_forward(d_acts, d_w, d_inp, d_tgt, cfg, h_losses);
        gpt2_backward(d_dw, d_gacts, d_acts, d_w, d_inp, d_tgt, cfg);

        // lines 6-9: the whole update in one pass over dw.
        float bc1 = 1.0f - powf(G1, (float)step);
        float bc2 = 1.0f - powf(G2, (float)step);
        k_vadam_update<<<grid_n, BS>>>(d_mu, d_m, d_s, d_dw, n, N, LAMBDA,
                                       bc1, bc2, LR, G1, G2);

        CUDA_CHECK(cudaEventRecord(ev1));
        CUDA_CHECK(cudaEventSynchronize(ev1));
        float ms; CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));

        if (step == 1 || step % 10 == 0 || step == STEPS) {
            k_vadam_stats<<<stat_blocks, BS>>>(d_klp, d_sigp, d_smin, d_smax,
                                               d_mu, d_s, n, N, LAMBDA);
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
            // Same line format as train_bnn_gpu.cu — the scorecard parses both.
            printf("step %4d | nll %8.4f | kl %10.1f | sigma %.6f [%.6f, %.6f] | %7.2f ms\n",
                   step, nll, kl, sg / (double)n, lo, hi, ms);
        }
    }

    if (!overfit) {
        ckpt_save("vadam", "mu", d_mu, n, cfg, STEPS);
        ckpt_save("vadam", "s",  d_s,  n, cfg, STEPS);   // sigma is derived from s
    }
    printf("\ndone.\n");
    free(h_inp); free(h_tgt); free(h_losses); free(h_klp); free(h_sigp); free(h_smin); free(h_smax);
    free(ld.tokens);
    cudaFree(d_mu); cudaFree(d_m); cudaFree(d_s); cudaFree(d_w); cudaFree(d_dw);
    cudaFree(d_acts); cudaFree(d_gacts); cudaFree(d_inp); cudaFree(d_tgt);
    cudaFree(d_klp); cudaFree(d_sigp); cudaFree(d_smin); cudaFree(d_smax);
    cublasDestroy(cublas_handle);
    return 0;
}

#endif  // !UNCERTAIN_AS_LIBRARY
