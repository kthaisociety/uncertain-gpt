/*
=============================================================================
 train_ensemble.cu — deep ensemble baseline.  (Lakshminarayanan et al. 2017)

 THE CONTROL ARM. E independent deterministic GPT-2s, differing ONLY in the
 random seed; uncertainty comes from their disagreement, not from a
 posterior. This is the method the Bayesian arms have to beat — if it wins,
 that is the finding and it gets reported as such.

     Arm A  train_bnn_vadam.cu   sigma from Adam's second moment
     Arm B  (not written yet)    learned sigma, low-rank covariance
     Arm C  THIS FILE            E=5 deep ensemble

 No new CUDA kernels: members train with the shared deterministic
 forward/backward and the shared k_adamw. This file deliberately includes
 no sampling code — the ensemble path and the BNN sampling path cannot
 share state because they share no code beyond the network itself.

 WHAT VARIES / WHAT IS HELD FIXED (per the design):
     varies      the seed -> weight init only (this codebase has no dropout,
                 and the dataloader is sequential, so init is the ONLY
                 source of diversity; that is the recommended setup — no
                 bagging, since data resampling shrinks the effective
                 training set per member)
     held fixed  architecture, data order, optimizer, lr schedule, steps,
                 batch size, precision

 INFERENCE: average PROBABILITIES, never logits:
     p_bar(y|x) = (1/E) sum_e softmax(logits_e)
 A logit average is a geometric mean — not a mixture, wrong uncertainty.

 UNCERTAINTY DECOMPOSITION (per token, nats, double precision):
     total     = H[p_bar]
     aleatoric = (1/E) sum_e H[p_e]
     epistemic = (1/E) sum_e KL(p_e || p_bar)
 total = aleatoric + epistemic is an exact identity; it is ASSERTED at
 1e-5 and the run exits nonzero if it fails. Epistemic — member
 disagreement — is the analogue of the BNN's predictive variance.

 FAIR-COMPARISON NOTE: the BNN arms use S Monte Carlo forward passes at
 eval; with S = E = 5 the inference FLOPs are identical across arms. Where
 the ensemble pays is MEMORY (E x params — worst of all arms) and TRAIN
 COST (E x steps).

 Deviations from the full spec, deliberate at this scale:
   - members live in VRAM, no checkpoint files (5 x 340 MB fits; seeds are
     printed, so any member is reproducible from its seed)
   - "OOD set" is uniform-random token ids — TinyShakespeare has no second
     domain. Synthetic but directional: epistemic MUST rise on it, and the
     run warns if it does not.
   - ECE uses 15 bins on next-token argmax confidence.

 Build:   make train_ensemble
 Run:     ./train_ensemble            train E members, then evaluate
          ./train_ensemble overfit    E members memorize the same batch
=============================================================================
*/

#include "gpt2_kernels.cuh"

#ifndef NBLOCKS
#define NBLOCKS 6      // transformer blocks; override with make L=12
#endif

#define E_MEMBERS 5
static const int SEEDS[E_MEMBERS] = { 1337, 2337, 3337, 4337, 5337 };

/* ===================================================================== *
 * Host-side evaluation: consumes E prob buffers for one batch, accumulates
 * NLPD / ECE / the three uncertainty terms. All math in double — the
 * identity check at 1e-5 is only meaningful if the arithmetic is cleaner
 * than the tolerance.
 * ===================================================================== */
typedef struct {
    double nlpd_ens, nlpd_single;      // sums over positions
    double total, alea, epi;           // sums over positions
    double id_err_max;                 // worst |total-(alea+epi)| seen
    // ECE, 15 bins on confidence = max_y p_bar
    double bin_conf[15], bin_acc[15]; long bin_n[15];
    long npos;
} EvalAcc;

static void eval_batch(EvalAcc *acc, float **probs /*E x BT*V*/, const int *tgt,
                       int BT, int V, double *pbar /*V scratch*/, int with_nlpd) {
    for (int p = 0; p < BT; p++) {
        // p_bar: mixture of the E members, in double
        for (int y = 0; y < V; y++) pbar[y] = 0.0;
        for (int e = 0; e < E_MEMBERS; e++) {
            const float *pe = probs[e] + (size_t)p * V;
            for (int y = 0; y < V; y++) pbar[y] += pe[y];
        }
        double h_bar = 0.0;
        int argmax = 0;
        for (int y = 0; y < V; y++) {
            pbar[y] /= E_MEMBERS;
            if (pbar[y] > 0.0) h_bar -= pbar[y] * log(pbar[y]);
            if (pbar[y] > pbar[argmax]) argmax = y;
        }

        // aleatoric = mean member entropy; epistemic = mean KL(p_e || p_bar),
        // computed DIRECTLY (not as the difference) so the identity check
        // below actually tests something.
        double h_mean = 0.0, kl_mean = 0.0;
        for (int e = 0; e < E_MEMBERS; e++) {
            const float *pe = probs[e] + (size_t)p * V;
            double h = 0.0, kl = 0.0;
            for (int y = 0; y < V; y++) {
                double q = pe[y];
                if (q > 0.0) {
                    double lq = log(q);
                    h  -= q * lq;
                    kl += q * (lq - log(pbar[y]));
                }
            }
            h_mean += h; kl_mean += kl;
        }
        h_mean /= E_MEMBERS; kl_mean /= E_MEMBERS;

        double id_err = fabs(h_bar - (h_mean + kl_mean));
        if (id_err > acc->id_err_max) acc->id_err_max = id_err;

        acc->total += h_bar; acc->alea += h_mean; acc->epi += kl_mean;

        if (with_nlpd) {
            int y = tgt[p];
            acc->nlpd_ens    -= log(fmax(pbar[y], 1e-30));
            acc->nlpd_single -= log(fmax((double)probs[0][(size_t)p * V + y], 1e-30));
            int b = (int)(pbar[argmax] * 15.0); if (b > 14) b = 14;
            acc->bin_conf[b] += pbar[argmax];
            acc->bin_acc[b]  += (argmax == y) ? 1.0 : 0.0;
            acc->bin_n[b]++;
        }
        acc->npos++;
    }
}

int main(int argc, char **argv) {
    bool overfit = (argc > 1 && argv[1][0] == 'o');

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s  sm_%d%d  %.1f GB\n\n", prop.name, prop.major, prop.minor,
           prop.totalGlobalMem / 1e9);
    CUBLAS_CHECK(cublasCreate(&cublas_handle));

    Config cfg = { .B = 4, .T = 64, .C = 768, .V = VOCAB, .NH = 12, .maxT = 1024, .L = NBLOCKS };
    ParamOff po = param_offsets(cfg);
    ActOff   a  = act_offsets(cfg);
    size_t n = po.total;
    int BT = cfg.B * cfg.T, V = cfg.V;

    const float LR = overfit ? 1e-3f : 1e-4f;   // same policy as the other arms
    const float B1 = 0.9f, B2 = 0.999f, AEPS = 1e-8f, WD = 0.0f;
    // steps: default per mode, overridable as `./bin <mode> <steps>`
    int STEPS = overfit ? 200 : 100;      // per member;
    if (argc > 2) STEPS = atoi(argv[2]);
    const int EVAL_BATCHES = overfit ? 1 : 2;   // ID batches (and as many OOD)

    printf("params %zu (%.0f MB/buffer)   E=%d members\n", n, n * 4 / 1e6, E_MEMBERS);
    // members + AdamW m,v + dw + two act buffers
    printf("total VRAM ~%.2f GB  (members alone: %.2f — E x params, the arm's price)\n\n",
           ((E_MEMBERS + 3.0) * n + 2.0 * a.total) * 4 / 1e9,
           E_MEMBERS * (double)n * 4 / 1e9);

    // ---- allocate ---------------------------------------------------------
    size_t pb = n * sizeof(float);
    float *d_member[E_MEMBERS];
    for (int e = 0; e < E_MEMBERS; e++) CUDA_CHECK(cudaMalloc(&d_member[e], pb));
    float *d_m, *d_v, *d_dw, *d_acts, *d_gacts;
    CUDA_CHECK(cudaMalloc(&d_m,  pb));
    CUDA_CHECK(cudaMalloc(&d_v,  pb));
    CUDA_CHECK(cudaMalloc(&d_dw, pb));
    CUDA_CHECK(cudaMalloc(&d_acts,  a.total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gacts, a.total * sizeof(float)));
    int *d_inp, *d_tgt;
    CUDA_CHECK(cudaMalloc(&d_inp, BT * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_tgt, BT * sizeof(int)));

    int *h_inp = (int *)malloc(BT * sizeof(int));
    int *h_tgt = (int *)malloc(BT * sizeof(int));
    float *h_losses = (float *)malloc(BT * sizeof(float));
    float *h_par = (float *)malloc(pb);

    Loader ld;
    loader_init(&ld, DATA_TRAIN);
    printf("\n%s: %d members x %d steps, seed varies, data order fixed\n\n",
           overfit ? "OVERFIT MODE" : "TRAINING", E_MEMBERS, STEPS);

    if (overfit) {   // one fixed batch, shared by every member
        loader_next(&ld, h_inp, h_tgt, BT);
        CUDA_CHECK(cudaMemcpy(d_inp, h_inp, BT * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_tgt, h_tgt, BT * sizeof(int), cudaMemcpyHostToDevice));
    }

    cudaEvent_t ev0, ev1;
    CUDA_CHECK(cudaEventCreate(&ev0)); CUDA_CHECK(cudaEventCreate(&ev1));
    int grid_n = ceil_div((int)n, BS);

    // ---- train the E members, sequentially --------------------------------
    for (int e = 0; e < E_MEMBERS; e++) {
        printf("member %d | seed %d\n", e, SEEDS[e]);

        // the seed's ONLY job: a different starting point in weight space
        srand(SEEDS[e]);
        for (size_t i = 0; i < n; i++) {
            float u1 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 2.0f);
            float u2 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 2.0f);
            h_par[i] = 0.02f * sqrtf(-2.0f * logf(u1)) * cosf(6.2831853f * u2);
        }
        CUDA_CHECK(cudaMemcpy(d_member[e], h_par, pb, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_m, 0, pb));
        CUDA_CHECK(cudaMemset(d_v, 0, pb));
        ld.pos = 0;                     // identical data order for every member

        for (int step = 1; step <= STEPS; step++) {
            CUDA_CHECK(cudaEventRecord(ev0));
            if (!overfit) {
                loader_next(&ld, h_inp, h_tgt, BT);
                CUDA_CHECK(cudaMemcpy(d_inp, h_inp, BT * sizeof(int), cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(d_tgt, h_tgt, BT * sizeof(int), cudaMemcpyHostToDevice));
            }
            CUDA_CHECK(cudaMemset(d_dw,    0, pb));
            CUDA_CHECK(cudaMemset(d_gacts, 0, a.total * sizeof(float)));

            // deterministic: forward/backward straight on the member's params
            float nll = gpt2_forward(d_acts, d_member[e], d_inp, d_tgt, cfg, h_losses);
            gpt2_backward(d_dw, d_gacts, d_acts, d_member[e], d_inp, d_tgt, cfg);

            float bc1 = 1.0f - powf(B1, (float)step);
            float bc2 = 1.0f - powf(B2, (float)step);
            k_adamw<<<grid_n, BS>>>(d_member[e], d_dw, d_m, d_v, n, bc1, bc2,
                                    LR, B1, B2, AEPS, WD);

            CUDA_CHECK(cudaEventRecord(ev1));
            CUDA_CHECK(cudaEventSynchronize(ev1));
            float ms; CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));

            // standard scorecard line; sigma/kl are 0 — nothing is sampled here
            if (step == 1 || step % 10 == 0 || step == STEPS)
                printf("step %4d | nll %8.4f | kl        0.0 | sigma 0.000000 [0.000000, 0.000000] | %7.2f ms\n",
                       step, nll, ms);
        }
    }

    /* ---- ensemble evaluation ---------------------------------------------
       Batch by batch: run all E members on the same inputs, pull their prob
       rows to the host, fold into the accumulator. p_bar / H / KL in double. */
    float *h_probs[E_MEMBERS];
    for (int e = 0; e < E_MEMBERS; e++)
        h_probs[e] = (float *)malloc((size_t)BT * V * sizeof(float));
    double *pbar = (double *)malloc((size_t)V * sizeof(double));
    EvalAcc id_acc  = {0}, ood_acc = {0};

    // Held-out: text the training never reached (training consumed the first
    // STEPS*BT tokens; overfit consumed one batch and evaluates on it).
    if (!overfit) ld.pos = (long)STEPS * BT;
    for (int b = 0; b < EVAL_BATCHES; b++) {
        if (!overfit) {
            loader_next(&ld, h_inp, h_tgt, BT);
            CUDA_CHECK(cudaMemcpy(d_inp, h_inp, BT * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_tgt, h_tgt, BT * sizeof(int), cudaMemcpyHostToDevice));
        }
        for (int e = 0; e < E_MEMBERS; e++) {
            gpt2_forward(d_acts, d_member[e], d_inp, d_tgt, cfg, h_losses);
            CUDA_CHECK(cudaMemcpy(h_probs[e], d_acts + a.prb,
                                  (size_t)BT * V * sizeof(float), cudaMemcpyDeviceToHost));
        }
        eval_batch(&id_acc, h_probs, h_tgt, BT, V, pbar, 1);
    }

    // OOD: uniform-random token ids. Not another text domain — a direction:
    // member disagreement must RISE out here or the ensemble is not
    // capturing model uncertainty.
    srand(99991);
    for (int b = 0; b < EVAL_BATCHES; b++) {
        for (int i = 0; i < BT; i++) { h_inp[i] = rand() % V; h_tgt[i] = rand() % V; }
        CUDA_CHECK(cudaMemcpy(d_inp, h_inp, BT * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_tgt, h_tgt, BT * sizeof(int), cudaMemcpyHostToDevice));
        for (int e = 0; e < E_MEMBERS; e++) {
            gpt2_forward(d_acts, d_member[e], d_inp, d_tgt, cfg, h_losses);
            CUDA_CHECK(cudaMemcpy(h_probs[e], d_acts + a.prb,
                                  (size_t)BT * V * sizeof(float), cudaMemcpyDeviceToHost));
        }
        eval_batch(&ood_acc, h_probs, h_tgt, BT, V, pbar, 0);
    }

    // ---- report -----------------------------------------------------------
    double np = (double)id_acc.npos;
    double ece = 0.0;
    for (int b = 0; b < 15; b++)
        if (id_acc.bin_n[b])
            ece += fabs(id_acc.bin_acc[b] - id_acc.bin_conf[b]) / np;

    double epi_id  = id_acc.epi  / np;
    double epi_ood = ood_acc.epi / (double)ood_acc.npos;

    // The correctness gate of this file: total = aleatoric + epistemic.
    if (id_acc.id_err_max > 1e-5 || ood_acc.id_err_max > 1e-5) {
        fprintf(stderr, "IDENTITY VIOLATION: |total-(alea+epi)| = %.2e (tol 1e-5)\n",
                fmax(id_acc.id_err_max, ood_acc.id_err_max));
        return 1;
    }

    printf("\nsummary: E=%d, S=E so inference FLOPs match the BNN arms | member mem %d x params\n",
           E_MEMBERS, E_MEMBERS);
    printf("summary: NLPD single %.4f -> ensemble %.4f | ECE(15) %.4f | identity_err %.1e\n",
           id_acc.nlpd_single / np, id_acc.nlpd_ens / np, ece,
           fmax(id_acc.id_err_max, ood_acc.id_err_max));
    printf("summary: uncertainty (nats)  ID: total %.3f = alea %.3f + epi %.4f   OOD: epi %.4f  %s\n",
           id_acc.total / np, id_acc.alea / np, epi_id, epi_ood,
           epi_ood > epi_id ? "(OOD > ID, as required)" : "(WARN: OOD <= ID!)");

    // final standard step line: the scorecard's "last nll" for this arm is
    // the ENSEMBLE NLPD — the number the arm actually competes on.
    printf("step %4d | nll %8.4f | kl        0.0 | sigma 0.000000 [0.000000, 0.000000] |    0.00 ms\n",
           STEPS, id_acc.nlpd_ens / np);

    printf("\ndone.\n");
    for (int e = 0; e < E_MEMBERS; e++) { free(h_probs[e]); cudaFree(d_member[e]); }
    free(pbar); free(h_inp); free(h_tgt); free(h_losses); free(h_par); free(ld.tokens);
    cudaFree(d_m); cudaFree(d_v); cudaFree(d_dw);
    cudaFree(d_acts); cudaFree(d_gacts); cudaFree(d_inp); cudaFree(d_tgt);
    cublasDestroy(cublas_handle);
    return 0;
}
