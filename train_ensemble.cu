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

 UNCERTAINTY DECOMPOSITION — now in gpt2_kernels.cuh, shared by all four
 arms as "S samples from a per-arm sample_fn". It was written here, and
 this file is still the reference implementation; it moved because three
 of four arms produced no uncertainty numbers while it lived here.

     total     = H[p_bar]
     aleatoric = (1/S) sum_s H[p_s]
     epistemic = (1/S) sum_s KL(p_s || p_bar)

 total = aleatoric + epistemic is an exact identity; it is ASSERTED at
 1e-5 and the run exits nonzero if it fails — for every arm now, not just
 this one. Epistemic — sample disagreement — is the analogue of the BNN's
 predictive variance.

 FAIR-COMPARISON NOTE: the BNN arms use S Monte Carlo forward passes at
 eval; with S = E = 5 the inference FLOPs are identical across arms. Where
 the ensemble pays is MEMORY (E x params — worst of all arms) and TRAIN
 COST (E x steps). And S IS NOT FREE HERE: this arm's S is pinned to the
 number of members it trained, while the Bayesian arms buy an extra sample
 for one forward pass. The harness prints that asymmetry rather than
 quietly evaluating the arms at different S.

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
          ./train_ensemble eval [steps] [S] [batches] [split]
                                      uncertainty from SAVED members, on
                                      held-out val/test. No training. S is
                                      ignored (see the note above).
=============================================================================
*/

#include "gpt2_kernels.cuh"

#ifndef NBLOCKS
#define NBLOCKS 6      // transformer blocks; override with make L=12
#endif

#define E_MEMBERS 5
static const int SEEDS[E_MEMBERS] = { 1337, 2337, 3337, 4337, 5337 };

/* ===================================================================== *
 * The uncertainty decomposition MOVED to gpt2_kernels.cuh (uq_fold_batch /
 * uq_evaluate / uq_report). It was written here and it is still the
 * reference, but three of four arms could not reach it while it lived in
 * this file. Generalizing "E members" to "S samples from a sample_fn" is
 * the whole change: pass S = E and the arithmetic is identical term for
 * term, so the control arm's numbers are fixed by construction.
 *
 * The ensemble's sample_fn is the degenerate one — sample e IS member e,
 * already in VRAM, so it hands back a pointer and copies nothing.
 * ===================================================================== */
typedef struct { const float *member[E_MEMBERS]; } EnsCtx;

static const float *ens_draw(int s, void *vctx) {
    return ((EnsCtx *)vctx)->member[s];   // no sampling: this arm has no posterior
}

/* ===================================================================== *
 * eval mode — uncertainty numbers from the SAVED members, no training.
 *
 *     ./train_ensemble eval [steps] [S] [batches] [split]
 *
 * S is accepted for CLI symmetry with the other three arms and then
 * ignored, loudly: this arm's S is E_MEMBERS and cannot be raised without
 * paying for another full training run. That asymmetry is the honest cost
 * of the control arm and it is printed rather than hidden.
 * ===================================================================== */
static int ens_eval(int argc, char **argv) {
    if (uq_selftest()) return 1;
    UQArgs u = uq_parse_args(argc, argv);
    if (u.S != E_MEMBERS)
        printf("note: S is FIXED at %d for this arm (one sample per trained member);\n"
               "      the requested S=%d is ignored. The Bayesian arms can draw any S\n"
               "      from a posterior for the price of a forward pass; this one cannot\n"
               "      exceed the members it paid to train. Report that asymmetry.\n",
               E_MEMBERS, u.S);

    CUBLAS_CHECK(cublasCreate(&cublas_handle));
    Config cfg = { .B = 4, .T = 64, .C = 384, .V = VOCAB, .NH = 6, .maxT = 1024, .L = NBLOCKS };
    ParamOff po = param_offsets(cfg);
    ActOff   a  = act_offsets(cfg);
    size_t n = po.total, pb = n * sizeof(float);
    int BT = cfg.B * cfg.T;

    EnsCtx ec;
    float *d_member[E_MEMBERS];
    for (int e = 0; e < E_MEMBERS; e++) {
        char w[32]; snprintf(w, sizeof w, "member%d", e);
        CUDA_CHECK(cudaMalloc(&d_member[e], pb));
        if (ckpt_load("ens", w, d_member[e], n, cfg, u.steps) < 0) return 1;
        ec.member[e] = d_member[e];
    }

    float *d_acts; int *d_inp, *d_tgt;
    CUDA_CHECK(cudaMalloc(&d_acts, a.total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_inp, BT * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_tgt, BT * sizeof(int)));
    int   *h_inp    = (int *)malloc(BT * sizeof(int));
    int   *h_tgt    = (int *)malloc(BT * sizeof(int));
    float *h_losses = (float *)malloc(BT * sizeof(float));

    Loader ld; loader_init(&ld, uq_split_path(u.split));
    UQSpec spec = { .cfg = cfg, .S = E_MEMBERS, .nbatches = u.nbatches,
                    .arm = "ens", .split = u.split, .mix_label = "ensemble",
                    .s_fixed_by_arm = 1, .fixed_batch = 0 };
    UQResult r;
    int bad = uq_evaluate(&spec, ens_draw, &ec, &ld, d_acts, d_inp, d_tgt,
                          h_inp, h_tgt, h_losses, &r);
    if (!bad) uq_report(&spec, &r);

    free(h_inp); free(h_tgt); free(h_losses); free(ld.tokens);
    for (int e = 0; e < E_MEMBERS; e++) cudaFree(d_member[e]);
    cudaFree(d_acts); cudaFree(d_inp); cudaFree(d_tgt);
    cublasDestroy(cublas_handle);
    return bad;
}

/* ===================================================================== *
 * main — compiled out when this file is INCLUDED by the Python extension
 * (see the note over train_bnn_gpu.cu's main). ens_draw above is shared with
 * uncertain::EnsembleTrainer unchanged.
 * ===================================================================== */
#ifndef UNCERTAIN_AS_LIBRARY

int main(int argc, char **argv) {
    if (argc > 1 && argv[1][0] == 'e') return ens_eval(argc, argv);
    bool overfit = (argc > 1 && argv[1][0] == 'o');
    // ./train_ensemble bench [B] [steps] [warmup]. The CONTROL on the
    // amortization claim: this arm samples nothing, so its O(P) work is the
    // plain AdamW every trainer pays, and it must amortize for the same
    // reason. One member is timed — the arm's cost is that times E.
    //
    // AND THE CONTROL ON THE FUSION. The other three arms fuse their
    // parameter update; this one has nothing to fuse. Its whole per-step
    // O(P) work is one memset and one AdamW — 8 parameter-sized touches,
    // which is the floor for ANY Adam-trained model, Bayesian or not. Folding
    // the memset into AdamW would save a launch and no bytes, at the price of
    // editing the arm whose in-training output is this repo's reference. Left
    // alone deliberately: an unfused control is what makes the other three
    // arms' before/after numbers mean something.
    bool bench = (argc > 1 && argv[1][0] == 'b');

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s  sm_%d%d  %.1f GB\n\n", prop.name, prop.major, prop.minor,
           prop.totalGlobalMem / 1e9);
    CUBLAS_CHECK(cublasCreate(&cublas_handle));

    Config cfg = { .B = 4, .T = 64, .C = 384, .V = VOCAB, .NH = 6, .maxT = 1024, .L = NBLOCKS };
    if (bench && argc > 2) cfg.B = atoi(argv[2]);   // the swept axis; T is held
    ParamOff po = param_offsets(cfg);
    ActOff   a  = act_offsets(cfg);
    size_t n = po.total;
    int BT = cfg.B * cfg.T;   // V now lives in the shared harness's UQSpec.cfg

    const float LR = overfit ? 1e-3f : 1e-4f;   // same policy as the other arms
    const float B1 = 0.9f, B2 = 0.999f, AEPS = 1e-8f, WD = 0.0f;
    // steps: default per mode, overridable as `./bin <mode> <steps>`
    int STEPS = overfit ? 200 : 100;      // per member;
    int bench_warm = 20;
    if (bench) {
        STEPS = (argc > 3) ? atoi(argv[3]) : 200;
        if (argc > 4) bench_warm = atoi(argv[4]);
    } else if (argc > 2) STEPS = atoi(argv[2]);
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
    Bench bm; bench_init(&bm, bench, STEPS, bench_warm);

    // Out of VRAM headroom: report the point as out of range rather than
    // spending the full step count measuring a spilling allocation.
    if (bench && bm.aborted) return 3;

    // ---- train the E members, sequentially --------------------------------
    // bench times ONE member: every member's step is identical, so E copies of
    // it would cost 5x the wall clock and measure the same thing.
    for (int e = 0; e < (bench ? 1 : E_MEMBERS); e++) {
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
            bench_step_begin(&bm);
            if (!overfit) {
                loader_next(&ld, h_inp, h_tgt, BT);
                CUDA_CHECK(cudaMemcpy(d_inp, h_inp, BT * sizeof(int), cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(d_tgt, h_tgt, BT * sizeof(int), cudaMemcpyHostToDevice));
            }
            bench_mark(&bm, "data");
            CUDA_CHECK(cudaMemset(d_dw,    0, pb));
            bench_mark(&bm, "+zero_dw");
            CUDA_CHECK(cudaMemset(d_gacts, 0, a.total * sizeof(float)));
            bench_mark(&bm, "zero_gacts");

            // deterministic: forward/backward straight on the member's params
            float nll = gpt2_forward(d_acts, d_member[e], d_inp, d_tgt, cfg, h_losses);
            gpt2_backward(d_dw, d_gacts, d_acts, d_member[e], d_inp, d_tgt, cfg);
            bench_mark(&bm, "fwd_bwd");

            float bc1 = 1.0f - powf(B1, (float)step);
            float bc2 = 1.0f - powf(B2, (float)step);
            k_adamw<<<grid_n, BS>>>(d_member[e], d_dw, d_m, d_v, n, bc1, bc2,
                                    LR, B1, B2, AEPS, WD);
            bench_mark(&bm, "+adamw");

            CUDA_CHECK(cudaEventRecord(ev1));
            CUDA_CHECK(cudaEventSynchronize(ev1));
            float ms; CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
            bench_step_end(&bm, step);

            if (bench) {
                if (step == 1 || step % 20 == 0) {
                    if (bench_probe(&bm, step)) break;
                    fprintf(stderr, "  bench B=%d step %d/%d  %.2f ms\n",
                            cfg.B, step, STEPS, ms);
                }
                continue;
            }

            // standard scorecard line; sigma/kl are 0 — nothing is sampled here
            if (step == 1 || step % 10 == 0 || step == STEPS)
                printf("step %4d | nll %8.4f | kl        0.0 | sigma 0.000000 [0.000000, 0.000000] | %7.2f ms\n",
                       step, nll, ms);
        }
    }

    if (bench) {   // no ensemble eval, no checkpoints — this trained nothing
        bench_report(&bm, "ens", cfg, n);
        for (int e = 0; e < E_MEMBERS; e++) cudaFree(d_member[e]);
        free(h_inp); free(h_tgt); free(h_losses); free(h_par); free(ld.tokens);
        cudaFree(d_m); cudaFree(d_v); cudaFree(d_dw);
        cudaFree(d_acts); cudaFree(d_gacts); cudaFree(d_inp); cudaFree(d_tgt);
        cublasDestroy(cublas_handle);
        return bm.aborted ? 2 : 0;
    }

    /* ---- ensemble evaluation ---------------------------------------------
       Now the SHARED path (gpt2_kernels.cuh), with S = E_MEMBERS. Same batch
       order, same member order, same double-precision folding, same OOD seed
       — so every number this block printed before, it prints now.

       The split here is deliberately NOT DATA_VAL: this is the in-training
       eval, and its job is to stay byte-comparable with every run already in
       runs/results.csv. It reads the training file past the tokens training
       consumed (overfit re-reads its one memorized batch). For the val/test
       numbers the four-arm comparison is built on, use `./train_ensemble eval`. */
    EnsCtx ec;
    for (int e = 0; e < E_MEMBERS; e++) ec.member[e] = d_member[e];

    // Held-out: text the training never reached (training consumed the first
    // STEPS*BT tokens; overfit consumed one batch and evaluates on it).
    if (!overfit) ld.pos = (long)STEPS * BT;
    UQSpec spec = { .cfg = cfg, .S = E_MEMBERS, .nbatches = EVAL_BATCHES,
                    .arm = "ens",
                    .split = overfit ? "train[the one memorized batch]"
                                     : "train[tail, past what training read]",
                    .mix_label = "ensemble", .s_fixed_by_arm = 1,
                    .fixed_batch = overfit };
    UQResult r;
    // The correctness gate of this file, now shared by all four arms.
    if (uq_evaluate(&spec, ens_draw, &ec, &ld, d_acts, d_inp, d_tgt,
                    h_inp, h_tgt, h_losses, &r) != 0) return 1;

    printf("\nsummary: E=%d, S=E so inference FLOPs match the BNN arms | member mem %d x params\n",
           E_MEMBERS, E_MEMBERS);
    uq_report(&spec, &r);

    // final standard step line: the scorecard's "last nll" for this arm is
    // the ENSEMBLE NLPD — the number the arm actually competes on.
    printf("step %4d | nll %8.4f | kl        0.0 | sigma 0.000000 [0.000000, 0.000000] |    0.00 ms\n",
           STEPS, r.nlpd_mix);

    if (!overfit)
        for (int e = 0; e < E_MEMBERS; e++) {
            char w[32]; snprintf(w, sizeof w, "member%d", e);
            ckpt_save("ens", w, d_member[e], n, cfg, STEPS);
        }
    printf("\ndone.\n");
    for (int e = 0; e < E_MEMBERS; e++) cudaFree(d_member[e]);
    free(h_inp); free(h_tgt); free(h_losses); free(h_par); free(ld.tokens);
    cudaFree(d_m); cudaFree(d_v); cudaFree(d_dw);
    cudaFree(d_acts); cudaFree(d_gacts); cudaFree(d_inp); cudaFree(d_tgt);
    cublasDestroy(cublas_handle);
    return 0;
}

#endif  // !UNCERTAIN_AS_LIBRARY
