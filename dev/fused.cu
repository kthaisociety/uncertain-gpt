/*
Equivalence check for the FUSED parameter updates.

    make -C dev fused && ./dev/fused

An optimization paper's claim is not "it got faster", it is "it got faster and
it computes the same thing". This file is the second half. For each arm it
runs the fused kernel and the sequence of kernels it replaced on the SAME
inputs, and compares every output buffer.

THE KERNELS UNDER TEST ARE THE REAL ONES. This file includes ../train_bnn_gpu.cu
and ../train_bnn_vadam.cu and defines UNCERTAIN_AS_LIBRARY — the guard the
Python extension already uses to compile out each arm's main() — so the test
cannot pass against a stale copy while the trainer runs something else. That is
also why it does NOT include dev/common.h: gpt2_kernels.cuh defines its own
cuda_check_ and ceil_div, and having both in one translation unit is a
redefinition. Same arrangement as dev/vadam_update.cu.

-----------------------------------------------------------------------------
WHAT "THE SAME THING" MEANS HERE, AND HOW THE ONE RESIDUAL IS SETTLED

Both paths are float32 evaluating the same expressions in the same order, so
the only licensed difference is where nvcc contracts a multiply-add into an
FMA — and it does contract differently, because the fused kernel's unrolled
r-loop is a different register-pressure problem from a standalone AdamW over
a flat buffer. A tolerance of 1e-4 — right for a double reference like
dev/vadam_update.cu's — would pass a fusion that had dropped a whole term, so
that is not how this is settled.

Instead it is settled by REMOVING the licence. `make -C dev fused_nofma`
rebuilds with -fmad=false and requires EXACT BIT EQUALITY on every output.
If the two paths agree bit for bit once the compiler may not contract, then
contraction is the whole of the difference and the arithmetic is the same.
The default build keeps production flags (--use_fast_math, FMA on) and allows
a few ULP; the no-FMA build allows none. Both must pass.

    fused        production flags   worst <= FUSED_ULP_TOL ULP
    fused_nofma  -fmad=false        bit-identical, no exceptions

BBB and Vadam are bit-identical under BOTH builds — their fused kernels
happen to contract the same way. Only LowR needs the argument.

-----------------------------------------------------------------------------
THE ERROR IS NOT NORMALIZED BY |want|

BBB's mu-gradient is  g + klw*mu/sigma_p^2 : two signed terms that cancel to
near zero for particular inputs, where the relative error against |want| is
100% on perfectly correct arithmetic. Same trap dev/vadam_update.cu documents
for Vadam's line 5 and train_bnn_lowrank.cu's check 3 for its off-diagonals.
Each buffer is scored in ULP at max(|value|, the buffer's RMS), which is the
scale the computation actually ran at. See ulp_dist() for what happened when
it was scored at the value's own magnitude instead.

-----------------------------------------------------------------------------
THE INPUTS ARE THE ONES THE REAL RUN OCCUPIES

rho spans [-8, -2], so softplus(rho) — the posterior sigma — spans 3.4e-4 to
0.127, bracketing the 0.003..0.100 the trained BBB checkpoint reports. Vadam's
s is drawn so that sigma = 1/sqrt(N*s + lambda) spans its trained range too.
Gradients are log-uniform over 1e-9..1e0 rather than +-1: at the trained
config the prior term klw*mu/sp2 is ~1e-8, so a test driven at |g| ~ 1 would
never exercise the regime where the two terms are the same size.

AND ADAM'S m AND v ARE DRAWN TOGETHER, NOT INDEPENDENTLY. They are both
accumulated from the same gradient, so a real state has v ~ m^2 and the
update m/(sqrt(v)+eps) is O(1) — i.e. the step is about lr, which is the
whole point of Adam. Drawing them independently over wide log ranges
manufactures pairs like m=1, v=1e-14, where the update is 9e6*lr and the
parameter moves by 1e5 times its own magnitude in one step. That regime does
not occur, and it amplifies a 1-ULP contraction difference into something
that looks like a broken fusion. adam_state() below draws v from m.
*/

#define UNCERTAIN_AS_LIBRARY      // compile out the arms' main(); see above
#include "../train_bnn_gpu.cu"
#include "../train_bnn_vadam.cu"
#include "../train_bnn_lowrank.cu"

// With -fmad=false the compiler may not contract, so the two paths must agree
// EXACTLY. With production flags a handful of ULP is the contraction licence.
#ifdef FUSED_NO_FMAD
#define FUSED_ULP_TOL 0.0
#define FUSED_BUILD   "no-FMA (-fmad=false)"
#else
#define FUSED_ULP_TOL 8.0
#define FUSED_BUILD   "production (--use_fast_math, FMA on)"
#endif

static int failures = 0;

// Distance in ULP — but ULP AT max(|value|, the buffer's RMS), never at the
// value's own magnitude alone.
//
// This is the trap CLAUDE.md records for every numerical check in this repo,
// and it caught this file on the first run. A's update is
// A_new = A - lr*(m/(sqrt(v)+eps)), two signed terms of the same size, so
// some elements cancel to ~1e-10 out of a buffer whose RMS is ~1e-4. Scored
// at their OWN magnitude, a 6e-13 absolute difference there is 21475 ULP and
// looks like a broken fusion; scored at the scale the computation actually
// ran at, it is 0.1 ULP, which is what it is. Taking the max keeps the unit
// honest in the other direction too: an element far ABOVE the RMS is judged
// at its own magnitude, where a 1-ULP rounding really is 1 ULP.
static double ulp_dist(float x, float y, double rms) {
    if (x == y) return 0.0;
    double scale = fabs((double)x) > fabs((double)y) ? fabs((double)x) : fabs((double)y);
    if (scale < rms) scale = rms;
    int e; frexp(scale, &e);
    double ulp = ldexp(1.0, e - 24);        // float32 has 24 significand bits
    return fabs((double)x - (double)y) / ulp;
}

// ---------------------------------------------------------------------------
// Compare two device buffers. Scored in ULP, and against the buffer's own RMS
// where a value cancelled to zero — never against |want|. See the header.
// ---------------------------------------------------------------------------
static void cmp(const char *what, const float *d_a, const float *d_b, size_t n) {
    float *a = (float *)malloc(n * sizeof(float));
    float *b = (float *)malloc(n * sizeof(float));
    CUDA_CHECK(cudaMemcpy(a, d_a, n * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(b, d_b, n * sizeof(float), cudaMemcpyDeviceToHost));

    double sq = 0.0;
    for (size_t i = 0; i < n; i++) sq += (double)a[i] * a[i];
    double rms = sqrt(sq / (double)n);
    if (rms == 0.0) rms = 1.0;         // an all-zero buffer (dw) is exact or it is not

    double worst = 0.0; size_t worst_i = 0, exact = 0, nan_seen = 0;
    for (size_t i = 0; i < n; i++) {
        if (isnan(a[i]) != isnan(b[i])) { nan_seen++; continue; }
        if (a[i] == b[i]) { exact++; continue; }
        double e = ulp_dist(a[i], b[i], rms);
        if (e > worst) { worst = e; worst_i = i; }
    }
    int bad = (worst > FUSED_ULP_TOL) || nan_seen;
    failures += bad;
    printf("  %-10s  worst %7.2f ULP  bit-identical %6.2f%%  %s\n",
           what, worst, 100.0 * (double)exact / (double)n, bad ? "FAIL" : "ok");
    if (bad)
        printf("      at [%zu]: unfused %.9g  fused %.9g   (%zu NaN mismatch)\n",
               worst_i, a[worst_i], b[worst_i], nan_seen);
    free(a); free(b);
}

// Host RNG, so both paths start from byte-identical inputs.
static float frand(void) { return (float)rand() / (float)RAND_MAX; }
static float nrand(void) {
    float u1 = (frand() + 1e-7f), u2 = frand();
    return sqrtf(-2.0f * logf(u1)) * cosf(6.2831853f * u2);
}
// Log-uniform over [lo, hi] with a random sign — the range the trained run
// actually occupies, not the +-1 a lazy test would use.
static float logu(float lo, float hi) {
    float m = expf(logf(lo) + frand() * (logf(hi) - logf(lo)));
    return frand() < 0.5f ? -m : m;
}

// One plausible Adam state for a parameter whose gradients are of size |g|.
// m and v come from the SAME gradient stream, so v tracks m^2 and the update
// m/(sqrt(v)+eps) stays O(1). See the header for what independent draws do.
static void adam_state(float g, float *m, float *v) {
    float mag = fabsf(g) * (0.5f + frand());        // m is a decayed mean of g
    *m = frand() < 0.5f ? -mag : mag;
    float w = 0.5f + frand();                       // v is a decayed mean of g^2
    *v = mag * mag * w * w;
}

static float *dev_dup(const float *h, size_t n) {
    float *d; CUDA_CHECK(cudaMalloc(&d, n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d, h, n * sizeof(float), cudaMemcpyHostToDevice));
    return d;
}

/* ===================================================================== *
 * BBB: k_bbb_step_fused  vs  memset x3 + k_bayes_backward + k_kl_backward
 *                            + k_adamw x2 + k_bayes_sample
 * ===================================================================== */
static void check_bbb(size_t n, float klw, int step) {
    const float SP = 0.1f, LR = 1e-4f, B1 = 0.9f, B2 = 0.999f;
    const float AEPS = 1e-8f, WD = 0.0f;
    const unsigned long long SEED = 20240520ull;
    float bc1 = 1.0f - powf(B1, (float)step), bc2 = 1.0f - powf(B2, (float)step);

    float *h_mu = (float *)malloc(n * 4), *h_rho = (float *)malloc(n * 4);
    float *h_mmu = (float *)malloc(n * 4), *h_vmu = (float *)malloc(n * 4);
    float *h_mr = (float *)malloc(n * 4), *h_vr = (float *)malloc(n * 4);
    float *h_dw = (float *)malloc(n * 4);
    srand(4242);
    for (size_t i = 0; i < n; i++) {
        h_mu[i]  = 0.02f * nrand();
        h_rho[i] = -8.0f + 6.0f * frand();          // sigma 3.4e-4 .. 0.127
        h_dw[i]  = logu(1e-9f, 1e0f);
        adam_state(h_dw[i], &h_mmu[i], &h_vmu[i]);
        adam_state(h_dw[i], &h_mr[i],  &h_vr[i]);
    }

    // --- path A: the kernels the fusion replaced, in the loop's own order.
    float *a_mu = dev_dup(h_mu, n), *a_rho = dev_dup(h_rho, n);
    float *a_mmu = dev_dup(h_mmu, n), *a_vmu = dev_dup(h_vmu, n);
    float *a_mr = dev_dup(h_mr, n), *a_vr = dev_dup(h_vr, n);
    float *a_dw = dev_dup(h_dw, n);
    float *a_dmu, *a_drho, *a_w;
    CUDA_CHECK(cudaMalloc(&a_dmu, n * 4));
    CUDA_CHECK(cudaMalloc(&a_drho, n * 4));
    CUDA_CHECK(cudaMalloc(&a_w, n * 4));
    int g = ceil_div((int)n, BS);

    CUDA_CHECK(cudaMemset(a_dmu, 0, n * 4));
    CUDA_CHECK(cudaMemset(a_drho, 0, n * 4));
    k_bayes_backward<<<g, BS>>>(a_dmu, a_drho, a_dw, a_rho, n, SEED, step);
    if (klw > 0.0f)
        k_kl_backward<<<g, BS>>>(a_dmu, a_drho, a_mu, a_rho, n, SP, klw);
    k_adamw<<<g, BS>>>(a_mu,  a_dmu,  a_mmu, a_vmu, n, bc1, bc2, LR, B1, B2, AEPS, WD);
    k_adamw<<<g, BS>>>(a_rho, a_drho, a_mr,  a_vr,  n, bc1, bc2, LR, B1, B2, AEPS, WD);
    // the next step's sample, off the parameters just written
    k_bayes_sample<<<g, BS>>>(a_w, a_mu, a_rho, n, SEED, step + 1);
    CUDA_CHECK(cudaMemset(a_dw, 0, n * 4));         // the next step's memset
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());

    // --- path B: one kernel.
    float *b_mu = dev_dup(h_mu, n), *b_rho = dev_dup(h_rho, n);
    float *b_mmu = dev_dup(h_mmu, n), *b_vmu = dev_dup(h_vmu, n);
    float *b_mr = dev_dup(h_mr, n), *b_vr = dev_dup(h_vr, n);
    float *b_dw = dev_dup(h_dw, n);
    float *b_w; CUDA_CHECK(cudaMalloc(&b_w, n * 4));
    k_bbb_step_fused<<<g, BS>>>(b_mu, b_rho, b_mmu, b_vmu, b_mr, b_vr, b_w,
                                b_dw, n, SEED, step, SP, klw, bc1, bc2,
                                LR, B1, B2, AEPS, WD);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());

    printf("BBB  n=%zu klw=%.0e step=%d\n", n, klw, step);
    cmp("mu",    a_mu,  b_mu,  n);
    cmp("rho",   a_rho, b_rho, n);
    cmp("m_mu",  a_mmu, b_mmu, n);
    cmp("v_mu",  a_vmu, b_vmu, n);
    cmp("m_rho", a_mr,  b_mr,  n);
    cmp("v_rho", a_vr,  b_vr,  n);
    cmp("w",     a_w,   b_w,   n);
    cmp("dw",    a_dw,  b_dw,  n);

    cudaFree(a_mu); cudaFree(a_rho); cudaFree(a_mmu); cudaFree(a_vmu);
    cudaFree(a_mr); cudaFree(a_vr); cudaFree(a_dw); cudaFree(a_dmu);
    cudaFree(a_drho); cudaFree(a_w);
    cudaFree(b_mu); cudaFree(b_rho); cudaFree(b_mmu); cudaFree(b_vmu);
    cudaFree(b_mr); cudaFree(b_vr); cudaFree(b_dw); cudaFree(b_w);
    free(h_mu); free(h_rho); free(h_mmu); free(h_vmu); free(h_mr);
    free(h_vr); free(h_dw);
}

/* ===================================================================== *
 * Vadam: k_vadam_step_fused  vs  memset + k_vadam_update + k_vadam_sample
 * ===================================================================== */
static void check_vadam(size_t n, int step) {
    const float N = 9e7f, LAMBDA = 2500.0f, LR = 1e-4f, G1 = 0.9f, G2 = 0.999f;
    const unsigned long long SEED = 20240520ull;
    float bc1 = 1.0f - powf(G1, (float)step), bc2 = 1.0f - powf(G2, (float)step);

    float *h_mu = (float *)malloc(n * 4), *h_m = (float *)malloc(n * 4);
    float *h_s = (float *)malloc(n * 4), *h_dw = (float *)malloc(n * 4);
    srand(1234);
    for (size_t i = 0; i < n; i++) {
        h_mu[i] = 0.02f * nrand();
        h_m[i]  = logu(1e-9f, 1e0f);   // Vadam's m and s are checked separately
                                       // in dev/vadam_update.cu; here only the
                                       // fusion is under test, and it was
                                       // bit-identical in every regime tried.
        // s enters only as N*s + lambda. Spanning 1e-14..1e-2 puts N*s from
        // 1e-6 (prior-dominated, sigma = 0.02) to 1e6 (data-dominated,
        // sigma = 1e-3) — the whole range the trained checkpoint reports.
        h_s[i]  = fabsf(logu(1e-14f, 1e-2f));
        h_dw[i] = logu(1e-9f, 1e0f);
    }

    float *a_mu = dev_dup(h_mu, n), *a_m = dev_dup(h_m, n);
    float *a_s = dev_dup(h_s, n), *a_dw = dev_dup(h_dw, n);
    float *a_w; CUDA_CHECK(cudaMalloc(&a_w, n * 4));
    int g = ceil_div((int)n, BS);

    k_vadam_update<<<g, BS>>>(a_mu, a_m, a_s, a_dw, n, N, LAMBDA,
                              bc1, bc2, LR, G1, G2);
    k_vadam_sample<<<g, BS>>>(a_w, a_mu, a_s, n, N, LAMBDA, SEED, step + 1);
    CUDA_CHECK(cudaMemset(a_dw, 0, n * 4));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());

    float *b_mu = dev_dup(h_mu, n), *b_m = dev_dup(h_m, n);
    float *b_s = dev_dup(h_s, n), *b_dw = dev_dup(h_dw, n);
    float *b_w; CUDA_CHECK(cudaMalloc(&b_w, n * 4));
    k_vadam_step_fused<<<g, BS>>>(b_mu, b_m, b_s, b_dw, b_w, n, N, LAMBDA,
                                  bc1, bc2, LR, G1, G2, SEED, step);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());

    printf("Vadam  n=%zu step=%d\n", n, step);
    cmp("mu", a_mu, b_mu, n);
    cmp("m",  a_m,  b_m,  n);
    cmp("s",  a_s,  b_s,  n);
    cmp("w",  a_w,  b_w,  n);
    cmp("dw", a_dw, b_dw, n);

    cudaFree(a_mu); cudaFree(a_m); cudaFree(a_s); cudaFree(a_dw); cudaFree(a_w);
    cudaFree(b_mu); cudaFree(b_m); cudaFree(b_s); cudaFree(b_dw); cudaFree(b_w);
    free(h_mu); free(h_m); free(h_s); free(h_dw);
}

/* ===================================================================== *
 * LowR: k_lowrank_step_fused  vs  memset x2 + k_lowrank_backward
 *                                 + k_lowrank_kl_grad + k_adamw x2
 *
 * One synthetic layer covering the whole buffer, which is what the real
 * per-layer launch does with its own (poff, aoff, boff). Minv only has to
 * be the SAME matrix on both paths for an equivalence test, but it is built
 * near I anyway so the magnitudes are the ones the run actually sees.
 * ===================================================================== */
template <int R>
static void check_lowrank(int P, float klw, int step) {
    const float LAMBDA = 100.0f, LR = 1e-5f, B1 = 0.9f, B2 = 0.999f;
    const float AEPS = 1e-8f, WD = 0.0f;
    const unsigned long long SEED = 20240520ull;
    float bc1 = 1.0f - powf(B1, (float)step), bc2 = 1.0f - powf(B2, (float)step);
    size_t nP = (size_t)P, nA = (size_t)P * R;

    float *h_th = (float *)malloc(nP * 4), *h_dw = (float *)malloc(nP * 4);
    float *h_rh = (float *)malloc(nP * 4), *h_md = (float *)malloc(nP * 4);
    float *h_vd = (float *)malloc(nP * 4);
    float *h_A = (float *)malloc(nA * 4), *h_mA = (float *)malloc(nA * 4);
    float *h_vA = (float *)malloc(nA * 4);
    float h_Minv[R * R], h_z[R];
    srand(777);
    for (int k = 0; k < R; k++) h_z[k] = nrand();
    // rhod so that dvec = softplus(rhod) spans 1e-4 .. 1.5e-2, bracketing the
    // 1.07e-3 .. 1.5e-2 the trained LowR8 checkpoint reports.
    for (size_t i = 0; i < nP; i++) {
        h_th[i] = 0.02f * nrand();
        h_rh[i] = -9.0f + 5.0f * frand();
        h_dw[i] = logu(1e-9f, 1e0f);
        adam_state(h_dw[i], &h_md[i], &h_vd[i]);
    }
    for (size_t i = 0; i < nA; i++) {
        h_A[i]  = 1e-4f * nrand();
        // A's gradient is g*z[k], so its Adam state is scaled by |z|, not |g|.
        adam_state(h_dw[i / R] * h_z[i % R], &h_mA[i], &h_vA[i]);
    }

    for (int a = 0; a < R; a++)
        for (int b = 0; b < R; b++)
            h_Minv[a * R + b] = (a == b) ? 1.0f + 0.05f * frand() : 0.01f * nrand();

    float *d_z, *d_Mi;
    CUDA_CHECK(cudaMalloc(&d_z, R * 4));   CUDA_CHECK(cudaMalloc(&d_Mi, R * R * 4));
    CUDA_CHECK(cudaMemcpy(d_z, h_z, R * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Mi, h_Minv, R * R * 4, cudaMemcpyHostToDevice));

    // --- path A
    float *a_A = dev_dup(h_A, nA), *a_mA = dev_dup(h_mA, nA), *a_vA = dev_dup(h_vA, nA);
    float *a_rh = dev_dup(h_rh, nP), *a_md = dev_dup(h_md, nP), *a_vd = dev_dup(h_vd, nP);
    float *a_dw = dev_dup(h_dw, nP), *a_th = dev_dup(h_th, nP);
    float *a_dA, *a_drh;
    CUDA_CHECK(cudaMalloc(&a_dA, nA * 4)); CUDA_CHECK(cudaMalloc(&a_drh, nP * 4));
    int g = ceil_div(P, BS);

    CUDA_CHECK(cudaMemset(a_dA, 0, nA * 4));
    CUDA_CHECK(cudaMemset(a_drh, 0, nP * 4));
    k_lowrank_backward<R><<<g, BS>>>(a_dA, a_drh, a_dw, a_rh, d_z, 0, 0, 0, P,
                                     SEED, step);
    if (klw > 0.0f)
        k_lowrank_kl_grad<R><<<g, BS>>>(a_dw, a_dA, a_drh, a_th, a_A, a_rh, d_Mi,
                                        0, 0, 0, P, LAMBDA, klw);
    k_adamw<<<ceil_div((int)nA, BS), BS>>>(a_A, a_dA, a_mA, a_vA, nA,
                                           bc1, bc2, LR, B1, B2, AEPS, WD);
    k_adamw<<<ceil_div((int)nP, BS), BS>>>(a_rh, a_drh, a_md, a_vd, nP,
                                           bc1, bc2, LR, B1, B2, AEPS, WD);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());

    // --- path B
    float *b_A = dev_dup(h_A, nA), *b_mA = dev_dup(h_mA, nA), *b_vA = dev_dup(h_vA, nA);
    float *b_rh = dev_dup(h_rh, nP), *b_md = dev_dup(h_md, nP), *b_vd = dev_dup(h_vd, nP);
    float *b_dw = dev_dup(h_dw, nP), *b_th = dev_dup(h_th, nP);
    k_lowrank_step_fused<R><<<g, BS>>>(b_dw, b_A, b_rh, b_mA, b_vA, b_md, b_vd,
                                       b_dw, b_th, d_Mi, d_z, 0, 0, 0, P,
                                       LAMBDA, klw, SEED, step,
                                       bc1, bc2, LR, B1, B2, AEPS, WD);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());

    printf("LowR  R=%d P=%d klw=%.0e step=%d\n", R, P, klw, step);
    cmp("A",     a_A,  b_A,  nA);
    cmp("m_A",   a_mA, b_mA, nA);
    cmp("v_A",   a_vA, b_vA, nA);
    cmp("rhod",  a_rh, b_rh, nP);
    cmp("m_rhod", a_md, b_md, nP);
    cmp("v_rhod", a_vd, b_vd, nP);
    cmp("dtheta", a_dw, b_dw, nP);   // the KL's dKL/dmu contribution

    cudaFree(a_A); cudaFree(a_mA); cudaFree(a_vA); cudaFree(a_rh);
    cudaFree(a_md); cudaFree(a_vd); cudaFree(a_dw); cudaFree(a_th);
    cudaFree(a_dA); cudaFree(a_drh);
    cudaFree(b_A); cudaFree(b_mA); cudaFree(b_vA); cudaFree(b_rh);
    cudaFree(b_md); cudaFree(b_vd); cudaFree(b_dw); cudaFree(b_th);
    cudaFree(d_z); cudaFree(d_Mi);
    free(h_th); free(h_dw); free(h_rh); free(h_md); free(h_vd);
    free(h_A); free(h_mA); free(h_vA);
}

int main(void) {
    printf("build: %s   tolerance %.0f ULP\n\n",
           FUSED_BUILD, (double)FUSED_ULP_TOL);
    const size_t n = 1u << 20;    // 1M elements: enough for the tails to show

    // step matters: the Adam bias corrections differ, and step 1 is the one
    // the priming path has to agree with.
    check_bbb(n, 1.0f / 9e7f, 1);      // the derived KL weight, first step
    check_bbb(n, 1.0f / 9e7f, 137);    // and a steady-state one
    check_bbb(n, 0.0f, 137);           // overfit mode: the KL launch is skipped
    check_vadam(n, 1);
    check_vadam(n, 137);
    // Both the binary's rank and one it is not built at, so the template is
    // exercised rather than just the RANK macro's instantiation.
    check_lowrank<8>(1 << 18, 1.0f / 9e7f, 137);
    check_lowrank<8>(1 << 18, 0.0f, 137);        // overfit mode: no KL launch
    check_lowrank<4>(1 << 18, 1.0f / 9e7f, 1);

    printf("\n[%s] %s\n", FUSED_BUILD,
           failures ? "FUSION CHANGED THE MATHEMATICS — do not ship"
                    : "fused == unfused on every output.");
    return failures != 0;
}
