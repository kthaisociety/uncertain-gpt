/*
CPU reference for k_vadam_update — Vadam lines 5-8 (Khan et al. 2018, Fig. 1):

    m  <- g1*m + (1-g1) * (g + lambda*mu/N)
    s  <- g2*s + (1-g2) * (g o g)
    m^ <- m/(1-g1^t),   s^ <- s/(1-g2^t)
    mu <- mu - alpha * m^ / (sqrt(s^) + lambda/N)

Vadam was the ONE arm in this repo with no correctness check, and it is the arm
that produced the strangest checkpoint, so the coincidence had to be removed.

THE KERNEL UNDER TEST IS THE REAL ONE. This file includes ../train_bnn_vadam.cu
rather than restating the kernel, so the test cannot pass against a stale copy
while the trainer runs something else. UNCERTAIN_AS_LIBRARY is the guard the
Python extension already uses to compile out each arm's main() — one mechanism,
not a second one for tests. That is why this does NOT include dev/common.h:
gpt2_kernels.cuh defines its own cuda_check_ and ceil_div, and having both in
one translation unit is a redefinition.

-----------------------------------------------------------------------------
WHY THE GRADIENT MAGNITUDES ARE LOG-SPACED OVER 1e-9..1e0

The whole question about this kernel is whether the lambda/N term in the line-8
denominator is negligible against sqrt(s^). At the trained config lambda/N =
2500/9e7 = 2.78e-5, and in the real checkpoint the MEDIAN sqrt(s^) is 2.71e-5 —
the two are the same size. A test driven with the usual +-1 random gradients
would put sqrt(s^) ~ 0.5, five orders of magnitude above lambda/N, and would
therefore pass while testing none of the arithmetic anyone cares about.

So g is drawn log-uniformly across the range the real run actually occupies,
and the summary prints the worst error separately for the lambda/N-dominated
half. Both regimes must pass.

Compile:  cd dev && make vadam_update
Run:      ./vadam_update
*/

#define UNCERTAIN_AS_LIBRARY      // compile out the arm's main(); see above
#include "../train_bnn_vadam.cu"
#include <float.h>

// Relative tolerance on each output's own scale. --use_fast_math makes the
// device sqrt and divide approximate, so ~1e-6 is the floor; 1e-4 leaves room
// without hiding anything that would matter to the update.
#define TOL 1e-4

// Host reference. Written from the paper's Fig. 1, in double, independently of
// the kernel above — a float copy of the kernel would agree with it by
// construction and prove nothing.
//
// It also returns a SCALE for each output. Line 5 adds two signed terms, so mi
// can cancel to near zero for particular (m, g) pairs; there the relative error
// against |mi| is 100% no matter how correct the kernel is, and it measures how
// close a true zero got to zero. Same reasoning as train_bnn_lowrank.cu's
// check 3, which normalizes its off-diagonals by the covariance scale rather
// than by |want|. The honest denominator is the size of the terms that went in.
static void vadam_update_cpu(double *mu, double *m, double *s, const double *g,
                             double *msc, double *dsc, size_t n,
                             double N, double lambda, double bc1, double bc2,
                             double alpha, double g1, double g2) {
    for (size_t i = 0; i < n; i++) {
        double prior = lambda * mu[i] / N;
        double a = g1 * m[i], b = (1.0 - g1) * (g[i] + prior);
        double mi = a + b;
        double si = g2 * s[i] + (1.0 - g2) * g[i] * g[i];
        m[i] = mi;
        s[i] = si;
        double den = sqrt(si / bc2) + lambda / N;
        mu[i] -= alpha * (mi / bc1) / den;
        msc[i] = fabs(a) + fabs(b);                    // no cancellation
        dsc[i] = alpha * (msc[i] / bc1) / den;         // the step m could take
    }
}

int main(int argc, const char **argv) {
    (void)argc; (void)argv;
    srand(1337);

    // The trained config, so the constants under test are the real ones.
    const size_t n      = 1u << 20;
    const double N      = 90000000.0;      // text8 training tokens
    const double LAMBDA = 2500.0;          // 1/init_std^2, init_std = 0.02
    const double ALPHA  = 1e-4;
    const double G1 = 0.9, G2 = 0.999;
    const double LT = LAMBDA / N;
    printf("n = %zu   N = %.0f   lambda = %.0f   lambda/N = %.4e\n\n",
           n, N, LAMBDA, LT);

    float  *h_mu = (float *)malloc(n * sizeof(float));
    float  *h_m  = (float *)malloc(n * sizeof(float));
    float  *h_s  = (float *)malloc(n * sizeof(float));
    float  *h_g  = (float *)malloc(n * sizeof(float));
    double *c_mu = (double *)malloc(n * sizeof(double));
    double *c_m  = (double *)malloc(n * sizeof(double));
    double *c_s  = (double *)malloc(n * sizeof(double));
    double *c_g  = (double *)malloc(n * sizeof(double));
    double *c_ms = (double *)malloc(n * sizeof(double));   // scale for m
    double *c_ds = (double *)malloc(n * sizeof(double));   // scale for delta mu
    float  *got  = (float *)malloc(n * sizeof(float));
    double *mu0  = (double *)malloc(n * sizeof(double));

    for (size_t i = 0; i < n; i++) {
        double u = (double)rand() / RAND_MAX;
        double mag = pow(10.0, -9.0 + 9.0 * u);              // 1e-9 .. 1e0
        double sgn = (rand() & 1) ? 1.0 : -1.0;
        h_g[i]  = (float)(sgn * mag);
        h_mu[i] = 0.02f * (((float)rand() / RAND_MAX) * 2.0f - 1.0f);
        h_m[i]  = (float)(sgn * mag * 0.5);                  // m is an EMA of g
        h_s[i]  = (float)(mag * mag);                        // s is an EMA of g*g
    }

    float *d_mu, *d_m, *d_s, *d_g;
    CUDA_CHECK(cudaMalloc(&d_mu, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_m,  n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_s,  n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_g,  n * sizeof(float)));

    // Three step counts: t=1 is where the bias correction is extreme
    // (bc2 = 1e-3), t=100000 is where the checkpoint was written (bc2 = 1).
    const int steps[] = { 1, 1000, 100000 };
    int fails = 0;

    for (size_t k = 0; k < sizeof steps / sizeof *steps; k++) {
        int t = steps[k];
        double bc1 = 1.0 - pow(G1, (double)t);
        double bc2 = 1.0 - pow(G2, (double)t);

        for (size_t i = 0; i < n; i++) {
            c_mu[i] = mu0[i] = h_mu[i];
            c_m[i]  = h_m[i];
            c_s[i]  = h_s[i];
            c_g[i]  = h_g[i];
        }
        vadam_update_cpu(c_mu, c_m, c_s, c_g, c_ms, c_ds, n, N, LAMBDA,
                         bc1, bc2, ALPHA, G1, G2);

        CUDA_CHECK(cudaMemcpy(d_mu, h_mu, n * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_m,  h_m,  n * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_s,  h_s,  n * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_g,  h_g,  n * sizeof(float), cudaMemcpyHostToDevice));
        k_vadam_update<<<ceil_div((int)n, BS), BS>>>(d_mu, d_m, d_s, d_g, n,
                                                     (float)N, (float)LAMBDA,
                                                     (float)bc1, (float)bc2,
                                                     (float)ALPHA, (float)G1, (float)G2);
        CUDA_CHECK(cudaGetLastError());

        // mu itself is ~0.02 while one step moves it by ~1e-4, so comparing mu
        // would hide a 100% error in the update. The DELTA is the sharp test.
        struct { const char *name; double *ref; float *dev; double *scale;
                 int is_delta; } chk[] = {
            { "m",         c_m,  d_m,  c_ms, 0 },
            { "s",         c_s,  d_s,  c_s,  0 },   // all-positive: scale == result
            { "delta mu",  c_mu, d_mu, c_ds, 1 },
        };
        for (size_t c = 0; c < 3; c++) {
            CUDA_CHECK(cudaMemcpy(got, chk[c].dev, n * sizeof(float),
                                  cudaMemcpyDeviceToHost));
            double worst = 0.0, worst_lo = 0.0; size_t nlo = 0, iworst = 0;
            for (size_t i = 0; i < n; i++) {
                double want = chk[c].ref[i], have = got[i];
                if (chk[c].is_delta) { want -= mu0[i]; have -= mu0[i]; }
                // Budget = relative tolerance on the computation's own scale,
                // PLUS, for the delta, the floor set by storing mu in float32:
                // `mu -= delta` cannot resolve anything below half an ulp of
                // mu, so at |mu| ~ 0.02 any step under ~9.3e-10 is lost. That
                // is a property of the buffer, not of the kernel. Same shape as
                // dev/common.h's validate_result (tolerance + |ref|*FLT_EPSILON).
                double budget = TOL * chk[c].scale[i]
                              + (chk[c].is_delta ? fabs(mu0[i]) * FLT_EPSILON : 0.0);
                double rel = fabs(want - have) / (budget > 1e-300 ? budget : 1e-300);
                if (rel > worst) { worst = rel; iworst = i; }
                // lambda/N-dominated half: sqrt(s^) < lambda/N
                if (sqrt(c_s[i] / bc2) < LT) {
                    nlo++;
                    if (rel > worst_lo) worst_lo = rel;
                }
            }
            int ok = worst <= 1.0;
            printf("  t=%-6d %-9s worst err/budget %.3f | in the lambda/N-"
                   "dominated %.0f%% : %.3f  %s\n",
                   t, chk[c].name, worst, 100.0 * nlo / n, worst_lo,
                   ok ? "PASS" : "FAIL");
            // A relative error is meaningless without knowing whether the
            // quantity cancelled to near zero, so a failure always shows the
            // inputs and the size of the term that cancelled.
            if (!ok) {
                size_t i = iworst;
                double prior = LAMBDA * mu0[i] / N;
                printf("           worst at i=%zu: g %+.6e  mu %+.6e  m_in %+.6e\n"
                       "             prior lambda*mu/N %+.6e | line-5 terms sum "
                       "to %.1e of their own magnitude\n",
                       i, h_g[i], mu0[i], (double)h_m[i], prior,
                       fabs(c_m[i]) / c_ms[i]);
            }
            fails += !ok;
        }
    }

    printf("\n%s\n", fails ? "k_vadam_update FAILED"
                           : "k_vadam_update matches the CPU reference.");
    free(h_mu); free(h_m); free(h_s); free(h_g);
    free(c_mu); free(c_m); free(c_s); free(c_g); free(c_ms); free(c_ds);
    free(got); free(mu0);
    cudaFree(d_mu); cudaFree(d_m); cudaFree(d_s); cudaFree(d_g);
    return fails ? 1 : 0;
}
