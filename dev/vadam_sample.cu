/*
Distributional test for k_vadam_sample — Vadam lines 1-3 (Khan et al. 2018):

    sigma <- 1/sqrt(N*s + lambda)
    eps   ~  N(0, I)
    w     <- mu + sigma o eps

Modelled on train_bnn_lowrank.cu's check 3: fix (mu, s), draw NS samples, and
compare the EMPIRICAL first and second moments against the closed form. There
is nothing to compare bitwise here — the kernel's output is random by design —
so the test is statistical, and the Monte-Carlo floor is printed next to every
tolerance so a "pass" cannot be confused with an exact one.

Three things are checked, because three different bugs are possible:

  [1] mean    -> mu             a wrong mu offset, or eps with a nonzero mean
  [2] var     -> 1/(N*s+lambda) the sigma formula itself, over the full range
                                of s the real checkpoint spans
  [3] cov_ij  -> 0 for i != j   THE ONE THAT MATTERS FOR A MEAN-FIELD SAMPLER.
                                philox_normal(seed, step, index) must decorrelate
                                across index; if it did not (a dropped index
                                term, a step/index swap) every weight would be
                                perturbed by the same eps, the posterior would
                                be rank-1 instead of diagonal, and every
                                marginal variance above would STILL be correct.

As in the low-rank check, off-diagonal error is normalized by the covariance
scale sqrt(Sigma_ii*Sigma_jj) and not by |want|: the off-diagonals are zero by
construction, so dividing by |want| measures how close a true zero got to zero
and diverges no matter how right the kernel is.

THE KERNEL UNDER TEST IS THE REAL ONE — see dev/vadam_update.cu's header for
why this includes ../train_bnn_vadam.cu instead of restating it, and why the
guard is UNCERTAIN_AS_LIBRARY.

Compile:  cd dev && make vadam_sample
Run:      ./vadam_sample
*/

#define UNCERTAIN_AS_LIBRARY      // compile out the arm's main(); see above
#include "../train_bnn_vadam.cu"

int main(int argc, const char **argv) {
    (void)argc; (void)argv;
    srand(1337);

    const int    n   = 4096;      // elements sampled
    const int    SUB = 48;        // leading block used for the covariance
    const int    NS  = 20000;     // draws
    const double N      = 90000000.0;
    const double LAMBDA = 2500.0;
    const unsigned long long SEED = 20240520ull;

    // s spans the range the trained checkpoint actually occupies: s=0 gives
    // exactly the prior sigma (0.02, where 99.75% of the real checkpoint sits)
    // and s=1e-3 gives sigma = 3.3e-5, below its measured minimum.
    float *h_mu = (float *)malloc(n * sizeof(float));
    float *h_s  = (float *)malloc(n * sizeof(float));
    for (int i = 0; i < n; i++) {
        h_mu[i] = 0.02f * (((float)rand() / RAND_MAX) * 2.0f - 1.0f);
        h_s[i]  = (i % 8 == 0) ? 0.0f
                : (float)pow(10.0, -11.0 + 8.0 * ((double)rand() / RAND_MAX));
    }

    float *d_mu, *d_s, *d_w;
    CUDA_CHECK(cudaMalloc(&d_mu, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_s,  n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_w,  n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_mu, h_mu, n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_s,  h_s,  n * sizeof(float), cudaMemcpyHostToDevice));

    float  *w    = (float *)malloc(n * sizeof(float));
    double *mean = (double *)calloc(n, sizeof(double));
    double *m2   = (double *)calloc(n, sizeof(double));
    double *cov  = (double *)calloc((size_t)SUB * SUB, sizeof(double));

    for (int s = 1; s <= NS; s++) {
        k_vadam_sample<<<ceil_div(n, BS), BS>>>(d_w, d_mu, d_s, n,
                                                (float)N, (float)LAMBDA, SEED, s);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(w, d_w, n * sizeof(float), cudaMemcpyDeviceToHost));
        for (int i = 0; i < n; i++) {
            double di = (double)w[i] - h_mu[i];
            mean[i] += di;
            m2[i]   += di * di;
        }
        for (int i = 0; i < SUB; i++)
            for (int j = 0; j < SUB; j++)
                cov[i * SUB + j] += ((double)w[i] - h_mu[i]) * ((double)w[j] - h_mu[j]);
    }

    // MC floors. Variance of a sample variance over NS draws is 2/NS relative;
    // the mean's standard error is sigma/sqrt(NS), i.e. 1/sqrt(NS) in units of
    // sigma. Tolerances below are ~5 sigma of these.
    const double var_mc  = sqrt(2.0 / NS);        // 1.00% at NS = 20000
    const double mean_mc = 1.0 / sqrt((double)NS); // 0.71% of sigma
    printf("NS = %d draws   MC floor: variance %.2f%%, mean %.2f%% of sigma\n\n",
           NS, 100 * var_mc, 100 * mean_mc);

    int fails = 0;
    double worst_var = 0.0, worst_mean = 0.0;
    double wv_prior = 0.0, wv_tight = 0.0;   // split by regime, reported separately
    for (int i = 0; i < n; i++) {
        double want = 1.0 / (N * (double)h_s[i] + LAMBDA);   // sigma^2
        double have = m2[i] / NS - (mean[i] / NS) * (mean[i] / NS);
        double rel  = fabs(have - want) / want;
        if (rel > worst_var) worst_var = rel;
        if (h_s[i] == 0.0f) { if (rel > wv_prior) wv_prior = rel; }
        else                { if (rel > wv_tight) wv_tight = rel; }
        double rel_m = fabs(mean[i] / NS) / sqrt(want);       // in units of sigma
        if (rel_m > worst_mean) worst_mean = rel_m;
    }
    printf("  [1] mean -> mu               : %s (worst %.3f sigma, tol %.3f)\n",
           worst_mean < 5 * mean_mc ? "PASS" : "FAIL", worst_mean, 5 * mean_mc);
    fails += !(worst_mean < 5 * mean_mc);
    printf("  [2] var  -> 1/(N*s+lambda)   : %s (worst rel %.4f, tol %.4f)\n",
           worst_var < 5 * var_mc ? "PASS" : "FAIL", worst_var, 5 * var_mc);
    printf("        s = 0 (prior sigma 0.02) %.4f   |   s > 0 (informed) %.4f\n",
           wv_prior, wv_tight);
    fails += !(worst_var < 5 * var_mc);

    double worst_off = 0.0;
    for (int i = 0; i < SUB; i++)
        for (int j = 0; j < SUB; j++) {
            if (i == j) continue;
            double si = 1.0 / (N * (double)h_s[i] + LAMBDA);
            double sj = 1.0 / (N * (double)h_s[j] + LAMBDA);
            double c  = cov[i * SUB + j] / NS - (mean[i] / NS) * (mean[j] / NS);
            double rel = fabs(c) / sqrt(si * sj);            // = |correlation|
            if (rel > worst_off) worst_off = rel;
        }
    // Correlation of two independent streams over NS draws is O(1/sqrt(NS));
    // over SUB^2 pairs the max of that is a few sigma more.
    double off_tol = 6.0 / sqrt((double)NS);
    printf("  [3] cov_ij -> 0, i != j      : %s (worst |corr| %.4f, tol %.4f)\n",
           worst_off < off_tol ? "PASS" : "FAIL", worst_off, off_tol);
    fails += !(worst_off < off_tol);

    printf("\n%s\n", fails ? "k_vadam_sample FAILED"
                           : "k_vadam_sample draws N(mu, 1/(N*s+lambda)), diagonal.");
    free(h_mu); free(h_s); free(w); free(mean); free(m2); free(cov);
    cudaFree(d_mu); cudaFree(d_s); cudaFree(d_w);
    return fails ? 1 : 0;
}
