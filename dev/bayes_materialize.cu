/*
Kernel for the reparameterization trick:

    w[i] = mu[i] + softplus(rho[i]) * eps[i]

Build the sampled weight vector from the variational parameters and an
already-drawn eps. Flat, elementwise, 1:1 — layout is irrelevant, only the
total count matters.

Compile:
    make bayes_materialize
Run:
    ./bayes_materialize

--------------------------------------------------------------------------
WHY THIS ONE MATTERS MORE THAN THE LAST THREE

residual/gelu/encoder run at the ACTIVATION count (B*T*C = 6.3M here). This
one runs at the PARAMETER count — 85,120,849 — and it is pure overhead that
the deterministic model does not pay at all. Every millisecond here is a
millisecond of the Bayesian tax.

Four buffers touched, 16 bytes/element:
    read  mu       340 MB
    read  rho      340 MB
    read  eps      340 MB   <- the one to attack
    write w        340 MB
                  ------
                  1.36 GB per sample, per step

At 264 GB/s peak that floors at 5.2 ms. Whatever this measures is the number
Philox has to beat: regenerating eps from (seed, index) instead of reading it
deletes one 340 MB read AND the 340 MB write that filled it earlier — 25% of
the traffic in this kernel, before counting the buffer you stop allocating.

gelu_forward already told you the arithmetic for that will be ~free.

FIRST PRINCIPLES
  CGMA?        ~4 flops (exp, log1p, mul, add) / 16 bytes = 0.25. Memory-bound.
  Coalesced?   thread i touches element i in all four buffers -> yes.
  Divergence?  softplus branches on x > 20. Real divergence, but only in warps
               that straddle the threshold — rho init is small and negative, so
               in practice ~every lane takes the log1pf path.
  Roofline:    16 bytes/element. Expect the block-size curve to look like
               gelu's (arithmetic present -> peaks late, around 512) rather
               than residual's (peaks at 128).
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "common.h"

// The CPU reference, linked from train_bnn.c (compiled with -DSKIP_MAIN).
extern "C" void bayes_materialize(float *w, const float *mu, const float *rho,
                                  const float *eps, size_t n);

// ----------------------------------------------------------------------------
// GPU kernel

// Must match train_bnn.c's softplus EXACTLY, including the x > 20 shortcut —
// otherwise the two disagree in the tail and validate_result blames your
// indexing for what is really a numerics mismatch.
__device__ inline float d_softplus(float x) {
    return x > 20.0f ? x : log1pf(expf(x));
}

__global__ void bayes_materialize_kernel(float *w, const float *mu,
                                         const float *rho, const float *eps,
                                         size_t n) {
    size_t i = (size_t)blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        w[i] = mu[i] + d_softplus(rho[i]) * eps[i];
    }
}

// ----------------------------------------------------------------------------
// kernel launcher

void bayes_materialize_gpu(float *w, const float *mu, const float *rho,
                           const float *eps, size_t n, const int block_size) {
    const int grid_size = ceil_div((int)n, block_size);
    bayes_materialize_kernel<<<grid_size, block_size>>>(w, mu, rho, eps, n);
    cudaCheck(cudaGetLastError());
}

// ----------------------------------------------------------------------------

int main(int argc, const char **argv) {
    (void)argc; (void)argv;
    srand(1337);
    print_device();

    // The real parameter count of the model in train_bnn.c at C=768, V=50257,
    // 12 layers. Not a made-up benchmark size — this is what a step costs.
    size_t n = 85120849;

    printf("n = %zu params, %.0f MB per buffer, %.2f GB touched\n\n",
           n, n * 4 / 1e6, n * 4.0 * 4 / 1e9);

    // rho near -5 is the real init: softplus(-5) ~ 0.0067, a small sigma.
    float *mu  = make_random_float(n);
    float *rho = make_random_float(n);
    for (size_t i = 0; i < n; i++) rho[i] -= 5.0f;
    float *eps = make_random_float(n);
    float *w   = (float *)malloc(n * sizeof(float));

    // CPU reference first — the thing the kernel must reproduce.
    bayes_materialize(w, mu, rho, eps, n);

    float *d_mu, *d_rho, *d_eps, *d_w;
    cudaCheck(cudaMalloc(&d_mu,  n * sizeof(float)));
    cudaCheck(cudaMalloc(&d_rho, n * sizeof(float)));
    cudaCheck(cudaMalloc(&d_eps, n * sizeof(float)));
    cudaCheck(cudaMalloc(&d_w,   n * sizeof(float)));
    cudaCheck(cudaMemcpy(d_mu,  mu,  n * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_rho, rho, n * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_eps, eps, n * sizeof(float), cudaMemcpyHostToDevice));

    int block_sizes[] = {32, 64, 128, 256, 512, 1024};
    for (size_t j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];
        printf("  checking block size %d\n", block_size);
        cudaCheck(cudaMemset(d_w, 0, n * sizeof(float)));
        bayes_materialize_gpu(d_w, d_mu, d_rho, d_eps, n, block_size);
        validate_result(d_w, w, "w", n, 1e-5f);
    }
    printf("\nAll results match. Starting benchmarks.\n\n");

    double peak = peak_bandwidth_gbs();
    printf("roofline floor at peak: %.3f ms\n\n", n * 4.0 * 4 / (peak * 1e6));
    for (size_t j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];
        float ms = benchmark_kernel(100, bayes_materialize_gpu,
                                    d_w, d_mu, d_rho, d_eps, n, block_size);
        // napkin math: 3 reads + 1 write of 4 bytes per element
        double gbs = (double)n * 4 * sizeof(float) / (ms * 1e6);
        printf("block_size %4d | time %.4f ms | bandwidth %7.2f GB/s | %4.1f%% of peak\n",
               block_size, ms, gbs, 100.0 * gbs / peak);
    }

    free(mu); free(rho); free(eps); free(w);
    cudaCheck(cudaFree(d_mu));  cudaCheck(cudaFree(d_rho));
    cudaCheck(cudaFree(d_eps)); cudaCheck(cudaFree(d_w));
    return 0;
}
