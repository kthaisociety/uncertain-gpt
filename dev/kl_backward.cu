/*
KL backward — the prior's pull on the variational parameters.

    sigma   = softplus(rho)
    dmu[i]  += weight * mu[i] / sp^2
    drho[i] += weight * (sigma/sp^2 - 1/sigma) * sigmoid(rho[i])

Flat, elementwise, 1:1. Parameter-sized (85.1M), so this is Bayesian tax.

Traffic: read mu, rho + read/write dmu, drho = 6 touches = 24 B/element.
Note the -1/sigma branch: that term is what makes certainty expensive and
stops sigma collapsing to zero. It is also the only division in the kernel.

Compile:  make kl_backward
Run:      ./kl_backward
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include "common.h"

extern "C" void kl_backward(float *dmu, float *drho, const float *mu,
                            const float *rho, size_t n, float sigma_prior,
                            float weight);

// Must match train_bnn.c exactly.
__device__ inline float d_softplus(float x) { return x > 20.0f ? x : log1pf(expf(x)); }
__device__ inline float d_sigmoid(float x)  { return 1.0f / (1.0f + expf(-x)); }

__global__ void kl_backward_kernel(float *dmu, float *drho, const float *mu,
                                   const float *rho, size_t n,
                                   float sigma_prior, float weight) {
    size_t i = (size_t)blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        float sp2 = sigma_prior * sigma_prior;
        float r = rho[i];
        float sigma = d_softplus(r);
        dmu[i]  += weight * (mu[i] / sp2);
        float dsigma = sigma / sp2 - 1.0f / sigma;
        drho[i] += weight * dsigma * d_sigmoid(r);
    }
}

void kl_backward_gpu(float *dmu, float *drho, const float *mu, const float *rho,
                     size_t n, float sigma_prior, float weight,
                     const int block_size) {
    const int grid_size = ceil_div((int)n, block_size);
    kl_backward_kernel<<<grid_size, block_size>>>(dmu, drho, mu, rho, n,
                                                  sigma_prior, weight);
    cudaCheck(cudaGetLastError());
}

int main(int argc, const char **argv) {
    (void)argc; (void)argv;
    srand(1337);
    print_device();

    size_t n = 85120849;
    float sigma_prior = 0.1f, weight = 1e-3f;
    const int TOUCH = 6;

    printf("n = %zu params, %.2f GB touched\n\n", n, n * 4.0 * TOUCH / 1e9);

    float *mu  = make_random_float(n);
    float *rho = make_random_float(n);
    for (size_t i = 0; i < n; i++) rho[i] -= 5.0f;

    // Accumulating op: the starting contents are part of the answer.
    float *dmu0  = make_random_float(n);
    float *drho0 = make_random_float(n);
    float *dmu   = (float *)malloc(n * sizeof(float));
    float *drho  = (float *)malloc(n * sizeof(float));
    memcpy(dmu,  dmu0,  n * sizeof(float));
    memcpy(drho, drho0, n * sizeof(float));

    kl_backward(dmu, drho, mu, rho, n, sigma_prior, weight);

    float *d_dmu, *d_drho, *d_mu, *d_rho;
    cudaCheck(cudaMalloc(&d_dmu,  n * sizeof(float)));
    cudaCheck(cudaMalloc(&d_drho, n * sizeof(float)));
    cudaCheck(cudaMalloc(&d_mu,   n * sizeof(float)));
    cudaCheck(cudaMalloc(&d_rho,  n * sizeof(float)));
    cudaCheck(cudaMemcpy(d_mu,  mu,  n * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_rho, rho, n * sizeof(float), cudaMemcpyHostToDevice));

    int block_sizes[] = {32, 64, 128, 256, 512, 1024};
    for (size_t j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int bs = block_sizes[j];
        printf("  checking block size %d\n", bs);
        cudaCheck(cudaMemcpy(d_dmu,  dmu0,  n * sizeof(float), cudaMemcpyHostToDevice));
        cudaCheck(cudaMemcpy(d_drho, drho0, n * sizeof(float), cudaMemcpyHostToDevice));
        kl_backward_gpu(d_dmu, d_drho, d_mu, d_rho, n, sigma_prior, weight, bs);
        validate_result(d_dmu,  dmu,  "dmu",  n, 1e-5f);
        validate_result(d_drho, drho, "drho", n, 1e-5f);
    }
    printf("\nAll results match. Starting benchmarks.\n\n");

    double peak = peak_bandwidth_gbs();
    printf("roofline floor at peak: %.3f ms\n\n", n * 4.0 * TOUCH / (peak * 1e6));
    for (size_t j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int bs = block_sizes[j];
        float ms = benchmark_kernel(100, kl_backward_gpu, d_dmu, d_drho, d_mu,
                                    d_rho, n, sigma_prior, weight, bs);
        double gbs = (double)n * TOUCH * sizeof(float) / (ms * 1e6);
        printf("block_size %4d | time %.4f ms | bandwidth %7.2f GB/s | %4.1f%% of peak\n",
               bs, ms, gbs, 100.0 * gbs / peak);
    }

    free(mu); free(rho); free(dmu0); free(drho0); free(dmu); free(drho);
    cudaCheck(cudaFree(d_dmu)); cudaCheck(cudaFree(d_drho));
    cudaCheck(cudaFree(d_mu));  cudaCheck(cudaFree(d_rho));
    return 0;
}
