/*
Kernel for the Bayesian backward pass:

    dmu[i]  += dw[i]
    drho[i] += dw[i] * eps[i] * sigmoid(rho[i])

Push the gradient sitting on the sampled weights back onto the variational
parameters. Flat, elementwise, 1:1 — mu and rho never mix, which is why this
is two independent lines and not a reduction.

Compile:
    make bayes_backward
Run:
    ./bayes_backward

--------------------------------------------------------------------------
THE EXPENSIVE ONE

Seven buffer touches per element, 28 bytes, 2.38 GB at the real parameter
count. That is more than bayes_materialize (16 B) and more than any GPT-2
activation kernel in this directory.

Two of those seven exist purely because the op ACCUMULATES:

    dmu[i] += dw[i]     is a read AND a write
    dmu[i]  = dw[i]     would be a write alone

The += convention is inherited from every other backward op in train_bnn.c,
where it is load-bearing (a parameter used twice gets gradient from both
paths). Here it is not: bayes_backward is called once per step on a freshly
zeroed buffer. So the two reads are pure protocol cost — 680 MB per step
bought by a convention this particular op does not need.

That is a fusion target, not a kernel target. Zeroing + accumulating + the
AdamW update that follows are three passes over the same 340 MB buffers; done
as one kernel they are one pass. Measure it here first, then fuse.

Also note eps is read AGAIN here, having already been read in materialize.
Philox kills both reads, not one.

FIRST PRINCIPLES
  CGMA?        ~4 flops / 28 bytes = 0.14. Memory-bound.
  Coalesced?   thread i touches element i in all five buffers -> yes.
  Divergence?  none; expf has no branch.
  Roofline:    28 bytes/element -> 2.38 GB -> 9.03 ms at 264 GB/s.
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include "common.h"

// The CPU reference, linked from train_bnn.c (compiled with -DSKIP_MAIN).
extern "C" void bayes_backward(float *dmu, float *drho, const float *dw,
                               const float *eps, const float *rho, size_t n);

// ----------------------------------------------------------------------------
// GPU kernel

// Must match train_bnn.c's sigmoidf_ exactly.
__device__ inline float d_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__global__ void bayes_backward_kernel(float *dmu, float *drho, const float *dw,
                                      const float *eps, const float *rho,
                                      size_t n) {
    size_t i = (size_t)blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        float g = dw[i];                    // read once, used twice
        dmu[i]  += g;
        drho[i] += g * eps[i] * d_sigmoid(rho[i]);
    }
}

// ----------------------------------------------------------------------------
// kernel launcher

void bayes_backward_gpu(float *dmu, float *drho, const float *dw,
                        const float *eps, const float *rho, size_t n,
                        const int block_size) {
    const int grid_size = ceil_div((int)n, block_size);
    bayes_backward_kernel<<<grid_size, block_size>>>(dmu, drho, dw, eps, rho, n);
    cudaCheck(cudaGetLastError());
}

// ----------------------------------------------------------------------------

int main(int argc, const char **argv) {
    (void)argc; (void)argv;
    srand(1337);
    print_device();

    size_t n = 85120849;   // real parameter count of the model in train_bnn.c

    printf("n = %zu params, %.0f MB per buffer, %.2f GB touched\n\n",
           n, n * 4 / 1e6, n * 4.0 * 7 / 1e9);

    float *dw  = make_random_float(n);
    float *eps = make_random_float(n);
    float *rho = make_random_float(n);
    for (size_t i = 0; i < n; i++) rho[i] -= 5.0f;   // realistic init

    // Because the op ACCUMULATES, the starting contents of dmu/drho are part
    // of the answer. Keep a pristine copy so every run starts from the same
    // state — otherwise the second block size validates against a buffer the
    // first block size already added to.
    float *dmu0  = make_random_float(n);
    float *drho0 = make_random_float(n);
    float *dmu   = (float *)malloc(n * sizeof(float));
    float *drho  = (float *)malloc(n * sizeof(float));
    memcpy(dmu,  dmu0,  n * sizeof(float));
    memcpy(drho, drho0, n * sizeof(float));

    // CPU reference first — the thing the kernel must reproduce.
    bayes_backward(dmu, drho, dw, eps, rho, n);

    float *d_dmu, *d_drho, *d_dw, *d_eps, *d_rho;
    cudaCheck(cudaMalloc(&d_dmu,  n * sizeof(float)));
    cudaCheck(cudaMalloc(&d_drho, n * sizeof(float)));
    cudaCheck(cudaMalloc(&d_dw,   n * sizeof(float)));
    cudaCheck(cudaMalloc(&d_eps,  n * sizeof(float)));
    cudaCheck(cudaMalloc(&d_rho,  n * sizeof(float)));
    cudaCheck(cudaMemcpy(d_dw,  dw,  n * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_eps, eps, n * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_rho, rho, n * sizeof(float), cudaMemcpyHostToDevice));

    int block_sizes[] = {32, 64, 128, 256, 512, 1024};
    for (size_t j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];
        printf("  checking block size %d\n", block_size);
        cudaCheck(cudaMemcpy(d_dmu,  dmu0,  n * sizeof(float), cudaMemcpyHostToDevice));
        cudaCheck(cudaMemcpy(d_drho, drho0, n * sizeof(float), cudaMemcpyHostToDevice));
        bayes_backward_gpu(d_dmu, d_drho, d_dw, d_eps, d_rho, n, block_size);
        validate_result(d_dmu,  dmu,  "dmu",  n, 1e-5f);
        validate_result(d_drho, drho, "drho", n, 1e-5f);
    }
    printf("\nAll results match. Starting benchmarks.\n\n");

    double peak = peak_bandwidth_gbs();
    printf("roofline floor at peak: %.3f ms\n\n", n * 4.0 * 7 / (peak * 1e6));
    for (size_t j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];
        float ms = benchmark_kernel(100, bayes_backward_gpu,
                                    d_dmu, d_drho, d_dw, d_eps, d_rho, n, block_size);
        // napkin math: 5 reads + 2 writes of 4 bytes per element
        double gbs = (double)n * 7 * sizeof(float) / (ms * 1e6);
        printf("block_size %4d | time %.4f ms | bandwidth %7.2f GB/s | %4.1f%% of peak\n",
               block_size, ms, gbs, 100.0 * gbs / peak);
    }

    free(dw); free(eps); free(rho); free(dmu0); free(drho0); free(dmu); free(drho);
    cudaCheck(cudaFree(d_dmu)); cudaCheck(cudaFree(d_drho)); cudaCheck(cudaFree(d_dw));
    cudaCheck(cudaFree(d_eps)); cudaCheck(cudaFree(d_rho));
    return 0;
}
