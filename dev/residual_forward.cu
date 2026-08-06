/*
Kernels for the residual forward pass:  out[i] = x[i] + y[i]

This file is the TEMPLATE for every other kernel in this directory. Copy it,
change the op, keep the shape:

    header comment documenting each version
    -> GPU kernels (kernel1, kernel2, ...)
    -> one launcher per version
    -> a switch dispatching on argv[1]
    -> main: validate at every block size, THEN benchmark at every block size

Compile:
    make residual_forward
Run:
    ./residual_forward 1

version 1 is the naive one-thread-per-element port
./residual_forward 1

version 2 is a grid-stride loop (add it yourself and compare)
./residual_forward 2

--------------------------------------------------------------------------
NOTE — one deliberate difference from llm.c's dev/cuda files.

llm.c copy-pastes the CPU reference into each dev file (`gelu_forward_cpu`).
We LINK the real one out of train_bnn.c instead. That matters: a copied
reference only proves two copies of the same misunderstanding agree, and it
silently rots when the original changes. Your CPU op is already gradient-checked
and mutation-tested — check against THAT, not against a transcription of it.
--------------------------------------------------------------------------

FIRST PRINCIPLES for this op (fill these in before you optimize):
  CGMA?        2 loads + 1 store, 1 flop  -> ~0.08 flops/byte. Hopelessly
               memory-bound, which means the ONLY goal is hitting peak
               bandwidth. There is no arithmetic to optimize.
  Coalesced?   thread i touches element i -> yes, by construction.
  Divergence?  only the tail guard `if (i < N)`. One warp affected.
  Occupancy?   trivial register use; this should saturate easily.
  Roofline:    3 buffer-touches * 4 bytes = 12 bytes/element. At 264 GB/s the
               floor for N elements is  N*12/264e9  seconds. If you are not
               near that, something is wrong with the launch config, not the math.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "common.h"

// The CPU reference, linked from train_bnn.c (compiled with -DSKIP_MAIN).
extern "C" void residual_forward(float *out, float *x, float *y, int N);

// ----------------------------------------------------------------------------
// GPU kernels

// version 1: one thread per element.
__global__ void residual_forward_kernel1(float *out, const float *x,
                                         const float *y, int N) {
    // Who am I? Block 3, thread 5, with 256 threads per block -> i = 773.
    // Every thread computes a different i; together they cover 0 .. N-1.
    // This replaces the CPU's `for (int i = 0; i < N; i++)` entirely: the
    // loop counter becomes the thread's identity.
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // ceil_div rounds UP, so the last block can contain threads with no
    // element to work on. Without this they would write past the buffer.
    if (i < N) {
        out[i] = x[i] + y[i];   // the CPU loop body, unchanged
    }
}

// ----------------------------------------------------------------------------
// kernel launchers

void residual_forward1(float *out, const float *x, const float *y,
                       int N, const int block_size) {
    const int grid_size = ceil_div(N, block_size);
    residual_forward_kernel1<<<grid_size, block_size>>>(out, x, y, N);
    cudaCheck(cudaGetLastError());
}

// ----------------------------------------------------------------------------
// kernel version dispatch

void residual_forward_gpu(int kernel_num, float *out, const float *x,
                          const float *y, int N, int block_size) {
    switch (kernel_num) {
        case 1: residual_forward1(out, x, y, N, block_size); break;
        default: printf("Invalid kernel number\n"); exit(1);
    }
}

// ----------------------------------------------------------------------------

int main(int argc, const char **argv) {
    srand(1337);
    print_device();

    int B = 8, T = 1024, C = 768;
    size_t N = (size_t)B * T * C;

    int kernel_num = (argc > 1) ? atoi(argv[1]) : 1;
    printf("Using kernel %d   (N = %zu, %.0f MB per buffer)\n\n",
           kernel_num, N, N * 4 / 1e6);

    float *x   = make_random_float(N);
    float *y   = make_random_float(N);
    float *out = (float *)malloc(N * sizeof(float));

    // CPU reference first — the thing the kernel must reproduce.
    residual_forward(out, x, y, (int)N);

    float *d_x, *d_y, *d_out;
    cudaCheck(cudaMalloc(&d_x,   N * sizeof(float)));
    cudaCheck(cudaMalloc(&d_y,   N * sizeof(float)));
    cudaCheck(cudaMalloc(&d_out, N * sizeof(float)));
    cudaCheck(cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_y, y, N * sizeof(float), cudaMemcpyHostToDevice));

    // Correctness at EVERY block size before any timing. A kernel that is
    // right at 256 and wrong at 1024 usually has an indexing bug that a single
    // block size hides.
    int block_sizes[] = {32, 64, 128, 256, 512, 1024};
    for (size_t j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];
        printf("  checking block size %d\n", block_size);
        residual_forward_gpu(kernel_num, d_out, d_x, d_y, (int)N, block_size);
        validate_result(d_out, out, "out", N, 1e-5f);
    }
    printf("\nAll results match. Starting benchmarks.\n\n");

    double peak = peak_bandwidth_gbs();
    for (size_t j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];
        float ms = benchmark_kernel(100, residual_forward_gpu, kernel_num,
                                    d_out, d_x, d_y, (int)N, block_size);
        // napkin math: 2 reads + 1 write of 4 bytes per element
        double gbs = (double)N * 3 * sizeof(float) / (ms * 1e6);
        printf("block_size %4d | time %.4f ms | bandwidth %7.2f GB/s | %4.1f%% of peak\n",
               block_size, ms, gbs, 100.0 * gbs / peak);
    }

    free(x); free(y); free(out);
    cudaCheck(cudaFree(d_x)); cudaCheck(cudaFree(d_y)); cudaCheck(cudaFree(d_out));
    return 0;
}
