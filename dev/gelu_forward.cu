/*
Kernels for the GELU forward pass.

    GELU(x) = 0.5x * (1 + tanh( sqrt(2/pi) * (x + 0.044715 x^3) ))

Compile:
    make gelu_forward
Run:
    ./gelu_forward 1

version 1 is the naive one-thread-per-element port
./gelu_forward 1

--------------------------------------------------------------------------
WHY THIS ONE AFTER residual_forward

Structurally identical — flat, elementwise, one thread per element. The only
difference is that the body has real arithmetic in it instead of a single add:
a cube, a tanh, four multiplies.

That makes it the cheapest possible experiment for one question:

    does adding arithmetic to a memory-bound kernel cost anything?

Predict before you run it. residual moved 12 bytes per element and hit 90% of
peak. GELU moves only 8 bytes per element (1 read + 1 write instead of 2 reads
+ 1 write) but does ~10x the math. If the GPU is truly bandwidth-bound, the
extra flops should be FREE — hidden entirely behind the memory wait — and
you should still land near 90%.

If that holds, you have measured principle 1 directly, and you have the
justification for the Philox trick later: regenerating eps costs ~30 integer
ops per element, and this experiment tells you whether that is free.

--------------------------------------------------------------------------
FIRST PRINCIPLES
  CGMA?        ~10 flops / 8 bytes = 1.25. Still memory-bound, but 15x
               higher intensity than residual. That is the whole point of
               running it.
  Coalesced?   thread i touches element i -> yes.
  Divergence?  only the tail guard.
  Occupancy?   tanhf costs registers. Watch whether the block-size curve
               looks different from residual's — if high block sizes fall
               off, registers are capping occupancy.
  Roofline:    2 buffer-touches * 4 bytes = 8 bytes/element.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "common.h"

// The CPU reference, linked from train_bnn.c (compiled with -DSKIP_MAIN).
extern "C" void gelu_forward(float *out, float *inp, int N);

// ----------------------------------------------------------------------------
// GPU kernels

// version 1: one thread per element.
__global__ void gelu_forward_kernel1(float *out, const float *inp, int N) {
    // Step 1 — index. The four launch built-ins are camelCase:
    //   blockDim.x  threads per block   (your host `block_size`)
    //   blockIdx.x  which block am I
    //   threadIdx.x which thread inside that block
    //   gridDim.x   how many blocks total
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    // Step 2 — guard. ceil_div rounds UP, so the last block can hold threads
    // with i >= N. Without this they read and write past both buffers.
    if (i < N) {
        // Step 3 — the CPU loop body, unchanged.
        const float k = 0.7978845608f;          // sqrt(2/pi)
        float x = inp[i];
        float inner = k * (x + 0.044715f * x * x * x);
        out[i] = 0.5f * x * (1.0f + tanhf(inner));
    }
}

// ----------------------------------------------------------------------------
// kernel launchers

void gelu_forward1(float *out, const float *inp, int N, const int block_size) {
    const int grid_size = ceil_div(N, block_size);
    gelu_forward_kernel1<<<grid_size, block_size>>>(out, inp, N);
    cudaCheck(cudaGetLastError());
}

// ----------------------------------------------------------------------------
// kernel version dispatch

void gelu_forward_gpu(int kernel_num, float *out, const float *inp,
                      int N, int block_size) {
    switch (kernel_num) {
        case 1: gelu_forward1(out, inp, N, block_size); break;
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

    float *inp = make_random_float(N);
    float *out = (float *)malloc(N * sizeof(float));

    // CPU reference first — the thing the kernel must reproduce.
    gelu_forward(out, inp, (int)N);

    float *d_inp, *d_out;
    cudaCheck(cudaMalloc(&d_inp, N * sizeof(float)));
    cudaCheck(cudaMalloc(&d_out, N * sizeof(float)));
    cudaCheck(cudaMemcpy(d_inp, inp, N * sizeof(float), cudaMemcpyHostToDevice));

    int block_sizes[] = {32, 64, 128, 256, 512, 1024};
    for (size_t j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];
        printf("  checking block size %d\n", block_size);
        gelu_forward_gpu(kernel_num, d_out, d_inp, (int)N, block_size);
        // Looser than residual's 1e-5: tanhf on the GPU is not bit-identical
        // to the host's, and --use_fast_math widens that further. This is a
        // real difference between the two machines, not a bug in your kernel.
        validate_result(d_out, out, "out", N, 1e-4f);
    }
    printf("\nAll results match. Starting benchmarks.\n\n");

    double peak = peak_bandwidth_gbs();
    for (size_t j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];
        float ms = benchmark_kernel(100, gelu_forward_gpu, kernel_num,
                                    d_out, d_inp, (int)N, block_size);
        // napkin math: 1 read + 1 write of 4 bytes per element
        double gbs = (double)N * 2 * sizeof(float) / (ms * 1e6);
        printf("block_size %4d | time %.4f ms | bandwidth %7.2f GB/s | %4.1f%% of peak\n",
               block_size, ms, gbs, 100.0 * gbs / peak);
    }

    free(inp); free(out);
    cudaCheck(cudaFree(d_inp)); cudaCheck(cudaFree(d_out));
    return 0;
}
