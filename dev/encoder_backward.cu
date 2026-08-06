/*
Encoder backward — pure scatter-add, and the sharpest example of why atomics
exist.

    dwte[inp[b,t], c] += dout[b,t,c]
    dwpe[t, c]        += dout[b,t,c]

The forward was a GATHER: many tokens read the same wte row, which is
harmless. The backward is the transpose — a SCATTER: many tokens write the
same wte row. Every repeated token in the batch is a collision, and common
tokens ("the") repeat constantly.

Without atomicAdd this races and silently drops updates. The failure does not
crash; it just produces slightly wrong gradients, which is far worse.

Contention is very uneven here:
  dwpe  — every one of the B sequences hits position t, so exactly B=8
          collisions per address. Mild.
  dwte  — depends entirely on token frequency. Rare tokens: no contention.
          Frequent tokens: hundreds. This is the same skew that makes
          embedding-gradient kernels annoying in every real framework.

Compile:  make encoder_backward
Run:      ./encoder_backward
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include "common.h"

extern "C" void encoder_backward(float *dwte, float *dwpe, float *dout,
                                 int *inp, int B, int T, int C);

__global__ void encoder_backward_kernel(float *dwte, float *dwpe,
                                        const float *dout, const int *inp,
                                        int B, int T, int C) {
    size_t idx = (size_t)blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < (size_t)B * T * C) {
        int c  = idx % C;
        int bt = idx / C;
        int t  = bt % T;
        int ix = inp[bt];
        float g = dout[idx];
        atomicAdd(&dwte[(size_t)ix * C + c], g);
        atomicAdd(&dwpe[(size_t)t  * C + c], g);
    }
}

void encoder_backward_gpu(float *dwte, float *dwpe, const float *dout,
                          const int *inp, int B, int T, int C,
                          const int block_size) {
    const int grid_size = ceil_div(B * T * C, block_size);
    encoder_backward_kernel<<<grid_size, block_size>>>(dwte, dwpe, dout, inp,
                                                       B, T, C);
    cudaCheck(cudaGetLastError());
}

int main(int argc, const char **argv) {
    (void)argc; (void)argv;
    srand(1337);
    print_device();

    int B = 8, T = 1024, C = 768, V = 50257;
    size_t N = (size_t)B * T * C;

    printf("B=%d T=%d C=%d V=%d  (%zu scatter-adds x2)\n\n", B, T, C, V, N);

    int   *inp  = make_random_int(B * T, V);
    float *dout = make_random_float(N);
    float *dwte0 = make_random_float((size_t)V * C);
    float *dwpe0 = make_random_float((size_t)T * C);
    float *dwte  = (float *)malloc((size_t)V * C * sizeof(float));
    float *dwpe  = (float *)malloc((size_t)T * C * sizeof(float));
    memcpy(dwte, dwte0, (size_t)V * C * sizeof(float));
    memcpy(dwpe, dwpe0, (size_t)T * C * sizeof(float));

    encoder_backward(dwte, dwpe, dout, inp, B, T, C);

    int   *d_inp;
    float *d_dout, *d_dwte, *d_dwpe;
    cudaCheck(cudaMalloc(&d_inp,  (size_t)B * T * sizeof(int)));
    cudaCheck(cudaMalloc(&d_dout, N * sizeof(float)));
    cudaCheck(cudaMalloc(&d_dwte, (size_t)V * C * sizeof(float)));
    cudaCheck(cudaMalloc(&d_dwpe, (size_t)T * C * sizeof(float)));
    cudaCheck(cudaMemcpy(d_inp,  inp,  (size_t)B*T*sizeof(int),   cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_dout, dout, N * sizeof(float),         cudaMemcpyHostToDevice));

    int block_sizes[] = {32, 64, 128, 256, 512, 1024};
    for (size_t j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int bs = block_sizes[j];
        printf("  checking block size %d\n", bs);
        cudaCheck(cudaMemcpy(d_dwte, dwte0, (size_t)V*C*sizeof(float), cudaMemcpyHostToDevice));
        cudaCheck(cudaMemcpy(d_dwpe, dwpe0, (size_t)T*C*sizeof(float), cudaMemcpyHostToDevice));
        encoder_backward_gpu(d_dwte, d_dwpe, d_dout, d_inp, B, T, C, bs);
        // loose: atomic summation order is nondeterministic
        validate_result(d_dwte, dwte, "dwte", (size_t)V * C, 1e-3f);
        validate_result(d_dwpe, dwpe, "dwpe", (size_t)T * C, 1e-3f);
    }
    printf("\nAll results match. Starting benchmarks.\n\n");

    double peak = peak_bandwidth_gbs();
    for (size_t j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int bs = block_sizes[j];
        float ms = benchmark_kernel(100, encoder_backward_gpu, d_dwte, d_dwpe,
                                    d_dout, d_inp, B, T, C, bs);
        // read dout + 2 atomic read-modify-writes = 5 touches
        double gbs = (double)N * 5 * sizeof(float) / (ms * 1e6);
        printf("block_size %4d | time %.4f ms | effective %7.2f GB/s | %4.1f%% of peak\n",
               bs, ms, gbs, 100.0 * gbs / peak);
    }

    free(inp); free(dout); free(dwte0); free(dwpe0); free(dwte); free(dwpe);
    cudaCheck(cudaFree(d_inp)); cudaCheck(cudaFree(d_dout));
    cudaCheck(cudaFree(d_dwte)); cudaCheck(cudaFree(d_dwpe));
    return 0;
}
