/*
Residual backward. The local derivative of x+y is exactly 1 on both branches,
so the gradient passes straight through to both inputs.

    dx[i] += dout[i]
    dy[i] += dout[i]

Traffic: read dout + read/write dx, dy = 5 touches = 20 B/element.
Activation-sized, not parameter-sized.

Worth noticing against residual_forward: the forward moved 12 B/element, the
backward moves 20. Backward passes are usually the more expensive direction,
and the reason is the += — accumulation turns each output into a load AND a
store.

Compile:  make residual_backward
Run:      ./residual_backward
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include "common.h"

extern "C" void residual_backward(float *dx, float *dy, float *dout, int N);

__global__ void residual_backward_kernel(float *dx, float *dy,
                                         const float *dout, int N) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N) {
        float g = dout[i];
        dx[i] += g;
        dy[i] += g;
    }
}

void residual_backward_gpu(float *dx, float *dy, const float *dout, int N,
                           const int block_size) {
    const int grid_size = ceil_div(N, block_size);
    residual_backward_kernel<<<grid_size, block_size>>>(dx, dy, dout, N);
    cudaCheck(cudaGetLastError());
}

int main(int argc, const char **argv) {
    (void)argc; (void)argv;
    srand(1337);
    print_device();

    int B = 8, T = 1024, C = 768;
    size_t N = (size_t)B * T * C;
    const int TOUCH = 5;

    printf("N = %zu, %.2f GB touched\n\n", N, N * 4.0 * TOUCH / 1e9);

    float *dout = make_random_float(N);
    float *dx0  = make_random_float(N);
    float *dy0  = make_random_float(N);
    float *dx   = (float *)malloc(N * sizeof(float));
    float *dy   = (float *)malloc(N * sizeof(float));
    memcpy(dx, dx0, N * sizeof(float));
    memcpy(dy, dy0, N * sizeof(float));

    residual_backward(dx, dy, dout, (int)N);

    float *d_dx, *d_dy, *d_dout;
    cudaCheck(cudaMalloc(&d_dx,   N * sizeof(float)));
    cudaCheck(cudaMalloc(&d_dy,   N * sizeof(float)));
    cudaCheck(cudaMalloc(&d_dout, N * sizeof(float)));
    cudaCheck(cudaMemcpy(d_dout, dout, N * sizeof(float), cudaMemcpyHostToDevice));

    int block_sizes[] = {32, 64, 128, 256, 512, 1024};
    for (size_t j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int bs = block_sizes[j];
        printf("  checking block size %d\n", bs);
        cudaCheck(cudaMemcpy(d_dx, dx0, N * sizeof(float), cudaMemcpyHostToDevice));
        cudaCheck(cudaMemcpy(d_dy, dy0, N * sizeof(float), cudaMemcpyHostToDevice));
        residual_backward_gpu(d_dx, d_dy, d_dout, (int)N, bs);
        validate_result(d_dx, dx, "dx", N, 1e-5f);
        validate_result(d_dy, dy, "dy", N, 1e-5f);
    }
    printf("\nAll results match. Starting benchmarks.\n\n");

    double peak = peak_bandwidth_gbs();
    printf("roofline floor at peak: %.3f ms\n\n", N * 4.0 * TOUCH / (peak * 1e6));
    for (size_t j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int bs = block_sizes[j];
        float ms = benchmark_kernel(100, residual_backward_gpu, d_dx, d_dy,
                                    d_dout, (int)N, bs);
        double gbs = (double)N * TOUCH * sizeof(float) / (ms * 1e6);
        printf("block_size %4d | time %.4f ms | bandwidth %7.2f GB/s | %4.1f%% of peak\n",
               bs, ms, gbs, 100.0 * gbs / peak);
    }

    free(dout); free(dx0); free(dy0); free(dx); free(dy);
    cudaCheck(cudaFree(d_dx)); cudaCheck(cudaFree(d_dy)); cudaCheck(cudaFree(d_dout));
    return 0;
}
