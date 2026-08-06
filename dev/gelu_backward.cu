/*
GELU backward. Unlike residual, the local derivative depends on x, so the
saved forward INPUT is needed — that is why this op reads an extra buffer.

    u     = k*(x + 0.044715 x^3),  k = sqrt(2/pi)
    local = 0.5*(1+tanh u) + 0.5*x*(1 - tanh^2 u)*k*(1 + 3*0.044715 x^2)
    dinp[i] += local * dout[i]

Traffic: read inp, dout + read/write dinp = 4 touches = 16 B/element.
Arithmetic: ~15 flops including a tanhf. CGMA ~0.9 — the highest in the
directory so far, and still nowhere near compute-bound.

Compile:  make gelu_backward
Run:      ./gelu_backward
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include "common.h"

extern "C" void gelu_backward(float *dinp, float *inp, float *dout, int N);

__global__ void gelu_backward_kernel(float *dinp, const float *inp,
                                     const float *dout, int N) {
    const float k = 0.7978845608f;
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N) {
        float x = inp[i];
        float u = k * (x + 0.044715f * x * x * x);
        float tanh_u = tanhf(u);
        float sech2  = 1.0f - tanh_u * tanh_u;
        float du_dx  = k * (1.0f + 3.0f * 0.044715f * x * x);
        float local_grad = 0.5f * (1.0f + tanh_u) + 0.5f * x * sech2 * du_dx;
        dinp[i] += local_grad * dout[i];
    }
}

void gelu_backward_gpu(float *dinp, const float *inp, const float *dout,
                       int N, const int block_size) {
    const int grid_size = ceil_div(N, block_size);
    gelu_backward_kernel<<<grid_size, block_size>>>(dinp, inp, dout, N);
    cudaCheck(cudaGetLastError());
}

int main(int argc, const char **argv) {
    (void)argc; (void)argv;
    srand(1337);
    print_device();

    int B = 8, T = 1024, C = 768;
    size_t N = (size_t)B * T * C;
    const int TOUCH = 4;

    printf("N = %zu, %.2f GB touched\n\n", N, N * 4.0 * TOUCH / 1e9);

    float *inp   = make_random_float(N);
    float *dout  = make_random_float(N);
    float *dinp0 = make_random_float(N);
    float *dinp  = (float *)malloc(N * sizeof(float));
    memcpy(dinp, dinp0, N * sizeof(float));

    gelu_backward(dinp, inp, dout, (int)N);

    float *d_dinp, *d_inp, *d_dout;
    cudaCheck(cudaMalloc(&d_dinp, N * sizeof(float)));
    cudaCheck(cudaMalloc(&d_inp,  N * sizeof(float)));
    cudaCheck(cudaMalloc(&d_dout, N * sizeof(float)));
    cudaCheck(cudaMemcpy(d_inp,  inp,  N * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_dout, dout, N * sizeof(float), cudaMemcpyHostToDevice));

    int block_sizes[] = {32, 64, 128, 256, 512, 1024};
    for (size_t j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int bs = block_sizes[j];
        printf("  checking block size %d\n", bs);
        cudaCheck(cudaMemcpy(d_dinp, dinp0, N * sizeof(float), cudaMemcpyHostToDevice));
        gelu_backward_gpu(d_dinp, d_inp, d_dout, (int)N, bs);
        // Looser than 1e-5: GPU tanhf under --use_fast_math is not bit-identical
        // to the host's, same reason as gelu_forward.
        validate_result(d_dinp, dinp, "dinp", N, 1e-4f);
    }
    printf("\nAll results match. Starting benchmarks.\n\n");

    double peak = peak_bandwidth_gbs();
    printf("roofline floor at peak: %.3f ms\n\n", N * 4.0 * TOUCH / (peak * 1e6));
    for (size_t j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int bs = block_sizes[j];
        float ms = benchmark_kernel(100, gelu_backward_gpu, d_dinp, d_inp,
                                    d_dout, (int)N, bs);
        double gbs = (double)N * TOUCH * sizeof(float) / (ms * 1e6);
        printf("block_size %4d | time %.4f ms | bandwidth %7.2f GB/s | %4.1f%% of peak\n",
               bs, ms, gbs, 100.0 * gbs / peak);
    }

    free(inp); free(dout); free(dinp0); free(dinp);
    cudaCheck(cudaFree(d_dinp)); cudaCheck(cudaFree(d_inp)); cudaCheck(cudaFree(d_dout));
    return 0;
}
