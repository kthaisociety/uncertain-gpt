/*
Matmul forward, via cuBLAS.

    out[b,t,oc] = sum_c inp[b,t,c] * weight[c,oc] + bias[oc]

~85% of the model's FLOPs, and the FIRST kernel in this directory that is
compute-bound rather than memory-bound:

    FLOPs = 2*B*T*C*OC = 38.7 GFLOP
    bytes = (B*T*C + C*OC + B*T*OC)*4 = 135 MB
    CGMA  = 287 flops/byte

Everything else here has been under 1.5. That is why this one is borrowed
rather than written: at CGMA 287 the winning kernel is decided by register
blocking, shared-memory tiling and tensor cores, and cuBLAS already does all
three at ~95% of achievable. Writing it yourself is a real education but it
contributes nothing a BNN needs — the matmul is identical whether the weights
came from mu+sigma*eps or straight out of a file.

--------------------------------------------------------------------------
THE ROW-MAJOR / COLUMN-MAJOR TRICK

cuBLAS is column-major; C is row-major. Rather than transposing anything,
exploit that a row-major (M,K) matrix IS a column-major (K,M) matrix in the
same bytes. So to get row-major C = A@B, ask cuBLAS for column-major
C' = B'@A' with M and N swapped — no data moves, it is purely a relabelling
of the same memory.

    cublasSgemm(N, N, OC, B*T, C, ... weight, OC, inp, C, ... out, OC)
                      ^^  ^^^^  ^
                      N   M     K   <- swapped on purpose

Bias is prefilled into `out` and picked up with beta=1, so the GEMM adds to
it instead of overwriting — one fewer pass over 100 MB.

Compile:  make matmul_forward
Run:      ./matmul_forward
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "common.h"

extern "C" void matmul_forward(float *out, float *inp, float *weight,
                               float *bias, int B, int T, int C, int OC);

static cublasHandle_t cublas_handle;

// Broadcast bias across all B*T rows so the GEMM can accumulate onto it.
__global__ void fill_bias_kernel(float *out, const float *bias, int OC, size_t N) {
    size_t idx = (size_t)blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N) out[idx] = bias[idx % OC];
}

void matmul_forward_gpu(float *out, const float *inp, const float *weight,
                        const float *bias, int B, int T, int C, int OC,
                        const int block_size) {
    size_t N = (size_t)B * T * OC;
    fill_bias_kernel<<<ceil_div((int)N, block_size), block_size>>>(out, bias, OC, N);
    cudaCheck(cudaGetLastError());

    const float alpha = 1.0f, beta = 1.0f;   // beta=1 -> add onto the bias
    cublasStatus_t st = cublasSgemm(cublas_handle,
                                    CUBLAS_OP_N, CUBLAS_OP_N,
                                    OC, B * T, C,
                                    &alpha,
                                    weight, OC,
                                    inp,    C,
                                    &beta,
                                    out,    OC);
    if (st != CUBLAS_STATUS_SUCCESS) { printf("cuBLAS error %d\n", st); exit(1); }
}

int main(int argc, const char **argv) {
    (void)argc; (void)argv;
    srand(1337);
    print_device();
    cublasCreate(&cublas_handle);

    // The MLP up-projection: the biggest matmul in the model.
    int B = 8, T = 1024, C = 768, OC = 3072;
    size_t N = (size_t)B * T * OC;
    double flops = 2.0 * B * T * C * OC;

    printf("(%d x %d) @ (%d x %d)   %.1f GFLOP   CGMA %.0f\n\n",
           B * T, C, C, OC, flops / 1e9,
           flops / (((double)B*T*C + (double)C*OC + (double)N) * 4));

    float *inp    = make_random_float((size_t)B * T * C);
    float *weight = make_random_float((size_t)C * OC);
    float *bias   = make_random_float(OC);
    float *out    = (float *)malloc(N * sizeof(float));

    matmul_forward(out, inp, weight, bias, B, T, C, OC);

    float *d_inp, *d_weight, *d_bias, *d_out;
    cudaCheck(cudaMalloc(&d_inp,    (size_t)B * T * C * sizeof(float)));
    cudaCheck(cudaMalloc(&d_weight, (size_t)C * OC * sizeof(float)));
    cudaCheck(cudaMalloc(&d_bias,   OC * sizeof(float)));
    cudaCheck(cudaMalloc(&d_out,    N * sizeof(float)));
    cudaCheck(cudaMemcpy(d_inp,    inp,    (size_t)B*T*C*sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_weight, weight, (size_t)C*OC*sizeof(float),  cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_bias,   bias,   OC * sizeof(float),          cudaMemcpyHostToDevice));

    int block_sizes[] = {256};   // only feeds the bias prefill; cuBLAS picks its own
    for (size_t j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int bs = block_sizes[j];
        printf("  checking block size %d\n", bs);
        cudaCheck(cudaMemset(d_out, 0, N * sizeof(float)));
        matmul_forward_gpu(d_out, d_inp, d_weight, d_bias, B, T, C, OC, bs);
        // 768-term dot products in a different summation order than the CPU's
        // sequential loop; relative agreement is what matters here.
        validate_result(d_out, out, "out", N, 1e-2f);
    }
    printf("\nAll results match. Starting benchmarks.\n\n");

    for (size_t j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int bs = block_sizes[j];
        float ms = benchmark_kernel(50, matmul_forward_gpu, d_out, d_inp,
                                    d_weight, d_bias, B, T, C, OC, bs);
        printf("cuBLAS sgemm  | time %.4f ms | %.2f TFLOP/s\n", ms, flops / (ms * 1e9));
    }

    free(inp); free(weight); free(bias); free(out);
    cudaCheck(cudaFree(d_inp)); cudaCheck(cudaFree(d_weight));
    cudaCheck(cudaFree(d_bias)); cudaCheck(cudaFree(d_out));
    cublasDestroy(cublas_handle);
    return 0;
}
