/*
Matmul backward, via cuBLAS.

Forward:  out[b,t,oc] = sum_c inp[b,t,c] * weight[c,oc] + bias[oc]

Three separate gradients, three separate shapes:

    dinp[b,t,c]   += sum_oc dout[b,t,oc] * weight[c,oc]     (B*T, C)
    dweight[c,oc] += sum_bt inp[b,t,c]  * dout[b,t,oc]      (C, OC)
    dbias[oc]     += sum_bt dout[b,t,oc]                    (OC,)

Two of the three are GEMMs. Writing M = B*T so the shapes read cleanly:

    dinp   (M,C)  = dout (M,OC) @ weight^T (OC,C)
    dweight(C,OC) = inp^T (C,M) @ dout     (M,OC)

Note they contract over DIFFERENT axes. dinp sums over the output channels;
dweight sums over the tokens. Same two matrices, transposed differently — the
classic reason a backward pass costs about twice a forward.

dbias is not a GEMM at all: it is a column sum of dout over M rows. That gets
its own kernel, one block per output channel.

--------------------------------------------------------------------------
THE TRANSPOSE BOOKKEEPING

Everything here is row-major and cuBLAS is column-major, so a row-major (M,K)
matrix IS a column-major (K,M) in the same bytes — reading it "transposed"
costs nothing. Row-major C = A @ B becomes column-major C' = B' @ A' with M
and N swapped. To ALSO transpose one of the operands, flip its op flag and
swap its logical dims; the leading dimension always stays the row length of
how the data actually sits in memory.

    dinp    = dout @ weight^T   ->  sgemm(OP_T, OP_N, C, M, OC, weight, OC, dout, OC, dinp, C)
    dweight = inp^T @ dout      ->  sgemm(OP_N, OP_T, OC, C, M,  dout, OC,  inp,  C,  dweight, OC)

If you get one wrong, the result is usually still a valid matrix of the right
shape with completely wrong numbers — which is exactly why this file validates
against the CPU before it times anything.

beta = 1.0f everywhere, because all three gradients ACCUMULATE (+=). cuBLAS
does that for free; a beta of 0 would silently drop whatever was already there.

Compile:  make matmul_backward
Run:      ./matmul_backward
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "common.h"

extern "C" void matmul_backward(float *dinp, float *dweight, float *dbias,
                                float *inp, float *weight, float *dout,
                                int B, int T, int C, int OC);

static cublasHandle_t cublas_handle;

// dbias[oc] += sum over all M=B*T rows of dout[.,oc].
// One block per output channel; the block strides down the column and reduces.
// Column-major-ish access (stride OC between rows) is not coalesced within a
// thread, but adjacent THREADS take adjacent rows of the same column... which
// is also strided. This is the transposed-reduction problem; the fix is one
// block per TILE of columns with a shared-memory transpose. Left simple here
// because dbias is OC floats out of a 100 MB dout — it is not the bottleneck.
__global__ void dbias_kernel(float *dbias, const float *dout, int M, int OC) {
    __shared__ float scratch[32];
    int oc = blockIdx.x;
    float sum = 0.0f;
    for (int m = threadIdx.x; m < M; m += blockDim.x) sum += dout[(size_t)m * OC + oc];

    int lane = threadIdx.x % warpSize, wid = threadIdx.x / warpSize;
    int nwarps = (blockDim.x + warpSize - 1) / warpSize;
    for (int off = warpSize / 2; off > 0; off >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, off);
    if (lane == 0) scratch[wid] = sum;
    __syncthreads();
    if (wid == 0) {
        sum = (threadIdx.x < nwarps) ? scratch[threadIdx.x] : 0.0f;
        for (int off = warpSize / 2; off > 0; off >>= 1)
            sum += __shfl_down_sync(0xffffffff, sum, off);
        if (lane == 0) dbias[oc] += sum;
    }
}

void matmul_backward_gpu(float *dinp, float *dweight, float *dbias,
                         const float *inp, const float *weight,
                         const float *dout, int B, int T, int C, int OC,
                         const int block_size) {
    const int M = B * T;
    const float one = 1.0f;

    // dinp (M,C) = dout (M,OC) @ weight^T (OC,C)
    cublasStatus_t s1 = cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                    C, M, OC,
                                    &one, weight, OC, dout, OC,
                                    &one, dinp, C);

    // dweight (C,OC) = inp^T (C,M) @ dout (M,OC)
    cublasStatus_t s2 = cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T,
                                    OC, C, M,
                                    &one, dout, OC, inp, C,
                                    &one, dweight, OC);

    if (s1 || s2) { printf("cuBLAS error %d %d\n", s1, s2); exit(1); }

    dbias_kernel<<<OC, block_size>>>(dbias, dout, M, OC);
    cudaCheck(cudaGetLastError());
}

int main(int argc, const char **argv) {
    (void)argc; (void)argv;
    srand(1337);
    print_device();
    cublasCreate(&cublas_handle);

    int B = 8, T = 1024, C = 768, OC = 3072;
    size_t NI = (size_t)B * T * C;
    size_t NW = (size_t)C * OC;
    size_t NO = (size_t)B * T * OC;
    // two GEMMs of the same size as the forward
    double flops = 2.0 * 2.0 * B * T * C * OC;

    printf("M=%d C=%d OC=%d   %.1f GFLOP (two GEMMs)\n\n", B * T, C, OC, flops / 1e9);

    float *inp    = make_random_float(NI);
    float *weight = make_random_float(NW);
    float *dout   = make_random_float(NO);

    // all three gradients accumulate, so keep pristine copies
    float *dinp0 = make_random_float(NI);
    float *dw0   = make_random_float(NW);
    float *db0   = make_random_float(OC);
    float *dinp  = (float *)malloc(NI * sizeof(float));
    float *dw    = (float *)malloc(NW * sizeof(float));
    float *db    = (float *)malloc(OC * sizeof(float));
    memcpy(dinp, dinp0, NI * sizeof(float));
    memcpy(dw,   dw0,   NW * sizeof(float));
    memcpy(db,   db0,   OC * sizeof(float));

    printf("running CPU reference (slow)...\n");
    matmul_backward(dinp, dw, db, inp, weight, dout, B, T, C, OC);
    printf("done.\n\n");

    float *d_inp, *d_weight, *d_dout, *d_dinp, *d_dw, *d_db;
    cudaCheck(cudaMalloc(&d_inp,    NI * sizeof(float)));
    cudaCheck(cudaMalloc(&d_weight, NW * sizeof(float)));
    cudaCheck(cudaMalloc(&d_dout,   NO * sizeof(float)));
    cudaCheck(cudaMalloc(&d_dinp,   NI * sizeof(float)));
    cudaCheck(cudaMalloc(&d_dw,     NW * sizeof(float)));
    cudaCheck(cudaMalloc(&d_db,     OC * sizeof(float)));
    cudaCheck(cudaMemcpy(d_inp,    inp,    NI * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_weight, weight, NW * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_dout,   dout,   NO * sizeof(float), cudaMemcpyHostToDevice));

    int block_sizes[] = {256};
    for (size_t j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int bs = block_sizes[j];
        printf("  checking block size %d\n", bs);
        cudaCheck(cudaMemcpy(d_dinp, dinp0, NI * sizeof(float), cudaMemcpyHostToDevice));
        cudaCheck(cudaMemcpy(d_dw,   dw0,   NW * sizeof(float), cudaMemcpyHostToDevice));
        cudaCheck(cudaMemcpy(d_db,   db0,   OC * sizeof(float), cudaMemcpyHostToDevice));
        matmul_backward_gpu(d_dinp, d_dw, d_db, d_inp, d_weight, d_dout,
                            B, T, C, OC, bs);
        validate_result(d_dinp, dinp, "dinp",    NI, 1e-2f);
        validate_result(d_dw,   dw,   "dweight", NW, 1e-1f);   // sums over 8192 terms
        validate_result(d_db,   db,   "dbias",   OC, 1e-1f);   // sums over 8192 terms
    }
    printf("\nAll results match. Starting benchmarks.\n\n");

    for (size_t j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int bs = block_sizes[j];
        float ms = benchmark_kernel(20, matmul_backward_gpu, d_dinp, d_dw, d_db,
                                    d_inp, d_weight, d_dout, B, T, C, OC, bs);
        printf("cuBLAS x2 + dbias | time %.4f ms | %.2f TFLOP/s\n",
               ms, flops / (ms * 1e9));
    }

    free(inp); free(weight); free(dout);
    free(dinp0); free(dw0); free(db0); free(dinp); free(dw); free(db);
    cudaCheck(cudaFree(d_inp)); cudaCheck(cudaFree(d_weight)); cudaCheck(cudaFree(d_dout));
    cudaCheck(cudaFree(d_dinp)); cudaCheck(cudaFree(d_dw)); cudaCheck(cudaFree(d_db));
    cublasDestroy(cublas_handle);
    return 0;
}
