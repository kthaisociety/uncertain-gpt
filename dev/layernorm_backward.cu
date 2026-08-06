/*
LayerNorm backward — reduction AND scatter-add in one kernel.

Two row-wide averages must exist before any element can be written:
    dnorm_mean      = mean_c(dy[c]*weight[c])
    dnorm_xhat_mean = mean_c(dy[c]*weight[c] * xhat[c])
then
    dinp[c]   += r * (dnorm - dnorm_mean - xhat*dnorm_xhat_mean)
    dbias[c]  += dy[c]
    dweight[c]+= dy[c]*xhat

--------------------------------------------------------------------------
THE NEW PROBLEM: dweight and dbias are shared

dinp is row-private — block bt owns row bt and nobody else touches it. But
dweight and dbias are only C floats, and EVERY one of the 8192 blocks adds
into the same C slots. Two blocks doing read-add-write on the same address
at the same time lose one of the updates.

atomicAdd makes the read-modify-write indivisible. The cost is real: 8192
blocks all hammering the same 768 addresses serializes hard. This is the
honest version-1 kernel; the fix is per-block partial sums in shared memory
followed by one atomic per block instead of one per element, which cuts the
atomic count by the block size.

CONSEQUENCE FOR TESTING: atomics complete in nondeterministic order, so the
float sum for dweight/dbias is a different order every run. It will not match
the CPU bit-for-bit and it will not even match itself run to run. The
tolerance for those two is deliberately loose; dinp keeps the strict one.

Compile:  make layernorm_backward
Run:      ./layernorm_backward
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include "common.h"

extern "C" void layernorm_backward(float *dinp, float *dweight, float *dbias,
                                   float *dout, float *inp, float *weight,
                                   float *mean, float *rstd,
                                   int B, int T, int C);

__device__ inline float block_reduce_sum(float val, float *scratch) {
    int lane = threadIdx.x % warpSize;
    int wid  = threadIdx.x / warpSize;
    int nwarps = (blockDim.x + warpSize - 1) / warpSize;
    for (int off = warpSize / 2; off > 0; off >>= 1)
        val += __shfl_down_sync(0xffffffff, val, off);
    if (lane == 0) scratch[wid] = val;
    __syncthreads();
    val = (threadIdx.x < nwarps) ? scratch[threadIdx.x] : 0.0f;
    if (wid == 0) {
        for (int off = warpSize / 2; off > 0; off >>= 1)
            val += __shfl_down_sync(0xffffffff, val, off);
        if (lane == 0) scratch[0] = val;
    }
    __syncthreads();
    return scratch[0];
}

__global__ void layernorm_backward_kernel(float *dinp, float *dweight,
                                          float *dbias, const float *dout,
                                          const float *inp, const float *weight,
                                          const float *mean, const float *rstd,
                                          int C) {
    __shared__ float scratch[32];

    int bt = blockIdx.x;
    const float *x  = inp  + (size_t)bt * C;
    const float *dy = dout + (size_t)bt * C;
    float       *dx = dinp + (size_t)bt * C;
    float m = mean[bt];
    float r = rstd[bt];

    // pass 1: the two row-wide averages
    float s1 = 0.0f, s2 = 0.0f;
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        float xhat  = (x[c] - m) * r;
        float dnorm = dy[c] * weight[c];
        s1 += dnorm;
        s2 += dnorm * xhat;
    }
    float dnorm_mean = block_reduce_sum(s1, scratch) / C;
    __syncthreads();
    float dnorm_xhat_mean = block_reduce_sum(s2, scratch) / C;

    // pass 2: write all three gradients
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        float xhat  = (x[c] - m) * r;
        float dnorm = dy[c] * weight[c];
        atomicAdd(&dbias[c],   dy[c]);          // shared across all rows
        atomicAdd(&dweight[c], dy[c] * xhat);   // shared across all rows
        dx[c] += r * (dnorm - dnorm_mean - xhat * dnorm_xhat_mean);
    }
}

void layernorm_backward_gpu(float *dinp, float *dweight, float *dbias,
                            const float *dout, const float *inp,
                            const float *weight, const float *mean,
                            const float *rstd, int B, int T, int C,
                            const int block_size) {
    layernorm_backward_kernel<<<B * T, block_size>>>(dinp, dweight, dbias, dout,
                                                     inp, weight, mean, rstd, C);
    cudaCheck(cudaGetLastError());
}

int main(int argc, const char **argv) {
    (void)argc; (void)argv;
    srand(1337);
    print_device();

    int B = 8, T = 1024, C = 768;
    size_t N = (size_t)B * T * C;
    const int TOUCH = 4;

    printf("B=%d T=%d C=%d  (%d rows)  %.2f GB touched\n\n",
           B, T, C, B * T, N * 4.0 * TOUCH / 1e9);

    float *inp    = make_random_float(N);
    float *dout   = make_random_float(N);
    float *weight = make_random_float(C);
    float *mean   = make_random_float((size_t)B * T);
    float *rstd   = (float *)malloc((size_t)B * T * sizeof(float));
    for (int i = 0; i < B * T; i++) rstd[i] = 0.5f + (float)rand() / RAND_MAX;

    float *dinp0 = make_random_float(N);
    float *dw0   = make_random_float(C);
    float *db0   = make_random_float(C);
    float *dinp  = (float *)malloc(N * sizeof(float));
    float *dw    = (float *)malloc(C * sizeof(float));
    float *db    = (float *)malloc(C * sizeof(float));
    memcpy(dinp, dinp0, N * sizeof(float));
    memcpy(dw,   dw0,   C * sizeof(float));
    memcpy(db,   db0,   C * sizeof(float));

    layernorm_backward(dinp, dw, db, dout, inp, weight, mean, rstd, B, T, C);

    float *d_dinp, *d_dw, *d_db, *d_dout, *d_inp, *d_weight, *d_mean, *d_rstd;
    cudaCheck(cudaMalloc(&d_dinp,   N * sizeof(float)));
    cudaCheck(cudaMalloc(&d_dw,     C * sizeof(float)));
    cudaCheck(cudaMalloc(&d_db,     C * sizeof(float)));
    cudaCheck(cudaMalloc(&d_dout,   N * sizeof(float)));
    cudaCheck(cudaMalloc(&d_inp,    N * sizeof(float)));
    cudaCheck(cudaMalloc(&d_weight, C * sizeof(float)));
    cudaCheck(cudaMalloc(&d_mean,   (size_t)B * T * sizeof(float)));
    cudaCheck(cudaMalloc(&d_rstd,   (size_t)B * T * sizeof(float)));
    cudaCheck(cudaMemcpy(d_dout,   dout,   N * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_inp,    inp,    N * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_weight, weight, C * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_mean,   mean,   (size_t)B*T*sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_rstd,   rstd,   (size_t)B*T*sizeof(float), cudaMemcpyHostToDevice));

    int block_sizes[] = {32, 64, 128, 256, 512, 1024};
    for (size_t j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int bs = block_sizes[j];
        printf("  checking block size %d\n", bs);
        cudaCheck(cudaMemcpy(d_dinp, dinp0, N * sizeof(float), cudaMemcpyHostToDevice));
        cudaCheck(cudaMemcpy(d_dw,   dw0,   C * sizeof(float), cudaMemcpyHostToDevice));
        cudaCheck(cudaMemcpy(d_db,   db0,   C * sizeof(float), cudaMemcpyHostToDevice));
        layernorm_backward_gpu(d_dinp, d_dw, d_db, d_dout, d_inp, d_weight,
                               d_mean, d_rstd, B, T, C, bs);
        validate_result(d_dinp, dinp, "dinp",    N, 1e-5f);
        // loose: atomic summation order is nondeterministic, see header
        validate_result(d_dw,   dw,   "dweight", C, 5e-2f);
        validate_result(d_db,   db,   "dbias",   C, 5e-2f);
    }
    printf("\nAll results match. Starting benchmarks.\n\n");

    double peak = peak_bandwidth_gbs();
    for (size_t j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int bs = block_sizes[j];
        float ms = benchmark_kernel(100, layernorm_backward_gpu, d_dinp, d_dw,
                                    d_db, d_dout, d_inp, d_weight, d_mean,
                                    d_rstd, B, T, C, bs);
        double gbs = (double)N * TOUCH * sizeof(float) / (ms * 1e6);
        printf("block_size %4d | time %.4f ms | bandwidth %7.2f GB/s | %4.1f%% of peak\n",
               bs, ms, gbs, 100.0 * gbs / peak);
    }

    free(inp); free(dout); free(weight); free(mean); free(rstd);
    free(dinp0); free(dw0); free(db0); free(dinp); free(dw); free(db);
    cudaCheck(cudaFree(d_dinp)); cudaCheck(cudaFree(d_dw)); cudaCheck(cudaFree(d_db));
    cudaCheck(cudaFree(d_dout)); cudaCheck(cudaFree(d_inp)); cudaCheck(cudaFree(d_weight));
    cudaCheck(cudaFree(d_mean)); cudaCheck(cudaFree(d_rstd));
    return 0;
}
