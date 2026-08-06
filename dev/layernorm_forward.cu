/*
LayerNorm forward — the first REDUCTION in this directory.

    per token row of C floats:
        m = mean(x)
        r = 1/sqrt(var(x) + eps)
        o[c] = (x[c] - m) * r * weight[c] + bias[c]

--------------------------------------------------------------------------
WHY THIS IS A DIFFERENT KIND OF KERNEL

Every kernel so far was 1:1 — thread i read element i and wrote element i,
and no thread needed to know anything about any other thread. LayerNorm
breaks that: the value written at position c depends on ALL 768 values in
the row, through the mean and the variance.

So threads have to COOPERATE, and that means a new mapping:

    one BLOCK per token row (grid = B*T)
    the threads inside that block split the row's C elements between them,
    then combine their partial sums

Combining is done in two stages, cheapest first:

  1. __shfl_down_sync — lanes inside one warp hand each other values through
     the register file directly. No memory of any kind is involved. 5 steps
     collapse 32 lanes to 1.
  2. __shared__ memory — the one survivor per warp writes to a 32-float
     scratchpad visible to the whole block, then warp 0 reduces that.

__syncthreads() between the stages is not optional: without it, warp 0 can
read the scratchpad before warp 7 has written to it. That is a race, and it
will look like a wrong answer at some block sizes and a right one at others.

The row is read TWICE from global (once for the mean, once for the variance).
That is the honest first version. A one-pass version using sum and sum-of-
squares together is the obvious next move, and it halves the read traffic.

Traffic (2-pass): read x twice, read weight+bias, write out, write mean+rstd
                  ~3 x 4 B/element = 12 B/element.

Compile:  make layernorm_forward
Run:      ./layernorm_forward
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "common.h"

extern "C" void layernorm_forward(float *out, float *mean, float *rstd,
                                  float *inp, float *weight, float *bias,
                                  int B, int T, int C);

// ----------------------------------------------------------------------------
// Sum `val` across every thread in the block. Returns the total in EVERY
// thread (not just thread 0) so callers do not need a second broadcast.
__device__ inline float block_reduce_sum(float val, float *scratch) {
    int lane = threadIdx.x % warpSize;
    int wid  = threadIdx.x / warpSize;
    int nwarps = (blockDim.x + warpSize - 1) / warpSize;

    // stage 1: within the warp, through the register file
    for (int off = warpSize / 2; off > 0; off >>= 1)
        val += __shfl_down_sync(0xffffffff, val, off);

    // stage 2: one value per warp, through shared memory
    if (lane == 0) scratch[wid] = val;
    __syncthreads();                       // every warp must have written

    val = (threadIdx.x < nwarps) ? scratch[threadIdx.x] : 0.0f;
    if (wid == 0) {
        for (int off = warpSize / 2; off > 0; off >>= 1)
            val += __shfl_down_sync(0xffffffff, val, off);
        if (lane == 0) scratch[0] = val;   // park the total
    }
    __syncthreads();                       // every thread must see it
    return scratch[0];
}

__global__ void layernorm_forward_kernel(float *out, float *mean, float *rstd,
                                         const float *inp, const float *weight,
                                         const float *bias, int T, int C) {
    __shared__ float scratch[32];          // one slot per warp, max 1024/32
    const float eps = 1e-5f;

    int bt = blockIdx.x;                   // this block owns one token row
    const float *x = inp + (size_t)bt * C;
    float       *o = out + (size_t)bt * C;

    // 1. mean
    float sum = 0.0f;
    for (int c = threadIdx.x; c < C; c += blockDim.x) sum += x[c];
    float m = block_reduce_sum(sum, scratch) / C;
    __syncthreads();                       // scratch is about to be reused

    // 2. variance
    float varsum = 0.0f;
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        float d = x[c] - m;
        varsum += d * d;
    }
    float r = rsqrtf(block_reduce_sum(varsum, scratch) / C + eps);

    // 3. file the readings away for backward, then normalize
    if (threadIdx.x == 0) { mean[bt] = m; rstd[bt] = r; }
    for (int c = threadIdx.x; c < C; c += blockDim.x)
        o[c] = (x[c] - m) * r * weight[c] + bias[c];
}

void layernorm_forward_gpu(float *out, float *mean, float *rstd,
                           const float *inp, const float *weight,
                           const float *bias, int B, int T, int C,
                           const int block_size) {
    // ONE BLOCK PER ROW — the grid is the token count, not the element count.
    layernorm_forward_kernel<<<B * T, block_size>>>(out, mean, rstd, inp,
                                                    weight, bias, T, C);
    cudaCheck(cudaGetLastError());
}

int main(int argc, const char **argv) {
    (void)argc; (void)argv;
    srand(1337);
    print_device();

    int B = 8, T = 1024, C = 768;
    size_t N = (size_t)B * T * C;
    const int TOUCH = 3;

    printf("B=%d T=%d C=%d  (%d rows of %d)  %.2f GB touched\n\n",
           B, T, C, B * T, C, N * 4.0 * TOUCH / 1e9);

    float *inp    = make_random_float(N);
    float *weight = make_random_float(C);
    float *bias   = make_random_float(C);
    float *out    = (float *)malloc(N * sizeof(float));
    float *mean   = (float *)malloc((size_t)B * T * sizeof(float));
    float *rstd   = (float *)malloc((size_t)B * T * sizeof(float));

    layernorm_forward(out, mean, rstd, inp, weight, bias, B, T, C);

    float *d_out, *d_mean, *d_rstd, *d_inp, *d_weight, *d_bias;
    cudaCheck(cudaMalloc(&d_out,    N * sizeof(float)));
    cudaCheck(cudaMalloc(&d_mean,   (size_t)B * T * sizeof(float)));
    cudaCheck(cudaMalloc(&d_rstd,   (size_t)B * T * sizeof(float)));
    cudaCheck(cudaMalloc(&d_inp,    N * sizeof(float)));
    cudaCheck(cudaMalloc(&d_weight, C * sizeof(float)));
    cudaCheck(cudaMalloc(&d_bias,   C * sizeof(float)));
    cudaCheck(cudaMemcpy(d_inp,    inp,    N * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_weight, weight, C * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_bias,   bias,   C * sizeof(float), cudaMemcpyHostToDevice));

    int block_sizes[] = {32, 64, 128, 256, 512, 1024};
    for (size_t j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int bs = block_sizes[j];
        printf("  checking block size %d\n", bs);
        cudaCheck(cudaMemset(d_out, 0, N * sizeof(float)));
        layernorm_forward_gpu(d_out, d_mean, d_rstd, d_inp, d_weight, d_bias,
                              B, T, C, bs);
        validate_result(d_out,  out,  "out",  N, 1e-5f);
        validate_result(d_mean, mean, "mean", (size_t)B * T, 1e-5f);
        validate_result(d_rstd, rstd, "rstd", (size_t)B * T, 1e-5f);
    }
    printf("\nAll results match. Starting benchmarks.\n\n");

    double peak = peak_bandwidth_gbs();
    printf("roofline floor at peak: %.3f ms\n\n", N * 4.0 * TOUCH / (peak * 1e6));
    for (size_t j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int bs = block_sizes[j];
        float ms = benchmark_kernel(100, layernorm_forward_gpu, d_out, d_mean,
                                    d_rstd, d_inp, d_weight, d_bias, B, T, C, bs);
        double gbs = (double)N * TOUCH * sizeof(float) / (ms * 1e6);
        printf("block_size %4d | time %.4f ms | bandwidth %7.2f GB/s | %4.1f%% of peak\n",
               bs, ms, gbs, 100.0 * gbs / peak);
    }

    free(inp); free(weight); free(bias); free(out); free(mean); free(rstd);
    cudaCheck(cudaFree(d_out)); cudaCheck(cudaFree(d_mean)); cudaCheck(cudaFree(d_rstd));
    cudaCheck(cudaFree(d_inp)); cudaCheck(cudaFree(d_weight)); cudaCheck(cudaFree(d_bias));
    return 0;
}
