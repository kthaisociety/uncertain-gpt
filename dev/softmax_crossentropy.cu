/*
Softmax + cross-entropy, fused, one row per block.

    per row of V logits:
        max  = max_v(logit[v])                 <- reduction 1
        sum  = sum_v exp(logit[v] - max)       <- reduction 2
        prob[v] = exp(logit[v] - max) / sum
        loss = -log(prob[target])

Two reductions, and the first one is a MAX rather than a sum — same shuffle
skeleton, different combiner. The max subtraction is not an optimization: V
logits can exceed 88, expf overflows to inf, and the row becomes nan. Shift
by the max and the largest exponent is exactly exp(0)=1.

V = 50257 here versus C = 768 in layernorm, so each thread strides over ~50x
more elements. That makes the per-thread accumulation loop dominate and the
reduction tail almost free — the opposite balance from layernorm.

NOTE ON SIZE: B=4, T=512 instead of 8/1024. probs is (B,T,V) = 411 MB at
these numbers and 1.6 GB at the larger ones, which with logits alongside it
does not leave room on a 6 GB card.

Compile:  make softmax_crossentropy
Run:      ./softmax_crossentropy
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "common.h"

extern "C" void softmax_crossentropy(float *losses, float *probs, float *logits,
                                     int *targets, int B, int T, int V);

__device__ inline float block_reduce(float val, float *scratch, bool is_max) {
    int lane = threadIdx.x % warpSize;
    int wid  = threadIdx.x / warpSize;
    int nwarps = (blockDim.x + warpSize - 1) / warpSize;
    for (int off = warpSize / 2; off > 0; off >>= 1) {
        float o = __shfl_down_sync(0xffffffff, val, off);
        val = is_max ? fmaxf(val, o) : val + o;
    }
    if (lane == 0) scratch[wid] = val;
    __syncthreads();
    val = (threadIdx.x < nwarps) ? scratch[threadIdx.x] : (is_max ? -1e30f : 0.0f);
    if (wid == 0) {
        for (int off = warpSize / 2; off > 0; off >>= 1) {
            float o = __shfl_down_sync(0xffffffff, val, off);
            val = is_max ? fmaxf(val, o) : val + o;
        }
        if (lane == 0) scratch[0] = val;
    }
    __syncthreads();
    return scratch[0];
}

__global__ void softmax_crossentropy_kernel(float *losses, float *probs,
                                            const float *logits,
                                            const int *targets, int V) {
    __shared__ float scratch[32];
    int bt = blockIdx.x;
    const float *lg = logits + (size_t)bt * V;
    float       *pr = probs  + (size_t)bt * V;

    // 1. max
    float mx = -1e30f;
    for (int v = threadIdx.x; v < V; v += blockDim.x) mx = fmaxf(mx, lg[v]);
    mx = block_reduce(mx, scratch, true);
    __syncthreads();

    // 2. exp and sum, storing the unnormalized values so we only expf once
    float sum = 0.0f;
    for (int v = threadIdx.x; v < V; v += blockDim.x) {
        float e = expf(lg[v] - mx);
        pr[v] = e;
        sum += e;
    }
    sum = block_reduce(sum, scratch, false);

    // 3. normalize
    float inv = 1.0f / sum;
    for (int v = threadIdx.x; v < V; v += blockDim.x) pr[v] *= inv;
    __syncthreads();                       // pr[target] must be normalized first

    if (threadIdx.x == 0) losses[bt] = -logf(pr[targets[bt]]);
}

void softmax_crossentropy_gpu(float *losses, float *probs, const float *logits,
                              const int *targets, int B, int T, int V,
                              const int block_size) {
    softmax_crossentropy_kernel<<<B * T, block_size>>>(losses, probs, logits,
                                                       targets, V);
    cudaCheck(cudaGetLastError());
}

int main(int argc, const char **argv) {
    (void)argc; (void)argv;
    srand(1337);
    print_device();

    int B = 4, T = 512, V = 50257;
    size_t N = (size_t)B * T * V;

    printf("B=%d T=%d V=%d  (%d rows of %d)  probs = %.0f MB\n\n",
           B, T, V, B * T, V, N * 4 / 1e6);

    float *logits  = make_random_float(N);
    int   *targets = make_random_int(B * T, V);
    float *probs   = (float *)malloc(N * sizeof(float));
    float *losses  = (float *)malloc((size_t)B * T * sizeof(float));

    softmax_crossentropy(losses, probs, logits, targets, B, T, V);

    float *d_logits, *d_probs, *d_losses;
    int   *d_targets;
    cudaCheck(cudaMalloc(&d_logits,  N * sizeof(float)));
    cudaCheck(cudaMalloc(&d_probs,   N * sizeof(float)));
    cudaCheck(cudaMalloc(&d_losses,  (size_t)B * T * sizeof(float)));
    cudaCheck(cudaMalloc(&d_targets, (size_t)B * T * sizeof(int)));
    cudaCheck(cudaMemcpy(d_logits,  logits,  N * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_targets, targets, (size_t)B*T*sizeof(int), cudaMemcpyHostToDevice));

    int block_sizes[] = {32, 64, 128, 256, 512, 1024};
    for (size_t j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int bs = block_sizes[j];
        printf("  checking block size %d\n", bs);
        cudaCheck(cudaMemset(d_probs, 0, N * sizeof(float)));
        softmax_crossentropy_gpu(d_losses, d_probs, d_logits, d_targets, B, T, V, bs);
        validate_result(d_probs,  probs,  "probs",  N, 1e-6f);
        // losses are ~10.7 in magnitude and come from -log(x), which amplifies
        // the tree-vs-sequential difference in the softmax denominator. The
        // observed gap is ~1.4e-6 RELATIVE; 1e-5 absolute was the wrong test.
        validate_result(d_losses, losses, "losses", (size_t)B * T, 1e-4f);
    }
    printf("\nAll results match. Starting benchmarks.\n\n");

    double peak = peak_bandwidth_gbs();
    for (size_t j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int bs = block_sizes[j];
        float ms = benchmark_kernel(100, softmax_crossentropy_gpu, d_losses,
                                    d_probs, d_logits, d_targets, B, T, V, bs);
        // read logits twice + write probs twice = 4 touches
        double gbs = (double)N * 4 * sizeof(float) / (ms * 1e6);
        printf("block_size %4d | time %.4f ms | bandwidth %7.2f GB/s | %4.1f%% of peak\n",
               bs, ms, gbs, 100.0 * gbs / peak);
    }

    free(logits); free(targets); free(probs); free(losses);
    cudaCheck(cudaFree(d_logits)); cudaCheck(cudaFree(d_probs));
    cudaCheck(cudaFree(d_losses)); cudaCheck(cudaFree(d_targets));
    return 0;
}
