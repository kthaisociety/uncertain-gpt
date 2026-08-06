/*
Cross-entropy + softmax backward, fused. The famous simplification:

    dlogits[b,t,v] += (probs[b,t,v] - [v == target]) / (B*T)

"The gradient is how much probability mass you put in the wrong place."

Flat and elementwise over B*T*V, so it is a 1:1 kernel despite living at the
end of a reduction. The indicator is written as a ternary rather than an `if`
for the same reason the CPU version does: a branch here would split the warp,
and 50256 of every 50257 lanes take the same side of it anyway.

Traffic: read probs + read/write dlogits = 3 touches = 12 B/element, over
B*T*V elements. At the real vocab this is the single largest activation
buffer in the model.

Compile:  make crossentropy_softmax_backward
Run:      ./crossentropy_softmax_backward
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include "common.h"

extern "C" void crossentropy_softmax_backward(float *dlogits, float *probs,
                                              int *targets, int B, int T, int V);

__global__ void crossentropy_softmax_backward_kernel(float *dlogits,
                                                     const float *probs,
                                                     const int *targets,
                                                     int V, float dloss,
                                                     size_t N) {
    size_t idx = (size_t)blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N) {
        int v  = idx % V;
        int bt = idx / V;
        float indicator = (v == targets[bt]) ? 1.0f : 0.0f;
        dlogits[idx] += (probs[idx] - indicator) * dloss;
    }
}

void crossentropy_softmax_backward_gpu(float *dlogits, const float *probs,
                                       const int *targets, int B, int T, int V,
                                       const int block_size) {
    size_t N = (size_t)B * T * V;
    float dloss = 1.0f / (B * T);
    const int grid_size = ceil_div((int)N, block_size);
    crossentropy_softmax_backward_kernel<<<grid_size, block_size>>>(
        dlogits, probs, targets, V, dloss, N);
    cudaCheck(cudaGetLastError());
}

int main(int argc, const char **argv) {
    (void)argc; (void)argv;
    srand(1337);
    print_device();

    int B = 4, T = 512, V = 50257;
    size_t N = (size_t)B * T * V;
    const int TOUCH = 3;

    printf("B=%d T=%d V=%d  N=%zu  %.2f GB touched\n\n",
           B, T, V, N, N * 4.0 * TOUCH / 1e9);

    float *probs   = make_random_float(N);
    int   *targets = make_random_int(B * T, V);
    float *dl0     = make_random_float(N);
    float *dlogits = (float *)malloc(N * sizeof(float));
    memcpy(dlogits, dl0, N * sizeof(float));

    crossentropy_softmax_backward(dlogits, probs, targets, B, T, V);

    float *d_dlogits, *d_probs;
    int   *d_targets;
    cudaCheck(cudaMalloc(&d_dlogits, N * sizeof(float)));
    cudaCheck(cudaMalloc(&d_probs,   N * sizeof(float)));
    cudaCheck(cudaMalloc(&d_targets, (size_t)B * T * sizeof(int)));
    cudaCheck(cudaMemcpy(d_probs,   probs,   N * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_targets, targets, (size_t)B*T*sizeof(int), cudaMemcpyHostToDevice));

    int block_sizes[] = {32, 64, 128, 256, 512, 1024};
    for (size_t j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int bs = block_sizes[j];
        printf("  checking block size %d\n", bs);
        cudaCheck(cudaMemcpy(d_dlogits, dl0, N * sizeof(float), cudaMemcpyHostToDevice));
        crossentropy_softmax_backward_gpu(d_dlogits, d_probs, d_targets, B, T, V, bs);
        validate_result(d_dlogits, dlogits, "dlogits", N, 1e-5f);
    }
    printf("\nAll results match. Starting benchmarks.\n\n");

    double peak = peak_bandwidth_gbs();
    printf("roofline floor at peak: %.3f ms\n\n", N * 4.0 * TOUCH / (peak * 1e6));
    for (size_t j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int bs = block_sizes[j];
        float ms = benchmark_kernel(100, crossentropy_softmax_backward_gpu,
                                    d_dlogits, d_probs, d_targets, B, T, V, bs);
        double gbs = (double)N * TOUCH * sizeof(float) / (ms * 1e6);
        printf("block_size %4d | time %.4f ms | bandwidth %7.2f GB/s | %4.1f%% of peak\n",
               bs, ms, gbs, 100.0 * gbs / peak);
    }

    free(probs); free(targets); free(dl0); free(dlogits);
    cudaCheck(cudaFree(d_dlogits)); cudaCheck(cudaFree(d_probs));
    cudaCheck(cudaFree(d_targets));
    return 0;
}
