/*
AdamW update. Elementwise, parameter-sized, and the heaviest per-element
traffic in the whole step:

    m = b1*m + (1-b1)*g
    v = b2*v + (1-b2)*g*g
    params -= lr * (m/bc1 / (sqrt(v/bc2) + eps) + wd*params)

Traffic: read grads + read/write params, m, v = 7 touches = 28 B/element.

In the Bayesian model this runs TWICE per step (once on mu, once on rho) and
carries two optimizer states for each — which is where bottleneck B, memory
capacity, actually comes from: 6 parameter-sized buffers live permanently
(mu, rho, m_mu, v_mu, m_rho, v_rho) versus 3 for the deterministic model.

Compile:  make adamw
Run:      ./adamw
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include "common.h"

extern "C" void adamw_update(float *params, float *grads, float *m, float *v,
                             size_t n, int t, float lr, float beta1,
                             float beta2, float eps, float wd);

__global__ void adamw_kernel(float *params, const float *grads, float *m,
                             float *v, size_t n, float bc1, float bc2,
                             float lr, float beta1, float beta2,
                             float eps, float wd) {
    size_t i = (size_t)blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        float g = grads[i];
        float mi = beta1 * m[i] + (1.0f - beta1) * g;
        float vi = beta2 * v[i] + (1.0f - beta2) * g * g;
        m[i] = mi;
        v[i] = vi;
        float m_hat = mi / bc1;
        float v_hat = vi / bc2;
        params[i] -= lr * (m_hat / (sqrtf(v_hat) + eps) + wd * params[i]);
    }
}

// Bias corrections depend only on t, so compute them once on the host rather
// than making every one of 85 million threads call powf.
void adamw_gpu(float *params, const float *grads, float *m, float *v,
               size_t n, int t, float lr, float beta1, float beta2,
               float eps, float wd, const int block_size) {
    float bc1 = 1.0f - powf(beta1, (float)t);
    float bc2 = 1.0f - powf(beta2, (float)t);
    const int grid_size = ceil_div((int)n, block_size);
    adamw_kernel<<<grid_size, block_size>>>(params, grads, m, v, n, bc1, bc2,
                                            lr, beta1, beta2, eps, wd);
    cudaCheck(cudaGetLastError());
}

int main(int argc, const char **argv) {
    (void)argc; (void)argv;
    srand(1337);
    print_device();

    size_t n = 85120849;
    int t = 7;
    float lr = 1e-4f, beta1 = 0.9f, beta2 = 0.999f, eps = 1e-8f, wd = 0.01f;
    const int TOUCH = 7;

    printf("n = %zu params, %.2f GB touched\n\n", n, n * 4.0 * TOUCH / 1e9);

    float *grads = make_random_float(n);
    // params, m, v are all read-modify-write, so keep pristine copies.
    float *p0 = make_random_float(n);
    float *m0 = make_random_float(n);
    float *v0 = (float *)malloc(n * sizeof(float));   // v is a variance: keep it positive
    for (size_t i = 0; i < n; i++) v0[i] = (float)rand() / RAND_MAX;
    float *params = (float *)malloc(n * sizeof(float));
    float *m      = (float *)malloc(n * sizeof(float));
    float *v      = (float *)malloc(n * sizeof(float));
    memcpy(params, p0, n * sizeof(float));
    memcpy(m,      m0, n * sizeof(float));
    memcpy(v,      v0, n * sizeof(float));

    adamw_update(params, grads, m, v, n, t, lr, beta1, beta2, eps, wd);

    float *d_p, *d_g, *d_m, *d_v;
    cudaCheck(cudaMalloc(&d_p, n * sizeof(float)));
    cudaCheck(cudaMalloc(&d_g, n * sizeof(float)));
    cudaCheck(cudaMalloc(&d_m, n * sizeof(float)));
    cudaCheck(cudaMalloc(&d_v, n * sizeof(float)));
    cudaCheck(cudaMemcpy(d_g, grads, n * sizeof(float), cudaMemcpyHostToDevice));

    int block_sizes[] = {32, 64, 128, 256, 512, 1024};
    for (size_t j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int bs = block_sizes[j];
        printf("  checking block size %d\n", bs);
        cudaCheck(cudaMemcpy(d_p, p0, n * sizeof(float), cudaMemcpyHostToDevice));
        cudaCheck(cudaMemcpy(d_m, m0, n * sizeof(float), cudaMemcpyHostToDevice));
        cudaCheck(cudaMemcpy(d_v, v0, n * sizeof(float), cudaMemcpyHostToDevice));
        adamw_gpu(d_p, d_g, d_m, d_v, n, t, lr, beta1, beta2, eps, wd, bs);
        validate_result(d_p, params, "params", n, 1e-5f);
        validate_result(d_m, m,      "m",      n, 1e-5f);
        validate_result(d_v, v,      "v",      n, 1e-5f);
    }
    printf("\nAll results match. Starting benchmarks.\n\n");

    double peak = peak_bandwidth_gbs();
    printf("roofline floor at peak: %.3f ms\n\n", n * 4.0 * TOUCH / (peak * 1e6));
    for (size_t j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int bs = block_sizes[j];
        float ms = benchmark_kernel(100, adamw_gpu, d_p, d_g, d_m, d_v, n, t,
                                    lr, beta1, beta2, eps, wd, bs);
        double gbs = (double)n * TOUCH * sizeof(float) / (ms * 1e6);
        printf("block_size %4d | time %.4f ms | bandwidth %7.2f GB/s | %4.1f%% of peak\n",
               bs, ms, gbs, 100.0 * gbs / peak);
    }

    free(grads); free(p0); free(m0); free(v0); free(params); free(m); free(v);
    cudaCheck(cudaFree(d_p)); cudaCheck(cudaFree(d_g));
    cudaCheck(cudaFree(d_m)); cudaCheck(cudaFree(d_v));
    return 0;
}
