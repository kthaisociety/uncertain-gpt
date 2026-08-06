/*
Shared helpers for the per-kernel development files in this directory.

This mirrors llm.c's `dev/cuda/common.h` — same idea, same function names, so
that reading their kernel files and writing yours feel like the same activity.
Kept deliberately smaller: no bf16/fp16 packing, no multi-GPU, no cuBLAS. Those
get added when a kernel actually needs them, not before.
*/
#ifndef DEV_COMMON_H
#define DEV_COMMON_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <utility>
#include <cuda_runtime.h>

// ----------------------------------------------------------------------------
// error checking

#define cudaCheck(err) cuda_check_((err), __FILE__, __LINE__)
inline void cuda_check_(cudaError_t error, const char *file, int line) {
    if (error != cudaSuccess) {
        printf("[CUDA ERROR] at %s:%d\n%s\n", file, line, cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
}

template<class T>
__host__ __device__ T ceil_div(T dividend, T divisor) {
    return (dividend + divisor - 1) / divisor;
}

// ----------------------------------------------------------------------------
// random test data

inline float *make_random_float(size_t N) {
    float *arr = (float *)malloc(N * sizeof(float));
    for (size_t i = 0; i < N; i++) arr[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    return arr;
}

inline float *make_random_float_scaled(size_t N, float scale) {
    float *arr = (float *)malloc(N * sizeof(float));
    for (size_t i = 0; i < N; i++) arr[i] = (((float)rand() / RAND_MAX) - 0.5f) * scale;
    return arr;
}

inline int *make_random_int(size_t N, int V) {
    int *arr = (int *)malloc(N * sizeof(int));
    for (size_t i = 0; i < N; i++) arr[i] = rand() % V;
    return arr;
}

// ----------------------------------------------------------------------------
// correctness

/*
Compare a device buffer against the CPU reference.

The tolerance is RELATIVE to each element's magnitude plus an absolute floor,
because a fixed absolute tolerance is the wrong test for a buffer whose values
span several orders of magnitude — it is far too strict on the large elements
and meaningless on the small ones.

Exits on failure (after showing at most 10 mismatches) so a broken kernel stops
the run loudly instead of quietly polluting the benchmark numbers below it.
*/
inline void validate_result(const float *d_result, const float *cpu_ref,
                            const char *name, size_t n, float tolerance = 1e-4f) {
    float *gpu = (float *)malloc(n * sizeof(float));
    cudaCheck(cudaMemcpy(gpu, d_result, n * sizeof(float), cudaMemcpyDeviceToHost));

    int nfaults = 0;
    for (size_t i = 0; i < n; i++) {
        if (!isfinite(cpu_ref[i])) continue;              // skip masked entries
        if (i < 5) printf("    %f %f\n", cpu_ref[i], gpu[i]);
        float t_eff = tolerance + fabsf(cpu_ref[i]) * FLT_EPSILON;
        if (fabsf(cpu_ref[i] - gpu[i]) > t_eff) {
            printf("  MISMATCH %s at %zu: CPU %f vs GPU %f\n", name, i, cpu_ref[i], gpu[i]);
            if (++nfaults >= 10) break;
        }
    }
    free(gpu);
    if (nfaults) { printf("  -> %s FAILED\n", name); exit(EXIT_FAILURE); }
}

// ----------------------------------------------------------------------------
// timing

/*
Time a kernel launcher over `repeats` runs.

The L2 flush is not optional. These are bandwidth-bound kernels; without
scrubbing L2 between repeats, part of the working set stays resident and the
reported GB/s is inflated by an amount that depends on buffer size — i.e. the
benchmark lies more the smaller your test case is.
*/
template<class Kernel, class... KernelArgs>
float benchmark_kernel(int repeats, Kernel kernel, KernelArgs&&... kernel_args) {
    cudaEvent_t start, stop;
    cudaDeviceProp prop;
    cudaCheck(cudaGetDeviceProperties(&prop, 0));
    void *flush_buffer;
    cudaCheck(cudaMalloc(&flush_buffer, prop.l2CacheSize));

    cudaCheck(cudaEventCreate(&start));
    cudaCheck(cudaEventCreate(&stop));
    float total = 0.0f;
    for (int i = 0; i < repeats; i++) {
        cudaCheck(cudaMemset(flush_buffer, 0, prop.l2CacheSize));
        cudaCheck(cudaEventRecord(start, nullptr));
        kernel(std::forward<KernelArgs>(kernel_args)...);
        cudaCheck(cudaEventRecord(stop, nullptr));
        cudaCheck(cudaEventSynchronize(stop));
        float ms;
        cudaCheck(cudaEventElapsedTime(&ms, start, stop));
        total += ms;
    }
    cudaCheck(cudaFree(flush_buffer));
    cudaCheck(cudaEventDestroy(start));
    cudaCheck(cudaEventDestroy(stop));
    return total / repeats;
}

// Peak DRAM bandwidth of device 0, GB/s. Your card reports 264 (Max-Q is
// downclocked from the desktop 2060's 336) — always quote % of THIS, not of
// the number on the spec sheet for the desktop part.
inline double peak_bandwidth_gbs() {
    cudaDeviceProp prop;
    cudaCheck(cudaGetDeviceProperties(&prop, 0));
    return 2.0 * prop.memoryClockRate * 1e3 * (prop.memoryBusWidth / 8) / 1e9;
}

inline void print_device(void) {
    cudaDeviceProp prop;
    cudaCheck(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s  sm_%d%d  %.1f GB  peak BW %.0f GB/s\n\n",
           prop.name, prop.major, prop.minor, prop.totalGlobalMem / 1e9,
           peak_bandwidth_gbs());
}

#endif  // DEV_COMMON_H
