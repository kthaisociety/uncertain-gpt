/*
Kernel for the encoder forward pass:

    out[b,t,c] = wte[ inp[b,t] , c ] + wpe[t, c]

token embedding (a gather out of a 50257-row table) plus position embedding,
summed into the activation buffer.

Compile:
    make encoder_forward
Run:
    ./encoder_forward

One thread per OUTPUT ELEMENT, not per token. Per token would launch B*T
threads — 256 at the real config, 0.8% of the card — and the count would not
grow if the model got wider, only the serial loop inside each thread would.

--------------------------------------------------------------------------
NOTE ON SIZES: this harness runs B=8, T=1024, not the B=4, T=64 in
train_bnn.c. The real config touches 786 KB, which fits entirely in the 3 MB
L2 — the benchmark would be measuring cache, not DRAM, and would report a
bandwidth number that means nothing. Correctness is size-independent; timing
is not.

FIRST PRINCIPLES
  CGMA?        1 add per output element, 12 bytes touched -> 0.08 flops/byte.
               Same ratio as residual. Bandwidth is the only target.
  Coalesced?   thread i handles element i, adjacent threads take adjacent c
               in the same row -> yes.
  Roofline:    12 bytes/element, identical to residual_forward. That kernel
               hit 0.3149 ms / 239.8 GB/s. Same traffic, harder index — the
               gap between the two is the cost of the gather.
  Gather cost: wte is 154 MB and token ids are random, so those row reads
               miss L2 every time. wpe is 3 MB and is re-read by all B
               sequences, so it should stay cached. The roofline counts the
               two reads the same; the hardware will not.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "common.h"

// The CPU reference, linked from train_bnn.c (compiled with -DSKIP_MAIN).
extern "C" void encoder_forward(float *out, int *inp, float *wte, float *wpe,
                                int B, int T, int C);

// ----------------------------------------------------------------------------
// GPU kernel

__global__ void encoder_forward_kernel(float *out, const int *inp,
                                       const float *wte, const float *wpe,
                                       int B, int T, int C) {
    // Step 1 — who am I? One id per output float, running 0 .. B*T*C-1.
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    // Step 2 — guard. ceil_div rounds up, so the last block overshoots.
    if (idx < B * T * C) {

        // Step 3 — un-flatten. This is the INVERSE of the CPU's loop nest.
        // The CPU walked forward: given b,t,c it computed b*T*C + t*C + c.
        // We go backward: given that number, recover b,t,c. Because c is the
        // fastest-varying index, / and % peel the layers off from the inside.
        int c  = idx % C;   // position inside the 768-float row
        int bt = idx / C;   // which token — this is exactly b*T + t
        int t  = bt % T;    // wpe is indexed by position alone, so peel t out
        // b on its own is never needed: bt already names the token, and both
        // inp and out are addressed by bt directly.

        // Step 4 — the body. No loop. This thread's ONE element.
        int ix = inp[bt];                                 // the token id here
        out[idx] = wte[ix * C + c] + wpe[t * C + c];
    }
}

// ----------------------------------------------------------------------------
// kernel launcher

void encoder_forward_gpu(float *out, const int *inp, const float *wte,
                         const float *wpe, int B, int T, int C,
                         const int block_size) {
    const int grid_size = ceil_div(B * T * C, block_size);
    encoder_forward_kernel<<<grid_size, block_size>>>(out, inp, wte, wpe, B, T, C);
    cudaCheck(cudaGetLastError());
}

// ----------------------------------------------------------------------------

int main(int argc, const char **argv) {
    (void)argc; (void)argv;
    srand(1337);
    print_device();

    int B = 8, T = 1024, C = 768, V = 50257;
    size_t N = (size_t)B * T * C;

    printf("out = %zu floats / %.0f MB, wte = %.0f MB\n\n",
           N, N * 4 / 1e6, (double)V * C * 4 / 1e6);

    int   *inp = make_random_int(B * T, V);
    float *wte = make_random_float((size_t)V * C);
    float *wpe = make_random_float((size_t)T * C);
    float *out = (float *)malloc(N * sizeof(float));

    // CPU reference first — the thing the kernel must reproduce.
    encoder_forward(out, inp, wte, wpe, B, T, C);

    int   *d_inp;
    float *d_wte, *d_wpe, *d_out;
    cudaCheck(cudaMalloc(&d_inp, (size_t)B * T * sizeof(int)));
    cudaCheck(cudaMalloc(&d_wte, (size_t)V * C * sizeof(float)));
    cudaCheck(cudaMalloc(&d_wpe, (size_t)T * C * sizeof(float)));
    cudaCheck(cudaMalloc(&d_out, N * sizeof(float)));
    cudaCheck(cudaMemcpy(d_inp, inp, (size_t)B * T * sizeof(int),   cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_wte, wte, (size_t)V * C * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_wpe, wpe, (size_t)T * C * sizeof(float), cudaMemcpyHostToDevice));

    // Correctness at EVERY block size before any timing. An indexing bug in a
    // 3-deep un-flatten is exactly the kind one block size hides.
    int block_sizes[] = {32, 64, 128, 256, 512, 1024};
    for (size_t j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];
        printf("  checking block size %d\n", block_size);
        cudaCheck(cudaMemset(d_out, 0, N * sizeof(float)));   // a partial write can't pass
        encoder_forward_gpu(d_out, d_inp, d_wte, d_wpe, B, T, C, block_size);
        validate_result(d_out, out, "out", N, 1e-5f);
    }
    printf("\nAll results match. Starting benchmarks.\n\n");

    double peak = peak_bandwidth_gbs();
    for (size_t j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];
        float ms = benchmark_kernel(100, encoder_forward_gpu,
                                    d_out, d_inp, d_wte, d_wpe, B, T, C, block_size);
        // napkin math: 2 reads + 1 write of 4 bytes per output element
        double gbs = (double)N * 3 * sizeof(float) / (ms * 1e6);
        printf("block_size %4d | time %.4f ms | bandwidth %7.2f GB/s | %4.1f%% of peak\n",
               block_size, ms, gbs, 100.0 * gbs / peak);
    }

    free(inp); free(wte); free(wpe); free(out);
    cudaCheck(cudaFree(d_inp)); cudaCheck(cudaFree(d_wte));
    cudaCheck(cudaFree(d_wpe)); cudaCheck(cudaFree(d_out));
    return 0;
}
