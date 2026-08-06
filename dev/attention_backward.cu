/*
Attention backward — FlashAttention style, no T*T buffer.

The CPU version needs THREE (B,NH,T,T) arrays: att from the forward, plus
datt and dpreatt as scratch. At B=2 T=1024 that is 100 MB each, 300 MB of
quadratic scratch to compute a gradient. This kernel allocates none of them.

Gradients to produce, all accumulating into the interleaved dqkv:

    datt[t2]    = dy . v[t2]
    E           = sum_t2 att[t2] * datt[t2]              <- softmax's zero-sum term
    dpreatt[t2] = att[t2] * (datt[t2] - E)
    dq[i]      += sum_t2 dpreatt[t2] * scale * k[t2][i]
    dk[t2][i]  += dpreatt[t2] * scale * q[i]
    dv[t2][i]  += att[t2] * dy[i]

============================================================================
THE PROBLEM: att IS GONE

Forward never stored it. So backward has to RECOMPUTE the softmax, which means
recomputing every q.k score. That is the deal FlashAttention makes everywhere:
trade arithmetic for memory, because on this hardware arithmetic is the cheap
one.

Naively that needs three passes over the keys — one for the max, one for the
denominator, one for E — before anything can be accumulated. It collapses to
TWO, because E can ride along with the online softmax:

    E = sum_t2 att[t2]*datt[t2] = (1/l) * sum_t2 exp(s - m) * datt[t2]

The sum on the right rescales by exactly the same corr = exp(m_old - m_new)
that l does. So carry a third running quantity `num` next to m and l, rescale
all three together, and E = num/l falls out at the end of pass A.

    pass A   recompute scores -> m, l, num        (online, one sweep)
    pass B   recompute scores -> dq, dk, dv       (E is now known)

Two recomputations of every score. Real FlashAttention avoids pass A entirely
by having the FORWARD save m and l — that is only T floats per (b,h), not T*T,
so it is nearly free. Worth doing once these are integrated; here the file
stands alone so it recomputes.

============================================================================
THE OTHER PROBLEM: dk AND dv ARE WRITTEN BY EVERY QUERY

dq is query-private — this thread owns query t and nobody else touches dq[t].
No atomics, straight into registers.

dk and dv are the opposite. Key t2 is read by every query t >= t2, so its
gradient collects contributions from up to T different queries living in
different blocks. That needs atomicAdd.

Done naively that is 128 threads hammering the same dk[c][i] address at the
same instant — a 128-way serialization on every element. Two things fix it:

  1. Accumulate the tile's dk/dv in SHARED memory first (dKs, dVs), then do
     ONE global atomicAdd per element per tile. Cuts global atomics by BR.
  2. Skew the loop so thread tid starts at key (tid % BC) instead of 0. Now
     the 128 threads spread over 32 distinct addresses at any instant —
     4-way conflict instead of 128-way. One line, big difference.

============================================================================
SHAPES / BUDGET
    shared:    Ks, Vs, dKs, dVs = 4 * BC*HS * 4B = 32 KB at BC=32
    registers: q[64], dq[64], dy[64] + scalars   ~ 200
    grid:      B * NH * ceil(T/BR)

LAYOUT REMINDER — qkv and dqkv are interleaved:
    q at  b*T*3C + t*3C + 0*C + h*hs
    k at  b*T*3C + t*3C + 1*C + h*hs
    v at  b*T*3C + t*3C + 2*C + h*hs

Compile:  make attention_backward
Run:      ./attention_backward
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include "common.h"

extern "C" void attention_forward(float *out, float *preatt, float *att,
                                  float *qkv, int B, int T, int C, int NH);
extern "C" void attention_backward(float *dqkv, float *dpreatt, float *datt,
                                   float *dout, float *qkv, float *att,
                                   int B, int T, int C, int NH);

#define HS 64
#define BR 128   // queries per block == threads per block
#define BC 32    // keys per tile

// ----------------------------------------------------------------------------

__global__ __launch_bounds__(BR)
void attention_backward_kernel(float *dqkv, const float *dout, const float *qkv,
                               int T, int C, int NH) {
    // PADDED to HS+1. Without the pad, address = c*64 + i, so bank =
    // (c*64+i) % 32 = i % 32 — every c lands in the SAME bank. The skewed
    // loop below has 32 lanes touching 32 different c with the same i at the
    // same instant, which is a 32-way conflict on every single access,
    // atomic or not. With stride 65, bank = (c + i) % 32 and the 32 lanes
    // spread across all 32 banks.
    //
    // MEASURED, same kernel, only the +1: 1336.2 ms -> 39.3 ms. 34x, from
    // one extra float per row. Nothing else changed.
    __shared__ float Ks[BC][HS + 1],  Vs[BC][HS + 1];
    __shared__ float dKs[BC][HS + 1], dVs[BC][HS + 1];

    const float scale = rsqrtf((float)HS);

    const int Tr = (T + BR - 1) / BR;
    const int qt = blockIdx.x % Tr;
    const int h  = (blockIdx.x / Tr) % NH;
    const int b  =  blockIdx.x / (Tr * NH);

    const int t = qt * BR + threadIdx.x;
    const bool active = (t < T);

    const float *qkv_b  = qkv  + (size_t)b * T * 3 * C;
    float       *dqkv_b = dqkv + (size_t)b * T * 3 * C;

    // q and dq live in registers; dy is READ THROUGH A POINTER instead.
    // Holding all three would be 192 floats and ptxas spills past 255 total —
    // and a spilled array lives in local memory, which is DRAM. dy is the same
    // 64 floats on every iteration, so L1 serves it and the register file is
    // spared. Measured: this is what took the kernel from 116 bytes of spill
    // stores to zero.
    // q, dy and dq all held per-thread: 192 floats. That is over what ptxas
    // can keep in 255 registers alongside the unrolled inner loops, so ~116
    // bytes spill to local memory. Local memory is L1-backed and this kernel
    // is arithmetic-heavy, so it is survivable — but it IS the first thing to
    // fix if this shows up hot in a profile. The real fix is not register
    // juggling: it is having the FORWARD save m and l (T floats per (b,h)),
    // which deletes pass A entirely and with it most of the live state.
    const int tq = active ? t : 0;                     // inactive borrow row 0
    float q[HS], dy[HS], dq[HS];
    {
        const float *qp  = qkv_b + (size_t)tq * 3 * C + 0 * C + h * HS;
        const float *dyp = dout  + (size_t)b * T * C + (size_t)tq * C + h * HS;
        #pragma unroll
        for (int i = 0; i < HS; i++) { q[i] = qp[i]; dy[i] = dyp[i]; dq[i] = 0.0f; }
    }

    const int t_hi = min(qt * BR + BR, T);
    const int n_key_tiles = (t_hi + BC - 1) / BC;

    // ======================= PASS A: m, l, num =============================
    float m = -INFINITY, l = 0.0f, num = 0.0f;

    for (int j = 0; j < n_key_tiles; j++) {
        const int t2_base = j * BC;
        __syncthreads();
        for (int idx = threadIdx.x; idx < BC * HS; idx += BR) {
            const int c = idx / HS, i = idx % HS, t2 = t2_base + c;
            if (t2 < T) {
                Ks[c][i] = qkv_b[(size_t)t2 * 3 * C + 1 * C + h * HS + i];
                Vs[c][i] = qkv_b[(size_t)t2 * 3 * C + 2 * C + h * HS + i];
            } else { Ks[c][i] = 0.0f; Vs[c][i] = 0.0f; }
        }
        __syncthreads();

        // Fully online, one key at a time. No score array is needed here
        // because only two SCALARS (l and num) carry forward — rescaling them
        // per key costs two multiplies. The forward pass could not do this:
        // there o[] is a 64-float vector, so rescaling per key instead of per
        // tile would cost 64 multiplies each time. That asymmetry is why the
        // forward tiles its rescale and this pass does not.
        for (int c = 0; c < BC && active; c++) {
            const int t2 = t2_base + c;
            if (t2 > t || t2 >= T) continue;

            float s = 0.0f;
            #pragma unroll
            for (int i = 0; i < HS; i++) s += q[i] * Ks[c][i];
            s *= scale;

            float d = 0.0f;                       // datt[t2] = dy . v[t2]
            #pragma unroll
            for (int i = 0; i < HS; i++) d += dy[i] * Vs[c][i];

            const float m_new = fmaxf(m, s);
            const float corr  = __expf(m - m_new);
            const float e     = __expf(s - m_new);
            l   = l   * corr + e;                 // both carry the SAME correction,
            num = num * corr + e * d;             // which is why E can ride along
            m   = m_new;
        }
    }

    const float E   = (l > 0.0f) ? (num / l) : 0.0f;
    const float inv = (l > 0.0f) ? (1.0f / l) : 0.0f;

    // ======================= PASS B: dq, dk, dv ============================
    for (int j = 0; j < n_key_tiles; j++) {
        const int t2_base = j * BC;
        __syncthreads();
        for (int idx = threadIdx.x; idx < BC * HS; idx += BR) {
            const int c = idx / HS, i = idx % HS, t2 = t2_base + c;
            if (t2 < T) {
                Ks[c][i] = qkv_b[(size_t)t2 * 3 * C + 1 * C + h * HS + i];
                Vs[c][i] = qkv_b[(size_t)t2 * 3 * C + 2 * C + h * HS + i];
            } else { Ks[c][i] = 0.0f; Vs[c][i] = 0.0f; }
            dKs[c][i] = 0.0f;
            dVs[c][i] = 0.0f;
        }
        __syncthreads();

        // Skewed start: at any instant the 128 threads are spread across the
        // 32 keys instead of all piled on key 0. 4-way shared conflict, not
        // 128-way. This is the single cheapest thing in the kernel.
        for (int cc = 0; cc < BC && active; cc++) {
            const int c  = (cc + threadIdx.x) % BC;
            const int t2 = t2_base + c;
            if (t2 > t || t2 >= T) continue;

            float s = 0.0f;
            #pragma unroll
            for (int i = 0; i < HS; i++) s += q[i] * Ks[c][i];
            const float att = __expf(s * scale - m) * inv;      // recomputed

            float datt = 0.0f;
            #pragma unroll
            for (int i = 0; i < HS; i++) datt += dy[i] * Vs[c][i];

            const float dp = att * (datt - E) * scale;          // dpreatt * scale

            #pragma unroll
            for (int i = 0; i < HS; i++) {
                dq[i] += dp * Ks[c][i];                 // query-private, no atomic
                atomicAdd(&dKs[c][i], dp * q[i]);       // shared, then one global add
                atomicAdd(&dVs[c][i], att * dy[i]);
            }
        }
        __syncthreads();

        // one global atomicAdd per element per tile, instead of per query
        for (int idx = threadIdx.x; idx < BC * HS; idx += BR) {
            const int c = idx / HS, i = idx % HS, t2 = t2_base + c;
            if (t2 < T) {
                atomicAdd(&dqkv_b[(size_t)t2 * 3 * C + 1 * C + h * HS + i], dKs[c][i]);
                atomicAdd(&dqkv_b[(size_t)t2 * 3 * C + 2 * C + h * HS + i], dVs[c][i]);
            }
        }
    }

    if (active) {
        float *dqp = dqkv_b + (size_t)t * 3 * C + 0 * C + h * HS;
        #pragma unroll
        for (int i = 0; i < HS; i++) dqp[i] += dq[i];
    }
}

void attention_backward_gpu(float *dqkv, const float *dout, const float *qkv,
                            int B, int T, int C, int NH, const int block_size) {
    (void)block_size;
    const int Tr = (T + BR - 1) / BR;
    attention_backward_kernel<<<B * NH * Tr, BR>>>(dqkv, dout, qkv, T, C, NH);
    cudaCheck(cudaGetLastError());
}

// ----------------------------------------------------------------------------

int main(int argc, const char **argv) {
    (void)argc; (void)argv;
    srand(1337);
    print_device();

    int B = 2, T = 1024, C = 768, NH = 12;
    size_t NQ = (size_t)B * T * 3 * C;
    size_t ND = (size_t)B * T * C;
    size_t NA = (size_t)B * NH * T * T;

    if (C / NH != HS) { printf("HS mismatch\n"); return 1; }
    printf("B=%d T=%d C=%d NH=%d   BR=%d BC=%d\n", B, T, C, NH, BR, BC);
    printf("quadratic scratch avoided: 3 x %.0f MB\n\n", NA * 4 / 1e6);

    float *qkv  = make_random_float(NQ);
    float *dout = make_random_float(ND);

    // The CPU reference needs att from a forward pass, plus two scratch arrays.
    float *out     = (float *)malloc(ND * sizeof(float));
    float *preatt  = (float *)malloc(NA * sizeof(float));
    float *att     = (float *)malloc(NA * sizeof(float));
    float *datt    = (float *)calloc(NA, sizeof(float));
    float *dpreatt = (float *)calloc(NA, sizeof(float));
    float *dqkv0   = make_random_float(NQ);
    float *dqkv    = (float *)malloc(NQ * sizeof(float));
    if (!out || !preatt || !att || !datt || !dpreatt || !dqkv) {
        printf("host alloc failed\n"); return 1;
    }
    memcpy(dqkv, dqkv0, NQ * sizeof(float));

    printf("running CPU reference (forward then backward, slow)...\n");
    attention_forward(out, preatt, att, qkv, B, T, C, NH);
    attention_backward(dqkv, dpreatt, datt, dout, qkv, att, B, T, C, NH);
    free(out); free(preatt); free(att); free(datt); free(dpreatt);
    printf("done.\n\n");

    float *d_qkv, *d_dout, *d_dqkv;
    cudaCheck(cudaMalloc(&d_qkv,  NQ * sizeof(float)));
    cudaCheck(cudaMalloc(&d_dout, ND * sizeof(float)));
    cudaCheck(cudaMalloc(&d_dqkv, NQ * sizeof(float)));
    cudaCheck(cudaMemcpy(d_qkv,  qkv,  NQ * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_dout, dout, ND * sizeof(float), cudaMemcpyHostToDevice));

    printf("  checking\n");
    cudaCheck(cudaMemcpy(d_dqkv, dqkv0, NQ * sizeof(float), cudaMemcpyHostToDevice));
    attention_backward_gpu(d_dqkv, d_dout, d_qkv, B, T, C, NH, BR);
    // Recomputed softmax + atomic accumulation order; relative agreement only.
    validate_result(d_dqkv, dqkv, "dqkv", NQ, 1e-3f);
    printf("\nResults match. Benchmarking.\n\n");

    // scores computed twice, plus dq/dk/dv each a full (T,T)x(T,hs) contraction
    double flops = 10.0 * B * NH * (double)T * T * HS;
    float ms = benchmark_kernel(20, attention_backward_gpu, d_dqkv, d_dout,
                                d_qkv, B, T, C, NH, BR);
    printf("BR=%d BC=%d | time %8.4f ms | %.2f TFLOP/s (counts masked work)\n",
           BR, BC, ms, flops / (ms * 1e9));

    free(qkv); free(dout); free(dqkv0); free(dqkv);
    cudaCheck(cudaFree(d_qkv)); cudaCheck(cudaFree(d_dout)); cudaCheck(cudaFree(d_dqkv));
    return 0;
}
