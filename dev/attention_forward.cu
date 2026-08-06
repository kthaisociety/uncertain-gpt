/*
Attention forward — FlashAttention.

    per (b, h, t):
        s[t2]  = (q . k[t2]) * scale     for t2 <= t      (causal)
        a[t2]  = softmax_t2(s)
        out[t] = sum_t2 a[t2] * v[t2]

Compile:  make attention_forward
Run:      ./attention_forward

============================================================================
WHAT PROBLEM IS BEING SOLVED

The direct port writes preatt and att, both (B, NH, T, T):

    B=4 NH=12 T=1024  ->  50.3M floats  ->  201 MB EACH

written once and read twice — 805 MB of traffic. It is quadratic in T, and it
exists for one reason only: softmax needs the whole row before it can
normalize any single element of it.

FlashAttention never forms that buffer. Same FLOPs, same answer, no T*T array
anywhere. Two ideas do all the work.

============================================================================
IDEA 1 — ONLINE SOFTMAX  (this is the whole algorithm)

Ordinary softmax needs three passes: find the max, exponentiate and sum,
divide. Pass 2 cannot start until pass 1 has seen every element. That is what
forces the row to exist in memory.

Online softmax computes it in ONE pass by keeping running state and fixing up
the past whenever the future surprises it:

    m   the largest score seen so far
    l   the denominator, sum of exp(s - m), consistent with THAT m
    o   the numerator, sum of exp(s - m) * v, also consistent with that m

When a new chunk arrives with a bigger max, the old l and o were divided by
the wrong exponential. But the correction is exactly one scalar:

    m_new = max(m_old, m_chunk)
    corr  = exp(m_old - m_new)          <- always <= 1, never overflows

    l_new = l_old*corr + sum_chunk exp(s - m_new)
    o_new = o_old*corr + sum_chunk exp(s - m_new)*v

Why that is exact: every term in l_old was exp(s_i - m_old). Multiplying by
exp(m_old - m_new) turns it into exp(s_i - m_old + m_old - m_new) =
exp(s_i - m_new). The old terms are now stated in the new scale, so old and
new can simply be added. No approximation — this is an identity.

At the very end, out = o / l. The division happens ONCE, at the end, instead
of requiring a completed row up front.

Edge case that matters: m starts at -inf, so corr = exp(-inf - finite) = 0,
which correctly zeroes an empty accumulator. But -inf minus -inf is NaN, so a
tile that is entirely masked must be skipped BEFORE the correction is
computed. That is the `if (m_tile == -INFINITY) continue;` below.

============================================================================
IDEA 2 — TILING, AND WHY THE QUERY TILE IS THE POINT

Online softmax alone removes the T*T buffer but still reads K and V from
global memory once per query. Tiling fixes that, and the direction of the
tiling is the part people get backwards.

A block owns BR = 128 QUERIES and walks the keys in tiles of BC = 32. Each
key tile is loaded into shared memory ONCE and then consumed by all 128
queries. That is where reuse comes from — not from the key tile, from the
query tile. With BR = 1 there would be no reuse at all and shared memory
would be pointless: you would copy global -> shared -> register instead of
global -> register.

    shared per block:  Ks[32][64] + Vs[32][64] = 16 KB
    threads per block: 128, one per query
    grid:              B * NH * ceil(T/128) = 384 blocks

Sizing is a tug of war. Bigger BR = more reuse per byte loaded, but more
registers per thread and fewer blocks resident to hide latency. Smaller BR =
better occupancy, but K and V get re-read from global more times. 128x32 is a
starting point, not a derived optimum — measure 64 and 256.

Per-thread state, all in registers:
    qr[64]  this thread's query          (loaded once, reused every tile)
    o [64]  running numerator
    s [32]  this tile's scores
    m, l    running max and denominator
~160 floats. Turing allows 255 registers/thread; if this spills, the compiler
puts arrays in LOCAL memory (which is DRAM) and performance collapses — check
with `nvcc -Xptxas -v` and watch for "spill stores".

============================================================================
WHAT TO EXPECT

FLOPs are IDENTICAL to the naive version: 13.4 GFLOP. Flash does not do less
arithmetic. It only stops moving 805 MB. So the win is bounded by how much of
the naive version's time was that traffic — at T=1024 on a 6 GB card, that is
a real saving but not a dramatic one. Report it honestly.

Two things that will limit this kernel, both visible in the code below:

1. The Q load is NOT coalesced. Adjacent threads own adjacent queries, whose
   q rows sit 3*C = 2304 floats apart in the interleaved qkv layout. Each
   thread reads 64 contiguous floats (2 full cache lines), so it is not
   catastrophic, but it is 128 separate strided reads instead of one stream.
   The fix is staging Q through shared memory with a coalesced load pattern.

2. Load imbalance. Query t attends to t+1 keys, so the first query tile does
   1/8 the work of the last. Blocks finish at wildly different times and the
   SMs drain unevenly. No block size fixes this; it needs work-stealing or a
   smarter block-to-tile assignment.

============================================================================
LAYOUT REMINDER — qkv is interleaved, not three separate tensors:
    q at  b*T*3C + t*3C + 0*C + h*hs
    k at  b*T*3C + t*3C + 1*C + h*hs
    v at  b*T*3C + t*3C + 2*C + h*hs
each hs = 64 contiguous floats.
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include "common.h"

// The CPU reference, linked from train_bnn.c (compiled with -DSKIP_MAIN).
// It still fills preatt/att for its own use; we only compare `out`.
extern "C" void attention_forward(float *out, float *preatt, float *att,
                                  float *qkv, int B, int T, int C, int NH);

#define HS 64    // head size, C/NH — compile-time so the arrays can be registers
#define BR 128   // queries per block == threads per block
#define BC 32    // keys per tile

// ----------------------------------------------------------------------------
// GPU kernel

__global__ __launch_bounds__(BR)
void attention_forward_flash_kernel(float *out, const float *qkv,
                                    int T, int C, int NH) {
    // The only memory this kernel allocates beyond registers. 16 KB.
    __shared__ float Ks[BC][HS];
    __shared__ float Vs[BC][HS];

    const float scale = rsqrtf((float)HS);

    // ---- decode blockIdx.x -> (b, h, query tile) --------------------------
    // grid was launched as B * NH * Tr, with the query tile varying fastest.
    const int Tr = (T + BR - 1) / BR;
    const int qt = blockIdx.x % Tr;
    const int h  = (blockIdx.x / Tr) % NH;
    const int b  =  blockIdx.x / (Tr * NH);

    const int t = qt * BR + threadIdx.x;   // THIS thread's query index
    const bool active = (t < T);           // last tile can overhang

    const float *qkv_b = qkv + (size_t)b * T * 3 * C;

    // ---- load this thread's query into registers, once ---------------------
    // Reused by every key tile below, so it is worth 64 registers.
    float qr[HS];
    if (active) {
        const float *qp = qkv_b + (size_t)t * 3 * C + 0 * C + h * HS;
        #pragma unroll
        for (int i = 0; i < HS; i++) qr[i] = qp[i];
    } else {
        #pragma unroll
        for (int i = 0; i < HS; i++) qr[i] = 0.0f;
    }

    // ---- the running softmax state ----------------------------------------
    float m = -INFINITY;   // largest score seen so far
    float l = 0.0f;        // denominator, consistent with m
    float o[HS];           // numerator,   consistent with m
    #pragma unroll
    for (int i = 0; i < HS; i++) o[i] = 0.0f;

    // ---- how far do we have to walk? --------------------------------------
    // Causality: query t only sees keys t2 <= t. The largest query in this
    // block is qt*BR + BR - 1, so every key tile past that is entirely masked
    // and can be skipped outright — that is roughly half the work saved for
    // free, without a single mask test.
    const int t_hi = min(qt * BR + BR, T);          // exclusive
    const int n_key_tiles = (t_hi + BC - 1) / BC;

    for (int j = 0; j < n_key_tiles; j++) {
        const int t2_base = j * BC;

        // Barrier BEFORE overwriting: some thread may still be reading the
        // previous tile's Ks/Vs in the accumulation loop below.
        __syncthreads();

        // ---- cooperative load of the key tile into shared memory ----------
        // 32*64 = 2048 floats each for K and V, 128 threads, 16 elements per
        // thread. Consecutive threadIdx.x -> consecutive `i` within a row, so
        // this load IS coalesced (unlike the Q load above).
        for (int idx = threadIdx.x; idx < BC * HS; idx += BR) {
            const int c  = idx / HS;
            const int i  = idx % HS;
            const int t2 = t2_base + c;
            if (t2 < T) {
                const float *kp = qkv_b + (size_t)t2 * 3 * C + 1 * C + h * HS;
                const float *vp = qkv_b + (size_t)t2 * 3 * C + 2 * C + h * HS;
                Ks[c][i] = kp[i];
                Vs[c][i] = vp[i];
            } else {
                Ks[c][i] = 0.0f;
                Vs[c][i] = 0.0f;
            }
        }
        __syncthreads();   // tile is complete and visible to the whole block

        // ---- scores for this thread's query against the whole tile --------
        // Every thread reads the SAME Ks[c][i] at the same time, so these are
        // broadcast reads: one shared-memory transaction serves all 128
        // threads, zero bank conflicts.
        float s[BC];
        float m_tile = -INFINITY;
        #pragma unroll
        for (int c = 0; c < BC; c++) {
            const int t2 = t2_base + c;
            if (t2 > t || t2 >= T) { s[c] = -INFINITY; continue; }  // causal mask
            float acc = 0.0f;
            #pragma unroll
            for (int i = 0; i < HS; i++) acc += qr[i] * Ks[c][i];
            acc *= scale;
            s[c] = acc;
            m_tile = fmaxf(m_tile, acc);
        }

        // Fully masked tile: nothing to merge. Skipping here also avoids
        // computing (-inf) - (-inf) = NaN in the correction below.
        if (m_tile == -INFINITY) continue;

        // ---- THE MERGE — this is online softmax, all four lines of it -----
        const float m_new = fmaxf(m, m_tile);
        const float corr  = __expf(m - m_new);   // rescales everything already accumulated

        float l_tile = 0.0f;
        #pragma unroll
        for (int c = 0; c < BC; c++) {
            // reuse s[] to hold the probabilities, so no second array
            const float p = (s[c] == -INFINITY) ? 0.0f : __expf(s[c] - m_new);
            s[c] = p;
            l_tile += p;
        }
        l = l * corr + l_tile;

        // o is a full hs-vector, so the rescale is 64 multiplies — but only
        // ONCE PER TILE, not once per key. That is what the tiling bought.
        #pragma unroll
        for (int i = 0; i < HS; i++) {
            float acc = 0.0f;
            #pragma unroll
            for (int c = 0; c < BC; c++) acc += s[c] * Vs[c][i];
            o[i] = o[i] * corr + acc;
        }

        m = m_new;
    }

    // ---- the single division, at the very end ------------------------------
    if (active) {
        float *op = out + (size_t)b * T * C + (size_t)t * C + h * HS;
        const float inv = 1.0f / l;
        #pragma unroll
        for (int i = 0; i < HS; i++) op[i] = o[i] * inv;
    }
}

// ----------------------------------------------------------------------------
// kernel launcher

void attention_forward_gpu(float *out, const float *qkv, int B, int T, int C,
                           int NH, const int block_size) {
    (void)block_size;   // fixed at BR: the block size IS the query tile here
    const int Tr = (T + BR - 1) / BR;
    attention_forward_flash_kernel<<<B * NH * Tr, BR>>>(out, qkv, T, C, NH);
    cudaCheck(cudaGetLastError());
}

// ----------------------------------------------------------------------------

int main(int argc, const char **argv) {
    (void)argc; (void)argv;
    srand(1337);
    print_device();

    int B = 4, T = 1024, C = 768, NH = 12;
    size_t N  = (size_t)B * T * C;
    size_t NQ = (size_t)B * T * 3 * C;
    size_t NA = (size_t)B * NH * T * T;
    double flops = 4.0 * B * NH * (double)T * T * (C / NH);

    if (C / NH != HS) { printf("HS mismatch: rebuild with HS=%d\n", C / NH); return 1; }

    printf("B=%d T=%d C=%d NH=%d hs=%d   tiles BR=%d BC=%d\n",
           B, T, C, NH, C / NH, BR, BC);
    printf("att buffer avoided: %.0f MB  (805 MB of traffic the direct port pays)\n",
           NA * 4 / 1e6);
    printf("%.1f GFLOP\n\n", flops / 1e9);

    float *qkv    = make_random_float(NQ);
    float *out    = (float *)malloc(N  * sizeof(float));
    float *preatt = (float *)malloc(NA * sizeof(float));   // CPU reference needs
    float *att    = (float *)malloc(NA * sizeof(float));   // these; never compared
    if (!out || !preatt || !att) { printf("host alloc failed\n"); return 1; }

    printf("running CPU reference (slow, ~13 GFLOP single-threaded)...\n");
    attention_forward(out, preatt, att, qkv, B, T, C, NH);
    free(preatt); free(att);
    printf("done.\n\n");

    float *d_qkv, *d_out;
    cudaCheck(cudaMalloc(&d_qkv, NQ * sizeof(float)));
    cudaCheck(cudaMalloc(&d_out, N  * sizeof(float)));
    cudaCheck(cudaMemcpy(d_qkv, qkv, NQ * sizeof(float), cudaMemcpyHostToDevice));

    printf("  checking\n");
    cudaCheck(cudaMemset(d_out, 0, N * sizeof(float)));
    attention_forward_gpu(d_out, d_qkv, B, T, C, NH, BR);
    // hs=64-term dot products then a softmax merged across 32 tiles; the
    // summation order differs from the CPU's sequential loop, so this is a
    // relative-agreement test, not bit-exactness.
    validate_result(d_out, out, "out", N, 1e-4f);
    printf("\nResults match. Benchmarking.\n\n");

    float ms = benchmark_kernel(20, attention_forward_gpu, d_out, d_qkv,
                                B, T, C, NH, BR);
    printf("BR=%d BC=%d | time %8.4f ms | %.2f TFLOP/s\n", BR, BC, ms,
           flops / (ms * 1e9));

    free(qkv); free(out);
    cudaCheck(cudaFree(d_qkv)); cudaCheck(cudaFree(d_out));
    return 0;
}
