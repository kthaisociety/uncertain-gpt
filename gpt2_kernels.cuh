/*
=============================================================================
 gpt2_kernels.cuh — everything that is NOT the Bayesian optimizer.

 Shared by every optimizer variant:

     train_bnn_gpu.cu      Bayes-by-Backprop  (Blundell et al. 2015)
                           explicit rho, KL gradient, two AdamW passes
     train_bnn_vadam.cu    Vadam              (Khan et al. 2018)
                           sigma derived from Adam's second moment

 The split line is deliberate and it is the same line the paper draws:

     IN HERE   the network — kernels, layout, forward, backward, data.
               Identical for every variant. A deterministic GPT-2 would
               use this file unchanged.

     NOT HERE  how the weights are sampled and updated. That is the only
               thing the Bayesian method changes, and it is the only thing
               the experiments compare.

 So each optimizer file is ~150 lines and can be read end to end, while a
 fix to the forward pass lands in both at once. Nothing is duplicated and
 the two cannot drift apart.

 Everything below was developed one op per file in dev/, validated against
 the real CPU function in train_bnn.c at six block sizes, and benchmarked
 before being brought here.

 Build: nvcc -arch=sm_75 -O3 --use_fast_math <optimizer>.cu -lcublas -o <bin>
=============================================================================
*/

#ifndef GPT2_KERNELS_CUH
#define GPT2_KERNELS_CUH

/* ---------------------------------------------------------------------------
 THE DATASET UNDER STUDY — one definition, shared by every arm, so two
 trainers can never disagree about what they were trained on.

 CHARACTER-level TinyShakespeare, V=65, not GPT-2 BPE at V=50257. The reason
 is that the BPE vocabulary made the model an embedding table with a
 transformer attached:

     V=50257   wte = 38.6M params = 45% of the model
               logits/probs activation buffers = 103 MB EACH (B*T*V)
     V=65      wte = 49,920 params = 0.06%
               logits/probs = 133 KB each

 At V=65 the transformer is ~98% of the parameters, which is what these
 experiments are actually about, and the activation collapse is what makes
 the large-batch scaling experiment affordable. Regenerate the .bin files
 with `python3 data/prepare_char.py`.
--------------------------------------------------------------------------- */
#define VOCAB      65
#define DATA_TRAIN "data/tinyshakespeare/tiny_shakespeare_char_train.bin"
#define DATA_VAL   "data/tinyshakespeare/tiny_shakespeare_char_val.bin"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CUDA_CHECK(err) cuda_check_((err), __FILE__, __LINE__)
static void cuda_check_(cudaError_t e, const char *file, int line) {
    if (e != cudaSuccess) {
        printf("[CUDA ERROR] %s:%d  %s\n", file, line, cudaGetErrorString(e));
        exit(EXIT_FAILURE);
    }
}
#define CUBLAS_CHECK(st) do { cublasStatus_t s_ = (st); if (s_) { \
    printf("[cuBLAS ERROR] %s:%d  code %d\n", __FILE__, __LINE__, (int)s_); \
    exit(EXIT_FAILURE);} } while (0)

static inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

static cublasHandle_t cublas_handle;

/* ===================================================================== *
 * Philox 4x32-10 — counter-based RNG.
 *
 * eps = f(seed, step, index). No state, no memory, pure arithmetic. The
 * point is not speed, it is REPRODUCIBILITY BY INDEX: the backward pass
 * regenerates the exact eps the forward used, from nothing but the element
 * index, so eps never has to be stored.
 * ===================================================================== */

__host__ __device__ __forceinline__
unsigned int mulhi32(unsigned int a, unsigned int b) {
#ifdef __CUDA_ARCH__
    return __umulhi(a, b);
#else
    return (unsigned int)(((unsigned long long)a * (unsigned long long)b) >> 32);
#endif
}

__host__ __device__ __forceinline__
void philox4x32_10(unsigned int ctr[4], unsigned int key[2], unsigned int out[4]) {
    const unsigned int M0 = 0xD2511F53u, M1 = 0xCD9E8D57u;
    const unsigned int W0 = 0x9E3779B9u, W1 = 0xBB67AE85u;
    unsigned int c0 = ctr[0], c1 = ctr[1], c2 = ctr[2], c3 = ctr[3];
    unsigned int k0 = key[0], k1 = key[1];
    #pragma unroll
    for (int r = 0; r < 10; r++) {
        unsigned int hi0 = mulhi32(M0, c0), lo0 = M0 * c0;
        unsigned int hi1 = mulhi32(M1, c2), lo1 = M1 * c2;
        unsigned int n0 = hi1 ^ c1 ^ k0, n1 = lo1;
        unsigned int n2 = hi0 ^ c3 ^ k1, n3 = lo0;
        c0 = n0; c1 = n1; c2 = n2; c3 = n3;
        k0 += W0; k1 += W1;
    }
    out[0] = c0; out[1] = c1; out[2] = c2; out[3] = c3;
}

// uint32 -> uniform on the OPEN interval (0,1). Open matters: logf(0) = -inf
// would poison the weight buffer.
__host__ __device__ __forceinline__
float u01(unsigned int x) { return ((float)x + 0.5f) * 2.3283064365386963e-10f; }

__host__ __device__ __forceinline__
float philox_normal(unsigned long long seed, int step, size_t index) {
    unsigned int ctr[4], key[2], out[4];
    ctr[0] = (unsigned int)(index & 0xffffffffu);
    ctr[1] = (unsigned int)(index >> 32);
    ctr[2] = (unsigned int)step;
    ctr[3] = 0u;
    key[0] = (unsigned int)(seed & 0xffffffffu);
    key[1] = (unsigned int)(seed >> 32);
    philox4x32_10(ctr, key, out);
    float u1 = u01(out[0]), u2 = u01(out[1]);
    return sqrtf(-2.0f * logf(u1)) * cosf(6.28318530717958647692f * u2);
}

// Must match train_bnn.c's softplus/sigmoidf_ EXACTLY, including the x>20
// shortcut, or the GPU and CPU disagree in the tail.
__device__ __forceinline__ float d_softplus(float x) {
    return x > 20.0f ? x : log1pf(expf(x));
}
__device__ __forceinline__ float d_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

/* ===================================================================== *
 * Block reduction — shuffle within the warp, shared memory across warps.
 * Used by layernorm and softmax. Returns the total in EVERY thread.
 * ===================================================================== */
__device__ __forceinline__ float block_reduce_sum(float val, float *scratch) {
    int lane = threadIdx.x % warpSize, wid = threadIdx.x / warpSize;
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

__device__ __forceinline__ float block_reduce_max(float val, float *scratch) {
    int lane = threadIdx.x % warpSize, wid = threadIdx.x / warpSize;
    int nwarps = (blockDim.x + warpSize - 1) / warpSize;
    for (int off = warpSize / 2; off > 0; off >>= 1)
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, off));
    if (lane == 0) scratch[wid] = val;
    __syncthreads();
    val = (threadIdx.x < nwarps) ? scratch[threadIdx.x] : -1e30f;
    if (wid == 0) {
        for (int off = warpSize / 2; off > 0; off >>= 1)
            val = fmaxf(val, __shfl_down_sync(0xffffffff, val, off));
        if (lane == 0) scratch[0] = val;
    }
    __syncthreads();
    return scratch[0];
}

/* ===================================================================== *
 * GPT-2 kernels.  All ported in dev/ and validated against train_bnn.c.
 * ===================================================================== */

__global__ void k_encoder_forward(float *out, const int *inp, const float *wte,
                                  const float *wpe, int T, int C, size_t N) {
    size_t idx = (size_t)blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N) {
        int c = idx % C, bt = idx / C, t = bt % T;
        out[idx] = wte[(size_t)inp[bt] * C + c] + wpe[(size_t)t * C + c];
    }
}

__global__ void k_encoder_backward(float *dwte, float *dwpe, const float *dout,
                                   const int *inp, int T, int C, size_t N) {
    size_t idx = (size_t)blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N) {
        int c = idx % C, bt = idx / C, t = bt % T;
        float g = dout[idx];
        atomicAdd(&dwte[(size_t)inp[bt] * C + c], g);   // scatter: tokens repeat
        atomicAdd(&dwpe[(size_t)t * C + c], g);
    }
}

__global__ void k_layernorm_forward(float *out, float *mean, float *rstd,
                                    const float *inp, const float *weight,
                                    const float *bias, int C) {
    __shared__ float scratch[32];
    const float eps = 1e-5f;
    int bt = blockIdx.x;                       // one block per token row
    const float *x = inp + (size_t)bt * C;
    float *o = out + (size_t)bt * C;

    float sum = 0.0f;
    for (int c = threadIdx.x; c < C; c += blockDim.x) sum += x[c];
    float m = block_reduce_sum(sum, scratch) / C;
    __syncthreads();

    float varsum = 0.0f;
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        float d = x[c] - m; varsum += d * d;
    }
    float r = rsqrtf(block_reduce_sum(varsum, scratch) / C + eps);

    if (threadIdx.x == 0) { mean[bt] = m; rstd[bt] = r; }
    for (int c = threadIdx.x; c < C; c += blockDim.x)
        o[c] = (x[c] - m) * r * weight[c] + bias[c];
}

__global__ void k_layernorm_backward(float *dinp, float *dweight, float *dbias,
                                     const float *dout, const float *inp,
                                     const float *weight, const float *mean,
                                     const float *rstd, int C) {
    __shared__ float scratch[32];
    int bt = blockIdx.x;
    const float *x  = inp  + (size_t)bt * C;
    const float *dy = dout + (size_t)bt * C;
    float *dx = dinp + (size_t)bt * C;
    float m = mean[bt], r = rstd[bt];

    float s1 = 0.0f, s2 = 0.0f;
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        float xhat = (x[c] - m) * r, dnorm = dy[c] * weight[c];
        s1 += dnorm; s2 += dnorm * xhat;
    }
    float dnorm_mean = block_reduce_sum(s1, scratch) / C;
    __syncthreads();
    float dnorm_xhat_mean = block_reduce_sum(s2, scratch) / C;

    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        float xhat = (x[c] - m) * r, dnorm = dy[c] * weight[c];
        atomicAdd(&dbias[c],   dy[c]);          // shared across all rows
        atomicAdd(&dweight[c], dy[c] * xhat);
        dx[c] += r * (dnorm - dnorm_mean - xhat * dnorm_xhat_mean);
    }
}

__global__ void k_gelu_forward(float *out, const float *inp, size_t N) {
    size_t i = (size_t)blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N) {
        const float k = 0.7978845608f;
        float x = inp[i];
        out[i] = 0.5f * x * (1.0f + tanhf(k * (x + 0.044715f * x * x * x)));
    }
}

__global__ void k_gelu_backward(float *dinp, const float *inp,
                                const float *dout, size_t N) {
    size_t i = (size_t)blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N) {
        const float k = 0.7978845608f;
        float x = inp[i];
        float u = k * (x + 0.044715f * x * x * x);
        float th = tanhf(u), sech2 = 1.0f - th * th;
        float du_dx = k * (1.0f + 3.0f * 0.044715f * x * x);
        dinp[i] += (0.5f * (1.0f + th) + 0.5f * x * sech2 * du_dx) * dout[i];
    }
}

__global__ void k_residual_forward(float *out, const float *x, const float *y,
                                   size_t N) {
    size_t i = (size_t)blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N) out[i] = x[i] + y[i];
}

__global__ void k_residual_backward(float *dx, float *dy, const float *dout,
                                    size_t N) {
    size_t i = (size_t)blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N) { float g = dout[i]; dx[i] += g; dy[i] += g; }
}

__global__ void k_softmax_crossentropy(float *losses, float *probs,
                                       const float *logits, const int *targets,
                                       int V) {
    __shared__ float scratch[32];
    int bt = blockIdx.x;
    const float *lg = logits + (size_t)bt * V;
    float *pr = probs + (size_t)bt * V;

    float mx = -1e30f;
    for (int v = threadIdx.x; v < V; v += blockDim.x) mx = fmaxf(mx, lg[v]);
    mx = block_reduce_max(mx, scratch);
    __syncthreads();

    float sum = 0.0f;
    for (int v = threadIdx.x; v < V; v += blockDim.x) {
        float e = expf(lg[v] - mx); pr[v] = e; sum += e;
    }
    sum = block_reduce_sum(sum, scratch);

    float inv = 1.0f / sum;
    for (int v = threadIdx.x; v < V; v += blockDim.x) pr[v] *= inv;
    __syncthreads();                       // pr[target] must be normalized
    if (threadIdx.x == 0) losses[bt] = -logf(pr[targets[bt]]);
}

__global__ void k_crossentropy_softmax_backward(float *dlogits, const float *probs,
                                                const int *targets, int V,
                                                float dloss, size_t N) {
    size_t idx = (size_t)blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N) {
        int v = idx % V, bt = idx / V;
        float indicator = (v == targets[bt]) ? 1.0f : 0.0f;
        dlogits[idx] += (probs[idx] - indicator) * dloss;
    }
}

// Broadcast bias across all rows so the GEMM can accumulate onto it (beta=1).
__global__ void k_fill_bias(float *out, const float *bias, int OC, size_t N) {
    size_t idx = (size_t)blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N) out[idx] = bias[idx % OC];
}

// dbias[oc] = sum over rows. One block per output channel.
__global__ void k_dbias(float *dbias, const float *dout, int M, int OC) {
    __shared__ float scratch[32];
    int oc = blockIdx.x;
    float sum = 0.0f;
    for (int m = threadIdx.x; m < M; m += blockDim.x) sum += dout[(size_t)m * OC + oc];
    sum = block_reduce_sum(sum, scratch);
    if (threadIdx.x == 0) dbias[oc] += sum;
}

/* ===================================================================== *
 * FlashAttention.  No (B,NH,T,T) buffer is ever allocated.
 *
 * HS must equal C/NH. BR queries per block, BC keys per tile.
 * The +1 padding on the shared rows is NOT cosmetic: with a stride of 64
 * words, bank = (c*64+i) % 32 = i % 32, so every key row collides in one
 * bank. Measured on the backward kernel: 1336 ms -> 39 ms from that +1.
 * ===================================================================== */
#define HS 64
#define BR 128
#define BC 32

__global__ __launch_bounds__(BR)
void k_attention_forward(float *out, const float *qkv, int T, int C, int NH) {
    __shared__ float Ks[BC][HS + 1], Vs[BC][HS + 1];
    const float scale = rsqrtf((float)HS);

    const int Tr = (T + BR - 1) / BR;
    const int qt = blockIdx.x % Tr;
    const int h  = (blockIdx.x / Tr) % NH;
    const int b  =  blockIdx.x / (Tr * NH);
    const int t  = qt * BR + threadIdx.x;
    const bool active = (t < T);

    const float *qkv_b = qkv + (size_t)b * T * 3 * C;

    float qr[HS];
    if (active) {
        const float *qp = qkv_b + (size_t)t * 3 * C + 0 * C + h * HS;
        #pragma unroll
        for (int i = 0; i < HS; i++) qr[i] = qp[i];
    } else {
        #pragma unroll
        for (int i = 0; i < HS; i++) qr[i] = 0.0f;
    }

    float m = -INFINITY, l = 0.0f, o[HS];
    #pragma unroll
    for (int i = 0; i < HS; i++) o[i] = 0.0f;

    const int t_hi = min(qt * BR + BR, T);
    const int n_tiles = (t_hi + BC - 1) / BC;

    for (int j = 0; j < n_tiles; j++) {
        const int t2_base = j * BC;
        __syncthreads();
        for (int idx = threadIdx.x; idx < BC * HS; idx += BR) {
            int c = idx / HS, i = idx % HS, t2 = t2_base + c;
            if (t2 < T) {
                Ks[c][i] = qkv_b[(size_t)t2 * 3 * C + 1 * C + h * HS + i];
                Vs[c][i] = qkv_b[(size_t)t2 * 3 * C + 2 * C + h * HS + i];
            } else { Ks[c][i] = 0.0f; Vs[c][i] = 0.0f; }
        }
        __syncthreads();

        float s[BC], m_tile = -INFINITY;
        #pragma unroll
        for (int c = 0; c < BC; c++) {
            int t2 = t2_base + c;
            if (t2 > t || t2 >= T) { s[c] = -INFINITY; continue; }
            float acc = 0.0f;
            #pragma unroll
            for (int i = 0; i < HS; i++) acc += qr[i] * Ks[c][i];
            acc *= scale; s[c] = acc; m_tile = fmaxf(m_tile, acc);
        }
        if (m_tile == -INFINITY) continue;   // fully masked; also avoids -inf - -inf

        const float m_new = fmaxf(m, m_tile);
        const float corr  = __expf(m - m_new);

        float l_tile = 0.0f;
        #pragma unroll
        for (int c = 0; c < BC; c++) {
            float p = (s[c] == -INFINITY) ? 0.0f : __expf(s[c] - m_new);
            s[c] = p; l_tile += p;
        }
        l = l * corr + l_tile;

        #pragma unroll
        for (int i = 0; i < HS; i++) {
            float acc = 0.0f;
            #pragma unroll
            for (int c = 0; c < BC; c++) acc += s[c] * Vs[c][i];
            o[i] = o[i] * corr + acc;
        }
        m = m_new;
    }

    if (active) {
        float *op = out + (size_t)b * T * C + (size_t)t * C + h * HS;
        float inv = (l > 0.0f) ? 1.0f / l : 0.0f;
        #pragma unroll
        for (int i = 0; i < HS; i++) op[i] = o[i] * inv;
    }
}

__global__ __launch_bounds__(BR)
void k_attention_backward(float *dqkv, const float *dout, const float *qkv,
                          int T, int C, int NH) {
    __shared__ float Ks[BC][HS + 1],  Vs[BC][HS + 1];
    __shared__ float dKs[BC][HS + 1], dVs[BC][HS + 1];
    const float scale = rsqrtf((float)HS);

    const int Tr = (T + BR - 1) / BR;
    const int qt = blockIdx.x % Tr;
    const int h  = (blockIdx.x / Tr) % NH;
    const int b  =  blockIdx.x / (Tr * NH);
    const int t  = qt * BR + threadIdx.x;
    const bool active = (t < T);
    const int tq = active ? t : 0;                 // inactive threads borrow row 0

    const float *qkv_b  = qkv  + (size_t)b * T * 3 * C;
    float       *dqkv_b = dqkv + (size_t)b * T * 3 * C;

    float q[HS], dy[HS], dq[HS];
    {
        const float *qp  = qkv_b + (size_t)tq * 3 * C + 0 * C + h * HS;
        const float *dyp = dout  + (size_t)b * T * C + (size_t)tq * C + h * HS;
        #pragma unroll
        for (int i = 0; i < HS; i++) { q[i] = qp[i]; dy[i] = dyp[i]; dq[i] = 0.0f; }
    }

    const int t_hi = min(qt * BR + BR, T);
    const int n_tiles = (t_hi + BC - 1) / BC;

    // ---- pass A: recompute the softmax stats, and carry E along with them --
    // E = sum att*datt = (1/l) sum exp(s-m)*datt, and that sum rescales by the
    // same corr as l does. So one online sweep produces m, l and E together.
    float m = -INFINITY, l = 0.0f, num = 0.0f;
    for (int j = 0; j < n_tiles; j++) {
        const int t2_base = j * BC;
        __syncthreads();
        for (int idx = threadIdx.x; idx < BC * HS; idx += BR) {
            int c = idx / HS, i = idx % HS, t2 = t2_base + c;
            if (t2 < T) {
                Ks[c][i] = qkv_b[(size_t)t2 * 3 * C + 1 * C + h * HS + i];
                Vs[c][i] = qkv_b[(size_t)t2 * 3 * C + 2 * C + h * HS + i];
            } else { Ks[c][i] = 0.0f; Vs[c][i] = 0.0f; }
        }
        __syncthreads();

        for (int c = 0; c < BC && active; c++) {
            int t2 = t2_base + c;
            if (t2 > t || t2 >= T) continue;
            float s = 0.0f;
            #pragma unroll
            for (int i = 0; i < HS; i++) s += q[i] * Ks[c][i];
            s *= scale;
            float d = 0.0f;
            #pragma unroll
            for (int i = 0; i < HS; i++) d += dy[i] * Vs[c][i];
            float m_new = fmaxf(m, s);
            float corr = __expf(m - m_new), e = __expf(s - m_new);
            l = l * corr + e;
            num = num * corr + e * d;
            m = m_new;
        }
    }
    const float E   = (l > 0.0f) ? num / l : 0.0f;
    const float inv = (l > 0.0f) ? 1.0f / l : 0.0f;

    // ---- pass B: dq in registers (query-private), dk/dv via shared then global
    for (int j = 0; j < n_tiles; j++) {
        const int t2_base = j * BC;
        __syncthreads();
        for (int idx = threadIdx.x; idx < BC * HS; idx += BR) {
            int c = idx / HS, i = idx % HS, t2 = t2_base + c;
            if (t2 < T) {
                Ks[c][i] = qkv_b[(size_t)t2 * 3 * C + 1 * C + h * HS + i];
                Vs[c][i] = qkv_b[(size_t)t2 * 3 * C + 2 * C + h * HS + i];
            } else { Ks[c][i] = 0.0f; Vs[c][i] = 0.0f; }
            dKs[c][i] = 0.0f; dVs[c][i] = 0.0f;
        }
        __syncthreads();

        // Skewed start so the warp's 32 lanes touch 32 different keys at once
        // instead of all piling onto key 0.
        for (int cc = 0; cc < BC && active; cc++) {
            const int c = (cc + threadIdx.x) % BC;
            const int t2 = t2_base + c;
            if (t2 > t || t2 >= T) continue;

            float s = 0.0f;
            #pragma unroll
            for (int i = 0; i < HS; i++) s += q[i] * Ks[c][i];
            float att = __expf(s * scale - m) * inv;

            float datt = 0.0f;
            #pragma unroll
            for (int i = 0; i < HS; i++) datt += dy[i] * Vs[c][i];

            float dp = att * (datt - E) * scale;
            #pragma unroll
            for (int i = 0; i < HS; i++) {
                dq[i] += dp * Ks[c][i];
                atomicAdd(&dKs[c][i], dp * q[i]);
                atomicAdd(&dVs[c][i], att * dy[i]);
            }
        }
        __syncthreads();

        for (int idx = threadIdx.x; idx < BC * HS; idx += BR) {
            int c = idx / HS, i = idx % HS, t2 = t2_base + c;
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

/* ===================================================================== *
 * Model config + buffer layout.  Same as train_bnn.c, MINUS preatt/att —
 * FlashAttention does not need them, so they are not allocated.
 * ===================================================================== */

/*
 L transformer blocks. Per-block tensors are stored contiguously with a fixed
 stride, so block l's copy of tensor X is at   base + l*stride + rel(X).
 Everything else (embeddings, final LN, head) sits outside that region.

 The layout is the SINGLE source of truth: forward, backward, every arm's
 allocator and LowR8's Bayesian-layer list all slice from these two functions.
 Add a tensor = one line here and every caller sees it.
*/
typedef struct { int B, T, C, V, NH, maxT, L; } Config;

typedef struct {
    size_t wte, wpe;                          // shared, before the blocks
    size_t blk0, bstride;                     // block region base and stride
    size_t lng, lnb, qw, qb, pw, pb, ln2g, ln2b, w1, b1, w2, b2;  // WITHIN a block
    size_t flng, flnb, hw, hb, total;         // shared, after the blocks
} ParamOff;

typedef struct {
    size_t enc;
    size_t blk0, bstride;
    size_t lno, lnm, lnr, qkv, atty, apr, resa,               // WITHIN a block
           ln2o, ln2m, ln2r, fch, gelu, mlpo, res1;
    size_t flo, flm, flr, log, prb, los, total;
} ActOff;

// param base pointer for tensor `rel` in block `l`
#define PBLK(p, l, rel) ((p).blk0 + (size_t)(l) * (p).bstride + (p).rel)
#define ABLK(a, l, rel) ((a).blk0 + (size_t)(l) * (a).bstride + (a).rel)

static ParamOff param_offsets(Config cfg) {
    int C = cfg.C, V = cfg.V, maxT = cfg.maxT;
    ParamOff p;
    p.wte  = 0;
    p.wpe  = p.wte + (size_t)V * C;
    p.blk0 = p.wpe + (size_t)maxT * C;

    size_t o = 0;                              // relative offsets within a block
    p.lng  = o; o += C;
    p.lnb  = o; o += C;
    p.qw   = o; o += (size_t)C * 3 * C;
    p.qb   = o; o += 3 * C;
    p.pw   = o; o += (size_t)C * C;
    p.pb   = o; o += C;
    p.ln2g = o; o += C;
    p.ln2b = o; o += C;
    p.w1   = o; o += (size_t)C * 4 * C;
    p.b1   = o; o += 4 * C;
    p.w2   = o; o += (size_t)4 * C * C;
    p.b2   = o; o += C;
    p.bstride = o;

    size_t after = p.blk0 + (size_t)cfg.L * p.bstride;
    p.flng = after;
    p.flnb = p.flng + C;
    p.hw   = p.flnb + C;
    p.hb   = p.hw + (size_t)C * V;
    p.total = p.hb + V;
    return p;
}

static ActOff act_offsets(Config cfg) {
    int B = cfg.B, T = cfg.T, C = cfg.C, V = cfg.V;
    ActOff a;
    a.enc  = 0;
    a.blk0 = (size_t)B * T * C;

    size_t o = 0;
    a.lno  = o; o += (size_t)B * T * C;
    a.lnm  = o; o += (size_t)B * T;
    a.lnr  = o; o += (size_t)B * T;
    a.qkv  = o; o += (size_t)B * T * 3 * C;    // no preatt/att: flash
    a.atty = o; o += (size_t)B * T * C;
    a.apr  = o; o += (size_t)B * T * C;
    a.resa = o; o += (size_t)B * T * C;
    a.ln2o = o; o += (size_t)B * T * C;
    a.ln2m = o; o += (size_t)B * T;
    a.ln2r = o; o += (size_t)B * T;
    a.fch  = o; o += (size_t)B * T * 4 * C;
    a.gelu = o; o += (size_t)B * T * 4 * C;
    a.mlpo = o; o += (size_t)B * T * C;
    a.res1 = o; o += (size_t)B * T * C;        // this block's output stream
    a.bstride = o;

    size_t after = a.blk0 + (size_t)cfg.L * a.bstride;
    a.flo  = after;
    a.flm  = a.flo + (size_t)B * T * C;
    a.flr  = a.flm + (size_t)B * T;
    a.log  = a.flr + (size_t)B * T;
    a.prb  = a.log + (size_t)B * T * V;
    a.los  = a.prb + (size_t)B * T * V;
    a.total = a.los + (size_t)B * T;
    return a;
}

/* ===================================================================== *
 * Launcher wrappers — one per op, so forward/backward read like the C code.
 * ===================================================================== */

#define BS 256   // default block size; dev/ showed 90-92% of peak across 64..512

static void matmul_forward(float *out, const float *inp, const float *weight,
                           const float *bias, int M, int C, int OC) {
    size_t N = (size_t)M * OC;
    k_fill_bias<<<ceil_div((int)N, BS), BS>>>(out, bias, OC, N);
    CUDA_CHECK(cudaGetLastError());
    const float one = 1.0f;
    // row-major C=A@B  ==  column-major C'=B'@A' with M and N swapped
    CUBLAS_CHECK(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                             OC, M, C, &one, weight, OC, inp, C, &one, out, OC));
}

static void matmul_backward(float *dinp, float *dweight, float *dbias,
                            const float *inp, const float *weight,
                            const float *dout, int M, int C, int OC) {
    const float one = 1.0f;
    // dinp (M,C) = dout (M,OC) @ weight^T (OC,C)
    CUBLAS_CHECK(cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                             C, M, OC, &one, weight, OC, dout, OC, &one, dinp, C));
    // dweight (C,OC) = inp^T (C,M) @ dout (M,OC)
    CUBLAS_CHECK(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T,
                             OC, C, M, &one, dout, OC, inp, C, &one, dweight, OC));
    k_dbias<<<OC, BS>>>(dbias, dout, M, OC);
    CUDA_CHECK(cudaGetLastError());
}

/* ===================================================================== *
 * Forward / backward.  Line-for-line the same order as train_bnn.c.
 * ===================================================================== */

static float gpt2_forward(float *acts, const float *params, const int *inputs,
                          const int *targets, Config cfg, float *h_losses) {
    int B = cfg.B, T = cfg.T, C = cfg.C, V = cfg.V, NH = cfg.NH;
    ParamOff p = param_offsets(cfg);
    ActOff   a = act_offsets(cfg);
    int M = B * T;
    size_t NBTC = (size_t)B * T * C;
    int Tr = (T + BR - 1) / BR;

    k_encoder_forward<<<ceil_div((int)NBTC, BS), BS>>>(
        acts + a.enc, inputs, params + p.wte, params + p.wpe, T, C, NBTC);

    // ---- L transformer blocks --------------------------------------------
    // The residual stream threads through: block l reads the previous block's
    // res1 (or the encoder output for l=0) and writes its own res1.
    size_t stream = a.enc;
    for (int l = 0; l < cfg.L; l++) {
        // TALKING half: LN1 -> attention -> +residual
        k_layernorm_forward<<<M, BS>>>(acts + ABLK(a,l,lno), acts + ABLK(a,l,lnm),
            acts + ABLK(a,l,lnr), acts + stream,
            params + PBLK(p,l,lng), params + PBLK(p,l,lnb), C);
        matmul_forward(acts + ABLK(a,l,qkv), acts + ABLK(a,l,lno),
            params + PBLK(p,l,qw), params + PBLK(p,l,qb), M, C, 3 * C);
        k_attention_forward<<<B * NH * Tr, BR>>>(acts + ABLK(a,l,atty),
            acts + ABLK(a,l,qkv), T, C, NH);
        matmul_forward(acts + ABLK(a,l,apr), acts + ABLK(a,l,atty),
            params + PBLK(p,l,pw), params + PBLK(p,l,pb), M, C, C);
        k_residual_forward<<<ceil_div((int)NBTC, BS), BS>>>(
            acts + ABLK(a,l,resa), acts + stream, acts + ABLK(a,l,apr), NBTC);

        // THINKING half: LN2 -> MLP -> +residual
        k_layernorm_forward<<<M, BS>>>(acts + ABLK(a,l,ln2o), acts + ABLK(a,l,ln2m),
            acts + ABLK(a,l,ln2r), acts + ABLK(a,l,resa),
            params + PBLK(p,l,ln2g), params + PBLK(p,l,ln2b), C);
        matmul_forward(acts + ABLK(a,l,fch), acts + ABLK(a,l,ln2o),
            params + PBLK(p,l,w1), params + PBLK(p,l,b1), M, C, 4 * C);
        k_gelu_forward<<<ceil_div((int)(NBTC * 4), BS), BS>>>(
            acts + ABLK(a,l,gelu), acts + ABLK(a,l,fch), NBTC * 4);
        matmul_forward(acts + ABLK(a,l,mlpo), acts + ABLK(a,l,gelu),
            params + PBLK(p,l,w2), params + PBLK(p,l,b2), M, 4 * C, C);
        k_residual_forward<<<ceil_div((int)NBTC, BS), BS>>>(
            acts + ABLK(a,l,res1), acts + ABLK(a,l,resa), acts + ABLK(a,l,mlpo), NBTC);

        stream = ABLK(a,l,res1);
    }

    // HEAD: final LN -> logits -> loss
    k_layernorm_forward<<<M, BS>>>(acts + a.flo, acts + a.flm, acts + a.flr,
                                   acts + stream, params + p.flng, params + p.flnb, C);
    matmul_forward(acts + a.log, acts + a.flo, params + p.hw, params + p.hb, M, C, V);
    k_softmax_crossentropy<<<M, BS>>>(acts + a.los, acts + a.prb, acts + a.log,
                                      targets, V);
    CUDA_CHECK(cudaGetLastError());

    // The only D2H copy in the step: B*T floats.
    CUDA_CHECK(cudaMemcpy(h_losses, acts + a.los, (size_t)M * sizeof(float),
                          cudaMemcpyDeviceToHost));
    float mean = 0.0f;
    for (int i = 0; i < M; i++) mean += h_losses[i];
    return mean / M;
}

static void gpt2_backward(float *grads, float *grads_acts, const float *acts,
                          const float *params, const int *inputs,
                          const int *targets, Config cfg) {
    int B = cfg.B, T = cfg.T, C = cfg.C, V = cfg.V, NH = cfg.NH;
    ParamOff p = param_offsets(cfg);
    ActOff   a = act_offsets(cfg);
    int M = B * T;
    size_t NBTC = (size_t)B * T * C, NBTV = (size_t)B * T * V;
    int Tr = (T + BR - 1) / BR;

    k_crossentropy_softmax_backward<<<ceil_div((int)NBTV, BS), BS>>>(
        grads_acts + a.log, acts + a.prb, targets, V, 1.0f / M, NBTV);

    // the last block's res1 is what the head consumed
    size_t last = (cfg.L > 0) ? ABLK(a, cfg.L - 1, res1) : a.enc;

    matmul_backward(grads_acts + a.flo, grads + p.hw, grads + p.hb,
                    acts + a.flo, params + p.hw, grads_acts + a.log, M, C, V);
    k_layernorm_backward<<<M, BS>>>(grads_acts + last, grads + p.flng, grads + p.flnb,
                                    grads_acts + a.flo, acts + last, params + p.flng,
                                    acts + a.flm, acts + a.flr, C);

    // ---- blocks in REVERSE ------------------------------------------------
    for (int l = cfg.L - 1; l >= 0; l--) {
        // this block's input stream: the previous block's res1, or the encoder
        size_t stream = (l == 0) ? a.enc : ABLK(a, l - 1, res1);

        k_residual_backward<<<ceil_div((int)NBTC, BS), BS>>>(
            grads_acts + ABLK(a,l,resa), grads_acts + ABLK(a,l,mlpo),
            grads_acts + ABLK(a,l,res1), NBTC);

        matmul_backward(grads_acts + ABLK(a,l,gelu), grads + PBLK(p,l,w2),
            grads + PBLK(p,l,b2), acts + ABLK(a,l,gelu), params + PBLK(p,l,w2),
            grads_acts + ABLK(a,l,mlpo), M, 4 * C, C);
        k_gelu_backward<<<ceil_div((int)(NBTC * 4), BS), BS>>>(
            grads_acts + ABLK(a,l,fch), acts + ABLK(a,l,fch),
            grads_acts + ABLK(a,l,gelu), NBTC * 4);
        matmul_backward(grads_acts + ABLK(a,l,ln2o), grads + PBLK(p,l,w1),
            grads + PBLK(p,l,b1), acts + ABLK(a,l,ln2o), params + PBLK(p,l,w1),
            grads_acts + ABLK(a,l,fch), M, C, 4 * C);

        // += : resa fed both the MLP and the skip, so it collects from both.
        k_layernorm_backward<<<M, BS>>>(grads_acts + ABLK(a,l,resa),
            grads + PBLK(p,l,ln2g), grads + PBLK(p,l,ln2b),
            grads_acts + ABLK(a,l,ln2o), acts + ABLK(a,l,resa),
            params + PBLK(p,l,ln2g), acts + ABLK(a,l,ln2m), acts + ABLK(a,l,ln2r), C);
        k_residual_backward<<<ceil_div((int)NBTC, BS), BS>>>(
            grads_acts + stream, grads_acts + ABLK(a,l,apr),
            grads_acts + ABLK(a,l,resa), NBTC);

        matmul_backward(grads_acts + ABLK(a,l,atty), grads + PBLK(p,l,pw),
            grads + PBLK(p,l,pb), acts + ABLK(a,l,atty), params + PBLK(p,l,pw),
            grads_acts + ABLK(a,l,apr), M, C, C);
        k_attention_backward<<<B * NH * Tr, BR>>>(grads_acts + ABLK(a,l,qkv),
            grads_acts + ABLK(a,l,atty), acts + ABLK(a,l,qkv), T, C, NH);
        matmul_backward(grads_acts + ABLK(a,l,lno), grads + PBLK(p,l,qw),
            grads + PBLK(p,l,qb), acts + ABLK(a,l,lno), params + PBLK(p,l,qw),
            grads_acts + ABLK(a,l,qkv), M, C, 3 * C);
        // second contribution to the input stream (the first came from the
        // attention-side residual fork above); += is what makes that work.
        k_layernorm_backward<<<M, BS>>>(grads_acts + stream, grads + PBLK(p,l,lng),
            grads + PBLK(p,l,lnb), grads_acts + ABLK(a,l,lno), acts + stream,
            params + PBLK(p,l,lng), acts + ABLK(a,l,lnm), acts + ABLK(a,l,lnr), C);
    }

    k_encoder_backward<<<ceil_div((int)NBTC, BS), BS>>>(
        grads + p.wte, grads + p.wpe, grads_acts + a.enc, inputs, T, C, NBTC);
    CUDA_CHECK(cudaGetLastError());
}

/* ===================================================================== *
 * Dataloader — llm.c's .bin format: 256 int32 header (magic 20240520),
 * then uint16 token ids. Read once into host memory, serve batches.
 * ===================================================================== */

typedef struct { int *tokens; long n; long pos; } Loader;

static void loader_init(Loader *ld, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { printf("cannot open %s\n", path); exit(1); }
    int header[256];
    if (fread(header, sizeof(int), 256, f) != 256) { printf("bad header\n"); exit(1); }
    if (header[0] != 20240520) { printf("bad magic %d\n", header[0]); exit(1); }
    long ntok = header[2];
    unsigned short *raw = (unsigned short *)malloc(ntok * sizeof(unsigned short));
    if (fread(raw, sizeof(unsigned short), ntok, f) != (size_t)ntok) {
        printf("short read\n"); exit(1);
    }
    fclose(f);
    ld->tokens = (int *)malloc(ntok * sizeof(int));
    for (long i = 0; i < ntok; i++) ld->tokens[i] = (int)raw[i];   // uint16 -> int
    free(raw);
    ld->n = ntok; ld->pos = 0;
    printf("loaded %ld tokens from %s\n", ntok, path);
}

// inputs = tokens[pos .. pos+BT), targets = the same window shifted by one.
static void loader_next(Loader *ld, int *h_inp, int *h_tgt, int BT) {
    if (ld->pos + BT + 1 > ld->n) ld->pos = 0;
    memcpy(h_inp, ld->tokens + ld->pos,     BT * sizeof(int));
    memcpy(h_tgt, ld->tokens + ld->pos + 1, BT * sizeof(int));
    ld->pos += BT;
}

/* ===================================================================== *
 * AdamW. Lives here rather than in an optimizer file because it is the
 * GENERIC update every variant leans on: BBB runs it on mu and rho, the
 * deep-ensemble baseline runs it on each member. The header's "no optimizer
 * logic" rule is really "no BAYESIAN logic" — plain AdamW carries none.
 * bc1/bc2 are the bias corrections 1-beta^t, computed once on the host.
 * ===================================================================== */
__global__ void k_adamw(float *params, const float *grads, float *m, float *v,
                        size_t n, float bc1, float bc2, float lr, float beta1,
                        float beta2, float eps, float wd) {
    size_t i = (size_t)blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        float g = grads[i];
        float mi = beta1 * m[i] + (1.0f - beta1) * g;
        float vi = beta2 * v[i] + (1.0f - beta2) * g * g;
        m[i] = mi; v[i] = vi;
        params[i] -= lr * ((mi / bc1) / (sqrtf(vi / bc2) + eps) + wd * params[i]);
    }
}

#endif  // GPT2_KERNELS_CUH
