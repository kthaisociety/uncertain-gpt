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
// text8: 90M training tokens, vocab 27. TinyShakespeare had 1.0M tokens
// against 43M parameters — 43x more parameters than data — and the ELBO's KL
// term is per-dataset, so the prior outweighed the data ~35:1 even at the
// correct weight. Measured on Vadam, where N sits inside sigma:
//     N=1e6  best-informed parameter 35% data-driven
//     N=1e8  best-informed parameter 98% data-driven
// text8 is also the standard char-level LM benchmark. Regenerate the .bin
// files with `python3 data/prepare_text8.py`.
#define VOCAB      27
#define DATA_TRAIN "data/text8/text8_train.bin"
#define DATA_VAL   "data/text8/text8_val.bin"
#define DATA_TEST  "data/text8/text8_test.bin"   // Mikolov split: 90M/5M/5M

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

/* ---------------------------------------------------------------------------
 HOW A FATAL ERROR ENDS.

 A standalone trainer that hits a CUDA error has nothing useful left to do, so
 it prints and exits — unchanged. The same code compiled into the Python
 extension (uncertain/, which defines UNCERTAIN_AS_LIBRARY) must not kill its
 host interpreter, so there the same message is printed and then thrown as a
 std::runtime_error, which pybind turns into a Python exception.

 Every printf is identical in both modes: the binaries' output does not change,
 and neither does their exit status (EXIT_FAILURE == 1, which is what the
 loader's exit(1) calls already used).
--------------------------------------------------------------------------- */
#ifdef UNCERTAIN_AS_LIBRARY
#include <stdexcept>
#define UNCERTAIN_ABORT(msg) throw std::runtime_error(msg)
#else
#define UNCERTAIN_ABORT(msg) exit(EXIT_FAILURE)
#endif

#define CUDA_CHECK(err) cuda_check_((err), __FILE__, __LINE__)
static void cuda_check_(cudaError_t e, const char *file, int line) {
    if (e != cudaSuccess) {
        printf("[CUDA ERROR] %s:%d  %s\n", file, line, cudaGetErrorString(e));
        UNCERTAIN_ABORT("CUDA error");
    }
}
#define CUBLAS_CHECK(st) do { cublasStatus_t s_ = (st); if (s_) { \
    printf("[cuBLAS ERROR] %s:%d  code %d\n", __FILE__, __LINE__, (int)s_); \
    UNCERTAIN_ABORT("cuBLAS error");} } while (0)

static inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

/* ---------------------------------------------------------------------------
 THE cuBLAS HANDLE.

 This was a plain `static cublasHandle_t cublas_handle;`. For a one-file binary
 that is exactly right: one translation unit, one handle, created in main().

 It stops being right the moment two arms are compiled into the same module,
 which is what the Python extension does. `static` at file scope means INTERNAL
 linkage: every translation unit that includes this header gets its own private
 copy. Whichever copy nobody called cublasCreate on stays null, and every GEMM
 routed through it fails — with nothing in the type system to catch it and no
 symptom that points here.

 Routing every use through an inline accessor makes it exactly ONE handle per
 binary or shared object, whatever the translation-unit layout is: C++
 guarantees a single instance of an inline function's local static across the
 linked image. The macro keeps every existing call site spelled the same —
 `cublasCreate(&cublas_handle)`, `cublasDestroy(cublas_handle)`, and the two
 GEMM wrappers below — so no arm file changes.

 Contract: one handle for the process, on the default stream, shared by every
 Trainer. See uncertain/SPEC.md for why the extension owns this rather than
 borrowing PyTorch's handle (that one follows torch's current stream; these
 kernels launch on the default stream, and mixing the two would race).
--------------------------------------------------------------------------- */
inline cublasHandle_t &cublas_handle_ref(void) {
    static cublasHandle_t h = nullptr;
    return h;
}
#define cublas_handle (cublas_handle_ref())

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

    // HS is compile-time in the FlashAttention kernels and they index q/k/v
    // with it, so a config whose head size is not HS reads past every head and
    // returns confident garbage with no error. Caught here rather than in the
    // numbers: dev/block_loop.cu walked into it at C=8, NH=2.
    if (C / NH != HS || C % NH != 0) {
        printf("config error: head size C/NH = %d, but the attention kernels are "
               "compiled for HS = %d\n", C / NH, HS);
        UNCERTAIN_ABORT("gpt2_forward: head size must equal HS");
    }

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
    if (!f) { printf("cannot open %s\n", path); UNCERTAIN_ABORT("loader: cannot open file"); }
    int header[256];
    if (fread(header, sizeof(int), 256, f) != 256) { printf("bad header\n"); UNCERTAIN_ABORT("loader: bad header"); }
    if (header[0] != 20240520) { printf("bad magic %d\n", header[0]); UNCERTAIN_ABORT("loader: bad magic"); }
    long ntok = header[2];
    unsigned short *raw = (unsigned short *)malloc(ntok * sizeof(unsigned short));
    if (fread(raw, sizeof(unsigned short), ntok, f) != (size_t)ntok) {
        printf("short read\n"); UNCERTAIN_ABORT("loader: short read");
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
 * RUN PERSISTENCE — so a 4-hour run never has to be repeated.
 *
 * Two separate things get saved:
 *   runs/results.csv   the measured numbers (written by scorecard.c)
 *   runs/*.bin         the trained parameters, written here
 *
 * The checkpoints matter more. The uncertainty evaluation (NLPD, ECE,
 * epistemic ID/OOD at S samples) is still unwritten for three of four arms.
 * When it lands, it must run against ALREADY TRAINED models — otherwise
 * every new metric costs another full training run.
 *
 * Format: 8-int header {magic, n_lo, C, L, V, NH, steps, n_hi} then n floats.
 * ===================================================================== */

static void ckpt_save(const char *arm, const char *what, const float *d_buf,
                      size_t n, Config cfg, int steps) {
    if (system("mkdir -p runs") != 0) { /* already exists */ }
    char path[512];
    snprintf(path, sizeof path, "runs/%s_C%d_L%d_%dk_%s.bin",
             arm, cfg.C, cfg.L, steps / 1000, what);
    float *h = (float *)malloc(n * sizeof(float));
    if (!h) { printf("ckpt: host alloc failed for %s\n", path); return; }
    CUDA_CHECK(cudaMemcpy(h, d_buf, n * sizeof(float), cudaMemcpyDeviceToHost));
    FILE *f = fopen(path, "wb");
    if (!f) { printf("ckpt: cannot write %s\n", path); free(h); return; }
    int hdr[8] = { 20240520, (int)(n & 0xffffffffu), cfg.C, cfg.L,
                   cfg.V, cfg.NH, steps, (int)(n >> 32) };
    fwrite(hdr, sizeof(int), 8, f);
    fwrite(h, sizeof(float), n, f);
    fclose(f); free(h);
    printf("ckpt: %s  (%zu floats, %.0f MB)\n", path, n, n * 4 / 1e6);
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

/* =====================================================================  *
 *                                                                        *
 *   THE SHARED UNCERTAINTY EVALUATION HARNESS                            *
 *                                                                        *
 * Everything below is new code that no training step touches. It exists   *
 * so that the four arms produce the SAME uncertainty numbers computed by  *
 * the SAME arithmetic, which is the only way the comparison means         *
 * anything. It lived inside train_ensemble.cu until now, which is why     *
 * three of four arms had no uncertainty numbers at all.                   *
 *                                                                        *
 * ---------------------------------------------------------------------- *
 * THE DECOMPOSITION (per token, nats, double precision)                   *
 *                                                                        *
 *     p_bar(y|x) = (1/S) sum_s p_s(y|x)        MIX PROBABILITIES, never   *
 *                                              logits — a logit average   *
 *                                              is a geometric mean and    *
 *                                              carries the wrong          *
 *                                              uncertainty.               *
 *     total     = H[p_bar]                                                *
 *     aleatoric = (1/S) sum_s H[p_s]                                      *
 *     epistemic = (1/S) sum_s KL(p_s || p_bar)                            *
 *                                                                        *
 * total = aleatoric + epistemic is EXACT ALGEBRA, for any S >= 1 and for  *
 * any way of drawing the samples:                                         *
 *                                                                        *
 *   alea + epi = -(1/S) SS p_sy log p_sy + (1/S) SS p_sy log p_sy         *
 *                                        - (1/S) SS p_sy log p_bar_y      *
 *              = - sum_y [(1/S) sum_s p_sy] log p_bar_y                   *
 *              = - sum_y p_bar_y log p_bar_y  =  H[p_bar]                 *
 *                                                                        *
 * Both halves are computed DIRECTLY (epistemic is never derived as the    *
 * difference), so asserting the identity actually tests something. It is  *
 * the correctness gate of this whole file: because the algebra is         *
 * independent of the posterior, a violation can only mean the             *
 * PROBABILITIES are wrong — an unnormalized softmax row, a stale prob     *
 * buffer, a sample callback that never wrote its weights — never a        *
 * modelling problem. It runs for every arm and the process exits nonzero. *
 *                                                                        *
 * ---------------------------------------------------------------------- *
 * S IS A PARAMETER, AND THE ENSEMBLE IS THE ASYMMETRY                     *
 *                                                                        *
 * Each arm supplies a uq_sample_fn: "give me draw s". The Bayesian arms   *
 * draw from a posterior, so S is free and the estimates converge as S     *
 * grows; the comparison is only fair at equal S, hence the parameter.     *
 *                                                                        *
 * The deep ensemble CANNOT do this. Its S-th sample is its S-th trained   *
 * member, so S is capped at E and every extra sample costs a full         *
 * training run rather than a forward pass. ens_eval() therefore IGNORES   *
 * a requested S, says so on stdout, and uq_evaluate() prints the S it     *
 * actually used with the reason. Do not paper over this when reporting:   *
 * "ensemble at S=5" and "BBB at S=5" cost the same at inference and       *
 * wildly different amounts to obtain.                                     *
 *                                                                        *
 * ---------------------------------------------------------------------- *
 * WHAT ELSE IS DELIBERATE                                                 *
 *                                                                        *
 *   NLPD single   the first draw alone, against the S-draw mixture. For   *
 *                 the ensemble that is member 0; for a BNN arm it is one  *
 *                 posterior sample. Same meaning, same column.            *
 *   ECE           UQ_BINS bins on next-token argmax confidence, ID only.  *
 *   OOD           uniform-random token ids, seeded 99991. Synthetic, but  *
 *                 directional: epistemic MUST rise out there, and the     *
 *                 report warns when it does not.                          *
 *   split         printed with every number. There is no default that is  *
 *                 safe to leave unstated in a paper.                      *
 * ===================================================================== */

// Mirror of ckpt_save. Returns the step count from the header, or -1.
// The shape fields are checked, not just read: loading a C=768 checkpoint into
// a C=384 run would otherwise read 44 MB of the wrong tensor and report
// confident nonsense rather than failing.
static int ckpt_load(const char *arm, const char *what, float *d_buf,
                     size_t n, Config cfg, int steps) {
    char path[512];
    snprintf(path, sizeof path, "runs/%s_C%d_L%d_%dk_%s.bin",
             arm, cfg.C, cfg.L, steps / 1000, what);
    FILE *f = fopen(path, "rb");
    if (!f) { printf("ckpt: cannot open %s\n", path); return -1; }
    int hdr[8];
    if (fread(hdr, sizeof(int), 8, f) != 8) {
        printf("ckpt: short header in %s\n", path); fclose(f); return -1;
    }
    if (hdr[0] != 20240520) {
        printf("ckpt: bad magic %d in %s\n", hdr[0], path); fclose(f); return -1;
    }
    size_t got = (size_t)(unsigned int)hdr[1] | ((size_t)(unsigned int)hdr[7] << 32);
    if (got != n || hdr[2] != cfg.C || hdr[3] != cfg.L ||
        hdr[4] != cfg.V || hdr[5] != cfg.NH) {
        printf("ckpt: shape mismatch in %s\n"
               "      file n=%zu C=%d L=%d V=%d NH=%d\n"
               "      run  n=%zu C=%d L=%d V=%d NH=%d\n",
               path, got, hdr[2], hdr[3], hdr[4], hdr[5],
               n, cfg.C, cfg.L, cfg.V, cfg.NH);
        fclose(f); return -1;
    }
    float *h = (float *)malloc(n * sizeof(float));
    if (!h) { printf("ckpt: host alloc failed for %s\n", path); fclose(f); return -1; }
    if (fread(h, sizeof(float), n, f) != n) {
        printf("ckpt: short read on %s\n", path); free(h); fclose(f); return -1;
    }
    fclose(f);
    CUDA_CHECK(cudaMemcpy(d_buf, h, n * sizeof(float), cudaMemcpyHostToDevice));
    free(h);
    printf("ckpt: loaded %s  (%zu floats, %.0f MB, %d steps)\n",
           path, n, n * 4 / 1e6, hdr[6]);
    return hdr[6];
}

// Token count from a .bin header WITHOUT loading the tokens. Vadam's sigma is
// 1/sqrt(N*s + lambda) with N = the TRAINING token count, so evaluating a
// Vadam checkpoint on the val split still needs the train split's N. Reading
// 1 KB of header beats re-reading 90M tokens to learn one integer.
static long loader_count(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { printf("cannot open %s\n", path); UNCERTAIN_ABORT("loader_count: cannot open file"); }
    int header[256];
    if (fread(header, sizeof(int), 256, f) != 256) {
        printf("bad header in %s\n", path); UNCERTAIN_ABORT("loader_count: bad header");
    }
    fclose(f);
    if (header[0] != 20240520) { printf("bad magic %d in %s\n", header[0], path); UNCERTAIN_ABORT("loader_count: bad magic"); }
    return header[2];
}

// A typo must not silently become a val-split number wearing a test-split
// label, so an unknown name exits rather than falling back.
static const char *uq_split_path(const char *split) {
    if (strcmp(split, "val")  == 0) return DATA_VAL;
    if (strcmp(split, "test") == 0) return DATA_TEST;
    printf("unknown split '%s' — use 'val' or 'test'\n", split);
    UNCERTAIN_ABORT("unknown split");
}

#define UQ_BINS 15
// Eval draws come from a Philox stream disjoint from every training stream, so
// a posterior sample at eval is never the exact eps some training step used.
#define UQ_EVAL_SEED  (20240520ull ^ 0xE7A1E7A1ull)

// "Populate a parameter vector for sample s and return a device pointer to it."
// Returning a pointer rather than filling a fixed buffer lets the ensemble hand
// back member s with no copy at all, while the BNN arms fill their scratch w.
typedef const float *(*uq_sample_fn)(int s, void *ctx);

typedef struct {
    Config cfg;
    int    S;                // posterior samples per position
    int    nbatches;         // eval batches of B*T positions (ID; same for OOD)
    const char *arm;         // "bbb" / "vadam" / "lowr" / "ens"
    const char *split;       // printed with every number
    const char *mix_label;   // what the S-sample mixture is called in the report
    int    s_fixed_by_arm;   // 1 for the ensemble: S is E, not a choice
    int    fixed_batch;      // 1 = evaluate whatever is already in d_inp/d_tgt
} UQSpec;

typedef struct {
    double nlpd_single, nlpd_mix;      // per position
    double total, alea, epi;           // per position, ID
    double epi_ood;                    // per position, OOD
    double ece;
    double id_err_max;                 // worst |total-(alea+epi)| over ID and OOD
    long   npos, npos_ood;
} UQResult;

// Sums over positions. All math in double — the identity check at 1e-5 is only
// meaningful if the arithmetic is cleaner than the tolerance.
typedef struct {
    double nlpd_mix, nlpd_single;
    double total, alea, epi;
    double id_err_max;
    double bin_conf[UQ_BINS], bin_acc[UQ_BINS]; long bin_n[UQ_BINS];
    long npos;
} UQAcc;

// Fold one batch: S prob rows per position -> the accumulator. This is the
// function that used to be train_ensemble.cu's eval_batch, with the ensemble's
// E replaced by S. Pass S = E and the arithmetic is identical, term for term
// and in the same order, which is what keeps the control arm's numbers fixed.
static void uq_fold_batch(UQAcc *acc, float **probs /*S x BT*V*/, const int *tgt,
                          int BT, int V, int S, double *pbar /*V scratch*/,
                          int with_nlpd) {
    for (int p = 0; p < BT; p++) {
        // p_bar: the mixture of the S samples, in double
        for (int y = 0; y < V; y++) pbar[y] = 0.0;
        for (int s = 0; s < S; s++) {
            const float *ps = probs[s] + (size_t)p * V;
            for (int y = 0; y < V; y++) pbar[y] += ps[y];
        }
        double h_bar = 0.0;
        int argmax = 0;
        for (int y = 0; y < V; y++) {
            pbar[y] /= S;
            if (pbar[y] > 0.0) h_bar -= pbar[y] * log(pbar[y]);
            if (pbar[y] > pbar[argmax]) argmax = y;
        }

        // aleatoric = mean sample entropy; epistemic = mean KL(p_s || p_bar),
        // computed DIRECTLY (not as the difference) so the identity check
        // below actually tests something.
        double h_mean = 0.0, kl_mean = 0.0;
        for (int s = 0; s < S; s++) {
            const float *ps = probs[s] + (size_t)p * V;
            double h = 0.0, kl = 0.0;
            for (int y = 0; y < V; y++) {
                double q = ps[y];
                if (q > 0.0) {                 // p_bar_y >= q/S > 0, so log is finite
                    double lq = log(q);
                    h  -= q * lq;
                    kl += q * (lq - log(pbar[y]));
                }
            }
            h_mean += h; kl_mean += kl;
        }
        h_mean /= S; kl_mean /= S;

        double id_err = fabs(h_bar - (h_mean + kl_mean));
        if (id_err > acc->id_err_max) acc->id_err_max = id_err;

        acc->total += h_bar; acc->alea += h_mean; acc->epi += kl_mean;

        if (with_nlpd) {
            int y = tgt[p];
            acc->nlpd_mix    -= log(fmax(pbar[y], 1e-30));
            acc->nlpd_single -= log(fmax((double)probs[0][(size_t)p * V + y], 1e-30));
            int b = (int)(pbar[argmax] * (double)UQ_BINS);
            if (b > UQ_BINS - 1) b = UQ_BINS - 1;
            acc->bin_conf[b] += pbar[argmax];
            acc->bin_acc[b]  += (argmax == y) ? 1.0 : 0.0;
            acc->bin_n[b]++;
        }
        acc->npos++;
    }
}

// A GPU-free check of the decomposition itself, on three cases whose answers
// are known on paper. Every arm's eval runs it first, so the shared math is
// gated before an hour of forward passes rather than after. Costs microseconds.
static int uq_selftest(void) {
    const int V = 2;
    double pbar[2];
    float pa[2] = { 1.00f, 0.00f };      // one-hot
    float pb[2] = { 0.00f, 1.00f };      // the opposite one-hot
    float pc[2] = { 0.25f, 0.75f };      // exactly representable in float
    float *one[1]      = { pc };
    float *same[2]     = { pc, pc };
    float *disjoint[2] = { pa, pb };
    const double L2 = log(2.0);
    const double Hc = -(0.25 * log(0.25) + 0.75 * log(0.75));
    const double TOL = 1e-12;            // exact cases; 1 ulp is ~1e-16
    UQAcc acc; int fails = 0;

    // S=1: one sample disagrees with nothing, so epistemic is exactly zero
    // and total collapses onto aleatoric.
    memset(&acc, 0, sizeof acc);
    uq_fold_batch(&acc, one, NULL, 1, V, 1, pbar, 0);
    if (fabs(acc.epi) > TOL || fabs(acc.alea - Hc) > TOL || fabs(acc.total - Hc) > TOL) {
        printf("uq_selftest: S=1 case failed (total %.17g alea %.17g epi %.17g)\n",
               acc.total, acc.alea, acc.epi); fails++;
    }
    // Two IDENTICAL samples: still nothing to disagree about.
    memset(&acc, 0, sizeof acc);
    uq_fold_batch(&acc, same, NULL, 1, V, 2, pbar, 0);
    if (fabs(acc.epi) > TOL || fabs(acc.total - Hc) > TOL) {
        printf("uq_selftest: identical-samples case failed (total %.17g epi %.17g)\n",
               acc.total, acc.epi); fails++;
    }
    // Maximal disagreement: two disjoint one-hots. Each sample is certain
    // (alea = 0) and the mixture is maximally uncertain (total = epi = log 2).
    memset(&acc, 0, sizeof acc);
    uq_fold_batch(&acc, disjoint, NULL, 1, V, 2, pbar, 0);
    if (fabs(acc.alea) > TOL || fabs(acc.epi - L2) > TOL || fabs(acc.total - L2) > TOL) {
        printf("uq_selftest: disjoint-one-hot case failed (total %.17g alea %.17g epi %.17g)\n",
               acc.total, acc.alea, acc.epi); fails++;
    }
    if (acc.id_err_max > TOL) {
        printf("uq_selftest: identity error %.2e exceeds %.0e\n", acc.id_err_max, TOL);
        fails++;
    }
    if (fails) printf("uq_selftest: %d CASE(S) FAILED — the shared decomposition is wrong\n", fails);
    return fails;
}

/* The driver. Device buffers come from the caller because every arm already
   owns them; only the host prob rows are allocated here (S x B*T*V floats —
   27 KB per sample at V=27, so S is bounded by patience, not memory).
   Returns 0, or 1 if the identity assertion fired. */
static int uq_evaluate(const UQSpec *spec, uq_sample_fn sample_fn, void *ctx,
                       Loader *ld, float *d_acts, int *d_inp, int *d_tgt,
                       int *h_inp, int *h_tgt, float *h_losses, UQResult *out) {
    Config cfg = spec->cfg;
    ActOff a = act_offsets(cfg);
    int BT = cfg.B * cfg.T, V = cfg.V, S = spec->S;

    printf("eval: arm %-5s | split %s | S=%d%s | %d batches x %d tokens"
           " = %d ID positions (+%d OOD)\n",
           spec->arm, spec->split, S,
           spec->s_fixed_by_arm ? " (FIXED by the arm: S = trained members)" : "",
           spec->nbatches, BT, spec->nbatches * BT, spec->nbatches * BT);

    float **h_probs = (float **)malloc((size_t)S * sizeof(float *));
    for (int s = 0; s < S; s++)
        h_probs[s] = (float *)malloc((size_t)BT * V * sizeof(float));
    double *pbar = (double *)malloc((size_t)V * sizeof(double));
    UQAcc id; memset(&id, 0, sizeof id);
    UQAcc ood; memset(&ood, 0, sizeof ood);

    // ---- in-distribution: held-out text -----------------------------------
    for (int b = 0; b < spec->nbatches; b++) {
        if (!spec->fixed_batch) {
            loader_next(ld, h_inp, h_tgt, BT);
            CUDA_CHECK(cudaMemcpy(d_inp, h_inp, BT * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_tgt, h_tgt, BT * sizeof(int), cudaMemcpyHostToDevice));
        }
        for (int s = 0; s < S; s++) {
            const float *w = sample_fn(s, ctx);
            gpt2_forward(d_acts, w, d_inp, d_tgt, cfg, h_losses);
            CUDA_CHECK(cudaMemcpy(h_probs[s], d_acts + a.prb,
                                  (size_t)BT * V * sizeof(float), cudaMemcpyDeviceToHost));
        }
        uq_fold_batch(&id, h_probs, h_tgt, BT, V, S, pbar, 1);
    }

    // ---- OOD: uniform-random token ids ------------------------------------
    // Not another text domain — a direction. Sample disagreement must RISE out
    // here or the arm is not capturing model uncertainty at all.
    srand(99991);
    for (int b = 0; b < spec->nbatches; b++) {
        for (int i = 0; i < BT; i++) { h_inp[i] = rand() % V; h_tgt[i] = rand() % V; }
        CUDA_CHECK(cudaMemcpy(d_inp, h_inp, BT * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_tgt, h_tgt, BT * sizeof(int), cudaMemcpyHostToDevice));
        for (int s = 0; s < S; s++) {
            const float *w = sample_fn(s, ctx);
            gpt2_forward(d_acts, w, d_inp, d_tgt, cfg, h_losses);
            CUDA_CHECK(cudaMemcpy(h_probs[s], d_acts + a.prb,
                                  (size_t)BT * V * sizeof(float), cudaMemcpyDeviceToHost));
        }
        uq_fold_batch(&ood, h_probs, h_tgt, BT, V, S, pbar, 0);
    }

    double np = (double)id.npos;
    double ece = 0.0;
    for (int b = 0; b < UQ_BINS; b++)
        if (id.bin_n[b]) ece += fabs(id.bin_acc[b] - id.bin_conf[b]) / np;

    out->nlpd_single = id.nlpd_single / np;
    out->nlpd_mix    = id.nlpd_mix    / np;
    out->total       = id.total / np;
    out->alea        = id.alea  / np;
    out->epi         = id.epi   / np;
    out->epi_ood     = ood.npos ? ood.epi / (double)ood.npos : 0.0;
    out->ece         = ece;
    out->id_err_max  = fmax(id.id_err_max, ood.id_err_max);
    out->npos        = id.npos;
    out->npos_ood    = ood.npos;

    for (int s = 0; s < S; s++) free(h_probs[s]);
    free(h_probs); free(pbar);

    // THE gate. See the header: this is exact algebra, so a violation means the
    // probabilities are wrong, not the model.
    if (out->id_err_max > 1e-5) {
        fprintf(stderr, "IDENTITY VIOLATION: |total-(alea+epi)| = %.2e (tol 1e-5)\n",
                out->id_err_max);
        return 1;
    }
    return 0;
}

// Two "summary:" lines, the format scorecard.c already parses and passes
// through. mix_label is what the S-sample mixture is called, so the control
// arm keeps saying "ensemble" and its output is unchanged.
static void uq_report(const UQSpec *spec, const UQResult *r) {
    printf("summary: NLPD single %.4f -> %s %.4f | ECE(%d) %.4f | identity_err %.1e\n",
           r->nlpd_single, spec->mix_label, r->nlpd_mix, UQ_BINS, r->ece, r->id_err_max);
    printf("summary: uncertainty (nats)  ID: total %.3f = alea %.3f + epi %.4f   OOD: epi %.4f  %s\n",
           r->total, r->alea, r->epi, r->epi_ood,
           r->epi_ood > r->epi ? "(OOD > ID, as required)" : "(WARN: OOD <= ID!)");
}

/* The eval CLI, identical for every arm, so the four are invoked the same way
   and an unequal S cannot creep in by accident:

       <binary> eval [steps] [S] [batches] [split]

   steps    which checkpoint to load — runs/<arm>_C<C>_L<L>_<steps/1000>k_*.bin
   S        posterior samples per position (ensemble: ignored, S = E)
   batches  eval batches, B*T positions each, ID and again for OOD
   split    "val" or "test". Never inferred, always printed.                */
typedef struct { int steps, S, nbatches; const char *split; } UQArgs;

static UQArgs uq_parse_args(int argc, char **argv) {
    UQArgs u; u.steps = 100000; u.S = 8; u.nbatches = 64; u.split = "val";
    if (argc > 2) u.steps    = atoi(argv[2]);
    if (argc > 3) u.S        = atoi(argv[3]);
    if (argc > 4) u.nbatches = atoi(argv[4]);
    if (argc > 5) u.split    = argv[5];
    if (u.S < 1)        u.S = 1;
    if (u.nbatches < 1) u.nbatches = 1;
    return u;
}

#endif  // GPT2_KERNELS_CUH
