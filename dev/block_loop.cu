/*
Finite-difference gradient check on the L>1 transformer block loop.

This is the one code path in the repo with no CPU reference: train_bnn.c is
single-block, so nothing validated the loop in gpt2_forward/gpt2_backward that
stacks L blocks and threads the residual stream between them. The only evidence
was that overfit still reaches 0.0070 at L=6, which a subtly wrong inter-block
gradient would not necessarily break. It affects ALL FOUR ARMS equally.

    dL/dw from gpt2_backward   vs   (L(w+h) - L(w-h)) / 2h

Run at L=1 first and L=2 second, deliberately. L=1 is the configuration
train_bnn.c already validates, so it calibrates the harness: if L=1's error is
at the finite-difference floor and L=2's is the same size, the block loop is
correct. If only L=2 degrades, the loop is where it degrades.

WHAT MAKES THIS TEST LOAD-BEARING is the choice of probes, not the loss. A
gradient for a parameter in BLOCK 0 must travel backwards through block 1 to
reach it, so a broken residual thread between blocks shows up there and nowhere
else. Block 0, block 1 and the shared tensors are therefore reported separately;
a pass on "block 1" alone would prove almost nothing.

-----------------------------------------------------------------------------
WHY C = 64 AND NH = 1, NOT THE C = 8 THIS WAS SPECIFIED AT

HS in the FlashAttention kernels is a compile-time `#define HS 64`, and the
kernels index q/k/v with it (`h * HS + i`) while sizing shared memory as
Ks[BC][HS+1]. The head size C/NH must therefore BE 64. At C=8, NH=2 the kernel
would read 64 floats per head out of a 4-float head and return garbage with no
error — the comment at gpt2_kernels.cuh's FlashAttention section says "HS must
equal C/NH" but nothing enforced it. This file's config satisfies it, and
gpt2_forward now checks it at runtime so the next person gets a message instead
of silent nonsense.

C=64/NH=1/T=4/B=2/L=2 is 103,835 parameters — small enough that a full forward
per probe is instant, large enough to exercise the loop and the residual thread.

Compile:  cd dev && make block_loop
Run:      ./block_loop
*/

#include "../gpt2_kernels.cuh"
#include <float.h>

/* ---------------------------------------------------------------------------
 DIRECTIONAL DERIVATIVES, NOT PER-PARAMETER ONES.

 The obvious test — nudge one weight, difference the loss — does not work on a
 float32 forward at this size, and failed identically at L=1 (which train_bnn.c
 validates) and L=2 before this was fixed. The arithmetic: a single weight's
 central difference has signal 2*h*g_i ~ 2e-6 while the float32 forward carries
 ~1e-6 of absolute noise in the loss. That is a ~50% error bar on a perfectly
 correct gradient, so the test measured its own floor and nothing else.

 Perturbing the WHOLE group along a Rademacher direction v (+-1 per element)
 fixes it. The signal becomes |sum_i g_i v_i| ~ sqrt(len) * |g|_rms, up by a
 factor of sqrt(len) — 128x for a 16K-element weight matrix — while the noise
 in the loss is unchanged. The analytic side is the same dot product, so the
 test is exactly as sharp about which parameters it covers: v is supported on
 ONE group, so "block 0" still means block 0.
--------------------------------------------------------------------------- */
// h is RELATIVE to each group's own scale, not absolute. The LayerNorm gains
// live at 1.0 and the weight matrices at 0.02, so one absolute h under-perturbs
// the gains by 50x and leaves their gradients uncertified — the run before this
// reported exactly that, "a 1% gradient error reads BELOW NOISE".
#define FD_REL 5e-3    // perturbation as a fraction of the group's RMS
#define NDIR   5       // random directions per group

typedef struct { const char *name; size_t off; size_t len; } Group;

static Config CFG = { .B = 2, .T = 4, .C = 64, .V = VOCAB, .NH = 1,
                      .maxT = 8, .L = 1 };

static float *d_w, *d_dw, *d_acts, *d_gacts;
static int   *d_inp, *d_tgt;
static float *h_w, *h_losses;
static size_t NP;

// The loss, summed in double on the host. gpt2_forward's own float return
// would cost a digit of the very quantity the differences are taken of.
static double loss_now(void) {
    gpt2_forward(d_acts, d_w, d_inp, d_tgt, CFG, h_losses);
    double s = 0.0;
    for (int i = 0; i < CFG.B * CFG.T; i++) s += h_losses[i];
    return s / (CFG.B * CFG.T);
}

static void upload(void) {
    CUDA_CHECK(cudaMemcpy(d_w, h_w, NP * sizeof(float), cudaMemcpyHostToDevice));
}

int main(int argc, const char **argv) {
    (void)argc; (void)argv;
    srand(1337);
    CUBLAS_CHECK(cublasCreate(&cublas_handle));

    int BT = CFG.B * CFG.T;
    int fails = 0;

    for (int L = 1; L <= 2; L++) {
        CFG.L = L;
        ParamOff p = param_offsets(CFG);
        ActOff   a = act_offsets(CFG);
        NP = p.total;
        printf("\n=========== L = %d   (C=%d NH=%d T=%d B=%d, %zu params) ===========\n",
               L, CFG.C, CFG.NH, CFG.T, CFG.B, NP);

        h_w      = (float *)malloc(NP * sizeof(float));
        h_losses = (float *)malloc(BT * sizeof(float));
        float *ana  = (float *)malloc(NP * sizeof(float));
        float *ana2 = (float *)malloc(NP * sizeof(float));

        // Same init the trainers use, except the LayerNorm gains. The trainers
        // draw every parameter from N(0, 0.02) including the gains, which makes
        // the network nearly dead at step 0 — fine for training, useless for a
        // gradient check, where the point is a well-conditioned function.
        for (size_t i = 0; i < NP; i++) {
            double u1 = ((double)rand() + 1.0) / ((double)RAND_MAX + 2.0);
            double u2 = ((double)rand() + 1.0) / ((double)RAND_MAX + 2.0);
            h_w[i] = (float)(0.02 * sqrt(-2.0 * log(u1)) * cos(6.2831853 * u2));
        }
        for (int l = 0; l < L; l++)
            for (int c = 0; c < CFG.C; c++) {
                h_w[PBLK(p, l, lng)  + c] = 1.0f;
                h_w[PBLK(p, l, ln2g) + c] = 1.0f;
            }
        for (int c = 0; c < CFG.C; c++) h_w[p.flng + c] = 1.0f;

        int *h_inp = (int *)malloc(BT * sizeof(int));
        int *h_tgt = (int *)malloc(BT * sizeof(int));
        for (int i = 0; i < BT; i++) {
            h_inp[i] = rand() % CFG.V;
            h_tgt[i] = rand() % CFG.V;
        }

        CUDA_CHECK(cudaMalloc(&d_w,     NP * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_dw,    NP * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_acts,  a.total * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_gacts, a.total * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_inp, BT * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_tgt, BT * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_inp, h_inp, BT * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_tgt, h_tgt, BT * sizeof(int), cudaMemcpyHostToDevice));
        upload();

        double L0 = loss_now();
        printf("loss at the probe point: %.6f\n", L0);

        // Analytic gradients, twice. layernorm_backward, encoder_backward and
        // attention_backward accumulate with atomics, so the same inputs do not
        // give bitwise-identical gradients. That spread is the floor below which
        // an FD disagreement means nothing, so it is measured, not assumed.
        for (int pass = 0; pass < 2; pass++) {
            CUDA_CHECK(cudaMemset(d_dw,    0, NP * sizeof(float)));
            CUDA_CHECK(cudaMemset(d_gacts, 0, a.total * sizeof(float)));
            gpt2_forward(d_acts, d_w, d_inp, d_tgt, CFG, h_losses);
            gpt2_backward(d_dw, d_gacts, d_acts, d_w, d_inp, d_tgt, CFG);
            CUDA_CHECK(cudaMemcpy(pass ? ana2 : ana, d_dw, NP * sizeof(float),
                                  cudaMemcpyDeviceToHost));
        }
        double atomic_spread = 0.0;
        for (size_t i = 0; i < NP; i++) {
            double d = fabs((double)ana[i] - ana2[i]);
            double s = fabs((double)ana[i]);
            if (s > 1e-9 && d / s > atomic_spread) atomic_spread = d / s;
        }
        printf("backward run-to-run spread (atomics): %.2e relative\n\n",
               atomic_spread);

        Group groups[8];
        int ng = 0;
        groups[ng++] = (Group){ "wte  (shared)",   p.wte,  (size_t)CFG.V * CFG.C };
        groups[ng++] = (Group){ "wpe  (shared)",   p.wpe,  (size_t)CFG.maxT * CFG.C };
        // Block 0 is the load-bearing group when L=2: its gradient only exists
        // if the residual stream was threaded back through block 1 correctly.
        groups[ng++] = (Group){ "blk0 qkv w",      PBLK(p,0,qw), (size_t)CFG.C * 3 * CFG.C };
        groups[ng++] = (Group){ "blk0 mlp w1",     PBLK(p,0,w1), (size_t)CFG.C * 4 * CFG.C };
        groups[ng++] = (Group){ "blk0 ln1 gain",   PBLK(p,0,lng), (size_t)CFG.C };
        if (L > 1) {
            groups[ng++] = (Group){ "blk1 qkv w",  PBLK(p,1,qw), (size_t)CFG.C * 3 * CFG.C };
            groups[ng++] = (Group){ "blk1 mlp w1", PBLK(p,1,w1), (size_t)CFG.C * 4 * CFG.C };
        }
        groups[ng++] = (Group){ "head hw",         p.hw,   (size_t)CFG.C * CFG.V };

        signed char *v = (signed char *)malloc(NP);
        for (int g = 0; g < ng; g++) {
            // Scale: ||g_G||, which is exactly E[(g.v)^2]^(1/2) over Rademacher
            // v. Normalizing by |g.v| itself would divide by a near-zero every
            // time a direction landed close to orthogonal to g, and report a
            // correct gradient as a 100% error — the same trap as the low-rank
            // check-3 off-diagonals. A wrong block loop perturbs g by O(||g||),
            // so this denominator gives up none of the test's power.
            double gnorm = 0.0, wrms = 0.0;
            for (size_t j = 0; j < groups[g].len; j++) {
                double gi = ana[groups[g].off + j], wi = h_w[groups[g].off + j];
                gnorm += gi * gi;
                wrms  += wi * wi;
            }
            gnorm = sqrt(gnorm);
            double fd_h = FD_REL * sqrt(wrms / groups[g].len);
            double worst = 0.0, sum = 0.0, sens = 0.0;
            for (int k = 0; k < NDIR; k++) {
                size_t off = groups[g].off, len = groups[g].len;
                double ana_dir = 0.0;
                for (size_t j = 0; j < len; j++) {
                    v[j] = (rand() & 1) ? 1 : -1;
                    ana_dir += (double)ana[off + j] * v[j];
                }
                double Ls[2];
                for (int sgn = 0; sgn < 2; sgn++) {
                    double step = sgn ? -fd_h : fd_h;
                    for (size_t j = 0; j < len; j++)
                        h_w[off + j] = (float)(h_w[off + j] + step * v[j]);
                    upload(); Ls[sgn] = loss_now();
                    for (size_t j = 0; j < len; j++)
                        h_w[off + j] = (float)(h_w[off + j] - step * v[j]);
                }
                upload();
                double num = (Ls[0] - Ls[1]) / (2.0 * fd_h);
                double rel = gnorm > 1e-12 ? fabs(num - ana_dir) / gnorm : 0.0;
                if (rel > worst) worst = rel;
                sum += rel;
                // How loudly would a 1% error in THIS group's gradient read?
                // Free — no extra forward — and it is what makes the PASS above
                // mean something: a check that cannot fail has proved nothing.
                if (gnorm > 1e-12) sens += 0.01 * fabs(ana_dir) / gnorm;
            }
            // 2%: the central-difference floor at this h, measured at L=1 where
            // the gradients are already validated against train_bnn.c. Far below
            // what a wrong block loop would produce — a dropped residual path
            // changes a gradient by O(1), not by a percent.
            int ok = worst < 2e-2;
            // A group only certifies its gradients if a wrong one would show:
            // the 1% probe must land clear of the noise this group achieved.
            double sens_avg = sens / NDIR;
            int sharp = sens_avg > 5.0 * (sum / NDIR);
            printf("  %-14s worst %.3e  mean %.3e  |g| %.3e  | a 1%% gradient "
                   "error reads %.3e %s  %s\n",
                   groups[g].name, worst, sum / NDIR, gnorm, sens_avg,
                   sharp ? "(detectable)" : "(BELOW NOISE)", ok ? "PASS" : "FAIL");
            fails += !sharp;
            fails += !ok;
        }
        free(v);

        free(h_w); free(h_losses); free(ana); free(ana2); free(h_inp); free(h_tgt);
        cudaFree(d_w); cudaFree(d_dw); cudaFree(d_acts); cudaFree(d_gacts);
        cudaFree(d_inp); cudaFree(d_tgt);
    }

    printf("\n%s\n", fails ? "BLOCK LOOP CHECK FAILED"
                           : "L=1 and L=2 both match finite differences — the "
                             "block loop and the residual thread are correct.");
    cublasDestroy(cublas_handle);
    return fails ? 1 : 0;
}
