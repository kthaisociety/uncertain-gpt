/*
Step 1 of the GPT-2 build: the data loader.

Its only job: get Shakespeare (already converted to numbers) off the disk
and into memory, in small handfuls the model can learn from.

Build and run from inside gpt2/:
    gcc -O2 train_gpt2.c -lm -o train_gpt2 && ./train_gpt2
*/

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>      // sqrtf — remember -lm at link time
#include <string.h>    // memset — zeroing the gradient buffers each step

// The file starts with a 256-slot label ("header") describing what's inside.
#define HEADER_INTS 256
#define MAGIC   20240520   // a fixed number that says "yes, this is one of our files"
#define VERSION 1

// NOTE: B and T are NOT #defines. They live inside main() as plain ints, so you
// can change them without recompiling logic around them. (A #define named B
// would also textually replace every letter B in the file — including the `int B`
// in a function's parameter list, which is a syntax error.)

/*
The encoder: turn token IDs into vectors.

    out[b][t][c] = wte[ inp[b][t] ][c]  +  wpe[t][c]

Shapes:
    out   (B, T, C)      what we write        e.g. (4, 64, 768)
    inp   (B, T)         token IDs            e.g. (4, 64)  = 256 ints
    wte   (V, C)         meaning per token    (50257, 768)
    wpe   (maxT, C)      meaning per position (1024, 768)

Every array is FLAT in memory. The shapes above are a story we tell about a
plain line of floats, and the index arithmetic is how we tell it.
*/
void encoder_forward(float *out, int *inp, float *wte, float *wpe,
                     int B, int T, int C) {

    for (int b = 0; b < B; b++) {          // which sequence
        for (int t = 0; t < T; t++) {      // which token inside it

            // The token ID sitting at row b, position t.
            // Looked up ONCE per token — all 768 floats below come from
            // this same row of wte.
            int ix = inp[b * T + t];

            // Three row pointers. Each one says "the start of a 768-float row".
            // Computing them here means the multiplies happen once per token
            // instead of once per float.
            float *out_bt = out + b * T * C + t * C;  // where this token's vector goes
            float *wte_ix = wte + ix * C;             // what this token MEANS
            float *wpe_t  = wpe + t * C;              // what this POSITION means

            // Add the two 768-long rows together, element by element.
            for (int c = 0; c < C; c++) {
                out_bt[c] = wte_ix[c] + wpe_t[c];
            }
        }
    }
}
/*
LayerNorm: the thermostat. Each token's 768 floats are re-centered to mean 0
and spread 1, then nudged by learned per-channel knobs (g scale, b shift).

    out[b][t][c] = g[c] * (inp[b][t][c] - mean[b][t]) * rstd[b][t] + b[c]

Shapes:
    inp, out    (B, T, C)
    weight,bias (C,)       learned; init g=1, b=0 (identity at birth)
    mean, rstd  (B, T)     per-token readings, SAVED — backward needs them

The math runs independently per token: 256 separate thermostat readings,
nothing mixes across rows. (That's why CUDA can later give each row its own
warp — prac4 reduction, one per (b,t).)
*/
void layernorm_forward(float *out, float *mean, float *rstd,
                       float *inp, float *weight, float *bias,
                       int B, int T, int C) {
    const float eps = 1e-5f;   // so we never divide by zero

    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {

            float *x = inp + b * T * C + t * C;   // this token's row, read
            float *o = out + b * T * C + t * C;   // same row, written
            // the pointer points to the start of C in this loop that is why you can use x that way in this case
            // 1. mean of the 768 floats
            float sum = 0.0f;
            for (int c = 0; c < C; c++) sum += x[c];
            float m = sum / C;

            // 2. variance: average squared distance from the mean
            float varsum = 0.0f;
            for (int c = 0; c < C; c++) {
                float d = x[c] - m;
                varsum += d * d;
            }
            float r = 1.0f / sqrtf(varsum / C + eps);

            // file the readings away — backward will ask for them
            mean[b * T + t] = m;
            rstd[b * T + t] = r;

            // 3. center, scale to unit spread, then the learned nudge
            for (int c = 0; c < C; c++) {
                o[c] = (x[c] - m) * r * weight[c] + bias[c];
            }
        }
    }
}

/*
Matmul: the workhorse. Each token's row (C floats) is dotted against every
column of the weight matrix, producing OC floats per token.

    out[b][t][o] = sum_c inp[b][t][c] * weight[c][o] + bias[o]

Shapes:
    inp    (B, T, C)
    weight (C, OC)    the bridge: size-in x size-out
    bias   (OC,)
    out    (B, T, OC)

The same W serves all B*T tokens — mentally it's one big (B*T, C) @ (C, OC).
This op is QKV, attention-out, both MLP layers, and the logits: ~85% of all
FLOPs in GPT-2.
*/
void matmul_forward(float *out, float *inp, float *weight, float *bias,
                    int B, int T, int C, int OC) {
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {

            float *x = inp + b * T * C + t * C;    // this token's row in
            float *o = out + b * T * OC + t * OC;  // this token's row out
                                                  // (row length OC, not C!)
            for (int oc = 0; oc < OC; oc++) {
                float sum = bias[oc];              // start from the bias
                for (int c = 0; c < C; c++) {
                    // walk down COLUMN o of W: row c, row length OC
                    sum += x[c] * weight[c * OC + oc];
                }
                o[oc] = sum;
            }
        }
    }
}

/*
GELU: the nonlinearity — the bouncer. Negative/small signals get squashed
toward 0, strong positive signals pass nearly unchanged. That kink is what
lets the network learn more than straight lines.

Pure elementwise: out[i] depends ONLY on inp[i]. No rows, no shapes —
that's why it takes a flat count N instead of B/T/C.

    GELU(x) = 0.5x * (1 + tanh( sqrt(2/pi) * (x + 0.044715 x^3) ))
*/
void gelu_forward(float *out, float *inp, int N) {
    const float k = 0.7978845608f;   // sqrt(2/pi)
    for (int i = 0; i < N; i++) {
        float x = inp[i];
        float inner = k * (x + 0.044715f * x * x * x);
        out[i] = 0.5f * x * (1.0f + tanhf(inner));
    }
}

/*
Residual: the skip path. The block's output is ADDED to the stream instead
of replacing it, so the original signal always has a clean highway through
all 12 blocks — and the gradient flows back undiluted (d/dx of x+f(x) has
that "1 +" in it).

    out[i] = x[i] + y[i]

Flat elementwise, like GELU but simpler: no formula, just +.
*/
void residual_forward(float *out, float *x, float *y, int N) {
    for (int i = 0; i < N; i++) {
        out[i] = x[i] + y[i];
    }
}

/*
Attention: the meeting. Tokens talk — each position looks BACK at earlier
positions, scores how relevant each one is, and mixes their Values.

The qkv buffer holds [ Q | K | V ] per token, laid out consecutively:
    qkv[b][t][ 0 ..  C) = query   what this token is looking for
    qkv[b][t][ C .. 2C) = key     what this token advertises
    qkv[b][t][2C .. 3C) = value   what it hands over if picked

The C channels are split into NH heads of hs = C/NH channels each.
Every (b, h) pair runs its OWN little attention: 12 parallel conversations.

    score(t, t2)   = (q_t · k_t2) / sqrt(hs)         for t2 <= t only
    att(t, t2)     = softmax over t2 of score        the mixing weights
    out(t)         = sum_t2 att(t,t2) * v_t2         the V-mix

THE CAUSAL MASK IS THE LOOP BOUND: t2 only runs up to t, so a token can
never see the future. No explicit mask array needed on CPU.

Shapes:
    qkv    (B, T, 3C)
    preatt (B, NH, T, T)   raw scores
    att    (B, NH, T, T)   post-softmax weights (each row sums to 1)
    out    (B, T, C)       the V-mix, heads concatenated back
*/
void attention_forward(float *out, float *preatt, float *att,
                       float *qkv, int B, int T, int C, int NH) {
    int hs = C / NH;                    // head size: 768/12 = 64
    float scale = 1.0f / sqrtf((float)hs);

    for (int b = 0; b < B; b++) { // each sequence
        for (int h = 0; h < NH; h++) { // each attention heads
            for (int t = 0; t < T; t++) { // each token 

                // this token's query, this head's 64 channels
                float *q = qkv + b*T*3*C + t*3*C + h*hs;
                float *pre_row = preatt + b*NH*T*T + h*T*T + t*T; // interim value 
                float *att_row = att    + b*NH*T*T + h*T*T + t*T; // attention value

                // 1. scores against every PAST token (t2 <= t), tracking max
                float max = -1e30f;
                for (int t2 = 0; t2 <= t; t2++) { // you can never look ahead in this case
                    float *k = qkv + b*T*3*C + t2*3*C + C + h*hs;  // K lives at +C
                    float s = 0.0f;
                    for (int i = 0; i < hs; i++) s += q[i] * k[i]; // dot product
                    s *= scale;
                    pre_row[t2] = s;
                    if (s > max) max = s;
                }

                // 2. softmax over the past: exp, sum, divide.
                //    (subtract max first — exp(1000) overflows, exp(x-max) can't)
                float sum = 0.0f;
                for (int t2 = 0; t2 <= t; t2++) {
                    float e = expf(pre_row[t2] - max);
                    att_row[t2] = e;
                    sum += e;
                }
                for (int t2 = 0; t2 <= t; t2++) att_row[t2] /= sum;

                // 3. V-mix: weighted sum of past tokens' values
                float *o = out + b*T*C + t*C + h*hs;
                for (int i = 0; i < hs; i++) o[i] = 0.0f;
                for (int t2 = 0; t2 <= t; t2++) {
                    float *v = qkv + b*T*3*C + t2*3*C + 2*C + h*hs;  // V lives at +2C
                    float w = att_row[t2];
                    for (int i = 0; i < hs; i++) o[i] += w * v[i]; // o still works since we are working in memory in this case
                }
            }
        }
    }
}

/*
Softmax + cross-entropy: the finish line.

For every (b, t) we turn the logits row into a probability distribution
over the vocabulary, then grade the model on the probability it assigned
to the correct next token (targets[b*T + t]).

    probs = softmax(logits)
    loss  = -log(probs[target])

We keep probs around because backward needs them.

Shapes:
    logits (B, T, V)
    probs  (B, T, V)
    losses (B, T)      one scalar per token
    targets (B, T)     integer token IDs
*/
void softmax_crossentropy(float *losses, float *probs, float *logits,
                          int *targets, int B, int T, int V) {
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float *logit_bt = logits + b * T * V + t * V;
            float *prob_bt  = probs  + b * T * V + t * V;

            // 1. softmax, numerically stable
            float max = -1e30f;
            for (int v = 0; v < V; v++) {
                if (logit_bt[v] > max) max = logit_bt[v];
            }
            float sum = 0.0f;
            for (int v = 0; v < V; v++) {
                prob_bt[v] = expf(logit_bt[v] - max);
                sum += prob_bt[v];
            }
            for (int v = 0; v < V; v++) prob_bt[v] /= sum;

            // 2. cross-entropy: negative log of the probability we assigned
            //    to the correct token.
            int target_token = targets[b * T + t];
            losses[b * T + t] = -logf(prob_bt[target_token]);
        }
    }
}

/*
=====================  THE BACKWARD PASS STARTS HERE  =========================

Forward asked: "given these weights, how wrong am I?"  -> one number, the loss.
Backward asks the mirror question: "if I nudged this ONE number up by a hair,
how much would the loss change?"  The answer, for every number in the network,
is its GRADIENT. Nothing more mystical than that: a gradient is a slope.

Three rules that make the whole pass mechanical:

  1. Every buffer gets a SHADOW of identical shape. `logits` is (B,T,V), so
     `dlogits` is (B,T,V). Same offsets, same slicing. d-prefix = "the loss's
     sensitivity to this".

  2. Every backward op has the same job description: it is handed d(its output)
     and must produce d(its input) and d(its weights). It knows nothing about
     the rest of the network. That local ignorance is what lets you chain them.

  3. You walk the forward list in REVERSE. Forward ended at the loss, so
     backward starts at the loss and works back to the embeddings.

Nothing here touches `params`. Backward only MEASURES slopes. Actually moving
the weights downhill is the optimizer's job (Adam, next module).
===============================================================================

First link in the chain: loss -> logits.

Forward did two things to each row of V logits: softmax turned them into
probabilities, then cross-entropy picked out the target's probability and took
-log of it. You would expect backward to need two separate steps. It doesn't —
the two derivatives multiply together and almost everything cancels, leaving:

    dlogits[v] = probs[v] - (1 if v is the target else 0)

That is the single most famous simplification in deep learning, and it's worth
reading in plain language:

    "The gradient is how much probability mass you put in the wrong place."

If the model was perfect (p=1 on the target, 0 elsewhere) the row is all zeros:
no slope, nothing to learn, you're already right. If the model put 0.7 on some
wrong token, that entry gets +0.7 — a positive slope, meaning "raising this
logit raises the loss", so the optimizer will push it DOWN. The target entry
gets p-1, which is always negative, meaning "push this one UP". The row always
sums to exactly zero: softmax is a zero-sum game, and pushing one token up
necessarily pushes the others down.

The 1/(B*T): your reported loss is the MEAN over all B*T positions, not the sum.
Every position therefore owns only 1/(B*T) of the blame, and that factor has to
ride along from the very first step or every gradient downstream is B*T times
too big.
*/
void crossentropy_softmax_backward(float *dlogits, float *probs,
                                   int *targets, int B, int T, int V) {
    float dloss = 1.0f / (B * T);   // mean, not sum -> each position's share

    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float *dlogit_bt = dlogits + b * T * V + t * V;
            float *prob_bt   = probs   + b * T * V + t * V;
            int target_token = targets[b * T + t];

            for (int v = 0; v < V; v++) {
                // 1.0 on the correct token, 0 everywhere else — written as an
                // expression instead of an if, so the inner loop stays branch-
                // free (this matters a lot once it becomes a CUDA kernel:
                // a branch here would split the warp).
                float indicator = (v == target_token) ? 1.0f : 0.0f;
                dlogit_bt[v] += (prob_bt[v] - indicator) * dloss;
            }
        }
    }
}

/*
Matmul backward: the workhorse in reverse.

Forward:  out[b,t,oc] = sum_c inp[b,t,c] * weight[c,oc] + bias[oc]
Backward contributions:
  dweight[c,oc] += sum_{b,t} inp[b,t,c] * dout[b,t,oc]
  dbias[oc]     += sum_{b,t} dout[b,t,oc]
  dinp[b,t,c]   += sum_{oc} dout[b,t,oc] * weight[c,oc]

All three use += to follow the gradient-accumulation convention.
*/
void matmul_backward(float *dinp, float *dweight, float *dbias,
                     float *inp, float *weight, float *dout,
                     int B, int T, int C, int OC) {
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float *x  = inp  + b * T * C + t * C;   // this token's input row
            float *dx = dinp + b * T * C + t * C;   // gradient for that row
            float *dy = dout + b * T * OC + t * OC; // upstream gradient row

            for (int oc = 0; oc < OC; oc++) {
                float dy_oc = dy[oc];               // upstream gradient for output oc
                dbias[oc] += dy_oc;                 // bias gets all of it

                for (int c = 0; c < C; c++) {
                    dweight[c * OC + oc] += x[c] * dy_oc;  // outer product
                    dx[c] += dy_oc * weight[c * OC + oc];  // propagate to input
                }
            }
        }
    }
}

/*
LayerNorm backward: the first op where the gradient is NOT elementwise.

Forward, per token row:
    m     = mean(x)                     over the C floats of this row
    r     = 1/sqrt(var(x) + eps)
    xhat  = (x - m) * r
    out   = xhat * weight + bias

The trap: it is tempting to write dinp = dout * weight * r. That would be right
only if m and r were constants. They are not — they are computed FROM the row.
Nudge x[c] and you move m and r, which moves ALL C outputs of that row. So the
gradient comes back along three paths: the direct one, the mean path, and the
variance path. The last two show up as row-wide averages that every c must
subtract.

    dnorm[c]  = dout[c] * weight[c]
    dinp[c]   = r * ( dnorm[c]
                    - mean_c(dnorm)                 <- the mean path
                    - xhat[c] * mean_c(dnorm*xhat) ) <- the variance path

Note the shape asymmetry: dweight/dbias are (C,) and accumulate across all B*T
rows, while dinp is (B,T,C) and is per-element. Two different sum axes in one
op — that is exactly what will make this awkward on the GPU later (the weight
grads want atomics or a separate reduction pass).
*/
void layernorm_backward(float *dinp, float *dweight, float *dbias,
                        float *dout, float *inp, float *weight,
                        float *mean, float *rstd,
                        int B, int T, int C) {
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {

            float *x  = inp  + b * T * C + t * C;   // this token's forward input
            float *dy = dout + b * T * C + t * C;   // upstream gradient
            float *dx = dinp + b * T * C + t * C;   // where we write
            float m = mean[b * T + t];              // saved in forward
            float r = rstd[b * T + t];

            // pass 1: the two row-wide averages. Both need the whole row before
            // any element can be written, which is why this op is two passes.
            float dnorm_mean = 0.0f;        // mean_c(dnorm)
            float dnorm_xhat_mean = 0.0f;   // mean_c(dnorm * xhat)
            for (int c = 0; c < C; c++) {
                float xhat  = (x[c] - m) * r;
                float dnorm = dy[c] * weight[c];
                dnorm_mean      += dnorm;
                dnorm_xhat_mean += dnorm * xhat;
            }
            dnorm_mean      /= C;
            dnorm_xhat_mean /= C;

            // pass 2: write all three gradients
            for (int c = 0; c < C; c++) {
                float xhat  = (x[c] - m) * r;
                float dnorm = dy[c] * weight[c];

                dbias[c]   += dy[c];           // bias adds, so it gets dout whole
                dweight[c] += dy[c] * xhat;    // weight multiplies xhat, so it gets dout*xhat

                dx[c] += r * (dnorm - dnorm_mean - xhat * dnorm_xhat_mean);
            }
        }
    }
}

/*
Residual backward: the cheapest op in the file.

Forward:  out[i] = x[i] + y[i]

Addition scales nothing, so d(out)/dx = 1 and d(out)/dy = 1. The upstream
gradient is handed to BOTH inputs unchanged — no weights, no reduction, no
saved state needed from the forward pass.

This is exactly why residual connections make deep networks trainable: the
gradient reaches the early layers along a path that is never multiplied down.
*/
void residual_backward(float *dx, float *dy, float *dout, int N) {
    for (int i = 0; i < N; i++) {
        dx[i] += dout[i];
        dy[i] += dout[i];
    }
}

/*
GELU backward: elementwise like residual, but NOT a free pass-through.

Forward (tanh approximation, the one llm.c and HF GPT-2 use):
    u    = k * (x + 0.044715 x^3),   k = sqrt(2/pi)
    gelu = 0.5 * x * (1 + tanh(u))

Residual could copy the gradient straight through because its local derivative
is exactly 1. GELU's local derivative depends on x, so we MULTIPLY by it — and
that is why this op needs the saved forward input (mlp_fch) while residual
needed nothing.

Product rule on 0.5*x*(1 + tanh(u)):
    d/dx = 0.5*(1 + tanh(u))            <- derivative of the 0.5*x factor
         + 0.5*x * sech^2(u) * du/dx    <- derivative of the tanh factor
    du/dx = k * (1 + 3*0.044715*x^2)
    sech^2(u) = 1 - tanh^2(u)           <- identity, cheaper than calling coshf

Sanity checks on the result: for large positive x the derivative -> 1 (GELU is
the identity out there), for large negative x it -> 0 (saturated, gradient
dies), and at x=0 it is exactly 0.5.
*/
void gelu_backward(float *dinp, float *inp, float *dout, int N) {
    const float k = 0.7978845608f;   // sqrt(2/pi), same constant as the forward
    for (int i = 0; i < N; i++) {
        float x = inp[i];                              // saved forward input
        float u = k * (x + 0.044715f * x * x * x);
        float tanh_u = tanhf(u);
        float sech2  = 1.0f - tanh_u * tanh_u;         // sech^2 = 1 - tanh^2
        float du_dx  = k * (1.0f + 3.0f * 0.044715f * x * x);

        float local_grad = 0.5f * (1.0f + tanh_u) + 0.5f * x * sech2 * du_dx;

        dinp[i] += local_grad * dout[i];               // chain rule: local * upstream
    }
}

/*
Attention backward: the crucible. Walks attention_forward's three steps in
reverse, staying inside the same (b, h, t) loop nest and reusing its exact
index arithmetic.

Naming, matching the derivation:
    s_j = preatt[.., t, j]      the scores
    p_i = att[.., t, i]         the softmax weights (SAVED by forward)
    g_i = datt[.., t, i]        gradient w.r.t. the weights

Step 3 reversed:  atty[t][c] = sum_{t2<=t} p[t2] * v[t2][c]
    a bilinear form, so each operand's gradient is the other operand:
        datt[t2]    = sum_c dout[t][c] * v[t2][c]      contract over channels
        dv[t2][c]  += p[t2] * dout[t][c]               contract over queries

Step 2 reversed:  p = softmax(s)
    The Jacobian is NOT diagonal — dp_i/ds_j = p_i*(delta_ij - p_j) — because
    the normalizer S coavuples every output to every input. Contracting it with g:
        dpreatt[j] = p_j * ( g_j - sum_i p_i g_i )
    The subtracted term is E_p[g], the p-weighted average of the upstream
    gradient. Same structural shape as layernorm's mean_c(dnorm*xhat) term, and
    for the same reason: a row-wide normalizer couples the row.
    Consequence worth knowing: sum_j dpreatt[j] == 0 always, which is exactly
    what softmax's shift-invariance (softmax(s+c) == softmax(s)) demands.

Step 1 reversed:  s[t][t2] = scale * dot(q[t], k[t2])
        dq[t][c]   += scale * sum_{t2<=t}  dpreatt[t2] * k[t2][c]
        dk[t2][c]  += scale * sum_{t1>=t2} dpreatt[t1][t2] * q[t1][c]

WHY dq gets away with less accumulation than dk/dv: dq[t] sums only over
t2 <= t, all of which happen inside this single iteration of the t loop. But
dk[t2] and dv[t2] collect from every FUTURE query t1 >= t2, which land on
different iterations. Hence the += on dk/dv is load-bearing. On the GPU, a
thread-per-query kernel needs atomics for K and V but not for Q — same fact.

The causal mask is enforced structurally: every loop runs t2 <= t, exactly as
forward did, so the future entries of dpreatt/datt are never touched and stay
at the zero calloc gave them.
*/
void attention_backward(float *dqkv, float *dpreatt, float *datt,
                        float *dout, float *qkv, float *att,
                        int B, int T, int C, int NH) {
    int hs = C / NH;
    float scale = 1.0f / sqrtf((float)hs);

    for (int b = 0; b < B; b++) {
        for (int h = 0; h < NH; h++) {
            for (int t = 0; t < T; t++) {

                // Same pointers forward used, plus their gradient partners.
                float *att_row     = att     + b*NH*T*T + h*T*T + t*T;
                float *datt_row    = datt    + b*NH*T*T + h*T*T + t*T;
                float *dpreatt_row = dpreatt + b*NH*T*T + h*T*T + t*T;
                float *q  = qkv  + b*T*3*C + t*3*C + h*hs;
                float *dq = dqkv + b*T*3*C + t*3*C + h*hs;
                float *dy = dout + b*T*C   + t*C   + h*hs;  // d(atty) for this token/head

                // ---- step 3 reversed: dout -> datt and dv -------------------
                for (int t2 = 0; t2 <= t; t2++) {
                    float *v  = qkv  + b*T*3*C + t2*3*C + 2*C + h*hs;
                    float *dv = dqkv + b*T*3*C + t2*3*C + 2*C + h*hs;
                    for (int i = 0; i < hs; i++) {
                        datt_row[t2] += dy[i] * v[i];        // how much did weight t2 matter
                        dv[i]        += att_row[t2] * dy[i]; // how much did value t2 matter
                    }
                }

                // ---- step 2 reversed: softmax -------------------------------
                // E_p[g]: the p-weighted average of the upstream gradient.
                float expectation = 0.0f;
                for (int t2 = 0; t2 <= t; t2++)
                    expectation += att_row[t2] * datt_row[t2];

                // dpreatt[j] = p_j * (g_j - E_p[g]). Only relative merit counts:
                // softmax is zero-sum, so j gains only at the others' expense.
                for (int t2 = 0; t2 <= t; t2++)
                    dpreatt_row[t2] += att_row[t2] * (datt_row[t2] - expectation);

                // ---- step 1 reversed: the QK dot product ---------------------
                for (int t2 = 0; t2 <= t; t2++) {
                    float *k  = qkv  + b*T*3*C + t2*3*C + C + h*hs;
                    float *dk = dqkv + b*T*3*C + t2*3*C + C + h*hs;
                    float d = dpreatt_row[t2] * scale;
                    for (int i = 0; i < hs; i++) {
                        dq[i] += d * k[i];   // this query's own row: one pass
                        dk[i] += d * q[i];   // this key is hit by every future query
                    }
                }
            }
        }
    }
}

/*
Encoder backward: a gather in reverse is a scatter.

Forward was  encoded[b,t,c] = wte[ix][c] + wpe[t][c],  ix = inputs[b,t]
Both inputs are just ADDED, so — exactly like residual — each receives the
upstream gradient unchanged. There is no local derivative to multiply by.

What makes this op interesting is not the math, it is the WRITE PATTERN.
Forward, each (b,t) READ one wte row. Backward, each (b,t) WRITES to one wte
row — and many tokens in the batch share the same ID, so several (b,t) pairs
land on the same row and must sum. On the CPU that is just `+=` in a serial
loop. On the GPU those writes race, which is why llm.c's encoder_backward is
the textbook `atomicAdd` case. Same for wpe: every batch row b hits wpe[t],
so wpe[t] collects B contributions.

Note the asymmetry in sizes: dwte is (V,C) = 154 MB of which only the rows for
tokens actually seen in this batch are touched (at most B*T = 256 of 50257).
dwpe is (maxT,C) and only its first T rows are touched.
*/
void encoder_backward(float *dwte, float *dwpe, float *dout, int *inp,
                      int B, int T, int C) {
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float *dy = dout + b * T * C + t * C;   // upstream gradient row
            int ix = inp[b * T + t];                // which dictionary row was read
            float *dwte_row = dwte + (size_t)ix * C;
            float *dwpe_row = dwpe + (size_t)t * C;

            for (int c = 0; c < C; c++) {
                dwte_row[c] += dy[c];   // scatter-add: races on the GPU
                dwpe_row[c] += dy[c];
            }
        }
    }
}

/*
AdamW: the step. One flat loop over every parameter in the model.

Note what is NOT in the signature: no B, no T, no C, no shapes at all. The
optimizer has no idea this buffer contains an embedding table and an attention
projection — it is just n floats with n slopes. That is the payoff for putting
every weight in one contiguous block: adding a second transformer block later
changes p.total and nothing else here. On the GPU this becomes ONE kernel over
124M contiguous floats, perfectly coalesced, zero branching.

    m = beta1*m + (1-beta1)*g          momentum: smoothed gradient
    v = beta2*v + (1-beta2)*g*g        smoothed SQUARED gradient
    p -= lr * ( m_hat/(sqrt(v_hat)+eps) + wd*p )

The idea worth holding: v is a PER-PARAMETER learning rate. Dividing by
sqrt(v_hat) means a weight whose gradient is consistently large takes smaller
steps, while a rarely-touched one (an embedding row for a rare token) takes
bigger ones. Gradient magnitudes vary enormously across a 124M-param model,
which is why plain SGD struggles here and Adam does not.

Bias correction: m and v start at zero, so early estimates are biased toward
zero. Dividing by (1 - beta^t) undoes exactly that. t starts at 1, not 0 —
at t=0 the denominator is 0.

The "W": weight decay is applied to p directly, OUTSIDE the adaptive
denominator. Plain Adam+L2 folds wd*p into the gradient instead, where it then
gets divided by sqrt(v_hat) — so weights with big gradients receive LESS decay,
which is backwards. Decoupling it is the whole contribution of the AdamW paper.

Known divergence from best practice (deliberate, matches llm.c): this decays
EVERY parameter, including biases and LayerNorm gains. nanoGPT excludes those,
since shrinking an LN gain fights the normalization rather than regularizing.
Excluding them would require the optimizer to know offset ranges, giving up the
structure-blindness above. Not worth it at TinyShakespeare scale.
*/
void adamw_update(float *params, float *grads, float *m, float *v,
                  size_t n, int t,
                  float lr, float beta1, float beta2, float eps, float wd) {
    // Bias-correction factors depend only on t, so hoist them out of the loop.
    float bc1 = 1.0f - powf(beta1, (float)t);
    float bc2 = 1.0f - powf(beta2, (float)t);

    for (size_t i = 0; i < n; i++) {
        float g = grads[i];

        m[i] = beta1 * m[i] + (1.0f - beta1) * g;
        v[i] = beta2 * v[i] + (1.0f - beta2) * g * g;

        float m_hat = m[i] / bc1;
        float v_hat = v[i] / bc2;

        params[i] -= lr * (m_hat / (sqrtf(v_hat) + eps) + wd * params[i]);
    }
}

/* ===================================================================== *
 * THE BAYESIAN LAYER — reparameterized weight sampling
 *
 * This is the entire difference between this file and train_gpt2.c, and it
 * is deliberately a WRAPPER: not one line of any forward or backward op
 * changes. The reason is worth stating plainly, because it is the whole
 * design.
 *
 * gpt2_forward takes `float *params` — a flat buffer of numbers. It does not
 * know or care where those numbers came from. So if we materialize a sampled
 * weight buffer w and hand THAT to the existing forward, the model runs
 * unchanged, and gpt2_backward hands back dL/dw for free. All that is left is
 * to push dL/dw one step further back, from w to the things that actually
 * own the memory: mu and rho.
 *
 *   deterministic:  params -----------------------> loss
 *   bayesian:       mu, rho --[sample w]--> w ----> loss
 *                                 ^
 *                                 eps (a constant, drawn once per forward)
 *
 * WHY NOT JUST CALL A GAUSSIAN RNG
 * --------------------------------
 * Writing w = gaussian(mu, sigma) puts the randomness BETWEEN the parameter
 * and the loss. A sampler is not a differentiable function of its arguments —
 * there is no derivative path from loss back to mu, and the whole thing is
 * untrainable. The reparameterization trick moves the randomness OFF the
 * parameter and into an input:
 *
 *     w_i = mu_i + sigma_i * eps_i ,    eps_i ~ N(0,1)
 *
 * w still has mean mu and variance sigma^2 — the distribution is identical.
 * But eps is now just a number that arrived from outside, like a data value,
 * and w is an ordinary arithmetic expression in mu and sigma. Differentiable.
 *
 * WHY rho AND NOT sigma
 * ---------------------
 * sigma must stay positive. An optimizer stepping an unconstrained sigma will
 * eventually push it below zero, and then sqrt/log of it is NaN and the run is
 * dead. So the stored parameter is rho, unconstrained, and
 *
 *     sigma = softplus(rho) = log(1 + e^rho)     always > 0
 *
 * which costs exactly one extra factor in the chain rule, since
 * d(softplus)/d(rho) = sigmoid(rho).
 *
 * THE GRADIENTS
 * -------------
 * With g_i = dL/dw_i (exactly what matmul_backward already writes into
 * `grads`), the local derivatives are trivial:
 *
 *     dw/dmu    = 1
 *     dw/dsigma = eps         and   dsigma/drho = sigmoid(rho)
 *
 *     dL/dmu_i  = g_i
 *     dL/drho_i = g_i * eps_i * sigmoid(rho_i)
 *
 * Note what this is NOT: no reduction, no cross-element coupling, no shape
 * change. Element i of the gradient depends only on element i. This is a pure
 * elementwise pass over p.total floats — one trivially coalesced CUDA kernel
 * later, and the cheapest thing in the entire model.
 *
 * THE TWO COSTS (both come due later, neither is fixed here)
 * ---------------------------------------------------------
 * 1. RNG volume. One normal per weight per forward. For qkv alone at C=768
 *    that is 768*2304 = 1.77M draws every single step.
 * 2. Gradient variance. ONE draw of w is shared by all B*T tokens in the
 *    batch, so every token's gradient is corrupted by the SAME eps. The
 *    errors are perfectly correlated and do not average out — the minibatch
 *    mean loses its usual 1/sqrt(B*T) variance reduction. This is the thing
 *    the local reparameterization trick fixes, and the reason the mean/
 *    variance GEMM split exists at all.
 * ===================================================================== */

// Standard normal, Box-Muller. Each call to the transform yields TWO
// independent normals, so cache the second rather than throwing it away.
static float randn(void) {
    static int have_spare = 0;
    static float spare;
    if (have_spare) { have_spare = 0; return spare; }

    // Map to the OPEN interval (0,1): rand() can legitimately return 0, and
    // logf(0) is -inf, which would poison the whole weight buffer.
    float u1 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 2.0f);
    float u2 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 2.0f);

    float r     = sqrtf(-2.0f * logf(u1));
    float theta = 6.28318530717958647692f * u2;

    spare = r * sinf(theta);
    have_spare = 1;
    return r * cosf(theta);
}

// softplus(x) = log(1 + e^x), guarded: expf overflows above ~88, and for large
// x the function is indistinguishable from x itself anyway.
static float softplus(float x) {
    return x > 20.0f ? x : log1pf(expf(x));
}

// d(softplus)/dx. No overflow risk: expf(-x) saturating to +inf yields 0,
// which is the correct limit.
static float sigmoidf_(float x) {
    return 1.0f / (1.0f + expf(-x));
}

/*
Build the sampled weight buffer from (mu, rho) and an ALREADY-DRAWN eps.

Split out from bayes_sample on purpose: the gradient checker has to perturb mu
or rho and recompute w while holding eps FIXED. Redrawing eps there would
compare two different networks and the check would be meaningless.
*/
void bayes_materialize(float *w, const float *mu, const float *rho,
                       const float *eps, size_t n) {
    for (size_t i = 0; i < n; i++)
        w[i] = mu[i] + softplus(rho[i]) * eps[i];
}

/*
Draw a fresh Monte Carlo sample of the whole parameter vector: one realization
W^(k) ~ q(W). Fills `eps` (which MUST be kept until backward) and `w`.
*/
void bayes_sample(float *w, float *eps, const float *mu, const float *rho,
                  size_t n) {
    for (size_t i = 0; i < n; i++) eps[i] = randn();
    bayes_materialize(w, mu, rho, eps, n);
}

/*
Push the gradient from the sampled weights back onto the real parameters.
`dw` is the `grads` buffer that gpt2_backward just filled.

Accumulates with += to match the convention every other backward op in this
file uses — so the KL gradient can be added on top in the next module without
either term having to know about the other.
*/
void bayes_backward(float *dmu, float *drho, const float *dw,
                    const float *eps, const float *rho, size_t n) {
    for (size_t i = 0; i < n; i++) {
        dmu[i]  += dw[i];
        drho[i] += dw[i] * eps[i] * sigmoidf_(rho[i]);
    }
}

/* ===================================================================== *
 * THE KL TERM — the prior, stated properly
 *
 * Training so far minimizes the data term alone, and the previous run showed
 * exactly what that does: sigma shrinks monotonically, because noise on the
 * weights can only ever hurt the fit. Left alone the model deletes its own
 * uncertainty and becomes an ordinary net wearing a Bayesian costume.
 *
 * The full objective is the negative ELBO:
 *
 *     L = E_q[ NLL(data | W) ]  +  KL( q(W) || p(W) )
 *         \_______ fit ______/     \____ stay near the prior ____/
 *
 * The first term is everything the file already does (with W sampled, which is
 * what makes it an expectation estimated by ONE Monte Carlo draw). The second
 * is new, and it is remarkable how little it costs: both q and p are diagonal
 * Gaussians, so the KL has a CLOSED FORM. No sampling, no forward pass, no
 * activations, no data. It is a pure function of (mu, rho).
 *
 * For q_i = N(mu_i, sigma_i^2) and prior p = N(0, sp^2), per parameter:
 *
 *     KL_i = log(sp/sigma_i) + (sigma_i^2 + mu_i^2)/(2*sp^2) - 1/2
 *
 * and the total is the plain sum over all n parameters. Three sanity checks
 * on that formula, all worth doing before trusting it:
 *   - mu=0, sigma=sp  ->  log(1) + (sp^2)/(2sp^2) - 1/2 = 0. Identical
 *     distributions cost nothing. Correct.
 *   - sigma -> 0  ->  log(sp/sigma) -> +inf. Infinite cost to be CERTAIN.
 *     This is the term that forbids the collapse.
 *   - mu -> large ->  mu^2 term grows quadratically. This is ordinary L2
 *     weight decay, and it is where weight decay comes from in the first
 *     place: a Gaussian prior on the weights. Which is exactly why wd=0 in
 *     the optimizer — running both would apply the same prior twice.
 *
 * THE GRADIENTS (elementwise, 1:1, no coupling — same shape as everything
 * else in the Bayesian layer)
 *
 *     dKL/dmu_i    = mu_i / sp^2
 *     dKL/dsigma_i = sigma_i/sp^2 - 1/sigma_i
 *     dKL/drho_i   = dKL/dsigma_i * sigmoid(rho_i)
 *
 * The sigma gradient is the interesting one. It is zero exactly at
 * sigma = sp, positive above (shrink toward the prior) and — because of the
 * -1/sigma term — violently negative below (grow, urgently). The data term
 * pushes sigma down, this pushes it up, and training finds where they balance.
 * A weight the data constrains tightly ends with small sigma; one the data
 * says nothing about relaxes back to sp. That equilibrium IS the posterior,
 * and it is the entire point of the model.
 *
 * SCALING — the part that is easy to get wrong
 *
 * The KL is a property of the WHOLE MODEL and is paid ONCE per pass over the
 * dataset. The NLL here is a mean over one minibatch. Mixing the two without
 * thought is the classic failure. The standard fix is to spread the KL evenly
 * across minibatches:
 *
 *     loss_batch = NLL_batch + KL / num_batches
 *
 * Get this too large and the posterior collapses onto the prior and the model
 * ignores the data entirely; too small and the KL is decorative and you are
 * back to a deterministic net. So kl_weight is an explicit argument here, not
 * a buried constant.
 * ===================================================================== */

/*
Total KL, summed over all n parameters. Returns the raw value in nats — the
caller applies kl_weight — so that what gets printed stays interpretable.

Returns DOUBLE, and that is not fussiness. This is the one genuinely
ill-conditioned sum in the whole file: every term is the same sign, so nothing
cancels and the total grows with n — ~4e3 at this toy size, ~1e8 at GPT-2
scale. In fp32 a total of 4e3 has an ulp of 4.9e-4, while perturbing ONE
parameter moves the sum by ~3e-4. The change is smaller than the representable
gap and rounds to exactly zero: the gradient becomes unmeasurable, and worse,
individual small terms stop contributing to the total at all. This is the
"large accumulator, small increments" failure, and the accumulator is where it
has to be fixed.
*/
double kl_forward(const float *mu, const float *rho, size_t n, float sigma_prior) {
    float sp2 = sigma_prior * sigma_prior;
    double kl = 0.0;
    for (size_t i = 0; i < n; i++) {
        // Each TERM is evaluated in double too, not just the running total.
        // Fixing only the accumulator leaves every term quantized to ~5e-7,
        // which is coarse enough to visibly corrupt a finite-difference probe
        // of a single parameter (~2e-5 of signal, i.e. ~40 ulps).
        double sigma = (double)softplus(rho[i]);
        double m = (double)mu[i];
        kl += log((double)sigma_prior / sigma)
            + (sigma * sigma + m * m) / (2.0 * (double)sp2)
            - 0.5;
    }
    return kl;
}

/*
Accumulate the KL gradient onto dmu/drho. Note the += : this is called AFTER
bayes_backward has already deposited the data-term gradient, and neither
function needs to know the other exists.

`weight` is the kl_weight discussed above, applied to the gradient so it stays
consistent with the loss the caller reports.
*/
void kl_backward(float *dmu, float *drho, const float *mu, const float *rho,
                 size_t n, float sigma_prior, float weight) {
    float sp2 = sigma_prior * sigma_prior;
    for (size_t i = 0; i < n; i++) {
        float sigma = softplus(rho[i]);

        dmu[i]  += weight * (mu[i] / sp2);

        // Zero at sigma == sigma_prior. The -1/sigma branch is what makes
        // certainty expensive and stops sigma from collapsing to 0.
        float dsigma = sigma / sp2 - 1.0f / sigma;
        drho[i] += weight * dsigma * sigmoidf_(rho[i]);
    }
}

/* ===================================================================== *
 * Model config + buffer layout
 *
 * The offsets used to be computed inline in main. They now live in these
 * two functions so that main, gpt2_forward, gpt2_backward and the gradient
 * checker all slice the SAME layout from one source of truth. Adding a
 * tensor = one new line here, and every caller sees it.
 * ===================================================================== */

typedef struct {
    int B, T, C, V, NH, maxT;
} Config;

typedef struct {
    size_t wte, wpe, lng, lnb, w1, b1, w2, b2, qw, qb, pw, pb,
           ln2g, ln2b, flng, flnb, hw, hb, total;
} ParamOff;

typedef struct {
    size_t enc, lno, lnm, lnr, fch, gelu, mlpo, res1, qkv, pat, att, atty,
           apr, resa, ln2o, ln2m, ln2r, flo, flm, flr, log, prb, los, total;
} ActOff;

ParamOff param_offsets(Config cfg) {
    int C = cfg.C, V = cfg.V, maxT = cfg.maxT;
    ParamOff p;
    p.wte  = 0;                                 // token embeddings (V*C)
    p.wpe  = p.wte + (size_t)V * C;             // position embeds  (maxT*C)
    p.lng  = p.wpe + (size_t)maxT * C;          // ln1 scale (C)
    p.lnb  = p.lng + C;                         // ln1 shift (C)
    p.w1   = p.lnb + C;                         // MLP up   (C*4C)
    p.b1   = p.w1 + (size_t)C * 4 * C;          //          (4C)
    p.w2   = p.b1 + 4 * C;                      // MLP down (4C*C)
    p.b2   = p.w2 + (size_t)4 * C * C;          //          (C)
    p.qw   = p.b2 + C;                          // qkv weight (C*3C)
    p.qb   = p.qw + (size_t)C * 3 * C;          //            (3C)
    p.pw   = p.qb + 3 * C;                      // attn proj  (C*C)
    p.pb   = p.pw + (size_t)C * C;              //            (C)
    p.ln2g = p.pb + C;                          // ln2 scale (C)
    p.ln2b = p.ln2g + C;                        // ln2 shift (C)
    p.flng = p.ln2b + C;                        // final ln scale (C)
    p.flnb = p.flng + C;                        // final ln shift (C)
    p.hw   = p.flnb + C;                        // lm_head weight (C*V)
    p.hb   = p.hw + (size_t)C * V;              // lm_head bias   (V)
    p.total = p.hb + V;
    return p;
}

ActOff act_offsets(Config cfg) {
    int B = cfg.B, T = cfg.T, C = cfg.C, V = cfg.V, NH = cfg.NH;
    ActOff a;
    a.enc  = 0;                                   // encoded  (B,T,C)
    a.lno  = a.enc + (size_t)B * T * C;           // ln1_out  (B,T,C)
    a.lnm  = a.lno + (size_t)B * T * C;           // ln1_mean (B,T)
    a.lnr  = a.lnm + (size_t)B * T;               // ln1_rstd (B,T)
    a.fch  = a.lnr + (size_t)B * T;               // mlp_fch  (B,T,4C)
    a.gelu = a.fch + (size_t)B * T * 4 * C;       // mlp_gelu (B,T,4C)
    a.mlpo = a.gelu + (size_t)B * T * 4 * C;      // mlp_out  (B,T,C)
    a.res1 = a.mlpo + (size_t)B * T * C;          // stream after MLP add
    a.qkv  = a.res1 + (size_t)B * T * C;          // qkv      (B,T,3C)
    a.pat  = a.qkv + (size_t)B * T * 3 * C;       // preatt   (B,NH,T,T)
    a.att  = a.pat + (size_t)B * NH * T * T;      // att      (B,NH,T,T)
    a.atty = a.att + (size_t)B * NH * T * T;      // V-mix    (B,T,C)
    a.apr  = a.atty + (size_t)B * T * C;          // attproj  (B,T,C)
    a.resa = a.apr + (size_t)B * T * C;           // stream after attn add
    a.ln2o = a.resa + (size_t)B * T * C;          // ln2_out  (B,T,C)
    a.ln2m = a.ln2o + (size_t)B * T * C;          // ln2_mean (B,T)
    a.ln2r = a.ln2m + (size_t)B * T;              // ln2_rstd (B,T)
    a.flo  = a.ln2r + (size_t)B * T;              // fln_out  (B,T,C)
    a.flm  = a.flo + (size_t)B * T * C;           // fln_mean (B,T)
    a.flr  = a.flm + (size_t)B * T;               // fln_rstd (B,T)
    a.log  = a.flr + (size_t)B * T;               // logits   (B,T,V)
    a.prb  = a.log + (size_t)B * T * V;           // probs    (B,T,V)
    a.los  = a.prb + (size_t)B * T * V;           // losses   (B,T)
    a.total = a.los + (size_t)B * T;
    return a;
}

/*
The whole forward pass, in one callable function. Returns the mean loss.

This used to be inlined in main. Pulling it out is what makes the gradient
checker possible: the checker has to run forward hundreds of times with a
slightly perturbed weight, which you cannot do with straight-line code.
*/
float gpt2_forward(float *acts, float *params, int *inputs, int *targets,
                   Config cfg) {
    int B = cfg.B, T = cfg.T, C = cfg.C, V = cfg.V, NH = cfg.NH;
    ParamOff p = param_offsets(cfg);
    ActOff   a = act_offsets(cfg);

    // TALKING half: LN1 -> attention -> +residual
    encoder_forward(acts + a.enc, inputs, params + p.wte, params + p.wpe, B, T, C);
    layernorm_forward(acts + a.lno, acts + a.lnm, acts + a.lnr,
                      acts + a.enc, params + p.lng, params + p.lnb, B, T, C);
    matmul_forward(acts + a.qkv, acts + a.lno, params + p.qw, params + p.qb,
                   B, T, C, 3 * C);
    attention_forward(acts + a.atty, acts + a.pat, acts + a.att, acts + a.qkv,
                      B, T, C, NH);
    matmul_forward(acts + a.apr, acts + a.atty, params + p.pw, params + p.pb,
                   B, T, C, C);
    residual_forward(acts + a.resa, acts + a.enc, acts + a.apr, B * T * C);

    // THINKING half: LN2 -> MLP -> +residual
    layernorm_forward(acts + a.ln2o, acts + a.ln2m, acts + a.ln2r,
                      acts + a.resa, params + p.ln2g, params + p.ln2b, B, T, C);
    matmul_forward(acts + a.fch, acts + a.ln2o, params + p.w1, params + p.b1,
                   B, T, C, 4 * C);
    gelu_forward(acts + a.gelu, acts + a.fch, B * T * 4 * C);
    matmul_forward(acts + a.mlpo, acts + a.gelu, params + p.w2, params + p.b2,
                   B, T, 4 * C, C);
    residual_forward(acts + a.res1, acts + a.resa, acts + a.mlpo, B * T * C);

    // HEAD: final LN -> logits -> loss
    layernorm_forward(acts + a.flo, acts + a.flm, acts + a.flr,
                      acts + a.res1, params + p.flng, params + p.flnb, B, T, C);
    matmul_forward(acts + a.log, acts + a.flo, params + p.hw, params + p.hb,
                   B, T, C, V);
    softmax_crossentropy(acts + a.los, acts + a.prb, acts + a.log, targets, B, T, V);

    float mean_loss = 0.0f;
    for (int i = 0; i < B * T; i++) mean_loss += acts[a.los + i];
    return mean_loss / (B * T);
}

/*
The backward pass so far. Walks the forward exactly in reverse.

INCOMPLETE by design: the chain currently stops at dmlp_fch. Everything from
the MLP up-projection back through attention to the encoder is still to be
written, so the gradients for w1/b1, ln2, qkv, proj, ln1, wte and wpe are all
still zero. The gradient checker below only tests the ones that are done.

Note it does NOT zero the grad buffers — the caller does that, because
gradients accumulate across micro-batches in real training.
*/
void gpt2_backward(float *grads, float *grads_acts, float *acts, float *params,
                   int *inputs, int *targets, Config cfg) {
    int B = cfg.B, T = cfg.T, C = cfg.C, V = cfg.V, NH = cfg.NH;
    ParamOff p = param_offsets(cfg);
    ActOff   a = act_offsets(cfg);

    // loss -> logits. No incoming gradient: d(loss)/d(loss) is 1 by definition.
    crossentropy_softmax_backward(grads_acts + a.log, acts + a.prb, targets, B, T, V);

    // logits matmul -> head weights + fln_out
    matmul_backward(grads_acts + a.flo, grads + p.hw, grads + p.hb,
                    acts + a.flo, params + p.hw, grads_acts + a.log,
                    B, T, C, V);

    // final layernorm -> fln scale/shift + res1
    layernorm_backward(grads_acts + a.res1, grads + p.flng, grads + p.flnb,
                       grads_acts + a.flo, acts + a.res1, params + p.flng,
                       acts + a.flm, acts + a.flr, B, T, C);

    // residual fork: res1 = res_att + mlp_out
    residual_backward(grads_acts + a.resa, grads_acts + a.mlpo,
                      grads_acts + a.res1, B * T * C);

    // MLP down-projection (4C -> C)
    matmul_backward(grads_acts + a.gelu, grads + p.w2, grads + p.b2,
                    acts + a.gelu, params + p.w2, grads_acts + a.mlpo,
                    B, T, 4 * C, C);

    // GELU
    gelu_backward(grads_acts + a.fch, acts + a.fch, grads_acts + a.gelu,
                  B * T * 4 * C);

    // MLP up-projection (C -> 4C). Lands on dln2_out.
    matmul_backward(grads_acts + a.ln2o, grads + p.w1, grads + p.b1,
                    acts + a.ln2o, params + p.w1, grads_acts + a.fch,
                    B, T, C, 4 * C);

    // ln2. NOTE the +=: dres_att already holds the contribution the residual
    // fork gave it. The MLP branch's contribution is ADDED here, not assigned —
    // res_att fed both the MLP and the skip, so it collects from both.
    layernorm_backward(grads_acts + a.resa, grads + p.ln2g, grads + p.ln2b,
                       grads_acts + a.ln2o, acts + a.resa, params + p.ln2g,
                       acts + a.ln2m, acts + a.ln2r, B, T, C);

    // The attention-side residual fork: res_att = encoded + attproj
    residual_backward(grads_acts + a.enc, grads_acts + a.apr,
                      grads_acts + a.resa, B * T * C);

    // Attention output projection (C -> C). Lands on datty.
    matmul_backward(grads_acts + a.atty, grads + p.pw, grads + p.pb,
                    acts + a.atty, params + p.pw, grads_acts + a.apr,
                    B, T, C, C);

    // Attention itself. Writes dQ, dK, dV interleaved into the one dqkv buffer.
    attention_backward(grads_acts + a.qkv, grads_acts + a.pat, grads_acts + a.att,
                       grads_acts + a.atty, acts + a.qkv, acts + a.att,
                       B, T, C, NH);

    // QKV projection (C -> 3C). Lands on dln_out.
    matmul_backward(grads_acts + a.lno, grads + p.qw, grads + p.qb,
                    acts + a.lno, params + p.qw, grads_acts + a.qkv,
                    B, T, C, 3 * C);

    // ln1. Second and final contribution to dencoded (the first came from the
    // attention-side residual fork above). Again += is what makes that work.
    layernorm_backward(grads_acts + a.enc, grads + p.lng, grads + p.lnb,
                       grads_acts + a.lno, acts + a.enc, params + p.lng,
                       acts + a.lnm, acts + a.lnr, B, T, C);

    // Encoder: scatter dencoded back into the embedding tables. End of chain.
    encoder_backward(grads + p.wte, grads + p.wpe, grads_acts + a.enc,
                     inputs, B, T, C);
}

/*
Finite-difference gradient check.

The idea, and it is the only honest test of a hand-written backward pass:
the gradient of the loss w.r.t. one weight is BY DEFINITION the slope of the
loss as you wiggle that weight. So wiggle it and measure.

    numeric = ( loss(w + h) - loss(w - h) ) / (2h)

The central difference (both directions) is used rather than the one-sided
version because its error is O(h^2) instead of O(h) — it buys ~4 extra digits
for one extra forward pass.

Run on a TINY config (V=13, C=8, ...) rather than the real one. Same code
paths, same functions, but two forward passes cost microseconds instead of
half a minute — so we can afford to probe many weights. A bug in the math is
a bug at any size; only performance cares how big V is.
*/
static float rand_weight(void) {
    return ((float)rand() / RAND_MAX - 0.5f) * 0.2f;
}

/*
Isolated check for encoder_backward.

The encoder sits at the very BOTTOM of the chain, so its gradient cannot be
tested through the full model until attention and ln1 backward exist. But an
op does not need the chain to be testable: invent a scalar loss directly from
its output and finite-difference against that.

    loss = sum_i encoded[i] * coef[i]        (coef = fixed random numbers)

Then d(loss)/d(encoded[i]) is exactly coef[i], so feeding `coef` in as the
upstream gradient is the same thing the real chain would eventually do. This
trick works for ANY op — it is how you test a backward before its turn comes.

The inputs deliberately repeat a token ID so the scatter-accumulate path (two
positions writing the same wte row) is actually exercised. Because the encoder
is linear, agreement here should be near-exact, not merely within tolerance.
*/
int encoder_backward_check(void) {
    int B = 2, T = 3, C = 4, V = 7, maxT = 3;
    int n_enc = B * T * C;

    float wte[7 * 4], wpe[3 * 4], coef[2 * 3 * 4];
    float encoded[2 * 3 * 4];
    float dwte[7 * 4] = {0}, dwpe[3 * 4] = {0};
    // token 5 appears three times -> three gradient contributions to wte row 5
    int inputs[6] = { 5, 2, 5, 0, 5, 6 };

    srand(7);
    for (int i = 0; i < V * C; i++)    wte[i] = rand_weight();
    for (int i = 0; i < maxT * C; i++) wpe[i] = rand_weight();
    for (int i = 0; i < n_enc; i++)    coef[i] = rand_weight();

    // analytic: the upstream gradient IS coef, by construction of the loss
    encoder_backward(dwte, dwpe, coef, inputs, B, T, C);

    const float h = 1e-2f;
    float worst = 0.0f;
    const char *worst_name = "";
    size_t worst_idx = 0;

    // probe both tables the same way
    struct { const char *name; float *w; float *dw; int n; } tabs[] = {
        { "wte", wte, dwte, V * C },
        { "wpe", wpe, dwpe, maxT * C },
    };

    for (int ti = 0; ti < 2; ti++) {
        for (int j = 0; j < tabs[ti].n; j++) {
            float saved = tabs[ti].w[j];

            tabs[ti].w[j] = saved + h;
            encoder_forward(encoded, inputs, wte, wpe, B, T, C);
            float lp = 0.0f;
            for (int i = 0; i < n_enc; i++) lp += encoded[i] * coef[i];

            tabs[ti].w[j] = saved - h;
            encoder_forward(encoded, inputs, wte, wpe, B, T, C);
            float lm = 0.0f;
            for (int i = 0; i < n_enc; i++) lm += encoded[i] * coef[i];

            tabs[ti].w[j] = saved;

            float numeric  = (lp - lm) / (2.0f * h);
            float analytic = tabs[ti].dw[j];
            float denom = fabsf(numeric) > fabsf(analytic) ? fabsf(numeric) : fabsf(analytic);
            float rel = denom > 1e-6f ? fabsf(numeric - analytic) / denom : 0.0f;
            if (rel > worst) { worst = rel; worst_name = tabs[ti].name; worst_idx = j; }
        }
    }

    // wte row 5 was written by three tokens; its gradient must be their sum.
    float expect_row5 = 0.0f;
    for (int i = 0; i < 6; i++)
        if (inputs[i] == 5) expect_row5 += coef[i * C + 0];   // channel 0 only
    printf("  encoder scatter: dwte[row 5][0] = %+.6f, sum of the 3 contributions = %+.6f\n",
           dwte[5 * C + 0], expect_row5);

    int ok = worst < 1e-3f;
    printf("  %-26s %s  worst rel err %.2e at %s[%zu]  (all %d weights probed)\n",
           "encoder (wte + wpe)", ok ? "PASS " : "FAIL ", worst, worst_name, worst_idx,
           V * C + maxT * C);
    return ok ? 0 : 1;
}

int gradient_check(void) {
    // Deliberately tiny, and deliberately awkward: T not a power of two, C not
    // divisible by anything convenient except NH, V small enough that softmax
    // is well-conditioned.
    Config cfg = { .B = 2, .T = 5, .C = 8, .V = 13, .NH = 2, .maxT = 5 };
    int B = cfg.B, T = cfg.T, V = cfg.V;

    ParamOff p = param_offsets(cfg);
    ActOff   a = act_offsets(cfg);

    float *params     = malloc(p.total * sizeof(float));
    float *acts       = malloc(a.total * sizeof(float));
    float *grads      = calloc(p.total, sizeof(float));
    float *grads_acts = calloc(a.total, sizeof(float));
    int *inputs  = malloc(B * T * sizeof(int));
    int *targets = malloc(B * T * sizeof(int));
    if (!params || !acts || !grads || !grads_acts || !inputs || !targets) {
        printf("gradient_check: out of memory\n");
        return 1;
    }

    srand(1337);
    for (size_t i = 0; i < p.total; i++) params[i] = rand_weight();
    // layernorm scales start at 1 like the real model, else rstd is meaningless
    for (int c = 0; c < cfg.C; c++) {
        params[p.lng  + c] = 1.0f;  params[p.lnb  + c] = 0.0f;
        params[p.ln2g + c] = 1.0f;  params[p.ln2b + c] = 0.0f;
        params[p.flng + c] = 1.0f;  params[p.flnb + c] = 0.0f;
    }
    for (int i = 0; i < B * T; i++) {
        inputs[i]  = rand() % V;
        targets[i] = rand() % V;
    }

    // analytic gradients: one forward, one backward
    float loss0 = gpt2_forward(acts, params, inputs, targets, cfg);
    gpt2_backward(grads, grads_acts, acts, params, inputs, targets, cfg);

    // The chain is complete, so every parameter is probed. Listed in backward
    // order — the order a bug would propagate through them.
    struct { const char *name; size_t off; size_t n; } probes[] = {
        { "head_w (lm head weight)", p.hw,   (size_t)cfg.C * V },
        { "head_b (lm head bias)",   p.hb,   (size_t)V },
        { "fln_g  (final ln scale)", p.flng, (size_t)cfg.C },
        { "fln_b  (final ln shift)", p.flnb, (size_t)cfg.C },
        { "mlp_w2 (MLP down W)",     p.w2,   (size_t)4 * cfg.C * cfg.C },
        { "mlp_b2 (MLP down bias)",  p.b2,   (size_t)cfg.C },
        { "mlp_w1 (MLP up W)",       p.w1,   (size_t)cfg.C * 4 * cfg.C },
        { "mlp_b1 (MLP up bias)",    p.b1,   (size_t)4 * cfg.C },
        { "ln2_g  (ln2 scale)",      p.ln2g, (size_t)cfg.C },
        { "ln2_b  (ln2 shift)",      p.ln2b, (size_t)cfg.C },
        { "proj_w (attn out W)",     p.pw,   (size_t)cfg.C * cfg.C },
        { "proj_b (attn out bias)",  p.pb,   (size_t)cfg.C },
        { "qkv_w  (QKV weight)",     p.qw,   (size_t)cfg.C * 3 * cfg.C },
        { "qkv_b  (QKV bias)",       p.qb,   (size_t)3 * cfg.C },
        { "ln_g   (ln1 scale)",      p.lng,  (size_t)cfg.C },
        { "ln_b   (ln1 shift)",      p.lnb,  (size_t)cfg.C },
        { "wte    (token embed)",    p.wte,  (size_t)V * cfg.C },
        { "wpe    (pos embed)",      p.wpe,  (size_t)cfg.maxT * cfg.C },
    };
    int num_probes = sizeof(probes) / sizeof(probes[0]);

    // Choosing h is the whole game in fp32. Two errors fight each other:
    //   truncation error ~ O(h^2)     -> wants h small
    //   cancellation error ~ eps/h    -> wants h large
    // The loss is ~2.5 and fp32 resolution there is ~1e-7, so with h=1e-3 and a
    // gradient of 5e-3 the measured difference is only 1e-5 — about 100 ulps,
    // i.e. a 1% noise floor. h=1e-2 buys 10x signal for negligible truncation
    // error, since the loss is smooth on that scale.
    const float h = 1e-2f;
    const float tol = 1e-2f;    // relative

    // The noise floor, derived rather than guessed.
    //
    // The forward pass is fp32, so the returned loss carries a few ulps of
    // accumulated rounding. A central difference divides that error by 2h, so
    // the NUMERIC gradient has an irreducible absolute noise of
    //
    //     noise = (a few ulps of loss) / (2h)
    //
    // A purely relative tolerance is the wrong test near that floor: it demands
    // 1% of a quantity that is itself mostly noise. So a probe passes if it
    // agrees relatively to `tol` OR absolutely to within the noise floor.
    // This is why the failures were all ~3e-6..2e-5 absolute regardless of
    // gradient size — constant ABSOLUTE error is measurement noise; a real math
    // error shows up as constant RELATIVE error.
    float loss_ulp = nextafterf(loss0, 1e30f) - loss0;   // ~2.4e-7 for loss ~2.5
    float noise = 4.0f * loss_ulp / (2.0f * h);
    // Below this the loss does not move at all and the difference returns an
    // exact zero — nothing can be concluded, so those probes are skipped and
    // counted rather than silently passed.
    const float resolution = 1e-6f;
    int failures = 0;

    printf("\n--- gradient check (B=%d T=%d C=%d V=%d NH=%d, central diff h=%.0e) ---\n",
           cfg.B, cfg.T, cfg.C, cfg.V, cfg.NH, h);
    printf("    loss %.4f, 1 ulp = %.2e -> numeric-gradient noise floor %.2e\n",
           loss0, loss_ulp, noise);

    for (int pi = 0; pi < num_probes; pi++) {
        float worst_rel = 0.0f;
        size_t worst_i = 0;
        float worst_ana = 0.0f, worst_num = 0.0f;
        int checked = 0, skipped = 0, violations = 0;

        // sample up to 8 weights per tensor, spread across it
        size_t stride = probes[pi].n / 8 ? probes[pi].n / 8 : 1;
        for (size_t k = 0; k < probes[pi].n; k += stride) {
            size_t idx = probes[pi].off + k;
            float saved = params[idx];

            params[idx] = saved + h;
            float loss_plus = gpt2_forward(acts, params, inputs, targets, cfg);
            params[idx] = saved - h;
            float loss_minus = gpt2_forward(acts, params, inputs, targets, cfg);
            params[idx] = saved;

            float numeric  = (loss_plus - loss_minus) / (2.0f * h);
            float analytic = grads[idx];

            float denom = fabsf(numeric) > fabsf(analytic) ? fabsf(numeric) : fabsf(analytic);
            if (denom < resolution) { skipped++; continue; }   // unmeasurable, not wrong

            float diff = fabsf(numeric - analytic);

            // REPORT the true relative error — never a noise-adjusted one, or
            // the output stops telling you anything.
            float rel = diff / denom;
            if (rel > worst_rel) {
                worst_rel = rel; worst_i = k;
                worst_ana = analytic; worst_num = numeric;
            }
            // DECIDE with the noise floor: agree to `tol` relatively, or to
            // within the measurement noise absolutely. Whichever is looser.
            float allowed = tol * denom > noise ? tol * denom : noise;
            if (diff > allowed) violations++;
            checked++;
        }

        // A tensor where every probe was too small to measure is NOT a pass —
        // nothing was actually verified.
        int ok = (checked > 0) && (violations == 0);
        if (!ok) failures++;
        printf("  %-26s %s  worst rel err %.2e at [%zu]  (analytic %+.6e vs numeric %+.6e)"
               "  [%d probed",
               probes[pi].name, checked == 0 ? "NOCHK" : (ok ? "PASS " : "FAIL "),
               worst_rel, worst_i, worst_ana, worst_num, checked);
        if (skipped) printf(", %d skipped below fp32 resolution", skipped);
        // A big relative error on a gradient smaller than the noise floor is
        // expected, not suspicious — say so rather than leaving it looking bad.
        if (ok && worst_rel > tol) printf(", worst probe is below the noise floor");
        printf("]\n");
    }

    printf("--- %d/%d chained tensors passed ---\n", num_probes - failures, num_probes);

    // Ops whose chain position is not reachable yet, tested standalone.
    printf("\n--- isolated op checks (not yet in the chain) ---\n");
    failures += encoder_backward_check();
    printf("\n");

    free(targets); free(inputs); free(grads_acts); free(grads); free(acts); free(params);
    return failures;
}

/*
Overfit-one-batch: the standard smoke test for a training loop.

Take ONE fixed batch and step on it repeatedly. A correct forward + backward +
optimizer must drive the loss to near zero — the model has enough capacity to
simply memorize 10 tokens, so anything else means a bug. This is the cheapest
way to catch an error the gradient check cannot see: the checker verifies that
gradients match the loss surface, but not that the optimizer USES them
correctly (wrong sign, wrong lr scaling, forgetting to zero the grads).

Starting loss should be ~ln(V) = ln(13) = 2.565 for random weights.
*/
int overfit_one_batch(void) {
    Config cfg = { .B = 2, .T = 5, .C = 8, .V = 13, .NH = 2, .maxT = 5 };
    int B = cfg.B, T = cfg.T, V = cfg.V;

    ParamOff p = param_offsets(cfg);
    ActOff   a = act_offsets(cfg);

    float *params     = malloc(p.total * sizeof(float));
    float *acts       = malloc(a.total * sizeof(float));
    float *grads      = calloc(p.total, sizeof(float));
    float *grads_acts = calloc(a.total, sizeof(float));
    float *m          = calloc(p.total, sizeof(float));   // Adam 1st moment
    float *v          = calloc(p.total, sizeof(float));   // Adam 2nd moment
    int *inputs  = malloc(B * T * sizeof(int));
    int *targets = malloc(B * T * sizeof(int));
    if (!params || !acts || !grads || !grads_acts || !m || !v || !inputs || !targets) {
        printf("overfit_one_batch: out of memory\n");
        return 1;
    }

    srand(1337);
    for (size_t i = 0; i < p.total; i++) params[i] = rand_weight();
    for (int c = 0; c < cfg.C; c++) {
        params[p.lng  + c] = 1.0f;  params[p.lnb  + c] = 0.0f;
        params[p.ln2g + c] = 1.0f;  params[p.ln2b + c] = 0.0f;
        params[p.flng + c] = 1.0f;  params[p.flnb + c] = 0.0f;
    }
    for (int i = 0; i < B * T; i++) { inputs[i] = rand() % V; targets[i] = rand() % V; }

    printf("\n--- overfit one batch (B=%d T=%d C=%d V=%d), AdamW lr=1e-2 ---\n",
           cfg.B, cfg.T, cfg.C, cfg.V);
    printf("    a correct loop must drive this to ~0; random-weight start is ln(%d) = %.3f\n",
           V, logf(V));

    float loss = 0.0f;
    int steps = 200;
    for (int step = 1; step <= steps; step++) {
        // Zero BOTH gradient buffers. Every backward op accumulates with +=, so
        // skipping this silently sums gradients across steps — the classic bug.
        memset(grads, 0, p.total * sizeof(float));
        memset(grads_acts, 0, a.total * sizeof(float));

        loss = gpt2_forward(acts, params, inputs, targets, cfg);
        gpt2_backward(grads, grads_acts, acts, params, inputs, targets, cfg);
        adamw_update(params, grads, m, v, p.total, step,
                     1e-2f, 0.9f, 0.999f, 1e-8f, 0.0f);

        if (step == 1 || step % 40 == 0)
            printf("    step %3d   loss %.6f\n", step, loss);
    }

    int ok = loss < 0.05f;
    printf("    %s final loss %.6f (want < 0.05)\n", ok ? "PASS " : "FAIL ", loss);

    free(targets); free(inputs); free(v); free(m);
    free(grads_acts); free(grads); free(acts); free(params);
    return ok ? 0 : 1;
}

/*
Gradient check for the KL term, on its own.

Deliberately NOT folded into bayes_check. The KL is orders of magnitude larger
than the data term at typical settings, so a combined check would let a broken
bayes_backward hide inside the KL's shadow — the numeric and analytic totals
would agree to within tolerance while the data gradient was nonsense. Two
independent tests, each able to fail alone, beat one test that can only fail
loudly.

The upside of isolation: no forward pass is involved at all, so this probes
EVERY parameter rather than a sampled stride, and still runs in microseconds.
*/
int kl_check(void) {
    const size_t n = 512;
    const float sp = 0.1f;
    const float h = 1e-3f;
    const float tol = 1e-3f;

    float *mu   = malloc(n * sizeof(float));
    float *rho  = malloc(n * sizeof(float));
    float *dmu  = calloc(n, sizeof(float));
    float *drho = calloc(n, sizeof(float));
    if (!mu || !rho || !dmu || !drho) { printf("kl_check: out of memory\n"); return 1; }

    srand(1337);
    for (size_t i = 0; i < n; i++) {
        mu[i] = rand_weight() * 5.0f;
        // Spread rho across the interesting range: sigma well below the prior
        // (where -1/sigma dominates and the gradient is large and negative),
        // at it (gradient ~0), and above it. A checker that only ever sees one
        // regime cannot catch a sign error in the other.
        rho[i] = -6.0f + 8.0f * ((float)i / (float)n);
    }

    kl_backward(dmu, drho, mu, rho, n, sp, 1.0f);

    printf("\n--- KL closed-form check (n=%zu, sigma_prior=%.2f, h=%.0e) ---\n", n, sp, h);

    struct { const char *name; float *par; float *grad; } probes[] = {
        { "dKL/dmu ", mu,  dmu  },
        { "dKL/drho", rho, drho },
    };
    int failures = 0;

    for (int pi = 0; pi < 2; pi++) {
        float worst = 0.0f, wa = 0.0f, wn = 0.0f;
        size_t wi = 0;
        for (size_t i = 0; i < n; i++) {
            float saved = probes[pi].par[i];
            probes[pi].par[i] = saved + h;
            double kp = kl_forward(mu, rho, n, sp);
            probes[pi].par[i] = saved - h;
            double km = kl_forward(mu, rho, n, sp);
            probes[pi].par[i] = saved;

            float numeric  = (float)((kp - km) / (2.0 * (double)h));
            float analytic = probes[pi].grad[i];
            float denom = fabsf(numeric) > fabsf(analytic) ? fabsf(numeric) : fabsf(analytic);
            if (denom < 1e-5f) continue;
            float rel = fabsf(numeric - analytic) / denom;
            if (rel > worst) { worst = rel; wi = i; wa = analytic; wn = numeric; }
        }
        int ok = worst < tol;
        if (!ok) failures++;
        printf("  %-10s %s  worst rel err %.2e at [%zu]  (analytic %+.6e vs numeric %+.6e)\n",
               probes[pi].name, ok ? "PASS " : "FAIL ", worst, wi, wa, wn);
    }

    // The three properties the formula must have, checked as values rather
    // than trusted from the derivation.
    float m0 = 0.0f, r_at_prior = logf(expf(sp) - 1.0f);  // softplus^-1(sp)
    double kl_zero = kl_forward(&m0, &r_at_prior, 1, sp);
    float rho_tiny = -12.0f, rho_big = 2.0f;
    double kl_certain = kl_forward(&m0, &rho_tiny, 1, sp);
    float g_dmu = 0.0f, g_drho_lo = 0.0f, g_drho_hi = 0.0f, junk = 0.0f;
    kl_backward(&junk, &g_drho_lo, &m0, &rho_tiny, 1, sp, 1.0f);
    kl_backward(&g_dmu, &g_drho_hi, &m0, &rho_big, 1, sp, 1.0f);

    printf("  identities  q==p costs %.2e nats (want 0) | certainty (sigma=%.1e) costs %.2f nats\n",
           kl_zero, (double)softplus(rho_tiny), kl_certain);
    printf("              drho at sigma<<sp = %+.3e (want <0, grow) | at sigma>>sp = %+.3e (want >0, shrink)\n",
           g_drho_lo, g_drho_hi);
    if (fabs(kl_zero) > 1e-5)  { printf("  FAIL q==p should cost exactly 0\n"); failures++; }
    if (!(g_drho_lo < 0.0f))     { printf("  FAIL gradient must PUSH UP a collapsed sigma\n"); failures++; }
    if (!(g_drho_hi > 0.0f))     { printf("  FAIL gradient must PULL DOWN an inflated sigma\n"); failures++; }

    printf("--- KL check %s ---\n", failures ? "FAILED" : "passed");
    free(drho); free(dmu); free(rho); free(mu);
    return failures;
}

/*
Gradient check for the Bayesian wrapper: does dL/dmu and dL/drho actually match
the loss surface?

ONE thing makes this different from gradient_check(), and it is the trap the
whole module lives or dies on: the loss is now STOCHASTIC. Finite differences
assume the function is fixed while you probe it. If eps is redrawn between the
+h and -h evaluations, the two forwards run two DIFFERENT networks, the
difference is dominated by the resampling noise, and the check reports garbage
no matter how correct the math is.

So eps is drawn exactly once, up front, and held frozen. Conditioned on eps,
the model is a perfectly ordinary deterministic function of (mu, rho) — and an
ordinary function is what finite differences can measure.

Second deliberate choice: rho starts near -1 (sigma ~ 0.31), not at the -5
(sigma ~ 0.0067) a real run would use. dL/drho carries a factor of
sigmoid(rho), so at rho = -5 the rho gradient is ~50x smaller than the mu
gradient and most probes would sink below the fp32 noise floor and be skipped.
A check that skips everything proves nothing.
*/
int bayes_check(void) {
    Config cfg = { .B = 2, .T = 5, .C = 8, .V = 13, .NH = 2, .maxT = 5 };
    int B = cfg.B, T = cfg.T, V = cfg.V;

    ParamOff p = param_offsets(cfg);
    ActOff   a = act_offsets(cfg);

    float *mu   = malloc(p.total * sizeof(float));
    float *rho  = malloc(p.total * sizeof(float));
    float *eps  = malloc(p.total * sizeof(float));
    float *w    = malloc(p.total * sizeof(float));
    float *dmu  = calloc(p.total, sizeof(float));
    float *drho = calloc(p.total, sizeof(float));
    float *grads      = calloc(p.total, sizeof(float));
    float *acts       = malloc(a.total * sizeof(float));
    float *grads_acts = calloc(a.total, sizeof(float));
    int *inputs  = malloc(B * T * sizeof(int));
    int *targets = malloc(B * T * sizeof(int));
    if (!mu || !rho || !eps || !w || !dmu || !drho || !grads || !acts ||
        !grads_acts || !inputs || !targets) {
        printf("bayes_check: out of memory\n");
        return 1;
    }

    srand(1337);
    for (size_t i = 0; i < p.total; i++) {
        mu[i]  = rand_weight();
        rho[i] = -1.0f + rand_weight();   // jittered so an index bug can't hide
    }
    for (int c = 0; c < cfg.C; c++) {
        mu[p.lng  + c] = 1.0f;  mu[p.lnb  + c] = 0.0f;
        mu[p.ln2g + c] = 1.0f;  mu[p.ln2b + c] = 0.0f;
        mu[p.flng + c] = 1.0f;  mu[p.flnb + c] = 0.0f;
    }
    for (int i = 0; i < B * T; i++) { inputs[i] = rand() % V; targets[i] = rand() % V; }

    // Draw ONCE. From here on eps is a constant of the problem.
    bayes_sample(w, eps, mu, rho, p.total);

    float loss0 = gpt2_forward(acts, w, inputs, targets, cfg);
    gpt2_backward(grads, grads_acts, acts, w, inputs, targets, cfg);
    bayes_backward(dmu, drho, grads, eps, rho, p.total);

    const float h = 1e-2f;
    const float tol = 1e-2f;
    float loss_ulp = nextafterf(loss0, 1e30f) - loss0;
    float noise = 4.0f * loss_ulp / (2.0f * h);
    const float resolution = 1e-6f;

    printf("\n--- bayesian reparam check (B=%d T=%d C=%d V=%d, eps frozen, h=%.0e) ---\n",
           cfg.B, cfg.T, cfg.C, cfg.V, h);
    printf("    loss %.4f, %zu params, noise floor %.2e\n", loss0, p.total, noise);

    struct { const char *name; float *par; float *grad; } probes[] = {
        { "dL/dmu   (posterior mean)",   mu,  dmu  },
        { "dL/drho  (posterior spread)", rho, drho },
    };
    int num_probes = sizeof(probes) / sizeof(probes[0]);
    int failures = 0;

    for (int pi = 0; pi < num_probes; pi++) {
        float worst_rel = 0.0f, worst_ana = 0.0f, worst_num = 0.0f;
        size_t worst_i = 0;
        int checked = 0, skipped = 0, violations = 0;
        // A probe only proves something if the gradient it is testing rises
        // ABOVE the measurement noise. Probes below it pass automatically —
        // counting them as verification is how a test goes green while blind.
        int decisive = 0;

        size_t stride = p.total / 40 ? p.total / 40 : 1;
        for (size_t idx = 0; idx < p.total; idx += stride) {
            float saved = probes[pi].par[idx];

            probes[pi].par[idx] = saved + h;
            bayes_materialize(w, mu, rho, eps, p.total);       // eps unchanged
            float loss_plus = gpt2_forward(acts, w, inputs, targets, cfg);

            probes[pi].par[idx] = saved - h;
            bayes_materialize(w, mu, rho, eps, p.total);
            float loss_minus = gpt2_forward(acts, w, inputs, targets, cfg);

            probes[pi].par[idx] = saved;

            float numeric  = (loss_plus - loss_minus) / (2.0f * h);
            float analytic = probes[pi].grad[idx];

            float denom = fabsf(numeric) > fabsf(analytic) ? fabsf(numeric) : fabsf(analytic);
            if (denom < resolution) { skipped++; continue; }

            float diff = fabsf(numeric - analytic);
            float rel  = diff / denom;
            // Rank the WORST decisive probe, not the worst probe overall — a
            // huge relative error on an unmeasurable gradient is expected and
            // would otherwise mask a real one.
            if (denom > noise) {
                decisive++;
                if (rel > worst_rel) {
                    worst_rel = rel; worst_i = idx;
                    worst_ana = analytic; worst_num = numeric;
                }
            }
            float allowed = tol * denom > noise ? tol * denom : noise;
            if (diff > allowed) violations++;
            checked++;
        }

        // Decisive probes, not merely probed ones, are what makes this a PASS.
        int ok = (decisive > 0) && (violations == 0);
        if (!ok) failures++;
        printf("  %-28s %s  worst rel err %.2e at [%zu]  (analytic %+.6e vs numeric %+.6e)"
               "  [%d probed, %d decisive",
               probes[pi].name, decisive == 0 ? "NOCHK" : (ok ? "PASS " : "FAIL "),
               worst_rel, worst_i, worst_ana, worst_num, checked, decisive);
        if (skipped) printf(", %d unmeasurable", skipped);
        printf("]\n");
    }

    // Restore w to the unperturbed sample so the buffer is not left mid-probe.
    bayes_materialize(w, mu, rho, eps, p.total);

    printf("--- %d/%d bayesian gradients passed ---\n\n", num_probes - failures, num_probes);

    free(targets); free(inputs); free(grads_acts); free(acts); free(grads);
    free(drho); free(dmu); free(w); free(eps); free(rho); free(mu);
    return failures;
}

/*
Overfit-one-batch, Bayesian edition. Same smoke test as the deterministic file:
a correct sample/forward/backward/unwrap/step loop must memorize one tiny batch.

There is NO KL term yet, and the consequence is worth watching rather than
hiding — it is the argument for the next module. Without KL, nothing rewards
uncertainty. Noise on the weights only ever makes the fit worse in expectation,
so the optimizer's cheapest move is to drive sigma toward zero and turn the
model back into an ordinary deterministic network. Mean sigma is printed every
step for exactly that reason: watch it collapse.

That collapse is not a bug in this code. It is what a Bayesian model does when
you delete its prior, and it is precisely what the KL term exists to fight.
*/
int bnn_overfit_one_batch(float kl_weight, float sigma_prior) {
    Config cfg = { .B = 2, .T = 5, .C = 8, .V = 13, .NH = 2, .maxT = 5 };
    int B = cfg.B, T = cfg.T, V = cfg.V;

    ParamOff p = param_offsets(cfg);
    ActOff   a = act_offsets(cfg);

    float *mu   = malloc(p.total * sizeof(float));
    float *rho  = malloc(p.total * sizeof(float));
    float *eps  = malloc(p.total * sizeof(float));
    float *w    = malloc(p.total * sizeof(float));
    float *dmu  = calloc(p.total, sizeof(float));
    float *drho = calloc(p.total, sizeof(float));
    float *grads      = calloc(p.total, sizeof(float));
    float *acts       = malloc(a.total * sizeof(float));
    float *grads_acts = calloc(a.total, sizeof(float));
    // Adam state is PER PARAMETER TENSOR, and mu and rho are two separate
    // parameter vectors — they need their own moments. This is where the
    // "optimizer memory doubles" claim actually shows up in the code.
    float *m_mu  = calloc(p.total, sizeof(float));
    float *v_mu  = calloc(p.total, sizeof(float));
    float *m_rho = calloc(p.total, sizeof(float));
    float *v_rho = calloc(p.total, sizeof(float));
    int *inputs  = malloc(B * T * sizeof(int));
    int *targets = malloc(B * T * sizeof(int));
    if (!mu || !rho || !eps || !w || !dmu || !drho || !grads || !acts ||
        !grads_acts || !m_mu || !v_mu || !m_rho || !v_rho || !inputs || !targets) {
        printf("bnn_overfit_one_batch: out of memory\n");
        return 1;
    }

    srand(1337);
    for (size_t i = 0; i < p.total; i++) {
        mu[i]  = rand_weight();
        // rho = -5 -> sigma = softplus(-5) = 0.0067. Small on purpose: start a
        // BNN with large sigma and the forward pass is pure noise, the loss is
        // flat, and it never gets off the ground.
        rho[i] = -5.0f;
    }
    for (int c = 0; c < cfg.C; c++) {
        mu[p.lng  + c] = 1.0f;  mu[p.lnb  + c] = 0.0f;
        mu[p.ln2g + c] = 1.0f;  mu[p.ln2b + c] = 0.0f;
        mu[p.flng + c] = 1.0f;  mu[p.flnb + c] = 0.0f;
    }
    for (int i = 0; i < B * T; i++) { inputs[i] = rand() % V; targets[i] = rand() % V; }

    printf("\n--- BNN overfit one batch (B=%d T=%d C=%d V=%d), AdamW lr=1e-2, "
           "kl_weight=%.0e, sigma_prior=%.2f ---\n",
           cfg.B, cfg.T, cfg.C, cfg.V, kl_weight, sigma_prior);
    printf("    random-weight start is ln(%d) = %.3f;  %s\n", V, logf(V),
           kl_weight == 0.0f ? "no prior -> sigma should COLLAPSE"
                             : "prior on -> sigma should settle, NOT collapse");

    float loss = 0.0f;
    int steps = 200;
    for (int step = 1; step <= steps; step++) {
        memset(grads, 0, p.total * sizeof(float));
        memset(grads_acts, 0, a.total * sizeof(float));
        memset(dmu, 0, p.total * sizeof(float));
        memset(drho, 0, p.total * sizeof(float));

        // ONE Monte Carlo draw per step: W^(k) ~ q(W).
        bayes_sample(w, eps, mu, rho, p.total);

        // Everything between these two lines is the unmodified GPT-2.
        loss = gpt2_forward(acts, w, inputs, targets, cfg);
        gpt2_backward(grads, grads_acts, acts, w, inputs, targets, cfg);

        // dL/dw -> dL/dmu, dL/drho. The whole Bayesian backward, one call.
        bayes_backward(dmu, drho, grads, eps, rho, p.total);

        // The second, entirely independent gradient source. It never looked at
        // the data, the activations, or eps — only at mu and rho. It lands on
        // the same two buffers via +=, and neither backward knows the other ran.
        double kl = kl_forward(mu, rho, p.total, sigma_prior);
        kl_backward(dmu, drho, mu, rho, p.total, sigma_prior, kl_weight);

        // wd = 0 deliberately: weight decay IS a Gaussian prior on the weights,
        // and the KL term above is that same prior stated properly. Running
        // both would apply one prior twice.
        adamw_update(mu,  dmu,  m_mu,  v_mu,  p.total, step,
                     1e-2f, 0.9f, 0.999f, 1e-8f, 0.0f);
        adamw_update(rho, drho, m_rho, v_rho, p.total, step,
                     1e-2f, 0.9f, 0.999f, 1e-8f, 0.0f);

        if (step == 1 || step % 40 == 0) {
            double sig = 0.0;
            for (size_t i = 0; i < p.total; i++) sig += softplus(rho[i]);
            printf("    step %3d   NLL %.6f   KL %9.2f   ELBO %10.4f   mean sigma %.6f\n",
                   step, loss, kl, loss + (double)kl_weight * kl, sig / (double)p.total);
        }
    }

    // The pass bar only applies without the prior. With KL on, refusing to
    // memorize 10 tokens with 1149 parameters is the CORRECT behaviour, not a
    // failure — so asserting loss<0.05 there would be asserting a bug.
    double sig_final = 0.0;
    for (size_t i = 0; i < p.total; i++) sig_final += softplus(rho[i]);
    sig_final /= (double)p.total;

    int ok;
    if (kl_weight == 0.0f) {
        ok = loss < 0.05f;
        printf("    %s final NLL %.6f (want < 0.05), sigma %.6f -> collapsing, as predicted\n",
               ok ? "PASS " : "FAIL ", loss, sig_final);
    } else {
        // Without a prior sigma only ever decreased. The test is that it no
        // longer does: it must have moved UP off its 0.006713 start, toward sp.
        ok = sig_final > 0.0068;
        printf("    %s final NLL %.6f, sigma %.6f -> %s (started 0.006713, prior %.3f)\n",
               ok ? "PASS " : "FAIL ", loss, sig_final,
               ok ? "held up by the prior" : "STILL COLLAPSING", sigma_prior);
    }

    free(targets); free(inputs);
    free(v_rho); free(m_rho); free(v_mu); free(m_mu);
    free(grads_acts); free(acts); free(grads);
    free(drho); free(dmu); free(w); free(eps); free(rho); free(mu);
    return ok ? 0 : 1;
}

/*
Guarded so this file can be compiled as a LIBRARY of CPU reference ops and
linked into the CUDA build (`-DSKIP_MAIN`), which supplies its own main. Every
GPU kernel is then checked against the exact function it replaces — not a
reimplementation of it, which would only prove a copy matches its original.
*/
#ifndef SKIP_MAIN
int main(int argc, char **argv) {

    // `./train_gpt2 check` runs only the gradient check (well under a second)
    // instead of the full-size forward+backward (~90 s). Use it while writing
    // backward ops; use the plain run to see the real model's numbers.
    if (argc > 1 && argv[1][0] == 'c') return gradient_check() == 0 ? 0 : 1;
    if (argc > 1 && argv[1][0] == 't') return overfit_one_batch();
    // `b` = the whole Bayesian layer: both gradient checks, then the same
    // training loop run twice — without and with the prior — so the effect of
    // the KL term is a controlled comparison rather than an assertion.
    if (argc > 1 && argv[1][0] == 'b') {
        int fails = 0;
        fails += (bayes_check() == 0 ? 0 : 1);
        fails += (kl_check() == 0 ? 0 : 1);
        fails += bnn_overfit_one_batch(0.0f,  0.1f);
        fails += bnn_overfit_one_batch(1e-3f, 0.1f);
        return fails ? 1 : 0;
    }

    // ---- 1. open the file -------------------------------------------------
    // "rb" = read, binary. Binary because this file is raw numbers, not text.
    const char *path = "data/tinyshakespeare/tiny_shakespeare_train.bin";
    FILE *f = fopen(path, "rb");
    if (f == NULL) {
        printf("could not open %s\n", path);
        return 1;
    }

    // ---- 2. read the label ------------------------------------------------
    // The first 256 numbers aren't Shakespeare. They describe the file.
    // Only the first three slots are used; the rest are zeros held in reserve.
    int32_t header[HEADER_INTS];
    if (fread(header, sizeof(int32_t), HEADER_INTS, f) != HEADER_INTS) {
        printf("file is too short to even hold a header\n");
        fclose(f);
        return 1;
    }

    if (header[0] != MAGIC) {
        printf("wrong magic: got %d, expected %d — is this the right file?\n",
               header[0], MAGIC);
        fclose(f);
        return 1;
    }
    if (header[1] != VERSION) {
        printf("wrong version: got %d, expected %d\n", header[1], VERSION);
        fclose(f);
        return 1;
    }

    int num_tokens = header[2];   // how many Shakespeare-numbers follow
    printf("file looks good. it holds %d tokens\n", num_tokens);

    // ---- 3. read all the tokens into memory -------------------------------
    // On disk each token is a uint16_t (2 bytes) — small, because there are
    // only ~50k possible tokens and that fits in 16 bits.
    //
    // The whole file is under a megabyte, so we just load all of it at once.
    // (Real datasets are too big for this and get streamed instead.)
    uint16_t *tokens = malloc(num_tokens * sizeof(uint16_t));
    if (tokens == NULL) {
        printf("out of memory\n");
        fclose(f);
        return 1;
    }

    if (fread(tokens, sizeof(uint16_t), num_tokens, f) != (size_t)num_tokens) {
        printf("file ended early — expected %d tokens\n", num_tokens);
        free(tokens);
        fclose(f);
        return 1;
    }

    fclose(f);   // everything we need is in memory now; done with the file

    // ---- 4. carve out one handful -----------------------------------------
    // A handful is B*T tokens of input, plus ONE extra token.
    //
    // Why one extra? Because the model's job is to predict the NEXT token.
    // So for every input token we need to know what actually followed it.
    // The answer for token i is simply token i+1 — the text is its own
    // answer key, shifted by one. That's why we grab B*T+1 and slide.
    //
    //   tokens:   [ 464, 3809, 286, 262, ... ]
    //   inputs:     464  3809  286        (what the model sees)
    //   targets:         3809  286  262   (what it should have said)
    //
    // The model doesn't get one guess per handful — it gets B*T guesses,
    // one at every position, all graded at once. That's what makes training
    // efficient.
    int B = 4;    // batch size:      how many sequences at once
    int T = 64;   // sequence length: how many tokens in each

    int inputs[B * T];
    int targets[B * T];

    int offset = 0;   // start at the beginning; later this walks through the file

    for (int i = 0; i < B * T; i++) {
        // widen from uint16_t to int: the model wants plain ints
        inputs[i]  = (int) tokens[offset + i];
        targets[i] = (int) tokens[offset + i + 1];   // the +1 is the whole trick
    }

    // ---- 5. build the model's tables — ONE block, sliced ------------------
    // These are WEIGHTS — the model itself. They exist before any data is seen
    // and they persist for the whole training run.
    //
    // Instead of one malloc per tensor, ALL weights live in a single block of
    // floats, and each tensor is just a pointer sliced out of it by an offset:
    //
    //   params:  [ wte ............ | wpe ... | ln1_g | ln1_b ]
    //            ^                  ^         ^       ^
    //            offset 0           V*C       +maxT*C +C
    //
    // Why: buffers that are born together and die together should travel
    // together. One malloc = one free (no leak paths), and this is exactly
    // llm.c's ParameterTensors — which pays off hugely on the GPU later.
    Config cfg = { .B = B, .T = T, .C = 768, .V = 50257, .NH = 12, .maxT = 1024 };
    int C = cfg.C, V = cfg.V, NH = cfg.NH, maxT = cfg.maxT;
    (void)NH;

    ParamOff p = param_offsets(cfg);
    ActOff   a = act_offsets(cfg);

    float *params = malloc(p.total * sizeof(float));
    if (params == NULL) {
        printf("out of memory allocating weights\n");
        free(tokens);
        return 1;
    }
    // Slicing: pointer + offset IS an array — nothing downstream changes.
    float *wte  = params + p.wte;   // ~154 MB
    float *wpe  = params + p.wpe;   // ~3 MB
    float *ln_g = params + p.lng;
    float *ln_b = params + p.lnb;
    float *mlp_w1 = params + p.w1;  // ~9 MB
    float *mlp_b1 = params + p.b1;
    float *mlp_w2 = params + p.w2;  // ~9 MB
    float *mlp_b2 = params + p.b2;
    float *qkv_w  = params + p.qw;
    float *qkv_b  = params + p.qb;
    float *proj_w = params + p.pw;
    float *proj_b = params + p.pb;
    float *ln2_g  = params + p.ln2g;
    float *ln2_b  = params + p.ln2b;
    float *fln_g  = params + p.flng;
    float *fln_b  = params + p.flnb;
    float *head_w = params + p.hw;
    float *head_b = params + p.hb;

    // Fill embeddings with small random numbers centered on zero. Real
    // training starts from noise like this and reshapes it.
    srand(42);   // fixed seed so every run gives identical numbers
    for (size_t i = 0; i < (size_t)V * C; i++)
        wte[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.02f;
    for (size_t i = 0; i < (size_t)maxT * C; i++)
        wpe[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.02f;

    // LayerNorm's knobs start as IDENTITY: g=1 (multiply by 1 = no change),
    // b=0 (add 0 = no change). At birth the thermostat just normalizes;
    // training will bend these if that helps.
    for (int c = 0; c < C; c++) {
        ln_g[c] = 1.0f;
        ln_b[c] = 0.0f;
    }

    // MLP weights: small random like the embeddings; biases start at 0
    // (a bias of zero = "no opinion yet").
    for (size_t i = 0; i < (size_t)C * 4 * C; i++)
        mlp_w1[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.02f;
    for (size_t i = 0; i < (size_t)4 * C * C; i++)
        mlp_w2[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.02f;
    for (int i = 0; i < 4 * C; i++) mlp_b1[i] = 0.0f;
    for (int i = 0; i < C; i++)     mlp_b2[i] = 0.0f;

    // Attention weights: same recipe — small random W, zero bias, ln2 identity.
    for (size_t i = 0; i < (size_t)C * 3 * C; i++)
        qkv_w[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.02f;
    for (size_t i = 0; i < (size_t)C * C; i++)
        proj_w[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.02f;
    for (int i = 0; i < 3 * C; i++) qkv_b[i] = 0.0f;
    for (int i = 0; i < C; i++)     proj_b[i] = 0.0f;
    for (int c = 0; c < C; c++) { ln2_g[c] = 1.0f; ln2_b[c] = 0.0f; }
    for (int c = 0; c < C; c++) { fln_g[c] = 1.0f; fln_b[c] = 0.0f; }
    for (size_t i = 0; i < (size_t)C * V; i++)
        head_w[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.02f;
    for (int i = 0; i < V; i++) head_b[i] = 0.0f;

    // ---- 6. activations: one block, sliced ---------------------------------
    // ACTIVATIONS, not weights: scratch space holding intermediate results for
    // one batch. Same one-malloc trick, layout owned by act_offsets().
    float *acts = malloc(a.total * sizeof(float));
    if (acts == NULL) {
        printf("out of memory allocating activations\n");
        free(params); free(tokens);
        return 1;
    }
    float *encoded  = acts + a.enc;
    float *mlp_fch  = acts + a.fch;
    float *mlp_out  = acts + a.mlpo;
    float *res1     = acts + a.res1;
    float *att      = acts + a.att;
    float *attproj  = acts + a.apr;

    // ---- 7. the forward pass, in one call ----------------------------------
    // Every op from encoder to cross-entropy now lives in gpt2_forward(). That
    // is what lets the gradient checker below re-run it with a nudged weight.
    float mean_loss = gpt2_forward(acts, params, inputs, targets, cfg);

    // Sanity checks on the way through — all reading buffers gpt2_forward filled.
    printf("\ntoken %d encodes to: ", inputs[0]);
    for (int c = 0; c < 10; c++) printf("%+.4f ", encoded[c]);
    printf("... (%d floats total)\n", C);

    // att rows are softmaxes — token 0 can only see itself, so its single
    // weight must be exactly 1.0.
    printf("att[batch 0, head 0, token 0, ->token 0] = %.6f (should be 1)\n", att[0]);
    printf("attention out row 0: ");
    for (int c = 0; c < 8; c++) printf("%+.4f ", attproj[c]);
    printf("\n");
    printf("mlp hidden row 0: ");
    for (int c = 0; c < 8; c++) printf("%+.4f ", mlp_fch[c]);
    printf("... (%d floats)\n", 4*C);
    printf("mlp output row 0: ");
    for (int c = 0; c < 8; c++) printf("%+.4f ", mlp_out[c]);
    printf("... (%d floats — back to C)\n", C);
    printf("stream after residual, row 0: ");
    for (int c = 0; c < 8; c++) printf("%+.4f ", res1[c]);
    printf("\n");

    // Average loss over all B*T predictions. Random weights should be close
    // to ln(V) ~ 10.82, because every token is roughly equally guessed.
    printf("mean loss: %.4f (expected ~%.4f for random weights)\n", mean_loss, logf(V));

    // ---- 8. the shadows: two more blocks, same shapes, same offsets --------
    // Every buffer that mattered on the way forward needs a partner holding
    // its gradient. The offsets are ALREADY WRITTEN — a gradient has the same
    // shape as the thing it measures, so `grads + p.wte` is dwte. This is the
    // payoff for having done the one-big-malloc thing.
    //
    // The two blocks have very different lifetimes:
    //   grads      = the DELIVERABLE. One slope per weight. Adam reads it next
    //                module, steps the weights downhill, then it's zeroed.
    //   grads_acts = SCRATCH. Only carries signal from one op's output back to
    //                the previous op's input. Garbage once the pass finishes.
    //
    // calloc, not malloc: gradients ACCUMULATE with `+=`. When a buffer is used
    // more than once in forward (the residual stream is used twice; a wte row
    // is used by every token that has that ID), each use contributes a separate
    // term, and they must sum onto a floor of zero.
    float *grads = calloc(p.total, sizeof(float));
    float *grads_acts = calloc(a.total, sizeof(float));
    if (grads == NULL || grads_acts == NULL) {
        printf("out of memory allocating gradients\n");
        free(acts); free(params); free(tokens);
        return 1;
    }

    // ---- 9. the backward pass, in one call ---------------------------------
    // Reaches: loss -> logits -> final LN -> residual fork -> MLP W2 -> GELU.
    // Still to write: MLP up-projection, ln2, attention, ln1, encoder.
    gpt2_backward(grads, grads_acts, acts, params, inputs, targets, cfg);

    float *dfln_b   = grads + p.flnb;
    float *dhead_b  = grads + p.hb;
    float *dres1    = grads_acts + a.res1;
    float *dres_att = grads_acts + a.resa;
    float *dmlp_out = grads_acts + a.mlpo;
    float *dmlp_gelu = grads_acts + a.gelu;
    float *dmlp_fch  = grads_acts + a.fch;

    // dhead_b is the column-sum of dlogits, and each dlogits row sums to 0.
    float head_bias_grad_sum = 0.0f;
    for (int v = 0; v < V; v++) head_bias_grad_sum += dhead_b[v];
    printf("dhead_b sum = %+.3e (should be ~0 because each dlogits row sums to 0)\n",
           head_bias_grad_sum);

    float dfln_b_absmax = 0.0f, dres1_absmax = 0.0f;
    for (int c = 0; c < C; c++)
        if (fabsf(dfln_b[c]) > dfln_b_absmax) dfln_b_absmax = fabsf(dfln_b[c]);
    for (int i = 0; i < B * T * C; i++)
        if (fabsf(dres1[i]) > dres1_absmax) dres1_absmax = fabsf(dres1[i]);
    printf("dfln_b absmax = %.3e, dres1 absmax = %.3e (both should be finite, non-zero)\n",
           dfln_b_absmax, dres1_absmax);

    // The residual fork copies dres1 to both branches. dmlp_out still equals it
    // exactly (nothing else writes there), but dres_att does NOT: by the time
    // the chain finishes, ln2's backward has ADDED its contribution on top. That
    // difference is the two-upstream-paths accumulation working, not a bug.
    printf("residual fork: dmlp_out[0]=%+.6e == dres1[0]=%+.6e | dres_att[0]=%+.6e (larger: fork + ln2)\n",
           dmlp_out[0], dres1[0], dres_att[0]);

    // GELU's local slope lives in [~-0.13, ~1.13], so the gradient must shrink
    // on average passing through it — never blow up.
    double gelu_in_abs = 0.0, gelu_out_abs = 0.0;
    for (int i = 0; i < B * T * 4 * C; i++) {
        gelu_in_abs  += fabsf(dmlp_gelu[i]);
        gelu_out_abs += fabsf(dmlp_fch[i]);
    }
    printf("gelu backward: mean|dout|=%.3e -> mean|dinp|=%.3e (ratio %.3f, should be <1.13)\n",
           gelu_in_abs / (B * T * 4 * C), gelu_out_abs / (B * T * 4 * C),
           gelu_out_abs / gelu_in_abs);

    // ---- 10. the actual test ------------------------------------------------
    // Everything above is a smell test. THIS is the check: finite differences
    // on a tiny model, comparing each analytic gradient against the measured
    // slope of the loss. Runs in well under a second.
    int failures = gradient_check();

    // Two mallocs in main -> two frees. Add a tensor later = bump an offset,
    // not a new malloc/free pair to keep track of.
    free(grads_acts);
    free(grads);
    free(acts);
    free(params);
    free(tokens);
    return failures == 0 ? 0 : 1;
}
#endif  /* SKIP_MAIN */
