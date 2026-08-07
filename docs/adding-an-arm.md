# Adding an arm

An *arm* is one method for putting uncertainty on the weights of the shared
GPT-2. It is one `.cu` file. Everything else — the network, the kernels, the
dataloader, the checkpoint format, the timing harness and the evaluation
harness — is inherited from `gpt2_kernels.cuh`, which is what makes two arms
comparable: they differ only in what they store and how many passes they make
over that store.

This document is written for someone who has not read the repository. Nothing
here needs a GPU until step 4.

---

## 0. What you are actually writing

Two functions.

**An update kernel.** It takes whatever your method stores plus `dw`, the
parameter-sized gradient the shared backward pass just accumulated, and writes
your state forward one step. If you fuse (§3) it also writes the next step's
weight draw `w`.

**A draw callback**, matching

```c
typedef const float *(*uq_sample_fn)(int s, void *ctx);
```

"Populate a parameter vector for sample `s` and return a device pointer to it."
It returns a pointer rather than filling a caller-supplied buffer so the deep
ensemble can hand back member `s` with no copy at all, while a variational arm
fills its own scratch `w`.

The four reference arms implement both in roughly 150 lines of CUDA each
(`train_bnn_lowrank.cu` is larger because rank-8 covariance is), against a
shared header of 1,610 lines.

## 1. Start from the closest reference arm

| file | method | state per parameter | touches/step |
|---|---|---|---|
| `train_bnn_vadam.cu` | Vadam (Khan et al. 2018) | `mu`, Adam `m`, `s` | 11, floor 9 |
| `train_bnn_gpu.cu` | Bayes-by-Backprop | `mu`, `rho`, an Adam pair for each | 32, floor 15 |
| `train_bnn_lowrank.cu` | low-rank + diagonal | `mu`, `A` (rank `r`), `d`, Adam state | 122/layer, floor 58 |
| `train_ensemble.cu` | deep ensemble (control) | none (E whole models) | 8 |

`train_bnn_vadam.cu` is the smallest and the best starting point: it stores the
least, and its `main()` shows the full shape — argument parsing, allocation,
initialisation, the training loop with bench marks, checkpoint save, and an
`eval` entry point — with nothing method-specific in the way.

Copy it to `train_<yourarm>.cu` and add a target to the root `Makefile`
alongside the others.

## 2. The buffer layout is one table

`param_offsets(cfg)` in `gpt2_kernels.cuh` is the single source of truth for
where every tensor of the network lives inside one flat parameter-sized buffer,
and `act_offsets(cfg)` does the same for activations. Index through them
(`PBLK(p, l, qw)` for block `l`'s attention weight) rather than recomputing an
offset; add a tensor there and every caller sees it.

Your own state is separate: allocate one device buffer of `p.total` floats per
parameter-sized thing your method keeps. `P = 11,061,531` at the shipped
`Config` (C=384, L=6, V=27), so each is 44.2 MB. Count them before you write
them — that count is your arm's cost (§4).

Two constraints that are not negotiable:

- **`HS` (head size) must be exactly 64.** It is a compile-time `#define` and
  the FlashAttention kernels index q/k/v with it, so any config where
  `C/NH != 64` reads past every head. `gpt2_forward` aborts on it now; before
  that it returned confident garbage.
- **Derive constants; do not type them.** The ELBO's KL weight is `1/N` from
  the loader, not a tuned knob. Vadam's `lambda` is `1/init_std^2`. Hardcoding
  the first cost 1.0 nats once already.

## 3. Fusing is optional, and it is a separate change

Get the unfused arm correct first: one kernel per logical step, priming the
loop with a plain sample kernel. Then, if your arm has redundant traffic to
delete, fold its parameter-sized passes into one kernel.

The observation that does most of the work in all three Bayesian arms here:
**a gradient accumulator for a variational parameter is write-only.** It is
zeroed, added to by exactly two kernels, read once by the update, and never
looked at again, so it never has to reach memory — it becomes a register. The
next step's draw folds in the same way, being a function of the parameters the
update just wrote.

**Keep the unfused kernels.** Eval draws with them, each loop primes with them,
and the equivalence check compares against them.

Fusion is not free: a kernel streaming eight buffers achieves less bandwidth
than several streaming three or four each. Vadam's fold cut traffic 1.22x and
returned no measurable time, which its pass count predicted. If your arm has
few touches to begin with, expect the same.

## 4. Count your passes before you measure them

A **touch** is one read or one write of a parameter-sized buffer in a step. A
deterministic AdamW step costs eight: a gradient zeroed and read, weights and
two moments each read and written. Anything above that is your method's.

Write the count down at the launch sites. `paper/tables/make_tables.py` reads
touch counts from a `TOUCHES` table in the generator, and the check on them is
`tab-bandwidth`: converting each region's time to achieved bandwidth has to put
structurally unrelated regions in one band (235–254 GB/s on this card). A
region sitting off the band means its touch count is wrong.

Label each bench region by cost class in the first character of its name:

- `*` — O(P) and Bayesian
- `+` — O(P) but a deterministic trainer pays it too
- unprefixed — O(B·T), the network

## 5. Three conventions the shared code depends on

Each fails silently rather than loudly. All three are also recorded in
`AGENTS.md`.

**`main()` lives inside `#ifndef UNCERTAIN_AS_LIBRARY`.** The Python extension
(`uncertain/_C.cu`) `#include`s every arm source verbatim into one translation
unit, and four `main()` symbols cannot coexist in one module. Everything else —
your kernels, your `uq_sample_fn` — stays *outside* the guard, so the module
gets your arithmetic with no second copy. `dev/` test files use the same guard
to test the real kernels.

**Reach cuBLAS only through the `cublas_handle` macro.** It expands to an inline
accessor's local static, which is one handle per linked image whatever the
translation-unit layout. A plain `static` at file scope silently gives every
translation unit its own null copy.

**Fatal paths go through `UNCERTAIN_ABORT`,** which is `exit()` in a binary and
`throw` in the module. A bare `exit()` in shared code kills the user's Python
interpreter.

## 6. Running the harness

Build:

```
make train_<yourarm>                 # ARCH=sm_XX to override sm_75
```

Train, and save a checkpoint the evaluator can load. The first argument selects
the mode by its first letter (`o`verfit, `e`val, `b`ench); anything else is
normal training, and the second argument is the step count:

```
./train_<yourarm> overfit            # one batch, smoke test: loss must collapse
./train_<yourarm>                    # normal training, default step count
./train_<yourarm> train 100000       # the run the tables are read from
```

Evaluate, from the checkpoint, never training:

```
./train_<yourarm> eval [steps] [S] [batches] [split]      # split = val | test
```

`uq_evaluate` prints NLPD (mixture and single draw), ECE, and the
total/aleatoric/epistemic decomposition, plus the same on an off-distribution
stream of uniform-random token ids. Every number carries its `S` and its split.

Time it:

```
./train_<yourarm> bench [B] [steps] [warmup]     # one point
./bench_sweep.sh <tag>                           # the whole B sweep
```

The sweep writes `bench:` and `bench-region:` lines into `runs/bench/<tag>/`.
Warm-up is discarded, the statistic is a median with p25/p75, and every line
carries the temperature and SM-clock range it was taken at.

## 7. What to run before you report a number

In roughly this order, cheapest first.

1. **`./train_<yourarm> overfit`** — one batch, loss must collapse. An arm
   whose injected noise never shrinks fails this; Vadam does, and that is a
   finding about Vadam rather than a bug.
2. **A per-kernel check in `dev/`** — one file, your kernel against a
   single-threaded CPU reference. `dev/vadam_update.cu` and
   `dev/vadam_sample.cu` are the pattern for kernels that live in an arm file:
   they `#include ../train_bnn_vadam.cu` with `UNCERTAIN_AS_LIBRARY` defined,
   so they test the real kernel rather than a copy that can drift.
3. **`make test`** — builds every arm and the scorecard, runs each in both
   modes, prints one table. It parses stdout by format string
   (`step %d | nll %f | ...`), so keep your step line in that shape.
4. **If you fused: `dev/fused.cu`** — run the fused kernel and the kernels it
   replaced on identical inputs and compare every output. It builds twice:
   `make -C dev fused fused_nofma`, the second with `-fmad=false`, where the
   requirement is exact bit equality. That second build is what makes the
   equivalence a result rather than a claim, because no tolerance can separate
   "the compiler contracted a multiply-add differently" from "the arithmetic
   changed".
5. **If you touched shared code: a trajectory diff.** Rebuild the previous
   `.cu` files from git into a scratch tree, run both for ~100 steps in
   *normal* mode, and diff the `step` lines with the `ms` column stripped.
   `train_ensemble` must be diffed too, step lines and both `summary:` lines:
   it is the reference this repository is calibrated against, and if a refactor
   changes any number it prints, that is a bug in the refactor.

Two traps any numerical check here will hit:

- **Never normalize an error by `|want|`.** Quantities that cancel to near zero
  report 100% relative error on correct arithmetic. Normalize by the scale of
  the computation (sum of input terms, or `max(|value|, buffer RMS)`).
- **Draw Adam's `m` and `v` together.** They accumulate from the same gradient,
  so `v ~ m^2` and the update is O(1). Drawing them independently over wide log
  ranges manufactures states that never occur and turns a 1-ULP difference into
  an apparent catastrophe.

Two gates that look sufficient and are not: the eval harness cannot see a
training-path change (it loads checkpoints and draws with the sample kernels),
and the `total == aleatoric + epistemic` assertion is normalisation-free, so it
holds for any non-negative vectors and passes at 1e-15 on an unnormalised row
or a stale sample buffer. Neither certifies your update.

## 8. The baseline to compare against

`paper/tables/generated/tab-posterior.tex` (per-step posterior cost across the
batch sweep) and `tab-quality.tex` / `tab-uncertainty.tex` / `tab-ood.tex`
(what the four arms return at 100,000 steps) are the reference rows, all at the
shared configuration of §2 on one RTX 2060 6 GB. Regenerate them with your
arm's logs present:

```
python3 paper/tables/make_tables.py
```

A cell with no measurement behind it renders as a loud red `TODO`, never as a
plausible value. Timings on another card are that card's numbers.
