# Project agent memory

This file is the project's committed home for project-intrinsic agent knowledge: build, test, release, architecture, and sharp-edge notes that should travel with the code.

**Read `RESUME.md` first.** It is the authoritative state file: what is trained,
what is not, what the open work is in priority order, and what was already
learned (and must not be re-derived). It is kept current across sessions.

## Layout

Four Bayesian arms, one shared network. `gpt2_kernels.cuh` holds everything that
is NOT the Bayesian optimizer — kernels, buffer layout, forward/backward,
dataloader, checkpointing, and the uncertainty harness. Each `train_*.cu` holds
only its own optimizer. **A fix to the network belongs in the header so it lands
in all four at once; do not fork the header per method.** `param_offsets` /
`act_offsets` are the single source of truth for buffer layout — add a tensor
there and every caller sees it.

## Uncertainty evaluation

All four arms report NLPD / ECE / aleatoric-epistemic-total through one shared
path (`uq_evaluate` in `gpt2_kernels.cuh`). Each arm supplies only a
`uq_sample_fn` — "give me draw s". Same CLI everywhere:

    <binary> eval [steps] [S] [batches] [split]     # split = val | test

Two invariants worth knowing before touching it:

- **`total == aleatoric + epistemic` is exact algebra, not an approximation.**
  It is asserted at 1e-5 in double precision and the process exits nonzero.
  A violation is an arithmetic error in `uq_fold_batch`, never a modelling
  problem. Keep the assertion — but **do not trust it further than that.** The
  identity is normalization-free (`p_bar` is the mean of the very vectors it is
  checked against), so it holds for ANY non-negative vectors: an unnormalized
  row and a stale sample buffer were both injected and both passed at 1e-15
  while the printed metrics were badly wrong. `uq_selftest` catches fold
  arithmetic first, more cheaply, at 1e-12 — but only at `V=2, S<=2`.
- **Eval loads checkpoints; it never trains.** Every new metric must be
  obtainable from `runs/*.bin` or it costs another multi-hour training run.

`train_ensemble.cu` is the control arm and its in-training output is the
reference. If a refactor changes any number it prints, that is a bug in the
refactor.

## The Python package (`uncertain/`)

`pip install -e . --no-build-isolation` compiles `uncertain/_C.cu` into
`uncertain._C`. That file `#include`s the four arm `.cu` files verbatim into ONE
translation unit and lifts what used to be their `main()`s into classes; it
implements no kernel, no optimizer and no metric of its own. Full API, design
rationale and verification checklist: `SPEC.md`.

Three conventions the package depends on. Break any of them and the binaries
still build while the module breaks, or worse, does not:

- **`main()` lives inside `#ifndef UNCERTAIN_AS_LIBRARY`.** Four `main()`
  symbols cannot coexist in one module. A NEW ARM must wrap its `main()` the
  same way and leave everything else — kernels, its `uq_sample_fn` — outside the
  guard, so the extension gets the arm's mathematics with no second copy.
- **Reach cuBLAS only through the `cublas_handle` macro.** It expands to an
  inline accessor's local static, which is one handle per linked image whatever
  the translation-unit layout. A plain `static` at file scope silently gives
  every TU its own null copy.
- **Fatal paths go through `UNCERTAIN_ABORT`,** which is `exit()` in a binary
  and `throw` in the module. A bare `exit()` in shared code kills the user's
  Python interpreter.

Building it needs an `nvcc` whose CUDA MAJOR version matches the one torch was
built against (`torch.version.cuda`) — torch refuses the build otherwise, and it
is right to. The system `nvcc` is not necessarily that one. `make NVCC=<path>`
builds the binaries with the same toolchain, which is what a timing comparison
against the wrapper requires.

## Performance measurement

`ms/step` used to be a lower bound rather than a measurement. It now has a
protocol, in the `bench_*` harness in `gpt2_kernels.cuh`:

    <binary> bench [B] [steps] [warmup]     # one arm, one batch size
    ./bench_sweep.sh <tag>                  # the whole B sweep into runs/bench/<tag>/

It is the REAL training step with marks inserted, not a copy of it — so it
cannot drift from what training does. Regions are labelled by cost class in
the first character of the name: `*` = O(P) and Bayesian, `+` = O(P) but a
deterministic trainer pays it too, unprefixed = O(B·T). Warm-up is discarded,
the statistic is a median with p25/p75, and every `bench:` line carries the
temperature and SM-clock range it was taken at. `make_tables.py` parses
`bench:` and `bench-region:` into `tab-{posterior,bandwidth,amortization,
fusion,crossover}.tex`, so those formats are pinned like the `step`/`summary:`
ones. The region lines carry the paper's leading claim, and `tab-bandwidth`
converts them
with touch counts read off the launch sites (`TOUCHES` in the generator) — the
one number there that comes from source rather than a log.

Three traps this hit, all of which produce plausible wrong numbers:

- **A `cudaMalloc` that SUCCEEDS is not proof the point is measurable.** Under
  ~384 MB of free VRAM the driver backs allocations with host memory and step
  time stops measuring the kernels — LowR at B=256 sat at 248 MB free and ran
  4x its own curve's prediction. `bench_vram_ok()` now refuses those points as
  out of range instead of reporting them.
- **Do not compare absolute ms across two sweeps.** The card's cooling fault
  means a later sweep runs 6–10 °C hotter and the IDENTICAL model region comes
  out 1–5% slower. `train_ensemble` is the drift control: its code does not
  change, so pre-vs-post on that arm is pure measurement drift.
- **Byte counts are cross-checked, not trusted.** Converting each region's
  time to achieved bandwidth must put every region in one band (235–254 GB/s
  here, aggregate 240 GB/s = the 92.4%-of-achievable ceiling). A region off
  the band means the touch count for it is wrong.

## Kernel correctness

`dev/` holds one file per kernel, each checked against a CPU reference; see
`dev/Makefile`. Three of them do not follow the `train_bnn.c` pattern because
what they test is not in `train_bnn.c`:

    make -C dev vadam block_loop     # then ./dev/{vadam_update,vadam_sample,block_loop}

`vadam_update` / `vadam_sample` include `../train_bnn_vadam.cu` and define
`UNCERTAIN_AS_LIBRARY` — the same guard the Python extension uses to compile out
an arm's `main()` — so they test the REAL kernels, not a copy that can drift.
`block_loop` gradient-checks the `L>1` block loop, which `train_bnn.c` cannot
cover at all (it is single-block).

`dev/fused.cu` is the gate on the FUSED parameter updates (`k_bbb_step_fused`,
`k_vadam_step_fused`, `k_lowrank_step_fused`): it runs each fused kernel and
the kernels it replaced on identical inputs and compares every output. It
builds TWICE —

    make -C dev fused fused_nofma && ./dev/fused && ./dev/fused_nofma

— because production flags let nvcc contract multiply-adds differently in a
fused kernel than in a standalone one, and no tolerance can separate "the
compiler rounded differently" from "the arithmetic changed". `fused_nofma`
adds `-fmad=false` and demands EXACT bit equality; that is what makes the
equivalence a result rather than a claim. **The unfused kernels must stay:**
eval draws with them, each loop primes with them, and this file checks against
them.

**Two gates that look sufficient and are not:**

- **The eval harness cannot see a training-path change.** It loads `runs/*.bin`
  and draws with the sample kernels, so a broken fused update passes it. The
  gate that works is a trajectory diff: rebuild the previous `.cu` files from
  git into a scratch tree, run both for ~100 steps in NORMAL mode (overfit is
  not reproducible, see Sharp edges) and diff the `step` lines with the `ms`
  column stripped — it differs run to run by construction.
- **`train_ensemble` must be diffed too,** step lines and both `summary:`
  lines. It is the reference this repo is calibrated against.

**Two traps these hit, which any new numerical check here will hit too:**

- **Never normalize an error by `|want|`.** Vadam's line 5 and a random
  directional derivative both cancel to near zero for some inputs, where the
  relative error against `|want|` is 100% on correct arithmetic. Normalize by
  the scale of the computation (sum of the input terms, or `||g||`), the way
  `train_bnn_lowrank.cu`'s check 3 already does for its off-diagonals.
  ULP-at-the-value's-own-magnitude is the same mistake in disguise, and it cost
  `dev/fused.cu` a false 21,475-ULP failure; it scores at
  `max(|value|, buffer RMS)` now.
- **Draw Adam's `m` and `v` TOGETHER.** They accumulate from the same
  gradient, so `v ~ m^2` and the update `m/(sqrt(v)+eps)` is O(1). Drawing them
  independently over wide log ranges manufactures states that never occur
  (m=1, v=1e-14 moves a parameter by 1e5x its own size in one step) and turns
  a 1-ULP difference into an apparent catastrophe. `adam_state()` in
  `dev/fused.cu`.
- **A float32 forward cannot resolve a single-parameter finite difference**
  at this size: the signal `2*h*g_i` is the size of the loss's own noise.
  Perturb a whole tensor along a Rademacher direction instead, with `h`
  relative to that tensor's RMS — LayerNorm gains sit at 1.0 and weights at
  0.02, so one absolute `h` leaves the gains uncertified.

## Sharp edges

- **Derive constants, don't type them.** The ELBO's KL weight is `1/N` from the
  loader, not a tuned knob; Vadam's `lambda` is `1/init_std^2`. Hardcoding the
  first cost 1.0 nats (recorded in `train_bnn_gpu.cu`'s header).
- **Vadam's sigma hides `N`.** `sigma = 1/sqrt(N*s + lambda)` uses the TRAINING
  token count, so evaluating on val/test still needs `DATA_TRAIN`'s N — see
  `loader_count`.
- **Low-rank rank is compile-time.** The rank-dependent code is templated on
  `R` so the r-loops unroll; the binary instantiates the `RANK` macro and the
  Python module instantiates a fixed list (`UNCERTAIN_LOWRANK_RANKS`). Either
  way, evaluate a checkpoint at the rank it was trained at — `ckpt_load`'s
  element-count check catches the mismatch.
- **Two things are not bit-reproducible; do not chase them as regressions.**
  `./train_bnn_lowrank check` fails check 2 on `main` itself (pre-existing, worth
  a real investigation, not caused by any refactor that reproduces it). And
  `overfit` runs diverge from themselves run to run — a last-bit difference
  amplifies in a steeply falling loss — so compare final nll and shape, not
  digits. Normal-mode runs ARE reproducible and are the regression check to use.
- **Some `.cu` file headers still quote C=768 numbers.** `train_bnn_lowrank.cu`
  says "7,077,888 params, 8.3% of 85,120,849". At the current `Config`
  (C=384, L=6) it is 10,616,832 of 11,061,531 (96%) and 44 MB/buffer.
  Recompute from the `Config` literal and `param_offsets` — never copy a count
  out of a header comment. (`train_bnn_gpu.cu`'s header was the other offender
  and now states buffer counts in units of P instead.)
- **Head size must be exactly `HS` = 64.** `HS` is a compile-time define and the
  FlashAttention kernels index q/k/v with it, so any config where `C/NH != 64`
  read past every head. `gpt2_forward` now `UNCERTAIN_ABORT`s with a message;
  before that it returned confident garbage. Small test configs must respect it — that
  is why `dev/block_loop.cu` is `C=64 NH=1`, not the `C=8` it was specified at.
- `scorecard.c` parses stdout by format string (`step %d | nll %f | ...` and
  `summary:` lines, first 3 only). Changing those formats breaks the scorecard.
- `paper/tables/make_tables.py` parses the SAME two formats plus the `eval:`
  header line, so a format change breaks the paper's tables too. It has a
  `--selftest` that pins them.

## The paper

`paper/` — see `paper/README.md` for build, table regeneration, and the
venue-template swap.

**The paper is a benchmark-suite paper: here is the suite, and here is what
was found using it.** The suite is the contribution and each finding is a
demonstration of what it surfaces, so the bar on a finding is "does this show
the instrument separates real differences", not "is this surprising". Two
framings were rejected before this one: ranking posteriors, and a mechanism
claim in which every result had to be interesting standalone. The shared CUDA
is load-bearing and the paper says so — comparisons are incomparable when each
method runs on different code, and you cannot share kernels you do not own.
The four arms are reference implementations of published methods, and the
quality numbers are kept but demoted. `docs/adding-an-arm.md` is the long form
of the paper's "Adding an arm" section and the two must not drift apart; a
suite nobody can extend is not a suite.

One rule holds everywhere: **no number is ever typed into the paper.** Cost and
quality tables are generated from `runs/` into `paper/tables/generated/`, and a
cell with no measurement behind it renders
as a loud red `TODO`, never as a plausible value. Those generated `.tex`
fragments and the figure PDF **are committed**, because CI (and any clone)
has no `runs/` and would otherwise publish a PDF of nothing but `TODO`: a
generator run with nothing measured now keeps them instead of overwriting
them, and each carries a `% generated: <commit> on <date>` stamp so a stale
one is detectable. Change a number by regenerating with `runs/` present —
never by editing a fragment. `paper/main.pdf` is tracked and rebuilt by
`.github/workflows/paper.yml` so GitHub can render it.

Four presentation choices there are load-bearing, not cosmetic;
`paper/README.md` says why. The quality and uncertainty tables carry **one row
per (arm, split) and prefer neither** — which split the paper standardises on is
an open decision, and a table that picks would settle it silently. `tab-ood.tex`
is a separate table because the OOD stream is a fixed synthetic sequence with
**no split dependence**; stating it per split implies one, and the generator
refuses to state it once if the two splits ever disagree. And the flat posterior
cost (`tab-posterior`) is stated for the **unfused** variant, because fusing puts
the Bayesian update in the same kernel as the deterministic one and only total
O(P) work stays comparable across the two. Finally `tab-crossover` decides where
folding stops paying against the **1% band the control arm reproduces itself
in** (`DRIFT_BAND`), not a round number: the control's code is identical in both
sweeps, so what it reproduces is the floor on what either can resolve, and the
generator complains on stderr if the sweeps ever drift wider than the band it
calls drift. `tab-cost.tex` is still generated as the training-run record but is
no longer `\input`: its `ms/step` predates the `bench_*` protocol that
supersedes it.

The prose is held near 4,200 words, with no em-dashes, no first person and no
bullet lists, and the count that budget refers to is

    cd paper && cat main.tex sections/*.tex | grep -v '^\s*%' | wc -w

The one open decision it deliberately does not settle is which split to
standardise on.

**The evaluation S is a rule, not a constant.** S is the first value at which
every arm free to choose one has converged, which for these four is 64: BBB is
flat by 32, Vadam and LowR are not. State the rule wherever the number would be
tempting — `paper/sections/suite.tex` and `docs/adding-an-arm.md` both carry it,
because an arm added later may converge later and must raise S rather than
inherit 64. `make_figures.py` derives it from `runs/sweep/*.log` instead of
taking it (`converged_S`, agreement to within the one printed digit
`uq_report` resolves, confirmed by a further doubling) and marks it on the
convergence figure; it leaves the figure unmarked if any sweepable arm is still
climbing at the end of the range.

The prose cites for real and `\nocite{*}` is gone, so an entry in
`references.bib` that loses its last citation silently vanishes from the
bibliography. `make -C paper` must stay clean of undefined-citation warnings.

## Maintaining this file

Keep this file for knowledge useful to almost every future agent session in this project.
Do not repeat what the codebase already shows; point to the authoritative file or command instead.
Prefer rewriting or pruning existing entries over appending new ones.
When updating this file, preserve this bar for all agents and keep entries concise.
