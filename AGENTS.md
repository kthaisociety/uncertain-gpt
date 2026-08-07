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
  Because the identity is independent of the posterior, a violation means the
  *probabilities* are wrong (unnormalized row, stale buffer, a sample callback
  that never wrote), never a modelling problem. Keep the assertion.
- **Eval loads checkpoints; it never trains.** Every new metric must be
  obtainable from `runs/*.bin` or it costs another multi-hour training run.

`train_ensemble.cu` is the control arm and its in-training output is the
reference. If a refactor changes any number it prints, that is a bug in the
refactor.

## Sharp edges

- **Derive constants, don't type them.** The ELBO's KL weight is `1/N` from the
  loader, not a tuned knob; Vadam's `lambda` is `1/init_std^2`. Hardcoding the
  first cost 1.0 nats (recorded in `train_bnn_gpu.cu`'s header).
- **Vadam's sigma hides `N`.** `sigma = 1/sqrt(N*s + lambda)` uses the TRAINING
  token count, so evaluating on val/test still needs `DATA_TRAIN`'s N — see
  `loader_count`.
- **Low-rank `RANK` is compile-time.** Build the binary at the rank the
  checkpoint was trained at; `ckpt_load`'s element-count check catches mismatch.
- `scorecard.c` parses stdout by format string (`step %d | nll %f | ...` and
  `summary:` lines, first 3 only). Changing those formats breaks the scorecard.
- `paper/tables/make_tables.py` parses the SAME two formats plus the `eval:`
  header line, so a format change breaks the paper's tables too. It has a
  `--selftest` that pins them.

## The paper

`paper/` — see `paper/README.md` for build, table regeneration, and the
venue-template swap. One rule holds everywhere: **no number is ever typed
into the paper.** Cost and quality tables are generated from `runs/` into
`paper/tables/generated/` (gitignored) and a cell with no measurement
behind it renders as a loud red `TODO`, never as a plausible value.

## Maintaining this file

Keep this file for knowledge useful to almost every future agent session in this project.
Do not repeat what the codebase already shows; point to the authoritative file or command instead.
Prefer rewriting or pruning existing entries over appending new ones.
When updating this file, preserve this bar for all agents and keep entries concise.
