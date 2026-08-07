# Resume state — paused 2026-08-06 late evening

## The run is SUSPENDED, not killed

Two processes are stopped with SIGSTOP. They still hold their VRAM and their
in-progress state:

```
56958  ./scorecard
78676  ./train_ensemble train 100000     <- member 3 of 5
```

**Resume:**   `kill -CONT 56958 78676`
**Abandon:**  `kill -9 56958 78676`  (loses Ens5's members 0-3; they are in
              VRAM only, checkpoints are written at END of training)

Paused because the GPU hit 87 C at only 46 W — a cooling fault, not a load
problem. Card is going in for service (likely dust / thermal paste).

If the machine is rebooted before resuming, the run is gone. That is fine —
see "what is lost" below.

## What is DONE and safe on disk

```
runs/bbb_C384_L6_100k_mu.bin      BBB trained 100K steps
runs/bbb_C384_L6_100k_rho.bin
runs/vadam_C384_L6_100k_mu.bin    Vadam trained 100K steps
runs/vadam_C384_L6_100k_s.bin     (sigma is DERIVED from s, not stored)
runs/BBB_train.log      runs/BBB_overfit.log
runs/Vadam_train.log    runs/Vadam_overfit.log
runs/Ens5_overfit.log
runs/results.csv                  appended per run, survives across sessions
```

## What is NOT done

```
Ens5  train   member 3 of 5, in progress   ~1.2 h remaining
LowR8 train   not started                  ~1.5 h
```

Rerun just those two after the repair:

```bash
cd ~/github/CUDA/gpt2
./train_ensemble   train 100000 2>&1 | tee runs/Ens5_train.log
./train_bnn_lowrank train 100000 2>&1 | tee runs/LowR8_train.log
```

or `make test` to redo everything (~4.1 h; BBB and Vadam would be recomputed).

## Current configuration

```
model     C=384  NH=6  L=6  V=27      11,061,531 params, 96% transformer
data      text8, 90M train / 5M val / 5M test tokens   params:data = 1 : 8.1
BBB       0.53 GB    17.68 ms/step
Vadam     0.53 GB    13.33 ms/step
LowR r=8  1.84 GB    52.07 ms/step
Ens5      ~1.0 GB    13.05 ms/step per member, x5
```

## THE OPEN WORK, in priority order

1. ~~**Shared eval harness.**~~ WRITTEN, NEVER EXECUTED (branch
   `fm/ug-eval-harness`, authored while the card was out for service). The
   decomposition now lives in gpt2_kernels.cuh as "S samples from a per-arm
   sample_fn"; all four arms have an `eval` mode that loads a checkpoint and
   reports NLPD / ECE / aleatoric-epistemic-total on DATA_VAL or DATA_TEST:

   ```bash
   ./train_bnn_gpu     eval 100000 8 64 val
   ./train_bnn_vadam   eval 100000 8 64 val
   ./train_bnn_lowrank eval 100000 8 64 val     # build at the ckpt's rank
   ./train_ensemble    eval 100000            # S fixed at E=5
   ```

   NOT ONE LINE OF IT HAS BEEN COMPILED OR RUN. First job when the GPU is
   back: `make` all four, then `./train_ensemble train 100000` and diff its
   summary lines against runs/Ens5_*.log — they must be identical, since the
   refactor is supposed to change nothing the control arm prints.

   Only BBB and Vadam have checkpoints, so only those two can be evaluated
   today; Ens5 and LowR8 need their training runs finished first (see above).

2. **Two unvalidated code paths.**
   - The L>1 block loop has no CPU reference (train_bnn.c is one block). Only
     evidence is that overfit still reaches 0.0070 at L=6. Proper test: finite
     -difference gradient check on a tiny multi-block config (C=8, L=2, T=4).
   - Vadam's k_vadam_update and k_vadam_sample have NO correctness check at
     all — written from pseudocode, never gradient-checked, sampler never
     distribution-tested. Every other arm has these. LowR4's five checks are
     the pattern to copy.

3. **Sigma sweep.** Loss currently tracks sigma monotonically across arms, so
   the experiment cannot yet distinguish "better posterior" from "less
   injected noise". Sweep sigma within each arm and compare CURVES, not
   points. Runs against checkpoints — no retraining.

4. Rank sweep {4, 8, 16} for LowR — now affordable at C=384 (r=8 is 1.84 GB).
5. Batch-size amortization curve: Bayesian cost is fixed per step, model cost
   scales with tokens, so overhead should fall sharply as B grows.
6. Fused parameter update (bottleneck A) — the remedy, not just the measurement.

## Findings from this session, already written into the source files

- **KL weight must be DERIVED, not tuned.** Hardcoded 1e-3 vs the correct
  1/N = 1e-6 cost 1.0 nats and made Bayesian methods look categorically worse
  than they are. Full sweep recorded in train_bnn_gpu.cu's header. This is
  bottleneck E measured.
- **Vadam's failure was the corpus, not the code.** sigma = 1/sqrt(N*s+lambda)
  puts N inside the formula, so 1M tokens meant the data supplied 0.04% of the
  precision for a typical parameter. At 100M it is ~4%, and 98% for the
  best-informed. Fixed by switching datasets; no code change.
- **Direct sigma blows up under a scale-free optimizer.** AdamW's step is ~lr
  regardless of gradient size, so an unconstrained dvec at init 1e-3 moves
  100%/step. softplus(rho) compresses it multiplicatively, which is why BBB
  never had the problem. LowR needed softplus + a separate variance lr.
- **Shared-memory row strides must not be a multiple of 32 words.** Padding
  [BC][HS] to [BC][HS+1] took attention_backward from 1336 ms to 39 ms.
- **92.4% of achievable bandwidth is the ceiling** on this card, hit by four
  structurally unrelated kernels. For that class of op the kernel is finished
  when it is correct; the only remaining lever is moving fewer bytes.

## Before the machine goes in

- [ ] `git commit` — 158 files uncommitted, last commit predates this entire
      project. Everything (GPU port, 17 dev kernels, FlashAttention, four arms)
      exists only on this disk.
- [ ] back up `runs/` off-machine — checkpoints are ~700 MB, not in git, and
      represent hours of compute.
