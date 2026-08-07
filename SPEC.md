# `uncertain` — specification

The C/CUDA Bayesian GPT-2 trainer in this repository, importable from Python as
a PyTorch extension.

This package was written under a no-execution instruction and has since been
**built and run**. Every claim below carries a label saying which:

| Label | Meaning |
|---|---|
| **[measured]** | Verified by execution. §7 gives the command and the output. |
| **[source]** | Reasoned by reading the code. Not run. |
| **[unverified]** | Neither. Belongs to §8, which is now short. |

The measurements were taken on the machine described in §7.0 — one RTX 2060
(sm_75, 6 GB, Max-Q, WSL2). **Numbers from any other card are that card's
numbers.** §6 is the timing result; §8 is what is still open.

The single most important section is **§7, the verification checklist**. It is
written for someone who has never seen this repository, and every step in it now
records what it actually produced here.

---

## 1. What this is, in one paragraph

`uncertain` moves the existing trainer across a language boundary. It ports
nothing. `uncertain/_C.cu` `#include`s the four arm `.cu` files verbatim, and
the only thing those files do differently is compile out their `main()`.
Everything that used to be in `main()` — the buffers, the loop body, the loader,
and the lifetime of all of it — is lifted into a C++ class that Python owns. The
kernels, the four optimizers, the forward/backward, the dataloader, the
checkpoint format and the uncertainty harness are the same code, launched in the
same order, with the same arguments. **[source]**

The standalone binaries and every `make` target keep working, and their output is
unchanged: the four binaries built from the pre-change commit and from this
branch print byte-identical output at the default configuration, apart from
digits where the pre-change binary already differs from *itself* run to run.
**[measured, §7.1]**

---

## 2. The public API

Import: `import uncertain`. Requires `torch` and an NVIDIA GPU at run time,
`nvcc` at install time. fp32 only.

### 2.1 `uncertain.Trainer`

```python
uncertain.Trainer(
    method,                    # "bbb" | "vadam" | "lowrank" | "ensemble"
    data,                      # str: path to an llm.c-format .bin
    *,
    val_data=None,             # str | None
    n_layer=6,                 # int
    n_head=6,                  # int
    n_embd=384,                # int   n_embd // n_head must == 64
    vocab_size=None,           # int | None -> the data file's header
    batch_size=4,              # int   B
    seq_len=64,                # int   T
    max_seq_len=1024,          # int   width of the learned position table
    lr=1e-4,                   # float
    beta1=0.9, beta2=0.999,    # float
    eps=1e-8,                  # float
    weight_decay=0.0,          # float
    init_std=0.02,             # float
    seed=20240520,             # int   Philox key for weight sampling
    device=None,               # int | None -> torch.cuda.current_device()
    fixed_batch=False,         # bool  the binaries' `overfit` mode
    validate_data=True,        # bool  scan the .bin for out-of-vocab ids
    **arm_specific,            # see below; wrong-arm keywords raise TypeError
)
```

Arm-specific keyword arguments. Passing one that does not belong to `method`
raises `TypeError` rather than being silently ignored:

| `method` | keyword | default | meaning |
|---|---|---|---|
| `bbb` | `sigma_prior` | `0.1` | prior std |
| `bbb` | `kl_weight` | `None` | `None` → derive `1/N`; `0` disables the prior term |
| `bbb` | `rho_init` | `-5.0` | `softplus(-5) ≈ 0.0067` |
| `vadam` | `lam` | `None` | `None` → derive `1/init_std**2` (= 2500 at 0.02) |
| `vadam` | `s0` | `0.0` | second-moment init; 0 → step-1 sigma is exactly the prior |
| `lowrank` | `rank` | `8` | must be in `uncertain.supported_ranks()` |
| `lowrank` | `lam` | `100.0` | prior precision |
| `lowrank` | `kl_weight` | `None` | as `bbb` |
| `lowrank` | `lr_var` | `1e-5` | separate lr for the variance group (`A`, `rhod`) |
| `lowrank` | `a_init` | `1e-4` | scale of `A` at init |
| `lowrank` | `dvec_init` | `1e-3` | `softplus(rhod)` at init |
| `ensemble` | `seeds` | `[1337, 2337, 3337, 4337, 5337]` | read from `train_ensemble.cu` |
| `ensemble` | `steps_per_member` | `100` | where one member ends and the next begins |

`kl_weight=None` with `fixed_batch=True` resolves to `0`, matching the binaries'
overfit mode, which runs with the prior term off. **[source]**

#### Methods

| Method | Returns | Notes |
|---|---|---|
| `step()` | `float` — mean NLL over the batch, nats | One training step. Blocks, as the binaries do. |
| `stats()` | `dict`: `kl`, `sigma_mean`, `sigma_min`, `sigma_max` | The arm's stats kernel + host reduction. **Not** run inside `step()`. Zeros for `ensemble`, which samples nothing. |
| `log_line(nll)` | `str` | The binaries' per-step line, the format `scorecard.c` parses. |
| `evaluate(batches=2, samples=None, split=None, full=False)` | `EvalResult` or `dict` | See §2.2. |
| `parameters()` | `dict[str, torch.Tensor]` | Zero-copy CUDA views of the variational parameters. |
| `buffers()` | `dict[str, torch.Tensor]` | Zero-copy views of optimizer state, gradients, the current weight sample. |
| `named_tensors(which)` | `dict[str, torch.Tensor]` | Named, correctly-shaped views into one full-size buffer, e.g. `t.named_tensors("mu")["h0.attn.qkv.weight"]` → `(C, 3C)`. |
| `save(tag=None)` | `None` | `ckpt_save` into `./runs`, the binaries' format and naming. |
| `load(steps, tag=None)` | `int` | `ckpt_load` from `./runs`. Parameters only — see §2.3. |

#### Read-only properties

`step_idx`, `last_step_ms`, `n_params`, `n_act`, `train_tokens`, `config`
(dict), `device`, `total_steps` (`-1` = open-ended), `hyperparams` (dict), and
the arm-specific `kl_weight`, `lam`, `rank`, `bayesian_params`, `members`,
`current_member`, `seeds`. Arm-specific ones raise `AttributeError` on the wrong
arm.

What each arm exposes in `parameters()` / `buffers()`:

| arm | `parameters()` | `buffers()` |
|---|---|---|
| `bbb` | `mu`, `rho` | `w`, `dw`, `dmu`, `drho`, `m_mu`, `v_mu`, `m_rho`, `v_rho` |
| `vadam` | `mu`, `s` | `m`, `w`, `dw` |
| `lowrank` | `theta`, `A`, `rhod` | `w`, `dw`, `m`, `v`, `mA`, `vA`, `md`, `vd`, `z` |
| `ensemble` | `member0` … `memberE-1` | `m`, `v`, `dw` |

### 2.2 `evaluate`

```python
nlpd, ece, ale, epi = t.evaluate()                       # EvalResult, 4 fields
d = t.evaluate(batches=64, samples=8, split="val", full=True)   # dict
```

This calls `uq_evaluate` in `gpt2_kernels.cuh` — the harness all four arms
already share — with this arm's own `uq_sample_fn`. The extension contributes
buffers and nothing else, so a number computed here and the same number computed
by `./train_bnn_gpu eval` come off the same arithmetic on the same draws.
**[source]**

Per token, in nats (ECE is a probability):

```
nlpd       -log p_bar(y)          p_bar = (1/S) sum_s p_s   (PROBABILITIES mixed)
ece        expected calibration error, 15 bins on argmax confidence
aleatoric  (1/S) sum_s H[p_s]
epistemic  (1/S) sum_s KL(p_s || p_bar)
```

`total = aleatoric + epistemic` is exact algebra for any `S` and any sampler.
The harness asserts it at `1e-5` and `evaluate()` **raises** `RuntimeError` if it
fails — a violation can only mean the predictive probabilities are wrong. The
same run also sweeps an OOD set of uniform-random token ids; `full=True` returns
`epistemic_ood`, which must exceed `epistemic`.

`full=True` keys: `nlpd`, `ece`, `aleatoric`, `epistemic`, `epistemic_ood`,
`total`, `nlpd_single`, `identity_err`, `positions`, `positions_ood`, `samples`,
`samples_fixed_by_arm`, `split`.

**`samples` is ignored for `ensemble`**, whose S *is* its trained member count;
a `RuntimeWarning` says so. Comparing arms at equal S equalizes inference cost,
not the cost of obtaining the samples. Do not report it as if it did.

`split` is a label carried onto every number, because the harness refuses to
leave it unstated. Default: `"val"` when the Trainer was given `val_data`,
`"train-tail"` when it was not — in that case evaluation reads the training
stream from where training has got to, which is text training has not seen but
is *not* a held-out split.

Evaluating does not disturb training: the data cursor and the current batch are
both saved and restored, and the draws come from `UQ_EVAL_SEED`, a Philox stream
disjoint from every training stream. **[source]**

### 2.3 Module-level

| Name | Meaning |
|---|---|
| `uncertain.selftest() -> int` | `uq_selftest`: three cases with known answers, checked to 1e-12. **Pure host arithmetic — no GPU, no data, no model.** Returns the failure count. |
| `uncertain.supported_ranks() -> list[int]` | `[1, 2, 4, 8, 16, 32]` as shipped. |
| `uncertain.default_ensemble_seeds() -> list[int]` | Read from `SEEDS` in `train_ensemble.cu`. |
| `uncertain.load_checkpoint(path) -> (ndarray, meta)` | Reads the 8-int-header format `ckpt_save` writes. |
| `uncertain.read_bin_header(path) -> dict` | `magic`, `version`, `tokens`, `vocab`. |
| `uncertain.METHODS`, `METHOD_SOURCE` | The four arms and the file each lives in. |
| `uncertain.HEAD_SIZE` | 64. `n_embd // n_head` must equal it. |
| `uncertain.MAX_ENSEMBLE_MEMBERS`, `ECE_BINS` | 5, 15. |
| `uncertain.DATA_TRAIN`, `DATA_VAL`, `DATA_TEST` | The paths compiled into the header. |

`load()` restores **parameters only**. AdamW moments were never in the
checkpoint format, so this puts a model in place to evaluate; it does not resume
a run. **[source]**

---

## 3. Method → source → algorithm

| `method` | Source file | Algorithm | Posterior over |
|---|---|---|---|
| `"bbb"` | `train_bnn_gpu.cu` | Bayes-by-Backprop (Blundell et al. 2015). `mu` and `rho` both free; `sigma = softplus(rho)` learned by backprop through the reparameterization; prior enters as a closed-form KL gradient. | all parameters |
| `"vadam"` | `train_bnn_vadam.cu` | Vadam (Khan et al. 2018). `sigma = 1/sqrt(N*s + lambda)` **derived** from Adam's second moment, never stored or learned; the prior is the `lambda*mu/N` term in the momentum update. | all parameters |
| `"lowrank"` | `train_bnn_lowrank.cu` | Low-rank + diagonal covariance. `Sigma = A A^T + diag(dvec^2)`, never formed; one shared `z ~ N(0, I_r)` per layer per pass couples the weights; KL via Woodbury with an `r x r` Cholesky. | **the four GEMM matrices only** — state this when comparing |
| `"ensemble"` | `train_ensemble.cu` | Deep ensemble (Lakshminarayanan et al. 2017). E independent deterministic models differing only in init seed. The control arm. | none — uncertainty is disagreement |

The kernels behind each are the arm file's own, unchanged. `uncertain` adds no
algorithm. **[preproc]** — the preprocessed library-mode output contains exactly
one definition of each shared kernel and all four arms' draw callbacks.

---

## 4. Design decisions

### 4.1 Why the trainer is the boundary, not the tensor op

The obvious PyTorch-shaped design is one `torch.autograd.Function` per kernel,
composable like any other op. That would destroy the thing this package exists
to ship.

`gpt2_forward` / `gpt2_backward` in `gpt2_kernels.cuh` are a single fused pass
over two flat slabs: one parameter vector and one activation vector, sliced by
`param_offsets()` / `act_offsets()`. Concretely, what decomposition would cost:

- **The slab becomes tensors.** Every op currently addresses `params + PBLK(p,
  l, qw)` — a pointer into one allocation. Autograd ops take and return
  `Tensor`s, so either every slice becomes a view whose lifetime autograd
  manages, or the parameters get copied. The Bayesian arms then lose the thing
  that makes them affordable: `k_bayes_sample` writes the whole 11M-element
  weight vector in **one** kernel launch, and `k_adamw` updates it in one more.
  Per-tensor, that is ~150 launches each.
- **The activation slab becomes allocations.** The backward pass relies on
  `+=` into pre-zeroed gradient buffers at known offsets — that is why one
  `cudaMemset` over `grads_acts` replaces per-op zeroing. Autograd would
  allocate and free per op, per step.
- **FlashAttention would be split from its neighbours.** `k_attention_forward`
  reads `qkv` in the layout `matmul_forward` just wrote. An autograd boundary
  between them is a materialization point.
- **Philox regeneration would break.** `eps` is never stored: the backward
  kernel regenerates the exact `eps` the forward used from `(seed, step, index)`.
  That works because forward and backward are one step with one step counter.
  Under autograd, "the step" is not a thing either side can name.

So the Python boundary is `step()`. **[source]**

This is also why there is no `nn.Module` wrapper and no optimizer protocol: the
arms *are* optimizers, and the update is fused into the same step.

### 4.2 Why `from_blob` rather than a copy

`parameters()` returns `torch::from_blob(ptr, {numel}, options)` views of the
live device buffers, with **no deleter**.

- **No device copy.** `mu` at the paper's configuration is 11,061,531 floats
  (44 MB). A checkpointing loop or an evaluation script that copies it every
  call pays that repeatedly, for data that has not moved.
- **The layout does not change.** The kernels index one flat vector by offsets
  from `param_offsets()`. A copy into per-tensor storage would either change
  that layout or need translating back, and the layout is the single source of
  truth every arm slices from.
- **Writes go through.** `t.parameters()["mu"].copy_(...)` writes the buffer the
  next `step()` reads. That is what makes `load()`, checkpoint surgery and
  sigma sweeps possible from Python at all.

**The cost, stated plainly: the tensor is a view and does not own or extend the
Trainer's memory.** Using one after the Trainer is garbage-collected reads freed
device memory. Clone anything you intend to keep. The no-deleter choice is
deliberate — a deleter would mean torch calling `cudaFree` on a pointer the
Trainer also frees. **[source]**

### 4.3 The `cublas_handle` global, and what would have gone wrong

`gpt2_kernels.cuh:87` was:

```c
static cublasHandle_t cublas_handle;
```

For a one-file binary that is exactly right: one translation unit, one handle,
`cublasCreate` in `main()`.

It stops being right the moment two arms are compiled into the same module.
`static` at file scope is **internal linkage**: every translation unit including
the header gets its own private copy. Four objects → four handles, of which at
most one is ever created. `matmul_forward` and `matmul_backward` — i.e. every
GEMM in the forward and backward pass — read whichever copy their own TU owns.
Nothing in the type system catches it and no symptom points at the header.

Two things were done:

1. **The header now routes every use through an inline accessor.**

   ```c
   inline cublasHandle_t &cublas_handle_ref(void) {
       static cublasHandle_t h = nullptr;
       return h;
   }
   #define cublas_handle (cublas_handle_ref())
   ```

   C++ guarantees a single instance of an inline function's local static across
   the linked image, so this is one handle per binary or shared object whatever
   the TU layout is. The macro keeps every existing call site spelled the same —
   `cublasCreate(&cublas_handle)`, `cublasDestroy(cublas_handle)`, and the two
   GEMM wrappers — so no arm file changed. **[preproc]** — the preprocessed
   output has zero unexpanded occurrences of the bare name in either mode.

2. **The extension compiles all four arms into ONE translation unit**, so the
   problem also cannot arise by construction. This was forced anyway: the
   header's fifteen `__global__` kernels have external linkage, and four objects
   each including it would be four definitions of `k_attention_forward` — a link
   error, or, had they been made `static` to dodge it, four silent copies of
   every kernel in the cubin.

Ownership is a **refcounted process singleton**: created by the first Trainer,
destroyed with the last. A Trainer cannot own it outright, because two Trainers
can be alive at once and the second must not be left holding a handle the first
destroyed.

**It is deliberately not torch's handle** (`at::cuda::getCurrentCUDABlasHandle`).
That handle follows torch's *current stream*, while every `<<<...>>>` launch in
this codebase goes to the legacy default stream. Borrowing it would put the
GEMMs on one stream and the elementwise kernels on another with nothing ordering
them. Today that is invisible, because torch's default current stream *is* the
legacy default stream — it would become a live data race the first time a user
wrote `with torch.cuda.stream(s):`. Owning the handle keeps all of it on the
default stream, exactly as the binaries have it. **[source]**

### 4.4 Model geometry at run time

Verified by reading all four arms: `Config` (`B, T, C, V, NH, maxT, L`) is a
plain runtime struct; `cfg.L` drives every block loop; `param_offsets` /
`act_offsets` compute every offset from it; `NBLOCKS` only supplies the default
in each `Config` literal inside `main()`. So geometry needed no redesign to
become constructor arguments. **[source]**

Two things are genuinely compile-time and are enforced as errors, not silently
worked around:

- **`n_embd // n_head` must be 64.** `HS` is a `#define` and
  `k_attention_forward` holds a whole head in `float qr[HS]` registers.
- **The low-rank rank.** See §4.5.

### 4.5 Low-rank rank as a run-time argument

`RANK` was a `#define` used for `__shared__ float zs[RANK]`, `float ar[RANK]`
registers and `#pragma unroll` bounds. Reading it from a variable would move
those arrays to local memory and kill the unroll — measurably the wrong trade in
an arm that already costs 52 ms/step.

Instead the rank-dependent definitions are now `template <int R>`: the kernels,
`chol_logdet_inv`, `lowr_build_layers` and `lowr_draw`. Inside an instantiation
`R` is a compile-time constant exactly as the macro was, so every unrolled
iteration survives. The binary instantiates `RANK`; the module instantiates the
list in `UNCERTAIN_LOWRANK_RANKS` (`1, 2, 4, 8, 16, 32`) and dispatches on the
constructor argument. An unsupported rank raises with the list and the
instruction to rebuild. **[source]**

### 4.6 The ensemble's `steps_per_member`

The one place an arm's structure shows through the API. `train_ensemble.cu`
trains member 0 to completion, then member 1, reusing **one** pair of AdamW
moment buffers. Interleaving the members under a single `step()` would need E
pairs — and this arm's memory footprint is a reported result. So `step()` walks
the members in the binary's order, resetting the moments and `ld.pos` at each
boundary, and `steps_per_member` (default 100, the binary's default) says where
the boundaries are. `total_steps == members * steps_per_member`; `step()` beyond
that raises. **[source]**

### 4.7 Errors, and not killing the interpreter

`CUDA_CHECK`, `CUBLAS_CHECK` and the loader called `exit()`. In a binary that is
correct. In a Python extension it terminates the user's interpreter with no
traceback and no chance to save anything. Both paths now go through
`UNCERTAIN_ABORT`, which is `exit(EXIT_FAILURE)` in a binary and
`throw std::runtime_error` in the module. **Every `printf` is unchanged, so the
binaries' output and exit status do not change.** **[preproc]** — the
preprocessed standalone output has zero `throw`; the library-mode output has
zero `exit`.

### 4.8 Init reproducibility

Each arm's `main()` seeds `srand(1337)` (`srand(seeds[e])` per ensemble member)
and fills with an inline Box–Muller loop. `host_normal_fill` in `_C.cu`
reproduces that loop character for character and is called in the same order, so
a Trainer's starting weights should be the binary's starting weights. The
ensemble initializes **all** members in the constructor rather than lazily,
which is equivalent because each member's init re-seeds and nothing else draws
`rand()` in between. **[source] — see §8; this is a claim about bit-identical
floating-point output that has not been run.**

---

## 5. Build and packaging

```
pyproject.toml     metadata; setuptools backend
setup.py           CUDAExtension over the single source uncertain/_C.cu
uncertain/_C.cu    the one translation unit
uncertain/__init__.py   the Python API
```

- `torch.utils.cpp_extension.CUDAExtension`, compiled on install. Requiring
  `nvcc` is an accepted trade; there is **no** pure-PyTorch fallback path and
  there should never be one.
- Flags match the `Makefile`: `-O3`, `--use_fast_math`, host `-O3`, and
  `-lcublas`. `--use_fast_math` is not cosmetic — the kernels were validated
  against the CPU reference with it, and the low-rank arm's bitwise checks
  depend on host and device agreeing on which transcendentals are in play.
- **`sm_75` is the default arch**, matching `ARCH` in the `Makefile`. Override
  with `UNCERTAIN_CUDA_ARCH=8.6` or `TORCH_CUDA_ARCH_LIST="7.5;8.6"`. `setup.py`
  sets `TORCH_CUDA_ARCH_LIST` when neither is given, which also stops torch
  probing the local GPU for arch flags — so the package builds on a machine
  whose card is unavailable.
- `--no-build-isolation` is required, because `setup.py` imports the installed
  torch. Declaring `torch` as a build requirement would download a second copy
  into an isolated environment and link the extension against the wrong one.
- The package sits **alongside** the existing sources. `Makefile`,
  `scorecard.c`, `dev/` and the binaries are untouched in behaviour.

### What changed in the existing files

| File | Change |
|---|---|
| `gpt2_kernels.cuh` | `UNCERTAIN_ABORT` macro; `cuda_check_`/`CUBLAS_CHECK`/`loader_init`/`loader_count`/`uq_split_path` route fatal paths through it; `cublas_handle` becomes an inline accessor plus a same-named macro. |
| all four `train_*.cu` | `main()` wrapped in `#ifndef UNCERTAIN_AS_LIBRARY`. Nothing else. |
| `train_bnn_lowrank.cu` | Rank-dependent definitions templated on `R` (§4.5); `main()`'s inline Bayesian-layer list collapsed onto `lowr_build_layers<RANK>`, which its own comment asked for. |

No arm's mathematics was changed.

---

## 6. Cost — the measurement

Per-step cost was measured on one RTX 2060 (§7.0), 3 runs per side per arm,
sampling the same step indices the binary prints and dropping each run's step 1
(first launch of every kernel, ~3x steady state on both sides). Both sides read
the **same instrument**: the CUDA event pair around the whole step body that the
binaries already had. `TrainerBase::tic/toc` is that pair, not a new one, and
`Trainer.last_step_ms` is what it returns. **[measured]**

| arm | binary median | wrapper median | delta | |
|---|---|---|---|---|
| `bbb` | 17.440 ms | 17.580 ms | **+0.140 ms** | +0.80 % |
| `vadam` | 13.900 ms | 13.873 ms | **−0.027 ms** | −0.19 % |
| `ensemble` | 13.490 ms | 13.467 ms | **−0.023 ms** | −0.17 % |
| `lowrank` (r=8) | 53.145 ms | 52.905 ms | **−0.240 ms** | −0.45 % |

n = 30 per cell (150 for `ensemble`, which prints per member).

**What this does and does not say.** The spread across arms straddles zero and
is smaller than each arm's own run-to-run spread — on `bbb`, the arm with the
largest delta, the binary's own p10–p90 is 17.26–17.88 ms, a band wider than the
0.140 ms difference. So the honest statement is: **the wrapper's per-step cost is
below what this instrument can resolve on this card.** It is *not* "zero
overhead", and the `bbb` row is not evidence of a 0.8 % regression any more than
the `lowrank` row is evidence of a 0.45 % speed-up.

An argument for why that is the expected result, kept clearly separate from the
measurement above: `step()` issues the same kernel launches, in the same order,
with the same arguments as the binary's loop body, and adds per step one
`cudaSetDevice` on the already-current device, one virtual call, and the pybind
entry/exit — all host-side and sub-microsecond against a 13–53 ms step. **[source]**

A Python-side wall clock around `t.step()` came out ~0.7 ms *below* the CUDA-event
figure, consistently across runs. That is a property of the two clocks, not a
negative overhead; it means the pybind and GIL cost is not separable from event
timing noise here and this instrument cannot bound it more tightly. **[measured]**

Two places where the wrapper does more work than the binary, both outside
`step()` and both deliberate:

- `stats()` is a separate call, so the KL/sigma kernel and its four D2H copies
  are *not* in the step. The binaries run them every tenth step. A caller who
  calls `stats()` every step will be slower than the binary; one who calls it
  every tenth step does the same work.
- `validate_data=True` scans the token file once at construction (~90 M tokens,
  a few seconds). Pass `validate_data=False` in a benchmark.

VRAM over an idle context, one Trainer plus one step, read from `nvidia-smi`
(includes torch's own context, so treat it as an upper bound): `bbb` 618 MiB,
`vadam` 398 MiB, `ensemble` 530 MiB, `lowrank` r=8 1862 MiB. **[measured]**

---

## 7. Verification checklist

Mechanical, in order. Written for someone who has never seen this repository.
Each step records what it produced on the machine in §7.0 — so a re-run is a
comparison, not an exploration.

### 7.0 Prerequisites

```bash
nvcc --version            # CUDA toolkit must be present
gcc --version             # must be a version your CUDA accepts
python3 -c "import torch; print(torch.__version__, torch.version.cuda)"
nvidia-smi                # a working GPU, for everything from 7.3 on
```

Reference machine for every number in this document: RTX 2060 Max-Q (sm_75,
6 GB), driver 581.15, WSL2, gcc 13.3.0, Python 3.12, torch 2.11.0+cu130,
nvcc 13.0.

> **`nvcc`'s CUDA major version must match the one torch was built against.**
> `torch.utils.cpp_extension` raises on a major mismatch, and it is right to:
> the extension and torch would otherwise load two CUDA runtimes into one
> process. On the reference machine the *system* nvcc was 12.0 while torch was
> `+cu130`, and the build fails until a CUDA 13 `nvcc` is on `CUDA_HOME`. Check
> the pair before anything else:
>
> ```bash
> python3 -c "import torch; print('torch wants CUDA', torch.version.cuda)"
> nvcc --version | grep release
> ```
>
> They must agree on the major version. If they do not, install the matching
> toolkit and point `CUDA_HOME` at it. **[measured — this is what actually blocked
> the first build here]**

> **gcc 13 is fine.** An earlier draft of this document warned that CUDA's
> `host_config.h` rejects `__GNUC__ > 12`. It does not, for the CUDA 12.0 and
> 13.0 toolkits on this machine: both compiled every target with gcc 13.3.0 and
> no `-allow-unsupported-compiler`. The escape hatch in `setup.py`
> (`UNCERTAIN_NVCC_EXTRA_FLAGS`) stays for toolchains that do complain, but it
> was not needed here. **[measured]**

### 7.1 The standalone binaries still build and behave

This is the regression that matters most: the binaries are how the experiments
are run today.

```bash
make clean && make                       # expect: train_bnn and train_bnn_gpu, no errors
make train_bnn_vadam train_ensemble train_bnn_lowrank    # expect: no errors
make train_bnn_lowrank R=4               # expect: no errors (template instantiates at 4)
```

```bash
./train_bnn_lowrank check
```
**Got:** checks 1, 3, 4, 5 PASS; **check 2 FAILS** with 2 mismatches, and the
run ends `SOME CHECKS FAILED`.

> **Check 2 fails on `main` as well, and this branch did not cause it.** The
> same binary built from the pre-change commit `c7ba8a0` — with the *system*
> CUDA 12.0 toolchain, i.e. the one used to run the experiments — prints
> character-for-character the same failure. The templating is exonerated by a
> stronger test than the checks: the whole `check` output is byte-identical
> before and after. **This is a pre-existing defect in `main` and is worth its
> own investigation; it is out of scope here.** **[measured]**
>
> ```bash
> diff <(git stash list >/dev/null; ./train_bnn_lowrank check) baseline_check.txt
> ```

```bash
./train_bnn_gpu overfit 200 | tail -3
./train_bnn_vadam overfit 200 | tail -3
./train_ensemble overfit 200 | tail -3
./train_bnn_lowrank overfit 200 | tail -3
```
**Got:** final `nll` 0.0172 (bbb), 0.3684 (vadam), 0.0123 (ensemble), 0.0137
(lowrank).

> **Do not diff overfit runs against a stored reference.** The binaries are not
> bit-reproducible run to run in this mode, on `main`: two consecutive runs of
> the *pre-change* `train_ensemble overfit 200` diverge from step 60 (1.6934 vs
> 1.6935) and drift to 0.3587 vs 0.3645 by step 100. An overfit run's loss falls
> steeply, so a last-bit difference amplifies. Compare the *final* nll and the
> shape, not the digits. **[measured]**

In normal (non-overfit) mode the binaries *are* reproducible, and that is the
regression check that carries weight:

```bash
./train_bnn_gpu | sed -E 's/\|[[:space:]]*[0-9.]+ ms$/| <ms>/' > new.txt
# ...same, from a checkout of the pre-change commit...
diff before.txt new.txt
```
**Got:** byte-identical for `bbb`, `vadam` and `lowrank`. `ensemble` differs in
one field only — `identity_err 2.2e-15` vs `1.8e-15` — and the pre-change binary
alternates between exactly those two values across its own runs. That field is
the double-precision residual of an exact algebraic identity, ten orders of
magnitude below the 1e-5 gate. **[measured]**

```bash
make test                                # the full scorecard, ~4.1 h
```
**Expect:** exit code 0. **Not run** — see §8.

### 7.2 Build the Python package

```bash
pip install -e . --no-build-isolation
```
**Got:** a clean build — one nvcc invocation on `uncertain/_C.cu`, no warnings,
no errors, ~4 minutes, producing `uncertain/_C.cpython-312-x86_64-linux-gnu.so`
(1.2 MB). The only thing that stood in the way was the CUDA-major mismatch in
§7.0. **[measured]**

### 7.3 Import and shared-math self-test — **no GPU required**

```bash
python3 -c "import uncertain; print(uncertain.__version__, uncertain.supported_ranks())"
```
**Got:** `0.1.0 [1, 2, 4, 8, 16, 32]` **[measured]**

```bash
python3 -c "import uncertain; assert uncertain.selftest() == 0; print('selftest OK')"
```
**Got:** `selftest OK` — 0 failures. This runs `uq_selftest`: three cases whose
answers are known on paper, checked to 1e-12, in pure host arithmetic. It proves
the module built, links and imports, and that the shared decomposition is right,
without touching a GPU. **[measured]**

```bash
python3 -c "
import uncertain
print(uncertain.METHOD_SOURCE)
print(uncertain.default_ensemble_seeds())
print(uncertain.HEAD_SIZE, uncertain.MAX_ENSEMBLE_MEMBERS, uncertain.ECE_BINS)"
```
**Got:** the four arms mapped to their `.cu` files; `[1337, 2337, 3337, 4337,
5337]`; `64 5 15`. **[measured]**

### 7.4 Timing parity — the measurement the package rests on

Run it first among the GPU checks, and run it on an otherwise idle card.

```bash
./train_bnn_vadam 2>&1 | grep '^step'           # note the ms/step column
```

```python
import uncertain
t = uncertain.Trainer("vadam", data="data/text8/text8_train.bin",
                      validate_data=False)
for _ in range(20):                    # warm up
    t.step()
ms = []
for _ in range(200):
    t.step()
    ms.append(t.last_step_ms)
ms.sort()
print("median ms/step:", ms[len(ms)//2])
```
**Got:** the four-arm table in §6 — deltas from −0.45 % to +0.80 %, straddling
zero and inside each arm's own run-to-run spread. `last_step_ms` comes from the
*same* CUDA event pair the binary prints, so this compares like with like.
**[measured]**

Two cautions for whoever re-runs this:

- **Build both sides with the same `nvcc`.** The comparison above used one
  toolchain for the extension and for `make`, precisely so the only difference
  is the wrapper. `make NVCC=/path/to/nvcc` does it without editing the
  Makefile. Comparing an extension built with CUDA 13 against a binary built
  with CUDA 12 measures the compilers, not the boundary.
- **Watch the card's temperature.** On the reference machine (a Max-Q 2060 with
  a known cooling fault) the die went 59 °C → 84 °C over a benchmark sweep, and
  a thermally drifting card will happily manufacture a several-percent
  "difference" between whichever side you time second. Interleave the two sides,
  as `bench_all.py`-style alternation does, rather than running all of one then
  all of the other.

**If it does not match, that is the finding, and this package's premise is
wrong. Report the number either way.**

### 7.5 The wrapper reproduces the binary's numbers

This is the strongest correctness result in this document, and the cheapest to
re-run. `log_line` emits the binary's exact format, so the two can be `diff`ed
once the ms column is blanked.

```bash
./train_bnn_gpu 2>&1 | grep '^step' | head -11 \
  | sed -E 's/\|[[:space:]]*[0-9.]+ ms$/| <ms>/' > binary.txt
```

```python
# wrapper.txt
import re, uncertain
t = uncertain.Trainer("bbb", data="data/text8/text8_train.bin",
                      validate_data=False)
for i in range(1, 101):
    nll = t.step()
    if i == 1 or i % 10 == 0:
        print(re.sub(r"\|\s*[\d.]+ ms$", "| <ms>", t.log_line(nll)))
```

```bash
diff binary.txt wrapper.txt
```

**Got, over 100 steps at the default configuration:**

| arm | result |
|---|---|
| `bbb` | **identical** — every `nll`, `kl`, `sigma`, `sigma_min`, `sigma_max` on all 11 lines |
| `lowrank` (r=8) | **identical**, all 11 lines |
| `vadam` | identical on 10 of 11 lines; line 11 differs by `kl 5478514.4` vs `5478514.5` |
| `ensemble` | 55 of 55 printed NLLs match across all 5 members (compared by value: the binary numbers steps per member, the Trainer counts globally) |

The one `vadam` line is not a wrapper defect: **the binary prints both values
itself across repeated runs** of the same executable on the same machine. The
underlying sum sits on a printf rounding boundary at ~5478514.45, ninth
significant digit. Verified by running `./train_bnn_vadam` three times and
diffing its output against itself. **[measured]**

Disagreement at step 1 would mean the init or the first sample differs;
disagreement only later would mean the update does. Neither occurred.

### 7.6 The four arms all run

```python
import uncertain
D = "data/text8/text8_train.bin"
for m in uncertain.METHODS:
    kw = {"rank": 8} if m == "lowrank" else {}
    t = uncertain.Trainer(m, data=D, **kw)
    print(f"{m:9s} params={t.n_params} nll={t.step():.4f} ms={t.last_step_ms:.2f}")
    del t
```
**Got:** four lines, `params=11061531` for all four, no crash on `del` (the
destructor frees the buffers and releases the cuBLAS handle refcount).
**[measured]**

### 7.7 Zero-copy is actually zero-copy

A shape check passes on a copy, so it proves nothing. Test **both directions**.

```python
import math, torch, uncertain
t = uncertain.Trainer("bbb", data="data/text8/text8_train.bin",
                      validate_data=False)
mu = t.parameters()["mu"]                       # hold THIS object across the step
assert t.parameters()["mu"].data_ptr() == mu.data_ptr()

before = mu.clone()                             # C -> Python
t.step()
assert (mu - before).abs().max().item() > 0     # a snapshot could not change

with torch.no_grad():                           # Python -> C
    mu.zero_()
nll = t.step()
V = t.config["vocab_size"]
assert abs(nll - math.log(V)) < 0.05            # flat logits: the write reached the kernels
```

**Got, on every arm** (mutating `mu`, `mu`, `theta`, `member0` respectively):

- **C → Python:** max |delta| after one step ≈ 1.004e-04 on all four — the AdamW
  step size, appearing in a tensor handed out *before* the step ran.
- **Python → C:** `bbb` 3.2796 → 3.2945, `vadam` 3.2884 → 3.2945, `lowrank`
  3.2768 → 3.2958, `ensemble` 3.2765 → 3.2958, against `ln 27 = 3.2958`. Zeroing
  the mean weights through the torch view collapses the network's logits to
  uniform on the next step. A copy leaves the loss where it was.
- **`named_tensors`:** all 78 named views satisfy `data_ptr() == base + offset*4`
  — no copy anywhere in the layout path.

**[measured]** If either direction fails, §4.2 is wrong.

### 7.8 Geometry really is a run-time argument

```python
import uncertain
D = "data/text8/text8_train.bin"
for L in (1, 2, 6, 12):
    t = uncertain.Trainer("bbb", data=D, n_layer=L)
    print(L, t.n_params, f"{t.step():.4f}")
    del t
t = uncertain.Trainer("bbb", data=D, n_layer=2, n_head=12, n_embd=768)
print("C=768:", t.n_params, f"{t.step():.4f}")
```
**Got:** finite losses throughout. `L=12` gives `params=21708315`. All four arms
also run at `L=2, C=128, NH=2, B=2, T=32` (`params=411931`), with `layout`
reporting exactly two blocks and `wte`/`wpe` reshaped to the new geometry — so
geometry really is a constructor argument, on every arm, not just `bbb`.
**[measured]**

```python
import uncertain
try:
    uncertain.Trainer("bbb", data="data/text8/text8_train.bin", n_head=8)  # 384/8 = 48
except ValueError as e:
    print("correctly rejected:", e)
```
**Got:** `ValueError: n_embd // n_head must equal 64 (got n_embd=384, n_head=8).
The FlashAttention kernel holds a head in HS=64 registers…` — a refusal, not a
crash or wrong numbers. Eleven other guard rails were exercised the same way
(unknown method, an arm's argument passed to a different arm, uncompiled rank,
`seq_len > max_seq_len`, missing data file, too many ensemble members,
`evaluate(batches=0)`, `evaluate(samples=0)`, `named_tensors` on a foreign
buffer, an arm property on the wrong arm, `load()` of a checkpoint that is not
there); every one raised the documented exception type. **[measured]**

### 7.9 Rank is a run-time argument

```python
import uncertain
D = "data/text8/text8_train.bin"
for r in (4, 8, 16):
    t = uncertain.Trainer("lowrank", data=D, rank=r)
    print(r, t.bayesian_params, f"{t.step():.4f}", f"{t.last_step_ms:.2f}ms")
    del t
try:
    uncertain.Trainer("lowrank", data=D, rank=7)
except ValueError as e:
    print("correctly rejected:", e)
```
**Got:** `rank=4` and `rank=8` both run, `rank=7` raises `ValueError: rank=7 is
not one of the compiled ranks [1, 2, 4, 8, 16, 32]`. **[measured]**

### 7.10 Evaluation, against the binary

```bash
for b in train_bnn_gpu train_bnn_vadam train_ensemble train_bnn_lowrank; do
  ./$b eval 100000 5 2 val 2>&1 | grep '^summary'
done
```

```python
import uncertain
t = uncertain.Trainer("bbb",
                      data="data/text8/text8_train.bin",
                      val_data="data/text8/text8_val.bin",
                      validate_data=False)
t.load(100000, tag="bbb")
d = t.evaluate(batches=2, samples=5, split="val", full=True)
for k in ("nlpd_single", "nlpd", "ece", "total", "aleatoric",
          "epistemic", "epistemic_ood", "identity_err"):
    print(f"{k:14s} {d[k]}")
```

**Got: every reported quantity matches the binary, on all four arms, to the full
precision the binary prints.** 28 comparisons, 0 mismatches. **[measured]**

| arm | NLPD single → mixture | ECE(15) | total = alea + epi | OOD epi |
|---|---|---|---|---|
| `bbb` | 1.5626 → 1.5503 | 0.0476 | 1.623 = 1.615 + 0.0089 | 0.0105 |
| `vadam` | 1.6892 → 1.6751 | 0.0383 | 1.737 = 1.705 + 0.0318 | 0.0340 |
| `ensemble` | 1.5189 → 1.4501 | 0.0770 | 1.623 = 1.552 + 0.0714 | 0.1845 |
| `lowrank` (r=8) | 1.6060 → 1.5892 | 0.0570 | 1.607 = 1.593 + 0.0138 | 0.0175 |

`identity_err` came out **0.0** from Python and `1.8e-15` from the binaries, both
far inside the 1e-5 gate; `epistemic_ood > epistemic` on all four, as required.
Note the checkpoint tags differ from the method names — `bbb`, `vadam`, `ens`,
`lowr` — so pass `tag=` to `load()` for the ensemble and low-rank arms.

Requires `runs/*_C384_L6_100k_*.bin`. Substitute another arm and its checkpoint
if you have only some of them.

### 7.11 The overfit smoke test, from Python

```python
import uncertain
t = uncertain.Trainer("bbb", data="data/text8/text8_train.bin",
                      fixed_batch=True, lr=1e-3)
for i in range(200):
    nll = t.step()
print("final nll:", nll)
```
**Expect:** `nll` falls to roughly what `./train_bnn_gpu overfit 200` reaches
(0.0172 at L=6 on this machine). This is the end-to-end check that sampling,
forward, backward and both AdamW passes are all wired correctly through the
Python boundary. Do not expect the digits to match: overfit runs are not
bit-reproducible even against themselves (§7.1). **Not run from Python** — §7.5
and §7.7 cover the same wiring with reproducible numbers. **[unverified]**

### 7.11b Save / load round-trip, and two Trainers at once

```python
import torch, uncertain
t = uncertain.Trainer("bbb", data="data/text8/text8_train.bin", validate_data=False)
for _ in range(3):
    t.step()
with torch.no_grad():
    t.parameters()["mu"].fill_(0.25)
t.save(tag="rtrip")                       # -> runs/rtrip_C384_L6_0k_{mu,rho}.bin
u = uncertain.Trainer("bbb", data="data/text8/text8_train.bin", validate_data=False)
assert u.load(3, tag="rtrip") == 3
assert torch.equal(u.parameters()["mu"], torch.full_like(u.parameters()["mu"], 0.25))
arr, meta = uncertain.load_checkpoint("runs/rtrip_C384_L6_0k_mu.bin")
print(meta)                               # {'n':11061531,'C':384,'L':6,'V':27,'NH':6,'steps':3}
```
**Got:** exactly that — `save()` writes the binaries' format and naming,
`load()` restores it bit for bit, and `load_checkpoint()` parses the same
8-int header. Two Trainers were also held alive at once (`bbb` and `ensemble`),
the first released, and the second kept stepping correctly — the refcounted
cuBLAS handle in §4.3 behaves. **[measured]**

### 7.12 Lifetime, as documented

```python
import uncertain
t = uncertain.Trainer("bbb", data="data/text8/text8_train.bin")
mu = t.parameters()["mu"]
del t
# mu now points at freed device memory. Do not touch it.
```
**Expect:** nothing to verify mechanically — this documents the contract in
§4.2. Do not "check" it by reading `mu`; that is undefined behaviour.

---

## 8. Still not verified

The list this section replaced covered everything, because nothing had been run.
What remains is:

1. **`make test` — the full scorecard.** ~4.1 h. Not run. It is the only check
   that exercises every arm in both modes end to end and gates on the result.
   §7.1's byte-identical binary output covers the same code far more cheaply,
   but not for the full step count.
2. **Any GPU other than the sm_75 RTX 2060 in §7.0.** Every number in this
   document is that card's. `sm_75` is the default arch; override with
   `UNCERTAIN_CUDA_ARCH=8.6` (see `setup.py`). Neither the build nor the timing
   has been tried at another arch.
3. **Multi-GPU.** `cudaSetDevice` per call and the `device=` argument are
   written for it; the reference machine has one GPU, so a second device was
   never selected.
4. **Non-default CUDA streams.** §4.3 explains why the extension owns its own
   cuBLAS handle rather than borrowing torch's — that reasoning predicts correct
   behaviour under `with torch.cuda.stream(...)`, and it was not tested.
5. **The `overfit` smoke test driven from Python** (§7.11). Skipped because
   overfit runs are not bit-reproducible even against themselves, so it would
   not have been a check. §7.5 and §7.7 cover the same wiring.
6. **CPython versions other than 3.12**, and torch versions other than
   2.11.0+cu130.

### A pre-existing failure found while verifying, not caused by this branch

`./train_bnn_lowrank check` fails check 2 (*A=0, dvec=0 → deterministic,
bitwise*, 2 mismatches). The identical failure is produced by the binary built
from the pre-change commit `c7ba8a0` with the system CUDA 12.0 toolchain, and
the entire `check` output is byte-identical before and after this branch — so
this branch neither caused it nor hid it. **It is a real defect on `main` and
deserves its own investigation.** **[measured]**

One incidental fix rides along and is called out so it can be reverted if
unwanted: `logdetM` in `train_bnn_lowrank.cu`'s `main()` was `malloc`'d and only
written when `KLW > 0`, so the printed `kl` column in overfit mode read
uninitialized memory. It is now `calloc`. This changes no value when
`KLW > 0`. **[source]**
