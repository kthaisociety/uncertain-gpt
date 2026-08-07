"""uncertain — the C/CUDA Bayesian GPT-2 trainer, importable from Python.

    import uncertain

    t = uncertain.Trainer(method="lowrank", rank=8,
                          n_layer=6, n_head=6, n_embd=384,
                          data="data/text8/text8_train.bin")
    t.step()                          # one training step, on the existing kernels
    p = t.parameters()                # zero-copy torch views of the live buffers
    nlpd, ece, ale, epi = t.evaluate()

`method` selects among the four arms of the comparison:

    "bbb"       train_bnn_gpu.cu       Bayes-by-Backprop   (Blundell et al. 2015)
    "vadam"     train_bnn_vadam.cu     Vadam               (Khan et al. 2018)
    "lowrank"   train_bnn_lowrank.cu   low-rank + diagonal covariance
    "ensemble"  train_ensemble.cu      deep ensemble       (Lakshminarayanan 2017)

The boundary is the TRAINING STEP, not the tensor op. `step()` runs the same
kernels in the same order as the standalone binary; there are no per-op
autograd wrappers, because the forward/backward in gpt2_kernels.cuh is one
fused pass over flat parameter and activation slabs and taking it apart would
cost the performance this package exists to ship. See SPEC.md.

Requires nvcc at install time and an NVIDIA GPU at run time. fp32 only.
"""

from __future__ import annotations

import os
import warnings
from typing import Dict, List, NamedTuple, Optional, Tuple

import torch

try:
    from . import _C
except ImportError as exc:  # pragma: no cover - install-time failure path
    raise ImportError(
        "uncertain._C is not built. Install the package from the repository root "
        "with:\n\n    pip install -e . --no-build-isolation\n\n"
        "This compiles the CUDA sources and needs nvcc on PATH.\n"
        f"Original error: {exc}"
    ) from exc

__version__ = "0.1.0"

METHODS = ("bbb", "vadam", "lowrank", "ensemble")

#: The file each method's mathematics lives in. `uncertain` adds no algorithm
#: of its own — these files are #included verbatim by uncertain/_C.cu.
METHOD_SOURCE = {
    "bbb": "train_bnn_gpu.cu",
    "vadam": "train_bnn_vadam.cu",
    "lowrank": "train_bnn_lowrank.cu",
    "ensemble": "train_ensemble.cu",
}

CKPT_MAGIC = 20240520

#: n_embd // n_head must equal this. HS is a compile-time constant in
#: gpt2_kernels.cuh: the FlashAttention kernel holds a whole head in registers.
HEAD_SIZE: int = _C.HEAD_SIZE

#: Most members the ensemble arm accepts — E_MEMBERS in train_ensemble.cu.
MAX_ENSEMBLE_MEMBERS: int = _C.MAX_ENSEMBLE_MEMBERS

#: Confidence bins the shared harness uses for ECE.
ECE_BINS: int = _C.UQ_BINS

#: The default data paths compiled into gpt2_kernels.cuh, relative to the
#: repository root. Handy defaults, not a promise that the files exist.
DATA_TRAIN: str = _C.DATA_TRAIN
DATA_VAL: str = _C.DATA_VAL
DATA_TEST: str = _C.DATA_TEST


def supported_ranks() -> List[int]:
    """Ranks the low-rank arm was compiled for.

    The r-loop is unrolled at compile time, so each rank is its own kernel
    instantiation. Adding one means editing UNCERTAIN_LOWRANK_RANKS in
    uncertain/_C.cu and rebuilding.
    """
    return list(_C.supported_ranks())


def selftest() -> int:
    """Run the shared decomposition's self-test. Returns the failure count.

    `uq_selftest` in gpt2_kernels.cuh: three cases whose answers are known on
    paper (one sample, two identical samples, two disjoint one-hots), checked
    to 1e-12. Pure host arithmetic — no GPU, no data, no model — so this is
    also the cheapest proof that the extension built and imports correctly::

        python3 -c "import uncertain; assert uncertain.selftest() == 0"
    """
    return _C.selftest()


def default_ensemble_seeds() -> List[int]:
    """The member seeds train_ensemble.cu uses, read from the C source."""
    return list(_C.default_ensemble_seeds())


class EvalResult(NamedTuple):
    """What `Trainer.evaluate()` returns, in nats (except ECE, a probability).

    Unpacks as ``nlpd, ece, ale, epi = t.evaluate()``. Ask for
    ``evaluate(full=True)`` to also get the total entropy, the single-predictor
    NLPD and the identity residual.
    """

    nlpd: float
    ece: float
    aleatoric: float
    epistemic: float


class _Unset:
    __slots__ = ()

    def __repr__(self) -> str:
        return "<unset>"


_UNSET = _Unset()

# Which constructor arguments belong to which arm. Passing one that belongs to
# a different arm is an error, not a silently ignored keyword.
_ARM_ARGS = {
    "bbb": ("sigma_prior", "kl_weight", "rho_init"),
    "vadam": ("lam", "s0"),
    "lowrank": ("rank", "lam", "kl_weight", "lr_var", "a_init", "dvec_init"),
    "ensemble": ("seeds", "steps_per_member"),
}

def read_bin_header(path: str) -> Dict[str, int]:
    """Header of an llm.c-format token file: magic, version, tokens, vocab."""
    import numpy as np

    with open(path, "rb") as f:
        raw = f.read(1024)
    if len(raw) < 1024:
        raise ValueError(f"{path}: shorter than the 256-int header")
    head = np.frombuffer(raw, dtype=np.int32)
    if int(head[0]) != CKPT_MAGIC:
        raise ValueError(f"{path}: bad magic {int(head[0])}, expected {CKPT_MAGIC}")
    return {
        "magic": int(head[0]),
        "version": int(head[1]),
        "tokens": int(head[2]),
        "vocab": int(head[3]),
    }


def load_checkpoint(path: str):
    """Read a checkpoint written by `Trainer.save()` or a standalone binary.

    Same 8-int header both write: ``{magic, n_lo, C, L, V, NH, steps, n_hi}``
    followed by n float32. Returns ``(array, meta)``.

    The point of this is that the uncertainty evaluation can run against models
    that are ALREADY TRAINED, rather than costing a full training run per
    metric::

        arr, meta = uncertain.load_checkpoint("runs/bbb_C384_L6_100k_mu.bin")
        t.parameters()["mu"].copy_(torch.from_numpy(arr))
    """
    import numpy as np

    with open(path, "rb") as f:
        hdr = np.frombuffer(f.read(32), dtype=np.int32)
        if hdr.size < 8:
            raise ValueError(f"{path}: truncated checkpoint header")
        if int(hdr[0]) != CKPT_MAGIC:
            raise ValueError(f"{path}: bad magic {int(hdr[0])}")
        n = (int(hdr[7]) << 32) | (int(hdr[1]) & 0xFFFFFFFF)
        arr = np.fromfile(f, dtype=np.float32, count=n)
    if arr.size != n:
        raise ValueError(f"{path}: expected {n} floats, found {arr.size}")
    meta = {
        "n": n,
        "C": int(hdr[2]),
        "L": int(hdr[3]),
        "V": int(hdr[4]),
        "NH": int(hdr[5]),
        "steps": int(hdr[6]),
    }
    return arr, meta


def _check_data(path: str, vocab_size: int, ntok: int) -> None:
    """Reject token ids the model has no embedding row for.

    An id >= vocab_size indexes past wte in k_encoder_forward. That is an
    out-of-bounds device read: no error, just quietly wrong numbers. Cheaper to
    catch here, once, than to debug later.
    """
    import numpy as np

    a = np.memmap(path, dtype=np.uint16, mode="r", offset=1024, shape=(ntok,))
    hi = 0
    for i in range(0, ntok, 1 << 24):
        chunk = a[i : i + (1 << 24)]
        if chunk.size:
            hi = max(hi, int(chunk.max()))
    if hi >= vocab_size:
        raise ValueError(
            f"{path} contains token id {hi} but vocab_size={vocab_size}. "
            "Either the wrong .bin or the wrong vocab_size."
        )


class Trainer:
    """One training run of one arm, owning its GPU buffers.

    Parameters
    ----------
    method : {"bbb", "vadam", "lowrank", "ensemble"}
        Which arm. See METHOD_SOURCE for the file each one lives in.
    data : str
        Path to an llm.c-format token file (256-int header, magic 20240520,
        then uint16 ids). `python3 data/prepare_text8.py` writes them.
    val_data : str, optional
        Held-out tokens for `evaluate()`. If omitted, evaluate() reads the
        training stream from wherever training has got to — which is exactly
        the data training has not reached yet, and is what train_ensemble.cu
        does. Loading a second file costs another host-side copy of it.
    n_layer, n_head, n_embd : int
        Model geometry. ``n_embd // n_head`` must equal 64 (HEAD_SIZE).
    vocab_size : int, optional
        Defaults to the vocab recorded in the data file's header.
    batch_size, seq_len, max_seq_len : int
        B, T and the width of the learned position table.
    lr : float
        AdamW step. The binaries use 1e-4 to train and 1e-3 with
        ``fixed_batch=True``, where the point is to memorize one batch fast.
    beta1, beta2, eps, weight_decay : float
        AdamW, as in the binaries.
    init_std : float
        Weight init scale. Also sets Vadam's derived prior precision.
    seed : int
        Philox key for the weight sampling. Not the init seed: init follows the
        binaries and uses srand(1337) (srand(seeds[e]) per ensemble member).
    device : int, optional
        CUDA device ordinal. Defaults to the current one.
    fixed_batch : bool
        Reuse one batch forever — the binaries' `overfit` smoke test.
    validate_data : bool
        Scan the token file for ids outside the vocabulary. Costs one pass over
        the file at construction.

    Arm-specific
    ------------
    bbb:      sigma_prior=0.1, kl_weight=None, rho_init=-5.0
    vadam:    lam=None, s0=0.0
    lowrank:  rank=8, lam=100.0, kl_weight=None, lr_var=1e-5,
              a_init=1e-4, dvec_init=1e-3
    ensemble: seeds=<the five in train_ensemble.cu>, steps_per_member=100

    ``kl_weight=None`` means DERIVE it as 1/N over the training tokens, which
    is what the ELBO says it is — the KL is per-dataset and the reported NLL is
    per-token. It is not a knob; hardcoding it once cost 1.0 nats. With
    ``fixed_batch=True`` it defaults to 0, matching the binaries' overfit mode.

    ``lam=None`` (Vadam) means derive the prior precision as 1/init_std**2, so
    the step-1 sigma equals the scale the weights are initialized at.

    Lifetime
    --------
    Tensors from `parameters()` and `buffers()` are VIEWS of memory this
    Trainer owns. They do not keep it alive. Using one after the Trainer is
    garbage-collected reads freed device memory. Clone anything you intend to
    keep.
    """

    def __init__(
        self,
        method: str,
        data: str,
        *,
        val_data: Optional[str] = None,
        n_layer: int = 6,
        n_head: int = 6,
        n_embd: int = 384,
        vocab_size: Optional[int] = None,
        batch_size: int = 4,
        seq_len: int = 64,
        max_seq_len: int = 1024,
        lr: float = 1e-4,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        init_std: float = 0.02,
        seed: int = 20240520,
        device: Optional[int] = None,
        fixed_batch: bool = False,
        validate_data: bool = True,
        # --- arm-specific; each is rejected unless it belongs to `method` ---
        sigma_prior=_UNSET,
        kl_weight=_UNSET,
        rho_init=_UNSET,
        lam=_UNSET,
        s0=_UNSET,
        rank=_UNSET,
        lr_var=_UNSET,
        a_init=_UNSET,
        dvec_init=_UNSET,
        seeds=_UNSET,
        steps_per_member=_UNSET,
    ):
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}, got {method!r}")

        supplied = {
            name: value
            for name, value in (
                ("sigma_prior", sigma_prior), ("kl_weight", kl_weight),
                ("rho_init", rho_init), ("lam", lam), ("s0", s0), ("rank", rank),
                ("lr_var", lr_var), ("a_init", a_init), ("dvec_init", dvec_init),
                ("seeds", seeds), ("steps_per_member", steps_per_member),
            )
            if not isinstance(value, _Unset)
        }
        stray = sorted(set(supplied) - set(_ARM_ARGS[method]))
        if stray:
            raise TypeError(
                f"method={method!r} does not take {', '.join(stray)} "
                f"(it takes {', '.join(_ARM_ARGS[method]) or 'no arm-specific arguments'})"
            )

        if not torch.cuda.is_available():
            raise RuntimeError("uncertain needs a CUDA device; torch reports none")
        if device is None:
            device = torch.cuda.current_device()
        torch.cuda.set_device(device)
        torch.cuda.init()

        if not os.path.exists(data):
            raise FileNotFoundError(data)
        header = read_bin_header(data)
        if vocab_size is None:
            vocab_size = header["vocab"] or _C.DEFAULT_VOCAB
        if val_data is not None:
            if not os.path.exists(val_data):
                raise FileNotFoundError(val_data)
            read_bin_header(val_data)

        if n_head <= 0 or n_embd % n_head != 0 or n_embd // n_head != HEAD_SIZE:
            raise ValueError(
                f"n_embd // n_head must equal {HEAD_SIZE} (got n_embd={n_embd}, "
                f"n_head={n_head}). The FlashAttention kernel holds a head in "
                f"HS={HEAD_SIZE} registers and HS is a compile-time constant."
            )
        if seq_len > max_seq_len:
            raise ValueError(f"seq_len ({seq_len}) must be <= max_seq_len ({max_seq_len})")

        if validate_data:
            _check_data(data, vocab_size, header["tokens"])
            if val_data is not None:
                vh = read_bin_header(val_data)
                _check_data(val_data, vocab_size, vh["tokens"])

        common = dict(
            data=data,
            val_data=val_data or "",
            B=int(batch_size), T=int(seq_len), C=int(n_embd), V=int(vocab_size),
            NH=int(n_head), maxT=int(max_seq_len), L=int(n_layer),
            device=int(device), seed=int(seed), lr=float(lr), beta1=float(beta1),
            beta2=float(beta2), eps=float(eps), weight_decay=float(weight_decay),
            init_std=float(init_std), fixed_batch=bool(fixed_batch),
        )

        # kl_weight: None -> derive 1/N, except in the overfit mode where the
        # binaries leave the prior term off entirely. -1.0 is the C-side "derive".
        def _klw(value):
            if isinstance(value, _Unset) or value is None:
                return 0.0 if fixed_batch else -1.0
            return float(value)

        if method == "bbb":
            self._t = _C.make_bbb(
                **common,
                sigma_prior=0.1 if isinstance(sigma_prior, _Unset) else float(sigma_prior),
                kl_weight=_klw(kl_weight),
                rho_init=-5.0 if isinstance(rho_init, _Unset) else float(rho_init),
            )
        elif method == "vadam":
            self._t = _C.make_vadam(
                **common,
                lam=-1.0 if (isinstance(lam, _Unset) or lam is None) else float(lam),
                s0=0.0 if isinstance(s0, _Unset) else float(s0),
            )
        elif method == "lowrank":
            r = 8 if isinstance(rank, _Unset) else int(rank)
            if r not in supported_ranks():
                raise ValueError(
                    f"rank={r} is not one of the compiled ranks {supported_ranks()}. "
                    "The r-loop unrolls at compile time; a new rank needs a rebuild "
                    "(UNCERTAIN_LOWRANK_RANKS in uncertain/_C.cu)."
                )
            self._t = _C.make_lowrank(
                **common,
                rank=r,
                lam=100.0 if isinstance(lam, _Unset) else float(lam),
                kl_weight=_klw(kl_weight),
                lr_var=1e-5 if isinstance(lr_var, _Unset) else float(lr_var),
                a_init=1e-4 if isinstance(a_init, _Unset) else float(a_init),
                dvec_init=1e-3 if isinstance(dvec_init, _Unset) else float(dvec_init),
            )
        else:  # ensemble
            sd = default_ensemble_seeds() if isinstance(seeds, _Unset) else [int(s) for s in seeds]
            if not sd:
                raise ValueError("ensemble needs at least one seed")
            if len(sd) > MAX_ENSEMBLE_MEMBERS:
                raise ValueError(
                    f"at most {MAX_ENSEMBLE_MEMBERS} ensemble members "
                    f"(E_MEMBERS in train_ensemble.cu, which is also the size of "
                    f"the context ens_draw indexes); got {len(sd)}"
                )
            self._t = _C.make_ensemble(
                **common,
                seeds=sd,
                steps_per_member=100 if isinstance(steps_per_member, _Unset)
                else int(steps_per_member),
            )

        self.method = method
        self.source = METHOD_SOURCE[method]
        # The harness prints the split with every number and never infers one.
        # Without val_data we are reading the training stream past the cursor:
        # unseen, but not a held-out split, and the label says so.
        self._default_split = "val" if val_data else "train-tail"

    # ---- the loop ---------------------------------------------------------
    def step(self) -> float:
        """One training step. Returns the mean NLL over the batch, in nats.

        Exactly the body of the arm's training loop: sample the weights, zero
        the accumulators, forward, backward, push the gradient onto the
        variational parameters, update. Blocks until the step is finished, as
        the binaries do (they time every step with CUDA events).
        """
        return self._t.step()

    def stats(self) -> Dict[str, float]:
        """KL and sigma summary — the arm's stats kernel plus a host reduction.

        NOT computed inside `step()`: the binaries only run this every tenth
        step, and doing it per step would add a kernel and four device-to-host
        copies to a step that is otherwise all model.

        The ensemble arm samples nothing and reports zeros, as its binary does.
        """
        s = self._t.stats()
        return {
            "kl": s.kl,
            "sigma_mean": s.sigma_mean,
            "sigma_min": s.sigma_min,
            "sigma_max": s.sigma_max,
        }

    def log_line(self, nll: float) -> str:
        """The binaries' per-step log line, verbatim — scorecard.c parses it."""
        s = self._t.stats()
        return (
            f"step {self.step_idx:4d} | nll {nll:8.4f} | kl {s.kl:10.1f} | "
            f"sigma {s.sigma_mean:.6f} [{s.sigma_min:.6f}, {s.sigma_max:.6f}] | "
            f"{self.last_step_ms:7.2f} ms"
        )

    # ---- evaluation -------------------------------------------------------
    def evaluate(
        self,
        batches: int = 2,
        samples: Optional[int] = None,
        split: Optional[str] = None,
        full: bool = False,
    ):
        """Uncertainty metrics over `batches` held-out batches of B*T tokens.

        This is `uq_evaluate` from gpt2_kernels.cuh, unchanged — the harness all
        four arms share, driven with this arm's own sample callback. It draws
        `samples` (S) predictive distributions per position, mixes the
        PROBABILITIES, and reports:

            nlpd       -log p_bar(y), the S-sample mixture   (per token, nats)
            ece        expected calibration error, 15 bins on argmax confidence
            aleatoric  mean sample entropy       (1/S) sum H[p_s]
            epistemic  mean disagreement         (1/S) sum KL(p_s || p_bar)

        `total = aleatoric + epistemic` is exact algebra for any S and any
        sampler, so the harness asserts it at 1e-5 and this call RAISES if it
        fails — a violation can only mean the predictive probabilities are
        wrong, never a modelling problem.

        The same run also sweeps an OOD set of uniform-random token ids, so
        `full=True` gives you `epistemic_ood`, which must be HIGHER than
        `epistemic` or the arm is not capturing model uncertainty at all.

        S and fairness: the Bayesian arms can draw as many samples as you like
        and their estimates converge as S grows. The ensemble cannot — its S-th
        sample is its S-th TRAINED MEMBER, so `samples` is ignored there and S
        is the member count. `full=True` reports `samples_fixed_by_arm` saying
        so. Comparing arms at equal S equalizes inference cost, not the cost of
        obtaining the samples; do not report it as if it did.

        `split` is a label carried onto every number, because the harness
        refuses to leave it unstated. It defaults to "val" when the Trainer was
        given `val_data` and "train-tail" when it was not — in that case
        evaluation reads the training stream from wherever training has got to,
        which is text training has not seen but is not a held-out split.

        Evaluating does not disturb training: the data cursor and the current
        batch are both restored, and the draws come from a Philox stream
        (UQ_EVAL_SEED) disjoint from every training stream.
        """
        if samples is not None and samples < 1:
            raise ValueError("samples must be >= 1")
        if batches < 1:
            raise ValueError("batches must be >= 1")
        if split is None:
            split = self._default_split
        r = self._t.evaluate(
            batches=int(batches), samples=int(samples or 0), split=str(split)
        )
        if samples is not None and r.samples_fixed_by_arm and samples != r.samples:
            warnings.warn(
                f"samples={samples} ignored: the {self.method!r} arm's S is its "
                f"member count ({r.samples}). Raising it costs another full "
                f"training run, not another forward pass.",
                RuntimeWarning,
                stacklevel=2,
            )
        if full:
            return {
                "nlpd": r.nlpd, "ece": r.ece, "aleatoric": r.aleatoric,
                "epistemic": r.epistemic, "epistemic_ood": r.epistemic_ood,
                "total": r.total, "nlpd_single": r.nlpd_single,
                "identity_err": r.identity_err, "positions": r.positions,
                "positions_ood": r.positions_ood, "samples": r.samples,
                "samples_fixed_by_arm": r.samples_fixed_by_arm, "split": r.split,
            }
        return EvalResult(r.nlpd, r.ece, r.aleatoric, r.epistemic)

    # ---- parameters -------------------------------------------------------
    def parameters(self) -> Dict[str, torch.Tensor]:
        """Zero-copy 1-D CUDA views of the live variational parameters.

        bbb      mu, rho
        vadam    mu, s          (sigma is derived from s, never stored)
        lowrank  theta, A, rhod
        ensemble member0 ... memberE-1

        These are `torch::from_blob` views of the buffers the kernels write:
        no device copy, no change to the memory layout the kernels expect, and
        writing through one writes the buffer the next step will read. They do
        NOT own or extend the Trainer's memory — see the class docstring.
        """
        return self._t.parameters()

    def buffers(self) -> Dict[str, torch.Tensor]:
        """Zero-copy views of the rest: optimizer state, gradients, the current
        weight sample. Same lifetime rules as `parameters()`."""
        return self._t.buffers()

    @property
    def layout(self) -> Dict[str, Tuple[int, Tuple[int, ...]]]:
        """name -> (offset, shape) for every tensor in the flat parameter vector.

        Straight out of `param_offsets()` in gpt2_kernels.cuh, which is the
        single source of truth the kernels themselves slice from.
        """
        return {k: (off, tuple(shape)) for k, (off, shape) in self._t.layout().items()}

    def named_tensors(self, which: str) -> Dict[str, torch.Tensor]:
        """Named, correctly-shaped views into one full-size parameter buffer.

            t.named_tensors("mu")["h0.attn.qkv.weight"]   # (C, 3C), no copy

        Only for buffers that span the whole parameter vector (mu, rho, theta,
        memberN, s) — not for the low-rank arm's packed A/rhod.
        """
        flat = self.parameters().get(which)
        if flat is None:
            raise KeyError(f"{which!r} is not a parameter of the {self.method} arm; "
                           f"have {sorted(self.parameters())}")
        if flat.numel() != self.n_params:
            raise ValueError(
                f"{which!r} has {flat.numel()} elements, not the model's "
                f"{self.n_params}: it is a packed per-arm buffer, not a copy of "
                "the parameter vector, so the layout does not describe it"
            )
        out = {}
        for name, (off, shape) in self.layout.items():
            numel = 1
            for d in shape:
                numel *= d
            out[name] = flat[off : off + numel].view(*shape)
        return out

    def load(self, steps: int, tag: Optional[str] = None) -> int:
        """Load a checkpoint from ./runs into the live buffers.

        Mirrors `save()` and the binaries' `eval` mode: reads
        ``runs/<tag>_C<C>_L<L>_<steps/1000>k_*.bin``. `ckpt_load` checks the
        shape fields, so loading a C=768 checkpoint into a C=384 run fails
        rather than reporting confident nonsense. Returns the step count
        recorded in the file.

        This is how the uncertainty numbers run against models that are ALREADY
        TRAINED, instead of costing a full training run per metric::

            t = uncertain.Trainer("bbb", data=..., val_data=...)
            t.load(100000)
            nlpd, ece, ale, epi = t.evaluate(batches=64, samples=8)

        PARAMETERS ONLY. AdamW moments are not in the checkpoint (they never
        were), so this restores a model to evaluate, not a run to resume.
        """
        got = self._t.load(tag if tag is not None else self._t.arm, int(steps))
        return got

    def save(self, tag: Optional[str] = None) -> None:
        """Write checkpoints into ./runs, in the binaries' format and naming.

        Same `ckpt_save` the standalone binaries call, so the files are
        interchangeable with the ones already on disk:
        ``runs/<tag>_C<C>_L<L>_<steps/1000>k_<what>.bin``. Defaults to the
        arm's own tag ("bbb", "vadam", "lowr", "ens").
        """
        self._t.save(tag if tag is not None else self._t.arm)

    # ---- read-only state --------------------------------------------------
    @property
    def step_idx(self) -> int:
        """Completed steps."""
        return self._t.step_idx

    @property
    def last_step_ms(self) -> float:
        """Wall time of the last step, from the CUDA events around it — the
        same measurement the binaries print."""
        return self._t.last_step_ms

    @property
    def n_params(self) -> int:
        return self._t.n_params

    @property
    def n_act(self) -> int:
        """Floats in one activation buffer. Two are allocated."""
        return self._t.n_act

    @property
    def train_tokens(self) -> int:
        return self._t.train_tokens

    @property
    def config(self) -> Dict[str, int]:
        return self._t.config

    @property
    def device(self) -> int:
        return self._t.device

    @property
    def total_steps(self) -> int:
        """Steps this Trainer accepts, or -1 for open-ended.

        Only the ensemble is bounded: members * steps_per_member, because the
        arm is E separate trainings sharing one pair of AdamW moment buffers.
        """
        return self._t.total_steps

    @property
    def hyperparams(self) -> Dict[str, float]:
        """The arm's own settings, including any that were derived rather than
        given: bbb/lowrank `kl_weight`, vadam `lam` and `N`, the low-rank
        `bayesian_params` count, the ensemble's `current_member`."""
        return dict(self._t.hyperparams)

    def _hp(self, key: str) -> float:
        hp = self._t.hyperparams
        if key not in hp:
            raise AttributeError(f"the {self.method!r} arm has no {key!r}")
        return hp[key]

    # --- arm-specific, each raising AttributeError on the wrong arm --------
    @property
    def kl_weight(self) -> float:
        return self._hp("kl_weight")

    @property
    def lam(self) -> float:
        return self._hp("lam")

    @property
    def rank(self) -> int:
        return int(self._hp("rank"))

    @property
    def bayesian_params(self) -> int:
        """Low-rank only: parameters given a posterior — the four GEMMs, not
        the whole model. State this when comparing arms."""
        return int(self._hp("bayesian_params"))

    @property
    def members(self) -> int:
        return int(self._hp("members"))

    @property
    def current_member(self) -> int:
        return int(self._hp("current_member"))

    @property
    def seeds(self) -> List[int]:
        """Ensemble only: the per-member init seeds. Empty for the other arms —
        the seed's only job in this arm is a different starting point."""
        return list(self._t.seeds)

    def __repr__(self) -> str:
        c = self.config
        return (
            f"<uncertain.Trainer method={self.method!r} source={self.source} "
            f"L={c['n_layer']} C={c['n_embd']} NH={c['n_head']} V={c['vocab_size']} "
            f"B={c['batch_size']} T={c['seq_len']} params={self.n_params} "
            f"step={self.step_idx}>"
        )


__all__ = [
    "Trainer",
    "EvalResult",
    "METHODS",
    "METHOD_SOURCE",
    "HEAD_SIZE",
    "MAX_ENSEMBLE_MEMBERS",
    "ECE_BINS",
    "DATA_TRAIN",
    "DATA_VAL",
    "DATA_TEST",
    "supported_ranks",
    "default_ensemble_seeds",
    "selftest",
    "load_checkpoint",
    "read_bin_header",
    "__version__",
]
