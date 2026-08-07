"""
text8 -> the .bin format the CUDA loader already reads.

WHY: TinyShakespeare has 1.0M training tokens against 43M model parameters —
43x more parameters than data points. The ELBO's KL term is per-dataset, so at
that ratio the prior outweighs the data ~35:1 even at the mathematically
correct weight, and every Bayesian arm is pinned near its prior. Measured on
Vadam, where N appears directly in sigma = 1/sqrt(N*s + lambda):

    TinyShakespeare  N=1e6   best-informed parameter: 35% data-driven
    text8            N=1e8   best-informed parameter: 98% data-driven

text8 is also the standard char-level LM benchmark, so numbers become
comparable to published work instead of only to each other.

Standard split (Mikolov): first 90M train, next 5M val, last 5M test.
Vocab is 27: lowercase a-z plus space.

    python3 data/prepare_text8.py
"""
import numpy as np, os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC  = os.path.join(HERE, "text8", "text8")
MAGIC, VERSION = 20240520, 1

text = open(SRC, "r").read()
vocab = sorted(set(text))
stoi = {c: i for i, c in enumerate(vocab)}
ids = np.array([stoi[c] for c in text], dtype=np.uint16)

splits = {"train": ids[:90_000_000],
          "val":   ids[90_000_000:95_000_000],
          "test":  ids[95_000_000:]}

for name, arr in splits.items():
    header = np.zeros(256, dtype=np.int32)
    header[0], header[1], header[2], header[3] = MAGIC, VERSION, len(arr), len(vocab)
    out = os.path.join(HERE, "text8", f"text8_{name}.bin")
    with open(out, "wb") as f:
        f.write(header.tobytes()); f.write(arr.tobytes())
    print(f"{name:5s} {len(arr):>11,} tokens -> {out}")

print(f"\nvocab size = {len(vocab)}  {repr(''.join(vocab))}")
print(f"set VOCAB = {len(vocab)} and DATA_TRAIN/DATA_VAL in gpt2_kernels.cuh")
