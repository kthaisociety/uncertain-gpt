"""
Character-level TinyShakespeare, in the SAME .bin format the CUDA loader
already reads (256 int32 header, magic 20240520, then uint16 token ids).

Why: with GPT-2 BPE the vocabulary is 50257, so wte alone is 50257*768 =
38.6M parameters — 45% of the model, and the logits/probs activation buffers
are B*T*50257 = 103 MB each. The model under study becomes an embedding
lookup table with a transformer attached.

TinyShakespeare has 65 distinct characters. At V=65 the embedding is 49,920
parameters (0.06%) and the logits buffer is 133 KB. The transformer becomes
~98% of the model, which is the thing the experiments are actually about.

Nothing in the CUDA code changes except Config.V — the loader reads integer
token ids and does not care what they mean.

    python3 data/prepare_char.py
"""
import numpy as np, os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC  = os.path.join(HERE, "tinyshakespeare", "tiny_shakespeare.txt")
MAGIC, VERSION = 20240520, 1

text = open(SRC, "r", encoding="utf-8").read()
vocab = sorted(set(text))
stoi = {c: i for i, c in enumerate(vocab)}
ids = np.array([stoi[c] for c in text], dtype=np.uint16)

# Same 90/10 split point the BPE prep used, so the two are comparable.
n_val = len(ids) // 10
splits = {"val": ids[:n_val], "train": ids[n_val:]}

for name, arr in splits.items():
    header = np.zeros(256, dtype=np.int32)
    header[0], header[1], header[2] = MAGIC, VERSION, len(arr)
    header[3] = len(vocab)                      # vocab size, for reference
    out = os.path.join(HERE, "tinyshakespeare", f"tiny_shakespeare_char_{name}.bin")
    with open(out, "wb") as f:
        f.write(header.tobytes()); f.write(arr.tobytes())
    print(f"{name:5s} {len(arr):>8,} tokens -> {out}")

print(f"\nvocab size = {len(vocab)}")
print("chars:", repr("".join(vocab)))
print("\nset Config.V = %d in the trainers." % len(vocab))
