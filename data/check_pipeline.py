#!/usr/bin/env python3
"""Report on the prepared data pipeline — no model, just confirms it's correct."""
import os

import tiktoken

from dataloader import DataLoader

DATA_DIR = os.path.join(os.path.dirname(__file__), "tinyshakespeare")
B, T = 4, 64  # llm.c's own documented quick/CPU-demo config (README: "batch size 4, context length only 64")
VOCAB_SIZE = 50257  # gpt-2 BPE via tiktoken, model->config.vocab_size in train_gpt2.c

enc = tiktoken.get_encoding("gpt2")
assert enc.n_vocab == VOCAB_SIZE

train_loader = DataLoader(os.path.join(DATA_DIR, "tiny_shakespeare_train.bin"), B, T)
val_loader = DataLoader(os.path.join(DATA_DIR, "tiny_shakespeare_val.bin"), B, T)
total_tokens = train_loader.num_tokens + val_loader.num_tokens

print(f"train tokens: {train_loader.num_tokens:,}")
print(f"val tokens:   {val_loader.num_tokens:,}")
print(f"total tokens: {total_tokens:,}")
print(f"vocab size:   {VOCAB_SIZE:,} (tiktoken 'gpt2' encoding)")
print(f"batch shape:  B={B}, T={T}\n")

inputs, targets = train_loader.next_batch()
assert inputs.shape == targets.shape == (B, T)
assert inputs.dtype == targets.dtype
assert (0 <= inputs).all() and (inputs < VOCAB_SIZE).all()
assert (0 <= targets).all() and (targets < VOCAB_SIZE).all()
# llm.c's shift convention: target[t] == input[t+1], except crossing the last row's boundary
assert (targets[:, :-1] == inputs[:, 1:]).all()

print(f"inputs.shape={inputs.shape} dtype={inputs.dtype}, targets.shape={targets.shape} dtype={targets.dtype}")
print("matches train_gpt2.c's gpt2_forward(inputs, targets, B, T): flat int arrays, B*T long, values in [0, V)\n")

row = 0
print(f"--- sample batch, row {row} ---")
print("input :", repr(enc.decode(inputs[row].tolist())))
print("target:", repr(enc.decode(targets[row].tolist())))
print("\n(target is input shifted left by one token — decode input[1:] and target[:-1] to see them line up)")
print("input[1:] :", repr(enc.decode(inputs[row, 1:].tolist())))
print("target[:-1]:", repr(enc.decode(targets[row, :-1].tolist())))
