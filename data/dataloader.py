"""
Python port of llm.c's llmc/dataloader.h (single shard, single process case).
Reads the exact .bin format written by data_common.py's write_datafile():
  - 256 int32 header: [magic=20240520, version=1, num_tokens, 0, 0, ...]
  - num_tokens uint16 token ids

Batching matches dataloader_load_batch() in llmc/dataloader.h exactly: each
batch reads B*T+1 contiguous tokens, inputs = tokens[0:B*T], targets =
tokens[1:B*T+1] (both reshaped to (B,T)) — so a batch's last target token
equals the next batch's first input token, and the last row's target crosses
into the next row's input. This is llm.c's actual convention, not a simplified
per-row-independent shift.
"""
import numpy as np

HEADER_SIZE = 256
MAGIC = 20240520
VERSION = 1


class DataLoader:
    def __init__(self, filename, B, T):
        with open(filename, "rb") as f:
            header = np.frombuffer(f.read(HEADER_SIZE * 4), dtype=np.int32)
            assert header[0] == MAGIC, f"bad magic {header[0]}, expected {MAGIC}"
            assert header[1] == VERSION, f"bad version {header[1]}, expected {VERSION}"
            self.num_tokens = int(header[2])
            self.tokens = np.frombuffer(f.read(self.num_tokens * 2), dtype=np.uint16)
        assert len(self.tokens) == self.num_tokens

        self.B, self.T = B, T
        self.shard_num_samples = (self.num_tokens * 2 - 2) // (B * T * 2)
        assert self.shard_num_samples > 0, "not enough tokens for even one B*T+1 batch"
        self.current_sample_idx = 0

    def reset(self):
        self.current_sample_idx = 0

    def next_batch(self):
        if self.current_sample_idx >= self.shard_num_samples:
            self.current_sample_idx = 0  # single shard: just wrap around
        B, T = self.B, self.T
        offset = self.current_sample_idx * B * T
        buf = self.tokens[offset : offset + B * T + 1].astype(np.int32)
        inputs = buf[: B * T].reshape(B, T)
        targets = buf[1 : B * T + 1].reshape(B, T)
        self.current_sample_idx += 1
        return inputs, targets
