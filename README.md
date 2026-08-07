<p align="center">
  <img src="logo.svg" alt="uncertain-gpt" width="620">
</p>

A GPT-2 written from scratch in C and CUDA, where the weights are distributions
instead of numbers. Same network every run; a different sample of it every step.

- `gpt2_kernels.cuh` — the network. Kernels, layout, forward, backward, data.
  Shared by every variant, so a fix lands everywhere at once.
- One `.cu` per Bayesian method, ~150 lines each, containing *only* how the
  weights are sampled and updated: Bayes-by-Backprop, Vadam, low-rank + diagonal
  covariance, and a deep-ensemble control arm.
- `train_bnn.c` — the deterministic CPU reference every kernel was checked
  against. `dev/` is where each kernel was built and validated one at a time.

```
python3 -m venv .venv && .venv/bin/pip install tiktoken numpy tqdm requests
.venv/bin/python3 data/tinyshakespeare.py --model=gpt-2  # download Shakespeare
.venv/bin/python3 data/prepare_char.py                   # char-level .bin the trainers read

make train_bnn_gpu && ./train_bnn_gpu overfit   # memorize one batch (smoke test)
make train_bnn_gpu && ./train_bnn_gpu           # train
make test                                       # run every variant, print one table
```

Built for an RTX 2060 (`sm_75`); override with `make ARCH=sm_XX`.

`paper/` holds the write-up: `make -C paper`. Its tables are generated from
`runs/`, never typed. Read it here: [`paper/main.pdf`](paper/main.pdf) (CI-built, placeholder tables).

The deterministic scaffolding (`train_gpt2.c`, the data pipeline) follows
Karpathy's [llm.c](https://github.com/karpathy/llm.c). MIT, same as llm.c.
