# GPT-2 replica — input/output data pipeline

Goal: match llm.c's own data pipeline exactly (dataset, tokenizer, `.bin` shard
format, batching convention), so this replica is directly comparable/debuggable
against llm.c as we build the model piece by piece. No model or training loop
here — just data in, data out.

`tinyshakespeare/` (raw text + `.bin` shards) and `.venv/` are gitignored —
regenerate with the commands below.

## 1. Dataset — TinyShakespeare

llm.c's own quick/dev-test dataset (its README's "quick start" walks through
exactly this). `tinyshakespeare.py` and `data_common.py` in this folder are
**unmodified copies** of `llm.c/dev/data/{tinyshakespeare,data_common}.py` —
not a reimplementation — so tokenization and the `.bin` format are byte-for-byte
identical to what llm.c itself produces. `llm.c/` stays untouched (read-only
reference); we just run our own copy of its script.

```
python3 -m venv .venv && .venv/bin/pip install tiktoken numpy tqdm requests transformers
.venv/bin/python3 tinyshakespeare.py --model=gpt-2
```

Output (matches llm.c's README example output exactly):
- `tinyshakespeare/tiny_shakespeare.txt` — raw text, downloaded from the same
  GitHub source llm.c uses (`karpathy/char-rnn`)
- `tinyshakespeare/tiny_shakespeare_val.bin` — 32,768 tokens (first ~10%)
- `tinyshakespeare/tiny_shakespeare_train.bin` — 305,260 tokens
- **338,028 tokens total**

## 2. Tokenizer — GPT-2 BPE via tiktoken

`tiktoken.get_encoding("gpt2")`, **vocab size 50,257** (`model->config.vocab_size`
in `train_gpt2.c`). Text is split on blank lines into "documents"; an
`<|endoftext|>` token (id `50256`) is inserted before each one, then encoded
with `enc.encode_ordinary()` (no other special-token handling).

## 3. `.bin` shard format — matches `data_common.py::write_datafile` exactly

- 256×`int32` header: `[magic=20240520, version=1, num_tokens, 0, 0, ...]`
- followed by `num_tokens` × `uint16` token ids

Verified by `check_pipeline.py` against `llmc/dataloader.h`'s own header checks
(same magic/version, file size == `1024 + num_tokens*2` bytes).

## 4. Dataloader — `dataloader.py`, ports `llmc/dataloader.h` exactly

Single-shard, single-process case (no distributed sharding/shuffling — not
needed at this scale). Per batch: reads `B*T+1` **contiguous** tokens,
`inputs = tokens[0:B*T]`, `targets = tokens[1:B*T+1]`, both reshaped to `(B,T)`.
This is llm.c's actual convention — note it's a shift over the *flattened*
B\*T stream, so row *b*'s last target token equals row *b+1*'s first input
token (not independently shifted per row). Batches themselves overlap by
exactly 1 token for the same reason (`total_batch_size_bytes` advances by
`B*T`, but each read grabs `B*T+1`).

Default `B=4, T=64` — llm.c's own documented quick/CPU-demo config (its
README: "batch size 4, context length only 64"), not the full `B=4, T=1024`
mainline default — kept small on purpose for iterating on the architecture.

## 5. Report — `check_pipeline.py`

```
.venv/bin/python3 check_pipeline.py
```

Prints total token count, vocab size, confirms `inputs`/`targets` are `(4,64)`
int32 arrays with values in `[0, 50257)` — exactly what `gpt2_forward(int*
inputs, int* targets, size_t B, size_t T)` in `train_gpt2.c` expects — and
decodes one sample batch's input/target side by side so you can see the
1-token shift directly in text.
