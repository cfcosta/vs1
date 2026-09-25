# GLiNER2.5-Decide

`GlinerDecide` runs [GLiNER2.5-Decide](https://huggingface.co/fastino/GLiNER2.5-Decide)
locally in Rust. It is a DeBERTa-v3-large encoder with GLiNER2's classifier
head: the caller names labels at request time, and the model picks one in a
single forward pass. The default repository is pinned to revision
`7ee5da4c2415e32259bcdc0b1a7367c32ce8d6f6`. A local checkpoint directory can
be passed instead. Custom repository IDs use their `main` revision.

## Run

```bash
cargo run --release -p vs1 -- \
  --backend gliner-decide research/gliner-decide/request.json

cargo run --release -p vs1 --features cuda -- \
  --backend gliner-decide --device cuda research/gliner-decide/request.json

# Inspect the exact token IDs, tasks and [L] marker positions.
target/release/vs1 --backend gliner-decide --dump-ids research/gliner-decide/request.json
```

`--batch-size` sets requests per forward pass (default 8). Requests are sorted
by length before batching, and responses keep input order. `--max-len` sets the
token budget per request (default 512).

CUDA defaults to BF16. CPU and Metal use F32. `--dtype f32` selects reference
precision on CUDA, and BF16 requires CUDA. The encoder is vs1's copy of
candle's DeBERTa-v2 (`crates/vs1/src/deberta.rs`). Its matmuls run in the
weight dtype. Attention scores, position terms, the mask and the softmax run in
F32. BF16 is approximate. On 33 validation questions, it kept every label, and
the largest probability difference from upstream F32 was 0.0094. This is a
measured tolerance, not a guarantee for all inputs.

## How requests map to the model

A request becomes one upstream `classify_text(text, tasks)` call. Every question
is one classification task, and all of a request's tasks share one encoder
pass. As upstream, each task can see the others' labels. The same question may
score differently alone than beside other questions.

| Question | Task labels      | Label descriptions                    | Answer                        |
| -------- | ---------------- | ------------------------------------- | ----------------------------- |
| `choice` | option IDs       | non-empty option descriptions         | argmax, softmax probabilities |
| `score`  | `"0"` .. `"n-1"` | each criterion                        | expected index, probabilities |
| `noul`   | `"yes"`, `"no"`  | the `true` / `false` criteria, if any | `P(yes)`                      |

The task name is the question ID, and non-empty instructions are its prompt
(`name: instructions`). Name questions the way the model card names heads
(`intent`, `urgency`, `needs_human`), because the model reads the name.
State is lowercased and split into words as upstream does. A final `.` is added
when it does not end in `.`, `!` or `?`.

Probabilities are the raw softmax over one task's labels. There is no
calibration file and no abstention. Confidence is vs1's normalized entropy of
that distribution. The model never produces Laya's action signal. Multi-label
tasks and few-shot examples are not exposed through the typed API.

## Limits

Questions, labels and descriptions are always kept whole. If they do not fit in
`--max-len`, the request fails. State is truncated on the right to fill the
rest. The dropped subword count is reported per question in
`usage.dropped_state_tokens`, and `request_fits` returns false in that case.
Upstream does not truncate by default. The 512 default is DeBERTa-v3's
pretraining length. The model uses only relative positions, so longer budgets
run. One 742-token request matched upstream in validation. Quality at longer
lengths has not been measured.

GLiNER2 marker strings (`[P]`, `[L]`, `[SEP_TEXT]`, `[DESCRIPTION]`, …) in
question IDs, instructions, labels, descriptions or state are rejected. Labels
must be distinct and non-empty, and each question needs at least two.

Parity evidence is in [the validation report](../research/gliner-decide/README.md).
