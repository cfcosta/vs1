# vs1

Runs [laya](https://github.com/NandhaKishorM/laya) System One decision
models on [candle](https://github.com/huggingface/candle), as a Rust
library and a small `vs1` command.

A System One model answers typed questions about a piece of _state_ in
a single, non-autoregressive forward pass. Nothing is generated: every
answer is read straight off the encoder as a calibrated probability, so
there is nothing to parse and nothing to hallucinate. laya's checkpoints
are a ModernBERT encoder with a small decision head, trained with
reinforcement learning against strictly proper scoring rules.

The request and response follow the TypeSafe `systemone` shape that
laya is API-compatible with.

## Primitives

| Question | Asks                        | Answer                                                            |
| -------- | --------------------------- | ----------------------------------------------------------------- |
| `choice` | pick one option from a set  | `choice`, `probabilities`, `confidence`                           |
| `score`  | place the state on a rubric | `score` (expected level), `legend`, `probabilities`, `confidence` |
| `noul`   | is this statement true?     | `noul` (P(true)), `confidence`                                    |

Every answer also carries `action.act_probability`, laya's learned
"act rather than escalate" signal.

## Usage

```rust
use vs1::{Question, SystemOne, SystemOneRequest};

let model: SystemOne = SystemOne::from("convaiinnovations/laya").try_into()?;

let request = SystemOneRequest::new("Please refund the duplicate charge today.")
    .question("wants_refund", Question::noul("Does the sender ask for money back?"))
    .question(
        "department",
        Question::choice(
            "Which team handles this?",
            [("billing", "invoices and payments"), ("other", "anything else")],
        ),
    )
    .question(
        "urgency",
        Question::score("How urgent is this?", ["not urgent", "soon", "critical"]),
    );

let response = model.system_one(&request)?;
println!("{}", serde_json::to_string_pretty(&response)?);
```

Requests and responses are `serde` types, so the same JSON a TypeSafe
client sends works here:

```json
{
  "state": { "from": "user@acme.com", "body": "We were billed twice." },
  "questions": {
    "churn_risk": { "type": "noul", "instructions": "Does the user threaten to leave?" }
  }
}
```

`system_one_batch` packs the questions of many requests into as few
forward passes as the batch size allows, which is the shape a reranker
needs: one relevance question about each of N candidate chunks.

## Checkpoints

| Checkpoint             | Builder                                     | Encoder          |
| ---------------------- | ------------------------------------------- | ---------------- |
| `laya`                 | `SystemOne::from("convaiinnovations/laya")` | ModernBERT-large |
| `laya-multilingual`    | `.with_subfolder("multilingual")`           | mmBERT-base      |
| `laya-typed-decisions` | `.with_subfolder("typed-decisions")`        | ModernBERT-large |

A local checkpoint directory works in place of the repo id. Weights are
memory-mapped from `model.safetensors`; the encoder is a ModernBERT port
vendored from docbert's `docbert-pylate` (itself a fork of LightOn's
pylate-rs, MIT, see `LICENSE-PYLATE`), and the head is a
weight-for-weight port of laya's PyTorch `TransformerEncoder` + scorer +
action head.

## Fidelity

Token sequences are built exactly as laya's Python `build_sequence`
does, including its option budget squeeze and CPython's `json.dumps`
layout for structured state. On the CPU in F32 the answers match the
Python implementation to within `5e-5`. On CUDA the model runs in BF16
by default.

## What it cannot do yet

The crate started inside [docbert](https://github.com/cfcosta/docbert)
to judge its retrieval candidates: fetch
three times as many chunks as asked for, ask "does the passage contain
the information the search query is looking for?" about each, and keep
the most probable. Measured zero-shot on BeIR SciFact and NFCorpus with
`lightonai/GTE-ModernColBERT-v1` as the retriever, the shipped
checkpoints do not do that job:

| Order of the 30-candidate pool (SciFact, hybrid) | nDCG@10 | R@10 |
| ------------------------------------------------ | ------- | ---- |
| retriever alone                                  | 0.60    | 0.70 |
| sorted by the `noul` answer                      | 0.16    | 0.35 |
| `1 / rank + 0.5 * answer`                        | 0.61    | 0.72 |

The answer separates relevant from irrelevant passages with an AUC of
0.5 to 0.7 (0.86 for the retriever's own rank), across four question
framings and both the English and multilingual checkpoints, while
costing about 440 ms per query for 30 candidates in BF16 on an RTX
3080 Ti. This matches laya's own note that the base checkpoints are a
fast base to specialise rather than a zero-shot decision engine. The
retrieval integration was therefore not kept; a checkpoint fine-tuned
on query and passage relevance pairs is what would make it worth
revisiting.

## Backends

Feature flags: `cuda`, `metal`, `mkl`, `accelerate`. `flash-attn`
additionally builds the encoder's packed flash-attention paths (needs
nvcc and cutlass); the masked path the decision head uses is fine
without it.

## Command line

```bash
cargo run --release -- request.json
cargo run --release --features cuda -- --device cuda --dtype bf16 request.json
```

`request.json` holds one request or an array of them in the JSON shape
above; every response is printed as JSON. `--dump-ids` also prints the
token sequence built for each question.

## Benchmarks

```bash
VS1_BENCH=1 cargo bench --features cuda
VS1_BENCH=1 VS1_BENCH_DTYPE=f32 cargo bench --features cuda
VS1_BENCH=1 VS1_BENCH_SUBFOLDER=multilingual cargo bench --features cuda
```

## Nix

```bash
nix build            # CPU
nix build .#vs1-cuda
nix build .#vs1-metal
nix develop          # toolchain, formatter, cargo-deny, cargo-nextest, CUDA on Linux
nix fmt
```
