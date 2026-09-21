# vs1

Runs [laya](https://github.com/NandhaKishorM/laya) System One decision
models on [candle](https://github.com/huggingface/candle), as a Rust
library and a small `vs1` command. The same build also supports hosted
Jev through the typed request/response interface.

A System One model answers typed questions about a piece of _state_ in
a single, non-autoregressive forward pass. Nothing is generated: every
answer is read straight off the encoder as a calibrated probability, so
there is nothing to parse and nothing to hallucinate. laya's checkpoints
are a ModernBERT encoder with a small decision head, trained with
reinforcement learning against strictly proper scoring rules.

The request and response follow the TypeSafe `systemone` shape that
laya is API-compatible with.

## Workspace

- `crates/vs1`: decision-model library and `vs1` CLI (the default Cargo package).
- `crates/vs1-browser`: browser automation CLI, depending directly on `vs1`.
- `crates/vs1-email`: local Maildir classification, with caller-selected inference.

Run `cargo test --workspace` and `cargo clippy --workspace --all-targets` to
check the workspace. All crates use the root `Cargo.lock` and `target/` directory.

## Caller-selected backend

`SystemOne` remains the local laya API. `JevClient` is included in normal builds; `DecisionModel` wraps either backend and exposes
`system_one` and `system_one_batch`. Backend and model selection are explicit:
there is no automatic cloud fallback, environment-based library selection or
checkpoint download when constructing a Jev client.

```rust
use vs1::{DecisionModel, JevClient, Question, SystemOneRequest};

let model: DecisionModel = JevClient::new(
    std::env::var("TYPESAFE_API_KEY")?,
    "jev-1.13.0",
)?.with_concurrency(16)?.into();
// Local alternative: let model: DecisionModel = local_system_one.into();
let request = SystemOneRequest::new("Please refund the duplicate charge.")
    .question("refund", Question::noul("Is a refund requested?"));
let response = model.system_one(&request)?;
if let DecisionModel::Jev(client) = &model {
    println!("{}", serde_json::to_string(&client.stats())?);
}
```

The client-selected hosted model overrides the optional request `model` field;
the response reports the actual provider model. The library does not read
credentials from the environment: that is the caller's choice, as above.
Jev responses retain provider probabilities and choices without normalization;
`Answer::action()` is a laya-only signal and is absent for hosted responses.

A Jev batch is bounded concurrent HTTP requests, not a provider batch endpoint.
Outputs retain input order. `JevClient::stats()` counts logical `calls`, HTTP
`attempts` (including failures), `retries`, parsed `successes`, and `questions`.
Token usage remains on each response. The client retries HTTP 429/503/529 up to
three total attempts, respects numeric Retry-After seconds up to 60, and otherwise
uses short backoff. It does not retry authentication, connection or decoding
errors. Requests time out after 30 seconds; redirects are disabled. A batch
error can occur after other requests in that batch succeeded; counters include
the work already attempted. Error strings omit credentials and provider bodies.

```bash
# TYPESAFE_API_KEY must already be set in your environment.
cargo run --release -p vs1 -- \
  --backend jev --model jev-1.13.0 --batch-size 16 request.json
nix run .#vs1 -- --backend jev --model jev-1.13.0 request.json
```

The core and email CLIs default to `--backend laya`; the browser keeps its hosted
default. All accept caller-selected model IDs. The CLIs print hosted call
counters on stderr; stdout remains the typed response/report. Jev's context and
option limits are enforced by the provider, not laya's tokenizer. No local
weights or GPU are required for hosted inference, though the core crate still
includes its CPU laya dependencies. The normal Nix packages (`vs1`, `vs1-email`, `vs1-browser`) include both
backends. No extra package or feature flag is needed to select Jev at runtime.
Acceleration features still control support for local GPU devices.

## Primitives

| Question | Asks                        | Answer                                                            |
| -------- | --------------------------- | ----------------------------------------------------------------- |
| `choice` | pick one option from a set  | `choice`, `probabilities`, `confidence`                           |
| `score`  | place the state on a rubric | `score` (expected level), `legend`, `probabilities`, `confidence` |
| `noul`   | is this statement true?     | `noul` (P(true))                                                  |

The JSON matches TypeSafe's `systemone` response field for field:
choice and score answers carry `confidence`, noul answers do not, and
`legend` echoes each level exactly as the question wrote it, text or
structure. laya's extra "act rather than escalate" signal is kept for
Rust callers as `Answer::action()` and is never serialised. Requests
may leave `instructions` out or set it to `null`, as TypeSafe allows.

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

For a browser-agent CLI like Jev Ultrafast's `examples/run.py`, see
[`crates/vs1-browser`](crates/vs1-browser/README.md). It accepts a URL
and a natural-language goal, calls this library directly from Rust, and executes
only observed browser actions through Chrome DevTools Protocol. It includes
independent outcome checks and a local-versus-Jev performance report; wire-format
compatibility does not imply equivalent browser-task accuracy.

```bash
cargo run --release -- request.json
cargo run --release --features cuda -- --device cuda --dtype bf16 request.json
```

`request.json` holds one request or an array of them in the JSON shape
above; the output is one response or an array of them, exactly as
TypeSafe would return. `--dump-ids` wraps each response with the token
sequence built for each question.

JSON scenario files with setup scripts and outcome checks are in
[`examples/`](examples/README.md); run them with `vs1-browser --scenario`.

## Benchmarks

```bash
VS1_BENCH=1 cargo bench --features cuda
VS1_BENCH=1 VS1_BENCH_DTYPE=f32 cargo bench --features cuda
VS1_BENCH=1 VS1_BENCH_SUBFOLDER=multilingual cargo bench --features cuda
```

## Nix

```bash
nix build            # CPU
nix build .#vs1-browser       # hosted Jev, no checkpoint loading
nix build .#vs1-browser-local # optional local CPU inference
nix run .#vs1-browser -- --help
nix build .#vs1-cuda
nix build .#vs1-metal
nix build .#vs1-browser-cuda
nix develop          # toolchain, formatter, cargo-deny, cargo-nextest, CUDA on Linux
nix fmt
```

The browser CLI defaults to Jev; local inference requires a build with the `local`
Cargo feature and `--backend local`. Accelerator features imply `local`.
Both CLIs have `-cuda`, `-flash-attn`, and `-metal` Nix package variants;
select a backend supported by your host. The browser CLI connects to an external
Chrome/Chromium instance through CDP; the package does not bundle a browser.

## Email categorization

[`vs1-email`](crates/vs1-email/README.md) categorizes a local Maildir mailbox with local
laya inference and ordered TOML `[[rules]]`. Start with
[`examples/email-rules.toml`](examples/email-rules.toml), fill in owner context,
and run `nix run .#vs1-email -- --config email.toml --dry-run --mailbox ~/Mail/account`.
The selected directory can be a Maildir or a sync root; all descendant Maildir folders are included.
No host or credentials are needed.
Dry-run emits JSON proposals without changing messages; execution without
`--dry-run` returns an explicit not-implemented error.
