# OpenJev (Verdict)

`OpenJev` runs [OpenJev (Verdict)](https://huggingface.co/heman10x/rlcd-modernbert-151m)
locally in Rust, using vs1's existing ModernBERT encoder. The Hugging Face
repository still uses its original `rlcd-modernbert-151m` identifier; API
responses identify the model as `OpenJev (Verdict)`.

The default repository is pinned to revision
`8af2496eb63c7fa66d7d234e1f62629380030eb4`. Its weights, tokenizer, configuration
and calibration artifact are loaded together. A local checkpoint directory can
be passed instead. Custom repository IDs use their `main` revision.

## Run

```bash
cargo run --release -p vs1 -- \
  --backend openjev research/openjev/request.json

cargo run --release -p vs1 --features flash-attn -- \
  --backend openjev --device cuda research/openjev/request.json

# Use an already downloaded checkpoint and inspect the exact model inputs.
target/release/vs1 --backend openjev --device cuda \
  --model artifacts/openjev/checkpoint --dump-ids research/openjev/request.json
```

This selector is available in the core `vs1` CLI. `--model` accepts a local
directory or Hugging Face repository; `--batch-size` controls questions per
forward pass. No API key or Python runtime is required. CUDA builds with
`flash-attn` default to BF16 weights and packed FlashAttention. CPU and builds
without `flash-attn` default to F32. `--dtype f32` selects reference precision;
explicit `--dtype bf16` requires CUDA and `flash-attn`.

BF16 is approximate: in the 95-case validation it preserved every choice and
abstention outcome, with maximum expected-score error 0.0234 on a 0–4 scale and
maximum noul error 0.0096. Full probability vectors differed from F32 by up to
0.0118 and between singleton/batch execution by up to 0.0228. Native score
argmaxes can change near ties even when the returned expected score barely
moves. This is a measured tolerance, not a guarantee for all inputs.

## Library and multiple resident models

```rust
use vs1::{DecisionModel, OpenJev, SystemOne};

let laya: SystemOne = SystemOne::from(vs1::DEFAULT_REPO_ID).try_into()?;
let openjev: OpenJev = OpenJev::from(vs1::openjev::DEFAULT_REPO_ID).try_into()?;
let models: [DecisionModel; 2] = [laya.into(), openjev.into()];
// Each instance owns its weights, tokenizer, calibration and device.
// Call models[index].system_one(&request) or .system_one_batch(&requests).
```

Builders accept `.with_device(device.clone())` to share a device. There is no
global active model or automatic routing between models. Loading both consumes
memory for both checkpoints. `DecisionModel::local()` remains the existing
Laya-specific accessor; match `DecisionModel::OpenJev(model)` for OpenJev's native
facilities. Hosted Jev is a separate backend.

## Answer semantics

The shared API accepts the existing `choice`, `score` and `noul` requests.
Each question adds an `insufficient evidence` candidate. When it wins, the
response uses the explicit extension:

```json
{
  "type": "abstain",
  "question_type": "choice",
  "probabilities": {
    "first": 0.1,
    "second": 0.2,
    "__insufficient_evidence__": 0.7
  },
  "reason": "insufficient_evidence"
}
```

Callers must handle `Answer::Abstain`; existing exhaustive Rust matches need an
additional arm. `answer.abstention()` exposes the abstention result. It has no
choice, score, noul value, confidence or Laya action signal. Existing Laya/Jev
answer JSON stays unchanged.

For non-abstained shared answers, probabilities are **conditional on sufficient
evidence**: remove the abstention candidate and renormalize. Choice preserves
the winning option ID; score is the expected zero-based rubric index; noul is
conditional P(true). Choice/score confidence uses vs1's normalized entropy of
that conditional distribution. OpenJev never produces Laya's action signal.

To retain all probability mass, use `build_input(state, id, question)` followed
by `predict(&inputs)`. `OpenJevPrediction` contains every raw logit, the full
calibrated distribution including abstention, the selected ID, temperature and
input-token count. The shared adapter's probabilities should not be substituted
for those native probabilities in calibration evaluations.

Choice and score support 2–24 substantive options, plus abstention. Noul uses
the proposition to construct its true/false labels; custom noul criteria are
rejected. The default context limit is 512 tokens, with right truncation of
the complete upstream prompt. Truncated candidate headers are rejected.
The Rust builder can override the limit up to the encoder's 8192 positions;
only the default 512-token setting was validated. Reserved marker strings
`<<LABEL>>` and `<<SEP>>`, duplicate option IDs and reserved abstention IDs are
rejected. These are explicit errors, with no option dropping or cloud fallback.

Implementation details, parity evidence and the BF16 re-evaluation are
in [the validation report](../research/openjev/README.md).
