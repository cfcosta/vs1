# Local labeled-example retrieval experiment

The retained experiment adds two similar labeled messages to laya's state as
`labeled_examples`, separate from the target `email`. Category rules, the
five-choice tournament, and length-weighted chunk pooling remain unchanged.
The normal `vs1-email` command does not enable this experiment.

`prepare.py` runs a frozen multilingual MiniLM encoder locally on CUDA. It
splits subject plus body into nonoverlapping 120-token chunks, normalizes each
chunk vector, takes their token-count-weighted sum, and normalizes that sum.
Cosine similarity selects two training messages. Each example contains its
subject, the first 160 body characters, and its reviewed reference category.
The model revision is pinned in the script. Python is an offline research
utility; the Rust inference path does not launch it.

Training JSON is an array of records with `subject`, `sender` (normalized email
address), `body`, and `label`. Targets have `subject`, `sender`, `date`, and
`body`; target labels are never read. Subject/date must uniquely identify each
target. The evaluation helper rejects targets sharing a sender or a normalized
body template with training data (five-word shingle Jaccard similarity >= 0.5).
Keep evaluation labels frozen before inference and separate from training.

A tested research environment used Python 3.13, PyTorch 2.11.0+cu128,
sentence-transformers 5.7.0, transformers 4.57.6, and NumPy 2.5.3. Install those
in an isolated environment with CUDA support. All email data remains local.

```sh
python3 -m unittest discover -s research/email-retrieval
python3 research/email-retrieval/prepare.py training.json targets.json examples.json
nix develop -c cargo build --release -p vs1-email --features cuda,flash-attn --example retrieval_audit
LD_LIBRARY_PATH=/run/opengl-driver/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} \
  target/release/examples/retrieval_audit email.toml SAMPLE_MAILDIR NEW_OUTPUT_DIR typed-decisions examples.json
```

The audit expects a frozen, clean 100-message Maildir sample and selects only
messages named by the examples map. It performs baseline/retrieval/retrieval/
baseline passes on exactly the same messages, with CUDA BF16, flash attention,
and batch size 16. Both fit checks and inference include the added examples;
chunk boundaries may therefore differ. It writes raw reports and timing
summaries to a new directory, and never changes mailbox files. Its timings
exclude model loading and example preparation. Treat outputs as private: the
audit also exports decoded selected messages for reproducibility.

See [experiment findings](../../docs/email-experiments.md#10-checkpoint-retrieval-embedding-and-larger-model-experiments)
for results and limitations. The private reproduction inputs and rejected
prototypes are under `~/.local/state/vs1-email/next-four-20260921/`.

## Three-backend ablations on a frozen sample

`backend_benchmark` also supports baseline and retrieval comparisons on an
arbitrary frozen sample. It remains a research example; the normal CLI does not
load example banks or launch an embedding model.

The private benchmark directory contains:

- `sample/`: a frozen Maildir with stable, lexically ordered numeric filenames.
- `benchmark.json`: `{"messages":300}` (defaults to 200 if absent).
- `email.toml`: frozen category definitions and owner context.
- `labels.json`: filename-to-category map, with `null` for unresolved references.
- `examples.json`: two selected examples per subject/newline/date key, produced
  by `prepare.py` using a separate reviewed training bank.

Freeze labels before inference. Exclude sender and template overlap between
training and test messages, check duplicate subject/date keys, and preserve raw
file hashes. The encoder consumes training and target text; target reference
labels must never enter example retrieval or inference.

Build with CUDA and FlashAttention:

```sh
nix develop -c cargo build --release -p vs1-email --features cuda,flash-attn --example backend_benchmark
```

The command is `backend_benchmark ROOT BACKEND RUN_NAME [AUDIT_MODE] [EXAMPLE_MODE]`.
`AUDIT_MODE` is `original` for laya/Jev and `compact512` for the native OpenJev
comparison. Example modes are:

| Mode      | Context during inference           | Context used to choose chunk boundaries |
| --------- | ---------------------------------- | --------------------------------------- |
| `none`    | None                               | None                                    |
| `matched` | None                               | Full examples                           |
| `full`    | Example subject, body and category | Full examples                           |
| `text`    | Example subject and body           | Full examples                           |
| `labels`  | Example category only              | Full examples                           |

`labels-native` sends the same category-only example context as `labels`, but
fits target chunks using that smaller context. Unlike `labels`, it is an
efficiency experiment and intentionally does not share full-example chunk
boundaries. It leaves the baseline and other ablations unchanged.

Jev is baseline-only: whole cleaned emails, no retrieval or local chunking.
OpenJev uses F32 CUDA, batch four; laya uses BF16 CUDA, batch 16. The normal
baseline and matched baseline are both necessary: adding examples otherwise
changes chunk boundaries and confounds context effects with rechunking.
The ablations share the token ceiling and target chunks, not identical sequence
lengths. Do not pad removed fields with arbitrary prose.

For tight contexts, `openjev ... compact512 prepare` performs tokenizer-only
fitting and writes `examples-fitted.json`. It repeatedly halves the longest
example subject/body, preserving labels and order. It drops a last example only
if all strings are empty and the context still does not fit. It reserves up to
64 tokens for the target body, or half the space remaining after metadata if
less is available. No predictions or reference labels guide this operation.
Preserve the original example file, then use the same fitted examples for both
local backends and every ablation. This fitting is separate from retrieval;
record how many examples/characters were removed.

Outputs include private raw results, summary metrics and `.chunks.json` files.
Compare chunk-state arrays exactly across `matched`, `full`, `text` and `labels`
within each backend. All target body text must be covered without truncation.
Record retrieval preparation time separately from classifier call/run/total
time. Local logical requests and questions are not HTTP calls; hosted statistics
include attempts and retries. OpenJev call timing excludes input preparation,
while laya's batch API includes tokenization, so cross-backend call timing has
different boundaries. End-to-end totals are also recorded.

## Neighbor agreement routing

`policies.py` contains the opt-in research policy from the follow-up experiment.
Pass the top five neighbors in descending cosine similarity order as
`(category, similarity)` pairs. `vote()` sums nonnegative similarities by category;
ties follow the first neighbor. `needs_laya()` decides the route before inference:
use Laya's unchanged baseline request if the nearest two labels disagree, fewer
than two neighbors exist, or all vote weights are zero. Otherwise use the vote.
`disagreement_hybrid()` combines a saved Laya result with that decision for replay.

Routing is separate from adding examples to a Laya prompt. It skips model calls
for agreement cases and preserves the normal request on disagreement cases.
The production CLI does not load this policy. See the
[follow-up report](../../docs/email-retrieval-followup.md) for fresh-sample results,
diagnostic regressions, bank coverage limits and actual routed call counts.
