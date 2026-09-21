# Mailbox backend comparison — 2026-09-21

This is a comparison of mailbox pipelines on the same 200 emails, not an
identical-prompt or identical-kernel model microbenchmark. No messages were
moved and no production classification settings were changed.

## Results

Final runs after stopping the unrelated CUDA compilation:

| Backend                       | Correct / 156 | Logical calls | Questions | Batches | Chunks | Call wall |     Run |   Total |
| ----------------------------- | ------------: | ------------: | --------: | ------: | -----: | --------: | ------: | ------: |
| Hosted Jev 1.13.0             |   142 (91.0%) |           200 |       200 |      13 |    200 |    5.456s |  5.463s |  5.540s |
| Laya BF16 / FlashAttention    |    87 (55.8%) |           704 |     1,760 |      56 |    352 |   24.364s | 27.483s | 28.675s |
| OpenJev BF16 / FlashAttention |    20 (12.8%) |           520 |       520 |      33 |    520 |    3.587s |  8.069s |  8.466s |
| OpenJev F32                   |    20 (12.8%) |           520 |       520 |     130 |    520 |   21.090s | 25.611s | 26.027s |

Jev made **200 HTTP attempts, 200 successes and zero retries** in the final
run. Summed individual HTTP request latency was 64.910s, averaging 324.5ms;
concurrency reduced the elapsed call interval to 5.456s. Across the initial
and final runs, this experiment made **400 hosted inference calls and 400
HTTP attempts**, without retries. Local logical calls are not HTTP requests.

OpenJev abstained on 134/200 emails in BF16 and 133/200 in F32. Among labeled
emails, this is 106/156 and 105/156, respectively. Accuracy conditional on
answering was 20/50 (40.0%) for BF16 and 20/51 (39.2%) for F32. Both modes
abstained on 318/520 individual chunks. Laya and hosted Jev did not abstain.

The initial run scored Jev 143/156, Laya 87/156 and OpenJev F32 20/156.
Initial timings overlapped a CUDA compilation and OpenJev batches stopped
at email boundaries, so those timings are not used above. The final harness
batches OpenJev chunks across emails. Labels and rules stayed unchanged.

The current OpenJev configuration is **not an accuracy improvement**. Its
faster BF16 path makes inference inexpensive, but most mail is rejected as
insufficient evidence. Discarding abstention and forcing the highest-scoring
substantive category is only a diagnostic, not a deployed policy. These
forced-category scores were 27/156 in F32 and 28/156 in BF16. Final email
predictions agreed between precisions on 199/200 messages; the difference
was one additional BF16 abstention. Across repeat runs, Laya and F32
OpenJev predictions were unchanged, while hosted Jev changed 3/200 labels.
These
results do not establish that a different OpenJev prompt or calibration
would behave the same way. No production backend, rules or aggregation
policy was replaced.

Aggregate machine-readable measurements, including loading and full process
timers, are in [email-backend-benchmark.json](email-backend-benchmark.json).

## Frozen sample and reference

The sample contains 200 files drawn without replacement from 23,641 regular
files in recursive Maildir `cur`/`new` directories, using Python
`random.Random(20260921)` on sorted paths. These are mailbox files, not
deduplicated conversations. Each copied message has a source path and SHA-256
in a private manifest. The existing Rust MIME parser and body cleanup decode
the sample. Sender, subject, date and body are separate; recipient is omitted
from model input. The same unchanged root `email.toml` supplies all 17 rules.

Reference labels were assistant-reviewed and frozen before sampled-email
predictions. There are 156 clear labels and 44 unresolved cases. Unresolved
cases include missing owner context, uncertain delivery-versus-receipt scope,
mixed purposes and insufficient content. These cases remain in every timed
run but are excluded from accuracy for **all** backends. An abstention on a
labeled email counts as incorrect. This is a preliminary assistant audit,
not user-adjudicated ground truth or full-mailbox accuracy.

The labeled set contains: bulk 43, receipts 25, other 20, ops 20, careers 14,
security 10, income 10, fiscal 6, identity 4, travel 3 and health 1. It cannot
establish accuracy for capture, bills, clients, papers, equity or household.

Frozen label SHA-256:
`6cb28a20e0359c312de132aa2318bc2b19e44e08df0d2c5e2acf2858288c3df8`.

## Protocol and timing definitions

- **Laya:** `convaiinnovations/laya`, `typed-decisions`, CUDA BF16 with
  FlashAttention, batch 16. Existing token-aware chunks, groups of at most
  five choices, winner tournament and character-weighted chunk pooling.
  Cached revision: `c5d78730f3493e4fe16d61507ef4b78eef7318cf`.
- **Hosted Jev:** `jev-1.13.0`, full cleaned email, one 17-choice question,
  16 concurrent HTTP requests. Existing caller-side probability normalization
  and argmax; raw provider decisions remain in private results.
- **OpenJev:** `heman10x/rlcd-modernbert-151m`, checkpoint revision
  `8af2496eb63c7fa66d7d234e1f62629380030eb4`. Same complete category descriptions
  and email state, one native 17-choice question plus abstention per chunk.
  The full empty-body request exhausts the default 512-token context, so
  this benchmark explicitly uses a **1,024-token budget**. The tokenizer has
  a 1,025-token guard; inputs reaching that extra token are split again.
  No accepted input is silently truncated. This longer context is an
  experimental mailbox configuration, beyond the original 512-token parity
  validation. Native calibrated probabilities, including abstention mass,
  are character-weighted across chunks before choosing an email category.
  CUDA F32 uses batches of 4; approximate CUDA BF16/FlashAttention uses 16.
  Batches span email boundaries.

All timed runs use the RTX 3080 Ti, release builds with `cuda,flash-attn`,
four Rayon threads, cached checkpoints and a fresh process. There is no
separate warmup: first-call setup is included. Local GPU workloads run
sequentially. Executable and source hashes are retained privately because
the OpenJev implementation was being updated concurrently by another agent.

**Logical calls** count request objects (native inputs for OpenJev), not
HTTP attempts or Rust batch invocations. **Questions** count individual
choice questions; Laya puts four first-round questions in one request.
**Call wall time** sums nonoverlapping backend invocation intervals, including
waiting for all concurrent Jev requests. OpenJev uses pretokenized native
inputs, so its tokenization is outside this interval; Laya's typed call
includes tokenization. **Run time** includes chunking, tokenization,
inference and aggregation. **Total** also includes mailbox parsing, model
loading and result serialization, but excludes compilation, sample creation
and reference review. The additional process timer includes process teardown.

Jev also records the sum of individual request latencies. These overlap
under concurrency and must not be compared directly to elapsed wall time.
Retries count toward both HTTP attempts and request latency. Local models
make zero HTTP inference calls.

## Reproduction

Private inputs, labels, source snapshots, executable snapshots, raw outputs,
per-batch timings and manifests are under
`~/.local/state/vs1-email/three-backends-200-20260921/`.
They are deliberately not committed.

```bash
nix develop -c cargo test -p vs1-email --example backend_benchmark
nix develop -c cargo clippy -p vs1-email --example backend_benchmark -- -D warnings
nix develop -c cargo build --release -p vs1-email \
  --features cuda,flash-attn --example backend_benchmark

# Repeat for laya, jev, openjev and openjev-bf16; choose a fresh run name.
LD_LIBRARY_PATH=/run/opengl-driver/lib RAYON_NUM_THREADS=4 \
  target/release/examples/backend_benchmark \
  ~/.local/state/vs1-email/three-backends-200-20260921 laya laya-repeat
```

Hosted runs require the existing `TYPESAFE_API_KEY`. OpenJev loads the pinned
local checkpoint under `artifacts/openjev/checkpoint`. Output files use
create-new semantics. Tests were written failing first for abstention-aware
scoring, native probability pooling and explicit precision selection.
