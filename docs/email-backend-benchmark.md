# Mailbox backend comparison — 2026-09-21

This is a comparison of mailbox pipelines on the same 200 emails, not an
identical-prompt or identical-kernel model microbenchmark. No messages were
moved and no production classification settings were changed.

## Correction: the original OpenJev input was a poor fit

The original **12.8% OpenJev score is a long-label baseline, not a useful
verdict on the model**. The benchmark's category definitions and exclusions
exhausted its normal 512-token window, and the benchmark expanded the window
to 1,024 instead of adapting the candidate descriptions. The
[upstream inference notes](https://github.com/Heman10x-NGU/Verdict-open-jev#what-changed-in-the-inference-engine)
specifically discuss shorter contexts and report training on very short
states. That limitation should have been investigated before the first
comparison was presented.

The follow-up kept the same checkpoint, emails, frozen labels, category IDs,
body cleanup and character-weighted aggregation. It replaced the verbose
candidate text with short positive descriptions, for example `receipts` →
`a purchase receipt or order confirmation`. This is a **benchmark-specific
description override**: it does not preserve every exclusion or owner
constraint, and it does not edit `email.toml` or production classification.
Owner-dependent categories were already excluded from the labeled score.

| OpenJev input                           | Precision |  Correct / 156 | Abstained / 200 | Chunks / logical calls | Batches |   Call wall |       Total |
| --------------------------------------- | --------- | -------------: | --------------: | ---------------------: | ------: | ----------: | ----------: |
| Original definitions, JSON, 1,024       | F32       |     20 (12.8%) |             133 |                    520 |     130 |     21.258s |     25.906s |
| Compact descriptions, JSON, 1,024       | F32       |     70 (44.9%) |              62 |                    335 |      84 |     13.537s |     15.965s |
| Compact descriptions, JSON, 1,024       | BF16      |     71 (45.5%) |              61 |                    335 |      21 |      1.651s |      4.137s |
| **Compact descriptions, JSON, 512**     | **F32**   | **68 (43.6%)** |          **51** |                **782** | **196** | **12.106s** | **15.866s** |
| **Compact descriptions, JSON, 512**     | **BF16**  | **67 (42.9%)** |          **53** |                **782** |  **49** |  **2.491s** |  **6.284s** |
| Compact descriptions, plain fields, 512 | BF16      |     72 (46.2%) |              28 |                    738 |      47 |      2.366s |      5.926s |

The same-precision, same-budget F32 comparison improves from 20 to 70 correct.
Thus the strongest measured issue is candidate formulation, not a faulty
forward pass or BF16. Reducing the budget from 1,024 to 512 did **not** improve
this sample after shortening the descriptions. The tests do not isolate
shorter wording, removed exclusions, changed label positions and the resulting
chunk boundaries from one another.

The first 50 sampled emails (44 labeled) were used to select a configuration.
Compact JSON at both budgets scored 21/44; the 512-token configuration was
selected before inspecting the remaining 150 emails because it stays within
the upstream default budget. It then scored **46/112** on those remaining
labeled emails, versus **14/112** for the original setup. F32 subsequently
scored 68/156 overall. This is exploratory confirmation on an already known
corpus, not a new independent test set.

The plain-field variant scored 20/44 on development and was not selected,
despite its higher whole-corpus result. First-chunk-only classification was
also inspected as a diagnostic: it scored 66/156 with compact JSON/512 and
78/156 with plain fields/512. It is not retained as a policy because it drops
later evidence. No labels were changed and no extra hosted Jev calls were
made during the audit.

### Independent numerical check

Actual email inputs were checked against GLiClass 0.1.20 and PyTorch
2.10.0+cu128, using the pinned tokenizer, checkpoint and calibration. Token IDs
matched exactly; inference used CUDA F32 with TF32 disabled.

- Original F32: 16/16 chunk decisions matched; maximum probability error
  **0.000000671**.
- Compact/512 F32: 16/16 chunk decisions matched; maximum probability error
  **0.000000760**.
- Compact/512 Rust BF16 versus Python F32: 15/16 decisions matched; maximum
  probability error **0.00693**. Reduced precision can flip close decisions;
  it does not explain the original collapse.

These are 16 chunks per configuration, not a parity test of every email.
Prompt structure was also checked against the upstream
[formatting contract](https://github.com/Heman10x-NGU/Verdict-open-jev/blob/main/core/formatting.py).
Captured prompts, token IDs, native predictions and the offline Python oracle
are preserved with the private audit artifacts.

The benchmark now defaults OpenJev to `compact512`. Explicit modes
`original`, `compact1024` and `plain512` retain reproducibility of the audit.
Laya and Jev keep their existing default benchmark paths. OpenJev remains
below Laya's 87/156 and hosted Jev's 142/156 on this reference set, but the
earlier 20/156 substantially understated its performance with suitable inputs.

## Original results (superseded OpenJev setup)

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

The original OpenJev configuration was **not an accuracy improvement**. Its
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
- **Original OpenJev setup:** `heman10x/rlcd-modernbert-151m`, checkpoint revision
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
# OpenJev now defaults to compact512; append "original" for the old setup.
LD_LIBRARY_PATH=/run/opengl-driver/lib RAYON_NUM_THREADS=4 \
  target/release/examples/backend_benchmark \
  ~/.local/state/vs1-email/three-backends-200-20260921 laya laya-repeat
```

Hosted runs require the existing `TYPESAFE_API_KEY`. OpenJev loads the pinned
local checkpoint under `artifacts/openjev/checkpoint`. Output files use
create-new semantics. Tests were written failing first for abstention-aware
scoring, native probability pooling and explicit precision selection.
