# What we have learned about laya

This note consolidates the evidence available on 2026-09-21, before further
email experiments. Keep laya as the decision model and preserve the current
checkpoint. Model replacement, weight changes and additional learned components
are not the default next step. The characteristics to preserve are local Rust
inference, non-autoregressive typed answers, explicit candidate probabilities,
small bounded requests, and efficient CUDA/flash-attention batching.

The detailed experiment history remains in [email-experiments.md](email-experiments.md).
This document explains what those results mean in light of the actual training
data. Observations, hypotheses and untested ideas are distinguished below.

## What training data we actually accessed

We downloaded the complete `all/train` Parquet from
[LocalLLaMA/typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions/tree/ea9306458d6e9563628369a3d1e72e362fb381d2),
at revision `ea9306458d6e9563628369a3d1e72e362fb381d2`. The file is 598,824 bytes;
its SHA-256 is
`46a58d63edfd86e23229c78afe8b72307bb4ca9fb0e8df180cabb3c67ec9dcd5`.
The [download manifest](../research/browser-training/download-manifest.json)
records the provenance. Raw data is local under
`artifacts/browser-training/samples/typed-decisions-train.parquet`.

This is the dataset named by laya's
[published fine-tuning notebook](https://github.com/NandhaKishorM/laya/blob/6a5819129eb220570792e417e49723d697efd76f/notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb).
It is **not the complete original laya training mixture**, which we have not
located and verified. The notebook establishes a published recipe; it does not
by itself prove every setting used to produce the downloaded weights.

The 2026-09-19 [sample audit](../research/browser-training/sample-audit.md) inspected
12 cases and 60 questions. For this note, we inspected all 1,200 training rows.
The resulting [full-file shape measurements](../research/browser-training/typed-training-shape.json)
are separate from that earlier sample report. No test split, model training,
or new inference experiment was needed for this inspection.

## What the full training file shows

| Workflow                  | Cases | State context                                              | Choice decisions                       |
| ------------------------- | ----: | ---------------------------------------------------------- | -------------------------------------- |
| Agent trace observability |   300 | Agent, task, constraints, trace summary                    | Action and outcome, four options each  |
| Customer service          |   300 | Account and customer thread                                | Action and category, five options each |
| Invoice processing        |   300 | Invoice, purchase order, delivery, payment, vendor history | Disposition, four options              |
| Security incidents        |   300 | Alert, context, history, principal                         | Disposition, four options              |

Every case contains five questions. Across the file there are 1,800 `choice`,
1,800 `noul`, and 2,400 `score` questions. Of the choices, 1,200 have four options
and 600 have five. Each workflow uses **one fixed question-schema string across
all 300 cases**: the same question wording, criteria and ordering within that
workflow. The data varies the situations, not thousands of arbitrary rubrics.

There is no 17-folder personal-mail taxonomy in this file. Customer-service
categories are account, billing, delivery, refund and technical. Invoice
processing asks whether to approve, hold, review or reject a vendor invoice
using explicit order and delivery evidence. That is different from deciding
whether a free-form message belongs in receipts, bills, income or fiscal.

State lengths measured with the pinned typed-checkpoint tokenizer, before any
truncation and excluding the question/options:

| Workflow                  | Minimum | Median | Maximum tokens |
| ------------------------- | ------: | -----: | -------------: |
| Agent trace observability |      91 |    105 |            120 |
| Customer service          |      76 |  269.5 |            528 |
| Invoice processing        |     268 |  296.5 |            348 |
| Security incidents        |     181 |    227 |            300 |
| All cases                 |      76 |    238 |            528 |

These are serialized JSON states, using `json.dumps(..., ensure_ascii=False)`
without tokenizer padding or truncation. They are not final sequence lengths.
Compared with long email threads, the training states are compact and provide
specific decision context. The source card identifies English synthetic data;
we have not performed an independent language audit of every field.

**Interpretation:** arbitrary Portuguese/English mailbox categories, repeated
paraphrased binary questions, tournament finalists and chunk aggregation all
require generalization beyond this observed fine-tuning task. The data explains
why we should measure that generalization. It does not prove laya cannot do it,
or establish what the original pretraining mixture taught it.

## Labels and training semantics

The [dataset card](https://huggingface.co/datasets/LocalLLaMA/typed-decisions/blob/ea9306458d6e9563628369a3d1e72e362fb381d2/README.md)
reports synthetic states and soft labels averaged from three teacher samples.
Agreement with these labels measures teacher agreement, not independently
verified correctness. Preserve that distinction when discussing benchmark
scores; a teacher's mistakes can also be learned.

The actual rows encode `state`, `questions`, `gold`, `factors` and
`label_agreement` as JSON strings. `factors` contains latent scenario information;
it and the answer/agreement fields must stay out of inference inputs. The `all`
configuration already includes the four workflows: adding their individual
configurations would duplicate cases. The largest probability-sum rounding
error measured across the full file is about 1e-6.

The inspected notebook turns each case/question pair into a separate token
sequence, maps soft targets into option order, and trains against those
distributions. It does not teach our length-weighted email pooling or a joint
consistency rule between answers. Its calibration sample is drawn from training
items (`all_items[::15][:400]`), not an independent email calibration set.

There is also a provenance caveat: preprocessing calls `build_sequence` using
configuration loaded from the base checkpoint; the training worker later sets
1,024/256 limits after those items have been saved. Consequently, a saved runtime
limit alone does not establish the sequence lengths used by that notebook run.
We have not established that this affected the released checkpoint.

## How the inference model behaves in our application

Our default email checkpoint is `typed-decisions`, revision
`c5d78730f3493e4fe16d61507ef4b78eef7318cf`. Its native runtime budgets are 1,024
total tokens and 256 header tokens. The renderer caps each option's description
at 48 tokens and can squeeze it further when the header is crowded. Long prose
in TOML is therefore not equivalent to more usable instruction.

The token layout is a typed question, option markers/descriptions and the state.
The question participates in encoding: the current implementation batches
question-conditioned sequences, rather than encoding an email once and getting
arbitrarily many answers for free. This is why dozens of probes increase work.
Batching is valuable, but total question count and sequence length still matter.
See [sequence construction](../crates/vs1/src/sequence.rs) and
[batched inference](../crates/vs1/src/model.rs).

The input audit matched **all 720 token sequences and option-marker positions**
against upstream on the fixed 100-message development sample. That argues
against a serialization mismatch as the cause of those errors. It does not
prove calibration, task suitability, or numerical-forward equivalence anew.

Choice probabilities are relative to that question's candidates. Separately
asked `noul` questions do not provide a demonstrated common ranking scale across
our categories. In the current tournament, eliminated categories receive zero
and finalists are normalized; those zeros mean eliminated, not impossible.

The reported confidence is `1 - entropy / log(option_count)`. At email level it
is recomputed after pooling across chunks. It is a concentration statistic,
not an empirically validated probability of correct filing. Neither a softmax
nor a high confidence number establishes cross-question comparability or repairs
missing semantic evidence. Thresholds require separate validation. For example, four equally likely
finalists with the other 13 categories zeroed yield about 0.51 confidence when
normalized over all 17 slots, despite no preference among those finalists.

## What the email experiments actually taught us

| Observation                                                                                           | Supported conclusion                                                       | What it does not establish                                                                  |
| ----------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| Purpose-focused descriptions improved 35→36/61 and 25→30/45                                           | Short, explicit distinctions helped on two selected subsets                | Arbitrarily longer prompts help                                                             |
| Reversing/shuffling category order changed 27/100 and 29/100 predictions and reduced labeled accuracy | Group membership/option order matters in our tournament                    | A searched-for ordering would generalize                                                    |
| Keeping two group winners fell 35→29/61                                                               | More surviving candidates was worse in this implementation                 | Wider candidate coverage guarantees better decisions                                        |
| Binary matching improved shortlist coverage but fell 36→34/61 at 2.93× runtime                        | Finding candidates and choosing correctly are separate problems            | Binary probabilities can safely replace categorical comparisons                             |
| 53 supporting questions plus 12 exclusions scored 35/61 or 34/61 versus 36/61                         | More paraphrases did not improve this classifier; question work rose 13.2× | The model performs a reliable vote across paraphrases                                       |
| Financial-event probes left accuracy at 36/61                                                         | The tested extraction/routing rule added no gain                           | A useful-looking intermediate answer is accurate enough to override a decision              |
| Opening-only input and email-level final decisions each fell 36→35/61                                 | Both tested shortcuts lost accuracy                                        | Only the opening matters, or another model call necessarily resolves chunk disagreement     |
| Multilingual checkpoint fell 36→20/61                                                                 | That checkpoint switch failed on this task                                 | All Portuguese input requires a different checkpoint                                        |
| Retrieved labeled examples improved 23→25/32, then 27→34/40 on fresh selected cases                   | Concrete local examples helped the tested laya pipeline                    | In-context learning is universally reliable or the examples' category alone caused the gain |

The retrieval result fixed eight cases and regressed one on the fresh set, with
identical repeats and 1.64× classification time. Added examples increased chunks
from 65 to 90. It used a separate frozen MiniLM retriever: we have **not** shown
that lexical retrieval preserves the gain. It remains an experimental harness,
not the default classifier. Its mechanism could involve example text, labels,
or their interaction; no ablation has isolated those contributions.

Plain-text MIME preference, HTML conversion, tracking-link cleanup and explicit
subject/from/date fields reduce input noise while preserving meaningful evidence.
The body is split with the actual tokenizer budget; all accepted chunks fit.
Recipients are omitted. Character-count-weighted pooling remains a heuristic,
not a procedure supported by the inspected training data. Missing owner context
also remains missing evidence: the empty owner lists cannot identify a spouse,
a fixed bill or a capture venue by themselves.

Our reference sets select clear cases, omit ambiguous/owner-dependent ones,
and do not cover all 17 categories adequately. The 40 fresh retrieval labels
were reviewed by the assistant, not independently confirmed by the user.
Previously consulted validation sets are no longer untouched holdouts. These
counts are useful paired evidence, not whole-mailbox accuracy estimates.

The [matched hosted Jev comparison](email-experiments.md#12-matched-hosted-jev-comparison)
subsequently scored 59/61 and 60/61 against laya's repeated 36/61, using identical
message states and preliminary questions under the current tournament. This
shows that those inputs can support much higher accuracy with another backend;
it does not distinguish local checkpoint limitations from runtime differences.
The experiment adapted Jev's rounded probabilities and occasional non-argmax
choice field explicitly. It did not replace laya or change the active config.

## Boundaries for subsequent work

The [200-message compact-description experiment](email-experiments.md#15-compact-category-descriptions-on-laya)
held all chunks and tournament settings fixed. Existing wording scored 87/156;
the OpenJev compact wording scored 68/156, and restoring all exclusions reached
84/156. Both candidates repeated exactly and were reverted. Shorter labels
are not automatically better for laya; the wording gain observed with OpenJev
did not transfer. Most compact-only regressions were informational broadcasts.

Keep the current laya checkpoint, typed API and local execution. First use this
training-data evidence to frame questions and interpret errors. Do not assume
that a generative prompting technique, extra questions, confidence threshold,
or changed pooling rule will transfer to this model without measurement.

An outstanding diagnostic is direct replay of frozen dataset cases through our
runtime against the supplied soft targets, reporting per-workflow and
per-question behavior. Training-case replay would be a compatibility diagnostic,
not a generalization benchmark. None of the email accuracy results establishes
how well this Rust runtime matches the released model on its fine-tuning task.

Other untested questions include how much failure comes from absent evidence,
question semantics, elimination or chunk aggregation. They should be separated
before changing the pipeline. This note does not start those experiments.
Continue to freeze labels before predictions, use failing tests before code
changes, revert regressions, document outcomes and Jujutsu-commit demonstrated
improvements. No model replacement or weight change is implied by this plan.
