# Browser decision model: training and evaluation specification

Status: proposed implementation, grounded in downloaded samples. The sampling
and inspection tools exist; the converters, trainer, teacher-labeling service and
cloud runner specified below do not yet exist. No training or cloud instance was
started for this document.

## 1. Outcome and scope

Build a fast, text-only browser decision model compatible with `vs1` and the Rust
browser-agent CLI. Input is a goal, current observed page, prior executed actions,
and runtime-defined options. Output is a distribution over operations and over
compatible observed targets. The first intended workloads are forms, search,
filters, navigation and reliable completion detection.

Evaluate two initializations with the same architecture and data:

- **Laya continuation:** retain the existing English Laya encoder and decision
  heads, then train for browser decisions.
- **ModernBERT initialization:** load pretrained `answerdotai/ModernBERT-large`
  encoder weights and initialize Laya-compatible typed decision heads. This is
  downstream training from pretrained weights, not pretraining from scratch.

Combine human browser demonstrations, general typed decisions, selected web
transitions and optional Jev soft labels. Dataset membership alone is never
sufficient evidence that an action is correct.

The deployment remains Rust-only. Python utilities and a future PyTorch trainer
are offline research tools; the CLI must not start a Python process or require
an inference server for local decisions. Field text stays with the existing text
helper. Generating pages, screenshots, selectors or arbitrary programs is outside
this model's output contract.

The $25 Vast.ai balance funds a bounded pilot, not the entire WebWorldData corpus
or an unrestricted experiment search. Jev usage is a separate budget.

## 2. Evidence available now

See [the sample audit](../research/browser-training/sample-audit.md),
[download manifest](../research/browser-training/download-manifest.json), and
[machine-readable inspection](../research/browser-training/inspection.json).
The raw downloads are local, ignored artifacts under
`artifacts/browser-training/samples/`. All sampled data is from training sources;
no official test split was downloaded.

| Source                     | Actually inspected                                                                 | Findings that constrain the design                                                                                                                               |
| -------------------------- | ---------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Mind2Web                   | 6 tasks, 49 actions, 3 websites                                                    | 6 steps have no positive candidate; 2 HOVER actions are collapsed to CLICK; median 213 negative candidates per step; select annotations need label/value mapping |
| LocalLLaMA/typed-decisions | 12 cases, 60 questions, all 4 workflows; complete 1,200-case train file downloaded | JSON embedded in strings; soft labels; latent factors must not enter model inputs                                                                                |
| Qwen/WebWorldData          | 12 records, 42 transitions, sampled at 3 file offsets                              | Actual schema is `conversations/from/value`; no explicit task goal or success metadata in these records; unsupported actions and an absent target occur          |

These are schema and feasibility samples, not representative accuracy estimates.
Do not extrapolate their action proportions or missing-label rates to an entire
dataset. All statistics below are measured with the English Laya tokenizer,
without automatic truncation, and exclude additional instructions/options unless
explicitly stated otherwise.

### 2.1 Pinned sources and availability

| Source                                                                        | Revision inspected                         | Declared license | Intended role                                                          |
| ----------------------------------------------------------------------------- | ------------------------------------------ | ---------------- | ---------------------------------------------------------------------- |
| [Mind2Web](https://huggingface.co/datasets/osunlp/Mind2Web)                   | `17ece8eb89862368edc0cc806acee6fca5163474` | CC BY 4.0        | Human action/target supervision                                        |
| [typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions) | `ea9306458d6e9563628369a3d1e72e362fb381d2` | Apache 2.0       | General typed-decision auxiliary task                                  |
| [WebWorldData](https://huggingface.co/datasets/Qwen/WebWorldData)             | `e108c5f8e35445c9ddff71cde2a5b1fc4db4020c` | Apache 2.0       | Transition learning and, where independently eligible, policy training |

“Laya data” in this pilot means the named typed-decisions dataset used by its
[released fine-tuning notebook](https://github.com/NandhaKishorM/laya/blob/6a5819129eb220570792e417e49723d697efd76f/notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb).
It does not mean Laya's complete original training mixture: that mixture has not
been located and verified for this project. The typed dataset is synthetic with
soft teacher labels, not a human-verified corpus or Jev's training data.

WebWorldData's repository contains one 52,243,347,728-byte JSONL file. Its card and
viewer give different aggregate counts. Count eligible records and transitions
from the pinned data during conversion; neither headline number is a training
budget. In particular, a row, a trajectory, a transition and a question are
separate units.

Retain source attribution, revisions, hashes and notices with derived datasets.
Raw webpages remain local artifacts; released examples should be synthetic or
reviewed. Record Jev output-use terms applicable to the account before undertaking
bulk distillation; these terms have not been verified by this audit.

## 3. Model and inference contract

### 3.1 Architecture

Use ModernBERT-large plus the existing Laya architecture:

1. Bidirectional text encoder.
2. Question-type embedding added to token states.
3. Two transformer decision-head layers.
4. Shared scorer applied at each option marker.
5. Existing auxiliary act/escalate head retained for file compatibility.

ModernBERT-large has 395M encoder parameters and supports a native context up to
8,192 tokens. The existing Laya checkpoint totals about 421M parameters with its
heads. Native context support does not imply good browser behavior at that
length; training and validation must match deployment lengths.
[Model reference](https://huggingface.co/answerdotai/ModernBERT-large)

Keep the architecture and weight names supported by `src/model.rs`,
`src/modernbert.rs` and `src/head.rs`. A smaller ModernBERT-base control is a later
experiment, not a hidden substitution for the proposed large model.

Pin the proposed ModernBERT initializer to
`45bb4654a4d5aaff24dd11d4781fa46d39bf8c13`. Pin the measured Laya baseline to
`c5d78730f3493e4fe16d61507ef4b78eef7318cf`, the local checkpoint used in the
canonical audit. The Hub's current Laya revision has since changed to
`1c5edc17a7acd8701df6fc341c0d179f1c62c982`; do not silently substitute it into
the baseline. A newer-checkpoint comparison is a separately named experiment.

Do not use the act/escalate head for fallback until it has separate validated
supervision. Its current output is not the browser operation head, and it is not
a reliable substitute for calibrated completion or error probabilities. Freeze
it in Laya continuation; initialize it reproducibly for the fresh-head model and
mark it unsupported for deployment decisions.

### 3.2 Browser output

Preserve current operation names:

`CLICK`, `TYPE_TEXT`, `SELECT`, `SCROLL_UP`, `SCROLL_DOWN`, `WAIT`, `DONE`, `BLOCKED`.

Offer only currently supported operations. `click_target`, `type_text_target`
and `select_target` contain only compatible observed controls/options. A select
target identifies both the control and an observed option. Never generate an ID
that was not offered. Keep singleton target resolution deterministic outside the
model as the CLI currently does.

One batched decision call may evaluate several questions, but the current
sequence layout encodes the state separately for each question. Batching is not
shared encoder-prefix computation. Count every question when estimating tokens,
training steps and latency.

Train target heads conditionally on their operation. A human click demonstration
supervises the operation and click target, not the speculative fill/select
heads. A teacher may label extra conditional heads, but their losses must remain
separate and must not make them executable without the corresponding operation.

### 3.3 Input layout and budgets

Use the existing sequence format and special token IDs:

`[CLS] type question: instructions [SEP] [MASK] option ... [SEP] state [SEP]`

The trainer must reproduce Rust's spacing, JSON rendering, Unicode handling,
marker placement, option cap, header squeeze and truncation exactly. Structured
instructions use ASCII JSON escaping; structured state and criteria retain
Unicode. Raw strings pass through. Include the recent Unicode regression case
in parity fixtures.

Proposed initial configuration, subject to the retrieval and memory gates:

| Setting                     | Pilot default                                                         |
| --------------------------- | --------------------------------------------------------------------- |
| Maximum sequence            | 2,048 tokens                                                          |
| Maximum header              | 640 tokens                                                            |
| Candidate shortlist         | At most 16, operation-specific                                        |
| Candidate descriptions      | Aim for at most 24 tokenizer tokens each                              |
| History                     | Up to 4 confirmed prior actions, additionally token-budgeted          |
| State                       | Goal, controls/current values, relevant page context, compact history |
| Temperature during training | 1.0; no inherited calibration scaling                                 |

The existing hard cap is 48 tokens per option. A 640-token header cannot hold
16 maximally long options plus useful instructions, so count rendered tokens
before training. A candidate whose distinguishing text is silently removed is a
conversion failure. Do not equate a larger sequence budget with a larger header.

A 1,024/384 budget is a cost ablation, not an automatic fallback with the same
quality claim. If 16 candidates miss too many targets, test a larger shortlist
and matching header budget, or improve retrieval, before fine-tuning.

## 4. Dataset conversion

### 4.1 Shared intermediate representation

Produce JSONL or Arrow records with a versioned schema. Store model inputs and
training-only evidence separately; render only an explicit allowlist of input
fields. An illustrative record (synthetic, not a downloaded example) is:

```json
{
  "schema_version": "browser-decision-v1",
  "id": "source:trajectory:step",
  "provenance": {
    "dataset": "source-repository",
    "revision": "immutable-revision",
    "source_record": "record-id-or-byte-offset",
    "trajectory_id": "trajectory-id",
    "step_index": 2,
    "site_group": "example.test",
    "split": "train",
    "converter_version": "commit-id"
  },
  "input": {
    "goal": "Apply the Design filter.",
    "page": { "title": "Stays", "url": "https://example.test/stays" },
    "elements": [
      {
        "id": "e0",
        "role": "combobox",
        "label": "Stay category",
        "value": "All stays",
        "visibility": "observed",
        "operations": ["SELECT"],
        "options": [{ "id": "e0:o0", "label": "Design", "value": "design" }]
      }
    ],
    "history": [],
    "page_context": ""
  },
  "supervision": {
    "operation": { "valid_labels": ["SELECT"], "kind": "human" },
    "selected_target_head": "select_target",
    "valid_target_ids": ["e0:o0"],
    "terminal": { "status": "unlabeled", "verifier": null },
    "teacher": null
  },
  "audit": {
    "source_target_ids": ["source-node-id"],
    "candidate_recall": true,
    "mapping": "unique_observed_option",
    "rejections": []
  }
}
```

The intermediate record is not itself a `SystemOneRequest`. A deterministic
renderer turns its `input` into `state` and `questions`, with an adjacent label
file. Both request and label hashes are stored. The current CLI must eventually
call the same rendering implementation or pass byte-identical golden fixtures.

Allow unknown values: absent `checked`, visibility or selection state is not
`false`. Preserve source node IDs only in provenance/mapping, then assign local
candidate IDs independently of labels. Never expose positive-candidate flags,
annotation IDs, gold, factors, teacher answers, source future actions or outcomes
as model inputs.

### 4.2 Mind2Web adapter

Read only `data/train/train_*.json` for development/training. Preserve official
test splits untouched for final evaluation. Required mappings:

| Source field                  | Destination                                       |
| ----------------------------- | ------------------------------------------------- |
| `confirmed_task`              | goal                                              |
| `actions[i].raw_html`         | source DOM for current state/values/options       |
| `cleaned_html`                | alternate context and audit, not sufficient alone |
| `operation.op`, `original_op` | operation eligibility and label                   |
| `pos_candidates`              | accepted target set, training-only                |
| `neg_candidates`              | candidate inventory, without negative-label flags |
| `backend_node_id`             | source-to-normalized target mapping               |
| `action_reprs[:i]`            | optional past-only history after normalization    |

Implementation requirements:

- Parse HTML inertly. Do not execute scripts or navigate captured webpages while
  extracting training examples.
- Map `TYPE` to `TYPE_TEXT`. For normalized `CLICK`, inspect `original_op` first.
  Reject `HOVER` and `ENTER` remappings until the runtime supports their actual
  semantics; do not teach a click as an equivalent operation.
- Treat all supplied positive candidates as potentially acceptable. Use a
  multi-positive loss rather than arbitrarily choosing the first. Human
  demonstrations remain one valid path, not proof all other paths are wrong.
- For absent positives, recover only if raw source evidence establishes a unique
  target identity. No fuzzy guessing from the action description. Otherwise
  reject the target example with `missing_target_annotation`; keep operation-only
  supervision only when its observation and operation are independently valid.
- Build candidates before looking at their labels. Do not inject a missing
  positive after retrieval. Report this as a retrieval miss.
- Normalize snapshot attributes such as `input_value`, `option_selected`,
  `aria_label`, `is_clickable` and geometry into semantic fields. Remove annotation
  instrumentation such as `data_pw_testid_buckeye*` from model-visible text.
- SVG/icon targets need a documented mapping to a compatible observable control.
  If mapping changes the action's semantics or cannot be verified, reject it.
- Static snapshots cannot establish all live visibility/occlusion properties.
  Record unknown visibility; compare extraction against live snapshots on owned
  pages before treating the representation as equivalent to our CDP reader.

For native selects, construct candidates as observed `(control, option)` pairs.
Resolve the label by exact option value, then exact normalized visible label.
A documented normalization may handle whitespace and time punctuation, but must
produce exactly one match. For example, the sample's `5 00 PM` identifies visible
`5:00 PM`, whose actual value is `17:00`; never fill `5 00 PM` as the value. Store
the resolution rule and actual option value. Reject ambiguous matches and
unsupported custom widgets. Learn to choose an option, not to invent one.

There is no automatic terminal label after the final recorded action. A final
pre-action snapshot does not demonstrate a completed state. Do not manufacture
`DONE=true` from trajectory exhaustion.

### 4.3 Typed-decisions adapter

Use exactly one copy of the `all/train` dataset; its four component configs are
not additional independent data. Parse `state`, `questions` and `gold` from their
JSON strings. Preserve criterion descriptions and label ordering.

The sample has all three primitives. Normalize probability dictionaries to the
renderer label order; validate finite nonnegative entries and matching supports.
Allow rounding error up to 1e-5 in the sum, renormalizing and logging it. Larger
errors are quarantined. Keep ordinal levels ordered and `noul` false/true order
fixed. Use the full soft distributions rather than turning them into hard labels.

`factors` and `label_agreement` are training metadata, not observations. The
teacher-generated labels measure agreement with a teacher, not objective truth.
Use this dataset for auxiliary decision-format training and regression testing,
not as evidence of browser competence or universal calibration.

### 4.4 WebWorldData adapter

The sampled records use `conversations` containing `from=human|gpt` and `value`,
which differs from the `messages/role/content` example in the card. Support both
through explicit schema adapters. Reject unknown or inconsistent role sequences.

For the observed format:

1. Extract the initial state before the `First Action` delimiter.
2. Parse the action from the final action delimiter preceding `Next Page State`.
3. Pair it with the following assistant state.
4. For later turns, use the preceding assistant state as the pre-state.
5. Keep the original record and byte offset so every transition can be traced.

Parse action expressions using a restricted AST/parser; never use `eval` or
execute dataset calls. Validate an allowlist of function names, primitive
arguments, target IDs and arities. Syntax validation does not establish semantic
correctness. Missing IDs are rejections, not invitations to search another state
for a convenient match. IDs are scoped to a particular state/document.

Route transitions into one of three lanes:

| Lane                 | Eligibility                                                                                                     | Supervision                                                     |
| -------------------- | --------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------- |
| Goal-directed policy | Explicit recoverable user goal, valid current state/target, supported action, credible success/quality evidence | Operation/target, with source and confidence recorded           |
| Transition auxiliary | Valid before/action/after triple and a mechanically verifiable relation                                         | Typed question about an action effect or transition consistency |
| Rejected/unlabeled   | Missing required evidence, unsupported semantics, ambiguous mapping                                             | Audit only; optionally queue for teacher labeling               |

All 12 currently sampled records lack explicit goal/success metadata and are
therefore ineligible for direct policy imitation without additional labeling.
This is a sample finding, not a claim that the entire release lacks goals.

For auxiliary training, the question may deliberately include both pre- and
post-state: e.g. whether an observed control's selected value changed as specified.
Declare that task separately from action prediction. Supervise only changes with
reliable corresponding elements; changing DOM IDs or navigation can invalidate
correspondence. Do not label every changed page a successful action, or every
unchanged page a failure. Do not train a target task that merely copies a target
ID already written in the question.

At browser-policy inference, neither the current demonstrated action nor the
resulting next state may appear in the input. Reconstructing a goal from the
future is allowed only as explicitly labeled hindsight/synthetic data with a
separate ablation and verification; it is not human goal supervision.

Supported policy mapping is limited to compatible `click`, `fill`, and native
`select_option`, plus demonstrably equivalent document scroll/wait actions.
`goto`, tabs, coordinate actions, `focus`, hover and arbitrary keyboard operations
are excluded from the first policy dataset. Do not map `send_msg_to_user` to
`DONE`, `infeasible` to `BLOCKED`, or arbitrary `noop(duration)` to our `WAIT`
without validating the goal/status or execution semantics.

Deduplicate alternative encodings of the same trajectory, and filter or separately
report non-English examples for the English encoder. If origin/site information
cannot be recovered reliably, quarantine those examples from the generalization
comparison rather than assuming they belong to a safe independent split.

## 5. Candidate retrieval and representation

The sample's full HTML has a median 38,309 tokens; even cleaned HTML has a median
13,203. Directly truncating those pages to 512 or 2,048 tokens is not the design.

Use a deterministic, label-blind preprocessing pipeline:

1. Extract candidate-compatible interactive nodes and semantic ancestor context.
2. Compute names using label relations, ARIA names, descendant text, placeholder
   and title in a documented order. Preserve distinguishing current values.
3. Rank using goal/history lexical relevance initially. Tie-break using source
   order or stable identity, never training labels.
4. Retain top-K candidates per operation and a small amount of surrounding page
   context. For selects rank control/option pairs with the same inference rules.
5. Assign local IDs and pack within explicit header/state budgets.

Produce a separate retrieval report: positive recall@8/16/32/64, retained versus
original supported actions, select-option coverage, and site/action breakdowns.
If top-16 recall is below the pilot's 95% target on labeled eligible validation
steps, improve retrieval or increase budgets before claiming classifier accuracy.

Report conditional target accuracy only alongside end-to-end accuracy that counts
retrieval misses as failures. Do not train a classifier on artificially perfect
shortlists and call the resulting metric browser grounding accuracy.

During training randomize candidate order, preserving identity and label mapping.
At evaluation include a fixed permutation robustness test. Candidates must retain
distinguishing descriptions; truncate/reject explicitly instead of silently
letting sequence construction collapse several options to the same prefix.

The shared browser observation schema must preserve URL/title, values, selection,
checked/expanded state, option labels and recent confirmed actions. Our existing
compact renderer omits some fields such as URL/selected state: aligning it with
the new training renderer is planned work, and the baseline must preserve the
old renderer for an honest before/after comparison.

## 6. Completion, failures and Jev distillation

### 6.1 Verified completion set

Collect owned fixture episodes with machine-checkable backend/DOM state and a
known goal. Include paired observations where only one required condition differs:
wrong destination, missing category, unchecked cancellation, unsubmitted form,
unopened property, completed task, loading state and genuinely unavailable action.
Vary names, values and layouts across disjoint template families.

Keep `DONE` in the operation options of unfinished examples. Positive `DONE`
labels require a success verifier; negative labels require evidence the task is
unfinished. Ambiguous state is not a confident negative. `BLOCKED` means no
supported progress remains, not merely that the student is uncertain.

The current hotel fixture is a regression check, not a sufficient completion
benchmark. Keep held-out layouts/tasks so passing it cannot mean memorizing one
page. Do not make a stricter completion gate appear to improve the model: report
model-only decisions and gated runtime results separately.

### 6.2 Teacher protocol

Optional after the human-supervised baseline. Ask Jev to score the exact same
observable state and candidate support available to the student. Pin the returned
model identifier/version, request hash, prompt/schema versions, candidate order,
timestamps, token usage and probabilities. Cache by the complete request and
teacher version. Query no official test states for training.

A teacher given full HTML while the student sees a truncated summary may produce
labels the student cannot infer. Such privileged-information experiments require
a separate name and ablation; they are not the pilot default.

Store human valid-action labels and teacher distributions separately. When they
disagree, sample for review and classify as alternative valid action, teacher
error, ambiguous state, unsupported operation or extraction error. Do not replace
human labels wholesale or use teacher confidence as proof of correctness.

Budget first for difficult/diverse training states and completion cases, rather
than requesting every speculative target head on every repeated state. The
initial cap is a configurable number of unique calls and a separately approved
currency ceiling. Stop before issuing a call that exceeds the budget. This spec
does not authorize spending teacher credits or prescribe an unverified API rate.

### 6.3 Student rollout correction

After an offline improvement, collect student trajectories on resettable training
environments. Query Jev on the states reached by the student, especially
uncertainty, repeated actions and completion claims. Retain verified corrections
and outcomes. Freeze evaluation tasks; their failures do not become training
examples for the same reported evaluation round.

This phase addresses distribution shift from human trajectories. It is a later
stage with a separate cost estimate, not part of the first offline mixture run.

## 7. Training objective and schedule

### 7.1 Objectives

For a hard label y, use cross-entropy `-log p(y)`. For a valid target set Y, use
`-log sum(p(y) for y in Y)` so equivalent valid elements are not penalized.
For teacher/generic soft labels q, use soft cross-entropy
`-sum(q[k] * log p[k])`, equivalent to KL up to a fixed target entropy.

Average operation and consumed-target losses within each browser step; average
steps within sampled trajectories or cap long-trajectory contribution. General
questions and auxiliary transitions form separate named loss groups. Log each
loss and its effective example count. No loss is assigned to unsupervised heads.

For a pilot teacher mix, start with equal human and teacher loss weights where
both are usable, then select any change using validation. Keep conflicts
inspectable. Maintain temperature 1 during training; clear old temperature bucket
values, particularly Laya's sharply scaled large-option bucket. Refit calibration
on a separate held-out calibration partition after selecting weights.

Optional Brier/ordinal-CDF losses and RLCD come after supervised baselines. The
pilot should test data and representation quality, not simultaneously introduce
an unvalidated reward algorithm.

### 7.2 Proposed optimization defaults

These are starting hyperparameters, not measured optimal settings:

- BF16 autocast, FP32 master/optimizer state where supported, AdamW.
- Encoder learning rate 1e-5; new head 1e-4, existing head 3e-5.
- Weight decay 0.01, gradient clipping 1.0, warmup 5% of updates then decay.
- Microbatch 1–2 questions at 2,048 tokens initially; accumulate to an effective
  32 questions. Length bucketing/dynamic padding; gradient checkpointing if needed.
- At most 2 epochs for the first paid run, with a hard wall-time cap. Validate
  every 250 updates or one quarter epoch, whichever is more frequent.
- Save optimizer, scheduler, RNG, sampler position and model state every 30
  minutes, plus best-validation and last checkpoints.
- Fixed seed for the pilot; repeat promising results with additional seeds later.

Full fine-tuning is the baseline if memory/throughput permit. LoRA is a separate
fallback/configuration; do not silently switch it in only one comparison arm.
If LoRA is used, record the targeted encoder projections, train decision heads,
and merge adapters before export to the Rust-compatible checkpoint format.

### 7.3 Stages and mixture

1. **Local overfit/sanity:** 128–256 clearly labeled examples; verify losses fall,
   the learned model beats a majority baseline, labels are not shifted, and
   shuffled labels do not produce a convincing held-out score.
2. **Initialization baseline:** frozen Laya evaluation; then Laya continuation on
   Mind2Web plus verified completion data and a small typed-decision mixture.
3. **ModernBERT arm:** same architecture, renderer, data, optimizer-update budget
   and evaluation; pretrained encoder with fresh decision heads.
4. **WebWorld ablation:** continue the better initialization with eligible policy
   records and/or transition auxiliaries. Compare with a matched extra-update
   control without WebWorld. Otherwise more compute is confounded with more data.
5. **Jev ablation:** add soft teacher labels to the same selected training states;
   compare against the same data/update budget without teacher loss.
6. **Live correction:** only after offline/closed-loop gates, as described above.

A proposed initial source sampler for the WebWorld stage is 60% browser policy
steps (Mind2Web plus owned completion episodes), 20% typed cases, 20% WebWorld
auxiliary/eligible policy cases. This is a tunable proposal, not a proven ratio.
Within the browser component, ensure completion episodes are represented. Report
both sampling proportions and effective question/token proportions: a typed case
contains five questions, unlike a typical two-question browser step. Cap repeats
of the small typed dataset; avoid swamping human browser demonstrations with
millions of easier auxiliary records.

The $25 pilot may only complete stages 1–3, or one paired comparison. Rank the
remaining experiments using observed throughput and quality; do not promise the
entire matrix within that budget.

## 8. Splits, contamination and evaluation

### 8.1 Partition before augmentation

Freeze group assignments before teacher labeling, candidate permutations,
negative generation or format conversions. Keep all steps/variants from a
trajectory in one partition. Group Mind2Web by website for development holdouts;
where a site has several tasks, do not randomly scatter near-identical page
states across partitions.

Proposed development grouping: 70% training, 15% model selection, 15% calibration
by source website groups where available. Typed cases split within workflow;
owned tasks split by template/layout family. Record exact group manifests and
seed rather than depending on a language's randomized hash.

Hash exact normalized goals/states and use near-duplicate page/trajectory checks
across sources. Remove overlaps with official held-out websites/tasks before
reporting cross-website generalization. WebWorld sources may overlap browser
benchmarks; uncertain provenance prevents a clean cross-domain claim. Preserve
a stricter evaluation with WebWorld excluded if decontamination is incomplete.

Official Mind2Web tests are evaluated once after model/renderer selection, using
its access/redistribution conventions. Its recorded-step metric remains distinct
from live browser execution.

### 8.2 Offline metrics

Report dataset, website, operation, option-count and sequence-length breakdowns:

- Eligibility/retention rates with every rejection reason and denominator.
- Candidate recall@K; target mapping coverage; context truncation frequency.
- Operation accuracy, target accuracy conditional on eligible operation, and
  joint operation+target accuracy over all supported labeled steps, with retrieval
  misses included as errors.
- Macro per-trajectory and per-website accuracy, plus micro step accuracy.
- Hard-label NLL/Brier and soft-target cross-entropy/KL, with the label provenance
  identified. Teacher agreement is not task correctness.
- Score MAE/ordinal error and noul metrics on general tasks.
- Calibration/reliability plots; do not use low ECE alone to declare a good model.
- Candidate permutation sensitivity and duplicate-description failure rate.

Report uncertainty by bootstrap over independent trajectory/site groups, not
individual correlated steps. Use paired comparisons on identical evaluation
records for data/initialization ablations.

### 8.3 Closed-loop metrics

Create at least 30 held-out owned tasks spanning 5 or more task/layout families
for the pilot. Include the known Continue and hotel regressions, but report those
separately from unseen tasks. For deterministic fixtures, repeats check stability;
they do not increase the number of independent tasks.

Measure independently verified task success, premature-DONE rate, step-budget
failures, unsupported/stale action attempts, repeat loops, decisions/actions per
success, and p50/p95 end-to-end latency. Keep the text helper and browser runtime
fixed across model comparisons. Separate model load, warmup, navigation, decision
compute, helper latency and final verification.

Compare original Laya, the trained models and hosted Jev on the same tasks when
teacher budget allows. Fixed-input replay results supplement, but never replace,
live results. Report local-only and Jev-fallback task success/cost separately.

### 8.4 Pilot gates (proposed acceptance criteria)

| Gate                | Requirement                                                                                                                                                    |
| ------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Data correctness    | All accepted records satisfy schema/label invariants; every rejected record has a reason; manually audit 20 examples per observed operation plus failures      |
| Candidate quality   | At least 95% positive recall@K on eligible validation steps, or explicitly re-scope/rework retrieval before training claims                                    |
| Retention           | Report total and per-operation coverage; investigate if fewer than 70% of source-supported labeled steps survive, rather than hiding exclusions                |
| Trainability        | Small-set supervised loss improves and valid labels are learned; no gain from leaked labels/future state                                                       |
| Offline improvement | Proposed target: at least +10 percentage points joint action accuracy over frozen Laya on the identical accepted development distribution; include uncertainty |
| Completion          | Zero premature DONE on the paired deterministic regression fixtures; broader held-out rate and confidence interval reported separately                         |
| Live utility        | Improvement on unseen closed-loop tasks; no claim of general browser competence from the fixture suite alone                                                   |
| Export              | Exact token/marker parity and CPU F32 probabilities within 1e-4 on golden fixtures; GPU drift characterized and no unexplained answer flips                    |
| Budget              | Stop/checkpoint before configured spend/time ceiling; do not consume all remaining balance waiting for a run to finish                                         |

Failure is a useful pilot outcome. If retrieval, representation or completion
labels fail, repair those before buying more epochs.

## 9. Export and Rust integration

Export a standard directory containing:

```text
model.safetensors
rl_agent_config.json
encoder/config.json
tokenizer/tokenizer.json
tokenizer/tokenizer_config.json
training-manifest.json
evaluation.json
```

Retain canonical weight names/shapes, including `encoder.*`, `type_emb.*`,
`head.layers.*`, `scorer.*`, `act_head.*` and required buffers. Export full merged
weights, not an adapter-only archive. Supply `head_layers`, `act_costs`, sequence
budgets, primitive temperatures and model name; test both canonical Python and
Rust loading before promoting the checkpoint.

The initial deploy command remains:

```sh
vs1-browser --device cuda --checkpoint /path/to/checkpoint \
  --task hotel --output /path/to/fresh-results
```

The command alone is insufficient if a new renderer/retriever was trained. Ship
and select its version alongside the checkpoint, and fail explicitly on a
renderer mismatch. Do not overwrite the default Laya checkpoint or silently alter
all existing inference behavior.

Golden parity corpus: all primitives, Unicode structured instructions, native
select mappings, singleton handling, varying option counts and mixed sequence
lengths at both training and truncation boundaries. Compare CPU reference first,
then masked CUDA and packed Flash Attention. Regression checks must include the
known local-window lengths around 64/65/66 and mixed batches.

Recalibrate on the held-out calibration set after fine-tuning. With limited data,
prefer a coarse temperature to overfitting many buckets. Confidence is the
library's entropy-derived statistic, not the selected option probability; report
both and fit any fallback threshold against empirical error/coverage.

## 10. Compute, cost and stopping policy

Prepare/download/normalize data locally before renting. The currently available
local GPU is an RTX 3080 Ti with 12 GiB VRAM; use it for sanity checks and export
verification. Target one 24 GiB 3090/4090-class rental, selected on measured total
cost and throughput rather than model name alone.

Reserve $5 of the $25 credit for non-training charges/contingency and cap planned
compute at $20. Prices below are arithmetic scenarios, not live offers:

| Compute price | $20 compute ceiling |
| ------------- | ------------------- |
| $0.40/hour    | 50 hours            |
| $0.60/hour    | 33.3 hours          |
| $1.00/hour    | 20 hours            |

Actual storage, bandwidth and idle charges must be included using the selected
[Vast offer's pricing](https://github.com/vast-ai/docs/blob/main/guides/instances/pricing.mdx).
Teacher labels and helper API calls are outside Vast credits. Do not provision
instances as part of dataset inspection.

Before a paid full run:

1. Verify package/container versions, GPU architecture, attention implementation
   and a complete forward/backward/checkpoint/resume cycle.
2. Benchmark 200–500 optimizer updates with the actual length and source mixture.
3. Measure questions/sec, non-padding tokens/sec, peak allocated VRAM, validation
   overhead and checkpoint/upload time.
4. Estimate remaining time and cost, then reduce sample/update budget if needed.

Use:

`training_seconds = epochs * total_supervised_question_examples / measured_questions_per_second`

Add warmup, validation, checkpointing, startup/download and export time separately.
If using token throughput, use measured mixture-weighted non-padding tokens; a
longer option header and duplicated state per question count toward this total.
Do not project 512-token throughput to 2,048 tokens with a fixed multiplier.

Checkpoint early enough to finish uploads before the spend ceiling. Monitor from
outside the instance; the training process exiting does not necessarily stop
billing. Copy and verify checkpoints locally, then stop/destroy resources and
account for any retained storage. No background instance is part of this spec's
current deliverable.

Earlier estimates of 8–24 GPU-hours for two small 512-token runs do not apply to
this larger 2,048-token, multi-source matrix. A complete sweep/full-corpus training
time is intentionally unspecified until eligible data counts and throughput are
measured. The pilot buys evidence, not a promised universal replacement for Jev.

## 11. Implementation milestones and deliverables

| Milestone                  | Deliverable                                                        | Exit condition                                           |
| -------------------------- | ------------------------------------------------------------------ | -------------------------------------------------------- |
| M0 — schema audit          | Pinned samples, manifests, source audit, this spec                 | Completed in this change; limitations explicit           |
| M1 — normalization         | Adapters, schema validator, rejection ledger, split manifests      | Manual review and data gates pass                        |
| M2 — representation        | Shared retriever/renderer, budget reports, golden requests         | Candidate recall and no-label-leakage checks pass        |
| M3 — training harness      | Reproducible pretrained/fine-tuned initializations, losses, resume | Small-set learnability and parity checks pass locally    |
| M4 — paid pilot            | Throughput report, one bounded paired experiment                   | Within $25 total rental cap; measured validation outcome |
| M5 — deployment evaluation | Exported checkpoint, Rust replay/live reports                      | Loading/parity and independent outcome checks pass       |
| M6 — expansion             | WebWorld/Jev matched-compute ablations and live correction         | Funded separately if pilot justifies scaling             |

Suggested future CLI modules: `prepare`, `audit`, `render`, `label-teacher`,
`train`, `evaluate`, `export`, `verify-parity`. These are proposed training-tool
commands, not existing commands and not additions to the user-facing browser
runtime. Pin environments in a lockfile/container when implementing M3.

Every trained release must include source revisions, sample IDs/splits, data and
renderer hashes, eligibility counts, model initialization, optimizer settings,
seed/update count, teacher provenance, actual spend, test contamination notes,
reference/Rust parity and live task success. A checkpoint without this report is
an experiment artifact, not a new default model.

## 12. Open decisions

- Locate and verify the original Laya training mixture if full reproduction is
  desired; the typed-decisions subset is the only defined substitute here.
- Determine whether a larger WebWorld sample contains recoverable goals/outcomes,
  and whether those subsets can be selected from the released metadata.
- Measure candidate recall after a label-blind static-to-live representation
  conversion; this may be the dominant bottleneck.
- Verify teacher output-use conditions and separately set a Jev spend ceiling.
- Choose the final context/shortlist size from coverage and GPU measurements.
- Establish whether Laya initialization or fresh heads generalize better before
  committing to a full-corpus schedule.

None of these uncertainties justify treating missing data as known or increasing
spend automatically. They are explicit measurements for the next implementation
stage.
