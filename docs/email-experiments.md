# Email classifier experiments

See [laya learnings](laya-learnings.md) for the training-data evidence, model
constraints and interpretation of these results.

Experiments use local Maildir copies and dry-run inference only, with the
`typed-decisions` checkpoint, CUDA BF16, flash attention, and batch size 16.
Descriptions, chunking and pooling stay fixed except for the factor under test.
The development sample has 100 messages and 61 existing labels; a separate
100-message validation sample has 45 labels frozen before viewing predictions.
Three exact-body duplicates were excluded from the initially labeled 48
validation entries before comparing any experiment there. These selected
clear-case subsets do not estimate whole-mailbox accuracy.

Unless a section records an incomplete run, each variant runs twice after model
warmup. Timings exclude model loading.
Only improvements on both labeled subsets qualify for adoption; changes that
fail the development comparison are reverted without tuning on validation.
Raw private messages and diagnostics stay outside the repository.

Baseline: 35/61 development, 25/45 validation. Development runtime is 9.71s
(mean of the preceding two identical-baseline runs); validation runtime 14.46s.

## 1. Keep two candidates from each preliminary group

For 17 categories, preserve two of each group's candidates, then run two
four-way contests and one final two-way contest: seven questions per chunk.
Stable ties and finalist order follow configuration order. A failing mock test
first established that a runner-up could not reach the final; the implementation
then passed that test and the existing small-choice tests.

Both passes regressed to 29/61; mean runtime 14.43s versus 9.71s.
Two errors were corrected and eight previously correct labels regressed.
Implementation and experimental tests were reverted. No improvement commit
was made. Validation was not used to tune this failed candidate.

## 2. Category-order sensitivity

Two permutations fixed before viewing predictions: reverse configuration order,
and a shuffle with seed 20260921. They change group membership/option order
without changing category definitions. This is a sensitivity check, not a search
for the best ordering.

Reverse order: 34/61, mean 10.75s, 27/100 predictions changed.
Seeded shuffle: 33/61, mean 10.72s, 29/100 predictions changed.
Both repetitions agreed. Neither ordering is retained. This exposes
group/option-order sensitivity but does not justify selecting an order.

## 3. Purpose distinctions in descriptions

One predeclared revision emphasizes paid versus unpaid, recruiting versus
client work, and developer broadcasts versus operational notifications. No
owner data or category names change.

Development: 36/61 on both passes, mean 10.86s; four fixes and three
regressions. Validation: 30/45 on both passes versus baseline 25/45, mean 14.60s
versus 14.46s. The revised descriptions are retained. This is a measured
gain on two small selected subsets, not evidence of reliable automatic filing.

## 4. Recognizable boilerplate removal

Target payment-provider support/anti-phishing footers and calendar dial-in
blocks. Preserve transaction fields, meeting purpose, and quoted history.

Both passes scored 35/61, unchanged from baseline. Chunks fell from 144 to
141 (705 questions versus 720), but mean runtime was 10.58s: no convincing
runtime improvement against the 9.71s earlier baseline or 10.7s contemporary
order controls. Implementation and tests for this experiment were reverted. Existing
tracking-link removal and mislabeled-HTML rendering remain intact.

## Final disposition

| Experiment                       | Development correct / 61 | Validation correct / 45 | Decision           |
| -------------------------------- | -----------------------: | ----------------------: | ------------------ |
| Baseline                         |                       35 |                      25 | Reference          |
| Top two per preliminary group    |                       29 |           Not evaluated | Reverted           |
| Reverse category order           |                       34 |           Not evaluated | Rejected           |
| Seeded shuffled category order   |                       33 |           Not evaluated | Rejected           |
| Purpose-specific descriptions    |                       36 |                      30 | Kept and committed |
| Recognizable boilerplate removal |                       35 |           Not evaluated | Reverted           |

Every completed condition repeated identical predictions. All runs classified
100 messages without processing failures. Rejected implementation changes and
the temporary runner were removed from the repository. Formatting, workspace
tests, and Clippy passed. The sole adopted change is the description revision;
tournament structure, category order, and chunk pooling remain unchanged.

Private reproducibility artifacts (frozen configurations, labels, per-chunk
results, timing records, runner source, and rejected patches) are stored in
`~/.local/state/vs1-email/four-experiments-20260921/`.

## 5. Binary category matching

Test all 17 categories independently using laya's `noul` primitive, with explicit
yes/no definitions and exclusions. Keep the five highest match probabilities,
then make one five-way final choice. Final choices follow configuration order;
chunking and length-weighted pooling are unchanged. This uses the improved
descriptions from experiment 3.

Two runs per method, reversing execution order on the second pass, used CUDA
BF16 with flash attention and batch size 16. Both methods processed the same
100 messages and 144 chunks.

| Method                            | Correct / 61 | Mean runtime | Questions |
| --------------------------------- | -----------: | -----------: | --------: |
| Current tournament                |           36 |        9.70s |       720 |
| Binary matching plus final choice |           34 |       28.38s |      2592 |

Runtime increased 2.93 times. Both runs produced identical
classifications for each method. Binary matching corrected six prior errors
but regressed eight previously correct labels. The correct category was absent
from every chunk's final shortlist for eight messages, down from 17, but better
shortlist coverage did not improve final accuracy.

Actual binary headers used 63–89 tokens including special tokens, below the
256-token header budget. All requests passed the body-capacity checks. Tests
for binary question construction, exclusions, ranking, ties and probability
bounds failed before implementation and passed afterward; package tests and
Clippy passed.

Decision: reject and remove the experimental implementation. No production
classifier or configuration changes are retained. Validation was not consulted
because the development comparison already failed. Private results, source and
header evidence are preserved in
`~/.local/state/vs1-email/binary-category-100-20260921/`.

## 6–7. Opening-only input and one final decision per email

Two separate experiments with the retained descriptions and the same 100-message
development sample (61 labeled cases), CUDA BF16, flash attention, batch 16.
Two passes repeat each condition after backend warmup; baseline and opening
order are reversed on the second pass. All runs complete without errors.

- Opening-only: keep headers and owner context, but use only the first nonempty
  body paragraph, capped at 600 Unicode characters. The existing tournament and
  chunk pooling otherwise stay the same.
- Email-level final: reuse unchanged baseline chunk decisions. Each chunk
  nominates its two highest-probability categories; retain up to five categories
  by their peak chunk probability, with configuration-order ties. For each
  candidate use the first 240 characters of its strongest supporting chunk,
  deduplicating shared chunks. Shorten excerpts if the exact token check requires
  it. One final choice sees the sender, subject, date, owner context and excerpts.
  Its answer replaces the weighted-average prediction. This is one specific
  nomination/excerpt policy, not an exhaustive test of email-level aggregation.

| Method                     | Correct / 61 | Mean runtime | Questions |
| -------------------------- | -----------: | -----------: | --------: |
| Current classifier         |           36 |        9.93s |       720 |
| Opening paragraph only     |           35 |        3.21s |       505 |
| One final choice per email |           35 |       10.61s |       820 |

All 100 predictions per method repeat identically. Opening-only fixes eight
prior errors but regresses nine; it is approximately
3.10 times faster but does not improve accuracy.
It produces 101 chunks rather than 144 because one opening still requires
splitting under the token budget. The final-choice experiment fixes four and
regresses five. Its reported time includes measured baseline processing plus
the extra final decision; baseline outputs are reused rather than inferred
again for the experiment.

Both prototypes were removed. No production behavior or configuration changes
are retained. Validation was not consulted after both failed the development
accuracy criterion. The opening-only speed tradeoff is recorded for a future
explicit fast-mode decision, not silently adopted.

Tests for Unicode-safe opening limits and bounded multi-chunk nominations
failed before implementation and passed afterward. Package tests, Clippy and
formatting passed. Frozen configuration, source, results and timing evidence:
`~/.local/state/vs1-email/email-context-100-20260921/`.

## 8. Financial-event extraction before category routing

Add one first-round choice question per chunk asking what financial event is
explicitly reported: completed purchase payment, unpaid obligation, incoming
money, statement/tax matter, or none/unclear. Route the first three to receipts,
bills, or income only when the selected score is at least 0.45 and exceeds the
runner-up by at least 0.10. Strong statement/tax evidence or conflicting
transaction types forces fallback to the current classifier. These thresholds
were fixed before inference; scores are not calibrated probabilities.

Run on all 100 development messages to expose nonfinancial false positives,
with the retained descriptions, CUDA BF16, flash attention and batch size 16.
Two repetitions reverse method order.

| Method                              | Correct / 61 | Mean runtime | Questions |
| ----------------------------------- | -----------: | -----------: | --------: |
| Current classifier                  |           36 |       10.83s |       720 |
| Financial-event probe plus fallback |           36 |       11.63s |       864 |

Both methods produced identical categories for all 100 messages, on both
passes. Only one message obtained an eligible financial route, and its existing
category was already correct. No errors were fixed. Baseline question answers
also remained unchanged when the probe was added to the batches. The extra
probe adds 20% more questions without an accuracy improvement. Runtime is
noisy: baseline passes were 9.73s and 11.93s; event passes 11.53s and 11.73s.

Raw answers indicate why lowering thresholds is not justified by this test:
both labeled incoming-money messages were called outgoing paid purchases, and
one tax/accounting message was also called a paid purchase. Their low scores
prevented overrides. Many receipt passages did select paid, but weak separation
and footer disagreement prevented a reliable route. This rejects the specific
event question and routing policy, not every possible evidence-extraction model.

The model-built question header used 189 tokens including special tokens,
below its 256-token budget. Every augmented request passed the exact state
capacity check. Routing tests failed before implementation and then passed;
package tests, Clippy and formatting passed. All 100 messages completed in
each run without processing failures.

Decision: remove the prototype and retain no production changes. Do not tune
thresholds against this outcome or spend validation labels on a candidate
without a development gain. Findings and private reproducibility artifacts:
`~/.local/state/vs1-email/money-events-100-20260921/`.

## 9. Multiple yes/no evidence questions per category

Added 3–4 category-specific `questions` to each rule in an isolated experimental
`email.toml`: 53 supporting questions in total. Optional schema support retained
compatibility with old configs and validated a 3–5 range, nonblank wording and
unique questions. Twelve existing exclusions were asked separately.

Every chunk received 65 binary (`noul`) questions. Two predeclared ways of
combining support were compared: strongest answer, and average of the strongest
two. These are alternative signals rather than a requirement that all answers
be yes. An exclusion score of at least 0.75 vetoed its category; this heuristic
was fixed before inference and does not imply calibrated probabilities.
Each method shortlisted five categories for the existing final choice, then
used unchanged length-weighted pooling across chunks. Both reused the same
binary answers so the pooling comparison did not repeat extraction.

The experiment ran against pinned, uncompressed source in an isolated Jujutsu
workspace because separate model-compression work was active in the main
working copy. It used the retained descriptions, CUDA BF16, flash attention,
batch size 16, and the same 100-message development sample with 61 labels.

| Method                                | Correct / 61, both passes | Mean runtime | Questions |
| ------------------------------------- | ------------------------: | -----------: | --------: |
| Current classifier                    |                        36 |       14.63s |       720 |
| Strongest answer, then final choice   |                        35 |      129.16s |      9504 |
| Strongest-two mean, then final choice |                        34 |      128.56s |      9504 |

All 100 predictions per method repeated identically. Strongest-answer pooling
fixed four cases but regressed five. Strongest-two pooling fixed five but
regressed seven. The correct category was missing from all chunk finalists
in 11 and nine labeled cases respectively, versus 17 for the baseline: better
shortlist coverage still did not improve final accuracy.

A post-hoc diagnostic that used the length-weighted supporting scores directly,
without the final choice, scored 28/61 and 25/61 on the first pass. It was not
adopted or tuned. Exclusions triggered 15 vetoes among 1,728 checks in that
pass. Raw supporting answers are retained for further analysis.

Timings varied considerably across passes (strongest-answer variant 142.98s
and 115.34s); treat them as local observations rather than a precise throughput
claim. The exact question-count increase is 13.2 times. The two experimental
variants reused baseline chunk boundaries; their times include probing and
final classification, not independent chunk-boundary preparation.

Schema and pooling tests failed before implementation, then passed. Package
tests, formatting and Clippy passed. Every probe and final request passed the
state-budget check; all messages completed without failures. No validation run
was needed after both development results regressed.

Decision: revert the schema, questions in the active config, and experimental
runner. The production classifier and active `email.toml` remain unchanged.
The proposed question set is preserved as `questions.toml`, with the full
prototype patch, source, raw answers, results and timings under
`~/.local/state/vs1-email/multi-questions-20260921/`. Only findings are committed;
unrelated model-compression changes are untouched.

## 10. Checkpoint, retrieval, embedding and larger-model experiments

Four experiments tested whether the remaining errors came from input
compatibility, missing examples, an unsuitable decision head, or model
capacity. The committed baseline and current `email.toml` were pinned in a
separate Jujutsu workspace, excluding concurrent model-compression changes.
The mailbox remained read-only and all email inference stayed local.

The baseline repeated at 36/61 on development and 30/45 on the existing
validation labels. Those validation labels have already influenced earlier
experiments and are not a new holdout. For the two supervised methods, only
the 61 development labels were training data. Removing validation senders
seen in training and normalized body-template overlaps left 32 cases, on
which the baseline scored 23/32. Template overlap uses five-word shingles,
case folding, digit normalization, and Jaccard similarity >= 0.5. This is a
conservative automated split, not a guarantee that every related template
has been identified.

### Checkpoint and input audit: no retained change

Compared all 720 question sequences from the 100-message development run
against upstream laya's `build_sequence`, pinned at
`6a5819129eb220570792e417e49723d697efd76f`. Token IDs and option-marker positions
matched exactly. The typed checkpoint uses its native 1,024-token sequence
and 256-token header budgets. This checks input serialization and construction;
it is not a new numerical-forward parity test or a claim about calibration.

[Upstream distinguishes the three checkpoints](https://github.com/NandhaKishorM/laya),
so the multilingual checkpoint was also tested with the unchanged classifier
and category descriptions. CUDA BF16 with flash attention, batch size 16:

| Checkpoint      | Correct / 61 | Chunks / 100 messages | Mean runtime |
| --------------- | -----------: | --------------------: | -----------: |
| Typed decisions |           36 |                   144 |       10.92s |
| Multilingual    |           20 |                   141 |        8.49s |

Predictions repeated identically on both passes. Reject the checkpoint switch;
there is no evidence here of a token-format bug to fix. The English base was
not tested in this experiment.

### Frozen embeddings plus a small classifier: rejected

Used the frozen
[paraphrase-multilingual-MiniLM-L12-v2 encoder](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2).
Subject and cleaned body were split into 120-token chunks. L2-normalized chunk
vectors were pooled with token-count weights and normalized again. A logistic
regression classifier used C=1, no class weighting, max_iter=1,000, and seed
20260921, without tuning on validation labels.

Training used 61 messages, covering only ten of the 17 categories. The small
and uneven label set is a material limitation. On the 32 validation cases it
scored **20/32 versus baseline 23/32**. The 658 chunks for all 106 labeled
messages encoded in 0.67–0.70 seconds on CUDA; fitting and prediction took
under 0.02 seconds. Loading and tokenization are outside those encoder timings.
Reject the classifier despite its speed; no trained classifier or production
integration is retained.

### Retrieved labeled examples: retained as an opt-in experiment

Reused the same frozen embeddings to select the two nearest training messages
by cosine similarity. Each demonstration supplies its subject, first 160 body
characters, and category in a separate `labeled_examples` state field. Target
labels are never supplied to inference. Rules, tournament grouping and
length-weighted chunk pooling are unchanged. Both chunk fit checks and model
calls include the examples, so the original body remains fully covered rather
than being silently truncated to make room.

This scored **25/32 versus 23/32**, identically on both initial passes. Because
that gain was small and the validation set had already been used, a fresh
check used sample indices 200–299. Before any predictions for those messages,
67 clear cases were labeled by the assistant from decoded subject/body evidence;
ambiguous and owner-dependent cases were excluded. Removing training senders,
training-template overlaps, and repeated templates within the new validation
set left 40 labels. These new labels have not been independently confirmed by
the user. The training set remained the same 61 messages.

The final retained runner used baseline/retrieval/retrieval/baseline order on
exactly those 40 messages:

| Method             | Correct / 40 | Mean runtime | Chunks | Questions |
| ------------------ | -----------: | -----------: | -----: | --------: |
| Current classifier |           27 |        4.69s |     65 |       325 |
| Retrieved examples |           34 |        7.68s |     90 |       450 |

Both methods repeated identically, with no processing failures. Retrieval
fixed eight errors and regressed one: +17.5 percentage points, with about
1.64 times the classification runtime. Added context creates more chunks,
accounting for part of the cost. Timings exclude model loading and example
preparation. Earlier initial timings are not used for this ratio because model
loading for another experiment overlapped part of that run.

The retained example generator reproduced all 40 demonstration pairs exactly
and checked that every decoded embedding chunk fits the encoder budget.
Leakage-filter, neighbor-selection and separate-state tests failed before their
implementations and passed afterward. Email package tests and Clippy passed.
The small selected sample does not establish whole-mailbox accuracy or
coverage of all categories.

Keep [the Rust audit](../crates/vs1-email/examples/retrieval_audit.rs) and
[offline example preparation](../research/email-retrieval/README.md) as a
reproducible experimental path. The default CLI and active `email.toml` remain
unchanged; a production integration still needs a maintained labeled example
bank and runtime embedding support. No Python inference dependency is added
to the Rust CLI.

### Larger local generative model: rejected in the tested configuration

Tested [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) locally, with the same
category rules and empty owner context. The prompt asked for exactly one
category and treated email content as data. It used the official chat template,
non-thinking mode, greedy decoding, and a 16-token output limit. Inputs had
separate subject, normalized sender address and cleaned body; unlike laya's
chunk tournament, this was a whole-message classification pass, capped at
6,000 body tokens. Three of 106 messages exceeded that cap.

Full BF16 weights failed to fit alongside existing desktop GPU usage before
any prediction. The completed run used bitsandbytes NF4 weights, BF16 compute,
and PyTorch SDPA on CUDA. This was not the Candle flash-attention path.

| Evaluation labels   | Baseline | Qwen3-4B NF4 |
| ------------------- | -------: | -----------: |
| Development         |    36/61 |        27/61 |
| Existing validation |    30/45 |        19/45 |

All 106 outputs parsed as valid category names, so these were classification
errors, not parser failures. Synchronized generation took 31.21 seconds total,
excluding loading and tokenization. The parser's exact-label and rejection
tests passed. Reject this particular model/prompt/quantization configuration;
it does not demonstrate that larger models in general cannot help. No model
replacement or generative integration is retained.

### Artifacts and decision

Only the retrieval experiment is retained. The classifier, checkpoint switch,
and generative-model prototypes are removed from the working implementation.
The full private inputs, frozen labels and split indices, model revisions,
package versions, raw outputs, rejected source, and token-audit records live
under `~/.local/state/vs1-email/next-four-20260921/`. Laya checkpoint revision:
`c5d78730f3493e4fe16d61507ef4b78eef7318cf`; embedding revision:
`e8f8c211226b894fcb81acc59f3b34ba3efd5f42`; Qwen revision:
`1cfa9a7208912126459214e8b04321603b3df60c`. Private messages and labels are not
committed. Concurrent model-compression work is untouched.

## 11. Questions-only rules and whole-email evidence: rejected

Tested the existing typed-decisions checkpoint with 53 binary questions,
3–4 per category, and no `what`, `not_for`, examples or exclusion vetoes.
For each distinct question, take its strongest yes-support across all email
chunks. Rank categories by either their strongest question or the mean of their
two strongest distinct questions. Shortlist five categories, restore config
order, then make one final choice per email using questions, uncalibrated
supports and source excerpts. Choice options contain bare category names.
Subject, sender and date remain separate; recipients are omitted.

The fixed 100-message development sample produces 144 chunks. This asks
7,732 questions per variant, versus 720 for the current classifier. Both initial
passes reproduced the baseline at 36/61 labeled messages, averaging 9.62 seconds.

| Evidence format / pooling              | Correct / 61 | Fixed / regressed versus baseline | Classification time |
| -------------------------------------- | -----------: | --------------------------------: | ------------------: |
| Current classifier                     |           36 |                                 — |          9.62s mean |
| Initial verbose / strongest question   |           25 |                            3 / 14 |         87.79s mean |
| Initial verbose / strongest two        |           23 |                            3 / 16 |         87.87s mean |
| Corrected compact / strongest question |           28 |                            5 / 13 |     86.10s composed |
| Corrected compact / strongest two      |   Incomplete |                                 — |        Not reported |

Both initial variants repeated their predictions exactly. However, their
160/80/0-character excerpt fallback removed all excerpts from 3/100
strongest-question requests and 97/100 strongest-two requests. Those results
therefore do not adequately test the intended source-grounded final pass.

After inspecting that issue, compacted evidence to positional arrays, removed
the redundant category support field, and required nonempty excerpts whenever
the source chunk has text. The corrected budget fallback is 160/80/40
characters. Replayed the saved binary answers to isolate final evidence
formatting: strongest-question completed all 100 messages, using 160 characters
in 91 requests and 80 in nine. It still lost eight correct classifications
against baseline. Its 86.10 seconds combines the original probe time with newly
measured preparation and final inference; it is not a fresh end-to-end timing.

The corrected strongest-two request exceeded the exact encoder budget at
message index 24 even with 40-character excerpts. The replay stopped there,
before a second corrected pass. No accuracy claim is made for that incomplete
variant. Initial shortlists already excluded the expected category on 13/61
labeled emails for strongest-question and 10/61 for strongest-two; a final
choice cannot recover an eliminated category. Directly choosing the highest
pooled support, inspected only as a diagnostic, scored 27/61 and 28/61.

Four tests cover questions-only schema rejection, distinct-question pooling
across chunks, invalid inputs/stable ties, and compact evidence alignment.
They failed before implementation and passed afterward; release Clippy passed.
No validation-set run followed the development regression. This rejects these
specific question wording, pooling and final-evidence configurations, not every
possible questions-only design.

Removed the prototype and kept the active config and classifier unchanged.
Private protocols, amendments, source, tests, raw answers, requests and timing
records are archived under `~/.local/state/vs1-email/question-only-20260921/`.
