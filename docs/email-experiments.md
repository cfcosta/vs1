# Email classifier experiments

Experiments use local Maildir copies and dry-run inference only, with the
`typed-decisions` checkpoint, CUDA BF16, flash attention, and batch size 16.
Descriptions, chunking and pooling stay fixed except for the factor under test.
The development sample has 100 messages and 61 existing labels; a separate
100-message validation sample has 45 labels frozen before viewing predictions.
Three exact-body duplicates were excluded from the initially labeled 48
validation entries before comparing any experiment there. These selected
clear-case subsets do not estimate whole-mailbox accuracy.

Each variant runs twice after model warmup. Timings exclude model loading.
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
