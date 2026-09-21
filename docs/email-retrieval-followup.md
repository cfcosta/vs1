# Retrieval follow-up: coverage, voting, context and selective use

This follows the [300-message retrieval comparison](email-experiments.md#21-fresh-300-message-retrieval-ablations-across-three-backends).
The previous 300 messages are diagnostic data (257 scored references). A new
300-message sample has 271 scored references and 29 unresolved cases. References
were reviewed by the assistant and frozen before classifier predictions; they
are not independently verified ground truth. Both sets use unchanged model
weights and the existing root configuration.

## Frozen experiments

1. Expand the original 156-example bank with independently selected messages:
   five bills, five statements/brokerage notes and eight executed documents.
   Compare full retrieved context from the original and expanded banks on both
   local backends. Bank expansion can change retrieved text and chunk boundaries;
   this measures the whole bank change rather than an isolated semantic effect.
2. Classify by five nearest neighbors, summing nonnegative cosine similarities
   per category. Ties follow nearest-neighbor order. Compare standalone voting
   with a hybrid: use the vote if the top two neighbor labels agree, otherwise
   use Laya with its unchanged baseline request. No target labels guide routing.
3. Give OpenJev labels-only examples with chunk boundaries fitted to those
   labels, instead of reserving space for full examples. Compare both accuracy
   and cost with its natural and full-reservation controls.
4. Run Laya full retrieval selectively: the top two neighbor labels must agree,
   their similarities must be at least 0.6 and 0.5 respectively, and the baseline
   predicted category must have at least three examples in the bank. Otherwise
   keep the baseline prediction. These thresholds were fixed before fresh labels
   or predictions; no threshold search was performed.

The hybrid chooses whether to call Laya at all. Selective full retrieval needs a
baseline Laya prediction first and then makes additional calls. They are distinct
experiments with different cost structures.

## Data and reproducibility

Seed 20260921400 sampled fresh candidates from the mailbox. Excluded raw hashes
from the previous 500-candidate pool and old 200-message evaluation, bank senders,
and body templates with five-word-shingle Jaccard similarity >=0.5 to the bank,
previous diagnostic set or another selected fresh message. Of 600 candidates,
226 were rejected for bank senders and 47 for template overlap before reaching 300. Sender exclusion is by normalized email address, not corporate domain.
Heuristic template checks do not prove semantic independence.

Bank additions were manually reviewed from separate candidate pools, excluding
diagnostic senders and templates. Eighteen survived: one bill overlapped a
diagnostic template and one contract duplicated another bank template. Added
references were chosen for category coverage before observing new predictions.
No test message or test reference was added to the bank. Mailbox files were only
read; frozen copies and all per-message artifacts remain private.

Fresh reference distribution: bulk 117, other 40, careers 32, security 20,
ops 19, receipts 18, travel 8, bills 5, fiscal 4, papers 3, identity 2, clients 2,
income 1, unresolved 29. There are **no scored capture examples** in the fresh
sample, and only eight scored bills/papers combined. This substantially limits
fresh evidence about completing bank coverage. Many unresolved cases need owner
relationships, missing body/attachment content, or a statement-versus-bill tie
resolved. Reference labels were not revised after predictions.

Pinned multilingual MiniLM and the previous 120-token chunk / weighted embedding
pooling scheme were reused. A shared preparation encoded 174 bank messages and
600 targets: 5,201 embedding chunks in 163 forward batches, one encoder API call,
3.30 seconds of encoding and 9.51 seconds total preparation. This cached-model
research pass excludes sampling, reference review and independent-overlap checks;
it is not directly comparable with the earlier 17–18 second preparation timer
that included leakage checks. All 300 original diagnostic neighbor pairs were
reproduced exactly. Every local full-context run used tokenizer-only example
fitting before classification, retaining the original unfitted examples.

Laya uses BF16 CUDA with FlashAttention, batch 16. OpenJev uses F32 CUDA, batch
four, and its established compact 512-token request. Hosted Jev remains a
baseline only. GPU timing is observational on the shared workstation. Classifier
logical requests, questions and GPU batches are distinct counts; totals below
exclude retrieval preparation. OpenJev call timing excludes input construction,
whereas Laya's batch API includes tokenization.

## Why the hybrid can help

On the fresh original-bank split, the nearest two labels agree for 133 messages,
122 of them scored. Voting gets 84 of those right, versus Laya's 59. They disagree
for 167 messages, 149 scored; Laya gets 76 right there, versus voting's 56.
Combining those complementary slices yields 160/271 in the saved-output replay.
On the older diagnostic set the same hybrid scores only 133/257 against Laya's
134/257, so this is not evidence of a universal advantage. The routed inference
run below checks that re-batching does not invalidate the replay result.

## Decisions

- **Keep the original-bank hybrid as an opt-in research improvement.** Actual
  routing scores 160/271 (59.0%), versus Laya's 135/271 (49.8%). It fixes 36
  baseline errors and introduces 11. Macro F1 improves from .367 to .425;
  full retrieval scores 156/271 but has macro F1 .310. The hybrid uses 331
  classifier chunks and 662 logical calls, 42.7% fewer than the baseline's
  578 chunks / 1,156 calls. Classifier total time is 23.73 versus 40.46 seconds,
  before retrieval preparation. All 300 routed predictions match the replay.
  It still regresses two of eight travel references, and its older diagnostic
  score is 133/257 versus 134/257. This is evidence for a useful option, not a
  universally better default.
- **Keep native label-context budgeting as an opt-in benchmark mode.** On fresh
  OpenJev, it scores 118/271 versus 111/271 naturally and 100/271 with full-example
  space reserved. It uses 1,436 rather than 3,440 chunks, taking 28.38 rather than
  52.40 classifier seconds. On the diagnostic set it scores 95/257, still below
  the natural baseline's 97/257. The expanded bank scores 117/271 on the fresh
  native mode, so it provides no further gain.
- **Do not adopt the expanded bank.** Laya full retrieval falls 154→149 on the
  diagnostic set and ties 156→156 on fresh messages. The fresh tie consists of
  three fixes and three regressions; papers improve 0→2, bulk falls 84→81, and
  bills stay 1/5. OpenJev full retrieval changes 100→101 diagnostically and
  105→104 fresh. The expanded-bank hybrid's actual fresh score is 153/271,
  below the original bank's 160. Keep the bank only in private research
  artifacts; the original bank and production configuration remain untouched.
- **Reject the similarity/coverage gate.** Original-bank gating reaches 141/271
  fresh, versus 156 with full retrieval and 160 with the hybrid. It requires
  the baseline plus 404 additional calls, taking 56.55 classifier seconds—more
  than full retrieval's 53.75. On diagnostics it loses 134→131 against baseline.
  Expanded-bank gating scores 138/271 fresh and takes 57.95 seconds. Remove the
  gate from retained source; preserve its private prototype and findings.
- **Do not substitute standalone voting for the accuracy-oriented hybrid.** It
  scores 131/257 diagnostic and 140/271 fresh with the original bank, versus
  126 and 136 with the expanded bank. It avoids classifier calls, but the
  hybrid is more accurate on fresh messages. The voting helper remains as a
  required component of that hybrid.

The expanded hybrid demonstrates why actual routing matters: one prediction
changed after re-batching its Laya subset, reducing the replay's 154/271 to
153/271. The original hybrid and both gates exactly reproduced their replay
predictions. Report measured routed results rather than assuming saved-output
combinations are invariant to batching.

Observed failing tests before implementing native context budgeting, voting,
the gate and the pre-inference route helper. Seven Rust benchmark tests and six
retained Python research tests pass; Clippy and canonical formatting pass. The
rejected gate and its test remain only in private research artifacts. No new
retrieval or routing default was enabled in the production CLI.

Aggregate metrics, per-category scores, paired changes, model/input hashes and
repeat checks are in [email-retrieval-followup.json](email-retrieval-followup.json).
Private reproduction scripts, frozen raw messages, references, neighbor scores,
expanded bank and rejected prototypes are under
`~/.local/state/vs1-email/retrieval-followup-20260921/`.

## Measured results

All classifier totals exclude shared retrieval preparation and tokenizer-only example fitting. Macro F1 averages reference categories present in each split (12 diagnostic, 13 fresh).

| Fresh run                             | Correct /271 | Macro F1 | Chunks | Calls | Questions | Call wall s | Total s |
| ------------------------------------- | -----------: | -------: | -----: | ----: | --------: | ----------: | ------: |
| jev-baseline                          |          222 |    0.594 |    300 |   300 |       300 |        8.02 |    8.13 |
| openjev-baseline                      |          111 |    0.368 |  1,310 | 1,310 |     1,310 |       19.51 |   25.81 |
| laya-baseline                         |          135 |    0.367 |    578 | 1,156 |     2,890 |       35.04 |   40.46 |
| original/openjev-labels-native        |          118 |    0.379 |  1,436 | 1,436 |     1,436 |       21.42 |   28.38 |
| original/laya-full                    |          156 |    0.310 |    697 | 1,394 |     3,485 |       47.14 |   53.75 |
| original/openjev-full                 |          105 |    0.370 |  3,440 | 3,440 |     3,440 |       51.38 |   68.10 |
| original/openjev-labels-matched       |          100 |    0.350 |  3,440 | 3,440 |     3,440 |       35.82 |   52.40 |
| expanded/openjev-labels-native        |          117 |    0.379 |  1,436 | 1,436 |     1,436 |       21.47 |   28.54 |
| expanded/laya-full                    |          156 |    0.337 |    697 | 1,394 |     3,485 |       47.33 |   54.03 |
| expanded/openjev-full                 |          104 |    0.374 |  3,455 | 3,455 |     3,455 |       52.01 |   69.25 |
| original/hybrid_laya (actual routing) |          160 |    0.425 |    331 |   662 |     1,655 |       20.51 |   23.73 |
| original/gate_full (actual routing)   |          141 |    0.388 |    780 | 1,560 |     3,900 |       49.07 |   56.55 |
| expanded/hybrid_laya (actual routing) |          153 |    0.420 |    324 |   648 |     1,620 |       20.17 |   23.28 |
| expanded/gate_full (actual routing)   |          138 |    0.381 |    798 | 1,596 |     3,990 |       50.28 |   57.95 |

Standalone voting uses zero classifier calls: original bank 140/271 (macro F1 .281), expanded bank 136/271 (.304). It still pays the embedding/retrieval cost above.

| Diagnostic run         | Correct /257 |
| ---------------------- | -----------: |
| jev-baseline           |          217 |
| openjev-labels-native  |           95 |
| openjev-baseline       |           97 |
| laya-full              |          154 |
| openjev-labels         |           94 |
| openjev-full           |          100 |
| laya-baseline          |          134 |
| expanded/laya-full     |          149 |
| expanded/openjev-full  |          101 |
| original/vote          |          131 |
| original/hybrid_replay |          133 |
| original/gate_replay   |          131 |
| expanded/vote          |          126 |
| expanded/hybrid_replay |          136 |
| expanded/gate_replay   |          134 |

Fresh repeats reproduced all 300 OpenJev native-label predictions and all 167
Laya fallback predictions in the retained hybrid. Combined with the unchanged
voting branch, the hybrid repeats all 300 final decisions. Jev's fresh baseline
used 300 HTTP attempts, 300 successes and zero retries; no Jev variant was tuned.

Gated classifier totals sum separately measured baseline and subset runs,
including both model loads. They are not a measurement of one process reusing
its loaded model. Hybrid totals measure only the routed Laya subset; retrieval
preparation and the sub-millisecond Python decision-combination step are
recorded separately. The existing gates' extra calls remain even if model
loading is amortized.
