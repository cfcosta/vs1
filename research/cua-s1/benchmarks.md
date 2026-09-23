# Cua-S1 backend comparison — 2026-09-23

This compares email classification, captured Google Flights decisions and live
browser tasks against Jev, Laya and OpenJev. Local runs used CUDA BF16 on an
RTX 3080 Ti with 12 GB shared with the desktop. Cua-S1 used text input only and
`max_len=4096`: one forward pass per question, followed by softmax over only the
option-letter logits at the final position.

The pinned revisions are the `vs1::cua_s1` constants in
[`api.rs`](../../crates/vs1/src/cua_s1/api.rs):

| Model                                             | Constant           | Revision                                   |
| ------------------------------------------------- | ------------------ | ------------------------------------------ |
| `Qwen/Qwen3.5-4B`                                 | `BASE_REVISION`    | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` |
| `cua-ai/cua-s1-4b-0.2` text LoRA (r=16, alpha=32) | `ADAPTER_REVISION` | `16818868b0cc7813808aae4e87b417657046ab79` |

## Email: shared chunking

All backends processed the same 200 private emails, with 156 labeled cases.
Every backend used `split_body` against its own `context_tokens`, then pooled
chunk probabilities by body character count. Laya retained its per-chunk
tournament; OpenJev used `compact512` descriptions. This shares the splitting
and pooling policy, not identical prompts or chunk boundaries. Labeled
abstentions count as incorrect; unlabeled messages remain in the timed run.

| Backend      | context_tokens | Chunks | Correct / labeled | Accuracy | Labeled abstentions | Load seconds |  Run seconds |
| ------------ | -------------: | -----: | ----------------: | -------: | ------------------: | -----------: | -----------: |
| Laya         |           1024 |    352 |            89/156 |    57.1% |                   0 |  0.302876059 | 22.317345195 |
| OpenJev BF16 |            512 |    782 |            67/156 |    42.9% |                  33 |  0.241752504 |  6.135261119 |
| cua-s1       |           4096 |    216 |           126/156 |    80.8% |                   0 |  2.035681718 | 97.819919938 |
| Hosted Jev   |           8192 |    212 |           142/156 |    91.0% |                   0 |  0.000246588 |  9.465786689 |

These are cold-inference runs. Run time includes chunking, tokenization,
inference and pooling; loading is separate. Jev used 16 concurrent HTTP
requests, so its elapsed wall time is not comparable to local GPU time. Its
8192-token context is a vs1-declared budget counted with OpenJev's tokenizer,
not a documented provider limit or the hosted tokenizer.

Sources: `~/.local/state/vs1-email/three-backends-200-20260921/20260923b-{laya,openjev-bf16,cua-s1,jev}.summary.json`.
Only aggregate counts and timings are reported here; email inputs remain
private. The earlier `20260923-*.summary.json` runs predate shared chunking for
every backend and include whole-email inference. Cua-S1 ran out of GPU memory
on that path before its context limit existed (`20260923-cua-s1.stderr`);
there is no completed Cua-S1 summary from that attempt.

## Google Flights replay

The inputs are the captured
[`call3_request.json`](../../crates/vs1/tests/fixtures/jev/call3_request.json) and
[`call5_request.json`](../../crates/vs1/tests/fixtures/jev/call5_request.json).
Medians exclude loading and warmup, with 10 timed repeats per local backend
and 5 for Jev. Target columns show every returned target answer, including
answers unused by the chosen operation; a dash means the question was absent.

| Backend      | Call | Median ms | Operation | Click target | Type-text target |
| ------------ | ---: | --------: | --------- | ------------ | ---------------- |
| Laya         |    3 |    19.507 | CLICK     | 3            | —                |
| Laya         |    5 |    32.029 | CLICK     | 3            | 15               |
| Laya full    |    3 |    26.410 | CLICK     | 3            | —                |
| Laya full    |    5 |    94.140 | DONE      | 10           | 15               |
| OpenJev 1024 |    3 |    15.789 | BLOCKED   | abstain      | —                |
| OpenJev 1024 |    5 |    28.338 | BLOCKED   | abstain      | abstain          |
| cua-s1       |    3 |   471.545 | CLICK     | 2            | —                |
| cua-s1       |    5 |  1683.485 | TYPE_TEXT | 6            | 16               |
| Jev          |    3 |   330.413 | CLICK     | 2            | —                |
| Jev          |    5 |   310.889 | TYPE_TEXT | 16           | 16               |

On call 3, target 2 is “One way” and target 3 is “Multi-city”. On call 5,
targets 3, 6, 10, 15 and 16 are “Skip to main content”, “Flights”, “Google
apps”, “Where from?” and “Where to?”, respectively. **Cua-S1 was the only local
backend that chose the same operation and active target as Jev on both calls**:
CLICK 2, then TYPE_TEXT 16. Its unused click-target answer on call 5 differed.
This is agreement on captured decisions, not a completed Flights task.

Laya used `max_len=512`, `head_max_len=192`; Laya full used 4096 and 2048.
The smaller configuration reached its sequence limit on every call-5 question;
the full configuration did not. OpenJev needed `--max-len 1024`: at 512,
call 5's 24 click-target descriptions did not fit (`replay-openjev.stderr`).
It chose BLOCKED on both calls and abstained on their target questions.

Sources: `artifacts/browser-bench-20260923/replay-{laya,laya-full,openjev-1024,cua-s1,jev}/replay.json`.

## Live browser tasks

Each backend ran the hotel task three times and Wikipedia twice, with at most
15 actions and the built-in independent final checks. All used the compact
browser prompt. Browser model settings match the replay configurations above
(Laya used the smaller configuration). Passes require the independent checks,
not merely a DONE response. Actions and median decision latency are listed in
run order; status was the same in every run of each row.

| Backend | Task      | Passes | Status  | Actions per run | Median decision ms per run |
| ------- | --------- | -----: | ------- | --------------- | -------------------------- |
| Jev     | Hotel     |    0/3 | done    | 4, 4, 4         | 323.790, 294.717, 354.777  |
| cua-s1  | Hotel     |    0/3 | done    | 0, 0, 0         | 695.426, 678.853, 665.628  |
| Laya    | Hotel     |    0/3 | done    | 0, 0, 0         | 40.216, 34.682, 37.170     |
| OpenJev | Hotel     |    0/3 | blocked | 1, 1, 1         | 18.455, 18.068, 18.414     |
| Jev     | Wikipedia |    2/2 | done    | 2, 2            | 317.902, 343.765           |
| cua-s1  | Wikipedia |    0/2 | error   | 0, 0            | 2.194, 1.017               |
| Laya    | Wikipedia |    0/2 | done    | 1, 1            | 26.205, 27.895             |
| OpenJev | Wikipedia |    0/2 | error   | 0, 0            | 1.411, 1.229               |

Decision medians cover model attempts, excluding model load, warmup, navigation,
browser actions, text-helper calls and final verification. The Wikipedia error
rows measure candidate-limit rejection, with no accepted model responses;
they are not successful inference latencies.

- Jev found and opened Casa Flora in every hotel run, but the independent
  `filters` check failed each time. The traces show destination entry, category
  selection and the cancellation checkbox, then opening the property without
  submitting “Find stays”. The earlier report's 3/3 hotel passes are corrected
  to **0/3**. Jev passed Wikipedia **2/2**.
- Cua-S1 and Laya chose DONE on the first hotel decision, before any action.
  OpenJev scrolled once, then chose BLOCKED.
- On Wikipedia, Cua-S1 and OpenJev errored because the page offered more
  click targets than their then-current per-question limits of 26 and 24,
  respectively. Cua-S1 now supports larger sets through option tournaments;
  these recorded runs predate that change.
  Laya clicked the unrelated “quarantined” link, then chose DONE; the article
  check failed in both runs.

Sources: `artifacts/browser-bench-20260923/live-{jev,cua-s1,laya,openjev}-{hotel,wikipedia}/summary.json`
and each directory's `run-*/trace.json`.

## Cua-S1 browser policy mismatch

The live result is not yet a fair test of Cua-S1's intended browser policy.
`vs1-browser` asks a separate generic `operation` question over abstract
CLICK/TYPE_TEXT/SELECT/…/DONE options, and separate target questions. Cua-S1
was trained on one question over concrete (element, action) options such as
`[4] searchbox "Destination" -> fill`. In the first hotel decision it assigned
DONE probability 0.42 (41.6%, from `live-cua-s1-hotel/run-01/trace.json`) even
though the task was incomplete. This demonstrates the current integration's
failure; it does not isolate the model's ability with its native option format.

Follow-ups:

1. Done: add a Cua-S1-native browser policy with combined (element, action) options.
2. Done: [balanced option tournaments](README.md#rust-option-tournaments)
   over groups of at most 26 candidates, with live-task reruns recorded below.

## Follow-up results — 2026-09-23

### Native browser policy

`vs1-browser --backend cua-s1` now defaults to `--policy native`: one question
over concrete (element, action) options, with tournaments above 26 options.
The earlier sections retain the original questions-policy results. These reruns
use the native policy before and after the kernel changes below. Passes require
status `done` and passing independent checks. Statuses, actions and decision
medians are in run order, with the same timing exclusions as above.

| Build          | Task      | Passes | Status per run   | Actions per run | Median decision ms per run |
| -------------- | --------- | -----: | ---------------- | --------------- | -------------------------- |
| Before kernels | Hotel     |    0/3 | done, done, done | 3, 3, 3         | 221.596, 222.693, 221.980  |
| Before kernels | Wikipedia |    2/2 | done, done       | 11, 9           | 1961.946, 1933.833         |
| Final          | Hotel     |    0/3 | done, done, done | 3, 3, 3         | 148.050, 148.142, 149.524  |
| Final          | Wikipedia |    1/2 | done, error      | 11, 15          | 1299.283, 1348.941         |

In every native hotel run, before and after the kernel changes, Cua-S1 checks
“Free cancellation”, selects “Design”, opens Casa Flora, then chooses DONE.
It never types the destination. Jev's hotel run 1 types “Lisbon”, selects
“Design”, checks “Free cancellation”, opens Casa Flora, then chooses DONE.
Neither submits “Find stays”; both pass the `property` check and fail `filters`.

Final Wikipedia run 2 passed the `article` check but ended with status `error`
after hitting the 15-action limit without choosing DONE. The recorded error is
`model-call budget exhausted`; this run is not counted as a pass.

Sources: `artifacts/browser-bench-20260923/live-cua-s1-{native,final}-{hotel,wikipedia}/summary.json`
and their `run-*/trace.json`; Jev sequence:
`artifacts/browser-bench-20260923/live-jev-hotel/run-01/trace.json`.

### Kernel speed progression

The dispatcher's measurements repeat replay request `call5` four times and
report per-question latency, using alternating A/B pairs against the previous
build for each step. The desktop shares the GPU, so absolute values drift
between sessions; this is why each step was measured in alternating pairs.

| Change                         | Previous → new ms per question | Validation                         |
| ------------------------------ | -----------------------------: | ---------------------------------- |
| Fused zero-centered RMSNorm    |                      533 → 497 |                                    |
| Fused causal conv + SiLU       |                      504 → 442 | Bit-identical to the previous path |
| Fused gated RMSNorm            |                442 → about 412 | Faster in all five pairs           |
| Parallel delta-rule recurrence |                about 412 → 370 |                                    |

All six BF16 reference cases kept their top option after every step. These
paired timings and validation results are dispatcher-reported measurements.

### Final email and replay re-measurement

The final email build processed the same 200 messages in 216 chunks at
`context_tokens=4096`. Only aggregate counts and timings are reported.

| Build       | Correct / labeled | Accuracy | Labeled abstentions | Load seconds |  Run seconds |
| ----------- | ----------------: | -------: | ------------------: | -----------: | -----------: |
| `20260923b` |           126/156 |    80.8% |                   0 |  2.035681718 | 97.819919938 |
| `20260923c` |           127/156 |    81.4% |                   0 |  1.997534872 | 62.747356051 |

Sources: `~/.local/state/vs1-email/three-backends-200-20260921/20260923{b,c}-cua-s1.summary.json`.
Both use the cold-inference timing boundary documented above, with loading
separate from run time.

Replay retains `--policy questions`, with 10 timed repeats per call and loading
and warmup excluded. These are whole-request medians, not the per-question
kernel timings above. Every returned answer stayed unchanged, including the
unused click target on call 5.

| Call | Original median ms | Final median ms | Operation | Click target | Type-text target |
| ---: | -----------------: | --------------: | --------- | ------------ | ---------------- |
|    3 |            471.545 |         305.755 | CLICK     | 2            | —                |
|    5 |           1683.485 |        1075.101 | TYPE_TEXT | 6            | 16               |

Sources: `artifacts/browser-bench-20260923/replay-cua-s1/replay.json` and
`artifacts/browser-bench-20260923/replay-cua-s1-final/replay.json`.
