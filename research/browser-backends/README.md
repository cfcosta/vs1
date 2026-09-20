# Laya versus Jev on the Ultrafast JSON examples

Tested September 20, 2026, using the same Rust browser runner, goals, browser
guards, text helper, and independent JavaScript verifiers. This compares backends;
it does not compare Rust runtime speed against Ultrafast's Python runtime.

With the compact prompt, **Jev completed 3/4 autonomous tasks; Laya completed 0/4**.
Both passed all eight constrained cases. The constrained plans supply action
sequencing, field text, candidate filtering, and postconditions, so these are
separate results from autonomous task completion.

## Autonomous results

Each cell shows independent outcome and agent-loop elapsed seconds. Each arm ran
once per task/backend, with a 20-action limit (at most 40 model attempts).

| Task         | Laya, compact | Jev, compact  | Laya, upstream prompt | Jev, upstream prompt |
| ------------ | ------------- | ------------- | --------------------- | -------------------- |
| Hotel        | Fail, 4.90 s  | Fail, 3.44 s  | Fail, 104.34 s        | Fail, 3.50 s         |
| Reading room | Fail, 2.25 s  | Pass, 1.02 s  | Fail, 3.35 s          | Pass, 1.09 s         |
| Wikipedia    | Fail, 7.42 s  | Pass, 3.44 s  | Fail, 6.81 s          | Pass, 4.31 s         |
| Flights      | Fail, 4.82 s  | Pass, 11.07 s | Fail, 4.80 s          | Fail, 1.77 s         |

Compact Flights uses a fresh rerun after the verifier correction below; both
backends were rerun. The original compact run also reached the requested flight
results with Jev, but exited unsuccessfully under the original verifier.
All original outcomes remain in [results.json](results.json).

Laya declared DONE without acting on Hotel, Reading Room, and Flights under the
compact prompt. On Wikipedia it clicked an unrelated image and then reported
BLOCKED. Under the upstream prompt it clicked “Find a stay” 20 times in Hotel
until the action budget was exhausted; it did not complete any task.

Jev's hotel failure is substantive under both prompts: it typed Lisbon, selected
Design, checked Free cancellation, and opened Casa Flora, but never clicked
“Find stays” to apply the destination search. The final page explicitly said
“Destination anywhere.” Opening the right hotel alone is insufficient.

Jev's upstream Flights run clicked the ticket-type control twice, then observed
an empty page and reported BLOCKED. Its compact rerun completed the search.
These live pages are asynchronous and runs share the browser profile; this is
not evidence that one prompt reliably beats the other on Flights.

Across the compact runs in the table, median decision-call latency was **3.36 s
for local CPU Laya** (6 calls) versus **0.325 s for hosted Jev** (33 calls).
These are different hardware and different action trajectories, not a controlled
inference benchmark. Timings cover first decision through termination, including
actions, waits, and text generation, but exclude loading, initial navigation,
final verification, and screenshots. An upstream Reading Room screenshot failed
after successful verification; it is recorded separately and not task failure.
Whole-process times, including capture delays, are also retained in results.json.

## Constrained results

| Scenario                                                        | Laya | Jev | Non-singleton model calls per backend |
| --------------------------------------------------------------- | ---- | --- | ------------------------------------- |
| Hotel: base, distractors, reordered, already filtered, complete | 5/5  | 5/5 | 5 total                               |
| Reading room: base, reordered, complete                         | 3/3  | 3/3 | 0                                     |

These used the default whole-word overlap retrieval. Reading Room leaves a
singleton and is deterministic; its success supplies no evidence of model quality.
Already-satisfied steps are skipped. There was one run per variant, not repeated
trials or eight independent task families.

## Flight verifier correction

The original verifier copied upstream's exact `Where from?` label lookup.
Google's resulting page now exposed the fill control as `Where from? Zürich ZRH`,
with value `Zürich`. Jev had reached the requested one-way Zurich-to-London search,
September 20, 2026, with matching visible flight options, but the label lookup
falsely rejected it.

The JSON verifier now accepts origin/destination fill-control labels with an
appended space-separated suffix, while retaining exact values, route, ticket
type, date/year, and flight-result checks. It does not match unrelated “Open”
buttons. Five browser-executed checks accepted the captured matching page and
rejected changed origin, destination, date, and missing results. A fresh compact
live rerun then passed with Jev and failed with Laya. The fixed date is retained
from upstream. Passenger count and cabin remain unchecked, as in upstream.

## Configuration and reproduction

- Runner baseline: `9e7200fc9512`, plus the flight JSON verifier fix in this report's commit.
- Laya: `convaiinnovations/laya`, snapshot
  `c5d78730f3493e4fe16d61507ef4b78eef7318cf`, CPU F32, sequence limit 512,
  head limit 192; no retraining or checkpoint changes.
- Jev: requested `jev-latest` through TypeSafe; responses identified `jev-1.13.0`.
- Shared text helper: `inception/mercury-2.5` through OpenRouter.
- Chrome `152.0.7977.82`, CDP on port 9222, fresh tab for each run.
- No screenshots/recordings requested, although the runner always attempts a final capture.

Set `TYPESAFE_API_KEY` and `TEXT_MODEL_API_KEY` in the environment, with
`TEXT_MODEL_BASE_URL=https://openrouter.ai/api/v1` and
`TEXT_MODEL=inception/mercury-2.5`. For each of `hotel-agent`,
`reading-room-agent`, `wikipedia`, and `flights`:

```sh
vs1-browser --scenario examples/hotel-agent.json \
  --backend local --checkpoint "$CHECKPOINT" \
  --prompt compact --max-steps 20 --output artifacts/hotel-local-new

vs1-browser --scenario examples/hotel-agent.json \
  --backend typesafe --prompt compact --max-steps 20 \
  --output artifacts/hotel-jev-new
```

Repeat with `--prompt upstream` for the second arm. For constrained cases use
`examples/hotel.json` and `examples/reading-room.json`; the default retrieval is
`overlap`. Every output path must be fresh. Checkpoints and credentials are not
included in this report.

[results.json](results.json) contains outcomes, actions, timings, source hashes,
and relative paths to raw traces in ignored `artifacts/backend-comparison-*`
directories. Eighteen autonomous runs and sixteen constrained variant runs were
performed. Browser unit tests passed (17), as did the five verifier checks.

Jev is clearly more useful for autonomous tasks in this small comparison. The
results support using constrained plans with Laya; they do not isolate a kernel
bug or establish general model quality across websites.
