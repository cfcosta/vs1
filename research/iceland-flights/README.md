# São Paulo → Iceland: browser-agent benchmark

**Result: no verified flight fare and no cheapest departure date.** The default
agent failed its single attempt; three attempts with an adjusted goal and the
upstream prompt also failed. None reached a flight-results page. These results
measure failed searches, not the price or availability of flights to Iceland.

## Scope

Run September 20, 2026 using `vs1-browser` at `2e1231c5a9d2` and its default hosted
Jev backend. “Next month” was taken as **October 1–31, 2026**, for **one adult,
one-way, economy**, with **BRL** prices and **São Paulo (SAO/all airports)** as
origin. These were stated working assumptions; the user had not confirmed them.

The first task requested Iceland broadly. After that failure, the adjusted goal
used Keflavík (KEF) and instructed the agent to load October 1 results before
comparing fares across October. The adjusted goal also used `--prompt upstream`;
it is a separate arm, not a controlled test of only prompt wording. Both goals
required a month-wide fare comparison and prohibited booking.

## Results

| Run                              | Outcome                  | Agent loop | Actions | Decision calls | Median decision | p95 decision | Text-helper calls |
| -------------------------------- | ------------------------ | ---------- | ------- | -------------- | --------------- | ------------ | ----------------- |
| Default, Iceland, compact prompt | Action budget exhausted  | 32.76 s    | 60      | 80             | 303 ms          | 353 ms       | 3                 |
| Adjusted KEF, upstream, trial 1  | BLOCKED                  | 4.36 s     | 4       | 7              | 369 ms          | 948 ms       | 1                 |
| Adjusted KEF, upstream, trial 2  | Invalid text-helper JSON | 1.72 s     | 0       | 1              | 892 ms          | 892 ms       | 1                 |
| Adjusted KEF, upstream, trial 3  | BLOCKED                  | 3.92 s     | 4       | 9              | 322 ms          | 378 ms       | 1                 |

Independent route/date/results verification failed in every run. Month-wide
minimum-price verification also failed: there were no valid fare observations
from which to identify a cheapest date. The JSON verifier checks only route,
one-way selection, an October date/year, and visible flight options; even a pass
would not by itself prove the cheapest fare across the month, passenger count,
or cabin. No such limited pass occurred here.

Wall time was 33.89 seconds for the default invocation, 5.43 seconds for adjusted
trial 1, and 7.77 seconds for the invocation containing adjusted trials 2–3.
Individual repeated-run wall times were not instrumented. Agent-loop timing
includes decisions, text generation, actions, and waits, excluding setup and final
verification. No screenshots or recordings were requested or written, and there
were no screenshot delays. p95 is the nearest-rank percentile over model attempts.

## Failure evidence

The default agent selected Iceland, eventually changed to one-way, and selected
October 1. It then repeatedly opened/confirmed/navigated the calendar without
submitting a flight search. Its last observation displayed February–April 2027.
It encountered October's date labels but **no priced October calendar labels**;
seeing a date is not comparing its fare. The text helper also supplied “São Paulo,
Brazil” and “Iceland” when the selected input was Departure. Those values were
recorded in the executed history. This run exhausted its 60-action budget.

Adjusted trials 1 and 3 filled and selected Keflavík, clicked the round-trip
control twice, then observed empty page text and chose BLOCKED. This resembles
the transient empty-state failure observed in the earlier Ultrafast comparison.
The traces do not establish whether the underlying cause is rendering, observation,
or timing, and this is not evidence of a local inference-kernel bug: Jev was hosted.

Adjusted trial 2 failed on the first text-generation call with:

```text
text helper returned invalid JSON; nothing typed: trailing characters at line 2 column 1
```

The parser stopped before typing; the executor did not guess a field value.
No booking or purchase actions were performed.

## Configuration and reproduction

- `TYPESAFE_MODEL=jev-latest`; returned model `jev-1.13.0`.
- Text helper: `inception/mercury-2.5` through OpenRouter, reasoning disabled.
- Chrome `152.0.7977.82`, CDP on port 9222, fresh tab for each run, shared profile.
- Maximum 60 executed actions / 120 model attempts per run.
- Default `compact` prompt for the original arm; `upstream` for the adjusted arm.
- No local inference, manual browser corrections, hardcoded field values, or
  policy changes during the runs. Retrying the adjusted task started fresh tabs.

With credentials in `TYPESAFE_API_KEY` and `TEXT_MODEL_API_KEY`:

```sh
export TEXT_MODEL_BASE_URL=https://openrouter.ai/api/v1
export TEXT_MODEL=inception/mercury-2.5
export TEXT_MODEL_REASONING=none
export TYPESAFE_MODEL=jev-latest

vs1-browser --scenario research/iceland-flights/scenario.json \
  --max-steps 60 --output artifacts/iceland-new-default

vs1-browser --scenario research/iceland-flights/keflavik.json \
  --prompt upstream --max-steps 60 --repeat 3 \
  --output artifacts/iceland-new-adjusted
```

The adjusted observations in this report were collected as one trial followed by
a separate two-repeat invocation. The commands above reproduce the same task
settings, not the exact live-site state or service responses.

[results.json](results.json) includes configuration, scenario hashes, timings,
provider-reported decision-token usage, executed histories, and final observations.
Raw requests/responses and text-helper metadata remain under
`artifacts/iceland-flights/`. Reported token usage is not an estimate of billing.

This task exposed two requirements beyond the earlier fixed-date flight demo:
reliably reaching a populated results page, and verifying a comparison of priced
options across the requested month. Neither was achieved. A useful next diagnostic
would isolate the empty trip-type-menu observations and check text-helper field
selection before attempting another month-wide search.
