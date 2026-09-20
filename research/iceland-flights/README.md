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

## Original Ultrafast cross-check

The unmodified original Ultrafast agent completed **0/1 original task and 0/3
adjusted KEF attempts** on the same day, with the same scenario URLs/goals,
60-action limit, Jev endpoint/model, Mercury text helper, and JSON verifiers.

| Ultrafast task        | Result  | Agent loop | Actions | Decision calls |
| --------------------- | ------- | ---------- | ------- | -------------- |
| Original Iceland goal | BLOCKED | 4.34 s     | 3       | 7              |
| Adjusted KEF, trial 1 | BLOCKED | 3.58 s     | 3       | 8              |
| Adjusted KEF, trial 2 | BLOCKED | 4.52 s     | 7       | 10             |
| Adjusted KEF, trial 3 | BLOCKED | 3.72 s     | 3       | 8              |

All four final observations contained empty page text after interaction with
“Change ticket type. Round trip.” None reached a matching results page, exposed a
verified fare, or completed the month-wide comparison. The second KEF trial also
struggled to confirm the destination before reaching the trip-type control.
All text-helper responses parsed successfully in these Ultrafast attempts.

This reproduces the empty-observation failure in the canonical implementation:
it is not unique to our Rust port and does not involve local inference. It does
not isolate whether the cause is browser rendering, the shared snapshot logic,
interaction timing, or policy handling of a transient page state. Live browser
state and remote model responses vary between runs. Ultrafast always used its
original prompt; therefore only the adjusted task has the same prompt style as
the earlier Rust runs. The original-goal Rust attempt used the compact prompt.

Ultrafast checkout `b8d45982393915a52d935aec30080cd2aea8f411` remained clean.
Only an external measurement wrapper recorded ticks and model calls, without
changing browser actions or policy decisions. It used browser-harness 0.1.13,
a fresh tab for every attempt, the same Chrome profile, and no screenshots.
The dedicated measurement daemon was stopped afterward.

[ultrafast-results.json](ultrafast-results.json) contains the full comparison
summaries, final observations, scenario hashes, and executed actions. Raw state,
model request/response bodies, and the measurement script are under
`artifacts/iceland-flights/ultrafast/`. No credentials were recorded.

## Chrome-CDP diagnosis: background animation throttling

Direct Chrome-CDP inspection reproduced the empty snapshot **without any model
calls**. The page's trip-type menu is functional; the failure is a timing problem
between background rendering and the agents' observation/termination behavior.
No runtime or policy code was changed during this diagnostic.

Using the same 1120×780 viewport, background-tab creation, focus emulation, and
mousePressed/mouseReleased sequence as the agents:

| Time after first post-click probe | Background snapshot                     | One-way option's menu opacity |
| --------------------------------- | --------------------------------------- | ----------------------------- |
| 0 ms                              | Main page still exposed                 | 0                             |
| 53 ms                             | Main page still exposed                 | 0                             |
| 256 ms                            | Main page still exposed                 | 0                             |
| 1,258 ms                          | Empty text; only scroll/wait actions    | 0                             |
| 4,262 ms                          | Round trip, One way, Multi-city exposed | 1                             |

The One way DOM node existed with nonzero viewport geometry during the empty
phase. It had no aria-hidden or inert ancestor, but its menu ancestor's computed
opacity was zero. The snapshot correctly excluded that still-invisible option.
In the foreground comparison, all three options appeared by the 52 ms sample.
A screenshot and the accessibility tree independently showed the working menu.
These are sampled observation times, not exact transition timestamps.

Four requestAnimationFrame callbacks measured intervals of approximately
**1,017 ms in the background** versus **16.7 ms in the foreground**, despite
`document.visibilityState === "visible"` and `document.hasFocus() === true`
under focus emulation. Bringing the exact same background tab to the foreground
changed its intervals to **24.3, 16.7, and 16.7 ms**. This activation control
supports background frame throttling as the mechanism in this environment.

After waiting for the menu to become visible, a normal CDP click successfully
selected One way in the background tab. No hidden element was force-clicked,
no field value was injected, and no model was involved.

Both implementations create background tabs and use a **50 ms** settle timeout
for ordinary clicks (200 ms for autocomplete fills). That is much shorter than
this background menu transition. The agent can receive an old or empty snapshot,
click the opener again, or decide BLOCKED while the menu is still animating.
Our `observe()` retries missing/error snapshots, but accepts a valid snapshot
with empty text and no page controls, so its existing retries do not cover this
case. This explains the reproduced failure mode; it does not prove that every
historical failure or the month-wide planning loop had the same cause.

The next fix should bound a wait for usable post-click observations/menu readiness
before asking the model for another action or accepting BLOCKED. It should not
weaken visibility checks to expose transparent controls or replay mutations.
Foreground execution is another rendering option, but changes the user's active
tab. Merely applying focus emulation did not restore normal frame cadence here.

[cdp-diagnosis.json](cdp-diagnosis.json) preserves the reduced, account-free
measurements. Raw DOM probes, frame timings, and the screenshot remain locally
under ignored `output/chrome-cdp/iceland/`; the two diagnostic tabs were closed.

## Readiness fix and follow-up run

The runner now polls actual snapshots after input instead of depending on two
animation frames and a 50 ms fallback. Polling starts after a short settle delay
(100 ms, or 200 ms for fills), checks every 50 ms, and has a five-second readiness
budget. Empty post-input snapshots are not sent to the model. A non-editable
popup opener additionally requires visible options or a visible dialog; an
aria-controls placeholder alone is insufficient. Existing opacity, aria-hidden,
inert, and viewport checks remain in force. Old-document node IDs are not reused
to infer popup readiness after navigation. The separate CDP transport timeout
still applies if Chrome stops answering calls.

The exact snapshot that satisfied readiness is returned to the policy. Navigation
interruptions cause another read, not another input. Expiry is a terminal
readiness error rather than a retryable stale-mutation error. There are no new
screenshots, foreground-tab activations, or model calls during settling.

Validation passed: browser tests in default/local builds, Clippy in both builds,
the browser guard suite, and all five constrained hotel variants. A dedicated
Chrome regression test delayed a menu behind opacity zero and an empty
aria-controls placeholder; the observer waited for the visible option and the
opener's click count stayed exactly one. Unit tests cover intervening navigation,
read-only polling, and a terminal timeout without mutation replay.

A fresh adjusted KEF/upstream run **passed the trip-type bottleneck**, selected
One way, and reached a flight search and price graph. It did not complete the
full cheapest-flight task: the no-progress guard stopped it after repeated graph
scroll actions. The final view showed October 20 and a “From R$2,720” graph value,
not a verified cheapest itinerary across the month. Do not treat this as a flight
recommendation or a successful fare-search benchmark.

The run took **29.12 seconds** of agent-loop time, with **33 actions, 48 decision
calls**, and **341 ms median decision latency**. Independent final verification
failed because it ended in the graph rather than matching flight options. Earlier
failures remain recorded. [readiness-results.json](readiness-results.json)
preserves the follow-up summary and action history.

The live regression can be run against Chrome CDP on port 9222 with:

```sh
cargo test -p vs1-browser --bin vs1-browser delayed_transparent_menu -- --ignored
```
