# Original Ultrafast versus vs1-browser, both using Jev

A fresh paired run of all four autonomous tasks on September 20, 2026 produced
**3/4 passes for original Ultrafast and 4/4 for vs1-browser**. This is one trial per
runtime/task, not evidence of a reliable success-rate difference. Both use hosted
Jev; the earlier [Laya comparison](../browser-backends/README.md) is separate.

## Results

Times below cover the agent loop: decisions, text generation, actions, and waits.
They exclude initial navigation and independent final verification/capture.

| Task         | Ultrafast outcome | Ultrafast loop | vs1 outcome | vs1 loop | Actions, Ultrafast / vs1 |
| ------------ | ----------------- | -------------- | ----------- | -------- | ------------------------ |
| Hotel        | Pass              | 3.53 s         | Pass        | 3.47 s   | 5 / 5                    |
| Reading Room | Pass              | 1.33 s         | Pass        | 1.08 s   | 1 / 1                    |
| Wikipedia    | Pass              | 3.70 s         | Pass        | 3.55 s   | 2 / 2                    |
| Flights      | BLOCKED, fail     | 2.28 s         | Pass        | 11.21 s  | 2 / 12                   |

Hotel used the same action sequence in both runtimes: type Lisbon, submit Find
stays, choose Design, enable Free cancellation, and open Casa Flora. Both passed
the filter-state verifier. This corrects any impression from the previous runs
that Jev cannot complete Hotel in our implementation: it failed intermittently,
not consistently. Earlier failures are retained in the earlier report.

Reading Room opened the correct article. Wikipedia filled search and clicked
Search, reaching the correct article in both runtimes. The shared successes had
6, 2, and 5 decision calls respectively in each runtime. Their loop durations
are similar; one sample cannot establish a runtime speed advantage.

On Flights, original Ultrafast clicked the round-trip control twice, then
observed empty text and only two actions and chose BLOCKED. The Rust run selected
One way, filled and confirmed both cities, selected the date, submitted Search,
and waited for results. It passed the corrected independent flight verifier.
The earlier upstream-prompt Rust run failed similarly to Ultrafast's run here,
while other Rust runs succeeded. Dynamic page state and timing remain relevant;
this comparison does not establish the cause of the transient empty observation.

## Process overhead matters

| Task         | Ultrafast wall time | vs1 process wall time |
| ------------ | ------------------- | --------------------- |
| Hotel        | 3.99 s              | 33.79 s               |
| Reading Room | 1.37 s              | 31.40 s               |
| Wikipedia    | 4.40 s              | 31.09 s               |
| Flights      | 3.60 s (failed)     | 16.26 s               |

The Rust CLI unconditionally attempts a final screenshot even when neither
`--screenshots` nor `--record` is requested. Hotel and Reading Room each recorded
a `CDP connection interrupted` screenshot error after successful verification,
with approximately 30 seconds of post-loop delay. Wikipedia also had substantial
post-loop overhead but no screenshot error; individual post-loop operations were
not timed, so its delay cannot be attributed precisely from this measurement.

Ultrafast ran with screenshots disabled and did not capture a final screenshot.
This difference is intentional: these are the runtimes' current default capture
behaviors. Loop times are the closer comparison of agent execution; wall times
show actual runner overhead. Rust's final capture behavior is a concrete issue
worth fixing independently of model quality. No runtime code was changed here.

## Method and evidence

- Original Ultrafast checkout: clean `b8d45982393915a52d935aec30080cd2aea8f411`.
  Its real `Agent.run()` implementation was executed without policy, browser,
  prompt, or budget modifications; a temporary harness collected observations.
- Rust checkout: `5d309ed21624`; `--backend typesafe --prompt upstream --max-steps 60`.
  Ultrafast's default limit is also 60 actions / 120 decision calls.
- Both: `TYPESAFE_MODEL=jev-latest`; OpenRouter text helper
  `inception/mercury-2.5`, reasoning disabled; Chrome `152.0.7977.82`.
- Both used the exact goal and source URL from each agent JSON. The local tasks
  used the same vs1 fixture file; its differences from upstream's copy are
  formatting. Fresh tabs were used, with a shared browser profile.
- Each pair ran Ultrafast first, then Rust. Site/network state was not frozen;
  order was not randomized. Rust's upstream prompt preserves the upstream rules
  but is not byte-identical in whitespace/serialization.
- Fresh final observations were evaluated against the same JSON JavaScript
  verifier in both runtimes. Success additionally required DONE and no run error.
  Flights used the corrected airport-suffix field-label check in both arms.
- Ultrafast used its locked dependencies in an isolated temporary environment,
  including browser-harness 0.1.13. The original checkout remained clean. Its
  dedicated measurement daemon was stopped afterward; Chrome was left running.
- Ultrafast wall time starts before Agent construction and ends after verification,
  close, and artifact writes; Rust wall time includes process launch and exit.
  Neither number includes installing dependencies or building executables.

[results.json](results.json) preserves outcomes, timings, histories, configuration,
and source hashes. Raw observations, logs, and the temporary measurement harness
are in `artifacts/ultrafast-comparison-20260920/`. No Python integration was added
to vs1. The constrained examples are not included here because Ultrafast's agent
has no equivalent supplied-plan execution mode.
