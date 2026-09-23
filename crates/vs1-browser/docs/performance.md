# Browser CLI measurements — September 19, 2026

The Rust CLI works as the `examples/run.py` equivalent: an arbitrary URL and goal,
observed-element decisions, browser execution, and terminal progress. The browser
runtime passed verified hotel and Wikipedia tasks with Jev. **The shipped local
Laya checkpoints did not reproduce Jev's task success. There is no demonstrated
end-to-end speedup with local inference.**

All measured task summaries, including failed attempts, are in
[measurements.json](measurements.json). Raw traces and screenshots are in the
ignored `../artifacts/` directory. The final plain-CUDA comparison uses the
`final-*` entries; earlier measurements are retained as development diagnostics.

## Decision latency on the same captured request

Input: the repository's `crates/vs1/tests/fixtures/jev/call5_request.json`, containing a real
Google Flights observation and operation/click/text target questions. The request
is identical across these replay calls. Local inference uses an RTX 3080 Ti,
CUDA BF16, and the root Laya checkpoint. Jev is the live `jev-1.13.0` service.
Model loading and a shape-specific warmup are excluded from every reported median.

| Backend              | Context policy                              | Repeats | Median decision time | Last selected operation/target |
| -------------------- | ------------------------------------------- | ------: | -------------------: | ------------------------------ |
| vs1, plain CUDA      | Checkpoint default: 512 tokens per question |      20 |         **60.18 ms** | CLICK / 3                      |
| vs1, plain CUDA      | 4096 maximum, 2048 header budget            |       5 |        **640.07 ms** | DONE                           |
| vs1, Flash Attention | Checkpoint default: 512 tokens per question |      20 |         **37.16 ms** | CLICK / 3                      |
| vs1, Flash Attention | 4096 maximum, 2048 header budget            |      10 |        **108.27 ms** | DONE                           |
| Jev HTTP             | Provider's request handling                 |       5 |        **334.56 ms** | TYPE_TEXT / 16                 |

The 60 ms result is 5.56× faster than the live Jev call, **but it truncates the
input and selects a different action**. The original state alone tokenizes to
1,170 tokens locally; all three default-budget sequences hit 512. This is not an
accuracy-equivalent speed comparison.

With the larger budget, the sequences contain 1,546, 2,246, and 1,637 tokens and
no longer hit the sequence limit. On plain CUDA, that workload is 1.91× slower than Jev.
Option descriptions remain subject to the library's 48-token per-option cap.
Increasing context did not recover the expected decision. Different models and
tokenizers mean their reported token totals should not be treated as equivalent
compute workloads.

The then-separate `flash-attn` feature (now part of `cuda`) uses the library's packed attention
path. It reduces full-context latency from 640.07 to **108.27 ms**, making that
local replay **3.09× faster than Jev**. Its full-context operation remains the
incorrect DONE. Three additional hotel runs with this build also failed
independent verification after premature DONE (41.91 ms median to the incorrect
stop). The acceleration is measured; equivalent browser capability is not.

The first Flash Attention build failed when compiler temporary files exhausted
disk space. Retrying with compiler temporary files in RAM completed successfully.
Its binary hash, feature metadata, ten full-context samples, and task results are
recorded separately under `flash_attention` in the measurements JSON.

## End-to-end outcomes

Three alternating local/Jev pairs used the same Rust executable, Chrome profile,
1120×780 viewport, hotel fixture, and natural-language goal. Local used the compact
prompt; Jev used the upstream policy. The common text helper was
`inception/mercury-2.5` through OpenRouter, with reasoning disabled. Local never
reached a typing action. Both backends had a 15-action diagnostic limit.

| Backend and task                            | Attempts | Independently passed | Task time                                                              |
| ------------------------------------------- | -------: | -------------------: | ---------------------------------------------------------------------- |
| Local Laya, hotel                           |        3 |              **0/3** | 77.39 ms median to an incorrect DONE; not completion time              |
| Jev through Rust, hotel                     |        3 |              **3/3** | **3.064 s median**, range 3.034–3.429 s                                |
| Local Laya, Wikipedia diagnostic            |        1 |                  0/1 | 0.303 s to BLOCKED after opening an unrelated link                     |
| Jev through Rust, custom Wikipedia URL/goal |        2 |                  1/2 | First attempt rejected invalid helper JSON; second verified in 4.615 s |
| Local Laya, Flights diagnostic              |        1 |                  0/1 | 0.076 s to an incorrect DONE                                           |
| Jev through Rust, Flights diagnostic        |        1 |                  0/1 | 2.209 s to BLOCKED with the trip-type menu still visible               |

Every final Jev hotel run executed five actions, made six decision requests and
one text request, and used 39 CDP calls within the timed interval. The independent
checker requires the Casa Flora detail URL **and** the applied Design, Free
cancellation, and Lisbon filters. Finding the property without those filters does
not pass. Wikipedia's successful run used the public `--url` / `--goal` interface
and checked the exact final article URL.

Additional local diagnostics tested the multilingual and typed-decisions
checkpoints, a larger 2048/512 context budget, and the upstream prompt. The first
three still chose premature DONE on the hotel task. The upstream-prompt run
repeated the site's home link until its 15-action budget stopped it. A one-button
custom page also produced premature DONE. These failures are retained; they are
not counted as fast successful completions. The evidence points to a decision
quality limitation; it does not establish that every prompt or specialized
checkpoint will fail.

## Comparison with the other repository

The copied [matched-run data](upstream-full-speed-measurement.json) reports:

| Historical Jev runtime       | Task           | Verified | Median task time |
| ---------------------------- | -------------- | -------: | ---------------: |
| Original Python runtime      | Google Flights |      3/3 |          9.450 s |
| Optimized Python runtime     | Google Flights |      3/3 |          7.092 s |
| Optimized Python smoke check | Hotel fixture  |      1/1 |          1.896 s |
| Optimized Python smoke check | Wikipedia      |      1/1 |          2.798 s |

The separate [recorded Flights run](upstream-flights-measurement.json) took
7.073 seconds, with 17 Jev calls at 178 ms median and two text-helper calls at
581 and 346 ms. The hotel and Wikipedia historical smoke times are reported in
the source repository's `docs/performance.md`; they are not matched repeated runs.

Our current Jev hotel median is 1.168 seconds slower than the historical 1.896 s
smoke check. Our successful Wikipedia attempt is 1.817 seconds slower than its
historical smoke check, with another attempt failing. Neither is a controlled
Rust-versus-Python comparison: these were not paired runs on the same host at the
same time. Network latency, provider responses, browser state, and live pages differ.
Current Jev request latency on the replay is also higher than the historical
recording's 178 ms median. Since neither current Flights attempt passed, there is
no valid successful-run comparison against its 7.092-second historical median.

## Timing, reproducibility, and checks

Task timing starts immediately before the first prediction after the initial
observation and stops at a terminal decision or failure. It includes fresh-state
checks, all decision attempts, text calls, stale retries, browser input, and
post-input observations. Model load/warmup, initial navigation, fresh independent
final verification, and the final screenshot are outside that interval.

The root checkpoint loaded in about 0.27–0.30 seconds in the final hotel runs,
plus about 0.12 seconds of warmup. These are warm filesystem-cache measurements,
not download or first-install timings. Configuration, binary hash, source hashes
at build time, all latency samples, and input hash are recorded in the JSON.
Concurrent edits to the shared core's Flash Attention code were outside the
plain-CUDA feature used for this table; the executable stayed fixed during the
comparison.

Validation completed:

- Ten offline Rust tests, including invalid targets, deterministic singleton
  targets, invalid text, independent verification, interrupted mouse release,
  and uncertain SELECT outcomes that must not become retryable stale reads.
- Nine real Chrome checks covering movement, context/value/identity changes,
  disabled and covered targets, native text replacement, and native selection.
- CPU and CUDA release builds, ten tests on CPU/CUDA/Flash Attention, and Clippy
  with warnings denied for the CUDA and Flash Attention examples.
- A verified hotel recording run with per-step screenshots, six timestamped CDP
  screencast frames, an initial screenshot, and no reported recording errors.
  Capture cadence is lower while synchronous inference is running; frame
  timestamps preserve actual time.

Reproduce using the commands in the [example README](../README.md). For useful
local browser performance, the next problem is a checkpoint that chooses correct
browser actions, alongside context/attention efficiency. Replacing network calls
alone does not establish that result.
