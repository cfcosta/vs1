# Browser agent CLI

A Rust implementation of Jev Ultrafast's `examples/run.py`: supply a URL and one
natural-language goal. Hosted Jev chooses an operation and an observed target; a
text model supplies field values only for `TYPE_TEXT`. The default build has no
`vs1` or Candle dependency. Local inference is available with the `local` feature.
There is no Python runtime or binding.

From the repository root:

```sh
cargo build --release -p vs1-browser
export TYPESAFE_API_KEY=... # Jev decision backend
export TEXT_MODEL_API_KEY=... # only needed when the task requires typing
target/release/vs1-browser \
  --url https://en.wikipedia.org/wiki/Main_Page \
  --goal 'Find and open the Wikipedia article about Gödel’s incompleteness theorems.' \
  --output crates/vs1-browser/artifacts/wiki
```

On NixOS, run the build through `direnv exec .` or inside `nix develop`.
For CUDA execution, include `/run/opengl-driver/lib` in `LD_LIBRARY_PATH` if the
driver is not otherwise discoverable. For local CPU inference, build with
`--features local`, then run with `--backend local --device cpu`. CUDA builds use
`--features cuda` and `--backend local --device cuda`; Metal uses `--features metal`
and `--backend local --device metal`. Accelerator features enable `local` automatically.
Even builds with local support default to hosted Jev.

The `cuda` feature includes the packed flash-attention path and needs NVCC and
Cutlass, as described in the root README. The measured full-context replay took
108 ms with it, against 640 ms for the earlier CUDA build without flash attention,
on the test GPU. See the performance report for the separate task-success results.

Connect to Chrome's existing debugging endpoint with `--cdp http://127.0.0.1:9222`
(the default). Chrome must already permit remote debugging. The agent opens and
closes only its own tab, with a 1120×780 viewport. Repeating `--goal` joins the
values into one task; it does not create a site-specific plan.

The terminal prints elapsed time, action count, operation, and target label.
`--max-steps` defaults to 60; at most twice as many decisions are allowed, including
stale retries. A terminal `DONE` is a model claim. Add `--expect-url` and/or repeated
`--expect-text` flags for independent final checks; otherwise the report explicitly
marks verification as unavailable. Failed verification, BLOCKED, exhausted budgets,
and execution failures return a nonzero exit code.

## Models

Hosted Jev is the default (`--backend typesafe`) and requires `TYPESAFE_API_KEY`.
`--model` selects the hosted model ID; it defaults to `TYPESAFE_MODEL` when set,
otherwise `jev-latest`. `--backend jev` aliases `typesafe`, and `--backend laya`
aliases `local` in local-enabled builds. Hosted decisions use `vs1::JevClient`;
the CLI prints its actual HTTP call/retry counters on stderr when the backend
is dropped. Requests and provider answers use the same typed API as local laya. To use local decisions, build with
`--features local` and select `--backend local`. Only local-enabled builds expose
`--device`, `--checkpoint`, `--subfolder`, `--max-len`, and `--head-max-len`.
`--checkpoint` accepts a local directory or Hub
repository (default `convaiinnovations/laya`). `--subfolder multilingual` and
`--subfolder typed-decisions` select other checkpoints. Load and warmup happen
once, before browser task timing. With local inference, only `TYPE_TEXT` uses an
external model. Text-helper settings for either backend:

```sh
export TEXT_MODEL_BASE_URL=https://openrouter.ai/api/v1
export TEXT_MODEL=inception/mercury-2.5
export TEXT_MODEL_REASONING=none
```

These are also the default text settings. Credentials stay in environment
variables and are never written into configuration metadata. `.env.example` lists
the supported variables; this executable does not automatically load `.env`.

`--prompt compact` (default) shortens the instructions and places the goal and
observed controls before page prose. `--prompt upstream` preserves the original
Jev policy shape. `--max-len` and `--head-max-len` override checkpoint token
budgets. Larger contexts change the inference workload and may affect quality;
the trace records the configuration. Do not assume a wire-compatible checkpoint
has Jev's browser skill or that a short failing run is a speedup.

Each cycle submits all applicable operation/target questions in one backend call.
Only the chosen operation's target can execute. For local inference, single-candidate target questions
are resolved deterministically outside the model, because `vs1` requires at least
two options. This does not affect the operation choice.

## Traces and recording

A final screenshot is saved only with `--screenshots` or `--record`.
Every run writes `run-NN/trace.json` and an append-only
`events.jsonl`. Successful execution is flushed before observing its result;
uncertain mutations stop, while pre-input stale rejections can reobserve.
After input, the runner polls read-only snapshots for up to five seconds instead
of relying on background animation frames. Empty snapshots are held back; a click
opening a non-editable popup control waits for visible menu options or a dialog.
A readiness timeout stops the run without replaying input. The CDP transport has
its own timeout. Screenshots and foreground activation are not required.

Visible SVG labels are included as graphic observations, even when the SVG itself
is hidden from accessibility. Hidden ancestors and transparent labels stay
excluded. This does not infer prices or other values from bar heights. Focusable
ARIA regions containing SVG/canvas expose a small `PRESS_KEY` action set
(Left/Right, Home/End, Enter). Execution checks the observed node, visibility,
coverage, and focus before sending native CDP keys; it does not click the graphic.
Keyboard behavior remains application-defined. Snapshots omit controls whose
center is covered, using the same hit test that execution repeats before input.

If input initially appears ineffective, the observer allows another 1.2 seconds
of read-only polling. Progress fingerprints include graphic labels and ignore
node replacement and geometry; input freshness still checks identity and current
geometry separately. The agent temporarily removes actions that had no observed
effect in the same state, or were already executed twice there within the last
20 actions. Both prompt formats list excluded actions and suggest another
control or representation. Six consecutive ineffective non-wait actions stop
the run; the overall action and decision budgets still apply. State changes
allow previously excluded actions again. These heuristics can miss longer cycles
and do not prove that a visually changing graph has been fully inspected.

Text generation is cached across stale retries only if the entire helper context
is identical. Targets are code-owned DOM node references; model output cannot
become a selector, coordinate, or executable script.

`--screenshots` saves each observation. `--record` saves CDP screencast JPEG frames
and their original epoch timestamps in `screencast/frames.json`, plus an initial
screenshot. CDP events are drained during browser calls; inference can reduce
capture cadence. These artifacts retain real timing rather than inventing frames.
Raw traces, screenshots, and field values can contain private page content; the
crate's `artifacts/` directory is ignored. Choose a fresh `--output` each time.

## Verification and comparison

```sh
# No model or paid API: exercise actual browser guards, typing, and selects.
target/release/vs1-browser --check-browser

# Offline policy and verification tests.
cargo test -p vs1-browser

# Repeat a local fixture task, with independent final filter/property checks.
target/release/vs1-browser \
  --backend local --device cuda --task hotel --repeat 3 --output crates/vs1-browser/artifacts/hotel

# Other fixed tasks: wikipedia and flights (historical date: September 20, 2026).
# Same Rust browser runtime, original decision provider and policy:
target/release/vs1-browser \
  --backend typesafe --prompt upstream --task hotel --repeat 3 \
  --output crates/vs1-browser/artifacts/jev-hotel

# Exact captured decision requests, with shape-specific warmup excluded:
target/release/vs1-browser \
  --backend local --device cuda --replay crates/vs1/tests/fixtures/jev/call5_request.json --repeat 10 \
  --output crates/vs1-browser/artifacts/replay
```

The default TypeSafe backend requires `TYPESAFE_API_KEY`; local inference never
uses it. `summary.json` records all runs, errors, success checks, decision and text
latency, browser protocol counts, startup configuration, and timing boundaries.
Task timing starts at the first prediction after the initial observation and ends
at the terminal decision or failure. Initial navigation, model load/warmup, and
fresh final verification are excluded, matching the upstream measurement boundary.
Recording and per-step screenshots are included if enabled; disable both for the
closest comparison to upstream's matched runs.

Historical data is retained in `docs/upstream-*-measurement.json`. Its optimized
Flights median is 7.092 seconds (3/3 verified); the earlier runtime's median is
9.450 seconds. The 1.896-second hotel and 2.798-second Wikipedia results are single
smoke checks. Hardware, platform, network, provider version, and live page changes
prevent those historical results from being a controlled same-machine comparison.
See `docs/performance.md` for this implementation's measurements.

## Scope and attribution

The policy, DOM snapshot, fixture, and browser safeguards derive from Browser
Use's MIT-licensed Jev Ultrafast. See `LICENSE-UPSTREAM` and `UPSTREAM.json` for the
source and hashes; the Rust implementation calls `vs1` through a path dependency.
The root library's dependencies are unchanged.

Like the source MVP, the reader supports common HTML/ARIA controls. It does not
traverse frames or shadow roots, handle uploads or popup tabs, scroll nested
containers, or implement arbitrary keyboard widgets. `SELECT` uses observed native
dropdown options. The code never books a flight as part of the supplied benchmark.

## Nix packages

From the repository root:

```sh
nix build .#vs1-browser
nix run .#vs1-browser -- --help
nix build .#vs1-browser-cuda
```

`vs1-browser-metal` is also available for compatible
hosts. Connect Chrome/Chromium separately as described above; it is not bundled.

## JSON scenarios

Goals, constrained plans, variant setup scripts, and independent verification can
live in a JSON file:

```sh
cargo run --release -p vs1-browser -- \
  --scenario examples/hotel.json --chooser lexical --output artifacts/hotel-json
```

See [the examples](../../examples/README.md) for hotel and reading-room scenarios
and the format. Omit `--chooser lexical` to use the model backend. These scenarios
use explicit plans; they do not replace the free-form `--url`/`--goal` mode.
