# JSON browser scenarios

Run examples through the Rust CLI:

```sh
cargo run --release -p vs1-browser -- \
  --scenario examples/hotel.json --output artifacts/hotel

cargo run --release -p vs1-browser -- \
  --scenario examples/reading-room.json --retrieval none \
  --output artifacts/reading-room
```

Chrome/Chromium must expose CDP at `http://127.0.0.1:9222` (override with `--cdp`).
The default is hosted Jev (`--backend typesafe`), requiring `TYPESAFE_API_KEY`.
For local Laya, build with `--features local` and pass `--backend local`; use
`--checkpoint /path/to/model` for a pinned checkpoint. The default browser build
does not include the `vs1` library or local-model dependencies.
`--chooser lexical` runs constrained scenarios without a model or model credentials. `--repeat 3` repeats
every variant. Each invocation needs a fresh output directory.

- [hotel.json](hotel.json): search, submit, select, checkbox, and property navigation;
  includes distractors, reordered controls, pre-applied filters, and completion.
- [reading-room.json](reading-room.json): semantic article selection; includes
  reordered links and an already-completed state. Use `--retrieval none` to retain
  all link candidates when comparing model selection with lexical matching.

These two files are hand-authored constrained plans, not a benchmark of automatic planning.

## Ultrafast task coverage

| Upstream task / entry point                             | JSON scenario                                      | Mode  |
| ------------------------------------------------------- | -------------------------------------------------- | ----- |
| Hotel: demo travel tab, `scripts/smoke.py`              | [hotel-agent.json](hotel-agent.json)               | Agent |
| Reading room: demo research tab                         | [reading-room-agent.json](reading-room-agent.json) | Agent |
| Wikipedia: README invocation of `examples/run.py`       | [wikipedia.json](wikipedia.json)                   | Agent |
| Google Flights: `examples/flights.py`, demo flights tab | [flights.json](flights.json)                       | Agent |

Agent mode uses the existing browser policy to choose actions, DONE, or BLOCKED
from the observed page. By default, success requires DONE **and** the independent
JSON verifier. Scenarios can instead expect BLOCKED and can limit executed actions.
The generic `examples/run.py` equivalent is an agent JSON with your own source,
goal, and verifier. `scripts/measure_flights.py` and `scripts/record_flights.py`
reuse the flight task; use repetitions and recording flags:

```sh
cargo run --release -p vs1-browser -- \
  --scenario examples/flights.json --repeat 3 --record --screenshots \
  --output artifacts/flights
cargo run --release -p vs1-browser -- \
  --scenario examples/wikipedia.json --output artifacts/wikipedia
```

`--max-steps` bounds each run. Agent mode supports the usual backend, checkpoint,
prompt, recording, and screenshot flags. `--chooser lexical` is unavailable;
`--retrieval` applies only to constrained plans. TYPE_TEXT uses the existing text
helper (`TEXT_MODEL_API_KEY`, optionally `TEXT_MODEL_BASE_URL` and `TEXT_MODEL`).
See the root README for backend configuration.

The flight verifier ports upstream's route, one-way, date/year, and visible-result
checks. Origin/destination checks accept Google's labels with appended airport
text while still requiring the exact field values. It retains **September 20, 2026** for comparability; update the goal and all
verification date strings together when choosing a future date. Like upstream,
it does not independently check passenger count or cabin. Website changes, consent
screens, model quality, and missing text-helper credentials can fail these live
runs. Adding an example does not establish that Laya can complete it.

Upstream GIF/video renderers consume recordings and are not additional browser
tasks; this CLI saves recordings and screenshots but does not port those renderers.
Browser guard/unit tests remain tests, not example scenarios.

## Format

Each JSON document contains:

- `mode`: `constrained` (default) or `agent`. Agent scenarios omit `steps` and
  `completion`; constrained scenarios require both.
- `source`: `{"kind":"file","path":"...","query":"..."}` or
  `{"kind":"url","url":"https://..."}`. File paths resolve relative to the JSON
  file, not the working directory. File `query` is optional.
- `goal`: the task description.
- `expect_status`: agent mode only, `"done"` (default) or `"blocked"`. The final
  status must match, and the independent verifier must still pass.
- `max_actions`: agent mode only, an optional non-negative integer. The run fails
  if it executes more browser actions than this limit. This checks the result;
  `--max-steps` remains the execution budget. Model decisions, including DONE and
  BLOCKED, do not count as actions. Use `"expect_status": "done", "max_actions": 0`
  for a goal that is already satisfied. Omit either field to use its default;
  explicit `null` values and both fields in constrained mode are rejected.
- `steps`: ordered instructions with action `kind` (`click`, `fill`, `select`),
  `role`, retrieval `terms`, optional literal fill `text`, and observed `after`
  conditions. Conditions support `text`, `title`, `url_suffix`, `value`, and
  `checked`; see hotel.json for their fields. All listed conditions must hold.
- `completion`: final observed conditions. An empty list is rejected.
- `scripts`: named arrays of JavaScript lines, joined by newlines and evaluated
  in the page. These are scenario-author code, not model-generated actions.
- `variants`: named cases with optional `setup` lists referencing those scripts.
  Every variant/repetition starts in a fresh browser tab.
- `verify`: JavaScript lines evaluating to a boolean. This independently checks
  the final DOM/URL and has access to the fresh observed snapshot as `page`; `false`, a non-boolean value, or an exception fails the run.

Setup scripts can construct fixtures and distractors; they run before model actions.
Verification scripts should only inspect the resulting page. Run scenario files you
trust: their JavaScript executes directly in the selected source page.

In constrained mode, the runner filters observed candidates by kind/role and, by default, by whole-word
overlap with `terms`. `--retrieval none` disables the overlap filter. Singletons
execute deterministically; larger candidate sets go to the selected chooser.
Already-satisfied steps are skipped, and every executed action must satisfy its
postcondition before the next step. There is no model-selected DONE or fallback
that silently corrects a wrong target.

The output stores the original scenario JSON and `results.json` with requests,
responses, observed pages, selected actions, checks, and timings. Agent runs store
those details in `variant-NN-run-NN/trace.json` plus `events.jsonl` and captures. Tabs are closed
after successful runs and setup/verification failures. Any failed variant gives a
nonzero exit status. Scenarios with invalid schema, unknown setup scripts, or
missing local fixtures fail before a browser tab is opened.

Each agent run's result in `results.json` includes `expectation_failures`, an array
containing all applicable reasons: `status_mismatch`, `verifier_failure`, and
`too_many_actions`. An empty array means those expectations passed; run or cleanup
errors still fail the run and remain in `error` or `cleanup_error`.

On NixOS, use `direnv exec . cargo ...` or `nix run .#vs1-browser -- ...`.
The JSON files reference a fixture in this checkout; invoke them from the checkout
or preserve those relative paths when copying the examples.
