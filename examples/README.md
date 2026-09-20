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
Use `--checkpoint /path/to/model` for a pinned local checkpoint. The default is the
local Laya backend; `--backend typesafe` uses the existing hosted backend.
`--chooser lexical` runs without a model or model credentials. `--repeat 3` repeats
every variant. Each invocation needs a fresh output directory.

- [hotel.json](hotel.json): search, submit, select, checkbox, and property navigation;
  includes distractors, reordered controls, pre-applied filters, and completion.
- [reading-room.json](reading-room.json): semantic article selection; includes
  reordered links and an already-completed state. Use `--retrieval none` to retain
  all link candidates when comparing model selection with lexical matching.

These are hand-authored constrained plans, not a benchmark of automatic planning.

## Format

Each JSON document contains:

- `source`: `{"kind":"file","path":"...","query":"..."}` or
  `{"kind":"url","url":"https://..."}`. File paths resolve relative to the JSON
  file, not the working directory. File `query` is optional.
- `goal`: the task description.
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
  the final DOM/URL; `false`, a non-boolean value, or an exception fails the run.

Setup scripts can construct fixtures and distractors; they run before model actions.
Verification scripts should only inspect the resulting page. Run scenario files you
trust: their JavaScript executes directly in the selected source page.

The runner filters observed candidates by kind/role and, by default, by whole-word
overlap with `terms`. `--retrieval none` disables the overlap filter. Singletons
execute deterministically; larger candidate sets go to the selected chooser.
Already-satisfied steps are skipped, and every executed action must satisfy its
postcondition before the next step. There is no model-selected DONE or fallback
that silently corrects a wrong target.

The output stores the original scenario JSON and `results.json` with requests,
responses, observed pages, selected actions, checks, and timings. Tabs are closed
after successful runs and setup/verification failures. Any failed variant gives a
nonzero exit status. Scenarios with invalid schema, unknown setup scripts, or
missing local fixtures fail before a browser tab is opened.

On NixOS, use `direnv exec . cargo ...` or `nix run .#vs1-browser -- ...`.
The JSON files reference a fixture in this checkout; invoke them from the checkout
or preserve those relative paths when copying the examples.
