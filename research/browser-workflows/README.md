# Constrained browser workflow experiment

The hotel task **works with a hand-authored declarative plan, candidate retrieval,
and independent completion checks**. This is a prototype, not automatic planning
for arbitrary natural-language goals. The production browser policy is unchanged.

## Observed results

Pinned English Laya checkpoint `c5d78730f3493e4fe16d61507ef4b78eef7318cf`, local
CPU/F32 inference. Five scenarios, each repeated three times:

| Scenario                                       | Laya, role/kind candidates | Laya, plus term-overlap retrieval | Lexical baseline, role/kind candidates |
| ---------------------------------------------- | -------------------------- | --------------------------------- | -------------------------------------- |
| Original hotel page                            | 0/3                        | 3/3                               | 3/3                                    |
| Distractor controls and another matching hotel | 3/3                        | 3/3                               | 3/3                                    |
| Reordered controls with those distractors      | 3/3                        | 3/3                               | 3/3                                    |
| Search and filters already satisfied           | 0/3                        | 3/3                               | 3/3                                    |
| Already complete                               | 3/3                        | 3/3                               | 3/3                                    |
| Total                                          | 9/15                       | 15/15                             | 15/15                                  |

Excluding the already-complete scenario, success is **6/12**, **12/12**, and
**12/12** respectively. Repeats check consistency; they are not independent tasks
or evidence of cross-website generalization.

Without retrieval, Laya repeatedly selects **Find stays** when asked to **view Casa
Flora**. The controller observes that the property did not open and stops with an
error. It does not retry with an oracle-corrected target.

With retrieval, labels must share at least one whole, case-insensitive word with
the current step's planner-supplied search terms. This excludes Find stays during
property opening. It still leaves both **View Casa Flora** and **View Casa Azul**
in the distractor cases, and all three category choices during selection. Laya
makes **15 non-singleton decisions**, all with verified postconditions. Singleton
candidate sets are executed deterministically and never attributed to the model.
The unfiltered Laya arm makes 42 model calls, with six postcondition failures.

The lexical baseline matches terms against candidate labels, choosing the earliest
candidate on ties. It also passes all scenarios without model inference. Thus the
experiment supports workflow constraints and independent verification; it does
**not** establish incremental benefit from Laya on this task.

See [results.json](results.json) for scenario summaries, selected labels, candidate
counts, source hashes, and timing. Raw requests/responses and before/after page
observations are in the ignored `artifacts/constrained-hotel-*/results.json` files.
The original unconstrained agent is not rerun here; comparisons above share the
same new plan and executor.

## What is reusable

The shared Rust [runner](../../crates/vs1-browser/src/scenario.rs) consumes
[hotel.json](../../examples/hotel.json), whose steps define:

- An instruction and permitted action kind/role.
- An explicit value for filling a field, when applicable.
- Search terms for optional retrieval and the lexical baseline.
- Observable postconditions, independent of the model's response.
- A final conjunction of completion checks.

The executor skips already-satisfied steps, selects only observed targets, executes
through the existing browser freshness/visibility guards, and verifies the effect
before advancing. Missing evidence is not satisfaction. A model answer cannot
claim DONE: completion is code-verified. Unknown candidates and invalid probability
distributions are rejected. A failed effect stops the run without guessing a
correction. No general-purpose recovery or dynamic replanning is implemented.

The hotel JSON scenario uses the owned fixture and a DOM/URL verification script
equivalent to the original independent verifier. Fixture setup adds unrelated search/button/checkbox controls
and Casa Azul; it does not pick model answers. Reordering affects initial DOM
controls; the fixture may redraw result cards in its normal order after search.

## What the plan supplies

The plan is an essential part of the result. It supplies operation sequencing,
control roles, literal field text, retrieval terms, and task-specific postconditions.
The executor does not derive these from free-form language. The test does not
exercise the text-helper LLM: Lisbon is explicitly supplied in the plan.

Supporting arbitrary tasks still needs a planner, grounded and durable completion
predicates, candidate retrieval across changing layouts, and recovery from missing
controls. Exact labels and visible-text predicates in this sample are not universal
browser semantics. Once a step passes, the prototype advances; later disappearance
of its controls does not invalidate that ledger entry. The final independent check
is therefore essential to detect a later loss of the requested outcome.

## Reproduce

Start Chrome/Chromium with CDP on port 9222. From the repository root:

```sh
cargo run --release -p vs1-browser -- \
  --scenario examples/hotel.json \
  --checkpoint "$CHECKPOINT" \
  --retrieval overlap \
  --output artifacts/constrained-hotel-new
```

`CHECKPOINT` is the local pinned snapshot directory above. On NixOS, prefix Cargo
commands with `direnv exec .`. Use `--retrieval none` for the role/kind-only arm;
use `--chooser lexical` for the no-model baseline. Use `--repeat 3` to match these historical runs.
Each invocation requires a fresh output directory. Results are saved even on
failure; the command exits nonzero if any scenario fails independent verification. The runner creates and closes
its own browser tabs. It does not book anything, call a remote model, or alter the
cached checkpoint.

Checks: Rust example tests cover absent observations, checkbox `value="on"` versus
actual checked state, and retrieval retaining multiple plausible candidates. The
live runs additionally exercise input, submit, select, checkbox, property opening,
postcondition failure, and skipping already-completed work.

The results above predate the JSON-scenario migration. Their recorded plan hash
refers to the original plan in commit `0d6f8c572a2d`; the current JSON additionally
contains source, variants, setup scripts, and verification. See the
[scenario format](../../examples/README.md) for adding other tasks.
