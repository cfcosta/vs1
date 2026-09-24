# Local browser benchmark suite

Run every `examples/*-agent.json` whose file source resolves to
`crates/vs1-browser/assets/tasks.html`, in filename order, for one backend.
Each invocation runs all variants and repeats of each case sequentially.
The runner uses Python 3.11+ with no third-party dependencies, following the
`uv` script style in `research/cua-s1/reference.py`.

Chrome must already expose a CDP endpoint (default `http://127.0.0.1:9222`),
and run on the same machine so it can open the local fixture's `file://` URL.
The runner does not start Chrome. Pass `--cdp URL` or set `CDP_URL` for another
endpoint. Each scenario variant/repeat opens a fresh tab.

Set `OPENROUTER_API_KEY` for typing actions, including with local Cua-S1 decisions.
`TEXT_MODEL_API_KEY` overrides it when set. Hosted Jev also needs
`TYPESAFE_API_KEY`. See the [browser CLI documentation](../../crates/vs1-browser/README.md)
for text-model configuration and backend flags.

## Run

From the repository root, build and run hosted Jev:

```sh
direnv exec . cargo build --release -p vs1-browser
export TYPESAFE_API_KEY=...
export OPENROUTER_API_KEY=...
direnv exec . uv run research/browser-suite/run.py run \
  --output artifacts/browser-suite/jev --repeat 3 --max-steps 20 \
  -- --backend jev --policy questions
```

For Cua-S1 on CPU:

```sh
direnv exec . cargo build --release -p vs1-browser --features local
export OPENROUTER_API_KEY=...
direnv exec . uv run research/browser-suite/run.py run \
  --output artifacts/browser-suite/cua-s1 --repeat 3 --max-steps 20 \
  -- --backend cua-s1 --device cpu --policy native
```

Cua-S1 uses the backend's default checkpoint; add
`--checkpoint artifacts/cua-s1` after `--` to use an existing reference checkpoint
directory. On a CUDA machine, build with `--features cuda` and pass
`--device cuda`. Native policy uses text observations only.

`--binary PATH` selects a different `vs1-browser` executable (default:
`target/release/vs1-browser`). Arguments after `--` pass through unchanged to
every case, including `--backend`, `--device`, `--policy`, `--prompt`, and `--cdp`.
The suite owns `--scenario`, `--output`, `--repeat`, and `--max-steps`; put repeat
and step limits before `--`. Their defaults are 1 and 60.

The run directory must not exist. Omit `--output` for a fresh UTC timestamp
directory under `artifacts/browser-suite/`. Each case receives its own fresh
`<run>/<case>/` output directory; `<case>.log` captures stdout and stderr.
`suite.json` records selected scenarios, variant names, repeat count, commands,
and exit codes. Failed cases do not stop the remaining cases.

## Reports

The runner reads each case's `results.json`, whose `runs[].result` contains the
agent summary and the scenario's final pass verdict. It writes aggregate
`results.json` and `results.md` in the run directory and prints per-case counts
and the overall pass rate. Each table row is one case/variant/repeat, with the
original zero-based run number, pass verdict, failure reason, final status,
action/decision counts, median decision latency, and elapsed time in milliseconds.
Timings are copied from the agent summary, excluding model load, setup, and final
verification; they are not whole-process timings.

Failure reasons preserve all applicable categories: `status mismatch`,
`verifier`, `too many actions`, and `error`. An expected `blocked` result can pass,
and zero-action limits are enforced by the scenario. Execution and cleanup errors
count as errors; missing or unreadable results count every missing planned run
as an error, with unknown metrics left blank (`null` in JSON). The JSON also keeps
error details. The denominator includes every planned variant/repeat.

Exit status is 0 only if every planned run passes, 1 for benchmark failures, and
2 for runner argument or filesystem errors. Regenerate reports without Chrome:

```sh
direnv exec . uv run research/browser-suite/run.py summarize artifacts/browser-suite/jev
```

Test the summarizer and runner with synthetic results, without Chrome or models:

```sh
direnv exec . python3 -B -m unittest discover -s research/browser-suite -p 'test_*.py'
```
