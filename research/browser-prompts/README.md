# Laya browser prompt experiments

Fixed-state experiments on the pinned English Laya checkpoint
`c5d78730f3493e4fe16d61507ef4b78eef7318cf`. These are diagnostic probes, not
closed-loop browser success rates or independent generalization benchmarks.
Production prompts and model code are unchanged. The [machine-readable results](results.json)
include per-case decisions, probabilities, atomic probes, option-order controls,
and hashes of the local request/response artifacts.

## Design

The Rust [capture example](../../crates/vs1-browser/examples/prompt_probe.rs)
uses the existing CDP observation and policy code. It creates its own browser tab,
sets up eight hotel observations and four form observations, captures each with
the real snapshot extractor, then closes the tab. Setup values and gold labels
are stored separately from model inputs. No text helper or remote model is used.

Hotel states: initial, destination typed but not submitted, destination searched,
Design selected, all filters applied, Casa Flora opened correctly, Casa Flora
opened with wrong filters, and a different property opened. Form states: empty,
name entered, terms checked but not submitted, and confirmation. The wrong-property
state is a deliberately constructed counterexample, not a recorded trajectory.

There are ten unfinished and two completed observations. Initial/midway actions
can have several valid orders. Therefore, a plausible operation such as CLICK
is **not** counted as task success or correct target selection. The main metrics
are premature DONE and correctly recognizing completed states. Separate noul and
binary-choice questions probe completion; these do not gate the operation output.
The gold `valid_operations` arrays are permissive annotations, not target labels.

Eight variants use the same observation within each case:

| Variant           | Change                                                                    |
| ----------------- | ------------------------------------------------------------------------- |
| upstream          | Existing structured state and original instructions                       |
| compact           | Existing compact production prompt                                        |
| short_instruction | Compact state/options; shorter operation instruction                      |
| short_labels      | Also shorten operation descriptions                                       |
| observations      | Short labels; page text before full observed action objects               |
| next_requirement  | Observations representation; explicitly compare each goal requirement     |
| facts             | Current values before page text; omit option lists/geometry; short labels |
| flat_actions      | Facts representation; choose an individual observed action or DONE        |

The `facts` name means observed control values, not an oracle-computed list of
satisfied requirements. Completion questions receive the same state as the
operation question. Laya encodes questions separately, so this is not shared-prefix
inference. BF16 batched results can vary slightly with padding/batch composition.

## Default-budget results

CUDA BF16, batch size 4, max_len 512, head_max_len 192:

| Variant           | Premature DONE / 10 unfinished | Correct DONE / 2 completed |
| ----------------- | ------------------------------ | -------------------------- |
| upstream          | 1                              | 0                          |
| compact           | 8                              | 2                          |
| short_instruction | 8                              | 2                          |
| short_labels      | 5                              | 0                          |
| observations      | 0                              | 0                          |
| next_requirement  | 0                              | 0                          |
| facts             | 4                              | 1                          |
| flat_actions      | 0                              | 0                          |

Avoiding DONE everywhere is not an improvement: several variants merely continue
acting on completed pages. Flat action selection repeatedly chose the brand/home
link, including on completed hotel pages. None is a demonstrated replacement for
the production prompt.

## Numerical control

The full 96-request / 288-question matrix was also run on CPU/F32. Every token
sequence matched the CUDA input exactly. CPU results have the same DONE counts as
the table except `facts`: **3/10** premature DONE instead of 4/10. There is still
no reliable variant. A second CUDA/F32 run agrees with every CPU operation choice;
its maximum absolute probability difference is 0.006656.

CUDA/BF16 differs from CPU/F32 on two of 96 operation choices (`searched/flat_actions`
and `design/facts`). Maximum probability difference is 0.15957, in the flat-action
case; the largest observed noul difference is 0.07910. These are material for
threshold-based completion decisions. Do not attribute small probability gains or
borderline classifications to wording alone. This test does not establish which
numerical implementation detail causes the drift. Context-budget and atomic
results below are BF16 diagnostics; the default-budget conclusion is also checked
in CPU/F32.

## What the token inspection established

Decoded token IDs from the actual inference sequence show:

- The upstream initial-state operation instruction stops partway through the
  autocomplete rule. The later completion requirements are missing.
- Its initial-state page is cut off amid navigation elements, before the relevant
  destination/category/checkbox controls.
- Compact input includes the control values, but repeats the goal and spends
  substantial space on available dropdown options and decorative page text.
- Shortening alone does not fix the model: `short_instruction` has no operation
  sequence at the 512-token limit, yet produces eight premature DONE decisions.
- The `facts` form-empty operation sequence is only 141 tokens, but chooses DONE.
  It also selects TYPE_TEXT on the fully filtered hotel page, where that operation
  is unnecessary.

These findings support better observation packing, but not truncation as the sole
cause of the browser failures.

## Context-budget controls

Same requests/weights, CUDA BF16; only a local copy of the checkpoint configuration
changes. These are diagnostics outside the checkpoint's default budget, not a
claim that it was trained for this browser context length.

| Variant           | 1024/192: premature / correct DONE | 2048/512: premature / correct DONE |
| ----------------- | ---------------------------------- | ---------------------------------- |
| upstream          | 4 / 0                              | 2 / 1                              |
| compact           | 8 / 2                              | 8 / 2                              |
| short_instruction | 8 / 2                              | 8 / 2                              |
| short_labels      | 5 / 0                              | 5 / 0                              |
| observations      | 0 / 0                              | 0 / 0                              |
| next_requirement  | 2 / 0                              | 0 / 0                              |
| facts             | 4 / 1                              | 3 / 1                              |
| flat_actions      | 0 / 0                              | 0 / 0                              |

Denominators remain ten unfinished and two completed states. No operation sequence
hits the 2048-token limit. Increasing the sequence and header budgets does not
produce a reliable completion policy.

## Requirement decomposition and option order

[atomic.py](atomic.py) asks four separate noul questions about destination/category/
cancellation/opened property, or name/terms/submission/confirmation. It compares
the existing compact state with plain-language control values such as `unchecked`,
removing the potentially confusing checkbox `value="on"` representation.

- Compact: **27/48** individual facts correct at threshold 0.5.
- Natural control values: **28/48** individual facts correct.
- Requiring all four predictions to be true classifies completion correctly on
  **9/12** and **7/12** states respectively. Always predicting unfinished would
  score 10/12, so raw accuracy alone is misleading on this imbalanced set.
- Even an empty form receives high probability that Terms are accepted and that
  submission has happened. These are grounding errors, not just premature-DONE
  label wording.

Reversing only operation/action option order changes selections in 1/12 upstream,
2/12 compact, 0/12 short-instruction, 3/12 short-label, 0/12 observations,
1/12 next-requirement, 3/12 facts, and 2/12 flat-action cases. The information and
option identities stay unchanged. No order was selected as a preferred prompt.

## Recommendation

Do not promote a prompt variant from this experiment. Retain this probe suite for
future changes. A next implementation should preserve selected/checked/applied
state explicitly, remove decorative content and duplicate goal text, and measure
both target selection and completion on unseen fixtures. Completion verification
can protect execution, but its success must be reported separately from the model.

The small sample does not establish that every possible prompt fails, nor that
fine-tuning will succeed. It does show that shorter wording, explicit requirements,
larger context, flat choices, and independent completion questions are insufficient
in these tested forms.

## Reproduce

Chrome/Chromium must expose CDP at `127.0.0.1:9222`. From the workspace root:

```sh
cargo run --release -p vs1-browser --example prompt_probe -- artifacts/prompt-probe
cargo build --release -p vs1 --features cuda
# CHECKPOINT is the local pinned snapshot directory described above.
LD_LIBRARY_PATH=/run/opengl-driver/lib target/release/vs1 \
  --device cuda --model "$CHECKPOINT" --batch-size 4 --dump-ids \
  artifacts/prompt-probe/requests.json > artifacts/prompt-probe/responses.json
python research/browser-prompts/analyze.py artifacts/prompt-probe
python research/browser-prompts/atomic.py \
  artifacts/prompt-probe artifacts/prompt-probe-atomic
# Run vs1 on the generated atomic requests with the same flags.
```

Use `direnv exec .` on NixOS when the toolchain is not on PATH. CPU/F32 inference
uses `--device cpu --dtype f32`. To inspect tokens, decode `ids.<question>.ids`
with the pinned checkpoint tokenizer and `skip_special_tokens=False`.

For context controls, copy `rl_agent_config.json` into a new artifact model
directory and set `(max_len, head_max_len)` to `(1024,192)` or `(2048,512)`;
symlink the original `encoder/`, `tokenizer/`, and `model.safetensors` there.
Do not modify the cached checkpoint. Run the same requests with `--model` pointing
to that directory; pass the matching `--max-len` to `analyze.py`.
For option-order controls, reverse the insertion order of
`request.questions.operation.criteria`, preserving every key/value pair.

Local run directories: `artifacts/prompt-probe-v2/`,
`artifacts/prompt-probe-atomic/`, `artifacts/prompt-probe-context-1024/`,
`artifacts/prompt-probe-context-2048/`, and `artifacts/prompt-probe-permuted/`.
They contain requests, labels, responses/token IDs, and inference logs; snapshots
are in the v2 directory. These local artifacts are ignored by version control.
The initial v1 pass covered hotel states only; the table reports the expanded v2
matrix. Model inference timing is not browser-task latency.

Captured file URLs include the output directory. For exact token reproduction, replay
the archived request JSON and check its hash; recapturing into a differently named
directory can change the upstream prompt tokens.
