# Cua-S1 text reference

See [the 2026-09-23 benchmark comparison](benchmarks.md) for email, browser replay
and live-task results against Jev, Laya and OpenJev.

## Rust option tournaments

`CuaS1::score_options` preserves option order and uses one forward pass for
1–26 options. Larger sets are split in their original order into the fewest
balanced groups of at most 26, with smaller groups first. Each group's winner
advances, and finalists are scored recursively until one final round fits.
Every prompt uses the same app, task family, state and goal, with fresh letters
starting at A. State truncation applies separately to each prompt.

Each option's `probability` is its group probability multiplied by its
finalist's recursively computed probability. The full distribution sums to one.
`is_selected` identifies the final-round winner, which can differ from the
highest hierarchical probability; ties within each round favor the first option.
`letter`, `logit` and `dropped_state_tokens` describe the option's first-round
group. `forward_passes` is the total number of model calls for the decision,
repeated on each prediction: 1 for 26 options, 3 for 27, and 30 for 700.

`system_one` accepts choice and score questions with at least two candidates.
Choice uses the tournament winner; score retains the expected level over the
hierarchical probabilities. Token usage includes every round, and dropped state
tokens report the maximum across rounds for each question. `request_fits`
checks first-round prompts; finalist prompts enforce `max_len` when scored.
The low-level `CuaS1Input::encode` still accepts only 1–26 options per pass.

## Large-page scoring benchmark

`wikipedia-56-options.json` is a captured native browser decision with 56 options.
The ignored test loads the pinned local checkpoints on `cuda:0` in BF16,
runs two warmup calls, then times ten `score_options` calls. Device synchronization
surrounds each call; timings include prompt construction and all tournament rounds,
but exclude checkpoint loading and JSON serialization. Timed predictions must
exactly match the final warmup prediction.

```sh
VS1_CUA_S1_BENCH_OUTPUT=artifacts/cua-s1/wikipedia-56-bench.json \
  direnv exec . cargo test --release -p vs1 --features cuda --test cua_s1 \
  measures_wikipedia_scoring_on_cuda -- --ignored --exact --nocapture
```

Set `VS1_CUA_S1_BENCH_ITERATIONS` to a positive number to change the timed call
count. Set `VS1_CUA_S1_BENCH_OPTIONS=22` to score only the first 22 options in one
pass, keeping the same tree and goal:

```sh
VS1_CUA_S1_BENCH_OPTIONS=22 VS1_CUA_S1_BENCH_ITERATIONS=10 \
  VS1_CUA_S1_BENCH_OUTPUT=artifacts/cua-s1/wikipedia-22-bench.json \
  direnv exec . cargo test --release -p vs1 --features cuda --test cua_s1 \
  measures_wikipedia_scoring_on_cuda -- --ignored --exact --nocapture
```

JSON is printed and optionally written to `VS1_CUA_S1_BENCH_OUTPUT`. It records
the median and latencies in call order (milliseconds), option count, forward-pass
count per decision, selected option with its probability, and the five options
with highest probabilities. The selected option uses `is_selected`, which need
not be the first of those five in a tournament. Predictions retain their full
option fields, first-round letters and logits, and dropped-state-token counts.

## Reference prompts

Regenerate from the repository root:

```sh
direnv exec . uv run research/cua-s1/reference.py prompts
direnv exec . nix fmt
```

On NixOS, supply a compatible Nix Python and the C++ runtime for the wheels:

```sh
direnv exec . nix shell --inputs-from . nixpkgs#python313 --command bash -c '
  export LD_LIBRARY_PATH="$(dirname "$(c++ -print-file-name=libstdc++.so.6)")"
  export TORCHINDUCTOR_CACHE_DIR="$PWD/artifacts/cua-s1/torchinductor"
  uv run --python python3.13 research/cua-s1/reference.py prompts
'
direnv exec . nix fmt
```

The PEP 723 script pins `cua-s1` from the `trycua/cua` main commit
`a5f18829df026d7b9ef80c339194b44b1c61f856`, Transformers 5.17.0, and the
`Qwen/Qwen3.5-4B` tokenizer revision
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`. The `prompts` command downloads only
tokenizer files. The script installs CPU PyTorch 2.13.0 and PEFT 0.18.1, plus CPU
torchvision 0.28.0 and Pillow 12.3.0: the upstream loader constructs an image
processor even in text mode, so these dependencies are needed to load it unchanged.

`cases.json` contains six text decisions with two to six options, optional goals,
fill actions with entity IDs, and Portuguese text. Each entry has a fixture `name`
plus the text arguments accepted by `FourBModel.forward`; `options` contains
`Option` fields in their intended order.

The `prompts` command calls the real `assign_letters`, `build_prompt`, and
`FourBModel._letter_token_ids`. The model constructor is lazy: only its tokenizer
is attached, and neither `load` nor `forward` is called. Chat templating and
tokenization match the text branch of
[`FourBModel.forward`](https://github.com/trycua/cua/blob/a5f18829df026d7b9ef80c339194b44b1c61f856/libs/cua-s1/python/src/cua_s1/four_b.py),
including the tokenizer's default generation prompt and special-token behavior.
In particular, the assistant suffix ends with `<think>\n`; the fixture preserves
that default exactly as upstream does.

`prompts.json` records the installed Cua-S1 commit, Transformers version, tokenizer
revision, and each case's chat text, input token IDs, letters, and letter token IDs.
Use `prompts --cases PATH --output PATH` for alternate inputs or output locations.

## Model run

Run the real `FourBModel.load` and `FourBModel.forward` on CPU in float32:

```sh
direnv exec . nix shell --inputs-from . nixpkgs#python313 --command bash -c '
  export LD_LIBRARY_PATH="$(dirname "$(c++ -print-file-name=libstdc++.so.6)")"
  export TORCHINDUCTOR_CACHE_DIR="$PWD/artifacts/cua-s1/torchinductor"
  uv run --python python3.13 research/cua-s1/reference.py run --device cpu --dtype float32
'
direnv exec . nix fmt
```

`run` defaults to `--device cpu --dtype float32 --threads 4`. It downloads pinned
snapshots under `artifacts/cua-s1/` and passes their local paths to the upstream
loader, which has no revision argument:

| Repository             | Revision                                   | Files used                                                   |
| ---------------------- | ------------------------------------------ | ------------------------------------------------------------ |
| `Qwen/Qwen3.5-4B`      | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` | Base weights, config, tokenizer                              |
| `cua-ai/cua-s1-4b-0.2` | `16818868b0cc7813808aae4e87b417657046ab79` | `text/adapter_config.json`, `text/adapter_model.safetensors` |

The upstream text loader selects `AutoModelForCausalLM`, attaches the unmerged
PEFT adapter (r=16, alpha=32), and uses evaluation mode. Missing adapter keys are
treated as errors. No vision input is used. Each case calls `FourBModel.forward`
once, with no generation or shared cache. Hooks assert the actual input IDs equal
`prompts.json`, count that single model call, and capture its final-position
option-letter logits. The stored probabilities are the upstream return values;
the script asserts exact equality with a float32 softmax over those captured
letter logits alone.

`probabilities-f32.json` records both repository revisions, the Cua-S1 commit,
installed package versions, device, dtype, thread count, attention implementation,
timings, and each option's original fields, letter, token ID, raw logit, and
probability. `load_seconds` includes snapshot downloads and model loading;
`forward_seconds` includes prompt construction and hook capture/validation;
`total_seconds` also includes writing the layer dump. These are reference-run
timings, not a benchmark. Use `--cases`, `--prompts`, `--output`, and `--layers`
to select alternate matching fixtures and output paths.

The recorded run on an AMD Ryzen 9 7950X3D took 37.53 seconds with cached weights:
2.66 seconds to load, and 5.23–6.59 seconds per case. Full attention used PyTorch
SDPA; linear attention used Transformers' reference PyTorch convolution and
gated-delta-rule implementations. All six input-ID and probability assertions
passed. The saved dump was also checked for all 111 documented shapes/dtypes,
finite values, exact input IDs, and exact continuity between decoder layers.

The highest-probability letters are A, A, C, D, F, and B in case order. In
`search_portuguese`, the model favors clicking `Pesquisar` (69.3724%) over filling
the empty field (30.6127%); the fixture records that behavior without correction.

`probabilities-bf16.json` records the same six cases on CPU with
`--dtype bfloat16`, matching `FourBModel`'s default dtype, with the same pinned
revisions, package versions, prompts, and four threads. To regenerate it, use the
shell wrapper above and replace the `run` arguments with `run --device cpu --dtype
bfloat16 --output research/cua-s1/probabilities-bf16.json --layers
artifacts/cua-s1/layers-bf16.safetensors`; both F32 files remain unchanged. The
separate, gitignored BF16 layer dump contains the same 111 keys, stored as float32
activations and int64 input IDs. All six input-ID, single-forward-pass, and exact
float32 option-logit softmax assertions passed. BF16 preserves all six top
options; the largest absolute probability difference is 0.017742388 (1.774239
percentage points), for option B in `reset_password`. The table lists probabilities
in letter order (A onward), with the maximum absolute difference across all
options in each case. These CPU fixtures do not establish CUDA parity.

| Case                    | F32 probabilities (A onward)                                                 | BF16 probabilities (A onward)                                                | Top option, both | Max absolute difference |
| ----------------------- | ---------------------------------------------------------------------------- | ---------------------------------------------------------------------------- | ---------------- | ----------------------- |
| `save_note`             | 0.999515653, 0.000484310                                                     | 0.999447167, 0.000552779                                                     | A                | 0.000068486             |
| `reset_password`        | 0.909905910, 0.088813774, 0.001280418                                        | 0.892183781, 0.106556162, 0.001260076                                        | A                | 0.017742388             |
| `decline_invitation`    | 0.000101538, 0.000013200, 0.999640584, 0.000244682                           | 0.000102272, 0.000014734, 0.999621749, 0.000261160                           | C                | 0.000018835             |
| `download_invoice`      | 0.000089708, 0.000347091, 0.000092337, 0.978663445, 0.020807469              | 0.000083052, 0.000328478, 0.000088409, 0.979177892, 0.020322189              | D                | 0.000514448             |
| `notification_settings` | 0.000153660, 0.000056130, 0.000316314, 0.000261375, 0.000090138, 0.999122441 | 0.000148737, 0.000045362, 0.000295799, 0.000261042, 0.000079613, 0.999169469 | F                | 0.000047028             |
| `search_portuguese`     | 0.306126863, 0.693724453, 0.000148637                                        | 0.294169992, 0.705677152, 0.000152843                                        | B                | 0.011956871             |

## Layer dump

Only the first case (`save_note`, 171 tokens) gets intermediate hooks. The file
`artifacts/cua-s1/layers-f32.safetensors` is gitignored. Its metadata records the
case name, base revision, adapter revision, and computation dtype. All activations
are detached, copied, contiguous float32 tensors; `input_ids` retains int64 token
IDs. Hooks preserve native tensor layouts without squeezing the batch dimension.

There are 111 keys. In the table below, `S=171`, batch size is 1, and `i` ranges
from 0 through 31. Layers 3, 7, 11, 15, 19, 23, 27, and 31 use full attention;
the other layers use linear attention.

| Key                                | Shape           | Meaning                                                                                                          |
| ---------------------------------- | --------------- | ---------------------------------------------------------------------------------------------------------------- |
| `input_ids`                        | `[1, S]`        | Actual token IDs passed to the model, identical to the first prompt fixture                                      |
| `embeddings.output`                | `[1, S, 2560]`  | Token embedding output                                                                                           |
| `layers.i.input`                   | `[1, S, 2560]`  | Decoder layer input, before its input RMSNorm                                                                    |
| `layers.i.mixer_output`            | `[1, S, 2560]`  | `linear_attn` output or first `self_attn` return value, after output projection and before the residual add      |
| `layers.i.output`                  | `[1, S, 2560]`  | Complete decoder layer output, after both residual adds                                                          |
| `layers.0.linear_attn.in_proj_qkv` | `[1, S, 8192]`  | Concatenated Q/K/V projection, before convolution and reshaping                                                  |
| `layers.0.linear_attn.in_proj_z`   | `[1, S, 4096]`  | Raw gating projection, before reshaping                                                                          |
| `layers.0.linear_attn.in_proj_a`   | `[1, S, 32]`    | Raw decay projection, before softplus and scaling                                                                |
| `layers.0.linear_attn.in_proj_b`   | `[1, S, 32]`    | Raw beta projection, before sigmoid                                                                              |
| `layers.0.linear_attn.norm.input`  | `[S * 32, 128]` | Gated RMSNorm's first input: flattened delta-rule attention output                                               |
| `layers.0.linear_attn.norm.gate`   | `[S * 32, 128]` | Gated RMSNorm's second input: flattened raw `z`, before SiLU                                                     |
| `layers.0.linear_attn.norm.output` | `[S * 32, 128]` | Gated RMSNorm output, before reshaping and output projection                                                     |
| `layers.0.linear_attn.out_proj`    | `[1, S, 2560]`  | Linear attention output projection; equals `layers.0.mixer_output`                                               |
| `layers.3.self_attn.q_proj`        | `[1, S, 8192]`  | Complete LoRA-adapted projection, including interleaved query and gate blocks, before splitting, Q norm, or RoPE |
| `layers.3.self_attn.k_proj`        | `[1, S, 1024]`  | Complete LoRA-adapted key projection, before K norm or RoPE                                                      |
| `layers.3.self_attn.v_proj`        | `[1, S, 1024]`  | Complete LoRA-adapted value projection, before head reshaping                                                    |
| `layers.3.self_attn.o_proj.input`  | `[1, S, 4096]`  | Flattened attention output after sigmoid gating, entering the LoRA-adapted output projection                     |
| `norm.last`                        | `[1, 2560]`     | Final decoder RMSNorm output at the last sequence position, before the LM head                                   |
