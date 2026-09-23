# Cua-S1 text reference

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
