# Cua-S1 text prompt fixtures

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
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`. It installs CPU PyTorch for
`return_tensors="pt"`; it downloads only tokenizer files, never model or LoRA
weights.

`cases.json` contains six text decisions with two to six options, optional goals,
fill actions with entity IDs, and Portuguese text. Each entry has a fixture `name`
plus the text arguments accepted by `FourBModel.forward`; `options` contains
`Option` fields in their intended order.

The script calls the real `assign_letters`, `build_prompt`, and
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
