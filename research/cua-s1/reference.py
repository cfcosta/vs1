# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = [
#   "cua-s1 @ git+https://github.com/trycua/cua.git@a5f18829df026d7b9ef80c339194b44b1c61f856#subdirectory=libs/cua-s1/python",
#   "transformers==5.17.0",
#   "torch==2.13.0",
#   "peft==0.18.1",
#   "torchvision==0.28.0",
#   "pillow==12.3.0",
# ]
# [tool.uv.sources]
# torch = { index = "pytorch-cpu" }
# torchvision = { index = "pytorch-cpu" }
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
# ///
"""Text fixtures using Cua-S1's actual prompt, model, and letter readout.

uv run research/cua-s1/reference.py prompts
uv run research/cua-s1/reference.py run
"""

import argparse
import json
import time
import warnings
from dataclasses import asdict
from importlib.metadata import distribution, version
from pathlib import Path

from cua_s1.four_b import (
    DEFAULT_BASE_MODEL,
    FourBModel,
    Option,
    assign_letters,
    build_prompt,
)
from transformers import AutoTokenizer

TOKENIZER_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
BASE_REVISION = TOKENIZER_REVISION
ADAPTER_REPO = "cua-ai/cua-s1-4b-0.2"
ADAPTER_REVISION = "16818868b0cc7813808aae4e87b417657046ab79"
ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT.parents[1] / "artifacts" / "cua-s1"


def prompts(args):
    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(
        DEFAULT_BASE_MODEL, revision=TOKENIZER_REVISION
    )
    # Construction is lazy; attach only the tokenizer to use the real letter readout.
    model = FourBModel(device="cpu", modality="text")
    model._tokenizer = tokenizer
    records = []
    for case in cases:
        assignment = assign_letters([Option(**option) for option in case["options"]])
        letter_ids = model._letter_token_ids(assignment)
        messages = build_prompt(
            assignment,
            app=case["app"],
            task_family=case["task_family"],
            ax_tree=case["ax_tree"],
            modality="text",
            goal=case.get("goal"),
        )
        chat_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(chat_text, return_tensors="pt")
        records.append({
            "name": case["name"],
            "chat_text": chat_text,
            "input_ids": inputs["input_ids"][0].tolist(),
            "letters": assignment.letters,
            "letter_ids": letter_ids,
        })
    source = json.loads(distribution("cua-s1").read_text("direct_url.json"))
    output = {
        "cua_s1_commit": source["vcs_info"]["commit_id"],
        "transformers_version": version("transformers"),
        "tokenizer": DEFAULT_BASE_MODEL,
        "tokenizer_revision": TOKENIZER_REVISION,
        "cases": records,
    }
    args.output.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(records)} prompt cases to {args.output}")


def layer_hooks(decoder, tensors):
    import torch

    handles = []

    def save(key, tensor):
        assert key not in tensors, f"duplicate dump: {key}"
        tensor = tensor.detach().to(device="cpu", dtype=torch.float32)
        assert torch.isfinite(tensor).all(), f"non-finite dump: {key}"
        tensors[key] = tensor.contiguous().clone()

    def output(module, key, last=False):
        def hook(module, inputs, value):
            if isinstance(value, tuple):
                value = value[0]
            save(key, value[:, -1, :] if last else value)

        handles.append(module.register_forward_hook(hook))

    def input(module, key, index=0, name="hidden_states"):
        def hook(module, inputs, kwargs):
            save(key, inputs[index] if len(inputs) > index else kwargs[name])

        handles.append(module.register_forward_pre_hook(hook, with_kwargs=True))

    output(decoder.embed_tokens, "embeddings.output")
    for i, layer in enumerate(decoder.layers):
        input(layer, f"layers.{i}.input")
        mixer = layer.linear_attn if layer.block_type == "linear_attention" else layer.self_attn
        output(mixer, f"layers.{i}.mixer_output")
        output(layer, f"layers.{i}.output")

    linear = decoder.layers[0].linear_attn
    for name in ("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj"):
        output(getattr(linear, name), f"layers.0.linear_attn.{name}")
    input(linear.norm, "layers.0.linear_attn.norm.input")
    input(linear.norm, "layers.0.linear_attn.norm.gate", index=1, name="gate")
    output(linear.norm, "layers.0.linear_attn.norm.output")

    attention = decoder.layers[3].self_attn
    for name in ("q_proj", "k_proj", "v_proj"):
        output(getattr(attention, name), f"layers.3.self_attn.{name}")
    input(attention.o_proj, "layers.3.self_attn.o_proj.input")
    output(decoder.norm, "norm.last", last=True)
    return handles


def run(args):
    import torch
    from huggingface_hub import snapshot_download
    from safetensors.torch import save_file

    started = time.perf_counter()
    torch.set_num_threads(args.threads)
    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    fixtures = json.loads(args.prompts.read_text(encoding="utf-8"))
    source = json.loads(distribution("cua-s1").read_text("direct_url.json"))
    assert fixtures["cua_s1_commit"] == source["vcs_info"]["commit_id"]
    assert fixtures["transformers_version"] == version("transformers")
    assert fixtures["tokenizer_revision"] == BASE_REVISION
    assert [case["name"] for case in cases] == [case["name"] for case in fixtures["cases"]]

    base = snapshot_download(
        DEFAULT_BASE_MODEL, revision=BASE_REVISION,
        local_dir=ARTIFACTS / "base" / BASE_REVISION,
        allow_patterns=["*.json", "*.safetensors", "*.jinja"],
    )
    adapter = snapshot_download(
        ADAPTER_REPO, revision=ADAPTER_REVISION,
        local_dir=ARTIFACTS / "adapter" / ADAPTER_REVISION,
        allow_patterns=["text/adapter_config.json", "text/adapter_model.safetensors"],
    )
    config = json.loads((Path(adapter) / "text" / "adapter_config.json").read_text())
    assert config["r"] == 16 and config["lora_alpha"] == 32
    model = FourBModel(
        base_model=base, lora_adapter_path=adapter,
        device=args.device, dtype=args.dtype, modality="text",
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message="Found missing adapter keys")
        model.load()
    decoder = model._model.get_base_model().model
    assert decoder.layers[0].block_type == "linear_attention"
    assert decoder.layers[3].block_type == "full_attention"
    load_seconds = time.perf_counter() - started
    print(f"loaded pinned base and text adapter in {load_seconds:.2f}s", flush=True)

    records = []
    tensors = {}
    for i, (case, fixture) in enumerate(zip(cases, fixtures["cases"], strict=True)):
        options = [Option(**option) for option in case["options"]]
        assignment = assign_letters(options)
        letter_ids = model._letter_token_ids(assignment)
        assert assignment.letters == fixture["letters"]
        assert letter_ids == fixture["letter_ids"]
        calls = 0
        logits = None

        def check_input(module, inputs, kwargs, fixture=fixture, dump=i == 0):
            nonlocal calls
            calls += 1
            ids = kwargs["input_ids"]
            assert ids.tolist() == [fixture["input_ids"]], f"input IDs differ: {fixture['name']}"
            if dump:
                tensors["input_ids"] = ids.detach().cpu().contiguous().clone()

        def capture_logits(module, inputs, result, letter_ids=letter_ids):
            nonlocal logits
            logits = result.logits[0, -1, letter_ids].detach().float().cpu().clone()

        handles = layer_hooks(decoder, tensors) if i == 0 else []
        handles.append(model._model.register_forward_pre_hook(check_input, with_kwargs=True))
        handles.append(model._model.register_forward_hook(capture_logits))
        case_started = time.perf_counter()
        try:
            probabilities = model.forward(
                options, app=case["app"], task_family=case["task_family"],
                ax_tree=case["ax_tree"], goal=case.get("goal"),
            )
        finally:
            for handle in handles:
                handle.remove()
        elapsed = time.perf_counter() - case_started
        assert calls == 1, f"expected one forward pass: {case['name']}"
        assert logits is not None and torch.isfinite(logits).all()
        torch.testing.assert_close(
            torch.tensor([option.probability for option in probabilities]),
            torch.softmax(logits, dim=-1), rtol=0, atol=0,
        )
        records.append({
            "name": case["name"],
            "input_tokens": len(fixture["input_ids"]),
            "forward_seconds": elapsed,
            "options": [
                {**asdict(option), "token_id": token_id, "logit": logit}
                for option, token_id, logit in zip(
                    probabilities, letter_ids, logits.tolist(), strict=True
                )
            ],
        })
        print(f"{case['name']}: {[option.probability for option in probabilities]} ({elapsed:.2f}s)", flush=True)
        if i == 0:
            assert len(tensors) == 3 * len(decoder.layers) + 15
            args.layers.parent.mkdir(parents=True, exist_ok=True)
            save_file(tensors, str(args.layers), metadata={
                "case": case["name"], "base_revision": BASE_REVISION,
                "adapter_revision": ADAPTER_REVISION, "dtype": args.dtype,
            })
            print(f"wrote {len(tensors)} tensors to {args.layers}", flush=True)
            tensors.clear()

    output = {
        "cua_s1_commit": source["vcs_info"]["commit_id"],
        "versions": {name: version(name) for name in (
            "torch", "torchvision", "pillow", "transformers", "peft", "accelerate",
            "huggingface-hub", "safetensors"
        )},
        "base_model": DEFAULT_BASE_MODEL,
        "base_revision": BASE_REVISION,
        "adapter_repo": ADAPTER_REPO,
        "adapter_revision": ADAPTER_REVISION,
        "adapter_subfolder": "text",
        "device": args.device,
        "dtype": args.dtype,
        "threads": torch.get_num_threads(),
        "attention_implementation": decoder.config._attn_implementation,
        "load_seconds": load_seconds,
        "total_seconds": time.perf_counter() - started,
        "cases": records,
    }
    args.output.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(records)} probability cases to {args.output} ({output['total_seconds']:.2f}s total)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("prompts", help="write text prompt fixtures without weights")
    command.add_argument("--cases", type=Path, default=ROOT / "cases.json")
    command.add_argument("--output", type=Path, default=ROOT / "prompts.json")
    command.set_defaults(run=prompts)
    command = commands.add_parser("run", help="score text cases and dump the first case's layers")
    command.add_argument("--cases", type=Path, default=ROOT / "cases.json")
    command.add_argument("--prompts", type=Path, default=ROOT / "prompts.json")
    command.add_argument("--output", type=Path, default=ROOT / "probabilities-f32.json")
    command.add_argument("--layers", type=Path, default=ARTIFACTS / "layers-f32.safetensors")
    command.add_argument("--device", choices=["cpu"], default="cpu")
    command.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    command.add_argument("--threads", type=int, default=4)
    command.set_defaults(run=run)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
