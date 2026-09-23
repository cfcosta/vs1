# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = [
#   "cua-s1 @ git+https://github.com/trycua/cua.git@a5f18829df026d7b9ef80c339194b44b1c61f856#subdirectory=libs/cua-s1/python",
#   "transformers==5.17.0",
#   "torch==2.13.0",
# ]
# [tool.uv.sources]
# torch = { index = "pytorch-cpu" }
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
# ///
"""Text prompt fixtures using Cua-S1's actual prompt and letter mapping.

uv run research/cua-s1/reference.py prompts
"""

import argparse
import json
from importlib.metadata import distribution, version
from pathlib import Path

from cua_s1.four_b import DEFAULT_BASE_MODEL, FourBModel, Option, assign_letters, build_prompt
from transformers import AutoTokenizer

TOKENIZER_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
ROOT = Path(__file__).resolve().parent


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("prompts", help="write text prompt fixtures without weights")
    command.add_argument("--cases", type=Path, default=ROOT / "cases.json")
    command.add_argument("--output", type=Path, default=ROOT / "prompts.json")
    command.set_defaults(run=prompts)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
