"""Prepare local labeled examples for the Rust retrieval audit.

Training rows: subject, sender (normalized address), body, label.
Target rows: subject, sender, date, body. Target labels are never read.
"""

import argparse
import json
import pathlib

from methods import independent, nearest

MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
REVISION = "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"


def main():
    import numpy as np
    from sentence_transformers import SentenceTransformer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("training", type=pathlib.Path)
    parser.add_argument("targets", type=pathlib.Path)
    parser.add_argument("output", type=pathlib.Path)
    args = parser.parse_args()
    train = json.loads(args.training.read_text())
    targets = json.loads(args.targets.read_text())
    if len(train) < 2 or not targets:
        raise ValueError("Need at least two training examples and one target")
    if independent(train, targets) != targets:
        raise ValueError("Evaluation targets overlap training senders or templates")
    keys = [x["subject"] + "\n" + x["date"] for x in targets]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate subject/date target keys")
    model = SentenceTransformer(MODEL, revision=REVISION, device="cuda")
    rows = train + targets
    chunks, weights, owners = [], [], []
    for i, row in enumerate(rows):
        ids = model.tokenizer.encode(
            row["subject"] + "\n" + row["body"], add_special_tokens=False
        )
        for start in range(0, max(1, len(ids)), 120):
            part = ids[start : start + 120]
            text = model.tokenizer.decode(part, skip_special_tokens=True)
            if len(model.tokenizer.encode(text)) > model.max_seq_length:
                raise ValueError("A decoded chunk exceeds the encoder token budget")
            chunks.append(text)
            weights.append(max(1, len(part)))
            owners.append(i)
    vectors = model.encode(chunks, batch_size=32, normalize_embeddings=True)
    pooled = np.zeros((len(rows), vectors.shape[1]), dtype=np.float32)
    for vector, weight, i in zip(vectors, weights, owners):
        pooled[i] += vector * weight
    pooled /= np.maximum(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-9)
    examples = {}
    for i, key in enumerate(keys):
        neighbors = nearest(pooled, len(train) + i, list(range(len(train))))
        examples[key] = [
            {
                "subject": train[j]["subject"],
                "body": train[j]["body"][:160],
                "category": train[j]["label"],
            }
            for j in neighbors
        ]
    with args.output.open("x") as output:
        json.dump(examples, output, ensure_ascii=False, indent=2)
    print(
        f"Prepared {len(examples)} targets using {len(train)} labeled training messages"
    )


if __name__ == "__main__":
    main()
