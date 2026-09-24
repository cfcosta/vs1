"""Offline parity oracle using upstream gliner2, not a reimplemented head.

pip install "gliner2[local] @ git+https://github.com/fastino-ai/GLiNER2@55656fbfa01d3d4a77485e1a1eeeaf682990ccdf"
python research/gliner-decide/reference.py CHECKPOINT REQUESTS OUTPUT

REQUESTS is a vs1 request file (one request or an array). Each request
becomes one upstream ``classify_text`` call whose tasks are the request's
questions, mapped the same way ``vs1::GlinerDecide`` maps them.
"""
import json
import sys

import torch
from gliner2 import AutoExtractor

root, requests_path, output = sys.argv[1:4]
torch.set_num_threads(4)
model = AutoExtractor.from_pretrained(root).eval()
raw = json.load(open(requests_path))
requests = raw if isinstance(raw, list) else [raw]


def render(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def task(question):
    kind = question["type"]
    criteria = question.get("criteria")
    descriptions = {}
    if kind == "choice":
        if isinstance(criteria, dict):
            labels = list(criteria)
            descriptions = {k: render(v) for k, v in criteria.items() if render(v)}
        else:
            labels = list(criteria)
    elif kind == "score":
        labels = [str(i) for i in range(len(criteria))]
        descriptions = {str(i): render(d) for i, d in enumerate(criteria) if render(d)}
    else:
        labels = ["yes", "no"]
        criteria = criteria or {}
        for label, key in (("yes", "true"), ("no", "false")):
            if render(criteria.get(key)):
                descriptions[label] = render(criteria[key])
    config = {"labels": labels}
    prompt = render(question.get("instructions"))
    if prompt:
        config["prompt"] = prompt
    if descriptions:
        config["label_descriptions"] = descriptions
    return config


results = []
for request in requests:
    state = request["state"]
    text = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
    tasks = {name: task(q) for name, q in request["questions"].items()}
    schema = model._classification_schema(tasks).build()
    batch = model.processor.collate_fn_inference([(text, schema)])
    with torch.inference_mode():
        hidden = model.encoder(
            input_ids=batch.input_ids, attention_mask=batch.attention_mask
        ).last_hidden_state[0]
        answers = {}
        for name, positions in zip(tasks, batch.schema_special_indices[0]):
            logits = model.classifier(hidden[positions[1:]]).squeeze(-1)
            probs = torch.softmax(logits, -1)
            answers[name] = {
                "labels": tasks[name]["labels"],
                "logits": logits.tolist(),
                "probabilities": probs.tolist(),
            }
        upstream = model.classify_text(text, tasks, include_confidence=True)
    for name, answer in answers.items():
        best = max(range(len(answer["labels"])), key=lambda i: answer["probabilities"][i])
        got = upstream[name]
        assert got["label"] == answer["labels"][best], (name, got, answer)
        assert abs(got["confidence"] - answer["probabilities"][best]) < 1e-5
    results.append({
        "ids": batch.input_ids[0].tolist(),
        "markers": batch.schema_special_indices[0],
        "answers": answers,
        "upstream": upstream,
    })
json.dump(results, open(output, "w"), indent=1)
print(json.dumps([r["upstream"] for r in results], indent=1))
