"""Offline parity oracle using GLiClass's actual model, not a reimplemented head.

pip install gliclass==0.1.20 transformers==5.17.0 torch safetensors
python research/openjev/reference.py CHECKPOINT OUTPUT [cpu|cuda] [extended]
"""
import json
import sys
from pathlib import Path

import torch
from gliclass import GLiClassModel
from transformers import AutoTokenizer

root, output = map(Path, sys.argv[1:3])
device = sys.argv[3] if len(sys.argv) > 3 else "cpu"
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
model = GLiClassModel.from_pretrained(str(root)).eval().to(device)
tokenizer = AutoTokenizer.from_pretrained(str(root))
calibration = json.loads((root / "calibrator.json").read_text())
requests = []


def add(state, kind, instructions, criteria=None):
    question = {"type": kind, "instructions": instructions}
    if criteria is not None:
        question["criteria"] = criteria
    requests.append({"state": state, "questions": {"decision": question}})


add("The customer asks to reset a forgotten password.", "choice",
    "What should support do?", {"reset": "reset the password", "refund": "refund a purchase"})
add("The customer asks for a refund of a duplicate charge.", "choice",
    "What should support do?", {"reset": "reset the password", "refund": "refund a purchase"})
add("The sky is blue.", "choice", "Which database is down?", ["PostgreSQL", "MySQL"])
add("The package arrived late and damaged.", "score", "How satisfied is the customer?",
    ["very dissatisfied", "dissatisfied", "neutral", "satisfied", "very satisfied"])
add("An excellent product. I love it.", "score", "How satisfied is the customer?",
    ["very dissatisfied", "dissatisfied", "neutral", "satisfied", "very satisfied"])
add("A recipe calls for three eggs.", "score", "Rate the company's credit risk.",
    ["low", "medium", "high"])
add("The invoice is marked PAID.", "noul", "The invoice has been paid.")
add("The invoice is overdue and unpaid.", "noul", "The invoice has been paid.")
add("The invoice number is 123.", "noul", "The invoice has been paid.")
add("O cliente quer cancelar sua assinatura.", "choice", "Qual é a intenção?",
    {"cancel": "cancelar assinatura", "upgrade": "melhorar o plano", "help": "ajuda técnica"})
add("Route to department 19.", "choice", "Choose the requested department.",
    {str(i): f"department {i}" for i in range(24)})
add("Short.", "choice", "Choose.", [f"option {i}" for i in range(7)])
add("Background information. " * 600, "noul", "The account is suspended.")

if "extended" in sys.argv[4:]:
    # Cardinality, order, out-of-scope evidence, length and near-boundary rubrics.
    for count in [2, 3, 4, 6, 8, 10, 16, 24]:
        options = {str(i): f"department {i}" for i in range(count)}
        for state in ["Route to department 0.", f"Route to department {count - 1}.",
                      "No department was specified."]:
            for reverse in [False, True]:
                criteria = dict(reversed(list(options.items()))) if reverse else options
                add(state, "choice", "Choose the requested department.", criteria)
    for status in ["paid", "unpaid", "overdue", "partially paid", "unknown", "pending"]:
        for padding in [0, 30, 180]:
            add(f"The invoice status is {status}. " + "Background information. " * padding,
                "noul", "The invoice has been paid.")
    for state in ["Terrible.", "Bad, but usable.", "It is okay.", "Good.", "Excellent.",
                  "No opinion.", "The package arrived late and damaged.",
                  "I like the product but dislike the service."]:
        for levels in [["dissatisfied", "neutral", "satisfied"],
                       ["very dissatisfied", "dissatisfied", "neutral", "satisfied", "very satisfied"]]:
            add(state, "score", "How satisfied is the customer?", levels)

records = []
for request in requests:
    state = request["state"]
    q = request["questions"]["decision"]
    kind, instruction = q["type"], q["instructions"]
    if kind == "choice":
        options = q["criteria"]
        ids = list(options)
        descriptions = list(options.values()) if isinstance(options, dict) else options
        labels = [f"It is {description}" for description in descriptions]
        body = f"Question: {instruction}\n\nContext:\n{state}"
    elif kind == "score":
        ids = [str(i) for i in range(len(q["criteria"]))]
        labels = [f"{text} (Value: {float(i)})" for i, text in enumerate(q["criteria"])]
        body = f"Question: {instruction}\n\nContext:\n{state}"
    else:
        ids = ["true", "false"]
        labels = [f"true: {instruction}", f"false: not {instruction}"]
        body = f"Context:\n{state}\n\nEvaluate proposition: {instruction}"
    ids.append("__insufficient_evidence__")
    labels.append("insufficient evidence")
    prompt = "".join("<<LABEL>>" + label for label in labels) + "<<SEP>>" + body
    tokens = tokenizer(prompt, truncation=True, max_length=512, return_tensors="pt")
    with torch.inference_mode():
        logits = model(**tokens.to(device)).logits[0, :len(ids)].float().cpu()
    temperature = calibration.get("per_k", {}).get(str(len(ids)), calibration["temperature"])
    probabilities = (logits / temperature).softmax(-1).tolist()
    token_ids = tokens["input_ids"][0].cpu().tolist()
    records.append({
        "request": request, "prompt": prompt, "ids": token_ids,
        "markers": [i for i, token in enumerate(token_ids) if token == 50368],
        "logits": logits.tolist(), "probabilities": dict(zip(ids, probabilities)),
        "selected": ids[logits.argmax().item()], "temperature": temperature,
    })
output.write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n")
print(f"wrote {len(records)} reference cases to {output}")
