"""Compare `vs1 --dump-ids` output with reference.py output.

python research/gliner-decide/compare.py REFERENCE RUST
"""
import json
import sys

ref = json.load(open(sys.argv[1]))
got = json.load(open(sys.argv[2]))
assert len(ref) == len(got)
worst = 0.0
flips = []
questions = 0
for i, (r, g) in enumerate(zip(ref, got)):
    assert r["ids"] == g["ids"]["ids"], f"request {i}: token ids differ"
    assert [m[1:] for m in r["markers"]] == g["ids"]["markers"], f"request {i}: markers differ"
    for name, answer in r["answers"].items():
        questions += 1
        a = g["response"]["answers"][name]
        probs = a.get("probabilities") or {"yes": a["noul"], "no": 1 - a["noul"]}
        for label, p in zip(answer["labels"], answer["probabilities"]):
            worst = max(worst, abs(probs[label] - p))
        best = max(probs, key=probs.get)
        if best != r["upstream"][name]["label"]:
            flips.append((i, name, r["upstream"][name]["label"], best))
print(json.dumps({
    "requests": len(ref),
    "questions": questions,
    "ids_identical": True,
    "max_probability_error": worst,
    "label_mismatches": flips,
}))
