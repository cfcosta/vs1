"""Summarize the effect of precision on the public answer semantics.

python research/openjev/audit_precision.py PARITY_JSON [...]
No model execution or training; inputs are captured openjev_parity outputs.
"""
import json
import sys
from pathlib import Path

ABSTAIN = "__insufficient_evidence__"


def summarize(path):
    report = json.loads(Path(path).read_text())
    result = {k: v for k, v in report.items() if k != "details"}
    result["file"] = path
    result["comparisons"] = {}
    for mode in ["reference_batch", "reference_single", "batch_single"]:
        maxima = {"probability": 0., "score": 0., "normalized_score": 0., "noul": 0.}
        changes = []
        score_details = []
        for index, row in enumerate(report["details"]):
            q = row["request"]["questions"]["decision"]
            first = row["batch"]["probabilities"] if mode == "batch_single" else row["reference"]
            second = row["batch" if mode == "reference_batch" else "single"]["probabilities"]
            a, b = max(first, key=first.get), max(second, key=second.get)
            abstain_a, abstain_b = a == ABSTAIN, b == ABSTAIN
            if abstain_a != abstain_b or (q["type"] == "choice" and a != b):
                changes.append({"index": index, "kind": q["type"], "from": a, "to": b,
                                "reference_margin": sorted(first.values(), reverse=True)[0]
                                - sorted(first.values(), reverse=True)[1]})
            maxima["probability"] = max(maxima["probability"], max(abs(first[k]-second[k]) for k in first))
            if abstain_a or abstain_b:
                continue
            first = {k: v/(1-first[ABSTAIN]) for k, v in first.items() if k != ABSTAIN}
            second = {k: v/(1-second[ABSTAIN]) for k, v in second.items() if k != ABSTAIN}
            if q["type"] == "score":
                expected_a = sum(int(k)*v for k, v in first.items())
                expected_b = sum(int(k)*v for k, v in second.items())
                error = abs(expected_a-expected_b)
                maxima["score"] = max(maxima["score"], error)
                maxima["normalized_score"] = max(maxima["normalized_score"], error/(len(first)-1))
                if a != b:
                    score_details.append({"index": index, "argmax": [a, b], "score": [expected_a, expected_b], "error": error})
            elif q["type"] == "noul":
                maxima["noul"] = max(maxima["noul"], abs(first["true"]-second["true"]))
        result["comparisons"][mode] = {"max_error": maxima, "discrete_changes": changes, "score_argmax_changes": score_details}
    return result


print(json.dumps([summarize(path) for path in sys.argv[1:]], indent=2))
