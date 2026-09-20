"""Summarize timing changes and exact output differences; raw runs stay in artifacts."""
import json
import sys
from pathlib import Path


def differences(a, b, path=""):
    if isinstance(a, dict) and isinstance(b, dict) and a.keys() == b.keys():
        return [d for k in a for d in differences(a[k], b[k], f"{path}/{k}")]
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in differences(x, y, f"{path}/{i}")]
    return [] if a == b else [{"path": path, "before": a, "after": b}]


if __name__ == "__main__":
    before, after = (json.loads(Path(p).read_text()) for p in sys.argv[1:3])
    assert [c["name"] for c in before["cases"]] == [c["name"] for c in after["cases"]]
    result = []
    for a, b in zip(before["cases"], after["cases"]):
        drift = differences(a["outputs"], b["outputs"])
        drift += differences(a.get("alternate_outputs"), b.get("alternate_outputs"), "/alternate")
        result.append({"name": a["name"], "baseline_p50_ms": a["p50_ms"],
                       "candidate_p50_ms": b["p50_ms"],
                       "p50_change_percent": 100 * (b["p50_ms"] / a["p50_ms"] - 1),
                       "baseline_p95_ms": a["p95_ms"], "candidate_p95_ms": b["p95_ms"],
                       "baseline_questions_per_second": a["questions_per_second"],
                       "candidate_questions_per_second": b["questions_per_second"],
                       "differing_outputs": len(drift), "first_differences": drift[:5]})
    print(json.dumps(result, indent=2))
