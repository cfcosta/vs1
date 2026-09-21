"""Archive full run snapshots locally; retain compact, auditable screen results."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW = ROOT.parents[1] / "artifacts" / "model-compression"


def errors(run):
    by_field = {k: 0.0 for k in ["probabilities", "confidence", "noul", "score"]}
    for left, right in zip(
        run["baseline"]["responses"], run["candidate"]["responses"], strict=True
    ):
        assert left["answers"].keys() == right["answers"].keys()
        for key, a in left["answers"].items():
            b = right["answers"][key]
            for field, previous in by_field.items():
                if field not in a:
                    continue
                delta = (
                    max(abs(v - b[field][k]) for k, v in a[field].items())
                    if isinstance(a[field], dict)
                    else abs(a[field] - b[field])
                )
                by_field[field] = max(previous, delta)
    return by_field


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    files = list(ROOT.glob("*.json"))
    for path in files:
        if path.name in {"results.json", "screen-corpus.json", "training-results.json"}:
            continue
        value = json.loads(path.read_text())
        if "variant" in value and "baseline" in value:
            path.replace(RAW / path.name)
    runs = []
    baseline = None
    for path in sorted(RAW.glob("*.json")):
        value = json.loads(path.read_text())
        if "variant" not in value or "baseline" not in value:
            continue
        if baseline is None:
            baseline = value["baseline"]
            (ROOT / "screen-corpus.json").write_text(
                json.dumps(
                    {"requests": value["requests"], "baseline": baseline}, indent=2
                )
                + "\n"
            )
        assert value["baseline"] == baseline, f"baseline changed: {path}"
        assert value["restored_exact"]
        fields = errors(value)
        quality = value["quality"] | {"max_error_by_field": fields}
        quality["passes_numeric_screen"] = (
            quality["decision_flips"] == 0
            and max(fields[k] for k in ["probabilities", "confidence", "noul"]) <= 0.01
            and fields["score"] <= 0.02
            and quality["max_action_abs_error"] <= 0.01
        )
        variant = value["variant"]
        timing_status = (
            "not measured"
            if not value["timings"]
            else "exploratory paired screen; not an accepted improvement"
        )
        if variant == "quant-8":
            timing_status = "both timing runs overlapped another GPU process; no performance conclusion"
        elif variant.startswith("attention-"):
            timing_status = (
                "three pairs only; late sweep overlapped another GPU process"
            )
        runs.append(
            {
                "variant": variant,
                "quality": quality,
                "timings": value["timings"],
                "timing_status": timing_status,
                "restored_exact": True,
                "raw_file": str(path.relative_to(ROOT.parents[1])),
            }
        )
    (ROOT / "results.json").write_text(json.dumps(runs, indent=2) + "\n")
    training = {
        p.parent.name: json.loads(p.read_text())
        for p in sorted(RAW.glob("*/training.json"))
    }
    (ROOT / "training-results.json").write_text(json.dumps(training, indent=2) + "\n")
    print(
        f"{len(runs)} screens; {sum(r['quality']['passes_numeric_screen'] for r in runs)} numerical passes (includes unpruned control); identical baseline across all runs"
    )


if __name__ == "__main__":
    main()
