"""Collect local saved decision requests; keep the resulting corpus untracked."""
import json
from pathlib import Path

root = Path(__file__).resolve().parents[2]
sessions = []
# Preserve local filesystem session order, as in the recorded experiment.
for path in (root / "artifacts").rglob("trace.json"):
    try:
        trace = json.loads(path.read_text())
    except (OSError, ValueError):
        continue
    if not isinstance(trace, dict):
        continue
    requests = [
        decision["request"]
        for decision in trace.get("decisions", [])
        if isinstance(decision, dict)
        and isinstance(decision.get("request"), dict)
        and "state" in decision["request"]
        and "questions" in decision["request"]
    ]
    if requests:
        sessions.append({"source": str(path.relative_to(root)), "requests": requests})
output = root / "artifacts/inference-followups/replay-inputs.json"
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(sessions))
print(f"{len(sessions)} sessions, {sum(len(s['requests']) for s in sessions)} requests")
