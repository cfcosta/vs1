# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Run the local tasks.html agent scenarios for one browser backend."""

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts" / "browser-suite"
FAILURE_REASONS = {
    "status_mismatch": "status mismatch",
    "verifier_failure": "verifier",
    "too_many_actions": "too many actions",
}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def discover_cases():
    cases = []
    fixture = (ROOT / "crates/vs1-browser/assets/tasks.html").resolve()
    for path in sorted((ROOT / "examples").glob("*-agent.json")):
        scenario = json.loads(path.read_text(encoding="utf-8"))
        source = scenario["source"]
        if source["kind"] != "file" or (path.parent / source["path"]).resolve() != fixture:
            continue
        cases.append({
            "case": path.stem.removesuffix("-agent"),
            "scenario": str(path),
            "variants": [variant["name"] for variant in scenario["variants"]],
        })
    if not cases:
        raise ValueError("no local tasks.html agent scenarios found")
    return cases


def read_case_results(output, case, repeat):
    path = output / case["case"] / "results.json"
    try:
        records = json.loads(path.read_text(encoding="utf-8"))["runs"]
        results = {}
        for record in records:
            key = (record["variant"], record["run"])
            if (
                key in results
                or key[0] not in case["variants"]
                or type(key[1]) is not int
                or not 0 <= key[1] < repeat
                or not isinstance(record["result"], dict)
            ):
                raise ValueError("invalid or duplicate scenario run")
            results[key] = record["result"]
        return results, None
    except (OSError, ValueError, KeyError, TypeError) as error:
        return {}, f"{path.name}: {error}"


def summarize_suite(output):
    suite = json.loads((output / "suite.json").read_text(encoding="utf-8"))
    rows = []
    counts = []
    for case in suite["cases"]:
        results, read_error = read_case_results(output, case, suite["repeat"])
        case_rows = []
        for variant in case["variants"]:
            for run in range(suite["repeat"]):
                result = results.get((variant, run), {})
                errors = [result.get("error"), result.get("cleanup_error")]
                if not result:
                    errors.append(case.get("error") or read_error or "missing scenario run")
                failures = result.get("expectation_failures", [])
                reasons = list(dict.fromkeys(FAILURE_REASONS.get(f, "error") for f in failures))
                if any(errors) or (result.get("passed") is not True and not reasons):
                    reasons.append("error")
                case_rows.append({
                    "case": case["case"],
                    "variant": variant,
                    "run": run,
                    "passed": result.get("passed") is True and not reasons,
                    "failure_reason": " / ".join(dict.fromkeys(reasons)) or None,
                    "final_status": result.get("status"),
                    "actions": result.get("actions"),
                    "decisions": result.get("decisions"),
                    "decision_median_ms": result.get("decision_median_ms"),
                    "elapsed_ms": result.get("elapsed_ms"),
                    "error": "; ".join(str(error) for error in errors if error) or None,
                })
        # A failed process with only passing records still needs an error in the report.
        if case.get("exit_code") and all(row["passed"] for row in case_rows):
            case_rows[-1].update(
                passed=False,
                failure_reason="error",
                error=f"vs1-browser exited {case['exit_code']}; see {case['case']}.log",
            )
        passed = sum(row["passed"] for row in case_rows)
        counts.append({"case": case["case"], "passed_runs": passed, "total_runs": len(case_rows)})
        rows.extend(case_rows)
        print(f"{case['case']}: {passed}/{len(case_rows)} passed")
    passed = sum(row["passed"] for row in rows)
    pass_rate = passed / len(rows) if rows else 0.0
    overall = f"Overall: {passed}/{len(rows)} passed ({pass_rate:.1%})"
    write_json(output / "results.json", {
        "passed_runs": passed,
        "total_runs": len(rows),
        "pass_rate": pass_rate,
        "cases": counts,
        "runs": rows,
    })
    columns = {
        "case": "Case",
        "variant": "Variant",
        "run": "Run",
        "passed": "Passed",
        "failure_reason": "Failure reason",
        "final_status": "Final status",
        "actions": "Actions",
        "decisions": "Decisions",
        "decision_median_ms": "Median decision ms",
        "elapsed_ms": "Elapsed ms",
    }
    lines = [
        "# Browser suite results", "", overall, "",
        *[f"- {count['case']}: {count['passed_runs']}/{count['total_runs']} passed" for count in counts],
        "", "| " + " | ".join(columns.values()) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    lines.extend("| " + " | ".join(format_cell(row[key]) for key in columns) + " |" for row in rows)
    (output / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(overall)
    print(f"Reports: {output / 'results.json'} and {output / 'results.md'}")
    return 0 if rows and passed == len(rows) else 1


def format_cell(value):
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value).replace("\\", "\\\\").replace("|", "\\|").replace("\n", "<br>")


def run_suite(args):
    browser_args = args.browser_args
    if browser_args[:1] == ["--"]:
        browser_args = browser_args[1:]
    for argument in browser_args:
        if argument.split("=", 1)[0] in {"--scenario", "--output", "--repeat", "--max-steps"}:
            raise ValueError(f"{argument} is controlled by the suite runner")
    cases = discover_cases()
    args.output.mkdir(parents=True, exist_ok=False)
    suite = {"repeat": args.repeat, "max_steps": args.max_steps, "cases": cases}
    write_json(args.output / "suite.json", suite)
    for case in cases:
        command = [
            args.binary, "--scenario", case["scenario"],
            "--repeat", str(args.repeat), "--max-steps", str(args.max_steps),
            "--output", str(args.output.resolve() / case["case"]), *browser_args,
        ]
        case["command"] = command
        print(f"Running {case['case']} (log: {args.output / (case['case'] + '.log')})", flush=True)
        with (args.output / f"{case['case']}.log").open("w", encoding="utf-8") as log:
            try:
                case["exit_code"] = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT).returncode
            except OSError as error:
                case["error"] = str(error)
                log.write(str(error) + "\n")
        write_json(args.output / "suite.json", suite)
    return summarize_suite(args.output)


def parse_positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    runner = commands.add_parser("run", help="run every local case for one backend")
    runner.add_argument("--output", type=Path, default=ARTIFACTS / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    runner.add_argument("--binary", default=str(ROOT / "target/release/vs1-browser"))
    runner.add_argument("--repeat", type=parse_positive_int, default=1)
    runner.add_argument("--max-steps", type=parse_positive_int, default=60)
    runner.add_argument("browser_args", nargs=argparse.REMAINDER, help="vs1-browser flags after --")
    summary = commands.add_parser("summarize", help="rewrite reports without starting a browser")
    summary.add_argument("output", type=Path)
    args = parser.parse_args()
    try:
        return run_suite(args) if args.command == "run" else summarize_suite(args.output)
    except (OSError, ValueError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
