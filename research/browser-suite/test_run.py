import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import run


class BrowserSuiteTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.output = Path(self.directory.name)

    def invoke(self, *arguments):
        return subprocess.run(
            [sys.executable, "-B", str(Path(run.__file__)), *map(str, arguments)],
            capture_output=True,
            text=True,
            check=False,
        )

    def test_summarizes_verdicts_metrics_and_missing_runs(self):
        variants = ["done", "blocked", "status|mismatch", "verifier", "budget", "error", "cleanup", "missing"]
        run.write_json(self.output / "suite.json", {
            "repeat": 1,
            "cases": [{"case": "outcomes", "variants": variants, "exit_code": 1}],
        })
        folder = self.output / "outcomes"
        folder.mkdir()
        records = []
        for variant in variants[:-1]:
            result = {
                "passed": variant in {"done", "blocked"},
                "status": "blocked" if variant == "blocked" else "done",
                "actions": 0,
                "decisions": 2,
                "decision_median_ms": 12.375,
                "elapsed_ms": 51.5,
                "expectation_failures": [],
                "error": None,
            }
            if variant == "status|mismatch":
                result["expectation_failures"] = ["status_mismatch"]
            elif variant == "verifier":
                result["expectation_failures"] = ["verifier_failure"]
            elif variant == "budget":
                result["expectation_failures"] = ["too_many_actions"]
            elif variant == "error":
                result.update(status="error", error="action budget exhausted")
                result["expectation_failures"] = ["status_mismatch", "verifier_failure"]
            elif variant == "cleanup":
                result["cleanup_error"] = "close target failed"
            records.append({"variant": variant, "run": 0, "result": result})
        run.write_json(folder / "results.json", {"runs": records})

        completed = self.invoke("summarize", self.output)
        self.assertEqual(completed.returncode, 1, completed.stderr)
        self.assertIn("outcomes: 2/8 passed", completed.stdout)
        self.assertIn("Overall: 2/8 passed (25.0%)", completed.stdout)
        report = json.loads((self.output / "results.json").read_text())
        self.assertEqual(report["pass_rate"], 0.25)
        rows = report["runs"]
        self.assertEqual([row["failure_reason"] for row in rows], [
            None, None, "status mismatch", "verifier", "too many actions",
            "status mismatch / verifier / error", "error", "error",
        ])
        self.assertTrue(rows[1]["passed"])
        self.assertEqual(rows[1]["final_status"], "blocked")
        self.assertEqual(rows[0]["actions"], 0)
        self.assertEqual(rows[0]["decisions"], 2)
        self.assertEqual(rows[0]["decision_median_ms"], 12.375)
        self.assertEqual(rows[0]["elapsed_ms"], 51.5)
        self.assertEqual(rows[6]["error"], "close target failed")
        self.assertEqual(rows[7]["error"], "missing scenario run")
        self.assertIsNone(rows[7]["actions"])
        markdown = (self.output / "results.md").read_text()
        self.assertIn("status\\|mismatch", markdown)
        self.assertIn("12.38 | 51.50", markdown)
        self.assertIn("Median decision ms", markdown)

    def test_counts_unreadable_and_absent_case_results_as_errors(self):
        cases = [{"case": name, "variants": ["base"]} for name in ["unreadable", "absent"]]
        run.write_json(self.output / "suite.json", {"repeat": 2, "cases": cases})
        (self.output / "unreadable").mkdir()
        (self.output / "unreadable/results.json").write_text("{truncated")
        completed = self.invoke("summarize", self.output)
        self.assertEqual(completed.returncode, 1, completed.stderr)
        report = json.loads((self.output / "results.json").read_text())
        self.assertEqual(report["total_runs"], 4)
        self.assertTrue(all(row["failure_reason"] == "error" for row in report["runs"]))
        self.assertTrue(all(row["error"] for row in report["runs"]))

    def test_runs_only_local_cases_and_continues_after_failure(self):
        binary = self.output / "fake-browser"
        binary.write_text(f"#!{sys.executable}\n" + '''import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
for flag in ["scenario", "output", "repeat", "max-steps", "backend", "device", "policy"]:
    parser.add_argument("--" + flag, required=True)
args = parser.parse_args()
assert (args.backend, args.device, args.policy, args.max_steps) == ("cua-s1", "cpu", "native", "7")
output = Path(args.output)
output.mkdir()
scenario = json.loads(Path(args.scenario).read_text())
records = []
for variant in scenario["variants"]:
    for repeat in range(int(args.repeat)):
        passed = output.name != "already-done" or variant["name"] != "base" or repeat != 0
        result = {"passed": passed, "status": scenario.get("expect_status", "done"),
                  "error": None if passed else "synthetic failure", "actions": 0,
                  "decisions": 1, "decision_median_ms": 1.25, "elapsed_ms": 2.5}
        records.append({"variant": variant["name"], "run": repeat, "result": result})
(output / "results.json").write_text(json.dumps({"runs": records}))
raise SystemExit(0 if all(record["result"]["passed"] for record in records) else 1)
''')
        binary.chmod(0o755)
        output = self.output / "suite"
        arguments = (
            "run", "--binary", binary, "--output", output, "--repeat", "2", "--max-steps", "7",
            "--", "--backend", "cua-s1", "--device", "cpu", "--policy", "native",
        )
        completed = self.invoke(*arguments)
        self.assertEqual(completed.returncode, 1, completed.stderr)
        report = json.loads((output / "results.json").read_text())
        names = {case["case"] for case in report["cases"]}
        self.assertEqual(names, {
            "already-done", "big-index", "checkout", "contact-form",
            "out-of-stock", "paginated", "search-pick", "settings",
        })
        self.assertEqual(report["total_runs"], 46)
        self.assertEqual(report["passed_runs"], 45)
        self.assertTrue(all((output / name / "results.json").exists() for name in names))
        self.assertTrue(all((output / f"{name}.log").exists() for name in names))
        suite = json.loads((output / "suite.json").read_text())
        self.assertEqual([case["exit_code"] for case in suite["cases"]], [1, 0, 0, 0, 0, 0, 0, 0])
        original_report = (output / "results.json").read_bytes()
        self.assertEqual(self.invoke(*arguments).returncode, 2)
        self.assertEqual((output / "results.json").read_bytes(), original_report)

    def test_returns_success_only_for_all_passing_results(self):
        case = {"case": "complete", "variants": ["base"], "exit_code": 0}
        suite = {"repeat": 1, "cases": [case]}
        run.write_json(self.output / "suite.json", suite)
        (self.output / "complete").mkdir()
        run.write_json(self.output / "complete/results.json", {
            "runs": [{"variant": "base", "run": 0, "result": {"passed": True, "status": "done"}}],
        })
        self.assertEqual(self.invoke("summarize", self.output).returncode, 0)
        case["exit_code"] = 1
        run.write_json(self.output / "suite.json", suite)
        self.assertEqual(self.invoke("summarize", self.output).returncode, 1)

    def test_rejects_invalid_limits_and_owned_passthrough_flags(self):
        for flags in [
            ["--repeat", "0"], ["--max-steps", "-1"],
            ["--", "--output=elsewhere"], ["--", "--scenario", "elsewhere"],
        ]:
            with self.subTest(flags=flags):
                output = self.output / "unused"
                completed = self.invoke("run", "--output", output, *flags)
                self.assertEqual(completed.returncode, 2)
                self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
