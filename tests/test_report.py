from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.evaluate.report import build_report, load_jsonl, score_summary


class ReportTests(unittest.TestCase):
    def test_report_contains_metric_aggregates_and_training_history(self) -> None:
        records = [
            {
                "id": "1",
                "source": "one two three",
                "summary": "one two",
                "summary__minicheck_score": 0.8,
                "summary__rouge_score": None,
            },
            {
                "id": "2",
                "source": "four five",
                "summary": "four",
                "summary__minicheck_score": 0.6,
                "summary__rouge_score": 0.4,
            },
        ]
        with tempfile.TemporaryDirectory(prefix="vdt-report-test-") as temporary:
            output = Path(temporary) / "report.html"
            html_path, json_path = build_report(
                records,
                output,
                manifest={"run_name": "pilot", "status": "completed"},
                history={"log_history": [{"step": 1, "loss": 1.2}]},
            )
            html = html_path.read_text(encoding="utf-8")
            report = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertIn("summary / minicheck", html)
            self.assertIn("Training history", html)
            self.assertEqual(report["records"], 2)
            self.assertEqual(len(report["metrics"]), 2)
            minicheck = next(row for row in report["metrics"] if row["metric"] == "minicheck")
            self.assertAlmostEqual(minicheck["mean"], 0.7)

    def test_load_jsonl_and_score_summary(self) -> None:
        with tempfile.TemporaryDirectory(prefix="vdt-report-jsonl-") as temporary:
            path = Path(temporary) / "scores.jsonl"
            path.write_text(
                json.dumps({"id": "1", "summary__factcc_score": 0.5}) + "\n",
                encoding="utf-8",
            )
            records = load_jsonl(path)
            summary = score_summary(records)
            self.assertEqual(summary[0]["metric"], "factcc")
            self.assertEqual(summary[0]["count"], 1)


if __name__ == "__main__":
    unittest.main()
