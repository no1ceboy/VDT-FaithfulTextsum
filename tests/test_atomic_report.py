from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.evaluate.atomic_report import (
    build_atomic_report,
    build_report_data,
    summarize_atomic_rows,
    summarize_parent_rows,
)


class AtomicReportTests(unittest.TestCase):
    def test_atomic_rates_separate_contradiction_and_unsupported(self) -> None:
        rows = [
            {"judge_status": "ok", "llm_label": "supported"},
            {"judge_status": "ok", "llm_label": "contradicted"},
            {"judge_status": "ok", "llm_label": "not_supported"},
            {"judge_status": "ok", "llm_label": "unclear"},
            {"judge_status": "ok", "llm_label": "not_a_claim"},
            {"judge_status": "error", "llm_label": ""},
        ]
        summary = summarize_atomic_rows(rows)
        self.assertEqual(summary["label_counts"]["contradicted"], 1)
        self.assertEqual(summary["factual_claim_rows"], 4)
        self.assertAlmostEqual(summary["rates"]["contradiction_rate"], 0.25)
        self.assertAlmostEqual(summary["rates"]["potential_hallucination_rate"], 0.5)
        self.assertAlmostEqual(
            summary["rates"]["decisive_potential_hallucination_rate"], 2 / 3, places=6
        )

    def test_parent_report_counts_flagged_documents(self) -> None:
        rows = [
            {"id": "a", "aggregate_label": "supported"},
            {"id": "a", "aggregate_label": "contradicted"},
            {"id": "b", "aggregate_label": "supported"},
            {"id": "c", "aggregate_label": "not_supported"},
        ]
        summary = summarize_parent_rows(rows)
        self.assertEqual(summary["documents"]["total"], 3)
        self.assertEqual(summary["documents"]["problem"], 2)
        self.assertAlmostEqual(summary["rates"]["summary_flagged_rate"], 2 / 4)
        self.assertAlmostEqual(summary["rates"]["document_flagged_rate"], 2 / 3, places=6)

    def test_build_report_includes_human_llm_comparison(self) -> None:
        atomic = {
            "human": [{"judge_status": "ok", "llm_label": "supported"}],
            "llm": [{"judge_status": "ok", "llm_label": "contradicted"}],
        }
        parents = {
            "human": [{"id": "1", "aggregate_label": "supported"}],
            "llm": [{"id": "1", "aggregate_label": "contradicted"}],
        }
        data = build_report_data(atomic, parents)
        comparison = data["comparisons"]["human_vs_llm"]
        self.assertEqual(comparison["paired_documents"]["right_only_problem"], 1)
        self.assertLess(comparison["rate_delta_right_minus_left"]["supported_rate"], 0)

        with tempfile.TemporaryDirectory(prefix="vdt-atomic-report-") as directory:
            html_path, json_path = build_atomic_report(
                atomic,
                Path(directory) / "audit.html",
                parent_inputs=parents,
            )
            self.assertTrue(html_path.is_file())
            self.assertTrue(json_path.is_file())
            report = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertIn("potential_hallucination_rate", report["systems"]["llm"]["atomic"]["rates"])
            self.assertIn("human_vs_llm", report["comparisons"])


if __name__ == "__main__":
    unittest.main()
