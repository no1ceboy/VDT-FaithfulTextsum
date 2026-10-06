from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.evaluate.llm_claim_judge import (
    build_judge_prompt,
    parse_judgment,
    run_judgment,
    summarize_judgments,
)


class FakeJudge:
    model_name = "fake-model"

    def judge(self, row):
        return {
            "label": "supported",
            "reason": "The source states the same fact.",
            "evidence_quote": row["source"],
            "confidence": 0.9,
            "raw_response": '{"label":"supported"}',
        }


class LLMClaimJudgeTests(unittest.TestCase):
    def test_prompt_keeps_claim_and_source_separate(self) -> None:
        prompt = build_judge_prompt(
            {
                "source": "Nguồn nói rằng việc này xảy ra năm 2016.",
                "claim": "Việc này xảy ra năm 2016.",
                "evidence": [{"text": "việc này xảy ra năm 2016", "retrieval_score": 1.0}],
            }
        )
        self.assertIn("CLAIM:", prompt)
        self.assertIn("SOURCE DOCUMENT", prompt)
        self.assertIn("năm 2016", prompt)

    def test_parse_json_and_alias(self) -> None:
        result = parse_judgment(
            '```json\n{"label":"not supported","reason":"No evidence.","evidence_quote":"","confidence":0.2}\n```'
        )
        self.assertEqual(result["label"], "not_supported")
        self.assertEqual(result["confidence"], 0.2)

    def test_run_preserves_human_label_and_writes_summary(self) -> None:
        rows = [
            {
                "id": "1",
                "human_label": "not_a_claim",
                "source": "A title.",
                "claim": "A title.",
                "evidence": [],
            }
        ]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "judged.jsonl"
            result = run_judgment(rows, FakeJudge(), output, overwrite=True)
            saved = json.loads(output.read_text(encoding="utf-8"))
            summary = json.loads(output.with_suffix(".summary.json").read_text(encoding="utf-8"))
        self.assertEqual(result[0]["human_label"], "not_a_claim")
        self.assertEqual(saved["llm_label"], "supported")
        self.assertEqual(summary["human_llm_agreement_rows"], 1)
        self.assertEqual(summary["human_llm_agreement"], 0.0)

    def test_summary_counts_errors(self) -> None:
        summary = summarize_judgments(
            [
                {"judge_status": "ok", "llm_label": "supported", "human_label": "supported"},
                {"judge_status": "error", "llm_label": "", "human_label": ""},
            ]
        )
        self.assertEqual(summary["rows_judged"], 1)
        self.assertEqual(summary["rows_error"], 1)
        self.assertEqual(summary["human_llm_agreement"], 1.0)
