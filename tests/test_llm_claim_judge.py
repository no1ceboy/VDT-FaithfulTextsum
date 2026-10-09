from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.evaluate.llm_claim_judge import (
    JudgeError,
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


class FakeBatchJudge(FakeJudge):
    backend_name = "local"

    def __init__(self) -> None:
        self.batch_calls = 0

    def judge_many(self, rows):
        self.batch_calls += 1
        return [
            {
                "label": "supported",
                "reason": "The source states the same fact.",
                "evidence_quote": row["source"],
                "confidence": 0.9,
                "raw_response": '{"label":"supported"}',
                "raw_attempts": ['{"label":"supported"}'],
            }
            for row in rows
        ]


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

    def test_parse_multiple_evidence_quotes_keeps_legacy_first_quote(self) -> None:
        result = parse_judgment(
            '{"label":"supported","reason":"Both source sentences provide evidence.",'
            '"evidence_quotes":[{"sentence_index":1,"quote":"First fact."},'
            '{"sentence_index":4,"quote":"Second fact."}],"confidence":0.8}'
        )
        self.assertEqual(len(result["evidence_quotes"]), 2)
        self.assertEqual(result["evidence_quotes"][1]["sentence_index"], 4)
        self.assertEqual(result["evidence_quote"], "First fact.")

    def test_parse_judgment_rejects_reasoning_or_embedded_json(self) -> None:
        responses = (
            '<think>First I will inspect the source.</think>\n'
            '{"label":"supported","reason":"same fact","evidence_quote":"fact","confidence":0.9}',
            'The answer is supported. '
            '{"label":"supported","reason":"same fact","evidence_quote":"fact","confidence":0.9}',
        )
        for response in responses:
            with self.assertRaisesRegex(JudgeError, "invalid|exactly one"):
                parse_judgment(response)

    def test_parse_judgment_requires_all_fields_and_bounded_confidence(self) -> None:
        with self.assertRaisesRegex(JudgeError, "missing required"):
            parse_judgment('{"label":"supported"}')
        with self.assertRaisesRegex(JudgeError, "between 0 and 1"):
            parse_judgment(
                '{"label":"supported","reason":"same fact",'
                '"evidence_quote":"fact","confidence":2}'
            )

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

    def test_run_uses_local_batch_judge_when_requested(self) -> None:
        rows = [
            {"id": "1", "source": "A.", "claim": "A.", "evidence": []},
            {"id": "2", "source": "B.", "claim": "B.", "evidence": []},
        ]
        judge = FakeBatchJudge()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "judged.jsonl"
            result = run_judgment(
                rows,
                judge,
                output,
                overwrite=True,
                model_batch_size=2,
            )
        self.assertEqual(judge.batch_calls, 1)
        self.assertEqual(len(result), 2)
        self.assertTrue(all(row["judge_status"] == "ok" for row in result))

    def test_judgment_errors_preserve_raw_attempts(self) -> None:
        class FailingJudge:
            backend_name = "local"
            model_name = "local-model"
            last_raw_response = "bad first response\n--- repair attempt ---\nbad repair"
            last_raw_attempts = ["bad first response", "bad repair"]

            def judge(self, row):
                raise RuntimeError("invalid structured judgment")

        rows = [{"id": "1", "source": "Source.", "claim": "Claim."}]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "judged.jsonl"
            run_judgment(rows, FailingJudge(), output, overwrite=True)
            saved = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(saved["judge_status"], "error")
        self.assertEqual(saved["llm_raw_attempts"], ["bad first response", "bad repair"])
        self.assertTrue(saved["llm_repair_attempted"])
