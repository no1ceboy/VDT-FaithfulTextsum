from __future__ import annotations

import unittest

from src.evaluate.claim_extractor import ClaimnessClassifier, candidate_spans, extract_rows, parse_claimness
from src.evaluate.llm_claim_judge import JudgeError


class FakeGenerator:
    backend_name = "fake"
    model_name = "fake-model"

    def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        if "<candidate>\nTitle about regulations" in user_prompt:
            return '{"is_claim": false, "unit_type": "heading", "reason": "Topic heading."}'
        return '{"is_claim": true, "unit_type": "claim", "reason": "It asserts an event."}'


class ClaimExtractorTests(unittest.TestCase):
    def test_candidate_spans_are_exact_substrings(self) -> None:
        summary = "Title about regulations\nOn 1/7/2016 release immediately."
        spans = candidate_spans(summary, split_clauses=False)
        self.assertTrue(spans)
        for span in spans:
            self.assertEqual(summary[span["start_char"] : span["end_char"]], span["candidate"])

    def test_colon_heading_keeps_subject_with_predicate_by_default(self) -> None:
        summary = "Tuổi Thân: gặp nhiều rắc rối trong công việc."
        self.assertEqual(
            [span["candidate"] for span in candidate_spans(summary)],
            [summary],
        )
        self.assertEqual(
            [span["candidate"] for span in candidate_spans(summary, split_clauses=True)],
            ["Tuổi Thân", "gặp nhiều rắc rối trong công việc."],
        )

    def test_claimness_parser_accepts_yes_no_strings(self) -> None:
        result = parse_claimness('{"is_claim":"no","unit_type":"heading","reason":"title"}')
        self.assertFalse(result["is_claim"])
        self.assertEqual(result["unit_type"], "heading")

    def test_claimness_parser_accepts_common_instruct_model_variants(self) -> None:
        fenced = parse_claimness(
            "```json\n{\"isClaim\": 1, \"unit_type\": \"claim\", \"reason\": \"assertion\"}\n```"
        )
        self.assertTrue(fenced["is_claim"])

        python_dict = parse_claimness(
            "{'is_claim': False, 'unit_type': 'heading', 'reason': 'title'}"
        )
        self.assertFalse(python_dict["is_claim"])

        short_answer = parse_claimness("is_claim: false")
        self.assertFalse(short_answer["is_claim"])

    def test_claimness_parser_rejects_thinking_or_prose_around_json(self) -> None:
        responses = (
            '<think>First I will reason about the candidate.</think>\n'
            '{"is_claim": true, "unit_type": "claim", "reason": "assertion"}',
            'The answer is obvious. {"is_claim": true, "unit_type": "claim", "reason": "assertion"}',
        )
        for response in responses:
            with self.assertRaisesRegex(JudgeError, "invalid|exactly one"):
                parse_claimness(response)

    def test_claimness_parser_rejects_incomplete_structured_object(self) -> None:
        with self.assertRaisesRegex(JudgeError, "missing required"):
            parse_claimness('{"return": false, "is_claim": false}')

    def test_failed_claimness_response_is_preserved_for_debugging(self) -> None:
        class BadGenerator:
            backend_name = "fake"
            model_name = "fake-model"

            def generate_text(self, system_prompt: str, user_prompt: str) -> str:
                return "I cannot determine this."

        _, _, errors = extract_rows(
            [{"id": "1", "input": "Source.", "output": "A claim."}],
            ClaimnessClassifier(BadGenerator()),
            source_col="input",
            summary_col="output",
            split_clauses=False,
        )
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0]["claimness_raw"], "I cannot determine this.")
        self.assertEqual(errors[0]["claimness_raw_attempts"], ["I cannot determine this."])
        self.assertFalse(errors[0]["claimness_repair_attempted"])

    def test_local_claimness_gets_one_bounded_format_repair_attempt(self) -> None:
        class RepairGenerator:
            backend_name = "local"
            model_name = "local-model"

            def __init__(self) -> None:
                self.responses = [
                    '{"classification": "claim"}',
                    '{"is_claim": true, "unit_type": "claim", "reason": "assertion"}',
                ]

            def generate_text(self, system_prompt: str, user_prompt: str) -> str:
                return self.responses.pop(0)

        claims, non_claims, errors = extract_rows(
            [{"id": "1", "input": "Source.", "output": "A claim."}],
            ClaimnessClassifier(RepairGenerator()),
            source_col="input",
            summary_col="output",
            split_clauses=False,
        )
        self.assertEqual(len(claims), 1)
        self.assertEqual(non_claims, [])
        self.assertEqual(errors, [])
        self.assertTrue(claims[0]["claimness_repair_attempted"])
        self.assertEqual(len(claims[0]["claimness_raw_attempts"]), 2)

    def test_extraction_separates_claims_and_headings(self) -> None:
        summary = "Title about regulations. On 1/7/2016 release immediately."
        claims, non_claims, errors = extract_rows(
            [{"id": "1", "input": "Source.", "output": summary}],
            ClaimnessClassifier(FakeGenerator()),
            source_col="input",
            summary_col="output",
            split_clauses=False,
        )
        self.assertEqual(len(claims), 1)
        self.assertEqual(claims[0]["claim"], "On 1/7/2016 release immediately.")
        self.assertEqual(len(non_claims), 1)
        self.assertEqual(non_claims[0]["claim_type"], "heading")
        self.assertEqual(errors, [])
