from __future__ import annotations

import unittest

from src.evaluate.claim_extractor import ClaimnessClassifier, candidate_spans, extract_rows, parse_claimness


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

    def test_claimness_parser_accepts_yes_no_strings(self) -> None:
        result = parse_claimness('{"is_claim":"no","unit_type":"heading","reason":"title"}')
        self.assertFalse(result["is_claim"])
        self.assertEqual(result["unit_type"], "heading")

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
