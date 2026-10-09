from __future__ import annotations

import json
import unittest

from src.evaluate.atomic_claims import (
    AtomicClaimDecomposer,
    decompose_rows,
    decompose_original_records,
    parse_atomic_decomposition,
)
from src.evaluate.atomic_aggregate import aggregate_atomic_judgments
from src.evaluate.llm_claim_judge import JudgeError


class FakeAtomicGenerator:
    backend_name = "local"
    model_name = "fake-atomic"

    def __init__(self, response: str) -> None:
        self.response = response

    def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        del system_prompt, user_prompt
        return self.response


class FakeBatchAtomicGenerator(FakeAtomicGenerator):
    def __init__(self, responses: list[str]) -> None:
        super().__init__(responses[0])
        self.responses = responses
        self.batch_calls = 0

    def generate_text_batch(self, system_prompt: str, user_prompts: list[str]) -> list[str]:
        del system_prompt, user_prompts
        self.batch_calls += 1
        return self.responses


class AtomicClaimTests(unittest.TestCase):
    def test_decomposition_requires_exact_surface_spans(self) -> None:
        parent = "Alice won the award, and Bob published the report."
        result = parse_atomic_decomposition(
            json.dumps(
                {
                    "atomic_claims": [
                        {
                            "surface_text": "Alice won the award",
                            "verification_text": "Alice won the award.",
                        },
                        {
                            "surface_text": "Bob published the report",
                            "verification_text": "Bob published the report.",
                        },
                    ]
                }
            ),
            parent,
        )
        self.assertEqual(result[0]["start_char"], 0)
        self.assertEqual(result[1]["surface_text"], "Bob published the report")
        with self.assertRaises(JudgeError):
            parse_atomic_decomposition(
                '{"atomic_claims":[{"surface_text":"invented fact",'
                '"verification_text":"invented fact"}]}',
                parent,
            )

    def test_decompose_rows_retrieves_evidence_per_atomic_claim(self) -> None:
        parent = "Alice won the award, and Bob published the report."
        response = json.dumps(
            {
                "atomic_claims": [
                    {
                        "surface_text": "Alice won the award",
                        "verification_text": "Alice won the award.",
                    },
                    {
                        "surface_text": "Bob published the report",
                        "verification_text": "Bob published the report.",
                    },
                ]
            }
        )
        rows, errors = decompose_rows(
            [
                {
                    "id": "doc-1",
                    "source": "Alice won the award. Bob published the report.",
                    "claim": parent,
                    "start_char": 10,
                    "end_char": 10 + len(parent),
                }
            ],
            AtomicClaimDecomposer(FakeAtomicGenerator(response)),
            top_k=1,
        )
        self.assertEqual(errors, [])
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["evidence"][0]["sentence_index"], 0)
        self.assertEqual(rows[1]["evidence"][0]["sentence_index"], 1)
        self.assertEqual(rows[1]["start_char"], 10 + parent.index("Bob"))

    def test_direct_mode_uses_original_summary_and_keeps_coverage(self) -> None:
        parent = "Alice won the award, and Bob published the report."
        response = json.dumps(
            {
                "atomic_claims": [
                    {
                        "surface_text": "Alice won the award",
                        "verification_text": "Alice won the award.",
                    },
                    {
                        "surface_text": "Bob published the report",
                        "verification_text": "Bob published the report.",
                    },
                ]
            }
        )
        rows, coverage, errors = decompose_original_records(
            [
                {
                    "id": "doc-1",
                    "text": "Alice won the award. Bob published the report.",
                    "summary": parent,
                }
            ],
            AtomicClaimDecomposer(FakeAtomicGenerator(response)),
            source_col="text",
            summary_col="summary",
            top_k=1,
        )
        self.assertEqual(errors, [])
        self.assertEqual(len(rows), 2)
        self.assertEqual(len(coverage), 1)
        self.assertEqual(rows[0]["source"], "Alice won the award. Bob published the report.")
        self.assertEqual(rows[0]["summary_col"], "summary")
        self.assertEqual(rows[1]["start_char"], parent.index("Bob"))
        self.assertEqual(coverage[0]["atomic_claim_count"], 2)
        self.assertEqual(coverage[0]["decomposition_status"], "ok")

    def test_direct_mode_preserves_non_claim_summary_units(self) -> None:
        response = json.dumps({"atomic_claims": []})
        rows, coverage, errors = decompose_original_records(
            [{"id": "doc-1", "text": "Source.", "summary": "Daily advice."}],
            AtomicClaimDecomposer(FakeAtomicGenerator(response)),
            source_col="text",
            summary_col="summary",
        )
        self.assertEqual(rows, [])
        self.assertEqual(errors, [])
        self.assertEqual(coverage[0]["decomposition_status"], "no_atomic_claims")
        self.assertEqual(coverage[0]["atomic_claim_count"], 0)

    def test_direct_mode_batches_local_summary_units(self) -> None:
        generator = FakeBatchAtomicGenerator(
            [
                json.dumps(
                    {
                        "atomic_claims": [
                            {
                                "surface_text": "Alice won",
                                "verification_text": "Alice won.",
                            }
                        ]
                    }
                ),
                json.dumps(
                    {
                        "atomic_claims": [
                            {
                                "surface_text": "Bob published",
                                "verification_text": "Bob published.",
                            }
                        ]
                    }
                ),
            ]
        )
        rows, coverage, errors = decompose_original_records(
            [
                {"id": "doc-1", "text": "Alice won.", "summary": "Alice won."},
                {"id": "doc-2", "text": "Bob published.", "summary": "Bob published."},
            ],
            AtomicClaimDecomposer(generator),
            source_col="text",
            summary_col="summary",
            model_batch_size=2,
        )
        self.assertEqual(errors, [])
        self.assertEqual(generator.batch_calls, 1)
        self.assertEqual(len(rows), 2)
        self.assertEqual(len(coverage), 2)

    def test_aggregate_is_conservative_and_keeps_all_quotes(self) -> None:
        rows = [
            {
                "id": "doc-1",
                "summary_col": "summary",
                "parent_claim_index": 0,
                "parent_claim": "A and B.",
                "source": "A. B.",
                "atomic_claim_index": 0,
                "atomic_claim_count": 2,
                "claim": "A.",
                "judge_status": "ok",
                "llm_label": "supported",
                "llm_confidence": 0.9,
                "llm_evidence_quotes": [{"sentence_index": 0, "quote": "A."}],
                "evidence": [{"sentence_index": 0, "text": "A."}],
            },
            {
                "id": "doc-1",
                "summary_col": "summary",
                "parent_claim_index": 0,
                "parent_claim": "A and B.",
                "source": "A. B.",
                "atomic_claim_index": 1,
                "atomic_claim_count": 2,
                "claim": "B.",
                "judge_status": "ok",
                "llm_label": "contradicted",
                "llm_confidence": 0.7,
                "llm_evidence_quotes": [{"sentence_index": 1, "quote": "B."}],
                "evidence": [{"sentence_index": 1, "text": "B."}],
            },
        ]
        result = aggregate_atomic_judgments(rows)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["aggregate_label"], "contradicted")
        self.assertEqual(result[0]["aggregate_confidence"], 0.7)
        self.assertEqual(len(result[0]["atomic_judgments"]), 2)
        self.assertEqual(len(result[0]["llm_evidence_quotes"]), 2)


if __name__ == "__main__":
    unittest.main()
