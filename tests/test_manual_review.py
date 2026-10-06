from __future__ import annotations

import unittest

from src.evaluate.manual_review import LABELS, build_review_rows, select_document_ids


class ManualReviewTests(unittest.TestCase):
    def test_non_proposition_label_is_available(self) -> None:
        self.assertIn("not_a_claim", LABELS)

    def test_selection_is_deterministic_and_balanced(self) -> None:
        rows = [
            {"id": "flag-1", "flags": ["low_lexical_evidence"]},
            {"id": "flag-2", "flags": ["claim_number_or_date_not_found"]},
            {"id": "flag-3", "flags": ["claim_number_or_date_not_found"]},
            {"id": "ok-1", "flags": []},
            {"id": "ok-2", "flags": []},
            {"id": "ok-3", "flags": []},
        ]
        selected = select_document_ids(rows, document_count=4, seed=42)
        self.assertEqual(selected, select_document_ids(rows, document_count=4, seed=42))
        self.assertEqual(sum(record_id.startswith("flag-") for record_id in selected), 2)
        self.assertEqual(len(selected), 4)

    def test_review_rows_keep_evidence_and_blank_labels(self) -> None:
        data = [{"id": "1", "input": "Nguồn.", "output": "Tóm tắt."}]
        audit = [
            {
                "id": "1",
                "summary_col": "output",
                "claim_index": 0,
                "claim_count": 1,
                "claim": "Tóm tắt.",
                "evidence": [{"text": "Nguồn.", "retrieval_score": 0.8}],
                "flags": [],
            }
        ]
        [row] = build_review_rows(data, audit, ["1"])
        self.assertEqual(row["source"], "Nguồn.")
        self.assertEqual(row["human_summary"], "Tóm tắt.")
        self.assertEqual(row["retrieval_score"], 0.8)
        self.assertEqual(row["human_label"], "")


if __name__ == "__main__":
    unittest.main()
