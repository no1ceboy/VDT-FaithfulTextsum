from __future__ import annotations

import unittest

from src.evaluate.fact_audit import (
    audit_records,
    evidence_overlap,
    retrieve_evidence,
    split_claims,
)


class FactAuditTests(unittest.TestCase):
    def test_claim_splitting_is_deterministic_and_supports_clauses(self) -> None:
        text = "Một sự kiện xảy ra năm 2020. Sự kiện khác xảy ra năm 2021; ở Hà Nội."
        self.assertEqual(split_claims(text), [
            "Một sự kiện xảy ra năm 2020.",
            "Sự kiện khác xảy ra năm 2021; ở Hà Nội.",
        ])
        self.assertEqual(split_claims(text, split_clauses=True), [
            "Một sự kiện xảy ra năm 2020.",
            "Sự kiện khác xảy ra năm 2021",
            "ở Hà Nội.",
        ])

    def test_retrieval_ranks_matching_source_sentence(self) -> None:
        source = "Trời mưa vào buổi sáng. Thành phố tổ chức lễ hội vào tháng sáu."
        evidence = retrieve_evidence(source, "Thành phố tổ chức lễ hội.", top_k=2)
        self.assertEqual(evidence[0]["sentence_index"], 1)
        self.assertGreater(evidence[0]["retrieval_score"], evidence[1]["retrieval_score"])

    def test_overlap_is_bounded(self) -> None:
        scores = evidence_overlap("Một thông tin", "Một thông tin khác")
        self.assertTrue(all(0.0 <= value <= 1.0 for value in scores.values()))

    def test_audit_keeps_claim_rows_and_flags_missing_numbers(self) -> None:
        records = [{
            "id": "7",
            "source": "Sự kiện diễn ra năm 2020 tại Hà Nội.",
            "human_sum": "Sự kiện diễn ra năm 2020 tại Hà Nội.",
            "grpo_sum": "Sự kiện diễn ra năm 2021 tại Hà Nội.",
        }]

        def fake_score(sources: list[str], claims: list[str]) -> list[float]:
            del sources
            return [0.9 if "2020" in claim else 0.2 for claim in claims]

        rows, aggregates = audit_records(
            records,
            "source",
            ["human_sum", "grpo_sum"],
            score_fn=fake_score,
            top_k=1,
        )

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["source"], records[0]["source"])
        self.assertEqual(rows[0]["status"], "likely_supported")
        self.assertEqual(rows[1]["status"], "needs_review")
        self.assertIn("claim_number_or_date_not_found", rows[1]["flags"])
        self.assertEqual(aggregates[0]["likely_supported"], 1)
        self.assertEqual(aggregates[1]["needs_review"], 1)

    def test_hybrid_retrieval_records_lexical_and_embedding_scores(self) -> None:
        class FakeEmbeddingRetriever:
            def score_sentences(self, sentences, query):
                del query
                return [0.1 + 0.2 * index for index, _ in enumerate(sentences)]

        evidence = retrieve_evidence(
            "Alpha event. Beta event.",
            "Beta event.",
            top_k=2,
            retrieval_mode="hybrid",
            embedding_retriever=FakeEmbeddingRetriever(),
            embedding_weight=0.5,
        )
        self.assertEqual(evidence[0]["sentence_index"], 1)
        self.assertEqual(evidence[0]["retrieval_method"], "hybrid")
        self.assertIn("lexical_retrieval_score", evidence[0])
        self.assertIn("embedding_score", evidence[0])
        self.assertIn("embedding_retrieval_score", evidence[0])


if __name__ == "__main__":
    unittest.main()
