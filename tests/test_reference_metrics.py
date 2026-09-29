from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

from src.evaluate.bertscore_eval import BertScoreEvaluator
from src.evaluate.rouge_eval import RougeEvaluator, rouge_scores


class ReferenceMetricTests(unittest.TestCase):
    def test_rouge_scores_unicode_tokens_and_overlap(self) -> None:
        scores = rouge_scores("Hà Nội has two lakes", "HÀ NỘI has three lakes")

        self.assertAlmostEqual(scores["rouge1_f1"], 0.8)
        self.assertAlmostEqual(scores["rouge2_f1"], 0.5)
        self.assertAlmostEqual(scores["rougeL_f1"], 0.8)

    def test_rouge_normalizes_decomposed_vietnamese_diacritics(self) -> None:
        scores = rouge_scores("đường phố", "đường phố")

        self.assertEqual(scores["rouge1_f1"], 1.0)

    def test_rouge_evaluator_uses_reference_and_skips_reference_self_score(self) -> None:
        evaluator = RougeEvaluator(reference_col="human_sum")
        rows = [
            {"input": "source", "human_sum": "a b c", "llm_sum": "a b x"},
        ]

        human = evaluator._evaluate(rows, "input", "human_sum")
        llm = evaluator._evaluate(rows, "input", "llm_sum")

        self.assertIsNone(human[0]["rouge_score"])
        self.assertAlmostEqual(llm[0]["rouge_score"], 2 / 3)
        self.assertAlmostEqual(llm[0]["rouge_1_f1"], 2 / 3)

    def test_bertscore_alignment_is_one_for_identical_token_embeddings(self) -> None:
        evaluator = object.__new__(BertScoreEvaluator)
        embeddings = torch.tensor(
            [[[1.0, 0.0], [0.0, 1.0]]], dtype=torch.float32
        )
        mask = torch.tensor([[True, True]])

        with patch.object(evaluator, "_encode", side_effect=[(embeddings, mask), (embeddings, mask)]):
            scores = evaluator._score_batch(["candidate"], ["reference"])

        self.assertEqual(len(scores), 1)
        for value in scores[0]:
            self.assertAlmostEqual(value, 1.0)

    def test_bertscore_evaluator_leaves_reference_self_score_null(self) -> None:
        evaluator = object.__new__(BertScoreEvaluator)
        evaluator.reference_col = "human_sum"
        evaluator.batch_size = 2
        rows = [{"input": "source", "human_sum": "reference", "llm_sum": "candidate"}]

        with patch.object(evaluator, "_score_batch", return_value=[(0.7, 0.8, 0.7466667)]) as score:
            human = evaluator._evaluate(rows, "input", "human_sum")
            llm = evaluator._evaluate(rows, "input", "llm_sum")

        score.assert_called_once_with(["candidate"], ["reference"])
        self.assertIsNone(human[0]["bertscore_score"])
        self.assertAlmostEqual(llm[0]["bertscore_score"], 0.746667, places=6)


if __name__ == "__main__":
    unittest.main()
