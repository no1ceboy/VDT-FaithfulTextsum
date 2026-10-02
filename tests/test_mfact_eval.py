from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from src.evaluate.mfact_eval import MFactEvaluator, _faithful_class_index


class MFactEvaluatorTests(unittest.TestCase):
    def test_released_binary_label_defaults_to_class_one(self) -> None:
        self.assertEqual(_faithful_class_index(SimpleNamespace(num_labels=2, label2id={})), 1)

    def test_named_faithful_label_is_respected(self) -> None:
        config = SimpleNamespace(num_labels=2, label2id={"hallucinated": 0, "faithful": 1})
        self.assertEqual(_faithful_class_index(config), 1)

    def test_scores_source_summary_pairs_with_faithful_probability(self) -> None:
        evaluator = MFactEvaluator.__new__(MFactEvaluator)
        evaluator.device = "cpu"
        evaluator.batch_size = 2
        evaluator.max_length = 512
        evaluator._faithful_idx = 1
        evaluator._tokenizer = MagicMock(
            side_effect=lambda sources, summaries, **_: {
                "input_ids": torch.ones((len(sources), 8), dtype=torch.long),
                "attention_mask": torch.ones((len(sources), 8), dtype=torch.long),
            }
        )
        evaluator._model = MagicMock(
            return_value=SimpleNamespace(
                logits=torch.tensor([[0.0, 2.0], [2.0, 0.0]])
            )
        )

        rows = [
            {"source": "Nguồn một.", "summary": "Tóm tắt một."},
            {"source": "Nguồn hai.", "summary": "Tóm tắt hai."},
        ]
        scored = evaluator._evaluate(rows, "source", "summary")

        self.assertAlmostEqual(scored[0]["mfact_score"], 0.880797, places=5)
        self.assertEqual(scored[0]["mfact_pred"], 1)
        self.assertAlmostEqual(scored[1]["mfact_score"], 0.119203, places=5)
        self.assertEqual(scored[1]["mfact_pred"], 0)
        evaluator._tokenizer.assert_called_once()


if __name__ == "__main__":
    unittest.main()
