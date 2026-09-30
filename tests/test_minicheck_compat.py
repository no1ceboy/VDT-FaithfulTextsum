from __future__ import annotations

import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from src.evaluate._minicheck_compat import (
    MiniCheckCompatScorer,
    _aggregate_claim_support,
    _chunk_source,
)


class MiniCheckCompatibilityTests(unittest.TestCase):
    def test_source_chunking_keeps_sentence_order_and_respects_word_budget(self) -> None:
        with patch(
            "src.evaluate._minicheck_compat._source_sentences",
            return_value=["one two.", "three four.", "five six."],
        ):
            chunks = _chunk_source("ignored", chunk_words=3)

        self.assertEqual(chunks, ["one two.", "three four.", "five six."])

    def test_fusion_takes_best_source_chunk_per_claim_then_worst_claim(self) -> None:
        self.assertAlmostEqual(
            _aggregate_claim_support(
                [0.1, 0.8, 0.9, 0.2], source_chunk_count=2, claim_sentence_count=2
            ),
            0.8,
        )

    def test_score_returns_sentence_fusion_result_for_one_record(self) -> None:
        scorer = object.__new__(MiniCheckCompatScorer)
        scorer.batch_size = 2
        scorer.device = torch.device("cpu")
        scorer.max_length = 2048
        scorer.tokenizer = SimpleNamespace(eos_token="</s>")

        with (
            patch(
                "src.evaluate._minicheck_compat._source_sentences",
                side_effect=[["source A", "source B"], ["claim X", "claim Y"]],
            ),
            patch.object(scorer, "_score_pairs", return_value=[0.1, 0.8, 0.9, 0.2]),
        ):
            predictions, scores, used_chunks, pair_matrices = scorer.score(
                docs=["document"], claims=["claim"], chunk_size=1
            )

        self.assertEqual(predictions, [1])
        self.assertEqual(scores, [0.8])
        self.assertEqual(used_chunks, [["source A", "source B"]])
        self.assertEqual(pair_matrices, [[[0.1, 0.8], [0.9, 0.2]]])

    def test_only_bundled_flan_t5_variant_is_accepted(self) -> None:
        with self.assertRaisesRegex(ValueError, "supports only 'flan-t5-large'"):
            MiniCheckCompatScorer(model_name="roberta-large")

    def test_offline_mode_uses_explicit_hugging_face_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cache_dir = Path(temporary)
            tokenizer = SimpleNamespace(eos_token="</s>")
            model = MagicMock()
            model.to.return_value = model
            model.eval.return_value = model

            with (
                patch("src.evaluate._minicheck_compat.AutoTokenizer.from_pretrained", return_value=tokenizer) as load_tokenizer,
                patch("src.evaluate._minicheck_compat.AutoModelForSeq2SeqLM.from_pretrained", return_value=model) as load_model,
                patch.dict("os.environ", {"HF_HUB_OFFLINE": "1"}),
            ):
                MiniCheckCompatScorer(cache_dir=cache_dir, device="cpu")

            self.assertEqual(load_tokenizer.call_args.args[0], "lytang/MiniCheck-Flan-T5-Large")
            self.assertEqual(load_model.call_args.args[0], "lytang/MiniCheck-Flan-T5-Large")
            self.assertEqual(load_tokenizer.call_args.kwargs["cache_dir"], str(cache_dir))
            self.assertEqual(load_model.call_args.kwargs["cache_dir"], str(cache_dir))
            self.assertTrue(load_tokenizer.call_args.kwargs["local_files_only"])
            self.assertTrue(load_model.call_args.kwargs["local_files_only"])

    def test_extracted_model_folder_is_loaded_without_hub_lookup(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            model_dir = Path(temporary)
            for filename in ("config.json", "model.safetensors", "tokenizer.json"):
                (model_dir / filename).touch()
            tokenizer = SimpleNamespace(eos_token="</s>")
            model = MagicMock()
            model.to.return_value = model
            model.eval.return_value = model

            with (
                patch("src.evaluate._minicheck_compat.AutoTokenizer.from_pretrained", return_value=tokenizer) as load_tokenizer,
                patch("src.evaluate._minicheck_compat.AutoModelForSeq2SeqLM.from_pretrained", return_value=model) as load_model,
                patch.dict("os.environ", {}, clear=True),
            ):
                MiniCheckCompatScorer(cache_dir=model_dir, device="cpu")

            self.assertEqual(load_tokenizer.call_args.args[0], str(model_dir.resolve()))
            self.assertEqual(load_model.call_args.args[0], str(model_dir.resolve()))
            self.assertIsNone(load_tokenizer.call_args.kwargs["cache_dir"])
            self.assertTrue(load_tokenizer.call_args.kwargs["local_files_only"])


if __name__ == "__main__":
    unittest.main()
