from __future__ import annotations

import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.evaluate._alignscore_compat import (
    _AlignScoreModel,
    _aggregate_sentence_scores,
    _group_source_sentences,
    _load_backbone_assets,
    _unwrap_state_dict,
)


class AlignScoreCompatibilityTests(unittest.TestCase):
    def test_alignscore_model_initializes_from_local_pretrained_roberta(self) -> None:
        config = SimpleNamespace(hidden_size=4)
        base_model = MagicMock()
        base_model.config = config
        with patch(
            "src.evaluate._alignscore_compat.RobertaModel.from_pretrained",
            return_value=base_model,
        ) as load_base:
            model = _AlignScoreModel(config, "models/roberta-base", None, offline=True)

        self.assertEqual(model.bin_layer.weight.shape, (2, 4))
        self.assertEqual(load_base.call_args.args[0], "models/roberta-base")
        self.assertTrue(load_base.call_args.kwargs["local_files_only"])
        self.assertTrue(load_base.call_args.kwargs["add_pooling_layer"])

    def test_source_chunking_preserves_sentences_and_order(self) -> None:
        sentences = [f"Sentence {index}." for index in range(6)]
        source = " ".join(["word"] * 700)

        chunks = _group_source_sentences(source, sentences)

        self.assertEqual(chunks, ["Sentence 0. Sentence 1.", "Sentence 2. Sentence 3.", "Sentence 4. Sentence 5."])

    def test_empty_source_gets_one_empty_chunk(self) -> None:
        self.assertEqual(_group_source_sentences("", []), [""])

    def test_sp_aggregation_uses_best_evidence_per_claim_then_mean(self) -> None:
        # Row-major pairs: source chunk 0 against all claims, then chunk 1.
        self.assertAlmostEqual(
            _aggregate_sentence_scores([0.1, 0.5, 0.7, 0.3], source_chunk_count=2, claim_sentence_count=2),
            0.6,
        )

    def test_lightning_checkpoint_and_plain_state_dict_are_both_accepted(self) -> None:
        import torch

        tensors = {"weight": torch.ones(1)}
        self.assertEqual(_unwrap_state_dict({"state_dict": tensors}), tensors)
        self.assertEqual(_unwrap_state_dict(tensors), tensors)

    def test_module_prefix_from_distributed_checkpoints_is_removed(self) -> None:
        import torch

        tensors = {"module.weight": torch.ones(1)}
        self.assertIn("weight", _unwrap_state_dict(tensors))

    def test_cache_directory_is_passed_to_local_backbone_loaders(self) -> None:
        from src.evaluate._alignscore_compat import AlignScoreCompatScorer

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint_path = root / "AlignScore-base.ckpt"
            checkpoint_path.touch()
            cache_dir = root / "hf-cache"
            cache_dir.mkdir()
            tokenizer = SimpleNamespace(model_max_length=512)
            model = MagicMock()
            model.load_state_dict.return_value = ([], [])
            model.to.return_value = model
            model.eval.return_value = model

            with (
                patch("src.evaluate._alignscore_compat.AutoTokenizer.from_pretrained", return_value=tokenizer) as load_tokenizer,
                patch("src.evaluate._alignscore_compat.RobertaConfig.from_pretrained", return_value=object()) as load_config,
                patch("src.evaluate._alignscore_compat._AlignScoreModel", return_value=model),
                patch("src.evaluate._alignscore_compat.torch.load", return_value={}),
                patch.dict("os.environ", {"HF_HUB_OFFLINE": "1"}),
            ):
                AlignScoreCompatScorer(
                    ckpt_path=checkpoint_path,
                    cache_dir=cache_dir,
                    device="cpu",
                )

            expected_cache = str(cache_dir.resolve())
            self.assertEqual(load_tokenizer.call_args.args[0], "FacebookAI/roberta-base")
            self.assertEqual(load_config.call_args.args[0], "FacebookAI/roberta-base")
            self.assertEqual(load_tokenizer.call_args.kwargs["cache_dir"], expected_cache)
            self.assertEqual(load_config.call_args.kwargs["cache_dir"], expected_cache)
            self.assertTrue(load_tokenizer.call_args.kwargs["local_files_only"])
            self.assertTrue(load_config.call_args.kwargs["local_files_only"])

    def test_backbone_loader_falls_back_to_legacy_cache_name(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as temporary:
            tokenizer = object()
            config = object()
            with (
                patch(
                    "src.evaluate._alignscore_compat.AutoTokenizer.from_pretrained",
                    side_effect=[OSError("canonical cache miss"), tokenizer],
                ) as load_tokenizer,
                patch(
                    "src.evaluate._alignscore_compat.RobertaConfig.from_pretrained",
                    return_value=config,
                ) as load_config,
            ):
                actual_tokenizer, actual_config, reference = _load_backbone_assets(
                    "FacebookAI/roberta-base", temporary, offline=True
                )

            self.assertIs(actual_tokenizer, tokenizer)
            self.assertIs(actual_config, config)
            self.assertEqual(reference, "roberta-base")
            self.assertEqual(load_tokenizer.call_args_list[0].args[0], "FacebookAI/roberta-base")
            self.assertEqual(load_tokenizer.call_args_list[1].args[0], "roberta-base")
            self.assertEqual(load_config.call_args.args[0], "roberta-base")

    def test_local_backbone_folder_is_used_directly(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tokenizer = object()
            config = object()
            with (
                patch(
                    "src.evaluate._alignscore_compat.AutoTokenizer.from_pretrained",
                    return_value=tokenizer,
                ) as load_tokenizer,
                patch(
                    "src.evaluate._alignscore_compat.RobertaConfig.from_pretrained",
                    return_value=config,
                ) as load_config,
            ):
                actual_tokenizer, actual_config, reference = _load_backbone_assets(
                    str(root), None, offline=False
                )

            self.assertIs(actual_tokenizer, tokenizer)
            self.assertIs(actual_config, config)
            self.assertEqual(reference, str(root.resolve()))
            self.assertTrue(load_tokenizer.call_args.kwargs["local_files_only"])
            self.assertTrue(load_config.call_args.kwargs["local_files_only"])

    def test_tiny_local_roberta_and_alignscore_checkpoint_run_end_to_end(self) -> None:
        import gc
        import torch
        from transformers import RobertaConfig, RobertaModel
        from src.evaluate._alignscore_compat import AlignScoreCompatScorer

        class TinyTokenizer:
            model_max_length = 8

            def __call__(self, contexts, claims, **kwargs):
                del claims, kwargs
                input_ids = torch.ones((len(contexts), self.model_max_length), dtype=torch.long)
                return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}

        with tempfile.TemporaryDirectory(prefix="vdt-alignscore-smoke-") as temporary:
            root = Path(temporary)
            backbone = root / "roberta-base"
            backbone.mkdir()
            config = RobertaConfig(
                vocab_size=32,
                hidden_size=16,
                num_hidden_layers=1,
                num_attention_heads=4,
                intermediate_size=32,
                max_position_embeddings=514,
            )
            config.save_pretrained(backbone)
            torch.save(RobertaModel(config).state_dict(), backbone / "pytorch_model.bin")
            (backbone / "tokenizer.json").touch()
            model = _AlignScoreModel(
                config,
                str(backbone),
                None,
                offline=True,
            )
            checkpoint_path = root / "AlignScore-base.ckpt"
            torch.save({"state_dict": model.state_dict()}, checkpoint_path)

            with (
                patch(
                    "src.evaluate._alignscore_compat.AutoTokenizer.from_pretrained",
                    return_value=TinyTokenizer(),
                ),
                patch.dict("os.environ", {"HF_HUB_OFFLINE": "1"}),
            ):
                scorer = AlignScoreCompatScorer(
                    ckpt_path=checkpoint_path,
                    model=str(backbone),
                    device="cpu",
                    evaluation_mode="nli",
                )
                score = scorer.score(["source text"], ["summary text"])

            self.assertEqual(len(score), 1)
            self.assertGreaterEqual(score[0], 0.0)
            self.assertLessEqual(score[0], 1.0)
            del scorer, model
            gc.collect()


if __name__ == "__main__":
    unittest.main()
