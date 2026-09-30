from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.training.run_utils import REPO_ROOT
from src.training.train_grpo import _check_local_inputs, _parse_args


class TrainingCliTests(unittest.TestCase):
    def _args(self, root: Path, extra: list[str] | None = None):
        model = root / "model"
        factcc = root / "factcc"
        model.mkdir(exist_ok=True)
        factcc.mkdir(exist_ok=True)
        train = root / "train.jsonl"
        train.write_text("{}\n", encoding="utf-8")
        argv = [
            "train_grpo",
            "--model", str(model),
            "--train_jsonl", str(train),
            "--output_dir", str(root / "run"),
            "--faithfulness_metrics", "factcc",
            "--factcc_model_path", str(factcc),
        ]
        if extra:
            argv.extend(extra)
        with patch.object(sys, "argv", argv):
            return _parse_args()

    def test_tensorboard_and_vllm_are_configurable_with_safe_vllm_default(self) -> None:
        with tempfile.TemporaryDirectory(prefix="training-cli-defaults-") as temporary:
            args = self._args(Path(temporary))
            self.assertEqual(args.report_to, "tensorboard")
            self.assertFalse(args.use_vllm)
            self.assertEqual(args.vllm_mode, "colocate")
            self.assertEqual(args.finetuning_method, "lora")

        with tempfile.TemporaryDirectory(prefix="training-cli-vllm-") as temporary:
            args = self._args(Path(temporary), ["--use_vllm", "--vllm_gpu_memory_utilization", "0.2"])
            self.assertTrue(args.use_vllm)
            self.assertEqual(args.vllm_gpu_memory_utilization, 0.2)

    def test_finetuning_modes_and_qlora_settings_parse(self) -> None:
        for method in ("lora", "qlora", "fft"):
            with self.subTest(method=method), tempfile.TemporaryDirectory(prefix="training-cli-method-") as temporary:
                args = self._args(Path(temporary), ["--finetuning_method", method])
                self.assertEqual(args.finetuning_method, method)

        with tempfile.TemporaryDirectory(prefix="training-cli-qlora-") as temporary:
            args = self._args(
                Path(temporary),
                ["--finetuning_method", "qlora", "--qlora_quant_type", "fp4", "--no-qlora_double_quant"],
            )
            self.assertEqual(args.qlora_quant_type, "fp4")
            self.assertFalse(args.qlora_double_quant)

    def test_run_output_must_resolve_inside_repository(self) -> None:
        with tempfile.TemporaryDirectory(prefix="training-cli-outside-") as temporary:
            args = self._args(Path(temporary))
            with self.assertRaisesRegex(ValueError, "inside the repository"):
                _check_local_inputs(args)

    def test_run_paths_inside_repository_and_metric_weights_parse(self) -> None:
        with tempfile.TemporaryDirectory(prefix="training-cli-inside-", dir=REPO_ROOT) as temporary:
            root = Path(temporary)
            args = self._args(root, ["--metric_weights", "0.7"])
            model, base, train, validation, output = _check_local_inputs(args)
            self.assertEqual(model, base)
            self.assertEqual(train, root / "train.jsonl")
            self.assertIsNone(validation)
            self.assertEqual(output, root / "run")
            self.assertEqual(args.metric_weights, [0.7])

    def test_internal_validation_can_be_enabled_without_external_eval_file(self) -> None:
        with tempfile.TemporaryDirectory(prefix="training-cli-internal-validation-", dir=REPO_ROOT) as temporary:
            root = Path(temporary)
            args = self._args(
                root,
                ["--internal_validation_fraction", "0.1", "--eval_strategy", "epoch"],
            )
            _, _, _, validation, _ = _check_local_inputs(args)
            self.assertIsNone(validation)
            self.assertEqual(args.internal_validation_fraction, 0.1)

    def test_internal_validation_requires_an_evaluation_strategy(self) -> None:
        with tempfile.TemporaryDirectory(prefix="training-cli-internal-validation-no-eval-", dir=REPO_ROOT) as temporary:
            root = Path(temporary)
            args = self._args(root, ["--internal_validation_fraction", "0.1"])
            with self.assertRaisesRegex(ValueError, "requires --eval_strategy"):
                _check_local_inputs(args)

    def test_existing_adapter_rejects_non_lora_modes(self) -> None:
        with tempfile.TemporaryDirectory(prefix="training-cli-adapter-", dir=REPO_ROOT) as temporary:
            root = Path(temporary)
            adapter = root / "adapter"
            adapter.mkdir()
            (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
            base = root / "base"
            base.mkdir()
            train = root / "train.jsonl"
            train.write_text("{}\n", encoding="utf-8")
            argv = [
                "train_grpo", "--model", str(adapter), "--base_model", str(base),
                "--train_jsonl", str(train), "--output_dir", str(root / "run"),
                "--faithfulness_metrics", "factcc", "--factcc_model_path", str(base),
                "--finetuning_method", "fft",
            ]
            with patch.object(sys, "argv", argv):
                args = _parse_args()
            with self.assertRaisesRegex(ValueError, "supported only"):
                _check_local_inputs(args)


if __name__ == "__main__":
    unittest.main()
