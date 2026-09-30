from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from pathlib import Path

from src.training.grpo_data import (
    build_prompt,
    read_records,
    read_records_from_files,
    infer_columns_from_file,
    split_records,
    validate_prepared_record,
)
from src.training.grpo_rewards import (
    char_ngram_fbeta,
    completion_text,
    make_metric_reward,
    reference_char_reward,
)


class GrpoDataTests(unittest.TestCase):
    def test_human_reference_is_not_in_prompt(self) -> None:
        secret_reference = "A reference that must not leak into the prompt."
        prompt = build_prompt("Nguồn thông tin.", "daily")
        self.assertNotIn(secret_reference, " ".join(message["content"] for message in prompt))
        self.assertIn("Nguồn thông tin.", prompt[-1]["content"])
        self.assertIn("daily", prompt[-1]["content"])

    def test_included_example_conforms_to_prepared_schema(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        (row,) = read_records(project_root / "data" / "sample.jsonl", reference_col="abstract_sum")
        validate_prepared_record(row)
        self.assertEqual(row["id"], "7")
        self.assertNotIn(row["reference"], " ".join(message["content"] for message in row["prompt"]))

    def test_missing_reference_column_reports_available_sample_field(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        sample = project_root / "data" / "sample.jsonl"
        with self.assertRaisesRegex(ValueError, "Possible reference field: abstract_sum"):
            read_records(sample)

    def test_prepare_cli_combines_multiple_files_before_splitting(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        script = project_root / "scripts" / "prepare_grpo_data.py"
        env = dict(os.environ)
        env["PYTHONPATH"] = str(project_root)
        with tempfile.TemporaryDirectory(prefix="vdt-grpo-multi-input-") as temporary:
            temporary_path = Path(temporary)
            first_path = temporary_path / "part-a.jsonl"
            second_path = temporary_path / "part-b.jsonl"
            first_rows = [
                {"id": str(index), "input": f"source {index}", "human_sum": f"summary {index}"}
                for index in range(1, 6)
            ]
            second_rows = [
                {"id": str(index), "input": f"source {index + 5}", "human_sum": f"summary {index + 5}"}
                for index in range(1, 6)
            ]
            first_rows[0]["input"] = "shared source"
            second_rows[0]["input"] = " shared   source "
            first_path.write_text(
                "".join(json.dumps(row) + "\n" for row in first_rows), encoding="utf-8"
            )
            second_path.write_text(
                "".join(json.dumps(row) + "\n" for row in second_rows), encoding="utf-8"
            )
            output_dir = temporary_path / "splits"
            result = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--input",
                    str(first_path),
                    str(second_path),
                    "--output_dir",
                    str(output_dir),
                    "--allow_external_output",
                    "--validation_fraction",
                    "0.2",
                    "--test_fraction",
                    "0.2",
                ],
                cwd=project_root,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            manifest = json.loads((output_dir / "data_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(len(manifest["input_files"]), 2)
            self.assertEqual(manifest["records_total"], 10)
            combined = read_records_from_files(
                [first_path, second_path], source_col="input", reference_col="human_sum"
            )
            self.assertEqual(len({row["id"] for row in combined}), 10)
            self.assertEqual({row["source_file"] for row in combined}, {"part-a.jsonl", "part-b.jsonl"})
            splits = {
                split_name: read_records(output_dir / f"{split_name}.jsonl", "source", "reference")
                for split_name in ("train", "validation", "test")
            }
            shared_source_splits = [
                split_name
                for split_name, rows in splits.items()
                if any(" ".join(row["source"].split()) == "shared source" for row in rows)
            ]
            self.assertEqual(len(shared_source_splits), 1)
            self.assertTrue(
                all(":" in row["id"] for rows in splits.values() for row in rows)
            )

    def test_prepare_cli_smoke_test_writes_manifest_and_split_files(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        script = project_root / "scripts" / "prepare_grpo_data.py"
        sample = project_root / "data" / "sample.jsonl"
        env = dict(os.environ)
        env["PYTHONPATH"] = str(project_root)
        with tempfile.TemporaryDirectory(prefix="vdt-grpo-test-") as temporary:
            output_dir = Path(temporary) / "prepared"
            result = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--input",
                    str(sample),
                    "--reference_col",
                    "abstract_sum",
                    "--output_dir",
                    str(output_dir),
                    "--allow_external_output",
                    "--validation_fraction",
                    "0",
                    "--test_fraction",
                    "0",
                ],
                cwd=project_root,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            manifest = json.loads((output_dir / "data_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["records_train"], 1)
            self.assertEqual(manifest["records_validation"], 0)
            self.assertEqual(manifest["records_test"], 0)
            self.assertEqual(len(read_records(output_dir / "train.jsonl", "source", "reference")), 1)
            self.assertTrue((output_dir / "validation.jsonl").exists())
            self.assertTrue((output_dir / "test.jsonl").exists())

    def test_canonical_source_summary_data_is_auto_detected_without_mutating_input(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        script = project_root / "scripts" / "prepare_grpo_data.py"
        env = dict(os.environ)
        env["PYTHONPATH"] = str(project_root)
        with tempfile.TemporaryDirectory(prefix="vdt-canonical-data-") as temporary:
            temporary_path = Path(temporary)
            input_path = temporary_path / "human_data.jsonl"
            rows = [
                {"id": str(index), "source": f"document {index}", "summary": f"summary {index}"}
                for index in range(12)
            ]
            input_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            original_bytes = input_path.read_bytes()
            output_dir = temporary_path / "prepared"
            result = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--input",
                    str(input_path),
                    "--output_dir",
                    str(output_dir),
                    "--allow_external_output",
                    "--validation_fraction",
                    "0.2",
                    "--test_fraction",
                    "0.2",
                ],
                cwd=project_root,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            manifest = json.loads((output_dir / "data_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["source_column"], "source")
            self.assertEqual(manifest["reference_column"], "summary")
            self.assertTrue((output_dir / "train.jsonl").is_file())
            self.assertTrue((output_dir / "validation.jsonl").is_file())
            self.assertTrue((output_dir / "test.jsonl").is_file())
            self.assertEqual(input_path.read_bytes(), original_bytes)
            self.assertEqual(infer_columns_from_file(input_path), ("source", "summary"))

    def test_prepare_cli_refuses_external_output_without_explicit_override(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        script = project_root / "scripts" / "prepare_grpo_data.py"
        sample = project_root / "data" / "sample.jsonl"
        env = dict(os.environ)
        env["PYTHONPATH"] = str(project_root)
        with tempfile.TemporaryDirectory(prefix="vdt-grpo-external-guard-") as temporary:
            output_dir = Path(temporary) / "must-not-be-created"
            result = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--input",
                    str(sample),
                    "--reference_col",
                    "abstract_sum",
                    "--output_dir",
                    str(output_dir),
                    "--validation_fraction",
                    "0",
                    "--test_fraction",
                    "0",
                ],
                cwd=project_root,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("Refusing output outside the repository", result.stderr)
            self.assertFalse(output_dir.exists())

    def test_duplicate_sources_stay_in_same_split(self) -> None:
        rows = [
            {"id": "a", "source": "same   source", "reference": "r1"},
            {"id": "b", "source": "same source", "reference": "r2"},
            {"id": "c", "source": "another source", "reference": "r3"},
            {"id": "d", "source": "third source", "reference": "r4"},
            {"id": "e", "source": "fourth source", "reference": "r5"},
        ]
        train, validation, test = split_records(rows, 0.2, 0.2, 13)
        assignment = {
            row["id"]: split_name
            for split_name, split in (("train", train), ("validation", validation), ("test", test))
            for row in split
        }
        self.assertEqual(assignment["a"], assignment["b"])
        self.assertTrue(train)
        self.assertTrue(validation)
        self.assertTrue(test)

    def test_one_document_cannot_make_heldout_split(self) -> None:
        rows = [{"id": "1", "source": "one", "reference": "summary"}]
        with self.assertRaisesRegex(ValueError, "distinct source documents"):
            split_records(rows, 0.1, 0.1, 42)
        train, validation, test = split_records(rows, 0.0, 0.0, 42)
        self.assertEqual(len(train), 1)
        self.assertFalse(validation)
        self.assertFalse(test)


class GrpoRewardTests(unittest.TestCase):
    def test_unicode_normalization_and_whitespace(self) -> None:
        self.assertEqual(char_ngram_fbeta("đúng   dữ kiện", "đúng dữ kiện"), 1.0)
        self.assertEqual(char_ngram_fbeta("", "reference"), 0.0)

    def test_completion_formats_and_reference_reward(self) -> None:
        self.assertEqual(completion_text("  summary  "), "summary")
        self.assertEqual(
            completion_text([{"role": "assistant", "content": " summary "}]),
            "summary",
        )
        scores = reference_char_reward(
            completions=[[{"role": "assistant", "content": "Bản tóm tắt."}]],
            reference=["Bản tóm tắt."],
        )
        self.assertEqual(scores, [1.0])

    def test_minicheck_reward_uses_shared_cache_and_fixed_model_default(self) -> None:
        cache_dir = "models/hf-cache"
        with patch("src.evaluate.minicheck_eval.MiniCheckEvaluator") as evaluator_type:
            evaluator_type.return_value.evaluate.return_value = [{"minicheck_score": 0.75}]
            reward = make_metric_reward("minicheck", hf_cache_dir=cache_dir)
            scores = reward(["generated summary"], source=["source document"])

        self.assertEqual(scores, [0.75])
        evaluator_type.assert_called_once_with(device="cpu", batch_size=4, cache_dir=cache_dir)

    def test_alignscore_reward_passes_checkpoint_and_matching_local_backbone(self) -> None:
        checkpoint = "models/alignscore/AlignScore-base.ckpt"
        backbone = "models/roberta-base"
        with patch("src.evaluate.alignscore_eval.AlignScoreEvaluator") as evaluator_type:
            evaluator_type.return_value.evaluate.return_value = [{"alignscore_score": 0.6}]
            reward = make_metric_reward(
                "alignscore",
                alignscore_ckpt=checkpoint,
                alignscore_backbone_path=backbone,
            )
            scores = reward(["summary"], source=["source"])

        self.assertEqual(scores, [0.6])
        evaluator_type.assert_called_once_with(
            device="cpu",
            model_path=checkpoint,
            backbone_path=backbone,
            batch_size=4,
        )


if __name__ == "__main__":
    unittest.main()
