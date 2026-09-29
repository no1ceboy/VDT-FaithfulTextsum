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
        (row,) = read_records(project_root / "data" / "sample.jsonl")
        validate_prepared_record(row)
        self.assertEqual(row["id"], "7")
        self.assertNotIn(row["reference"], " ".join(message["content"] for message in row["prompt"]))

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
            self.assertEqual(result.returncode, 0, result.stderr)
            manifest = json.loads((output_dir / "data_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["records_train"], 1)
            self.assertEqual(manifest["records_validation"], 0)
            self.assertEqual(manifest["records_test"], 0)
            self.assertEqual(len(read_records(output_dir / "train.jsonl", "source", "reference")), 1)
            self.assertTrue((output_dir / "validation.jsonl").exists())
            self.assertTrue((output_dir / "test.jsonl").exists())

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


if __name__ == "__main__":
    unittest.main()
