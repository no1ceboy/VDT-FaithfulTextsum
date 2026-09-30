from __future__ import annotations

import tempfile
import unittest
import json
from pathlib import Path
from unittest.mock import patch

from src.evaluate.run_eval import (
    _default_metrics,
    _cached_roberta_base,
    _hf_cache_root,
    _nltk_data_root,
    _project_path,
    _reference_column,
    _resolve_roberta_base,
    _source_column,
    main,
    _summary_columns,
    _validate_roberta_base_folder,
)


class EvalCliPathTests(unittest.TestCase):
    def test_factcc_relative_path_is_anchored_at_project_root(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        self.assertEqual(_project_path("models/factcc"), project_root / "models" / "factcc")

    def test_current_paired_data_schema_is_selected_by_default(self) -> None:
        rows = [{"input": "source", "human_sum": "human", "llm_sum": "candidate"}]
        self.assertEqual(_summary_columns(rows), ["human_sum", "llm_sum"])
        self.assertEqual(
            _default_metrics(rows, "human_sum", ["human_sum", "llm_sum"]),
            ["factcc", "minicheck", "alignscore", "rouge"],
        )

    def test_canonical_source_summary_schema_is_selected_by_default(self) -> None:
        rows = [{"id": "1", "source": "source", "summary": "summary"}]
        self.assertEqual(_source_column(rows), "source")
        self.assertEqual(_summary_columns(rows), ["summary"])
        self.assertEqual(_reference_column(rows), "summary")
        self.assertEqual(
            _default_metrics(rows, "summary", ["summary"]),
            ["factcc", "minicheck", "alignscore"],
        )

    def test_cli_accepts_canonical_source_summary_rows(self) -> None:
        row = {"id": "1", "source": "A source document.", "summary": "A short summary."}
        with tempfile.TemporaryDirectory(prefix="vdt-canonical-eval-") as temporary:
            root = Path(temporary)
            data_path = root / "data.jsonl"
            output_path = root / "scores.jsonl"
            data_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            with patch(
                "sys.argv",
                [
                    "run_eval",
                    "--data",
                    str(data_path),
                    "--metrics",
                    "rouge",
                    "--output",
                    str(output_path),
                ],
            ):
                self.assertEqual(main(), 0)
            result = json.loads(output_path.read_text(encoding="utf-8").splitlines()[0])
            self.assertIsNone(result["summary__rouge_score"])

    def test_legacy_abstract_sum_schema_remains_supported(self) -> None:
        self.assertEqual(_summary_columns([{"input": "source", "abstract_sum": "summary"}]), ["abstract_sum"])
        self.assertEqual(
            _default_metrics([{"abstract_sum": "summary"}], "human_sum", ["abstract_sum"]),
            ["factcc", "minicheck", "alignscore"],
        )

    def test_roberta_base_requires_weights_tokenizer_and_config(self) -> None:
        with tempfile.TemporaryDirectory(prefix="vdt-roberta-test-") as temporary:
            folder = Path(temporary)
            for filename in ("config.json", "model.safetensors", "tokenizer.json"):
                (folder / filename).touch()
            _validate_roberta_base_folder(folder)
            (folder / "model.safetensors").unlink()
            with self.assertRaisesRegex(FileNotFoundError, "model.safetensors"):
                _validate_roberta_base_folder(folder)

    def test_roberta_base_can_be_resolved_from_transferred_hub_cache(self) -> None:
        with tempfile.TemporaryDirectory(prefix="vdt-roberta-cache-test-") as temporary:
            cache = Path(temporary)
            project_root = cache / "isolated-project"
            project_root.mkdir()
            snapshot = cache / "models--FacebookAI--roberta-base" / "snapshots" / "abc123"
            snapshot.mkdir(parents=True)
            for filename in ("config.json", "model.safetensors", "tokenizer.json"):
                (snapshot / filename).touch()

            self.assertEqual(_cached_roberta_base(cache), snapshot.resolve())
            with patch("src.evaluate.run_eval.PROJECT_ROOT", project_root):
                self.assertEqual(
                    _resolve_roberta_base("models/roberta-base", str(cache)),
                    snapshot.resolve(),
                )

    def test_cli_scores_human_and_llm_summaries_without_legacy_column_args(self) -> None:
        row = {
            "id": "1",
            "input": "The source says rain began at noon.",
            "human_sum": "Rain began at noon.",
            "llm_sum": "Rain started at noon.",
        }
        with tempfile.TemporaryDirectory(prefix="vdt-eval-cli-test-") as temporary:
            root = Path(temporary)
            data_path = root / "data.jsonl"
            output_path = root / "results.jsonl"
            data_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            with patch(
                "sys.argv",
                [
                    "run_eval",
                    "--data",
                    str(data_path),
                    "--metrics",
                    "rouge",
                    "--output",
                    str(output_path),
                ],
            ):
                self.assertEqual(main(), 0)

            result = json.loads(output_path.read_text(encoding="utf-8").splitlines()[0])
            self.assertIsNone(result["human_sum__rouge_score"])
            self.assertIsNotNone(result["llm_sum__rouge_score"])

    def test_nltk_argument_accepts_root_or_nested_punkt_tab_folder(self) -> None:
        with tempfile.TemporaryDirectory(prefix="vdt-nltk-test-") as temporary:
            root = Path(temporary)
            punkt_tab = root / "tokenizers" / "punkt_tab"
            (punkt_tab / "english").mkdir(parents=True)

            self.assertEqual(_nltk_data_root(str(root)), root)
            self.assertEqual(_nltk_data_root(str(root / "tokenizers")), root)
            self.assertEqual(_nltk_data_root(str(punkt_tab)), root)
            self.assertEqual(_nltk_data_root(str(punkt_tab / "english")), root)

    def test_hf_cache_root_is_found_inside_wrapped_archive_folder(self) -> None:
        with tempfile.TemporaryDirectory(prefix="vdt-hf-cache-test-") as temporary:
            archive_root = Path(temporary) / "uploaded-cache"
            actual_cache = archive_root / "huggingface" / "hub"
            (actual_cache / "models--lytang--MiniCheck-Flan-T5-Large").mkdir(parents=True)

            self.assertEqual(_hf_cache_root(str(archive_root)), actual_cache.resolve())


if __name__ == "__main__":
    unittest.main()
