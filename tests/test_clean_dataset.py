import json
import tempfile
import unittest
from pathlib import Path

from src.data.clean_dataset import clean_file, clean_text


class CleanDatasetTests(unittest.TestCase):
    def test_clean_text_replaces_links_and_removes_decorations(self) -> None:
        cleaned, stats = clean_text(
            "🔥 Tiêu đề\n---\nNội dung [xem thêm](https://example.com/a). ✅ <b>Quan trọng</b>",
        )

        self.assertEqual(cleaned, "Tiêu đề\nNội dung xem thêm [URL]. Quan trọng")
        self.assertEqual(stats.markdown_links_replaced, 1)
        self.assertEqual(stats.urls_replaced, 0)
        self.assertEqual(stats.html_tags_removed, 2)
        self.assertGreaterEqual(stats.icon_chars_removed, 1)
        self.assertEqual(stats.separator_lines_removed, 1)

    def test_clean_file_drops_bad_rows_and_preserves_reference(self) -> None:
        with tempfile.TemporaryDirectory(prefix="clean-dataset-") as temporary:
            root = Path(temporary)
            input_path = root / "input.jsonl"
            output_path = root / "cleaned.jsonl"
            rows = [
                {"id": "ok", "input": "🔥 Văn bản https://example.com.", "output": "Tóm tắt.", "qwen3_token_length": 12},
                {"id": "empty", "input": "", "output": "Có tóm tắt."},
                {"id": "corrupt", "input": "Văn bản � hỏng", "output": "Tóm tắt."},
            ]
            input_path.write_text(
                "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
                encoding="utf-8",
            )

            report = clean_file(
                input_path,
                output_path,
                source_col="input",
                reference_col="output",
                drop_stale_token_lengths=True,
            )

            cleaned_rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([row["id"] for row in cleaned_rows], ["ok"])
            self.assertEqual(cleaned_rows[0]["output"], "Tóm tắt.")
            self.assertNotIn("qwen3_token_length", cleaned_rows[0])
            self.assertEqual(report.records_read, 3)
            self.assertEqual(report.records_written, 1)
            self.assertEqual(report.records_dropped, 2)
            self.assertIn("empty_source", report.drop_reasons)
            self.assertIn("replacement_character", report.drop_reasons)
            self.assertTrue((root / "cleaned.jsonl.report.json").is_file())
            self.assertTrue((root / "cleaned.jsonl.audit.jsonl").is_file())

    def test_clean_file_autodetects_canonical_columns(self) -> None:
        with tempfile.TemporaryDirectory(prefix="clean-dataset-canonical-") as temporary:
            root = Path(temporary)
            input_path = root / "input.jsonl"
            output_path = root / "cleaned.jsonl"
            input_path.write_text(
                json.dumps({"id": "1", "text": "Nội dung ✅", "summary": "Tóm tắt."}, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

            report = clean_file(input_path, output_path)

            self.assertEqual(report.source_col, "text")
            self.assertEqual(report.reference_col, "summary")
            self.assertEqual(json.loads(output_path.read_text(encoding="utf-8"))["text"], "Nội dung")


if __name__ == "__main__":
    unittest.main()
