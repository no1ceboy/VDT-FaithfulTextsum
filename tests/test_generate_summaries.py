from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.generate_summaries import _read_jsonl


class GenerateSummariesTests(unittest.TestCase):
    def test_legacy_input_human_sum_rows_are_normalized(self) -> None:
        with tempfile.TemporaryDirectory(prefix="generate-summaries-legacy-") as temporary:
            path = Path(temporary) / "legacy.jsonl"
            path.write_text(
                json.dumps(
                    {"id": "sds-1", "input": "Nguồn văn bản.", "abstract_sum": "Tóm tắt."},
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            rows = _read_jsonl(path)

            self.assertEqual(rows[0]["id"], "sds-1")
            self.assertEqual(rows[0]["source"], "Nguồn văn bản.")
            self.assertEqual(rows[0]["reference"], "Tóm tắt.")
            self.assertEqual(rows[0]["prompt"][-1]["role"], "user")

    def test_legacy_input_output_rows_are_normalized(self) -> None:
        with tempfile.TemporaryDirectory(prefix="generate-summaries-output-") as temporary:
            path = Path(temporary) / "legacy.jsonl"
            path.write_text(
                json.dumps(
                    {"id": "batch-1", "input": "source text", "output": "reference text"},
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            rows = _read_jsonl(path)

            self.assertEqual(rows[0]["source"], "source text")
            self.assertEqual(rows[0]["reference"], "reference text")


if __name__ == "__main__":
    unittest.main()
