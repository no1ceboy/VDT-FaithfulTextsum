from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.evaluate.run_eval import _hf_cache_root, _nltk_data_root, _project_path


class EvalCliPathTests(unittest.TestCase):
    def test_factcc_relative_path_is_anchored_at_project_root(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        self.assertEqual(_project_path("models/factcc"), project_root / "models" / "factcc")

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
