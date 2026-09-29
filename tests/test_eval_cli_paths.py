from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.evaluate.run_eval import _nltk_data_root, _project_path


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


if __name__ == "__main__":
    unittest.main()
