"""QAFactEval evaluator using the upstream scoring API.

QAFactEval is an older English-language project with a fixed local model
layout. ``model_path`` must point to the directory created by its
``download_models.sh`` script.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from tqdm import tqdm

from .base import BaseEvaluator

logger = logging.getLogger(__name__)


class QAFactEvalEvaluator(BaseEvaluator):
    """Adapter for QAFactEval's QA plus LERC-QuIP factuality score."""

    metric_name = "qafacteval"

    def __init__(
        self,
        device: str = "cuda",
        model_path: str | None = None,
        batch_size: int = 8,
        cuda_device: int = 0,
    ) -> None:
        super().__init__(device=device, model_path=model_path, batch_size=batch_size)
        self._cuda_device = cuda_device if device.startswith("cuda") else -1
        self._model_folder = Path(model_path or "./models")
        self._scorer = None

    def _load(self) -> None:
        try:
            from qafacteval import QAFactEval  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "QAFactEval is not installed. Install the Salesforce/QAFactEval "
                "repository in a compatible environment."
            ) from exc

        required = [
            self._model_folder / "quip-512-mocha",
            self._model_folder / "generation" / "model.tar.gz",
            self._model_folder / "answering",
            self._model_folder / "lerc" / "model.tar.gz",
            self._model_folder / "lerc" / "pretraining.tar.gz",
        ]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "QAFactEval model files are missing:\n  " + "\n  ".join(missing)
            )

        logger.info(
            "Loading QAFactEval (model_folder=%s, cuda=%d)",
            self._model_folder,
            self._cuda_device,
        )
        self._scorer = QAFactEval(
            lerc_quip_path=str(self._model_folder / "quip-512-mocha"),
            generation_model_path=str(self._model_folder / "generation" / "model.tar.gz"),
            answering_model_dir=str(self._model_folder / "answering"),
            lerc_model_path=str(self._model_folder / "lerc" / "model.tar.gz"),
            lerc_pretrained_model_path=str(self._model_folder / "lerc" / "pretraining.tar.gz"),
            cuda_device=self._cuda_device,
            use_lerc_quip=True,
            verbose=False,
            generation_batch_size=self.batch_size,
            answering_batch_size=self.batch_size,
            lerc_batch_size=self.batch_size,
        )

    def _evaluate(
        self,
        records: list[dict[str, Any]],
        source_col: str,
        summary_col: str,
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for start in tqdm(range(0, len(records), self.batch_size), desc="QAFactEval", unit="batch"):
            batch = records[start : start + self.batch_size]
            sources = [str(record[source_col]) for record in batch]
            summaries = [[str(record[summary_col])] for record in batch]
            batch_results = self._scorer.score_batch_qafacteval(
                sources,
                summaries,
                return_qa_pairs=False,
            )
            if len(batch_results) != len(batch):
                raise RuntimeError("QAFactEval returned an unexpected number of results")

            for record, result in zip(batch, batch_results):
                metrics = result[0]
                qa_eval = metrics.get("qa-eval", {})
                # Keep the model's native LERC-QuIP score; it is not rescaled.
                score = float(qa_eval["lerc_quip"])
                out = dict(record)
                out[f"{self.metric_name}_score"] = round(score, 6)
                out[f"{self.metric_name}_f1"] = qa_eval.get("f1")
                out[f"{self.metric_name}_em"] = qa_eval.get("em")
                out[f"{self.metric_name}_is_answered"] = qa_eval.get("is_answered")
                results.append(out)
        return results
