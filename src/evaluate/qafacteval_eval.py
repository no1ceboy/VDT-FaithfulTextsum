"""QAFactEval evaluator.

Wraps the ``qafacteval`` library for QA-based faithfulness scoring.

Offline usage:
  1. Clone the repo and install: ``pip install -e .``
  2. Download pretrained models via the provided script:
       bash download_models.sh
  3. Pass ``model_path`` to the directory containing the QA/QG models.
     By default QAFactEval looks for models in ``./models`` relative to its
     package. Setting ``model_path`` overrides that base directory.

The score returned is the QAFactEval aggregate score (float, higher = more faithful).
We normalise it to [0, 1] by dividing by the max possible score (5.0) if the
raw score exceeds 1.0.

References
----------
* Fabbri et al., 2022 — "QAFactEval: Improved QA-Based Factual Consistency
  Evaluation for Summarization"
* https://github.com/salesforce/QAFactEval
"""

from __future__ import annotations

import logging
from typing import Any

from tqdm import tqdm

from .base import BaseEvaluator

logger = logging.getLogger(__name__)

_QAFACTEVAL_MAX_RAW = 1.0  # adjust if your install returns unnormalised scores


class QAFactEvalEvaluator(BaseEvaluator):
    """Wrapper around the QAFactEval QA-based faithfulness metric."""

    metric_name = "qafacteval"

    def __init__(
        self,
        device: str = "cuda",
        model_path: str | None = None,
        batch_size: int = 8,
        cuda_device: int = 0,
    ) -> None:
        super().__init__(device=device, model_path=model_path, batch_size=batch_size)
        # QAFactEval uses integer CUDA device ids
        self._cuda_device = cuda_device if device.startswith("cuda") else -1
        self._model_folder = model_path or "./models"
        self._scorer = None

    # ------------------------------------------------------------------

    def _load(self) -> None:
        try:
            from qafacteval import QAFactEval  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "QAFactEval is not installed. Run:\n"
                "  git clone https://github.com/salesforce/QAFactEval && "
                "cd QAFactEval && pip install -e ."
            ) from exc

        logger.info(
            "Loading QAFactEval (model_folder=%s, cuda=%d) …",
            self._model_folder,
            self._cuda_device,
        )
        kwargs = dict(
            cuda_device=self._cuda_device,
            use_lerc_quip=True,
            verbose=False,
            generation_batch_size=self.batch_size,
            answering_batch_size=self.batch_size,
            lerc_batch_size=self.batch_size,
        )
        self._scorer = QAFactEval(model_folder=self._model_folder, **kwargs)

    # ------------------------------------------------------------------

    def _evaluate(
        self,
        records: list[dict[str, Any]],
        source_col: str,
        summary_col: str,
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []

        for i in tqdm(
            range(0, len(records), self.batch_size),
            desc="QAFactEval",
            unit="batch",
        ):
            batch = records[i : i + self.batch_size]
            # QAFactEval expects: sources as list[list[str]] (each source is a
            # list of sentences) and summaries as list[str].
            sources = [[r[source_col]] for r in batch]  # treat doc as single "sentence"
            summaries = [r[summary_col] for r in batch]

            try:
                batch_results = self._scorer.score(sources, summaries, return_qa_pairs=False)
                # batch_results is a list of dicts, one per example
                scores = [res[0]["qa-eval"] for res in batch_results]
            except Exception as exc:  # noqa: BLE001
                logger.error("QAFactEval batch %d failed: %s", i, exc)
                scores = [float("nan")] * len(batch)

            for record, raw_score in zip(batch, scores):
                out = dict(record)
                if raw_score == raw_score:  # not NaN
                    # Normalise if necessary (raw scores can exceed 1.0)
                    normalised = min(float(raw_score) / _QAFACTEVAL_MAX_RAW, 1.0)
                else:
                    normalised = float("nan")
                out[f"{self.metric_name}_score"] = round(normalised, 6)
                out[f"{self.metric_name}_raw_score"] = round(float(raw_score), 6) if raw_score == raw_score else float("nan")
                results.append(out)

        return results
