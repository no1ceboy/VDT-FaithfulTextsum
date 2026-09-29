"""FENICE evaluator using the upstream ``score_batch`` API.

FENICE loads its claim extractor and NLI checkpoints by Hugging Face
repository ID. Offline runs need those repositories in the Hugging Face
cache; the upstream API does not accept a checkpoint-path override.

References
----------
* Scirè et al., 2024 — "FENICE: Factuality Evaluation of summarization based
  on Natural language Inference and Claim Extraction"
* https://github.com/Babelscape/FENICE
"""

from __future__ import annotations

import logging
import warnings
from typing import Any

from .base import BaseEvaluator

logger = logging.getLogger(__name__)

class FENICEEvaluator(BaseEvaluator):
    """Wrapper around the FENICE factuality evaluator.

    Notes
    -----
    FENICE was designed for English. On Vietnamese documents it will still
    produce a numeric score, but calibration may be lower. A warning is
    emitted once to remind users of this limitation.
    """

    metric_name = "fenice"

    def __init__(
        self,
        device: str = "cuda",
        batch_size: int = 8,
    ) -> None:
        super().__init__(device=device, batch_size=batch_size)
        self._fenice = None

    # ------------------------------------------------------------------

    def _load(self) -> None:
        try:
            from metric.FENICE import FENICE  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "FENICE is not installed. Install the Babelscape/FENICE "
                "repository and its pinned dependencies."
            ) from exc

        logger.info("Loading upstream FENICE models from the local HF cache …")
        warnings.warn(
            "FENICE was developed and evaluated primarily on English; interpret "
            "Vietnamese scores as exploratory.",
            UserWarning,
            stacklevel=2,
        )
        if self.device.startswith("cuda"):
            import torch

            if ":" in self.device:
                torch.cuda.set_device(int(self.device.split(":", 1)[1]))
        self._fenice = FENICE()

    # ------------------------------------------------------------------

    def _evaluate(
        self,
        records: list[dict[str, Any]],
        source_col: str,
        summary_col: str,
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for start in range(0, len(records), self.batch_size):
            batch = records[start : start + self.batch_size]
            inputs = [
                {"document": str(record[source_col]), "summary": str(record[summary_col])}
                for record in batch
            ]
            scores = self._fenice.score_batch(inputs)
            if len(scores) != len(batch):
                raise RuntimeError(
                    f"FENICE returned {len(scores)} scores for {len(batch)} records"
                )
            for record, score in zip(batch, scores):
                out = dict(record)
                out[f"{self.metric_name}_score"] = round(float(score["score"]), 6)
                results.append(out)
            # The upstream implementation caches all claim alignments. Clear
            # those per-batch caches to keep memory bounded on large datasets.
            for cache_name in ("sentences_cache", "coref_clusters_cache", "claims_cache", "alignments_cache"):
                getattr(self._fenice, cache_name, {}).clear()
        return results
