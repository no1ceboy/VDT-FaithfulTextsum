"""AlignScore evaluator using the project's inference-only compatibility port.

Default checkpoint: ``AlignScore-large`` (best quality).
Lighter alternative: ``AlignScore-base``.

Offline usage:
  1. Pre-download the checkpoint:
       huggingface-cli download yzha/AlignScore AlignScore-large.ckpt \\
           --local-dir ./models/alignscore
  2. Pass ``model_path="./models/alignscore/AlignScore-large.ckpt"``
  3. Set ``HF_HUB_OFFLINE=1``, ``TRANSFORMERS_OFFLINE=1``

References
----------
* Zha et al., 2023 — "AlignScore: Evaluating Factual Consistency with a
  Unified Alignment Function"
* https://github.com/yuh-zha/AlignScore (MIT; inference implementation adapted locally)
"""

from __future__ import annotations

import logging
from typing import Any

from tqdm import tqdm

from .base import BaseEvaluator

logger = logging.getLogger(__name__)

_DEFAULT_REPO = "yzha/AlignScore"
_DEFAULT_CKPT_FILE = "AlignScore-large.ckpt"


class AlignScoreEvaluator(BaseEvaluator):
    """Wrapper around the AlignScore alignment-based faithfulness metric."""

    metric_name = "alignscore"

    def __init__(
        self,
        device: str = "cuda",
        model_path: str | None = None,
        cache_dir: str | None = None,
        batch_size: int = 8,
        evaluation_mode: str = "nli_sp",  # "nli_sp" | "nli" | "bin_sp" | "bin"
    ) -> None:
        super().__init__(device=device, model_path=model_path, batch_size=batch_size)
        self.evaluation_mode = evaluation_mode
        self._cache_dir = cache_dir
        # If omitted, resolve the checkpoint through the HF cache (supports
        # HF_HUB_OFFLINE when the file has already been transferred).
        self._ckpt_path = model_path
        self._scorer = None

    # ------------------------------------------------------------------

    def _load(self) -> None:
        from ._alignscore_compat import AlignScoreCompatScorer

        logger.info(
            "Loading AlignScore (ckpt=%s, mode=%s) …",
            self._ckpt_path or f"{_DEFAULT_REPO}/{_DEFAULT_CKPT_FILE}",
            self.evaluation_mode,
        )

        checkpoint_path = self._ckpt_path
        if checkpoint_path is None:
            from huggingface_hub import hf_hub_download

            checkpoint_path = hf_hub_download(
                repo_id=_DEFAULT_REPO,
                filename=_DEFAULT_CKPT_FILE,
            )

        self._scorer = AlignScoreCompatScorer(
            ckpt_path=checkpoint_path,
            cache_dir=self._cache_dir,
            batch_size=self.batch_size,
            device=self.device,
            evaluation_mode=self.evaluation_mode,
        )

    # ------------------------------------------------------------------

    def _evaluate(
        self,
        records: list[dict[str, Any]],
        source_col: str,
        summary_col: str,
    ) -> list[dict[str, Any]]:
        sources = [r[source_col] for r in records]
        summaries = [r[summary_col] for r in records]

        logger.info("Running AlignScore on %d records …", len(records))
        # AlignScore.score() processes in internal batches.
        scores = self._scorer.score(contexts=sources, claims=summaries)
        if len(scores) != len(records):
            raise RuntimeError("AlignScore returned an unexpected number of scores")

        results: list[dict[str, Any]] = []
        for record, score in tqdm(
            zip(records, scores), total=len(records), desc="AlignScore", unit="doc"
        ):
            out = dict(record)
            out[f"{self.metric_name}_score"] = round(float(score), 6) if score == score else float("nan")
            results.append(out)

        return results
