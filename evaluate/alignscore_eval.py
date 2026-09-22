"""AlignScore evaluator.

Wraps ``alignscore`` for information-alignment-based faithfulness scoring.

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
* https://github.com/yuh-zha/AlignScore
"""

from __future__ import annotations

import logging
from typing import Any

from tqdm import tqdm

from evaluate.base import BaseEvaluator

logger = logging.getLogger(__name__)

# Default HuggingFace checkpoint identifier (used if model_path is None)
_DEFAULT_CKPT = "yzha/AlignScore"
_DEFAULT_CKPT_FILE = "AlignScore-large.ckpt"


class AlignScoreEvaluator(BaseEvaluator):
    """Wrapper around the AlignScore alignment-based faithfulness metric."""

    metric_name = "alignscore"

    def __init__(
        self,
        device: str = "cuda",
        model_path: str | None = None,
        batch_size: int = 8,
        evaluation_mode: str = "nli_sp",  # "nli_sp" | "nli" | "bin_sp" | "bin"
    ) -> None:
        super().__init__(device=device, model_path=model_path, batch_size=batch_size)
        self.evaluation_mode = evaluation_mode
        # If model_path is given it must point to a local .ckpt file.
        self._ckpt_path = model_path  # None → AlignScore will auto-download
        self._scorer = None

    # ------------------------------------------------------------------

    def _load(self) -> None:
        try:
            from alignscore import AlignScore  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "AlignScore is not installed. Run:\n"
                "  git clone https://github.com/yuh-zha/AlignScore && "
                "cd AlignScore && pip install ."
            ) from exc

        logger.info(
            "Loading AlignScore (ckpt=%s, mode=%s) …",
            self._ckpt_path or _DEFAULT_CKPT,
            self.evaluation_mode,
        )

        kwargs: dict[str, Any] = dict(
            model="roberta-large",
            batch_size=self.batch_size,
            device=self.device,
            evaluation_mode=self.evaluation_mode,
        )
        if self._ckpt_path:
            kwargs["ckpt_path"] = self._ckpt_path
        else:
            # Let AlignScore download the default checkpoint
            kwargs["ckpt_path"] = _DEFAULT_CKPT_FILE

        self._scorer = AlignScore(**kwargs)

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
        try:
            # AlignScore.score() processes in internal batches
            scores = self._scorer.score(contexts=sources, claims=summaries)
        except Exception as exc:  # noqa: BLE001
            logger.error("AlignScore failed: %s", exc)
            scores = [float("nan")] * len(records)

        results: list[dict[str, Any]] = []
        for record, score in tqdm(
            zip(records, scores), total=len(records), desc="AlignScore", unit="doc"
        ):
            out = dict(record)
            out[f"{self.metric_name}_score"] = round(float(score), 6) if score == score else float("nan")
            results.append(out)

        return results
