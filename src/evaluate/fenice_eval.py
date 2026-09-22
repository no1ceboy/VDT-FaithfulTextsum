"""FENICE evaluator.

Wraps the ``FENICE`` pip package for factual-consistency scoring.

Offline usage: FENICE internally downloads NLI and coreference models from
HuggingFace. For fully offline use, pre-download those models and set
``TRANSFORMERS_OFFLINE=1`` + ``HF_HUB_OFFLINE=1`` in your environment, or
pass ``model_path`` which will be forwarded to the FENICE ``nli_model``
argument if the FENICE API supports it.

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

from tqdm import tqdm

from .base import BaseEvaluator

logger = logging.getLogger(__name__)

_DEFAULT_NLI_MODEL = "roberta-large-mnli"


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
        model_path: str | None = None,
        batch_size: int = 8,
        granularity: str = "sentence",  # "sentence" | "clause"
    ) -> None:
        super().__init__(device=device, model_path=model_path, batch_size=batch_size)
        self.granularity = granularity
        self._nli_model_id = model_path or _DEFAULT_NLI_MODEL
        self._fenice = None

    # ------------------------------------------------------------------

    def _load(self) -> None:
        try:
            from fenice import Fenice  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "FENICE is not installed. Run: pip install FENICE"
            ) from exc

        logger.info("Loading FENICE (NLI model: %s) …", self._nli_model_id)
        warnings.warn(
            "FENICE was designed for English. Scores on Vietnamese text are "
            "computed but may be less calibrated.",
            UserWarning,
            stacklevel=2,
        )
        # FENICE constructor accepts device and nli_model_name
        self._fenice = Fenice(
            device=self.device,
            nli_model_name=self._nli_model_id,
            granularity=self.granularity,
        )

    # ------------------------------------------------------------------

    def _evaluate(
        self,
        records: list[dict[str, Any]],
        source_col: str,
        summary_col: str,
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []

        for record in tqdm(records, desc="FENICE", unit="doc"):
            source = record[source_col]
            summary = record[summary_col]
            try:
                score = self._fenice.score(article=source, summary=summary)
                # FENICE returns a dict with 'score' key (float in [0,1])
                if isinstance(score, dict):
                    score_val = float(score.get("score", score.get("fenice_score", 0.0)))
                else:
                    score_val = float(score)
            except Exception as exc:  # noqa: BLE001
                logger.warning("FENICE failed on record (id=%s): %s", record.get("id"), exc)
                score_val = float("nan")

            out = dict(record)
            out[f"{self.metric_name}_score"] = round(score_val, 6)
            results.append(out)

        return results
