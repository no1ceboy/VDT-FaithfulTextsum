"""MiniCheck evaluator.

Wraps the ``MiniCheck`` library for sentence-level fact-checking.

Default model: ``Bespoke-MiniCheck-7B`` (best quality, requires GPU).
Lighter alternative: ``MiniCheck-Flan-T5-Large`` (CPU-feasible).

Offline usage: pass ``model_path`` pointing to a local directory or set
``HF_HUB_OFFLINE=1`` after pre-downloading the checkpoint.

The score returned is the *mean support probability* over all sentences
in the summary (1.0 = all sentences supported; 0.0 = none supported).

References
----------
* Tang et al., 2024 — "MiniCheck: Efficient Fact-Checking of LLMs on
  Grounding Documents"
* https://github.com/Liyan06/MiniCheck
"""

from __future__ import annotations

import logging
from typing import Any

from tqdm import tqdm

from evaluate.base import BaseEvaluator

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "Bespoke-MiniCheck-7B"


class MiniCheckEvaluator(BaseEvaluator):
    """Wrapper around MiniCheck sentence-level fact-checker."""

    metric_name = "minicheck"

    def __init__(
        self,
        device: str = "cuda",
        model_path: str | None = None,
        batch_size: int = 16,
        model_name: str = _DEFAULT_MODEL,
    ) -> None:
        super().__init__(device=device, model_path=model_path, batch_size=batch_size)
        # model_path takes priority over model_name
        self.model_name = model_path or model_name
        self._checker = None

    # ------------------------------------------------------------------

    def _load(self) -> None:
        try:
            from minicheck.minicheck import MiniCheck  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "MiniCheck is not installed. Run:\n"
                '  pip install "minicheck @ git+https://github.com/Liyan06/MiniCheck.git@main"'
            ) from exc

        logger.info("Loading MiniCheck model: %s …", self.model_name)
        self._checker = MiniCheck(
            model_name=self.model_name,
            device=self.device,
        )

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
            desc="MiniCheck",
            unit="batch",
        ):
            batch = records[i : i + self.batch_size]
            docs = [r[source_col] for r in batch]
            claims = [r[summary_col] for r in batch]

            # MiniCheck.score() accepts parallel lists of docs and claims.
            # It returns (pred_labels, raw_probs, sent_scores, agg_scores)
            # where agg_scores is the aggregated per-document score.
            try:
                pred_labels, raw_probs, sent_scores, agg_scores = self._checker.score(
                    docs=docs,
                    claims=claims,
                    chunk_size=self.batch_size,
                )
            except Exception as exc:  # noqa: BLE001
                logger.error("MiniCheck batch %d failed: %s", i, exc)
                agg_scores = [float("nan")] * len(batch)
                pred_labels = [None] * len(batch)

            for record, score, label in zip(batch, agg_scores, pred_labels):
                out = dict(record)
                out[f"{self.metric_name}_score"] = round(float(score), 6) if score == score else float("nan")
                out[f"{self.metric_name}_pred"] = label  # 1 = supported, 0 = not
                results.append(out)

        return results
