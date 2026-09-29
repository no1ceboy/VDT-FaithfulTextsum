"""MiniCheck evaluator using the project's bundled FLAN-T5 inference port.

MiniCheck scores individual claims against source documents. This adapter
splits each summary into sentence-like claims, averages their support
probabilities, and preserves the individual probabilities in the output.

For offline use, provide a Hugging Face cache directory and set
``HF_HUB_OFFLINE=1``. The bundled adapter currently supports the upstream
``flan-t5-large`` model only; its weights remain a separate model asset.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from tqdm import tqdm

from .base import BaseEvaluator

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "flan-t5-large"


class MiniCheckEvaluator(BaseEvaluator):
    """Wrapper around MiniCheck sentence-level fact checking."""

    metric_name = "minicheck"

    def __init__(
        self,
        device: str = "cuda",
        batch_size: int = 16,
        model_name: str = _DEFAULT_MODEL,
        cache_dir: str | None = None,
        chunk_size: int | None = None,
    ) -> None:
        super().__init__(device=device, batch_size=batch_size)
        self.cache_dir = cache_dir
        self.model_name = model_name
        self.chunk_size = chunk_size
        self._checker = None

    def _load(self) -> None:
        from ._minicheck_compat import MiniCheckCompatScorer

        logger.info("Loading MiniCheck model: %s …", self.model_name)
        self._checker = MiniCheckCompatScorer(
            model_name=self.model_name,
            batch_size=self.batch_size,
            cache_dir=self.cache_dir,
            device=self.device,
        )

    def _evaluate(
        self,
        records: list[dict[str, Any]],
        source_col: str,
        summary_col: str,
    ) -> list[dict[str, Any]]:
        results = [dict(record) for record in records]
        docs: list[str] = []
        claims: list[str] = []
        claim_record_ids: list[int] = []
        for record_id, record in enumerate(records):
            pieces = [
                part.strip()
                for part in re.split(r"(?<=[.!?])\s+|[\r\n]+", str(record[summary_col]))
                if part.strip()
            ]
            for claim in pieces:
                docs.append(str(record[source_col]))
                claims.append(claim)
                claim_record_ids.append(record_id)

        per_record_scores: list[list[float]] = [[] for _ in records]
        for start in tqdm(range(0, len(claims), self.batch_size), desc="MiniCheck", unit="batch"):
            end = min(start + self.batch_size, len(claims))
            kwargs: dict[str, Any] = {"docs": docs[start:end], "claims": claims[start:end]}
            # In MiniCheck, chunk_size controls source-document chunk length.
            if self.chunk_size is not None:
                kwargs["chunk_size"] = self.chunk_size
            _, support_probs, _, _ = self._checker.score(**kwargs)
            if len(support_probs) != end - start:
                raise RuntimeError("MiniCheck returned an unexpected number of claim scores")
            for record_id, probability in zip(claim_record_ids[start:end], support_probs):
                per_record_scores[record_id].append(float(probability))

        for out, scores in zip(results, per_record_scores):
            score = sum(scores) / len(scores) if scores else float("nan")
            out[f"{self.metric_name}_score"] = round(score, 6) if score == score else score
            out[f"{self.metric_name}_pred"] = int(all(value >= 0.5 for value in scores)) if scores else None
            out[f"{self.metric_name}_sentence_scores"] = [round(value, 6) for value in scores]
        return results
