"""Vietnamese mFACT evaluator.

mFACT-vi_VN is a binary document--summary classifier released by the mFACT
authors.  It was trained as a Vietnamese faithfulness classifier using
translation-based transfer, so it is a more relevant exploratory signal for
this project than the English-oriented metrics.  The model's class-1
probability is the faithfulness score, matching the authors' inference script.

The evaluator uses only the repository's existing PyTorch/Transformers
dependencies.  For an offline company-machine run, ``model_path`` must point
to a complete extracted model directory containing ``config.json``, model
weights, and tokenizer files.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .base import BaseEvaluator

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "yfqiu-nlp/mFACT-vi_VN"
_DEFAULT_MAX_LENGTH = 512


def _faithful_class_index(config: Any) -> int:
    """Resolve mFACT's faithful class, which is class 1 in the release."""
    label2id = getattr(config, "label2id", {}) or {}
    for label, index in label2id.items():
        normalized = str(label).strip().lower().replace("-", "_").replace(" ", "_")
        if normalized in {
            "faithful",
            "consistent",
            "correct",
            "entailment",
            "label_1",
            "1",
        }:
            return int(index)

    # The released mFACT checkpoint has two labels and the official script
    # applies softmax(... )[:, 1] as the faithful probability.
    if int(getattr(config, "num_labels", 2)) == 2:
        return 1
    raise ValueError(
        "mFACT checkpoint does not expose a recognizable faithful class and is "
        f"not binary (num_labels={getattr(config, 'num_labels', None)!r})."
    )


class MFactEvaluator(BaseEvaluator):
    """Score Vietnamese source/summary pairs with mFACT-vi_VN."""

    metric_name = "mfact"

    def __init__(
        self,
        device: str = "cuda",
        model_path: str | None = None,
        batch_size: int = 8,
        max_length: int = _DEFAULT_MAX_LENGTH,
    ) -> None:
        super().__init__(device=device, model_path=model_path, batch_size=batch_size)
        if max_length < 1:
            raise ValueError("mFACT max_length must be positive")
        self._model_id = model_path or _DEFAULT_MODEL
        self.max_length = max_length
        self._tokenizer = None
        self._model = None
        self._faithful_idx: int | None = None

    def _load(self) -> None:
        logger.info("Loading Vietnamese mFACT from %s", self._model_id)
        local_files_only = Path(self._model_id).expanduser().is_dir()
        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_id,
            local_files_only=local_files_only,
        )
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self._model_id,
            local_files_only=local_files_only,
        )
        self._faithful_idx = _faithful_class_index(self._model.config)
        model_limit = int(getattr(self._model.config, "max_position_embeddings", self.max_length))
        self.max_length = min(self.max_length, model_limit)
        self._model.to(self.device).eval()

    def _evaluate(
        self,
        records: list[dict[str, Any]],
        source_col: str,
        summary_col: str,
    ) -> list[dict[str, Any]]:
        if self._faithful_idx is None or self._tokenizer is None or self._model is None:
            raise RuntimeError("mFACT model is not loaded")

        results: list[dict[str, Any]] = []
        for start in tqdm(range(0, len(records), self.batch_size), desc="mFACT", unit="batch"):
            batch = records[start : start + self.batch_size]
            sources = [str(record[source_col]) for record in batch]
            summaries = [str(record[summary_col]) for record in batch]
            encoded = self._tokenizer(
                sources,
                summaries,
                max_length=self.max_length,
                truncation=True,
                padding=True,
                return_tensors="pt",
            )
            encoded = {name: value.to(self.device) for name, value in encoded.items()}
            with torch.inference_mode():
                logits = self._model(**encoded).logits
                scores = torch.softmax(logits, dim=-1)[:, self._faithful_idx].cpu().tolist()

            for record, score in zip(batch, scores):
                out = dict(record)
                out[f"{self.metric_name}_score"] = round(float(score), 6)
                out[f"{self.metric_name}_pred"] = int(score >= 0.5)
                results.append(out)
        return results
