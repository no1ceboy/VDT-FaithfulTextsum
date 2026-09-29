"""FactCC evaluator.

Uses the ``manueldeprada/FactCC`` checkpoint via HuggingFace ``transformers``.
Offline usage: pass ``model_path`` pointing to a local directory that contains
the model files (downloaded via ``huggingface-cli download`` or ``snapshot_download``).

FactCC predicts each (document, claim) pair as CORRECT (label 1) or INCORRECT
(label 0). The score returned is the *probability of the CORRECT class*.

References
----------
* Kryscinski et al., 2020 — "Evaluating the Factual Consistency of Abstractive
  Text Summarization"
* HF hub: manueldeprada/FactCC
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

_DEFAULT_MODEL = "manueldeprada/FactCC"
_CORRECT_LABEL = "CORRECT"  # label used by the FactCC checkpoint


class FactCCEvaluator(BaseEvaluator):
    """Wrapper around the FactCC NLI-based faithfulness classifier."""

    metric_name = "factcc"

    def __init__(
        self,
        device: str = "cuda",
        model_path: str | None = None,
        batch_size: int = 8,
        max_length: int = 512,
    ) -> None:
        super().__init__(device=device, model_path=model_path, batch_size=batch_size)
        self.max_length = max_length
        self._model_id = model_path or _DEFAULT_MODEL
        self._tokenizer = None
        self._model = None
        self._correct_idx: int | None = None

    # ------------------------------------------------------------------

    def _load(self) -> None:
        logger.info("Loading FactCC from %s …", self._model_id)
        local_files_only = Path(self._model_id).expanduser().is_dir()
        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_id, local_files_only=local_files_only
        )
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self._model_id, local_files_only=local_files_only
        )
        self._model.to(self.device)
        self._model.eval()

        # Find the index corresponding to the CORRECT label
        label2id = self._model.config.label2id
        if _CORRECT_LABEL in label2id:
            self._correct_idx = label2id[_CORRECT_LABEL]
        else:
            # Fall back: assume label 0 = INCORRECT, label 1 = CORRECT
            self._correct_idx = 1
            logger.warning(
                "FactCC: label '%s' not found in config (%s). Defaulting to index 1.",
                _CORRECT_LABEL,
                list(label2id.keys()),
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
            desc="FactCC",
            unit="batch",
        ):
            batch = records[i : i + self.batch_size]
            sources = [r[source_col] for r in batch]
            summaries = [r[summary_col] for r in batch]

            encodings = self._tokenizer(
                sources,
                summaries,
                max_length=self.max_length,
                truncation=True,
                padding=True,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                logits = self._model(**encodings).logits  # (B, num_labels)
                probs = torch.softmax(logits, dim=-1)
                scores = probs[:, self._correct_idx].cpu().tolist()

            for record, score in zip(batch, scores):
                out = dict(record)
                out[f"{self.metric_name}_score"] = round(score, 6)
                results.append(out)

        return results
