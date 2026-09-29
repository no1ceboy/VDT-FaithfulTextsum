"""Local-checkpoint BERTScore-style contextual token alignment.

This portable implementation uses the project's existing PyTorch and
Transformers dependencies. It reports raw, unrescaled precision/recall/F1 with
uniform token weights (no IDF). The selected hidden layer and checkpoint must
be recorded for reproducible comparisons.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as functional

from .base import BaseEvaluator

logger = logging.getLogger(__name__)


class BertScoreEvaluator(BaseEvaluator):
    """Compute contextual token-alignment scores against a reference column."""

    metric_name = "bertscore"

    def __init__(
        self,
        device: str = "cuda",
        model_path: str | None = None,
        num_layers: int = 9,
        reference_col: str = "human_sum",
        batch_size: int = 8,
        max_length: int = 512,
    ) -> None:
        super().__init__(device=device, model_path=model_path, batch_size=batch_size)
        self.num_layers = num_layers
        self.reference_col = reference_col
        self.max_length = max_length
        self._model = None
        self._tokenizer = None

    def _load(self) -> None:
        if not self.model_path:
            raise ValueError("BERTScore requires --bertscore_model_path pointing to a local model folder")
        model_path = Path(self.model_path).expanduser().resolve()
        if not model_path.is_dir():
            raise FileNotFoundError(f"Local BERTScore model directory not found: {model_path}")

        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "BERTScore needs the already-provisioned Transformers package; ask IT to provide it."
            ) from exc

        self._tokenizer = AutoTokenizer.from_pretrained(
            str(model_path), local_files_only=True, use_fast=True
        )
        self._model = AutoModel.from_pretrained(
            str(model_path), local_files_only=True, output_hidden_states=True
        )
        layer_count = int(self._model.config.num_hidden_layers)
        if not 1 <= self.num_layers <= layer_count:
            raise ValueError(
                f"--bertscore_num_layers must be between 1 and {layer_count} "
                f"for this model (got {self.num_layers})"
            )
        self._model.to(self.device)
        self._model.eval()
        logger.info(
            "Loaded local BERTScore encoder from %s (layer=%d/%d, device=%s)",
            model_path,
            self.num_layers,
            layer_count,
            self.device,
        )

    def _encode(self, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with torch.inference_mode():
            output = self._model(**encoded)
        embeddings = output.hidden_states[self.num_layers]
        valid = encoded["attention_mask"].bool()
        for special_id in self._tokenizer.all_special_ids:
            valid &= encoded["input_ids"].ne(special_id)
        return functional.normalize(embeddings, p=2, dim=-1), valid

    def _score_batch(
        self, candidates: list[str], references: list[str]
    ) -> list[tuple[float, float, float]]:
        candidate_embeddings, candidate_mask = self._encode(candidates)
        reference_embeddings, reference_mask = self._encode(references)
        similarities = torch.bmm(candidate_embeddings, reference_embeddings.transpose(1, 2))

        # Ignore padding and special tokens when finding each token's best match.
        similarities_for_reference = similarities.masked_fill(
            ~reference_mask.unsqueeze(1), torch.finfo(similarities.dtype).min
        )
        similarities_for_candidate = similarities.masked_fill(
            ~candidate_mask.unsqueeze(2), torch.finfo(similarities.dtype).min
        )
        precision_by_token = similarities_for_reference.max(dim=2).values
        recall_by_token = similarities_for_candidate.max(dim=1).values
        precision_by_token = precision_by_token.masked_fill(~candidate_mask, 0.0)
        recall_by_token = recall_by_token.masked_fill(~reference_mask, 0.0)

        precision = precision_by_token.sum(dim=1) / candidate_mask.sum(dim=1).clamp_min(1)
        recall = recall_by_token.sum(dim=1) / reference_mask.sum(dim=1).clamp_min(1)
        has_tokens_on_both_sides = candidate_mask.any(dim=1) & reference_mask.any(dim=1)
        precision = precision.masked_fill(~has_tokens_on_both_sides, 0.0)
        recall = recall.masked_fill(~has_tokens_on_both_sides, 0.0)
        f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
        return [
            (float(p), float(r), float(f))
            for p, r, f in zip(precision.cpu(), recall.cpu(), f1.cpu())
        ]

    def _evaluate(
        self,
        records: list[dict[str, Any]],
        source_col: str,
        summary_col: str,
    ) -> list[dict[str, Any]]:
        del source_col  # Reference-based; source document is not used.
        scored_indices = [i for i, _ in enumerate(records) if summary_col != self.reference_col]
        values: dict[int, tuple[float, float, float]] = {}
        for start in range(0, len(scored_indices), self.batch_size):
            indices = scored_indices[start : start + self.batch_size]
            batch_scores = self._score_batch(
                [records[i][summary_col] for i in indices],
                [records[i][self.reference_col] for i in indices],
            )
            values.update(zip(indices, batch_scores))

        results: list[dict[str, Any]] = []
        for index, record in enumerate(records):
            out = dict(record)
            if index in values:
                precision, recall, f1 = values[index]
                precision, recall, f1 = round(precision, 6), round(recall, 6), round(f1, 6)
            else:
                precision = recall = f1 = None
            out[f"{self.metric_name}_score"] = f1
            out[f"{self.metric_name}_precision"] = precision
            out[f"{self.metric_name}_recall"] = recall
            results.append(out)
        return results
