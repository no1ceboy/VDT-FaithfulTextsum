"""Inference-only AlignScore adapter for the modern project runtime.

The scoring architecture and aggregation are adapted from AlignScore by
Yuheng Zha et al. (MIT; see ``third_party/licenses/ALIGNScore-MIT.txt``).
This module deliberately omits AlignScore's Lightning training interface and
loads the weights from its Lightning checkpoint into an equivalent PyTorch
inference model. Validate score parity against upstream before treating this
port as interchangeable in a research result.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from transformers import AutoTokenizer, RobertaConfig, RobertaModel

logger = logging.getLogger(__name__)

_SUPPORTED_MODES = {"nli_sp", "nli", "bin_sp", "bin"}
_MODEL_ID = "roberta-large"


def _split_sentences(text: str) -> list[str]:
    """Use the same English Punkt sentence segmentation as upstream AlignScore."""
    try:
        from nltk.tokenize import sent_tokenize
    except ImportError as exc:
        raise ImportError("The native AlignScore adapter requires NLTK for sentence splitting.") from exc

    try:
        return sent_tokenize(text, language="english")
    except LookupError as exc:
        raise RuntimeError(
            "NLTK's English Punkt data is missing. Provision the 'punkt_tab' NLTK resource "
            "in the approved environment and set NLTK_DATA to its local data directory."
        ) from exc


def _group_source_sentences(source: str, sentences: Sequence[str]) -> list[str]:
    """Reproduce AlignScore's sentence-preserving ~350-word source chunking."""
    if not sentences:
        return [""]
    target_chunks = len(source.strip().split()) // 350 + 1
    sentences_per_chunk = max(len(sentences) // target_chunks, 1)
    return [
        " ".join(sentences[start : start + sentences_per_chunk])
        for start in range(0, len(sentences), sentences_per_chunk)
    ]


def _aggregate_sentence_scores(
    pair_scores: Sequence[float], source_chunk_count: int, claim_sentence_count: int
) -> float:
    """For each claim sentence take its best source-chunk score, then average."""
    expected = source_chunk_count * claim_sentence_count
    if source_chunk_count < 1 or claim_sentence_count < 1 or len(pair_scores) != expected:
        raise ValueError("Pair-score dimensions do not match source/claim sentence counts")
    best_per_claim = [
        max(pair_scores[source_index * claim_sentence_count + claim_index]
            for source_index in range(source_chunk_count))
        for claim_index in range(claim_sentence_count)
    ]
    return sum(best_per_claim) / len(best_per_claim)


def _unwrap_state_dict(checkpoint: Any) -> Mapping[str, Tensor]:
    """Accept either a Lightning checkpoint or a plain tensor state dictionary."""
    if not isinstance(checkpoint, Mapping):
        raise ValueError("AlignScore checkpoint must contain a state-dictionary mapping")
    state = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state, Mapping) or not all(
        isinstance(name, str) and isinstance(value, Tensor) for name, value in state.items()
    ):
        raise ValueError("AlignScore checkpoint state_dict must map parameter names to tensors")
    if state and all(name.startswith("module.") for name in state):
        state = {name.removeprefix("module."): value for name, value in state.items()}
    return state


class _AlignScoreModel(nn.Module):
    """The upstream RoBERTa AlignScore heads without Lightning training code."""

    def __init__(self, config: RobertaConfig) -> None:
        super().__init__()
        # The AlignScore checkpoint contains the trained RoBERTa backbone.
        # Build its architecture from the small config file, then load all
        # inference weights from the checkpoint instead of downloading a
        # second copy of the 1.4 GB base-model weights.
        self.base_model = RobertaModel(config, add_pooling_layer=True)
        hidden_size = self.base_model.config.hidden_size
        self.bin_layer = nn.Linear(hidden_size, 2)
        self.tri_layer = nn.Linear(hidden_size, 3)
        self.reg_layer = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, encoded: Mapping[str, Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        output = self.base_model(**encoded)
        pooled = self.dropout(output.pooler_output)
        return (
            self.bin_layer(pooled),
            self.tri_layer(pooled),
            self.reg_layer(pooled),
        )


class AlignScoreCompatScorer:
    """Score source/summary pairs with a local upstream AlignScore checkpoint."""

    def __init__(
        self,
        *,
        ckpt_path: str | Path,
        model: str = _MODEL_ID,
        cache_dir: str | Path | None = None,
        batch_size: int = 8,
        device: str = "cuda",
        evaluation_mode: str = "nli_sp",
    ) -> None:
        if evaluation_mode not in _SUPPORTED_MODES:
            raise ValueError(f"Unsupported AlignScore mode {evaluation_mode!r}; choose from {sorted(_SUPPORTED_MODES)}")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.evaluation_mode = evaluation_mode
        offline = (
            os.environ.get("HF_HUB_OFFLINE") == "1"
            or os.environ.get("TRANSFORMERS_OFFLINE") == "1"
        )
        requested_model_path = Path(model).expanduser()
        is_explicit_path = requested_model_path.is_absolute() or requested_model_path.parent != Path(".")
        model_path = requested_model_path if requested_model_path.is_dir() else None
        if is_explicit_path and model_path is None:
            raise FileNotFoundError(f"AlignScore backbone folder not found: {requested_model_path}")
        if model_path is not None:
            if not (model_path / "config.json").is_file() or not any(
                (model_path / name).is_file()
                for name in ("tokenizer.json", "vocab.json", "vocab.txt", "spiece.model")
            ):
                raise FileNotFoundError(
                    f"{model_path} is not a complete RoBERTa tokenizer/config folder. "
                    "It must directly contain config.json and tokenizer files. If this is a "
                    "Hugging Face cache, use --hf_cache_dir instead."
                )
            model = str(model_path.resolve())
            offline = True
        hub_cache = str(Path(cache_dir).expanduser().resolve()) if cache_dir is not None else None
        self.tokenizer = AutoTokenizer.from_pretrained(
            model, cache_dir=hub_cache, local_files_only=offline
        )
        config = RobertaConfig.from_pretrained(
            model, cache_dir=hub_cache, local_files_only=offline
        )
        self.max_length = int(self.tokenizer.model_max_length)
        if self.max_length > 100_000:
            raise ValueError("Tokenizer has no finite model_max_length")

        checkpoint_path = Path(ckpt_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"AlignScore checkpoint not found: {checkpoint_path}")
        # AlignScore checkpoints are pickle-based Lightning files. Restrict
        # deserialization to tensors/primitives; do not execute arbitrary pickle globals.
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        state_dict = _unwrap_state_dict(checkpoint)

        scorer = _AlignScoreModel(config)
        missing_keys, unexpected_keys = scorer.load_state_dict(state_dict, strict=False)
        missing_heads = [
            name for name in ("tri_layer.weight", "tri_layer.bias", "bin_layer.weight", "bin_layer.bias")
            if name in missing_keys
        ]
        missing_backbone = [name for name in missing_keys if name.startswith("base_model.")]
        if missing_heads:
            raise ValueError(
                "AlignScore checkpoint is missing required classification head weights: "
                + ", ".join(missing_heads)
            )
        if missing_backbone:
            preview = ", ".join(missing_backbone[:8])
            suffix = " …" if len(missing_backbone) > 8 else ""
            raise ValueError(
                "AlignScore checkpoint is missing RoBERTa backbone weights; "
                f"refusing random initialization ({preview}{suffix})"
            )
        if unexpected_keys:
            logger.info("Ignoring %d non-inference AlignScore checkpoint keys", len(unexpected_keys))
        if missing_keys:
            logger.info("Using pretrained RoBERTa initialization for %d omitted checkpoint keys", len(missing_keys))

        self.model = scorer.to(self.device).eval()

    def _predict_pairs(self, contexts: Sequence[str], claims: Sequence[str]) -> tuple[list[float], list[float]]:
        if len(contexts) != len(claims):
            raise ValueError("AlignScore contexts and claims must have the same length")
        nli_scores: list[float] = []
        binary_scores: list[float] = []
        for start in range(0, len(contexts), self.batch_size):
            end = min(start + self.batch_size, len(contexts))
            try:
                encoded = self.tokenizer(
                    list(contexts[start:end]),
                    list(claims[start:end]),
                    truncation="only_first",
                    padding="max_length",
                    max_length=self.max_length,
                    return_tensors="pt",
                )
            except ValueError:
                # Match upstream's fallback when the claim itself exceeds the limit.
                encoded = self.tokenizer(
                    list(contexts[start:end]),
                    list(claims[start:end]),
                    truncation=True,
                    padding="max_length",
                    max_length=self.max_length,
                    return_tensors="pt",
                )
            encoded = {name: value.to(self.device) for name, value in encoded.items()}
            with torch.inference_mode():
                binary_logits, nli_logits, _ = self.model(encoded)
                nli_scores.extend(torch.softmax(nli_logits, dim=-1)[:, 0].cpu().tolist())
                binary_scores.extend(torch.softmax(binary_logits, dim=-1)[:, 1].cpu().tolist())
        return nli_scores, binary_scores

    def score(self, contexts: Sequence[str], claims: Sequence[str]) -> list[float]:
        if len(contexts) != len(claims):
            raise ValueError("AlignScore contexts and claims must have the same length")
        if self.evaluation_mode in {"nli", "bin"}:
            nli_scores, binary_scores = self._predict_pairs(contexts, claims)
            return nli_scores if self.evaluation_mode == "nli" else binary_scores

        use_nli = self.evaluation_mode == "nli_sp"
        record_scores: list[float] = []
        for context, claim in zip(contexts, claims):
            source_chunks = _group_source_sentences(context, _split_sentences(context))
            claim_sentences = _split_sentences(claim) or [claim]
            paired_contexts = [chunk for chunk in source_chunks for _ in claim_sentences]
            paired_claims = [sentence for _ in source_chunks for sentence in claim_sentences]
            nli_scores, binary_scores = self._predict_pairs(paired_contexts, paired_claims)
            pair_scores = nli_scores if use_nli else binary_scores
            record_scores.append(
                _aggregate_sentence_scores(pair_scores, len(source_chunks), len(claim_sentences))
            )
        return record_scores
