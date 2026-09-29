"""Inference-only MiniCheck adapter for the FLAN-T5-Large checkpoint.

Scoring logic is adapted from Liyan Tang et al.'s MiniCheck implementation
(Apache-2.0; see ``third_party/licenses/MiniCheck-Apache-2.0.txt``). This
project-local port loads the same model and candidate-token scores directly
with PyTorch/Transformers; it does not require the upstream ``minicheck``
package, vLLM, Accelerate, or ``device_map=\"auto\"``.

Modified for explicit device placement and offline local-model loading.

Validate score parity against the upstream implementation before treating
results from this port as interchangeable in a research result.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from pathlib import Path

import torch
from torch import Tensor
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

logger = logging.getLogger(__name__)

_SUPPORTED_MODEL = "flan-t5-large"
_MODEL_ID = "lytang/MiniCheck-Flan-T5-Large"
_NO_SUPPORT_TOKEN_ID = 3
_SUPPORT_TOKEN_ID = 209
_DEFAULT_MAX_LENGTH = 2048
_DEFAULT_SOURCE_CHUNK_WORDS = 500


def _source_sentences(text: str) -> list[str]:
    """Use MiniCheck's NLTK sentence splitting while preserving line breaks."""
    try:
        from nltk.tokenize import sent_tokenize
    except ImportError as exc:
        raise ImportError("The bundled MiniCheck adapter requires NLTK.") from exc

    sentences: list[str] = []
    blocks = text.split("\n")
    try:
        for index, block in enumerate(blocks):
            sentences.extend(sent_tokenize(block, language="english"))
            if index < len(blocks) - 1:
                sentences.append("\n")
    except LookupError as exc:
        raise RuntimeError(
            "NLTK's English Punkt data is missing. Transfer the 'punkt_tab' resource "
            "and set NLTK_DATA to its local data directory."
        ) from exc
    return sentences


def _chunk_source(text: str, chunk_words: int = _DEFAULT_SOURCE_CHUNK_WORDS) -> list[str]:
    """Pack whole source sentences into approximately ``chunk_words`` chunks."""
    if chunk_words < 1:
        raise ValueError("chunk_words must be positive")
    sentences = _source_sentences(text)
    if not sentences:
        return [""]

    chunks: list[str] = []
    current: list[str] = []
    word_count = 0
    for sentence in sentences:
        sentence_words = 0 if sentence == "\n" else len(sentence.split())
        if current and word_count + sentence_words > chunk_words:
            chunk = " ".join(current).replace(" \n ", "\n").strip()
            if chunk:
                chunks.append(chunk)
            current = []
            word_count = 0
        current.append(sentence)
        word_count += sentence_words

    if current:
        chunk = " ".join(current).replace(" \n ", "\n").strip()
        if chunk:
            chunks.append(chunk)
    return chunks or [""]


def _aggregate_claim_support(
    pair_scores: Sequence[float], source_chunk_count: int, claim_sentence_count: int
) -> float:
    """Match MiniCheck's fusion: max over source chunks, then min over claims."""
    expected = source_chunk_count * claim_sentence_count
    if source_chunk_count < 1 or claim_sentence_count < 1 or len(pair_scores) != expected:
        raise ValueError("Pair-score dimensions do not match source/claim dimensions")
    claim_best_scores = [
        max(
            pair_scores[source_index * claim_sentence_count + claim_index]
            for source_index in range(source_chunk_count)
        )
        for claim_index in range(claim_sentence_count)
    ]
    return min(claim_best_scores)


class MiniCheckCompatScorer:
    """Run MiniCheck's FLAN-T5-Large inference path without its pip package."""

    def __init__(
        self,
        *,
        model_name: str = _SUPPORTED_MODEL,
        cache_dir: str | Path | None = None,
        batch_size: int = 16,
        device: str = "cuda",
        max_length: int = _DEFAULT_MAX_LENGTH,
    ) -> None:
        if model_name != _SUPPORTED_MODEL:
            raise ValueError(
                f"The bundled MiniCheck adapter supports only {_SUPPORTED_MODEL!r}; "
                "other MiniCheck model variants are not included."
            )
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if max_length < 1:
            raise ValueError("max_length must be positive")

        self.device = torch.device(device)
        self.batch_size = batch_size
        self.max_length = max_length
        offline = os.environ.get("HF_HUB_OFFLINE") == "1" or os.environ.get("TRANSFORMERS_OFFLINE") == "1"
        model_reference = _MODEL_ID
        cache_path = Path(cache_dir) if cache_dir is not None else None
        if cache_path is not None and (cache_path / "config.json").is_file():
            weight_files = ("model.safetensors", "pytorch_model.bin", "model.safetensors.index.json")
            if any((cache_path / filename).is_file() for filename in weight_files):
                model_reference = str(cache_path.resolve())
                cache_path = None
                offline = True
        load_args = {
            "cache_dir": str(cache_path) if cache_path is not None else None,
            "local_files_only": offline,
        }
        self.tokenizer = AutoTokenizer.from_pretrained(model_reference, **load_args)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_reference, **load_args)
        self.model.to(self.device).eval()
        if self.tokenizer.eos_token is None:
            raise ValueError("MiniCheck's FLAN-T5 tokenizer must define an EOS token")

    def _score_pairs(self, contexts: Sequence[str], claims: Sequence[str]) -> list[float]:
        if len(contexts) != len(claims):
            raise ValueError("MiniCheck contexts and claims must have equal lengths")
        probabilities: list[float] = []
        eos = self.tokenizer.eos_token
        for start in range(0, len(contexts), self.batch_size):
            end = min(start + self.batch_size, len(contexts))
            joined = [eos.join((context, claim)) for context, claim in zip(contexts[start:end], claims[start:end])]
            encoded = self.tokenizer(
                ["predict: " + text for text in joined],
                max_length=self.max_length,
                truncation=True,
                padding=True,
                return_tensors="pt",
            )
            encoded = {name: value.to(self.device) for name, value in encoded.items()}
            decoder_input_ids = torch.zeros(
                (len(joined), 1), dtype=torch.long, device=self.device
            )
            with torch.inference_mode():
                logits = self.model(
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded["attention_mask"],
                    decoder_input_ids=decoder_input_ids,
                ).logits[:, 0, :]
                candidate_logits = torch.stack(
                    (logits[:, _NO_SUPPORT_TOKEN_ID], logits[:, _SUPPORT_TOKEN_ID]), dim=-1
                )
                probabilities.extend(torch.softmax(candidate_logits, dim=-1)[:, 1].cpu().tolist())
        return probabilities

    def score(
        self,
        *,
        docs: Sequence[str],
        claims: Sequence[str],
        chunk_size: int | None = None,
    ) -> tuple[list[int], list[float], list[list[str]], list[list[list[float]]]]:
        """Return MiniCheck-compatible predictions, scores, chunks, and pair scores."""
        if len(docs) != len(claims):
            raise ValueError("MiniCheck docs and claims must have equal lengths")
        source_chunk_words = chunk_size or _DEFAULT_SOURCE_CHUNK_WORDS
        if source_chunk_words < 1:
            raise ValueError("chunk_size must be positive")

        doc_chunks = [_chunk_source(str(doc), source_chunk_words) for doc in docs]
        claim_sentences = [_source_sentences(str(claim)) or [str(claim)] for claim in claims]
        pair_groups: list[list[tuple[str, str]]] = []
        flat_contexts: list[str] = []
        flat_claims: list[str] = []
        for chunks, sentences in zip(doc_chunks, claim_sentences):
            group = [(context, sentence) for context in chunks for sentence in sentences]
            pair_groups.append(group)
            flat_contexts.extend(context for context, _ in group)
            flat_claims.extend(sentence for _, sentence in group)

        flat_scores = self._score_pairs(flat_contexts, flat_claims)
        predictions: list[int] = []
        scores: list[float] = []
        pair_score_matrices: list[list[list[float]]] = []
        offset = 0
        for chunks, sentences, group in zip(doc_chunks, claim_sentences, pair_groups):
            count = len(group)
            group_scores = flat_scores[offset : offset + count]
            offset += count
            matrix = [
                group_scores[row * len(sentences) : (row + 1) * len(sentences)]
                for row in range(len(chunks))
            ]
            score = _aggregate_claim_support(group_scores, len(chunks), len(sentences))
            scores.append(score)
            predictions.append(int(score > 0.5))
            pair_score_matrices.append(matrix)

        return predictions, scores, doc_chunks, pair_score_matrices
