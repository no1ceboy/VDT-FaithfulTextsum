"""Dependency-free ROUGE-1/2/L F1 against a reference summary.

This project implementation lowercases text, tokenizes Unicode letter/number
runs, and does not stem words. It is intended for portable research comparison,
not bit-for-bit compatibility with a specific ROUGE package.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Any

from .base import BaseEvaluator

_TOKEN_PATTERN = re.compile(r"[^\W_]+", flags=re.UNICODE)


def _tokens(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFC", text).casefold()
    return _TOKEN_PATTERN.findall(normalized)


def _f1(candidate_count: int, reference_count: int, overlap: int) -> float:
    if candidate_count == 0 or reference_count == 0 or overlap == 0:
        return 0.0
    precision = overlap / candidate_count
    recall = overlap / reference_count
    return 2.0 * precision * recall / (precision + recall)


def _rouge_n(candidate: list[str], reference: list[str], n: int) -> float:
    candidate_ngrams = Counter(tuple(candidate[i : i + n]) for i in range(len(candidate) - n + 1))
    reference_ngrams = Counter(tuple(reference[i : i + n]) for i in range(len(reference) - n + 1))
    overlap = sum((candidate_ngrams & reference_ngrams).values())
    return _f1(sum(candidate_ngrams.values()), sum(reference_ngrams.values()), overlap)


def _lcs_length(first: list[str], second: list[str]) -> int:
    """Compute LCS length with O(min(len(first), len(second))) memory."""
    if len(second) > len(first):
        first, second = second, first
    previous = [0] * (len(second) + 1)
    for first_token in first:
        current = [0]
        for index, second_token in enumerate(second, start=1):
            if first_token == second_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def rouge_scores(candidate_text: str, reference_text: str) -> dict[str, float]:
    """Return ROUGE-1/2/L F1 for a candidate and one reference."""
    candidate = _tokens(candidate_text)
    reference = _tokens(reference_text)
    lcs = _lcs_length(candidate, reference)
    return {
        "rouge1_f1": _rouge_n(candidate, reference, 1),
        "rouge2_f1": _rouge_n(candidate, reference, 2),
        "rougeL_f1": _f1(len(candidate), len(reference), lcs),
    }


class RougeEvaluator(BaseEvaluator):
    """Score summaries against a separate human/reference summary column."""

    metric_name = "rouge"

    def __init__(self, reference_col: str = "human_sum", batch_size: int = 8) -> None:
        super().__init__(device="cpu", batch_size=batch_size)
        self.reference_col = reference_col

    def _load(self) -> None:
        # ROUGE uses only the Python standard library.
        return None

    def _evaluate(
        self,
        records: list[dict[str, Any]],
        source_col: str,
        summary_col: str,
    ) -> list[dict[str, Any]]:
        del source_col  # Reference-based; source document is not used.
        results: list[dict[str, Any]] = []
        for record in records:
            out = dict(record)
            if summary_col == self.reference_col:
                scores: dict[str, float | None] = {
                    "rouge_score": None,
                    "rouge_1_f1": None,
                    "rouge_2_f1": None,
                    "rouge_l_f1": None,
                }
            else:
                component_scores = rouge_scores(record[summary_col], record[self.reference_col])
                # ROUGE-L F1 is the primary scalar reported in summary.tsv.
                scores = {
                    "rouge_score": component_scores["rougeL_f1"],
                    "rouge_1_f1": component_scores["rouge1_f1"],
                    "rouge_2_f1": component_scores["rouge2_f1"],
                    "rouge_l_f1": component_scores["rougeL_f1"],
                }
            out.update(scores)
            results.append(out)
        return results
