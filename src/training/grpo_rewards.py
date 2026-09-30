"""Transparent reward components used by the Vietnamese summarization GRPO pilot."""

from __future__ import annotations

import math
import unicodedata
from collections import Counter
from typing import Any, Callable


def completion_text(completion: Any) -> str:
    """Normalize TRL's string or conversational completion representation."""
    if isinstance(completion, str):
        return completion.strip()
    if isinstance(completion, dict):
        content = completion.get("content", "")
        return content.strip() if isinstance(content, str) else ""
    if isinstance(completion, (list, tuple)):
        messages = [item for item in completion if isinstance(item, dict)]
        assistant_text = [
            item.get("content", "")
            for item in messages
            if item.get("role") == "assistant" and isinstance(item.get("content"), str)
        ]
        if assistant_text:
            return "\n".join(assistant_text).strip()
        if len(completion) == 1 and isinstance(completion[0], str):
            return completion[0].strip()
    return ""


def char_ngram_fbeta(hypothesis: str, reference: str, max_order: int = 6, beta: float = 2.0) -> float:
    """Whitespace-insensitive, Unicode-normalized character n-gram F-beta score.

    This is a small, dependency-free chrF-style reference-overlap proxy, not
    claimed to be the canonical SacreBLEU chrF implementation.
    """
    if max_order < 1 or beta <= 0:
        raise ValueError("max_order and beta must be positive")
    hypothesis = "".join(unicodedata.normalize("NFC", hypothesis).split())
    reference = "".join(unicodedata.normalize("NFC", reference).split())
    if not hypothesis or not reference:
        return 0.0

    matches = predicted = expected = 0
    for order in range(1, max_order + 1):
        hyp_ngrams = Counter(hypothesis[i : i + order] for i in range(len(hypothesis) - order + 1))
        ref_ngrams = Counter(reference[i : i + order] for i in range(len(reference) - order + 1))
        predicted += sum(hyp_ngrams.values())
        expected += sum(ref_ngrams.values())
        matches += sum((hyp_ngrams & ref_ngrams).values())
    if predicted == 0 or expected == 0:
        return 0.0
    precision = matches / predicted
    recall = matches / expected
    beta_squared = beta * beta
    denominator = beta_squared * precision + recall
    return (1 + beta_squared) * precision * recall / denominator if denominator else 0.0


def reference_char_reward(completions: list[Any], reference: list[str], **_: Any) -> list[float]:
    """Reward character n-gram overlap with the human reference (bounded [0, 1])."""
    if len(completions) != len(reference):
        raise ValueError("TRL supplied different numbers of completions and references")
    return [char_ngram_fbeta(completion_text(hyp), ref) for hyp, ref in zip(completions, reference)]


def _records_from_kwargs(
    completions: list[Any], source: list[str], summary_col: str = "_generated_summary"
) -> list[dict[str, str]]:
    if len(completions) != len(source):
        raise ValueError("TRL supplied different numbers of completions and source documents")
    return [
        {"_source": str(doc), summary_col: completion_text(summary)}
        for doc, summary in zip(source, completions)
    ]


def make_metric_reward(
    metric_name: str,
    *,
    device: str = "cpu",
    factcc_model_path: str | None = None,
    hf_cache_dir: str | None = None,
    alignscore_ckpt: str | None = None,
    alignscore_backbone_path: str | None = None,
    batch_size: int = 4,
) -> Callable[..., list[float]]:
    """Create a lazy, local-only reward adapter backed by an existing evaluator."""
    evaluator: Any = None

    def metric_reward(completions: list[Any], source: list[str], **_: Any) -> list[float]:
        nonlocal evaluator
        texts = [completion_text(value) for value in completions]
        if len(texts) != len(source):
            raise ValueError("TRL supplied different numbers of completions and source documents")
        if evaluator is None:
            if metric_name == "factcc":
                if not factcc_model_path:
                    raise ValueError("FactCC requires --factcc_model_path pointing to a local checkpoint")
                from ..evaluate.factcc_eval import FactCCEvaluator

                evaluator = FactCCEvaluator(device=device, model_path=factcc_model_path, batch_size=batch_size)
            elif metric_name == "minicheck":
                from ..evaluate.minicheck_eval import MiniCheckEvaluator

                evaluator = MiniCheckEvaluator(
                    device=device,
                    batch_size=batch_size,
                    cache_dir=hf_cache_dir,
                )
            elif metric_name == "alignscore":
                if not alignscore_ckpt:
                    raise ValueError("AlignScore requires --alignscore_ckpt pointing to a local checkpoint")
                from ..evaluate.alignscore_eval import AlignScoreEvaluator

                evaluator = AlignScoreEvaluator(
                    device=device,
                    model_path=alignscore_ckpt,
                    backbone_path=alignscore_backbone_path,
                    batch_size=batch_size,
                )
            else:
                raise ValueError(f"Unsupported faithfulness metric: {metric_name}")

        rows = _records_from_kwargs(texts, source)
        scored = evaluator.evaluate(rows, source_col="_source", summary_col="_generated_summary")
        values: list[float] = []
        for row in scored:
            value = float(row[f"{metric_name}_score"])
            if not math.isfinite(value):
                value = 0.0
            # Metric adapters differ in calibration and may produce out-of-range values.
            # Clipping provides a common bounded scale; it does not make scores comparable.
            values.append(min(1.0, max(0.0, value)))
        return values

    metric_reward.__name__ = f"{metric_name}_faithfulness_reward"
    metric_reward.__doc__ = f"Reward generated summaries using the existing {metric_name} evaluator."
    return metric_reward
