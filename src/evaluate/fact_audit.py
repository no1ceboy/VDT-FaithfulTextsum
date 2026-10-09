"""Claim/evidence audit for investigating summary faithfulness.

This is an interpretable companion to ``run_eval``.  It does not claim to
extract objective world facts.  It splits a summary into claim-like units,
retrieves source sentences that may support each unit, and optionally scores
the full source/claim pair with Vietnamese mFACT.  The retrieved evidence is
included for human review; lexical and embedding retrieval are ranking
heuristics only.

The implementation is intentionally dependency-light and CPU-friendly:
sentence splitting and lexical retrieval use the standard library, while the
optional embedding retriever reuses the existing PyTorch/Transformers stack.
mFACT is loaded once and evaluated in small batches.  A binary mFACT score
cannot by itself distinguish contradiction from unsupported content, so
automatic ``needs_review`` flags must not be reported as confirmed errors.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import unicodedata
from collections import defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from .run_eval import (
    _project_path,
    _source_column,
    _summary_columns,
    _validate_records,
    load_records,
)

logger = logging.getLogger(__name__)

_BOUNDARY_RE = re.compile(r"(?:\r?\n+|(?<=[.!?…。！？])\s+)")
_CLAUSE_RE = re.compile(r"\s*(?:[;；:：])\s*")
_TOKEN_RE = re.compile(r"[^\W_]+", flags=re.UNICODE)
_NUMBER_RE = re.compile(r"(?<![\w])\d+(?:[.,]\d+)?%?")


def split_claims(text: str, split_clauses: bool = False) -> list[str]:
    """Split summary text into reproducible claim-like units.

    This is deliberately conservative: it does not ask a generative model to
    rewrite or invent claims.  ``split_clauses`` additionally separates
    semicolon/colon-delimited clauses, which is useful for investigative runs
    but can over-split fluent Vietnamese prose.
    """
    sentence_like = [part.strip() for part in _BOUNDARY_RE.split(str(text)) if part.strip()]
    if not split_clauses:
        return sentence_like
    claims: list[str] = []
    for sentence in sentence_like:
        claims.extend(part.strip() for part in _CLAUSE_RE.split(sentence) if part.strip())
    return claims


def _tokens(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFC", str(text)).casefold()
    return _TOKEN_RE.findall(normalized)


def _character_ngrams(text: str, order: int = 3) -> set[str]:
    normalized = "".join(unicodedata.normalize("NFC", str(text)).casefold().split())
    if len(normalized) < order:
        return {normalized} if normalized else set()
    return {normalized[index : index + order] for index in range(len(normalized) - order + 1)}


def _recall(query: set[str], candidate: set[str]) -> float:
    if not query:
        return 0.0
    return len(query & candidate) / len(query)


class EmbeddingEvidenceRetriever:
    """Offline sentence retriever backed by an existing Transformers encoder.

    This is a retrieval component only.  Its cosine score is evidence ranking
    signal, not an entailment or contradiction decision.
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        device: str = "auto",
        dtype: str = "auto",
        batch_size: int = 8,
        max_length: int = 512,
        query_prefix: str = "",
        passage_prefix: str = "",
    ) -> None:
        if batch_size < 1 or max_length < 1:
            raise ValueError("embedding batch_size and max_length must be positive")
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Embedding retrieval requires the existing torch and transformers environment"
            ) from exc

        path = Path(model_path)
        if not path.is_dir():
            raise FileNotFoundError(f"Embedding model folder not found: {path}")
        if not (path / "config.json").is_file():
            raise FileNotFoundError(
                f"{path} is not a complete local embedding model: config.json is missing"
            )
        if not any(
            (path / filename).is_file()
            for filename in (
                "model.safetensors",
                "pytorch_model.bin",
                "model.safetensors.index.json",
                "pytorch_model.bin.index.json",
            )
        ):
            raise FileNotFoundError(
                f"{path} is not a complete local embedding model: model weights are missing"
            )
        if not any(
            (path / filename).is_file()
            for filename in (
                "tokenizer.json",
                "tokenizer_config.json",
                "tokenizer.model",
                "vocab.txt",
                "vocab.json",
                "spiece.model",
            )
        ):
            raise FileNotFoundError(
                f"{path} is not a complete local embedding model: tokenizer files are missing"
            )
        self.torch = torch
        self.model_path = str(path)
        self.batch_size = batch_size
        self.max_length = max_length
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("--embedding_device cuda was requested but CUDA is unavailable")
        self.device = torch.device(device)

        if self.device.type == "cpu" or dtype == "float32":
            torch_dtype = torch.float32
        elif dtype == "bfloat16" and torch.cuda.is_bf16_supported():
            torch_dtype = torch.bfloat16
        elif dtype in {"float16", "bfloat16"}:
            torch_dtype = torch.float16
        elif dtype == "auto":
            torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            raise ValueError(f"Unsupported embedding dtype: {dtype}")

        self.tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModel.from_pretrained(
            str(path), local_files_only=True, torch_dtype=torch_dtype
        )
        self.model.to(self.device)
        self.model.eval()
        self._passage_cache: dict[tuple[str, ...], Any] = {}

    def _encode(self, texts: Sequence[str]) -> Any:
        if not texts:
            return self.torch.empty((0, 1))
        batches: list[Any] = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with self.torch.inference_mode():
                outputs = self.model(**inputs)
            hidden = outputs.last_hidden_state
            mask = inputs.get("attention_mask")
            if mask is None:
                pooled = hidden.mean(dim=1)
            else:
                weights = mask.unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
            batches.append(self.torch.nn.functional.normalize(pooled, p=2, dim=1).cpu())
        return self.torch.cat(batches, dim=0)

    def score_sentences(self, sentences: Sequence[str], query: str) -> list[float]:
        sentence_tuple = tuple(str(sentence) for sentence in sentences)
        passage_embeddings = self._passage_cache.get(sentence_tuple)
        if passage_embeddings is None:
            passage_embeddings = self._encode(
                [self.passage_prefix + sentence for sentence in sentence_tuple]
            )
            self._passage_cache[sentence_tuple] = passage_embeddings
        query_embedding = self._encode([self.query_prefix + str(query)])
        if not len(sentence_tuple):
            return []
        return [
            float(score)
            for score in (passage_embeddings @ query_embedding[0]).tolist()
        ]


def evidence_overlap(claim: str, source_sentence: str) -> dict[str, float]:
    """Return cheap lexical recall features used only to rank evidence."""
    claim_tokens = set(_tokens(claim))
    source_tokens = set(_tokens(source_sentence))
    claim_chars = _character_ngrams(claim)
    source_chars = _character_ngrams(source_sentence)
    token_recall = _recall(claim_tokens, source_tokens)
    char_recall = _recall(claim_chars, source_chars)
    return {
        "token_recall": token_recall,
        "char_recall": char_recall,
        "retrieval_score": 0.7 * token_recall + 0.3 * char_recall,
    }


def retrieve_evidence(
    source: str,
    claim: str,
    top_k: int = 3,
    *,
    retrieval_mode: str = "lexical",
    embedding_retriever: EmbeddingEvidenceRetriever | None = None,
    embedding_weight: float = 0.5,
) -> list[dict[str, Any]]:
    """Return top source sentences using lexical, embedding, or hybrid ranking."""
    if top_k < 1:
        raise ValueError("top_k must be positive")
    if retrieval_mode not in {"lexical", "embedding", "hybrid"}:
        raise ValueError("retrieval_mode must be lexical, embedding, or hybrid")
    if retrieval_mode != "lexical" and embedding_retriever is None:
        raise ValueError("embedding_retriever is required for embedding and hybrid retrieval")
    if not 0.0 <= embedding_weight <= 1.0:
        raise ValueError("embedding_weight must be between 0 and 1")
    sentences = split_claims(source)
    if not sentences and str(source).strip():
        sentences = [str(source).strip()]
    candidates: list[dict[str, Any]] = []
    for sentence_index, sentence in enumerate(sentences):
        features = evidence_overlap(claim, sentence)
        candidates.append(
            {
                "sentence_index": sentence_index,
                "text": sentence,
                "retrieval_method": retrieval_mode,
                **{key: round(value, 6) for key, value in features.items()},
            }
        )
    if retrieval_mode != "lexical":
        assert embedding_retriever is not None
        embedding_scores = embedding_retriever.score_sentences(sentences, claim)
        for item, cosine_score in zip(candidates, embedding_scores):
            embedding_retrieval_score = max(0.0, min(1.0, (cosine_score + 1.0) / 2.0))
            lexical_score = float(item["retrieval_score"])
            item["embedding_score"] = round(cosine_score, 6)
            item["embedding_retrieval_score"] = round(embedding_retrieval_score, 6)
            item["lexical_retrieval_score"] = round(lexical_score, 6)
            item["retrieval_score"] = round(
                embedding_retrieval_score
                if retrieval_mode == "embedding"
                else (1.0 - embedding_weight) * lexical_score
                + embedding_weight * embedding_retrieval_score,
                6,
            )
    candidates.sort(key=lambda item: (-item["retrieval_score"], item["sentence_index"]))
    return candidates[:top_k]


def _number_values(text: str) -> set[str]:
    return {match.replace(",", ".") for match in _NUMBER_RE.findall(str(text))}


def _audit_status(
    claim: str,
    source: str,
    evidence: Sequence[dict[str, Any]],
    mfact_score: float | None,
    mfact_threshold: float,
    min_evidence_overlap: float,
    retrieval_mode: str = "lexical",
) -> tuple[str, list[str]]:
    """Assign a cautious investigation status and explain its flags."""
    flags: list[str] = []
    best_retrieval = float(evidence[0]["retrieval_score"]) if evidence else 0.0
    if not evidence or best_retrieval < min_evidence_overlap:
        flag_name = {
            "embedding": "low_embedding_evidence",
            "hybrid": "low_hybrid_evidence",
        }.get(retrieval_mode, "low_lexical_evidence")
        flags.append(flag_name)
    if not _number_values(claim).issubset(_number_values(source)):
        if _number_values(claim):
            flags.append("claim_number_or_date_not_found")
    if mfact_score is None:
        return "retrieval_only", flags
    if mfact_score < mfact_threshold:
        flags.append("low_mfact")
    if flags:
        return "needs_review", flags
    return "likely_supported", flags


def audit_records(
    records: Sequence[dict[str, Any]],
    source_col: str,
    summary_cols: Sequence[str],
    *,
    score_fn: Callable[[list[str], list[str]], list[float]] | None = None,
    top_k: int = 3,
    split_clauses: bool = False,
    mfact_threshold: float = 0.5,
    min_evidence_overlap: float = 0.05,
    retrieval_mode: str = "lexical",
    embedding_retriever: EmbeddingEvidenceRetriever | None = None,
    embedding_weight: float = 0.5,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Create claim rows and per-summary aggregates.

    ``score_fn`` receives repeated full sources and claim texts.  Keeping it
    injectable makes the extraction/retrieval logic testable without loading a
    checkpoint and allows another local scorer to be compared later.
    """
    claim_rows: list[dict[str, Any]] = []
    aggregates: list[dict[str, Any]] = []
    for summary_col in summary_cols:
        pending: list[dict[str, Any]] = []
        for record_index, record in enumerate(records):
            claims = split_claims(str(record[summary_col]), split_clauses=split_clauses)
            for claim_index, claim in enumerate(claims):
                pending.append(
                    {
                        "record_index": record_index,
                        "record_id": record.get("id", record_index),
                        "claim_index": claim_index,
                        "claim_count": len(claims),
                        "source": str(record[source_col]),
                        "claim": claim,
                    }
                )

        scores: list[float | None]
        if score_fn is None:
            scores = [None] * len(pending)
        else:
            scores = [float(value) for value in score_fn(
                [item["source"] for item in pending],
                [item["claim"] for item in pending],
            )]
            if len(scores) != len(pending):
                raise RuntimeError("Claim scorer returned an unexpected number of scores")

        for item, score in zip(pending, scores):
            evidence = retrieve_evidence(
                item["source"],
                item["claim"],
                top_k=top_k,
                retrieval_mode=retrieval_mode,
                embedding_retriever=embedding_retriever,
                embedding_weight=embedding_weight,
            )
            normalized_score = (
                score if score is not None and math.isfinite(score) else None
            )
            status, flags = _audit_status(
                item["claim"],
                item["source"],
                evidence,
                normalized_score,
                mfact_threshold,
                min_evidence_overlap,
                retrieval_mode,
            )
            claim_rows.append(
                {
                    "id": item["record_id"],
                    "record_index": item["record_index"],
                    "summary_col": summary_col,
                    "claim_index": item["claim_index"],
                    "claim_count": item["claim_count"],
                    "source": item["source"],
                    "claim": item["claim"],
                    "evidence": evidence,
                    "retrieval_mode": retrieval_mode,
                    "embedding_weight": embedding_weight if retrieval_mode == "hybrid" else None,
                    "mfact_score": round(normalized_score, 6) if normalized_score is not None else None,
                    "mfact_pred": int(normalized_score >= mfact_threshold)
                    if normalized_score is not None
                    else None,
                    "status": status,
                    "flags": flags,
                }
            )

        group = [row for row in claim_rows if row["summary_col"] == summary_col]
        finite_scores = [row["mfact_score"] for row in group if row["mfact_score"] is not None]
        supported = sum(row["status"] == "likely_supported" for row in group)
        review = sum(row["status"] == "needs_review" for row in group)
        aggregates.append(
            {
                "summary_col": summary_col,
                "retrieval_mode": retrieval_mode,
                "claims_total": len(group),
                "claims_scored": len(finite_scores),
                "mean_mfact": round(sum(finite_scores) / len(finite_scores), 6)
                if finite_scores
                else None,
                "likely_supported": supported,
                "likely_supported_rate": round(supported / len(group), 6) if group else None,
                "needs_review": review,
                "needs_review_rate": round(review / len(group), 6) if group else None,
                "number_or_date_flags": sum(
                    "claim_number_or_date_not_found" in row["flags"] for row in group
                ),
            }
        )
    return claim_rows, aggregates


def _validate_mfact_folder(path: Path) -> None:
    missing: list[str] = []
    if not (path / "config.json").is_file():
        missing.append("config.json")
    if not any(
        (path / name).is_file()
        for name in (
            "pytorch_model.bin",
            "model.safetensors",
            "pytorch_model.bin.index.json",
            "model.safetensors.index.json",
        )
    ):
        missing.append("model weights")
    if not any((path / name).is_file() for name in ("tokenizer.json", "vocab.txt", "tokenizer.model")):
        missing.append("tokenizer files")
    if missing:
        raise FileNotFoundError(
            f"Incomplete mFACT model folder {path}; missing: {', '.join(missing)}"
        )


def _make_mfact_score_fn(args: argparse.Namespace) -> Callable[[list[str], list[str]], list[float]] | None:
    if args.no_mfact:
        return None
    model_path = _project_path(args.mfact_model_path)
    if not model_path.is_dir():
        raise FileNotFoundError(
            f"mFACT model folder not found: {model_path}. Use --no_mfact for extraction-only mode."
        )
    _validate_mfact_folder(model_path)
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from .mfact_eval import MFactEvaluator

    evaluator = MFactEvaluator(
        device=args.device,
        model_path=str(model_path),
        batch_size=args.batch_size,
    )

    def score_fn(sources: list[str], claims: list[str]) -> list[float]:
        rows = [{"_source": source, "_claim": claim} for source, claim in zip(sources, claims)]
        scored = evaluator.evaluate(rows, source_col="_source", summary_col="_claim")
        return [float(row["mfact_score"]) for row in scored]

    return score_fn


def _make_embedding_retriever(args: argparse.Namespace) -> EmbeddingEvidenceRetriever | None:
    if args.retrieval_mode == "lexical":
        return None
    if not args.embedding_model_path:
        raise ValueError(
            "--embedding_model_path is required for --retrieval_mode "
            f"{args.retrieval_mode}"
        )
    model_path = _project_path(args.embedding_model_path)
    return EmbeddingEvidenceRetriever(
        model_path,
        device=args.embedding_device,
        dtype=args.embedding_dtype,
        batch_size=args.embedding_batch_size,
        max_length=args.embedding_max_length,
        query_prefix=args.embedding_query_prefix,
        passage_prefix=args.embedding_passage_prefix,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract claim-like units, retrieve source evidence, and audit summary behavior.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data", required=True, help="Input JSONL or JSON file")
    parser.add_argument("--source_col", help="Source field; auto-detects text, source, then input")
    parser.add_argument("--summary_col", help="One summary field")
    parser.add_argument("--summary_cols", nargs="+", help="Several summary fields to compare")
    parser.add_argument("--mfact_model_path", default="models/mfact-vi_VN")
    parser.add_argument("--no_mfact", action="store_true", help="Only extract claims and retrieve evidence")
    parser.add_argument("--device", default="cpu", help="mFACT device; CPU is the low-memory default")
    parser.add_argument("--batch_size", type=int, default=1, help="mFACT claim batch size")
    parser.add_argument("--top_k", type=int, default=3, help="Evidence sentences saved per claim")
    parser.add_argument(
        "--retrieval_mode",
        choices=("lexical", "embedding", "hybrid"),
        default="lexical",
        help="Evidence ranking method; lexical is the dependency-free default",
    )
    parser.add_argument(
        "--embedding_model_path",
        help="Complete local Transformers encoder folder for embedding retrieval",
    )
    parser.add_argument("--embedding_device", default="auto")
    parser.add_argument(
        "--embedding_dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
    )
    parser.add_argument("--embedding_batch_size", type=int, default=8)
    parser.add_argument("--embedding_max_length", type=int, default=512)
    parser.add_argument("--embedding_query_prefix", default="")
    parser.add_argument("--embedding_passage_prefix", default="")
    parser.add_argument("--embedding_weight", type=float, default=0.5)
    parser.add_argument("--mfact_threshold", type=float, default=0.5)
    parser.add_argument("--min_evidence_overlap", type=float, default=0.05)
    parser.add_argument("--split_clauses", action="store_true", help="Also split semicolon/colon-delimited clauses")
    parser.add_argument("--limit", type=int, help="Use only the first N records; omit for all")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--output", default="results/fact_audit.jsonl")
    args = parser.parse_args()
    if args.summary_col and args.summary_cols:
        parser.error("Use either --summary_col or --summary_cols, not both")
    if args.batch_size < 1 or args.top_k < 1:
        parser.error("--batch_size and --top_k must be positive")
    if args.embedding_batch_size < 1 or args.embedding_max_length < 1:
        parser.error("embedding batch size and max length must be positive")
    if not 0.0 <= args.embedding_weight <= 1.0:
        parser.error("--embedding_weight must be between 0 and 1")
    if not 0 <= args.mfact_threshold <= 1 or not 0 <= args.min_evidence_overlap <= 1:
        parser.error("thresholds must be in [0, 1]")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    return args


def _write_outputs(claim_rows: list[dict[str, Any]], aggregates: list[dict[str, Any]], output: str) -> None:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        for row in claim_rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    summary_path = output_path.with_suffix(".summary.tsv")
    fields = [
        "summary_col",
        "claims_total",
        "claims_scored",
        "mean_mfact",
        "likely_supported",
        "likely_supported_rate",
        "needs_review",
        "needs_review_rate",
        "number_or_date_flags",
    ]
    lines = ["\t".join(fields)]
    for aggregate in aggregates:
        lines.append("\t".join(str(aggregate[field]) for field in fields))
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary_json_path = output_path.with_suffix(".summary.json")
    summary_json_path.write_text(
        json.dumps(aggregates, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    logger.info("Saved %d claim rows to %s", len(claim_rows), output_path)
    logger.info("Saved audit summary to %s and %s", summary_path, summary_json_path)


def main() -> int:
    args = _parse_args()
    data_path = _project_path(args.data)
    records = load_records(str(data_path))
    if args.limit is not None:
        records = records[: args.limit]
    source_col = _source_column(records, args.source_col)
    summary_cols = _summary_columns(records, args.summary_col, args.summary_cols)
    _validate_records(records, [source_col, *summary_cols])
    score_fn = _make_mfact_score_fn(args)
    embedding_retriever = _make_embedding_retriever(args)
    claim_rows, aggregates = audit_records(
        records,
        source_col,
        summary_cols,
        score_fn=score_fn,
        top_k=args.top_k,
        split_clauses=args.split_clauses,
        mfact_threshold=args.mfact_threshold,
        min_evidence_overlap=args.min_evidence_overlap,
        retrieval_mode=args.retrieval_mode,
        embedding_retriever=embedding_retriever,
        embedding_weight=args.embedding_weight,
    )
    _write_outputs(claim_rows, aggregates, args.output)
    print(json.dumps(aggregates, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
