"""Extract auditable atomic claims directly from original summaries.

The primary workflow starts with the original source/summary JSONL. Each
summary sentence is used as a parent unit; there is no sentence-level LLM
claimness gate before decomposition. The model may return zero atomic claims
for a heading or fragment, and those units are preserved in the coverage file.

The decomposition model may repeat inherited context in ``verification_text``
so each unit can be judged independently. ``surface_text`` is always an exact
substring of the parent summary unit and remains the source-grounded span.
Evidence is retrieved separately for every atomic unit.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Protocol

from .claim_recovery import read_jsonl
from .fact_audit import EmbeddingEvidenceRetriever, retrieve_evidence, split_claims
from .llm_claim_judge import (
    DEFAULT_GEMINI_MODEL,
    GeminiClaimJudge,
    JudgeError,
    LocalClaimJudge,
    _parse_structured_object,
)
from .run_eval import _project_path

LOGGER = logging.getLogger(__name__)

ATOMIC_SYSTEM_PROMPT = """You extract atomic factual propositions from one Vietnamese
summary sentence or summary unit. This unit has not been preclassified by
another claim extractor. An atomic claim has one independently checkable
proposition. Split coordinated events, states, relations, quantities, dates,
or causes when they can be verified separately. Keep a causal relation
together when the relation itself is the proposition. If the input is already
atomic, return one item.

If the unit is only a heading, label, fragment, formatting artifact, or other
text with no checkable proposition, return an empty atomic_claims array. Do
not use that option to discard a factual proposition merely because it is
general, predictive, subjective, or difficult to verify.

For every item:
- surface_text must be an exact contiguous substring copied from the parent.
- verification_text must be a standalone wording of the same proposition. It
  may repeat an explicit subject from the parent when the surface uses a
  pronoun, but it must not add facts or change the claim.
- Do not judge truth and do not omit a proposition.

Return exactly one JSON object with an atomic_claims array. Do not emit
chain-of-thought, <think>, <analysis>, Markdown, prose, or a Python dictionary."""

ATOMIC_CLAIM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "atomic_claims": {
            "type": "array",
            "description": "Atomic propositions in original order.",
            "items": {
                "type": "object",
                "properties": {
                    "surface_text": {
                        "type": "string",
                        "description": "Exact contiguous text copied from the parent claim.",
                    },
                    "verification_text": {
                        "type": "string",
                        "description": "Standalone wording with inherited context repeated if needed.",
                    },
                },
                "required": ["surface_text", "verification_text"],
            },
        }
    },
    "required": ["atomic_claims"],
}


class AtomicGenerator(Protocol):
    backend_name: str
    model_name: str

    def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        """Generate one structured decomposition response."""


def _atomic_prompt(parent_claim: str) -> str:
    return f"""Extract all independently checkable atomic propositions from this
summary unit and preserve its meaning. If it has no checkable proposition,
return an empty atomic_claims array.

PARENT SUMMARY UNIT:
<summary_unit>
{parent_claim}
</summary_unit>

Return only the JSON object requested by the schema."""


def _atomic_repair_prompt(parent_claim: str, previous_response: str) -> str:
    return f"""The previous decomposition did not follow the required format.
Return exactly one JSON object with an atomic_claims array. Each item must have
surface_text copied exactly from the parent summary unit and verification_text
that adds no facts. If the parent is only a heading, label, fragment, or other
non-proposition, the array may be empty. Do not return reasoning, Markdown,
prose, or a Python dictionary. Any text outside the JSON object is invalid.

PARENT SUMMARY UNIT:
<summary_unit>
{parent_claim}
</summary_unit>

PREVIOUS INVALID RESPONSE:
<previous_response>
{previous_response}
</previous_response>"""


def parse_atomic_decomposition(text: str, parent_claim: str) -> list[dict[str, Any]]:
    """Parse, validate, and locate atomic surface spans in the parent claim."""
    value = _parse_structured_object(text)
    raw_items = value.get("atomic_claims")
    if raw_items is None:
        raw_items = value.get("claims")
    if not isinstance(raw_items, list):
        raise JudgeError("model JSON response must contain an atomic_claims array")
    if not raw_items:
        return []

    normalized: list[dict[str, Any]] = []
    cursor = 0
    for index, raw_item in enumerate(raw_items):
        if not isinstance(raw_item, dict):
            raise JudgeError(f"atomic claim {index} is not an object")
        raw_surface = raw_item.get("surface_text", raw_item.get("text"))
        if not isinstance(raw_surface, str) or not raw_surface.strip():
            raise JudgeError(f"atomic claim {index} has empty surface_text")
        surface_lookup = raw_surface.strip()
        start = parent_claim.find(surface_lookup, cursor)
        if start < 0:
            start = parent_claim.find(surface_lookup)
        if start < 0:
            raise JudgeError(
                f"atomic claim {index} surface_text is not an exact parent substring"
            )
        end = start + len(surface_lookup)
        if start < cursor:
            raise JudgeError(f"atomic claim {index} overlaps an earlier atomic claim")
        verification = raw_item.get("verification_text", raw_item.get("claim"))
        if not isinstance(verification, str) or not verification.strip():
            raise JudgeError(f"atomic claim {index} has empty verification_text")
        normalized.append(
            {
                "atomic_claim_index": index,
                "surface_text": parent_claim[start:end],
                "verification_text": verification.strip(),
                "start_char": start,
                "end_char": end,
            }
        )
        cursor = end
    return normalized


class AtomicClaimDecomposer:
    """Run atomic decomposition with bounded local format repair."""

    def __init__(self, generator: AtomicGenerator) -> None:
        self.generator = generator
        self.backend_name = generator.backend_name
        self.model_name = generator.model_name
        self.last_raw_response = ""
        self.last_raw_attempts: list[str] = []

    def decompose(self, parent_claim: str) -> dict[str, Any]:
        self.last_raw_response = ""
        self.last_raw_attempts = []
        raw = self.generator.generate_text(
            ATOMIC_SYSTEM_PROMPT,
            _atomic_prompt(parent_claim),
        )
        attempts = [raw]
        self.last_raw_response = raw
        self.last_raw_attempts = attempts.copy()
        try:
            atomic_claims = parse_atomic_decomposition(raw, parent_claim)
        except JudgeError:
            if self.backend_name != "local":
                raise
            repaired = self.generator.generate_text(
                ATOMIC_SYSTEM_PROMPT,
                _atomic_repair_prompt(parent_claim, raw),
            )
            attempts.append(repaired)
            self.last_raw_attempts = attempts.copy()
            self.last_raw_response = "\n--- repair attempt ---\n".join(attempts)
            atomic_claims = parse_atomic_decomposition(repaired, parent_claim)
        return {
            "atomic_claims": atomic_claims,
            "raw_response": attempts[-1],
            "raw_attempts": attempts,
        }

    def decompose_many(self, parent_claims: list[str]) -> list[dict[str, Any]]:
        """Decompose several parent units with a local batch generator.

        A malformed response is repaired independently with a single-item
        call. Errors are returned in their aligned result slot so one bad
        summary unit does not discard the rest of the batch.
        """
        if not parent_claims:
            return []
        batch_generator = getattr(self.generator, "generate_text_batch", None)
        if not callable(batch_generator):
            return [self.decompose(parent_claim) for parent_claim in parent_claims]
        raw_responses = list(
            batch_generator(
                ATOMIC_SYSTEM_PROMPT,
                [_atomic_prompt(parent_claim) for parent_claim in parent_claims],
            )
        )
        if len(raw_responses) != len(parent_claims):
            raise JudgeError(
                f"local batch returned {len(raw_responses)} responses for "
                f"{len(parent_claims)} summary units"
            )
        results: list[dict[str, Any]] = []
        for parent_claim, raw in zip(parent_claims, raw_responses):
            attempts = [raw]
            self.last_raw_response = raw
            self.last_raw_attempts = attempts.copy()
            try:
                atomic_claims = parse_atomic_decomposition(raw, parent_claim)
            except JudgeError as first_error:
                if self.backend_name != "local":
                    results.append(
                        {
                            "_error": f"{type(first_error).__name__}: {first_error}",
                            "atomic_claims": [],
                            "raw_response": raw,
                            "raw_attempts": attempts,
                        }
                    )
                    continue
                try:
                    repaired = self.generator.generate_text(
                        ATOMIC_SYSTEM_PROMPT,
                        _atomic_repair_prompt(parent_claim, raw),
                    )
                    attempts.append(repaired)
                    self.last_raw_response = "\n--- repair attempt ---\n".join(attempts)
                    self.last_raw_attempts = attempts.copy()
                    atomic_claims = parse_atomic_decomposition(repaired, parent_claim)
                except Exception as repair_error:
                    results.append(
                        {
                            "_error": f"{type(repair_error).__name__}: {repair_error}",
                            "atomic_claims": [],
                            "raw_response": attempts[-1],
                            "raw_attempts": attempts,
                            "first_error": str(first_error),
                        }
                    )
                    continue
            results.append(
                {
                    "atomic_claims": atomic_claims,
                    "raw_response": attempts[-1],
                    "raw_attempts": attempts,
                }
            )
        return results


def _source_value(row: dict[str, Any]) -> str:
    for key in ("source", "text", "input"):
        value = str(row.get(key, "")).strip()
        if value:
            return value
    return ""


def _parent_claim_value(row: dict[str, Any]) -> str:
    for key in ("claim", "claim_text", "summary_text"):
        value = str(row.get(key, "")).strip()
        if value:
            return value
    return ""


def _parent_start(row: dict[str, Any]) -> int:
    value = row.get("start_char", row.get("parent_start_char", 0))
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _coverage_row(
    row: dict[str, Any],
    source: str,
    parent_claim: str,
    decomposer: AtomicClaimDecomposer,
    *,
    result: dict[str, Any] | None = None,
    status: str,
    error: str | None = None,
) -> dict[str, Any]:
    """Build one row that accounts for every summary unit.

    Atomic output intentionally contains only units with at least one atomic
    proposition so it can be sent directly to ``llm_claim_judge``. This
    coverage row is the lossless companion for headings, fragments, and failed
    model calls.
    """
    result = result or {}
    atomic_claims = result.get("atomic_claims", [])
    parent_start = _parent_start(row)
    coverage = dict(row)
    coverage.update(
        {
            "source": source,
            "parent_claim": parent_claim,
            "parent_claim_index": row.get("parent_claim_index", row.get("claim_index")),
            "summary_unit_index": row.get("summary_unit_index", row.get("claim_index")),
            "parent_start_char": parent_start,
            "parent_end_char": parent_start + len(parent_claim),
            "atomic_claim_count": len(atomic_claims),
            "atomic_claim_indices": [item.get("atomic_claim_index") for item in atomic_claims],
            "decomposition_backend": decomposer.backend_name,
            "decomposition_model": decomposer.model_name,
            "decomposition_status": status,
            "decomposition_raw": result.get("raw_response", decomposer.last_raw_response),
            "decomposition_raw_attempts": result.get(
                "raw_attempts", decomposer.last_raw_attempts.copy()
            ),
            "decomposition_repair_attempted": len(
                result.get("raw_attempts", decomposer.last_raw_attempts)
            )
            > 1,
        }
    )
    if error:
        coverage["decomposition_error"] = error
    return coverage


def _decompose_parent_row(
    row: dict[str, Any],
    decomposer: AtomicClaimDecomposer,
    *,
    top_k: int,
    retrieval_mode: str,
    embedding_retriever: EmbeddingEvidenceRetriever | None,
    embedding_weight: float,
    decomposition_result: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any] | None]:
    """Decompose one parent unit and return atoms, coverage, and an error."""
    source = _source_value(row)
    parent_claim = _parent_claim_value(row)
    if not source or not parent_claim:
        message = "source or summary unit is empty"
        coverage = _coverage_row(
            row,
            source,
            parent_claim,
            decomposer,
            result={"atomic_claims": [], "raw_response": "", "raw_attempts": []},
            status="error",
            error=message,
        )
        return [], coverage, dict(coverage)

    result = decomposition_result
    try:
        if result is None:
            result = decomposer.decompose(parent_claim)
        if result.get("_error"):
            raise JudgeError(str(result["_error"]))
        parent_start = _parent_start(row)
        atomic_claims = result["atomic_claims"]
        output_rows: list[dict[str, Any]] = []
        for atomic_claim in atomic_claims:
            verification_text = atomic_claim["verification_text"]
            evidence = retrieve_evidence(
                source,
                verification_text,
                top_k=top_k,
                retrieval_mode=retrieval_mode,
                embedding_retriever=embedding_retriever,
                embedding_weight=embedding_weight,
            )
            annotated = dict(row)
            annotated.update(
                {
                    "source": source,
                    "parent_claim": parent_claim,
                    "parent_claim_index": row.get("parent_claim_index", row.get("claim_index")),
                    "atomic_claim_index": atomic_claim["atomic_claim_index"],
                    "atomic_claim_count": len(atomic_claims),
                    "atomic_surface_text": atomic_claim["surface_text"],
                    "verification_claim": verification_text,
                    "claim": verification_text,
                    "claim_text": verification_text,
                    "start_char": parent_start + atomic_claim["start_char"],
                    "end_char": parent_start + atomic_claim["end_char"],
                    "parent_start_char": parent_start,
                    "parent_end_char": parent_start + len(parent_claim),
                    "evidence": evidence,
                    "retrieval_mode": retrieval_mode,
                    "decomposition_backend": decomposer.backend_name,
                    "decomposition_model": decomposer.model_name,
                    "decomposition_status": "ok",
                    "decomposition_raw": result["raw_response"],
                    "decomposition_raw_attempts": result["raw_attempts"],
                    "decomposition_repair_attempted": len(result["raw_attempts"]) > 1,
                }
            )
            output_rows.append(annotated)
        status = "ok" if atomic_claims else "no_atomic_claims"
        coverage = _coverage_row(
            row,
            source,
            parent_claim,
            decomposer,
            result=result,
            status=status,
        )
        return output_rows, coverage, None
    except Exception as exc:  # preserve the input and continue auditing
        message = f"{type(exc).__name__}: {exc}"
        coverage = _coverage_row(
            row,
            source,
            parent_claim,
            decomposer,
            result=result,
            status="error",
            error=message,
        )
        LOGGER.error("atomic decomposition failed: %s", exc)
        return [], coverage, dict(coverage)


def decompose_rows(
    rows: list[dict[str, Any]],
    decomposer: AtomicClaimDecomposer,
    *,
    top_k: int = 3,
    retrieval_mode: str = "lexical",
    embedding_retriever: EmbeddingEvidenceRetriever | None = None,
    embedding_weight: float = 0.5,
    limit: int | None = None,
    sleep_seconds: float = 0.0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Decompose already prepared parent rows.

    This compatibility function remains useful for old experiment artifacts.
    The command-line workflow uses :func:`decompose_original_records` instead,
    so the old sentence-level claim extractor is not a prerequisite.
    """
    selected = rows if limit is None else rows[:limit]
    output_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for document_index, row in enumerate(selected, 1):
        rows_for_document, _coverage, error = _decompose_parent_row(
            row,
            decomposer,
            top_k=top_k,
            retrieval_mode=retrieval_mode,
            embedding_retriever=embedding_retriever,
            embedding_weight=embedding_weight,
        )
        output_rows.extend(rows_for_document)
        if error:
            errors.append(error)
        print(f"decomposed document {document_index}/{len(selected)}", flush=True)
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)
    return output_rows, errors


def _resolve_column(
    rows: list[dict[str, Any]],
    requested: str | None,
    candidates: tuple[str, ...],
    label: str,
) -> str:
    if requested:
        missing = [index for index, row in enumerate(rows) if requested not in row]
        if missing:
            preview = ", ".join(str(index) for index in missing[:5])
            raise ValueError(f"{label} column {requested!r} is missing in row(s): {preview}")
        return requested
    for candidate in candidates:
        if all(candidate in row for row in rows):
            return candidate
    options = ", ".join(candidates)
    raise ValueError(f"Could not infer the {label} field. Pass --{label}_col; tried {options}.")


def summary_spans(summary: str) -> list[dict[str, Any]]:
    """Return exact sentence-like parent units from the original summary."""
    units = split_claims(summary)
    spans: list[dict[str, Any]] = []
    cursor = 0
    for candidate_index, unit in enumerate(units):
        start = summary.find(unit, cursor)
        if start < 0:
            start = summary.find(unit)
        if start < 0:
            raise JudgeError("could not recover an exact summary-unit span")
        end = start + len(unit)
        spans.append(
            {
                "candidate_index": candidate_index,
                "text": summary[start:end],
                "start_char": start,
                "end_char": end,
            }
        )
        cursor = end
    if not spans and summary.strip():
        start = len(summary) - len(summary.lstrip())
        end = len(summary.rstrip())
        spans.append(
            {
                "candidate_index": 0,
                "text": summary[start:end],
                "start_char": start,
                "end_char": end,
            }
        )
    return spans


def decompose_original_records(
    rows: list[dict[str, Any]],
    decomposer: AtomicClaimDecomposer,
    *,
    source_col: str,
    summary_col: str,
    top_k: int = 3,
    retrieval_mode: str = "lexical",
    embedding_retriever: EmbeddingEvidenceRetriever | None = None,
    embedding_weight: float = 0.5,
    limit: int | None = None,
    sleep_seconds: float = 0.0,
    model_batch_size: int = 1,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Extract atoms directly from original records.

    Returns ``atomic_rows``, ``coverage_rows``, and ``errors``. Every original
    summary sentence gets one coverage row. Only rows with one or more atomic
    claims are emitted to ``atomic_rows`` so the result is immediately
    compatible with ``llm_claim_judge``.
    """
    if model_batch_size < 1:
        raise ValueError("model_batch_size must be positive")
    selected = rows if limit is None else rows[:limit]
    parent_rows: list[dict[str, Any]] = []
    for document_index, row in enumerate(selected, 1):
        source = str(row.get(source_col, "") or "").strip()
        summary = str(row.get(summary_col, "") or "")
        if not source or not summary.strip():
            parent_row = dict(row)
            parent_row.update(
                {
                    "source": source,
                    "source_col": source_col,
                    "summary_col": summary_col,
                    "summary_text": summary,
                    "parent_claim": summary,
                    "claim": summary,
                    "claim_text": summary,
                    "record_index": document_index - 1,
                    "summary_unit_index": None,
                }
            )
            parent_rows.append(parent_row)
            continue
        spans = summary_spans(summary)
        for span in spans:
            parent_row = dict(row)
            parent_row.update(
                {
                    "source": source,
                    "source_col": source_col,
                    "summary_col": summary_col,
                    "summary_text": summary,
                    "parent_claim": span["text"],
                    "claim": span["text"],
                    "claim_text": span["text"],
                    "claim_index": span["candidate_index"],
                    "claim_count": len(spans),
                    "record_index": document_index - 1,
                    "summary_unit_index": span["candidate_index"],
                    "start_char": span["start_char"],
                    "end_char": span["end_char"],
                }
            )
            parent_rows.append(parent_row)

    decomposition_results: list[dict[str, Any] | None] = [None] * len(parent_rows)
    valid_indices = [
        index
        for index, row in enumerate(parent_rows)
        if _source_value(row) and _parent_claim_value(row)
    ]
    for batch_start in range(0, len(valid_indices), model_batch_size):
        batch_indices = valid_indices[batch_start : batch_start + model_batch_size]
        parent_claims = [_parent_claim_value(parent_rows[index]) for index in batch_indices]
        if len(batch_indices) == 1:
            batch_results = [decomposer.decompose(parent_claims[0])]
        else:
            batch_results = decomposer.decompose_many(parent_claims)
        if len(batch_results) != len(batch_indices):
            raise JudgeError(
                f"decomposer returned {len(batch_results)} results for "
                f"{len(batch_indices)} summary units"
            )
        for index, result in zip(batch_indices, batch_results):
            decomposition_results[index] = result
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    atomic_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for unit_index, parent_row in enumerate(parent_rows):
        atoms, coverage, error = _decompose_parent_row(
            parent_row,
            decomposer,
            top_k=top_k,
            retrieval_mode=retrieval_mode,
            embedding_retriever=embedding_retriever,
            embedding_weight=embedding_weight,
            decomposition_result=decomposition_results[unit_index],
        )
        atomic_rows.extend(atoms)
        coverage_rows.append(coverage)
        if error:
            errors.append(error)
        print(f"decomposed summary unit {unit_index + 1}/{len(parent_rows)}", flush=True)
    return atomic_rows, coverage_rows, errors


def _write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def _make_decomposer(args: argparse.Namespace) -> AtomicClaimDecomposer:
    if args.backend == "gemini":
        api_key = os.environ.get(args.api_key_env, "")
        if not api_key:
            raise RuntimeError(
                f"Missing API key environment variable {args.api_key_env!r}; do not put the key in the command or repository"
            )
        generator = GeminiClaimJudge(api_key, args.model or DEFAULT_GEMINI_MODEL)

        class GeminiAtomicGenerator:
            backend_name = generator.backend_name
            model_name = generator.model_name

            def generate_text(self, system_prompt: str, user_prompt: str) -> str:
                return generator.generate_text(
                    system_prompt,
                    user_prompt,
                    ATOMIC_CLAIM_SCHEMA,
                    max_output_tokens=args.max_new_tokens,
                )

        return AtomicClaimDecomposer(GeminiAtomicGenerator())
    if not args.model:
        raise ValueError("--model is required for --backend local")
    return AtomicClaimDecomposer(
        LocalClaimJudge(
            _project_path(args.model),
            device=args.device,
            dtype=args.dtype,
            max_input_tokens=args.max_input_tokens,
            max_new_tokens=args.max_new_tokens,
            allow_download=args.allow_download,
        )
    )


def _make_embedding_retriever(args: argparse.Namespace) -> EmbeddingEvidenceRetriever | None:
    if args.retrieval_mode == "lexical":
        return None
    if not args.embedding_model_path:
        raise ValueError(
            "--embedding_model_path is required for --retrieval_mode "
            f"{args.retrieval_mode}"
        )
    return EmbeddingEvidenceRetriever(
        _project_path(args.embedding_model_path),
        device=args.embedding_device,
        dtype=args.embedding_dtype,
        batch_size=args.embedding_batch_size,
        max_length=args.embedding_max_length,
        query_prefix=args.embedding_query_prefix,
        passage_prefix=args.embedding_passage_prefix,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract atomic claims directly from original source/summary JSONL.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--backend", choices=("gemini", "local"), required=True)
    parser.add_argument("--input", required=True, help="Original source/summary JSONL")
    parser.add_argument("--source_col", help="Source field; autodetected when omitted")
    parser.add_argument("--summary_col", help="Summary field; autodetected when omitted")
    parser.add_argument("--output", required=True, help="Atomic-claim JSONL")
    parser.add_argument(
        "--coverage_output",
        help="Coverage JSONL for every summary unit; defaults beside --output",
    )
    parser.add_argument("--error_output", help="Optional decomposition-error JSONL")
    parser.add_argument("--model", help="Gemini model ID or local model folder")
    parser.add_argument("--api_key_env", default="GEMINI_API_KEY")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto")
    parser.add_argument("--max_input_tokens", type=int, default=8192)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--allow_download", action="store_true")
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--retrieval_mode", choices=("lexical", "embedding", "hybrid"), default="lexical")
    parser.add_argument("--embedding_model_path")
    parser.add_argument("--embedding_device", default="auto")
    parser.add_argument("--embedding_dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto")
    parser.add_argument("--embedding_batch_size", type=int, default=8)
    parser.add_argument("--embedding_max_length", type=int, default=512)
    parser.add_argument("--embedding_query_prefix", default="")
    parser.add_argument("--embedding_passage_prefix", default="")
    parser.add_argument("--embedding_weight", type=float, default=0.5)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--sleep_seconds", type=float, default=0.0)
    parser.add_argument(
        "--model_batch_size",
        type=int,
        default=1,
        help="Local model generation batch size; use 1 for the safest setting",
    )
    args = parser.parse_args()
    if args.backend == "gemini" and args.allow_download:
        parser.error("--allow_download applies only to --backend local")
    if args.max_input_tokens < 1 or args.max_new_tokens < 1 or args.top_k < 1:
        parser.error("token limits and --top_k must be positive")
    if args.embedding_batch_size < 1 or args.embedding_max_length < 1:
        parser.error("embedding batch size and max length must be positive")
    if args.model_batch_size < 1:
        parser.error("--model_batch_size must be positive")
    if not 0.0 <= args.embedding_weight <= 1.0:
        parser.error("--embedding_weight must be between 0 and 1")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.sleep_seconds < 0:
        parser.error("--sleep_seconds must be non-negative")
    if args.backend == "gemini" and args.model_batch_size != 1:
        parser.error("--model_batch_size applies only to --backend local")

    rows = read_jsonl(Path(args.input))
    if not rows:
        raise ValueError(f"No rows found in {args.input}")
    source_col = _resolve_column(rows, args.source_col, ("text", "source", "input"), "source")
    summary_col = _resolve_column(
        rows,
        args.summary_col,
        ("summary", "output", "human_sum", "abstract_sum", "llm_sum"),
        "summary",
    )
    decomposer = _make_decomposer(args)
    embedding_retriever = _make_embedding_retriever(args)
    atomic_rows, coverage_rows, errors = decompose_original_records(
        rows,
        decomposer,
        source_col=source_col,
        summary_col=summary_col,
        top_k=args.top_k,
        retrieval_mode=args.retrieval_mode,
        embedding_retriever=embedding_retriever,
        embedding_weight=args.embedding_weight,
        limit=args.limit,
        sleep_seconds=args.sleep_seconds,
        model_batch_size=args.model_batch_size,
    )
    output_path = Path(args.output)
    _write_jsonl(atomic_rows, output_path)
    coverage_path = (
        Path(args.coverage_output)
        if args.coverage_output
        else output_path.with_name(output_path.stem + ".coverage.jsonl")
    )
    _write_jsonl(coverage_rows, coverage_path)
    if args.error_output:
        _write_jsonl(errors, Path(args.error_output))
    output_path.with_suffix(".summary.json").write_text(
        json.dumps(
            {
                "input_records": len(rows) if args.limit is None else min(args.limit, len(rows)),
                "source_col": source_col,
                "summary_col": summary_col,
                "summary_units": len(coverage_rows),
                "atomic_rows": len(atomic_rows),
                "no_atomic_units": sum(
                    row.get("decomposition_status") == "no_atomic_claims"
                    for row in coverage_rows
                ),
                "errors": len(errors),
                "backend": decomposer.backend_name,
                "model": decomposer.model_name,
                "retrieval_mode": args.retrieval_mode,
                "model_batch_size": args.model_batch_size,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Atomic claims written to: {args.output}")
    print(f"Coverage written to: {coverage_path}")
    print(f"Atomic claims: {len(atomic_rows)}; errors: {len(errors)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
