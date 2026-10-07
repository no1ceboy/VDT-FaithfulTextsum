"""Locate factual claim spans before running source-support verification.

The splitter creates exact candidate spans in the original summary. A Gemini
or local model then answers only whether each candidate is a factual claim.
The accepted rows retain verbatim text and character offsets, so the next
verification stage never has to trust a model-generated paraphrase.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Protocol

from .fact_audit import retrieve_evidence, split_claims
from .llm_claim_judge import (
    DEFAULT_GEMINI_MODEL,
    GeminiClaimJudge,
    JudgeError,
    LocalClaimJudge,
    _extract_json_object,
)
from .claim_recovery import read_jsonl

LOGGER = logging.getLogger(__name__)
UNIT_TYPES = ("claim", "heading", "fragment", "instruction", "metadata", "other")

CLAIMNESS_SYSTEM_PROMPT = """You classify candidate spans from a Vietnamese summary.
Decide whether the exact candidate text asserts a checkable factual proposition.
A factual claim describes an event, state, relation, quantity, date, person,
place, legal requirement, or other proposition that can be supported or
contradicted by a source document.

Return is_claim=false for a title or topic heading, section label, fragment
without a proposition, instruction, formatting artifact, or other text that
does not have a truth value. A heading that itself states a complete factual
proposition can be a claim. Do not judge whether the claim is true; only judge
whether it is a claim. Never rewrite the candidate."""

CLAIMNESS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "is_claim": {
            "type": "boolean",
            "description": "Whether the candidate asserts a checkable factual proposition.",
        },
        "unit_type": {
            "type": "string",
            "enum": list(UNIT_TYPES),
            "description": "The role of the candidate in the summary.",
        },
        "reason": {
            "type": "string",
            "description": "A concise explanation for the classification.",
        },
    },
    "required": ["is_claim", "unit_type", "reason"],
}


class ClaimnessGenerator(Protocol):
    model_name: str
    backend_name: str

    def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        """Return structured model output as text."""


def candidate_spans(summary: str, *, split_clauses: bool = True) -> list[dict[str, Any]]:
    """Split a summary and recover exact character spans in original text."""
    candidates = split_claims(summary, split_clauses=split_clauses)
    spans: list[dict[str, Any]] = []
    cursor = 0
    for candidate_index, candidate in enumerate(candidates):
        start = summary.find(candidate, cursor)
        if start < 0:
            start = summary.find(candidate)
        if start < 0:
            LOGGER.warning("Could not locate candidate in original summary: %r", candidate[:120])
            continue
        end = start + len(candidate)
        spans.append(
            {
                "candidate_index": candidate_index,
                "candidate": candidate,
                "start_char": start,
                "end_char": end,
            }
        )
        cursor = end
    return spans


def _claimness_prompt(summary: str, candidate: str, start: int, end: int) -> str:
    return f"""Classify one candidate from the summary.

CANDIDATE (copied from the summary, characters {start}:{end}):
<candidate>
{candidate}
</candidate>

SUMMARY CONTEXT:
<summary>
{summary}
</summary>

Return only the JSON object requested by the schema. Do not add or rewrite text."""


def parse_claimness(text: str) -> dict[str, Any]:
    value = _extract_json_object(text)
    raw_is_claim = value.get("is_claim")
    if isinstance(raw_is_claim, bool):
        is_claim = raw_is_claim
    elif isinstance(raw_is_claim, str):
        normalized = raw_is_claim.strip().casefold()
        if normalized in {"yes", "true", "1"}:
            is_claim = True
        elif normalized in {"no", "false", "0"}:
            is_claim = False
        else:
            raise JudgeError(f"invalid is_claim value: {raw_is_claim!r}")
    else:
        raise JudgeError("model did not return boolean is_claim")
    unit_type = str(value.get("unit_type", "other")).strip().casefold().replace("-", "_")
    if unit_type not in UNIT_TYPES:
        unit_type = "other"
    if is_claim:
        unit_type = "claim"
    reason = str(value.get("reason", "")).strip()
    return {"is_claim": is_claim, "unit_type": unit_type, "reason": reason}


class ClaimnessClassifier:
    def __init__(self, generator: ClaimnessGenerator) -> None:
        self.generator = generator
        self.backend_name = generator.backend_name
        self.model_name = generator.model_name

    def classify(self, summary: str, candidate: str, start: int, end: int) -> dict[str, Any]:
        raw = self.generator.generate_text(
            CLAIMNESS_SYSTEM_PROMPT,
            _claimness_prompt(summary, candidate, start, end),
        )
        result = parse_claimness(raw)
        result["raw_response"] = raw
        return result


def _source_value(row: dict[str, Any], requested: str | None) -> str:
    if requested:
        return str(row.get(requested, ""))
    for name in ("source", "text", "input"):
        if str(row.get(name, "")).strip():
            return str(row[name])
    return ""


def _summary_value(row: dict[str, Any], requested: str | None) -> tuple[str, str]:
    if requested:
        return requested, str(row.get(requested, ""))
    for name in ("summary", "output", "human_sum", "abstract_sum", "llm_sum"):
        if str(row.get(name, "")).strip():
            return name, str(row[name])
    return "summary", ""


def _base_output_row(
    row: dict[str, Any],
    source: str,
    summary_col: str,
    summary: str,
    span: dict[str, Any],
    classifier: ClaimnessClassifier,
    result: dict[str, Any],
) -> dict[str, Any]:
    candidate = span["candidate"]
    output = {
        "id": row.get("id"),
        "source": source,
        "summary_col": summary_col,
        "summary_text": summary,
        "claim": candidate,
        "claim_text": candidate,
        "start_char": span["start_char"],
        "end_char": span["end_char"],
        "candidate_index": span["candidate_index"],
        "evidence": retrieve_evidence(source, candidate, top_k=3),
        "claim_type": result["unit_type"],
        "is_claim": result["is_claim"],
        "claimness_reason": result["reason"],
        "claimness_backend": classifier.backend_name,
        "claimness_model": classifier.model_name,
        "claimness_status": "ok",
        "claimness_raw": result.get("raw_response", ""),
    }
    return output


def extract_rows(
    rows: list[dict[str, Any]],
    classifier: ClaimnessClassifier,
    *,
    source_col: str | None = None,
    summary_col: str | None = None,
    limit: int | None = None,
    split_clauses: bool = True,
    sleep_seconds: float = 0.0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Return factual claims, non-claims, and candidate errors."""
    selected = rows if limit is None else rows[:limit]
    claims: list[dict[str, Any]] = []
    non_claims: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for document_index, row in enumerate(selected, 1):
        source = _source_value(row, source_col)
        resolved_summary_col, summary = _summary_value(row, summary_col)
        if not source.strip() or not summary.strip():
            errors.append(
                {
                    "id": row.get("id"),
                    "document_index": document_index - 1,
                    "claimness_status": "error",
                    "claimness_error": "source or summary is empty",
                }
            )
            continue
        spans = candidate_spans(summary, split_clauses=split_clauses)
        per_document_claims: list[dict[str, Any]] = []
        for span in spans:
            try:
                result = classifier.classify(
                    summary,
                    span["candidate"],
                    span["start_char"],
                    span["end_char"],
                )
                output = _base_output_row(
                    row, source, resolved_summary_col, summary, span, classifier, result
                )
                if result["is_claim"]:
                    per_document_claims.append(output)
                else:
                    non_claims.append(output)
            except Exception as exc:  # keep extraction auditable and continue
                errors.append(
                    {
                        "id": row.get("id"),
                        "source": source,
                        "summary_col": resolved_summary_col,
                        "summary_text": summary,
                        "candidate": span["candidate"],
                        "start_char": span["start_char"],
                        "end_char": span["end_char"],
                        "candidate_index": span["candidate_index"],
                        "claimness_backend": classifier.backend_name,
                        "claimness_model": classifier.model_name,
                        "claimness_status": "error",
                        "claimness_error": f"{type(exc).__name__}: {exc}",
                    }
                )
                LOGGER.error("document %d candidate %d failed: %s", document_index, span["candidate_index"], exc)
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
        for claim_index, output in enumerate(per_document_claims):
            output["claim_index"] = claim_index
            output["claim_count"] = len(per_document_claims)
            claims.append(output)
        print(f"extracted document {document_index}/{len(selected)}", flush=True)
    return claims, non_claims, errors


def _write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def _make_classifier(args: argparse.Namespace) -> ClaimnessClassifier:
    if args.backend == "gemini":
        api_key = os.environ.get(args.api_key_env, "")
        if not api_key:
            raise RuntimeError(
                f"Missing API key environment variable {args.api_key_env!r}; do not put the key in the command or repository"
            )
        generator = GeminiClaimJudge(api_key, args.model or DEFAULT_GEMINI_MODEL)
        # The shared Gemini generator needs the extraction schema, not the
        # verification schema. Wrap its method so both backends share one API.
        class GeminiClaimnessGenerator:
            backend_name = generator.backend_name
            model_name = generator.model_name

            def generate_text(self, system_prompt: str, user_prompt: str) -> str:
                return generator.generate_text(
                    system_prompt,
                    user_prompt,
                    CLAIMNESS_SCHEMA,
                    max_output_tokens=256,
                )

        return ClaimnessClassifier(GeminiClaimnessGenerator())
    if not args.model:
        raise ValueError("--model is required for --backend local")
    generator = LocalClaimJudge(
        args.model,
        device=args.device,
        dtype=args.dtype,
        max_input_tokens=args.max_input_tokens,
        max_new_tokens=args.max_new_tokens,
        allow_download=args.allow_download,
    )
    return ClaimnessClassifier(generator)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Locate exact factual claim spans before source verification.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--backend", choices=("gemini", "local"), required=True)
    parser.add_argument("--input", required=True, help="Summary JSONL")
    parser.add_argument("--output", required=True, help="Recovered factual-claim JSONL")
    parser.add_argument("--non_claim_output", help="Optional heading/fragment JSONL")
    parser.add_argument("--error_output", help="Optional candidate-error JSONL")
    parser.add_argument("--source_col")
    parser.add_argument("--summary_col")
    parser.add_argument("--model", help="Gemini model ID or local model folder")
    parser.add_argument("--api_key_env", default="GEMINI_API_KEY")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto")
    parser.add_argument("--max_input_tokens", type=int, default=8192)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--allow_download", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--no_split_clauses", action="store_true")
    parser.add_argument("--sleep_seconds", type=float, default=0.0)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.sleep_seconds < 0:
        parser.error("--sleep_seconds must be non-negative")
    if args.backend == "gemini" and args.allow_download:
        parser.error("--allow_download applies only to --backend local")
    rows = read_jsonl(Path(args.input))
    if not rows:
        raise ValueError(f"No rows found in {args.input}")
    classifier = _make_classifier(args)
    claims, non_claims, errors = extract_rows(
        rows,
        classifier,
        source_col=args.source_col,
        summary_col=args.summary_col,
        limit=args.limit,
        split_clauses=not args.no_split_clauses,
        sleep_seconds=args.sleep_seconds,
    )
    _write_jsonl(claims, Path(args.output))
    if args.non_claim_output:
        _write_jsonl(non_claims, Path(args.non_claim_output))
    if args.error_output:
        _write_jsonl(errors, Path(args.error_output))
    summary_path = Path(args.output).with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(
            {
                "documents_input": len(rows) if args.limit is None else min(args.limit, len(rows)),
                "claims": len(claims),
                "non_claims": len(non_claims),
                "errors": len(errors),
                "backend": classifier.backend_name,
                "model": classifier.model_name,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Claims written to: {args.output}")
    print(f"Claims: {len(claims)}; non-claims: {len(non_claims)}; errors: {len(errors)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
