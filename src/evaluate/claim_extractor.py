"""Locate factual claim spans before running source-support verification.

The splitter creates exact candidate spans in the original summary. A Gemini
or local model then answers only whether each candidate is a factual claim.
The accepted rows retain verbatim text and character offsets, so the next
verification stage never has to trust a model-generated paraphrase.
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Protocol

from .fact_audit import retrieve_evidence, split_claims
from .llm_claim_judge import (
    DEFAULT_GEMINI_MODEL,
    GeminiClaimJudge,
    JudgeError,
    LocalClaimJudge,
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
whether it is a claim. Never rewrite the candidate.
A prediction or generalization is still a claim. For example, a sentence
asserting that people born in the Year of the Monkey will face many problems
at work is_claim=true; do not mark it false merely because it is horoscope-like
or describes a broad group.

Your entire response must be exactly one JSON object on one line. Do not emit
chain-of-thought, <think>, <analysis>, or other reasoning blocks. Do not use
Markdown fences, prose, or a Python dictionary. Use the JSON boolean true or
false for is_claim, not a quoted word."""

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


def candidate_spans(summary: str, *, split_clauses: bool = False) -> list[dict[str, Any]]:
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


def _claimness_repair_prompt(
    summary: str, candidate: str, start: int, end: int, previous_response: str
) -> str:
    """Ask a local instruct model once to repair an invalid structured answer."""
    return f"""The previous response did not follow the required output format.
Reclassify the same candidate and return exactly one JSON object with these keys:
is_claim (JSON boolean), unit_type, and reason. Do not copy the previous format.
Do not return Markdown, prose, a Python dictionary, or any <think>/<analysis>
reasoning block. Any text outside the JSON object is invalid.

CANDIDATE (copied from the summary, characters {start}:{end}):
<candidate>
{candidate}
</candidate>

SUMMARY CONTEXT:
<summary>
{summary}
</summary>

PREVIOUS INVALID RESPONSE:
<previous_response>
{previous_response}
</previous_response>"""


def parse_claimness(text: str) -> dict[str, Any]:
    """Parse claimness without accepting model reasoning as structured output.

    A local instruct model may use a Markdown fence, Python booleans, or a
    short ``is_claim: false`` answer. Those narrow forms are tolerated, but
    the response must otherwise contain only the structured answer. A
    thinking/reasoning block or prose before/after the answer is invalid; the
    caller may make its bounded repair attempt and otherwise records an error.
    """
    raw_text = str(text).strip()
    if not raw_text:
        raise JudgeError("model returned an empty claimness response")
    if re.search(
        r"<\s*/?\s*(?:think|analysis|reasoning|thought)\s*>"
        r"|<\|(?:begin|end)_(?:of_)?(?:think|thinking|analysis|reasoning|thought)\|>",
        raw_text,
        flags=re.IGNORECASE,
    ):
        raise JudgeError("model emitted a thinking/reasoning block; response is invalid")

    cleaned = raw_text
    if cleaned.startswith("```"):
        fenced = re.fullmatch(
            r"```(?:json)?\s*(.*?)\s*```", cleaned, flags=re.IGNORECASE | re.DOTALL
        )
        if fenced is None:
            raise JudgeError("model returned an invalid Markdown wrapper around claimness JSON")
        cleaned = fenced.group(1).strip()

    value: Any = None
    short_answer = False
    json_error: Exception | None = None
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        json_error = exc
        try:
            value = ast.literal_eval(cleaned)
        except (SyntaxError, ValueError, TypeError):
            short_match = re.fullmatch(
                r"(?:is[_ -]?claim|is claim)\s*(?:is|=|:)\s*[\"']?"
                r"(true|false|yes|no|1|0)[\"']?",
                cleaned,
                flags=re.IGNORECASE,
            )
            if short_match:
                value = {"is_claim": short_match.group(1)}
                short_answer = True
            else:
                raise JudgeError(
                    "model did not return exactly one JSON object without reasoning or prose"
                ) from json_error
    if not isinstance(value, dict):
        raise JudgeError("model JSON response was not an object")

    if not short_answer:
        has_unit_type = any(key in value for key in ("unit_type", "unitType"))
        has_reason = any(key in value for key in ("reason", "explanation"))
        missing = []
        if not has_unit_type:
            missing.append("unit_type")
        if not has_reason:
            missing.append("reason")
        if missing:
            raise JudgeError(
                "model JSON response is missing required claimness keys: "
                + ", ".join(missing)
            )

    raw_is_claim = None
    for key in ("is_claim", "isClaim", "is claim"):
        if key in value:
            raw_is_claim = value[key]
            break
    if isinstance(raw_is_claim, bool):
        is_claim = raw_is_claim
    elif isinstance(raw_is_claim, int) and raw_is_claim in {0, 1}:
        is_claim = bool(raw_is_claim)
    elif isinstance(raw_is_claim, str):
        normalized = raw_is_claim.strip().casefold().strip(" .,!\"'")
        if normalized in {"yes", "true", "1"}:
            is_claim = True
        elif normalized in {"no", "false", "0"}:
            is_claim = False
        else:
            raise JudgeError(f"invalid is_claim value: {raw_is_claim!r}")
    else:
        raise JudgeError("model did not return boolean is_claim")
    unit_type = str(value.get("unit_type", value.get("unitType", "other"))).strip().casefold().replace("-", "_")
    if unit_type not in UNIT_TYPES:
        unit_type = "other"
    if is_claim:
        unit_type = "claim"
    reason = str(value.get("reason", value.get("explanation", ""))).strip()
    return {"is_claim": is_claim, "unit_type": unit_type, "reason": reason}


class ClaimnessClassifier:
    def __init__(self, generator: ClaimnessGenerator) -> None:
        self.generator = generator
        self.backend_name = generator.backend_name
        self.model_name = generator.model_name
        self.last_raw_response = ""
        self.last_raw_attempts: list[str] = []

    def classify(self, summary: str, candidate: str, start: int, end: int) -> dict[str, Any]:
        self.last_raw_response = ""
        self.last_raw_attempts = []
        raw = self.generator.generate_text(
            CLAIMNESS_SYSTEM_PROMPT,
            _claimness_prompt(summary, candidate, start, end),
        )
        self.last_raw_response = raw
        raw_attempts = [raw]
        self.last_raw_attempts = raw_attempts.copy()
        try:
            result = parse_claimness(raw)
        except JudgeError:
            if self.backend_name != "local":
                raise
            repaired = self.generator.generate_text(
                CLAIMNESS_SYSTEM_PROMPT,
                _claimness_repair_prompt(summary, candidate, start, end, raw),
            )
            raw_attempts.append(repaired)
            self.last_raw_attempts = raw_attempts.copy()
            self.last_raw_response = "\n--- repair attempt ---\n".join(raw_attempts)
            result = parse_claimness(repaired)
        result["raw_response"] = raw_attempts[-1]
        result["raw_attempts"] = raw_attempts
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
        "claimness_raw_attempts": result.get("raw_attempts", []),
        "claimness_repair_attempted": len(result.get("raw_attempts", [])) > 1,
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
                        "claimness_raw": classifier.last_raw_response,
                        "claimness_raw_attempts": classifier.last_raw_attempts.copy(),
                        "claimness_repair_attempted": len(classifier.last_raw_attempts) > 1,
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
    parser.add_argument(
        "--split_clauses",
        action="store_true",
        help="Opt in to colon/semicolon splitting; may detach a heading subject from its predicate.",
    )
    parser.add_argument(
        "--no_split_clauses",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--sleep_seconds", type=float, default=0.0)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.sleep_seconds < 0:
        parser.error("--sleep_seconds must be non-negative")
    if args.backend == "gemini" and args.allow_download:
        parser.error("--allow_download applies only to --backend local")
    if args.split_clauses and args.no_split_clauses:
        parser.error("--split_clauses and --no_split_clauses cannot be used together")
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
        split_clauses=args.split_clauses,
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
