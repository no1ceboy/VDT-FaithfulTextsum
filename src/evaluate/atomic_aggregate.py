"""Aggregate per-atomic-claim judgments into one parent-claim result.

Atomic decomposition intentionally produces one row per independently
checkable proposition.  This module combines those rows again for reporting
without hiding the individual judgments or their evidence.  Aggregation is a
conservative rule, not a second language-model judgment:

* one contradicted atom makes the parent contradicted;
* otherwise one not_supported atom makes it not_supported;
* otherwise one unclear atom makes it unclear;
* only an all-supported group is supported;
* missing/error/unknown atoms remain unresolved.

The original atomic rows and every selected evidence quote remain nested in
the output, so a reviewer can inspect exactly which proposition caused the
parent result.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .claim_recovery import read_jsonl

KNOWN_LABELS = frozenset(
    ("supported", "contradicted", "not_supported", "unclear", "not_a_claim")
)


def _group_key(row: dict[str, Any]) -> tuple[str, ...]:
    """Build a stable parent key even when input rows have no document ID."""
    return (
        str(row.get("id", "")),
        str(row.get("summary_col", "")),
        str(row.get("parent_claim_index", row.get("claim_index", ""))),
        str(row.get("parent_start_char", row.get("start_char", ""))),
        str(row.get("parent_end_char", row.get("end_char", ""))),
        str(row.get("parent_claim", row.get("claim", row.get("claim_text", "")))),
        str(row.get("source", row.get("input", row.get("text", "")))),
    )


def _quote_items(row: dict[str, Any]) -> list[dict[str, Any]]:
    raw = row.get("llm_evidence_quotes")
    if raw is None:
        legacy = str(row.get("llm_evidence_quote", "")).strip()
        raw = [] if not legacy else [{"sentence_index": None, "quote": legacy}]
    if not isinstance(raw, list):
        return []
    quotes: list[dict[str, Any]] = []
    for item in raw:
        if isinstance(item, str):
            quote = item.strip()
            sentence_index = None
        elif isinstance(item, dict):
            quote = str(item.get("quote", item.get("evidence_quote", ""))).strip()
            sentence_index = item.get("sentence_index", item.get("source_sentence_index"))
        else:
            continue
        if quote:
            quotes.append({"sentence_index": sentence_index, "quote": quote})
    return quotes


def _atomic_item(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "atomic_claim_index": row.get("atomic_claim_index"),
        "atomic_claim_count": row.get("atomic_claim_count"),
        "atomic_surface_text": row.get("atomic_surface_text", row.get("claim", "")),
        "verification_claim": row.get("verification_claim", row.get("claim", "")),
        "start_char": row.get("start_char"),
        "end_char": row.get("end_char"),
        "judge_status": row.get("judge_status", "unjudged"),
        "judge_error": row.get("judge_error", ""),
        "llm_label": row.get("llm_label", ""),
        "llm_reason": row.get("llm_reason", ""),
        "llm_confidence": row.get("llm_confidence"),
        "llm_evidence_quotes": _quote_items(row),
        "evidence": row.get("evidence", []),
    }


def _aggregate_label(items: list[dict[str, Any]]) -> tuple[str, str]:
    labels = [str(item.get("llm_label", "")).strip() for item in items]
    if any(item.get("judge_status") != "ok" for item in items):
        return "unresolved", "At least one atomic claim has no successful judgment."
    if any(label not in KNOWN_LABELS for label in labels):
        return "unresolved", "At least one atomic claim has an unknown or missing label."
    if "contradicted" in labels:
        return "contradicted", "At least one atomic proposition is contradicted by the source."
    if "not_supported" in labels:
        return "not_supported", "At least one atomic proposition lacks sufficient source support."
    if "unclear" in labels:
        return "unclear", "At least one atomic proposition remains ambiguous."
    if labels and all(label == "supported" for label in labels):
        return "supported", "Every atomic proposition was judged supported by the source."
    if labels and all(label == "not_a_claim" for label in labels):
        return "not_a_claim", "Every decomposed unit was judged not to assert a factual proposition."
    return "unresolved", "Atomic labels could not be combined into a definitive parent result."


def aggregate_atomic_judgments(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return one conservative parent-level row for each atomic row group."""
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    order: list[tuple[str, ...]] = []
    for row in rows:
        key = _group_key(row)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(row)

    aggregated: list[dict[str, Any]] = []
    for key in order:
        group = sorted(
            groups[key],
            key=lambda row: (
                int(row.get("atomic_claim_index", 0) or 0),
                int(row.get("start_char", 0) or 0),
            ),
        )
        items = [_atomic_item(row) for row in group]
        expected_counts = [
            int(row["atomic_claim_count"])
            for row in group
            if str(row.get("atomic_claim_count", "")).strip().isdigit()
            and int(row["atomic_claim_count"]) > 0
        ]
        expected_count = max(expected_counts, default=len(items))
        indices = [item.get("atomic_claim_index") for item in items]
        duplicate_indices = len([index for index in indices if index is not None]) != len(
            {index for index in indices if index is not None}
        )
        if len(items) != expected_count or duplicate_indices:
            label = "unresolved"
            reason = "The atomic-claim input is incomplete or contains duplicate units."
        else:
            label, reason = _aggregate_label(items)
        confidences = [
            float(item["llm_confidence"])
            for item in items
            if isinstance(item.get("llm_confidence"), (int, float))
            and 0.0 <= float(item["llm_confidence"]) <= 1.0
        ]
        combined_quotes: list[dict[str, Any]] = []
        seen_quotes: set[tuple[Any, str]] = set()
        for item in items:
            atomic_index = item.get("atomic_claim_index")
            for quote in item["llm_evidence_quotes"]:
                quote_key = (quote.get("sentence_index"), quote.get("quote", ""))
                if quote_key in seen_quotes:
                    continue
                seen_quotes.add(quote_key)
                combined_quotes.append(
                    {
                        "atomic_claim_index": atomic_index,
                        "sentence_index": quote.get("sentence_index"),
                        "quote": quote.get("quote", ""),
                    }
                )

        first = dict(group[0])
        parent_claim = str(
            first.get("parent_claim", first.get("claim", first.get("claim_text", "")))
        )
        aggregate_confidence = min(confidences) if confidences else None
        first["claim"] = parent_claim
        first["claim_text"] = parent_claim
        if first.get("parent_start_char") is not None:
            first["start_char"] = first.get("parent_start_char")
        if first.get("parent_end_char") is not None:
            first["end_char"] = first.get("parent_end_char")
        first.update(
            {
                "atomic_claim_count": expected_count,
                "atomic_rows_present": len(items),
                "atomic_missing_count": max(0, expected_count - len(items)),
                "atomic_judgments": items,
                "llm_label": label,
                "llm_reason": reason,
                "llm_confidence": aggregate_confidence,
                "judge_status": "ok" if label != "unresolved" else "unresolved",
                "aggregate_label": label,
                "aggregate_reason": reason,
                "aggregate_confidence": aggregate_confidence,
                "aggregate_status": "ok" if label != "unresolved" else "unresolved",
                "atomic_label_counts": dict(Counter(
                    str(item.get("llm_label", "")) for item in items
                )),
                "atomic_error_count": sum(
                    item.get("judge_status") != "ok" for item in items
                ),
                "llm_evidence_quotes": combined_quotes,
                "llm_evidence_quote": combined_quotes[0]["quote"] if combined_quotes else "",
                "evidence_by_atomic_claim": [
                    {
                        "atomic_claim_index": item.get("atomic_claim_index"),
                        "evidence": item.get("evidence", []),
                    }
                    for item in items
                ],
            }
        )
        aggregated.append(first)
    return aggregated


def _write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate judged atomic claims into parent-level results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", required=True, help="LLM-judged atomic-claim JSONL")
    parser.add_argument("--output", required=True, help="Parent-level aggregate JSONL")
    args = parser.parse_args()

    rows = read_jsonl(Path(args.input))
    if not rows:
        raise ValueError(f"No rows found in {args.input}")
    output_rows = aggregate_atomic_judgments(rows)
    output_path = Path(args.output)
    _write_jsonl(output_rows, output_path)
    summary = {
        "atomic_rows": len(rows),
        "parent_rows": len(output_rows),
        "aggregate_label_counts": dict(Counter(row["aggregate_label"] for row in output_rows)),
        "unresolved_parent_rows": sum(row["aggregate_label"] == "unresolved" for row in output_rows),
    }
    output_path.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"Parent aggregates written to: {output_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
