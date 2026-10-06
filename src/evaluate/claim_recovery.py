"""Recover factual claim rows after human or LLM claimness labeling.

A claim is a minimal text unit that asserts an event, state, relation,
quantity, or other proposition with a truth value that can be checked against
the source. A heading or topic label is retained for audit, but is not a
factual claim unless it itself makes such an assertion.

This module does not invent or rewrite claims. It filters already labeled
claim-like rows and preserves the original claim text and all judge fields.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

SUPPORT_LABELS = frozenset(("supported", "contradicted", "not_supported", "unclear"))
NON_CLAIM_LABEL = "not_a_claim"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(value)
    return rows


def recover_claims(
    rows: list[dict[str, Any]],
    *,
    label_col: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Return ``claims``, ``non_claims``, and ``unresolved`` rows.

    Support labels remain claims even when the support judgment is negative:
    ``not_supported`` means an actual proposition lacks source support, not
    that the text was a heading.
    """
    claims: list[dict[str, Any]] = []
    non_claims: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    for row in rows:
        label = str(row.get(label_col, "")).strip()
        annotated = dict(row)
        annotated["claim_recovery_label"] = label
        if label in SUPPORT_LABELS:
            annotated["claim_type"] = "claim"
            annotated["is_claim"] = True
            claims.append(annotated)
        elif label == NON_CLAIM_LABEL:
            annotated["claim_type"] = "non_claim"
            annotated["is_claim"] = False
            non_claims.append(annotated)
        else:
            annotated["claim_type"] = "unresolved"
            annotated["is_claim"] = None
            unresolved.append(annotated)
    return claims, non_claims, unresolved


def _write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def _write_summary(
    input_rows: list[dict[str, Any]],
    claims: list[dict[str, Any]],
    non_claims: list[dict[str, Any]],
    unresolved: list[dict[str, Any]],
    label_col: str,
    output_path: Path,
) -> None:
    summary = {
        "input_rows": len(input_rows),
        "recovered_claims": len(claims),
        "non_claims": len(non_claims),
        "unresolved": len(unresolved),
        "label_col": label_col,
        "labels": dict(Counter(str(row.get(label_col, "")) for row in input_rows)),
    }
    output_path.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Recover factual claims from labeled claim-like rows.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", required=True, help="Human- or LLM-labeled JSONL")
    parser.add_argument("--label_col", default="llm_label", help="Column containing the five-way label")
    parser.add_argument("--output", required=True, help="Recovered claim JSONL")
    parser.add_argument("--non_claim_output", help="Optional output for headings/fragments")
    parser.add_argument("--unresolved_output", help="Optional output for blank/unknown labels")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    rows = read_jsonl(input_path)
    if not rows:
        raise ValueError(f"No rows found in {input_path}")
    claims, non_claims, unresolved = recover_claims(rows, label_col=args.label_col)
    _write_jsonl(claims, output_path)
    if args.non_claim_output:
        _write_jsonl(non_claims, Path(args.non_claim_output))
    if args.unresolved_output:
        _write_jsonl(unresolved, Path(args.unresolved_output))
    _write_summary(rows, claims, non_claims, unresolved, args.label_col, output_path)
    print(f"Recovered claims: {len(claims)}")
    print(f"Non-claims: {len(non_claims)}")
    print(f"Unresolved: {len(unresolved)}")
    print(f"Claims written to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
