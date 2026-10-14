"""Report atomic faithfulness judgments and compare summary systems.

The atomic audit produces categorical judgments rather than one calibrated
hallucination score.  This module turns those judgments into explicit rates:

* ``contradiction_rate`` counts claims that conflict with the source;
* ``unsupported_rate`` counts claims for which the source is insufficient;
* ``potential_hallucination_rate`` combines those two failure categories;
* parent-summary and document-level rates show how many summaries are affected.

Human and LLM summaries should be audited against the same source documents,
then supplied as separate named inputs, for example ``human=...`` and
``llm=...``.  Atomic rows are compared by aggregate rates.  Parent rows are
also compared by document ID; atomic propositions are not force-matched
because human and LLM summaries can decompose into different claims.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from collections import Counter, defaultdict
from html import escape
from pathlib import Path
from typing import Any, Mapping, Sequence

from .report import load_jsonl

KNOWN_LABELS = (
    "supported",
    "contradicted",
    "not_supported",
    "unclear",
    "not_a_claim",
)
FACTUAL_LABELS = frozenset(("supported", "contradicted", "not_supported", "unclear"))
DECIDED_LABELS = frozenset(("supported", "contradicted", "not_supported"))
PROBLEM_LABELS = frozenset(("contradicted", "not_supported"))
PARENT_RESOLVED_LABELS = frozenset((*FACTUAL_LABELS, "not_a_claim"))


def _rate(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return round(numerator / denominator, 6)


def _ordered_counts(counter: Counter[str]) -> dict[str, int]:
    return {label: int(counter.get(label, 0)) for label in KNOWN_LABELS}


def _atomic_label(row: Mapping[str, Any]) -> str:
    return str(row.get("llm_label", "") or "").strip()


def _parent_label(row: Mapping[str, Any]) -> str:
    aggregate = str(row.get("aggregate_label", "") or "").strip()
    return aggregate or str(row.get("llm_label", "") or "").strip()


def summarize_atomic_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize one system's per-atomic-claim judgment JSONL."""
    labels = Counter(
        _atomic_label(row)
        for row in rows
        if str(row.get("judge_status", "ok") or "ok").strip() == "ok"
        and _atomic_label(row) in KNOWN_LABELS
    )
    rows_judged = sum(
        str(row.get("judge_status", "ok") or "ok").strip() == "ok"
        for row in rows
    )
    rows_error = sum(
        str(row.get("judge_status", "") or "").strip() == "error"
        for row in rows
    )
    valid_label_rows = sum(labels.values())
    factual_rows = sum(labels.get(label, 0) for label in FACTUAL_LABELS)
    decided_rows = sum(labels.get(label, 0) for label in DECIDED_LABELS)
    problem_rows = sum(labels.get(label, 0) for label in PROBLEM_LABELS)
    contradiction_rows = labels.get("contradicted", 0)
    unsupported_rows = labels.get("not_supported", 0)

    return {
        "rows_total": len(rows),
        "rows_judged": rows_judged,
        "rows_error": rows_error,
        "unknown_or_missing_label_rows": max(0, rows_judged - valid_label_rows),
        "label_counts": _ordered_counts(labels),
        "factual_claim_rows": factual_rows,
        "decided_claim_rows": decided_rows,
        "rates": {
            # The factual denominator excludes not_a_claim and failed rows.
            "supported_rate": _rate(labels.get("supported", 0), factual_rows),
            "contradiction_rate": _rate(contradiction_rows, factual_rows),
            "unsupported_rate": _rate(unsupported_rows, factual_rows),
            "potential_hallucination_rate": _rate(problem_rows, factual_rows),
            # This second denominator excludes unclear judgments as well.
            "decisive_potential_hallucination_rate": _rate(problem_rows, decided_rows),
            "unclear_rate": _rate(labels.get("unclear", 0), factual_rows),
            "not_a_claim_rate": _rate(labels.get("not_a_claim", 0), valid_label_rows),
            "judgment_coverage_rate": _rate(rows_judged, len(rows)),
        },
    }


def _document_key(row: Mapping[str, Any], row_index: int) -> str:
    identifier = str(row.get("id", "") or "").strip()
    if identifier:
        return identifier
    source = str(row.get("source", row.get("text", row.get("input", ""))) or "")
    if source:
        digest = hashlib.sha1(source.encode("utf-8")).hexdigest()[:16]
        return f"source:{digest}"
    return f"row:{row_index}"


def _document_status_map(rows: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for row_index, row in enumerate(rows):
        grouped[_document_key(row, row_index)].append(_parent_label(row))

    statuses: dict[str, str] = {}
    for key, labels in grouped.items():
        if any(label not in PARENT_RESOLVED_LABELS for label in labels):
            statuses[key] = "unresolved"
        elif "contradicted" in labels:
            statuses[key] = "contradicted"
        elif "not_supported" in labels:
            statuses[key] = "not_supported"
        elif "unclear" in labels:
            statuses[key] = "unclear"
        elif labels and all(label == "supported" for label in labels):
            statuses[key] = "supported"
        elif labels and all(label == "not_a_claim" for label in labels):
            statuses[key] = "not_a_claim"
        else:
            statuses[key] = "unresolved"
    return statuses


def summarize_parent_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize parent-summary rows produced by ``atomic_aggregate``."""
    labels = Counter(
        _parent_label(row)
        for row in rows
        if _parent_label(row)
    )
    evaluable_rows = sum(labels.get(label, 0) for label in FACTUAL_LABELS)
    problem_rows = sum(labels.get(label, 0) for label in PROBLEM_LABELS)
    statuses = _document_status_map(rows)
    document_counts = Counter(statuses.values())
    evaluable_documents = sum(document_counts.get(label, 0) for label in FACTUAL_LABELS)
    problem_documents = sum(document_counts.get(label, 0) for label in PROBLEM_LABELS)
    return {
        "parent_rows_total": len(rows),
        "parent_label_counts": {
            label: int(labels.get(label, 0))
            for label in (*KNOWN_LABELS, "unresolved")
        },
        "parent_evaluable_rows": evaluable_rows,
        "parent_unresolved_rows": labels.get("unresolved", 0),
        "rates": {
            "summary_supported_rate": _rate(labels.get("supported", 0), evaluable_rows),
            "summary_contradiction_rate": _rate(labels.get("contradicted", 0), evaluable_rows),
            "summary_unsupported_rate": _rate(labels.get("not_supported", 0), evaluable_rows),
            "summary_flagged_rate": _rate(problem_rows, evaluable_rows),
            "summary_unclear_rate": _rate(labels.get("unclear", 0), evaluable_rows),
            "summary_resolution_rate": _rate(
                sum(labels.get(label, 0) for label in PARENT_RESOLVED_LABELS),
                len(rows),
            ),
            "document_flagged_rate": _rate(problem_documents, evaluable_documents),
        },
        "documents": {
            "total": len(statuses),
            "evaluable": evaluable_documents,
            "problem": problem_documents,
            "status_counts": {
                label: int(document_counts.get(label, 0))
                for label in (*KNOWN_LABELS, "unresolved")
            },
        },
        "_document_statuses": statuses,
    }


def _comparison(
    left_name: str,
    right_name: str,
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> dict[str, Any]:
    left_rates = left.get("atomic", {}).get("rates", {})
    right_rates = right.get("atomic", {}).get("rates", {})
    atomic_rate_delta: dict[str, float | None] = {}
    for name in (
        "potential_hallucination_rate",
        "decisive_potential_hallucination_rate",
        "contradiction_rate",
        "unsupported_rate",
        "supported_rate",
    ):
        left_value = left_rates.get(name)
        right_value = right_rates.get(name)
        atomic_rate_delta[name] = (
            round(right_value - left_value, 6)
            if left_value is not None and right_value is not None
            else None
        )

    result: dict[str, Any] = {
        "left_system": left_name,
        "right_system": right_name,
        "rate_delta_right_minus_left": atomic_rate_delta,
    }
    left_statuses = left.get("parent", {}).get("_document_statuses", {})
    right_statuses = right.get("parent", {}).get("_document_statuses", {})
    common_ids = sorted(set(left_statuses) & set(right_statuses))
    matrix = Counter()
    for identifier in common_ids:
        left_status = left_statuses[identifier]
        right_status = right_statuses[identifier]
        if left_status not in PARENT_RESOLVED_LABELS or right_status not in PARENT_RESOLVED_LABELS:
            matrix["unresolved_on_either"] += 1
            continue
        left_problem = left_status in PROBLEM_LABELS
        right_problem = right_status in PROBLEM_LABELS
        matrix[(left_problem, right_problem)] += 1
    resolved_common = sum(
        matrix[key]
        for key in ((False, False), (False, True), (True, False), (True, True))
    )
    result["paired_documents"] = {
        "common_document_ids": len(common_ids),
        "resolved_common_documents": resolved_common,
        "unresolved_on_either": matrix["unresolved_on_either"],
        "both_clean": matrix[(False, False)],
        "right_only_problem": matrix[(False, True)],
        "left_only_problem": matrix[(True, False)],
        "both_problem": matrix[(True, True)],
        "left_problem_rate": _rate(
            matrix[(True, False)] + matrix[(True, True)], resolved_common
        ),
        "right_problem_rate": _rate(
            matrix[(False, True)] + matrix[(True, True)], resolved_common
        ),
    }
    return result


def build_report_data(
    atomic_inputs: Mapping[str, Sequence[Mapping[str, Any]]],
    parent_inputs: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Build the machine-readable comparison report."""
    parent_inputs = parent_inputs or {}
    names = sorted(set(atomic_inputs) | set(parent_inputs))
    systems: dict[str, dict[str, Any]] = {}
    for name in names:
        atomic_rows = atomic_inputs.get(name)
        parent_rows = parent_inputs.get(name)
        system: dict[str, Any] = {}
        if atomic_rows is not None:
            system["atomic"] = summarize_atomic_rows(atomic_rows)
        if parent_rows is not None:
            parent_summary = summarize_parent_rows(parent_rows)
            parent_summary.pop("_document_statuses", None)
            system["parent"] = parent_summary
        systems[name] = system

    comparisons: dict[str, Any] = {}
    for left_name, right_name in itertools.combinations(names, 2):
        if left_name in systems and right_name in systems:
            comparisons[f"{left_name}_vs_{right_name}"] = _comparison(
                left_name,
                right_name,
                {
                    "atomic": summarize_atomic_rows(atomic_inputs[left_name])
                    if left_name in atomic_inputs
                    else {},
                    "parent": summarize_parent_rows(parent_inputs[left_name])
                    if left_name in parent_inputs
                    else {},
                },
                {
                    "atomic": summarize_atomic_rows(atomic_inputs[right_name])
                    if right_name in atomic_inputs
                    else {},
                    "parent": summarize_parent_rows(parent_inputs[right_name])
                    if right_name in parent_inputs
                    else {},
                },
            )
    return {
        "report_type": "atomic_faithfulness",
        "systems": systems,
        "comparisons": comparisons,
        "definitions": {
            "factual_denominator": "supported + contradicted + not_supported + unclear; excludes not_a_claim and failed rows",
            "potential_hallucination": "contradicted + not_supported",
            "contradicted": "the source explicitly conflicts with the atomic claim",
            "not_supported": "the source does not provide enough evidence; this is not proof that the claim is false",
            "summary_flagged": "a parent summary unit contains at least one contradicted or not_supported atomic proposition",
            "document_flagged": "a document contains at least one flagged parent summary unit",
            "comparison_direction": "rate_delta_right_minus_left; negative is lower for the right system",
        },
    }


def _percent(value: Any) -> str:
    if value is None:
        return "N/A"
    return f"{float(value) * 100:.2f}%"


def _cell(value: Any) -> str:
    return escape("N/A" if value is None else str(value))


def build_atomic_report(
    atomic_inputs: Mapping[str, Sequence[Mapping[str, Any]]],
    output_path: str | Path,
    *,
    parent_inputs: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    title: str = "Atomic faithfulness audit",
) -> tuple[Path, Path]:
    """Write an HTML report and adjacent JSON report."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    data = build_report_data(atomic_inputs, parent_inputs)
    systems = data["systems"]

    system_rows: list[str] = []
    for name, system in systems.items():
        atomic = system.get("atomic", {})
        atomic_counts = atomic.get("label_counts", {})
        atomic_rates = atomic.get("rates", {})
        parent = system.get("parent", {})
        parent_rates = parent.get("rates", {})
        document_rates = parent_rates
        system_rows.append(
            "<tr>"
            f"<td>{escape(name)}</td>"
            f"<td>{_cell(atomic.get('rows_total'))}</td>"
            f"<td>{_cell(atomic_counts.get('supported'))}</td>"
            f"<td>{_cell(atomic_counts.get('contradicted'))}</td>"
            f"<td>{_cell(atomic_counts.get('not_supported'))}</td>"
            f"<td>{_cell(atomic_counts.get('unclear'))}</td>"
            f"<td>{_percent(atomic_rates.get('potential_hallucination_rate'))}</td>"
            f"<td>{_percent(atomic_rates.get('contradiction_rate'))}</td>"
            f"<td>{_percent(parent_rates.get('summary_flagged_rate'))}</td>"
            f"<td>{_percent(document_rates.get('document_flagged_rate'))}</td>"
            "</tr>"
        )

    comparison_sections: list[str] = []
    for name, comparison in data["comparisons"].items():
        delta = comparison["rate_delta_right_minus_left"]
        paired = comparison["paired_documents"]
        comparison_sections.append(
            f"<h2>{escape(name)}</h2>"
            "<table><tr><th>Metric</th><th>Right minus left</th></tr>"
            + "".join(
                f"<tr><td>{escape(metric)}</td><td>{_percent(value)}</td></tr>"
                for metric, value in delta.items()
            )
            + "</table>"
            "<table><tr><th>Paired document outcome</th><th>Count</th></tr>"
            + "".join(
                f"<tr><td>{escape(metric)}</td><td>{_cell(paired[metric])}</td></tr>"
                for metric in (
                    "common_document_ids",
                    "resolved_common_documents",
                    "unresolved_on_either",
                    "both_clean",
                    "left_only_problem",
                    "right_only_problem",
                    "both_problem",
                    "left_problem_rate",
                    "right_problem_rate",
                )
            )
            + "</table>"
        )

    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>{escape(title)}</title>
<style>
body {{ font: 15px/1.45 system-ui, sans-serif; color:#172033; margin:2rem auto; max-width:1250px; padding:0 1rem; }}
h1 {{ margin-bottom:.25rem; }} h2 {{ margin-top:2rem; }} .muted {{ color:#64748b; }}
table {{ border-collapse:collapse; width:100%; margin:.75rem 0 1.25rem; }}
th,td {{ border:1px solid #dbe3ef; padding:.45rem .6rem; text-align:left; }}
th {{ background:#f1f5f9; }} tr:nth-child(even) {{ background:#f8fafc; }}
code {{ background:#f1f5f9; padding:.1rem .25rem; }}
</style></head><body>
<h1>{escape(title)}</h1>
<p class="muted">Descriptive audit statistics, not calibrated probabilities or a significance test.</p>
<h2>System comparison</h2>
<table><tr><th>System</th><th>Atomic rows</th><th>Supported</th><th>Contradicted</th><th>Not supported</th><th>Unclear</th><th>Potential hallucination</th><th>Direct contradiction</th><th>Summary flagged</th><th>Document flagged</th></tr>
{''.join(system_rows) or '<tr><td colspan="10">No systems supplied.</td></tr>'}</table>
{''.join(comparison_sections)}
<h2>Definitions</h2>
<ul>
<li><b>Potential hallucination</b> = contradicted + not supported, divided by factual atomic claims.</li>
<li><b>Direct contradiction</b> is stronger evidence than not supported.</li>
<li><b>Summary flagged</b> means at least one atomic proposition in that summary unit is contradicted or not supported.</li>
<li><b>Document flagged</b> means at least one summary unit for that document is flagged.</li>
</ul>
<p class="muted">Inspect the source and quoted evidence for flagged rows before treating an automatic judgment as a confirmed human error.</p>
</body></html>
"""
    output.write_text(html, encoding="utf-8")
    json_output = output.with_suffix(".json")
    json_output.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output, json_output


def _named_inputs(specs: Sequence[str], option: str) -> dict[str, list[dict[str, Any]]]:
    if not specs:
        return {}
    result: dict[str, list[dict[str, Any]]] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"{option} must use LABEL=PATH, got {spec!r}")
        label, path_text = spec.split("=", 1)
        label = label.strip()
        path_text = path_text.strip()
        if not label or not path_text:
            raise ValueError(f"{option} must use a non-empty LABEL=PATH, got {spec!r}")
        if label in result:
            raise ValueError(f"Duplicate {option} label: {label}")
        result[label] = load_jsonl(Path(path_text))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Report atomic faithfulness rates and compare human/LLM summary audits.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--atomic_input",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="Judged atomic JSONL; repeat for human=... and llm=...",
    )
    parser.add_argument(
        "--parent_input",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="Optional aggregated parent JSONL with matching labels",
    )
    parser.add_argument("--output", default="results/atomic_faithfulness_report.html")
    parser.add_argument("--title", default="Atomic faithfulness audit")
    args = parser.parse_args()
    atomic_inputs = _named_inputs(args.atomic_input, "--atomic_input")
    parent_inputs = _named_inputs(args.parent_input, "--parent_input")
    html_path, json_path = build_atomic_report(
        atomic_inputs,
        args.output,
        parent_inputs=parent_inputs,
        title=args.title,
    )
    print(f"HTML report: {html_path}")
    print(f"JSON report: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
