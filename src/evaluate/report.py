"""Create dependency-free HTML and JSON reports from evaluation/run artifacts."""

from __future__ import annotations

import json
import math
import statistics
from html import escape
from pathlib import Path
from typing import Any


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load JSONL objects and fail early on malformed report input."""
    input_path = Path(path)
    rows: list[dict[str, Any]] = []
    with input_path.open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{input_path}:{line_number}: invalid JSON: {exc.msg}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{input_path}:{line_number}: every row must be a JSON object")
            rows.append(row)
    if not rows:
        raise ValueError(f"{input_path}: no non-empty JSONL records found")
    return rows


def _numeric_values(records: list[dict[str, Any]], field: str) -> list[float]:
    values: list[float] = []
    for record in records:
        value = record.get(field)
        if isinstance(value, bool):
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            values.append(number)
    return values


def score_fields(records: list[dict[str, Any]]) -> list[tuple[str, str, str]]:
    """Return ``(summary_column, metric, score_field)`` triples in stable order."""
    fields: set[str] = set()
    for record in records:
        fields.update(
            key
            for key in record
            if "__" in key and key.endswith("_score")
        )
    triples: list[tuple[str, str, str]] = []
    for field in sorted(fields):
        summary_col, metric_score = field.rsplit("__", 1)
        triples.append((summary_col, metric_score.removesuffix("_score"), field))
    return triples


def score_summary(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Calculate descriptive statistics for every scored summary/metric pair."""
    summary: list[dict[str, Any]] = []
    for summary_col, metric, field in score_fields(records):
        values = _numeric_values(records, field)
        missing = len(records) - len(values)
        summary.append(
            {
                "summary_col": summary_col,
                "metric": metric,
                "mean": sum(values) / len(values) if values else None,
                "median": statistics.median(values) if values else None,
                "stdev": statistics.stdev(values) if len(values) > 1 else 0.0 if values else None,
                "count": len(values),
                "missing": missing,
            }
        )
    return summary


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _fmt(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _bar_chart(summary: list[dict[str, Any]]) -> str:
    """Render a small inline SVG so reports work without matplotlib or JavaScript."""
    usable = [row for row in summary if row["mean"] is not None]
    if not usable:
        return "<p>No finite metric scores were available for a chart.</p>"
    width = 760
    row_height = 32
    height = max(90, 42 + row_height * len(usable))
    parts = [
        f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" aria-label="Mean metric scores">',
        f'<line x1="210" y1="20" x2="710" y2="20" stroke="#cbd5e1" />',
    ]
    for index, row in enumerate(usable):
        y = 30 + index * row_height
        label = escape(f"{row['summary_col']} / {row['metric']}")
        mean = max(0.0, min(1.0, float(row["mean"])))
        bar_width = 500 * mean
        parts.append(f'<text x="4" y="{y + 14}" class="chart-label">{label}</text>')
        parts.append(
            f'<rect x="210" y="{y}" width="{bar_width:.2f}" height="18" rx="3" fill="#2563eb" />'
        )
        parts.append(f'<text x="{min(718, 218 + bar_width):.2f}" y="{y + 14}" class="chart-value">{mean:.3f}</text>')
    parts.append("</svg>")
    return "".join(parts)


def _training_history_section(history: dict[str, Any] | None) -> str:
    if not history:
        return ""
    points = history.get("log_history")
    if not isinstance(points, list):
        return ""
    rows = [point for point in points if isinstance(point, dict)]
    if not rows:
        return ""
    keys = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if key not in {"step", "epoch"} and isinstance(value, (int, float)) and not isinstance(value, bool)
        }
    )
    if not keys:
        return ""
    header = "<tr><th>step</th><th>epoch</th>" + "".join(f"<th>{escape(key)}</th>" for key in keys) + "</tr>"
    body = []
    for row in rows[-25:]:
        cells = [escape(_fmt(row.get("step"))), escape(_fmt(row.get("epoch")))]
        cells.extend(escape(_fmt(row.get(key))) for key in keys)
        body.append("<tr>" + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>")
    return (
        "<h2>Training history (last 25 log entries)</h2>"
        "<p>Numeric Trainer/TensorBoard log values are shown for quick inspection; use TensorBoard for the full run.</p>"
        f"<div class=table-wrap><table><thead>{header}</thead><tbody>{''.join(body)}</tbody></table></div>"
    )


def build_report(
    records: list[dict[str, Any]],
    output_path: str | Path,
    title: str = "VDT FaithfulTextsum results",
    manifest: dict[str, Any] | None = None,
    history: dict[str, Any] | None = None,
) -> tuple[Path, Path]:
    """Write an HTML report and adjacent machine-readable JSON summary."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    metrics = score_summary(records)
    text_fields = [
        key
        for key in ("text", "source", "input", "summary", "human_sum", "llm_sum")
        if all(key in row and isinstance(row[key], str) for row in records)
    ]
    length_rows = []
    for field in text_fields:
        lengths = [len(str(row[field]).split()) for row in records]
        length_rows.append(
            f"<tr><td>{escape(field)}</td><td>{sum(lengths) / len(lengths):.1f}</td>"
            f"<td>{min(lengths)}</td><td>{max(lengths)}</td></tr>"
        )
    metric_rows = []
    for row in metrics:
        metric_rows.append(
            "<tr>"
            + "".join(
                f"<td>{escape(_fmt(row[key]))}</td>"
                for key in ("summary_col", "metric", "mean", "median", "stdev", "count", "missing")
            )
            + "</tr>"
        )
    metadata_rows = []
    for key in ("run_name", "ablation", "status", "finetuning_method", "base_model_path", "training_rows", "evaluation_rows"):
        if manifest and key in manifest:
            metadata_rows.append(f"<tr><td>{escape(key)}</td><td>{escape(_fmt(manifest[key]))}</td></tr>")
    report_data = {
        "title": title,
        "records": len(records),
        "metrics": metrics,
        "text_length_words": {
            field: {
                "mean": sum(len(str(row[field]).split()) for row in records) / len(records),
                "min": min(len(str(row[field]).split()) for row in records),
                "max": max(len(str(row[field]).split()) for row in records),
            }
            for field in text_fields
        },
        "manifest": manifest,
    }
    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>{escape(title)}</title>
<style>
body {{ font: 15px/1.45 system-ui, sans-serif; color:#172033; margin:2rem auto; max-width:1100px; padding:0 1rem; }}
h1 {{ margin-bottom:.25rem; }} h2 {{ margin-top:2rem; }}
.muted {{ color:#64748b; }} .table-wrap {{ overflow-x:auto; }}
table {{ border-collapse:collapse; width:100%; margin:.75rem 0; }}
th,td {{ border:1px solid #dbe3ef; padding:.45rem .6rem; text-align:left; }}
th {{ background:#f1f5f9; }} tr:nth-child(even) {{ background:#f8fafc; }}
.chart {{ width:100%; max-height:600px; background:#f8fafc; border:1px solid #dbe3ef; }}
.chart-label {{ font-size:12px; fill:#334155; }} .chart-value {{ font-size:12px; fill:#172033; }}
</style></head><body>
<h1>{escape(title)}</h1>
<p class="muted">{len(records)} scored record(s). This is a descriptive report, not a significance test.</p>
{('<h2>Run metadata</h2><div class=table-wrap><table><tr><th>Field</th><th>Value</th></tr>' + ''.join(metadata_rows) + '</table></div>') if metadata_rows else ''}
<h2>Mean score visualization</h2>
{_bar_chart(metrics)}
<h2>Metric summary</h2>
<div class=table-wrap><table><tr><th>Summary</th><th>Metric</th><th>Mean</th><th>Median</th><th>Std. dev.</th><th>Valid</th><th>Missing</th></tr>
{''.join(metric_rows) if metric_rows else '<tr><td colspan="7">No score fields found. Run evaluation first.</td></tr>'}</table></div>
<h2>Text length summary (words)</h2>
<div class=table-wrap><table><tr><th>Field</th><th>Mean</th><th>Min</th><th>Max</th></tr>
{''.join(length_rows) if length_rows else '<tr><td colspan="4">No text fields found.</td></tr>'}</table></div>
{_training_history_section(history)}
<p class="muted">The adjacent JSON file contains the same aggregate values for downstream analysis.</p>
</body></html>
"""
    output.write_text(html, encoding="utf-8")
    json_output = output.with_suffix(".json")
    json_output.write_text(json.dumps(_json_safe(report_data), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output, json_output
