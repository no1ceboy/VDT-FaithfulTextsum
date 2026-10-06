"""Create a small, auditable human-review packet from claim-audit JSONL."""

from __future__ import annotations

import argparse
import html
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

LABELS = ("supported", "contradicted", "not_supported", "unclear", "not_a_claim")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(value)
    if not rows:
        raise ValueError(f"No records found in {path}")
    return rows


def _flag_kind(rows: Iterable[dict[str, Any]]) -> str:
    flags = {flag for row in rows for flag in row.get("flags", [])}
    has_number = "claim_number_or_date_not_found" in flags
    has_overlap = "low_lexical_evidence" in flags
    if has_number and has_overlap:
        return "both"
    if has_overlap:
        return "low_lexical_evidence"
    if has_number:
        return "claim_number_or_date_not_found"
    return "unflagged"


def select_document_ids(
    audit_rows: list[dict[str, Any]],
    document_count: int = 20,
    flagged_fraction: float = 0.5,
    seed: int = 42,
) -> list[str]:
    """Select a deterministic balanced sample, preserving claim-file order."""
    if document_count < 1:
        raise ValueError("document_count must be positive")
    if not 0 <= flagged_fraction <= 1:
        raise ValueError("flagged_fraction must be between 0 and 1")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    order: list[str] = []
    for row in audit_rows:
        record_id = str(row.get("id"))
        if record_id not in grouped:
            order.append(record_id)
        grouped[record_id].append(row)
    if document_count > len(order):
        raise ValueError(f"Requested {document_count} documents but audit contains only {len(order)}")

    rng = random.Random(seed)
    flagged_groups: dict[str, list[str]] = defaultdict(list)
    for record_id in order:
        flagged_groups[_flag_kind(grouped[record_id])].append(record_id)
    for values in flagged_groups.values():
        rng.shuffle(values)

    target_flagged = min(
        len(order),
        max(0, min(document_count, round(document_count * flagged_fraction))),
    )
    flagged_ids: list[str] = []
    # Round-robin keeps low-overlap cases visible instead of sampling only the
    # much larger number/date-flagged group.
    for kind in ("low_lexical_evidence", "both", "claim_number_or_date_not_found"):
        while flagged_groups[kind] and len(flagged_ids) < target_flagged:
            flagged_ids.append(flagged_groups[kind].pop())
    remaining_flagged = [record_id for kind in flagged_groups for record_id in flagged_groups[kind]]
    rng.shuffle(remaining_flagged)
    flagged_ids.extend(remaining_flagged[: max(0, target_flagged - len(flagged_ids))])

    selected = set(flagged_ids)
    unflagged_ids = [record_id for record_id in order if _flag_kind(grouped[record_id]) == "unflagged"]
    rng.shuffle(unflagged_ids)
    for record_id in unflagged_ids:
        if len(selected) >= document_count:
            break
        selected.add(record_id)
    if len(selected) < document_count:
        fallback = [record_id for record_id in order if record_id not in selected]
        rng.shuffle(fallback)
        selected.update(fallback[: document_count - len(selected)])
    selected_ordered = [record_id for record_id in order if record_id in selected]
    return selected_ordered[:document_count]


def build_review_rows(
    data_rows: list[dict[str, Any]],
    audit_rows: list[dict[str, Any]],
    selected_ids: list[str],
    source_col: str = "input",
    summary_col: str = "output",
) -> list[dict[str, Any]]:
    data_by_id = {str(row.get("id")): row for row in data_rows}
    selected = set(selected_ids)
    output: list[dict[str, Any]] = []
    for audit_row in audit_rows:
        record_id = str(audit_row.get("id"))
        if record_id not in selected:
            continue
        source_row = data_by_id.get(record_id, {})
        evidence = audit_row.get("evidence", [])
        output.append(
            {
                "id": audit_row.get("id"),
                "source": source_row.get(source_col, audit_row.get("source", "")),
                "human_summary": source_row.get(summary_col, ""),
                "summary_col": audit_row.get("summary_col"),
                "claim_index": audit_row.get("claim_index"),
                "claim_count": audit_row.get("claim_count"),
                "claim": audit_row.get("claim", ""),
                "evidence": evidence,
                "automatic_flags": audit_row.get("flags", []),
                "retrieval_score": evidence[0].get("retrieval_score") if evidence else None,
                "human_label": "",
                "reviewer_notes": "",
            }
        )
    if not output:
        raise ValueError("The selected document IDs produced no review claims")
    return output


def _write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def _safe_embedded_json(rows: list[dict[str, Any]]) -> str:
    return (
        json.dumps(rows, ensure_ascii=False)
        .replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("</", "<\\/")
    )


def _write_html(rows: list[dict[str, Any]], path: Path, selected_ids: list[str]) -> None:
    cards: list[str] = []
    for index, row in enumerate(rows):
        evidence_items = "".join(
            "<li><strong>score="
            + html.escape(str(item.get("retrieval_score", "")))
            + "</strong> "
            + html.escape(str(item.get("text", "")))
            + "</li>"
            for item in row.get("evidence", [])
        )
        flags = ", ".join(str(flag) for flag in row.get("automatic_flags", [])) or "none"
        options = "".join(
            f'<option value="{label}">{label}</option>' for label in LABELS
        )
        cards.append(
            f"""
            <article class="claim" data-index="{index}">
              <h3>Document {html.escape(str(row['id']))} · claim {html.escape(str(row.get('claim_index')))}</h3>
              <p><b>Claim</b><br>{html.escape(str(row.get('claim', '')))}</p>
              <details><summary>Human summary</summary><p>{html.escape(str(row.get('human_summary', '')))}</p></details>
              <details><summary>Source document</summary><pre>{html.escape(str(row.get('source', '')))}</pre></details>
              <p><b>Retrieved evidence</b> · automatic flags: {html.escape(flags)}</p>
              <ol>{evidence_items or '<li>No evidence sentence retrieved</li>'}</ol>
              <label>Human label
                <select data-field="human_label">
                  <option value="">Choose one</option>{options}
                </select>
              </label>
              <label>Notes<textarea data-field="reviewer_notes" rows="3" placeholder="Why? Which evidence supports your label?"></textarea></label>
            </article>
            """
        )
    document_ids = ", ".join(map(str, selected_ids))
    data_json = _safe_embedded_json(rows)
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>VDT manual claim review</title>
<style>
body {{ font: 16px system-ui, sans-serif; max-width: 1100px; margin: 2rem auto; padding: 0 1rem; background: #f5f6f8; color: #1f2937; }}
header, .claim {{ background: white; border: 1px solid #d1d5db; border-radius: 10px; padding: 1rem; margin: 1rem 0; }}
.claim {{ border-left: 5px solid #9ca3af; }}
h1 {{ margin-bottom: .25rem; }}
pre {{ white-space: pre-wrap; max-height: 18rem; overflow: auto; background: #f3f4f6; padding: .75rem; border-radius: 6px; }}
label {{ display: block; margin: .75rem 0; font-weight: 650; }}
select, textarea {{ display: block; width: 100%; max-width: 700px; margin-top: .3rem; padding: .45rem; font: inherit; }}
button {{ padding: .65rem 1rem; margin-right: .5rem; cursor: pointer; }}
#progress {{ font-weight: 650; }}
</style></head><body>
<header><h1>Manual claim review</h1>
<p>Selected documents: {html.escape(document_ids)}</p>
<p>Labels: <b>supported</b> = source entails claim; <b>contradicted</b> = source conflicts; <b>not_supported</b> = source lacks evidence; <b>unclear</b> = cannot decide; <b>not_a_claim</b> = heading, label, fragment, or other non-proposition.</p>
<p id="progress">0/{len(rows)} claims labeled</p>
<button type="button" onclick="downloadReview()">Download completed JSONL</button>
<button type="button" onclick="clearSavedReview()">Clear browser draft</button>
<button type="button" onclick="window.scrollTo({{top: 0, behavior: 'smooth'}})">Back to top</button></header>
{''.join(cards)}
<script>
const records = {data_json};
const storageKey = 'vdt-manual-review-{html.escape(path.stem)}';
function collectReview() {{
  return records.map((record, index) => {{
    const card = document.querySelector(`[data-index="${{index}}"]`);
    return {{...record,
      human_label: card.querySelector('[data-field="human_label"]').value,
      reviewer_notes: card.querySelector('[data-field="reviewer_notes"]').value
    }};
  }});
}}
function saveReview() {{
  const draft = collectReview().map(x => ({{human_label: x.human_label, reviewer_notes: x.reviewer_notes}}));
  localStorage.setItem(storageKey, JSON.stringify(draft));
}}
function restoreReview() {{
  try {{
    const draft = JSON.parse(localStorage.getItem(storageKey) || '[]');
    draft.forEach((item, index) => {{
      const card = document.querySelector(`[data-index="${{index}}"]`);
      if (!card) return;
      card.querySelector('[data-field="human_label"]').value = item.human_label || '';
      card.querySelector('[data-field="reviewer_notes"]').value = item.reviewer_notes || '';
    }});
  }} catch (error) {{
    console.warn('Could not restore the browser draft', error);
  }}
}}
function clearSavedReview() {{
  if (!window.confirm('Clear the saved draft for this packet?')) return;
  localStorage.removeItem(storageKey);
  document.querySelectorAll('[data-field="human_label"]').forEach(x => x.value = '');
  document.querySelectorAll('[data-field="reviewer_notes"]').forEach(x => x.value = '');
  updateProgress();
}}
function updateProgress() {{
  const count = [...document.querySelectorAll('[data-field="human_label"]')].filter(x => x.value).length;
  document.getElementById('progress').textContent = `${{count}}/${{records.length}} claims labeled`;
}}
document.querySelectorAll('[data-field="human_label"]').forEach(x => x.addEventListener('change', () => {{ updateProgress(); saveReview(); }}));
document.querySelectorAll('[data-field="reviewer_notes"]').forEach(x => x.addEventListener('input', saveReview));
restoreReview();
updateProgress();
function downloadReview() {{
  saveReview();
  const text = collectReview().map(x => JSON.stringify(x)).join('\\n') + '\\n';
  const blob = new Blob([text], {{type: 'application/x-ndjson'}});
  const link = document.createElement('a');
  link.href = URL.createObjectURL(blob);
  link.download = 'batch_3_manual_review_20.completed.jsonl';
  link.click();
  URL.revokeObjectURL(link.href);
}}
</script></body></html>"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(page, encoding="utf-8")


def create_review_packet(
    data_path: str | Path,
    audit_path: str | Path,
    jsonl_path: str | Path,
    html_path: str | Path,
    *,
    document_count: int = 20,
    flagged_fraction: float = 0.5,
    seed: int = 42,
    source_col: str = "input",
    summary_col: str = "output",
) -> list[dict[str, Any]]:
    audit_rows = _read_jsonl(Path(audit_path))
    selected_ids = select_document_ids(audit_rows, document_count, flagged_fraction, seed)
    review_rows = build_review_rows(
        _read_jsonl(Path(data_path)), audit_rows, selected_ids, source_col, summary_col
    )
    _write_jsonl(review_rows, Path(jsonl_path))
    _write_html(review_rows, Path(html_path), selected_ids)
    return review_rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Cleaned source/reference JSONL")
    parser.add_argument("--audit", required=True, help="Claim-audit JSONL")
    parser.add_argument("--jsonl", required=True, help="Blank-label JSONL template")
    parser.add_argument("--html", required=True, help="Offline browser review form")
    parser.add_argument("--documents", type=int, default=20)
    parser.add_argument("--flagged_fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--source_col", default="input")
    parser.add_argument("--summary_col", default="output")
    args = parser.parse_args()
    rows = create_review_packet(
        args.data,
        args.audit,
        args.jsonl,
        args.html,
        document_count=args.documents,
        flagged_fraction=args.flagged_fraction,
        seed=args.seed,
        source_col=args.source_col,
        summary_col=args.summary_col,
    )
    print(f"Created {len(rows)} claim rows from {args.documents} documents")
    print(f"JSONL template: {args.jsonl}")
    print(f"HTML form: {args.html}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
