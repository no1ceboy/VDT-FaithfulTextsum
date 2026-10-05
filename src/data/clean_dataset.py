"""Create a clean, auditable copy of a summarization JSONL dataset.

The cleaner is deliberately dependency-free so it can run on the company
machine without downloading anything.  It cleans the source/document field
only by default and leaves the reference summary unchanged.  The input file
is never modified.

Supported source fields are ``text``, ``source``, and the legacy ``input``.
Supported reference fields are ``summary``, ``human_sum``, ``abstract_sum``,
``output``, and ``reference``.  Explicit column arguments always win.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tempfile
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator


URL_TOKEN = "[URL]"
REPLACEMENT_CHARACTER = "\ufffd"
SOURCE_CANDIDATES = ("text", "source", "input")
REFERENCE_CANDIDATES = ("summary", "human_sum", "abstract_sum", "output", "reference")

_MARKDOWN_LINK_RE = re.compile(
    r"\[([^\]]+)\]\(\s*(?:https?://|www\.)[^)]*\)", re.IGNORECASE
)
_URL_RE = re.compile(r"(?<![\w])(?:https?://|www\.)[^\s<>\[\]\"']+", re.IGNORECASE)
_HTML_TAG_RE = re.compile(r"</?[A-Za-z][^>]*>")
_SEPARATOR_RE = re.compile(r"^\s*(?:[-_=*~]){3,}\s*$")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_ICON_RE = re.compile(
    r"[\u2022\u25aa\u25ab\u25a0\u25a1\u25c6\u25c7\u25cf\u25cb\u200d\ufe0e\ufe0f"
    r"\U0001f000-\U0001faff\u2600-\u27bf\u2b00-\u2bff]"
)
_TRAILING_URL_PUNCTUATION = ".,;:!?)]}"

@dataclass
class TextCleaningStats:
    """Counters explaining changes made to one text field."""

    markdown_links_replaced: int = 0
    urls_replaced: int = 0
    html_tags_removed: int = 0
    icon_chars_removed: int = 0
    separator_lines_removed: int = 0
    control_chars_removed: int = 0
    replacement_chars_replaced: int = 0
    whitespace_normalized: bool = False

    @property
    def changed(self) -> bool:
        return any(
            (
                self.markdown_links_replaced,
                self.urls_replaced,
                self.html_tags_removed,
                self.icon_chars_removed,
                self.separator_lines_removed,
                self.control_chars_removed,
                self.replacement_chars_replaced,
                self.whitespace_normalized,
            )
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "markdown_links_replaced": self.markdown_links_replaced,
            "urls_replaced": self.urls_replaced,
            "html_tags_removed": self.html_tags_removed,
            "icon_chars_removed": self.icon_chars_removed,
            "separator_lines_removed": self.separator_lines_removed,
            "control_chars_removed": self.control_chars_removed,
            "replacement_chars_replaced": self.replacement_chars_replaced,
            "whitespace_normalized": self.whitespace_normalized,
        }


@dataclass
class CleaningReport:
    """Dataset-level report written beside the cleaned JSONL."""

    input_path: str
    output_path: str
    source_col: str
    reference_col: str | None
    replacement_policy: str
    clean_reference: bool
    drop_stale_token_lengths: bool
    records_read: int = 0
    records_written: int = 0
    records_dropped: int = 0
    records_changed: int = 0
    records_unchanged: int = 0
    records_source_changed: int = 0
    records_reference_changed: int = 0
    records_metadata_changed: int = 0
    drop_reasons: Counter[str] = field(default_factory=Counter)
    change_counts: Counter[str] = field(default_factory=Counter)
    dropped_ids: list[str] = field(default_factory=list)
    metadata_fields_removed: Counter[str] = field(default_factory=Counter)
    input_sha256: str = ""
    output_sha256: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "input_path": self.input_path,
            "output_path": self.output_path,
            "source_col": self.source_col,
            "reference_col": self.reference_col,
            "replacement_policy": self.replacement_policy,
            "clean_reference": self.clean_reference,
            "drop_stale_token_lengths": self.drop_stale_token_lengths,
            "records_read": self.records_read,
            "records_written": self.records_written,
            "records_dropped": self.records_dropped,
            "records_changed": self.records_changed,
            "records_unchanged": self.records_unchanged,
            "records_source_changed": self.records_source_changed,
            "records_reference_changed": self.records_reference_changed,
            "records_metadata_changed": self.records_metadata_changed,
            "drop_reasons": dict(self.drop_reasons),
            "change_counts": dict(self.change_counts),
            "dropped_ids": self.dropped_ids,
            "metadata_fields_removed": dict(self.metadata_fields_removed),
            "input_sha256": self.input_sha256,
            "output_sha256": self.output_sha256,
        }


def infer_column(record: dict[str, Any], explicit: str | None, candidates: Iterable[str], kind: str) -> str:
    """Resolve a field name without silently guessing a missing explicit field."""
    if explicit:
        if explicit not in record:
            available = ", ".join(sorted(map(str, record))) or "<none>"
            raise ValueError(f"{kind} column {explicit!r} is missing; fields found: {available}")
        return explicit
    for candidate in candidates:
        if candidate in record:
            return candidate
    available = ", ".join(sorted(map(str, record))) or "<none>"
    raise ValueError(f"Could not infer the {kind} column; fields found: {available}")


def _replace_url(match: re.Match[str], stats: TextCleaningStats) -> str:
    """Replace a URL but retain sentence punctuation after it."""
    value = match.group(0)
    trailing = ""
    while value and value[-1] in _TRAILING_URL_PUNCTUATION:
        trailing = value[-1] + trailing
        value = value[:-1]
    if not value:
        return match.group(0)
    stats.urls_replaced += 1
    return URL_TOKEN + trailing


def clean_text(text: str, *, replacement_policy: str = "keep") -> tuple[str, TextCleaningStats]:
    """Clean one source string and return the transformed text plus counters.

    ``replacement_policy`` controls U+FFFD only.  ``drop`` is handled by the
    file-level function because it needs to remove the complete record.
    """
    if not isinstance(text, str):
        raise TypeError("Text values must be strings")
    stats = TextCleaningStats()
    original = text
    text = unicodedata.normalize("NFC", text)

    def markdown_replacement(match: re.Match[str]) -> str:
        stats.markdown_links_replaced += 1
        label = match.group(1).strip()
        return f"{label} {URL_TOKEN}".strip()

    text = _MARKDOWN_LINK_RE.sub(markdown_replacement, text)

    def html_replacement(_match: re.Match[str]) -> str:
        stats.html_tags_removed += 1
        return " "

    text = _HTML_TAG_RE.sub(html_replacement, text)
    text = _URL_RE.sub(lambda match: _replace_url(match, stats), text)

    if replacement_policy == "replace":
        stats.replacement_chars_replaced = text.count(REPLACEMENT_CHARACTER)
        text = text.replace(REPLACEMENT_CHARACTER, " ")
    text, stats.icon_chars_removed = _ICON_RE.subn(" ", text)
    text, stats.control_chars_removed = _CONTROL_RE.subn(" ", text)

    lines: list[str] = []
    for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = re.sub(r"[ \t\f\v]+", " ", line).strip()
        if _SEPARATOR_RE.fullmatch(line):
            stats.separator_lines_removed += 1
            continue
        lines.append(line)
    text = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
    stats.whitespace_normalized = text != original and any(
        marker in original for marker in ("\r", "\t", "  ", "\n\n\n")
    )
    return text, stats


def _has_replacement_character(value: Any) -> bool:
    return isinstance(value, str) and REPLACEMENT_CHARACTER in value


def _read_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: each JSONL row must be an object")
            yield line_number, row


def _check_output_targets(paths: Iterable[Path], overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Refusing to overwrite existing output(s): {joined}; pass --overwrite to replace")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clean_file(
    input_path: str | Path,
    output_path: str | Path,
    *,
    source_col: str | None = None,
    reference_col: str | None = None,
    replacement_policy: str = "drop",
    clean_reference: bool = False,
    drop_stale_token_lengths: bool = False,
    report_path: str | Path | None = None,
    audit_log_path: str | Path | None = None,
    write_audit_log: bool = True,
    overwrite: bool = False,
) -> CleaningReport:
    """Clean a JSONL file without changing the input file.

    Rows with missing/empty source, missing/empty reference (when supplied),
    or replacement characters are dropped by default.  Every dropped or
    changed row is recorded in the audit JSONL.
    """
    if replacement_policy not in {"drop", "replace", "keep"}:
        raise ValueError("replacement_policy must be one of: drop, replace, keep")
    input_file = Path(input_path).expanduser().resolve()
    output_file = Path(output_path).expanduser().resolve()
    if not input_file.is_file():
        raise FileNotFoundError(f"Input JSONL does not exist: {input_file}")
    if input_file == output_file:
        raise ValueError("Output must be a different file from input; the cleaner never edits input in place")

    report_file = (
        Path(report_path).expanduser().resolve()
        if report_path
        else output_file.with_name(output_file.name + ".report.json")
    )
    audit_file = (
        Path(audit_log_path).expanduser().resolve()
        if audit_log_path
        else output_file.with_name(output_file.name + ".audit.jsonl")
    )
    targets = [output_file, report_file]
    if write_audit_log:
        targets.append(audit_file)
    _check_output_targets(targets, overwrite)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.parent.mkdir(parents=True, exist_ok=True)
    if write_audit_log:
        audit_file.parent.mkdir(parents=True, exist_ok=True)

    source_name: str | None = None
    reference_name: str | None = None
    report = CleaningReport(
        input_path=str(input_file),
        output_path=str(output_file),
        source_col="",
        reference_col=None,
        replacement_policy=replacement_policy,
        clean_reference=clean_reference,
        drop_stale_token_lengths=drop_stale_token_lengths,
        input_sha256=_sha256(input_file),
    )

    output_temp: Path | None = None
    audit_temp: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", delete=False,
            dir=output_file.parent, prefix=output_file.name + ".", suffix=".partial"
        ) as output_stream:
            output_temp = Path(output_stream.name)
            audit_stream = None
            if write_audit_log:
                audit_stream = tempfile.NamedTemporaryFile(
                    mode="w", encoding="utf-8", newline="\n", delete=False,
                    dir=audit_file.parent, prefix=audit_file.name + ".", suffix=".partial"
                )
                audit_temp = Path(audit_stream.name)
            try:
                for line_number, raw in _read_jsonl(input_file):
                    report.records_read += 1
                    if source_name is None:
                        source_name = infer_column(raw, source_col, SOURCE_CANDIDATES, "source")
                        reference_name = (
                            infer_column(raw, reference_col, REFERENCE_CANDIDATES, "reference")
                            if reference_col or any(name in raw for name in REFERENCE_CANDIDATES)
                            else None
                        )
                        report.source_col = source_name
                        report.reference_col = reference_name

                    record_id = str(raw.get("id", line_number)).strip() or str(line_number)
                    reasons: list[str] = []
                    if not isinstance(raw.get(source_name), str):
                        reasons.append("source_not_string")
                    elif not raw[source_name].strip():
                        reasons.append("empty_source")
                    if reference_name is not None:
                        if not isinstance(raw.get(reference_name), str):
                            reasons.append("reference_not_string")
                        elif not raw[reference_name].strip():
                            reasons.append("empty_reference")
                    if replacement_policy == "drop" and (
                        _has_replacement_character(raw.get(source_name))
                        or (reference_name is not None and _has_replacement_character(raw.get(reference_name)))
                    ):
                        reasons.append("replacement_character")

                    source_stats = TextCleaningStats()
                    reference_stats = TextCleaningStats()
                    cleaned_source = raw.get(source_name)
                    cleaned_reference = raw.get(reference_name) if reference_name is not None else None
                    if not reasons or not any(reason.endswith("not_string") for reason in reasons):
                        if isinstance(cleaned_source, str):
                            cleaned_source, source_stats = clean_text(
                                cleaned_source, replacement_policy=replacement_policy
                            )
                            if not cleaned_source and "empty_source" not in reasons:
                                reasons.append("empty_source_after_cleaning")
                        if isinstance(cleaned_reference, str):
                            if clean_reference:
                                cleaned_reference, reference_stats = clean_text(
                                    cleaned_reference, replacement_policy=replacement_policy
                                )
                            elif replacement_policy == "replace":
                                cleaned_reference = cleaned_reference.replace(REPLACEMENT_CHARACTER, " ")
                                reference_stats.replacement_chars_replaced = (
                                    raw[reference_name].count(REPLACEMENT_CHARACTER)
                                )
                            if reference_name is not None and not cleaned_reference.strip():
                                reasons.append("empty_reference_after_cleaning")

                    if reasons:
                        report.records_dropped += 1
                        for reason in reasons:
                            report.drop_reasons[reason] += 1
                        if len(report.dropped_ids) < 1000:
                            report.dropped_ids.append(record_id)
                        if audit_stream is not None:
                            audit_stream.write(json.dumps({
                                "line": line_number,
                                "id": record_id,
                                "action": "dropped",
                                "reasons": reasons,
                                "source_changes": source_stats.as_dict(),
                                "reference_changes": reference_stats.as_dict(),
                            }, ensure_ascii=False) + "\n")
                        continue

                    cleaned = dict(raw)
                    cleaned[source_name] = cleaned_source
                    if reference_name is not None and clean_reference:
                        cleaned[reference_name] = cleaned_reference
                    elif reference_name is not None and replacement_policy == "replace" and reference_stats.changed:
                        cleaned[reference_name] = cleaned_reference
                    metadata_changed = drop_stale_token_lengths and "qwen3_token_length" in cleaned
                    if metadata_changed:
                        del cleaned["qwen3_token_length"]
                        report.metadata_fields_removed["qwen3_token_length"] += 1

                    source_changed = cleaned_source != raw[source_name]
                    reference_changed = (
                        reference_name is not None
                        and cleaned_reference != raw[reference_name]
                    )
                    row_changed = cleaned != raw
                    if source_changed:
                        report.records_source_changed += 1
                    if reference_changed:
                        report.records_reference_changed += 1
                    if metadata_changed:
                        report.records_metadata_changed += 1
                    if row_changed:
                        report.records_changed += 1
                    else:
                        report.records_unchanged += 1
                    for key, value in source_stats.as_dict().items():
                        if isinstance(value, bool):
                            if value:
                                report.change_counts[key] += 1
                        elif value:
                            report.change_counts[key] += value
                    for key, value in reference_stats.as_dict().items():
                        if isinstance(value, bool):
                            if value:
                                report.change_counts[f"reference_{key}"] += 1
                        elif value:
                            report.change_counts[f"reference_{key}"] += value
                    if audit_stream is not None and row_changed:
                        audit_stream.write(json.dumps({
                            "line": line_number,
                            "id": record_id,
                            "action": "kept_and_changed",
                            "source_changes": source_stats.as_dict(),
                            "reference_changes": reference_stats.as_dict(),
                            "removed_fields": [
                                key for key in raw if key not in cleaned
                            ],
                        }, ensure_ascii=False) + "\n")
                    output_stream.write(json.dumps(cleaned, ensure_ascii=False) + "\n")
                    report.records_written += 1
            finally:
                if audit_stream is not None:
                    audit_stream.close()

        if source_name is None:
            raise ValueError(f"{input_file}: no non-empty JSONL records found")
        output_temp.replace(output_file)
        output_temp = None
        if audit_temp is not None:
            audit_temp.replace(audit_file)
            audit_temp = None
        report.output_sha256 = _sha256(output_file)
        report_file.write_text(
            json.dumps(report.as_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return report
    finally:
        if output_temp is not None:
            output_temp.unlink(missing_ok=True)
        if audit_temp is not None:
            audit_temp.unlink(missing_ok=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input JSONL; it is never modified")
    parser.add_argument("--output", required=True, help="Separate cleaned JSONL output")
    parser.add_argument("--source_col", help="Source field; auto-detects text, source, then input")
    parser.add_argument(
        "--reference_col",
        help="Optional reference field; auto-detects summary, human_sum, abstract_sum, output, or reference",
    )
    parser.add_argument(
        "--replacement_policy",
        choices=("drop", "replace", "keep"),
        default="drop",
        help="Handling for the Unicode replacement character U+FFFD (default: drop the row)",
    )
    parser.add_argument(
        "--clean_reference",
        action="store_true",
        help="Apply icon/link/formatting cleanup to the reference too; default leaves it unchanged",
    )
    parser.add_argument(
        "--drop_stale_token_lengths",
        action="store_true",
        help="Remove qwen3_token_length because source cleanup makes it stale",
    )
    parser.add_argument("--report", dest="report_path", help="JSON report path")
    parser.add_argument("--audit_log", dest="audit_log_path", help="JSONL change/drop audit path")
    parser.add_argument("--no_audit_log", action="store_true", help="Do not write the per-row audit JSONL")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing output/report/audit files")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        report = clean_file(
            args.input,
            args.output,
            source_col=args.source_col,
            reference_col=args.reference_col,
            replacement_policy=args.replacement_policy,
            clean_reference=args.clean_reference,
            drop_stale_token_lengths=args.drop_stale_token_lengths,
            report_path=args.report_path,
            audit_log_path=args.audit_log_path,
            write_audit_log=not args.no_audit_log,
            overwrite=args.overwrite,
        )
    except (OSError, TypeError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 2
    print(
        f"Cleaned {report.records_read} rows; wrote {report.records_written}, "
        f"dropped {report.records_dropped}, changed {report.records_changed}."
    )
    print(f"Output: {report.output_path}")
    print(f"Report: {args.report_path or report.output_path + '.report.json'}")
    if not args.no_audit_log:
        print(f"Audit: {args.audit_log_path or report.output_path + '.audit.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
