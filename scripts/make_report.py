#!/usr/bin/env python3
"""Build a dependency-free HTML/JSON report from scored JSONL and a run folder."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.evaluate.report import build_report, load_jsonl  # noqa: E402


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Scored evaluation JSONL")
    parser.add_argument("--output", default="results/report.html", help="HTML report path")
    parser.add_argument("--run_dir", help="Optional training run folder containing run_manifest.json and training_history.json")
    parser.add_argument("--manifest", help="Optional run manifest JSON path")
    parser.add_argument("--title", default="VDT FaithfulTextsum results")
    args = parser.parse_args()

    try:
        input_path = Path(args.input).expanduser().resolve()
        output_path = Path(args.output).expanduser().resolve()
        records = load_jsonl(input_path)
        manifest = _load_json(Path(args.manifest).expanduser().resolve()) if args.manifest else None
        history = None
        if args.run_dir:
            run_dir = Path(args.run_dir).expanduser().resolve()
            manifest_path = run_dir / "run_manifest.json"
            history_path = run_dir / "training_history.json"
            if manifest is None and manifest_path.is_file():
                manifest = _load_json(manifest_path)
            if history_path.is_file():
                history = _load_json(history_path)
        html_path, json_path = build_report(records, output_path, args.title, manifest, history)
        print(f"HTML report: {html_path}")
        print(f"JSON summary: {json_path}")
        return 0
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
