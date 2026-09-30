#!/usr/bin/env python3
"""Validate a source/reference JSONL file and make a deterministic grouped split."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.training.grpo_data import (  # noqa: E402
    distinct_source_count,
    infer_columns_from_file,
    read_records_from_files,
    split_records,
    write_jsonl,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="One or more same-schema JSONL files; combine them before splitting",
    )
    parser.add_argument("--output_dir", required=True, help="New or existing directory for split files")
    parser.add_argument(
        "--allow_external_output",
        action="store_true",
        help="Explicitly allow writing split files outside this repository (only when approved)",
    )
    parser.add_argument(
        "--source_col",
        default=None,
        help="Source field; auto-detects source for the canonical source/summary schema",
    )
    parser.add_argument(
        "--reference_col",
        default=None,
        help="Human reference field; auto-detects summary for the canonical source/summary schema",
    )
    parser.add_argument("--id_col", default="id")
    parser.add_argument("--validation_fraction", type=float, default=0.1)
    parser.add_argument("--test_fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing existing split files and the manifest",
    )
    args = parser.parse_args()

    try:
        input_paths = [Path(path).resolve() for path in args.input]
        output_dir_arg = Path(args.output_dir).expanduser()
        output_dir = (output_dir_arg if output_dir_arg.is_absolute() else REPO_ROOT / output_dir_arg).resolve()
        if output_dir == REPO_ROOT:
            raise ValueError("Refusing to write split files into the repository root; choose a dedicated results/ folder")
        try:
            output_dir.relative_to(REPO_ROOT)
        except ValueError:
            if not args.allow_external_output:
                raise ValueError(
                    f"Refusing output outside the repository: {output_dir}; use a path under results/ "
                    "or pass --allow_external_output only if that location is approved"
                )
        train_path = output_dir / "train.jsonl"
        validation_path = output_dir / "validation.jsonl"
        test_path = output_dir / "test.jsonl"
        manifest_path = output_dir / "data_manifest.json"
        targets = (train_path, validation_path, test_path, manifest_path)
        if not args.overwrite and any(path.exists() for path in targets):
            existing = ", ".join(str(path) for path in targets if path.exists())
            raise FileExistsError(f"Refusing to overwrite existing output(s): {existing}; pass --overwrite to replace")

        records = read_records_from_files(
            input_paths, args.source_col, args.reference_col, args.id_col
        )
        resolved_columns = [
            infer_columns_from_file(path, args.source_col, args.reference_col)
            for path in input_paths
        ]
        if len(set(resolved_columns)) != 1:
            raise ValueError(
                "All input files must resolve to the same source/reference columns; "
                f"found {resolved_columns}"
            )
        resolved_source_col, resolved_reference_col = resolved_columns[0]
        train, validation, test = split_records(
            records, args.validation_fraction, args.test_fraction, args.seed
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        write_jsonl(train_path, train, overwrite=args.overwrite)
        write_jsonl(validation_path, validation, overwrite=args.overwrite)
        write_jsonl(test_path, test, overwrite=args.overwrite)
        input_file_hashes = [
            {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for path in input_paths
        ]
        combined_digest = hashlib.sha256()
        for item in input_file_hashes:
            combined_digest.update(item["path"].encode("utf-8"))
            combined_digest.update(b"\0")
            combined_digest.update(item["sha256"].encode("ascii"))
        manifest = {
            "input_file_name": input_paths[0].name if len(input_paths) == 1 else None,
            "input_files": input_file_hashes,
            "input_sha256": combined_digest.hexdigest(),
            "id_namespacing": "<input-order>:<filename>:<original-id>" if len(input_paths) > 1 else "preserved",
            "source_column": resolved_source_col,
            "reference_column": resolved_reference_col,
            "id_column": args.id_col,
            "split_seed": args.seed,
            "requested_validation_fraction": args.validation_fraction,
            "requested_test_fraction": args.test_fraction,
            "split_method": "deterministic SHA-256 assignment grouped by normalized exact source text",
            "records_total": len(records),
            "records_train": len(train),
            "records_validation": len(validation),
            "records_test": len(test),
            "distinct_sources_train": distinct_source_count(train),
            "distinct_sources_validation": distinct_source_count(validation),
            "distinct_sources_test": distinct_source_count(test),
            "reference_in_prompt": False,
        }
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(
            f"Validated {len(records)} rows from {len(input_paths)} file(s); wrote train={len(train)}, validation={len(validation)}, "
            f"test={len(test)} to {output_dir}"
        )
        if not validation or not test:
            print("WARNING: at least one held-out split is empty; this is suitable only for a pipeline smoke test.")
        return 0
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
