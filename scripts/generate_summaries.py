#!/usr/bin/env python3
"""Generate summaries from prepared rows or raw ``id/text/summary`` JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.training.grpo_data import build_prompt, validate_prepared_record  # noqa: E402


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: each JSONL row must be an object")
            if isinstance(row.get("prompt"), list):
                validate_prepared_record(row)
            else:
                source = row.get("text", row.get("source"))
                reference = row.get("summary")
                if not isinstance(source, str) or not source.strip():
                    raise ValueError(
                        f"{path}:{line_number}: raw generation rows require a non-empty source field"
                    )
                if not isinstance(reference, str) or not reference.strip():
                    raise ValueError(
                        f"{path}:{line_number}: raw generation rows require a non-empty summary field"
                    )
                style = row.get("style")
                if style is not None and not isinstance(style, str):
                    raise ValueError(f"{path}:{line_number}: style must be a string when present")
                row = {
                    "id": str(row.get("id", line_number)).strip(),
                    "source": source.strip(),
                    "reference": reference.strip(),
                    "prompt": build_prompt(source, style),
                    **({"style": style} if style is not None else {}),
                }
                validate_prepared_record(row)
            rows.append(row)
    if not rows:
        raise ValueError(f"No records found in {path}")
    identifiers = [str(row["id"]) for row in rows]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError(f"{path}: duplicate id found")
    return rows


def _read_existing(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or "id" not in row:
                raise ValueError(f"{path}:{line_number}: existing output requires an id field")
            record_id = str(row["id"])
            if record_id in records:
                raise ValueError(f"{path}:{line_number}: duplicate id {record_id!r}")
            records[record_id] = row
    return records


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Local base model or PEFT adapter directory")
    parser.add_argument("--base_model", help="Local base checkpoint required when --model is an adapter")
    parser.add_argument(
        "--input_jsonl",
        required=True,
        help="Prepared JSONL, or raw canonical rows with id/text/summary",
    )
    parser.add_argument("--existing_jsonl", help="Optional prior generation output to merge by id")
    parser.add_argument("--output", required=True, help="New JSONL output compatible with scripts/run_eval.py")
    parser.add_argument("--summary_col", default="llm_sum", help="Name of generated summary field")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--max_input_tokens", type=int, default=4096)
    args = parser.parse_args()

    try:
        model_path = Path(args.model).expanduser().resolve()
        base_model_path = Path(args.base_model).expanduser().resolve() if args.base_model else model_path
        input_path = Path(args.input_jsonl).expanduser().resolve()
        existing_path = Path(args.existing_jsonl).expanduser().resolve() if args.existing_jsonl else None
        output_path = Path(args.output).expanduser().resolve()
        if not model_path.is_dir() or not base_model_path.is_dir():
            raise ValueError("--model and --base_model (when used) must be existing local directories")
        if (model_path / "adapter_config.json").exists() and not args.base_model:
            raise ValueError("--model is a PEFT adapter; pass --base_model with its local base checkpoint")
        if args.base_model and not (model_path / "adapter_config.json").exists():
            raise ValueError("--base_model is only used when --model is a PEFT adapter")
        if not input_path.is_file():
            raise ValueError(f"Input JSONL does not exist: {input_path}")
        if existing_path and not existing_path.is_file():
            raise ValueError(f"Existing generation JSONL does not exist: {existing_path}")
        if args.summary_col in {"id", "input", "source", "human_sum", "reference"}:
            raise ValueError("summary_col must not replace a reserved id/source/reference column")
        if output_path.exists():
            raise FileExistsError(f"Refusing to replace existing output: {output_path}")
        manifest_path = output_path.with_name(output_path.name + ".manifest.json")
        if manifest_path.exists():
            raise FileExistsError(f"Refusing to replace existing manifest: {manifest_path}")
        if args.batch_size < 1 or args.max_new_tokens < 1 or args.max_input_tokens < 1:
            raise ValueError("batch_size, max_new_tokens, and max_input_tokens must be positive")

        rows = _read_jsonl(input_path)
        existing = _read_existing(existing_path) if existing_path else None
        if existing is not None:
            expected_ids = {str(row["id"]) for row in rows}
            if set(existing) != expected_ids:
                raise ValueError("Existing output IDs do not exactly match the prepared input IDs")
            for row in rows:
                old = existing[str(row["id"])]
                old_source = old.get("text", old.get("source", old.get("input")))
                old_reference = old.get("human_sum", old.get("reference"))
                if old_source != row["source"] or old_reference != row["reference"]:
                    raise ValueError(f"Existing row {row['id']!r} has a different source or human reference")
                if args.summary_col in old:
                    raise ValueError(f"Existing row {row['id']!r} already has summary column {args.summary_col!r}")

        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        import torch
        from peft import PeftModel
        from importlib.metadata import version
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("Generation requires a CUDA GPU with BF16 support")
        from transformers import AutoConfig

        model_config = AutoConfig.from_pretrained(str(base_model_path), local_files_only=True)
        context_limit = getattr(model_config, "max_position_embeddings", None)
        tokenizer = AutoTokenizer.from_pretrained(str(base_model_path), local_files_only=True)
        if not tokenizer.chat_template:
            raise ValueError("Local tokenizer has no chat template")
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        model = AutoModelForCausalLM.from_pretrained(
            str(base_model_path), local_files_only=True, torch_dtype=torch.bfloat16
        )
        if (model_path / "adapter_config.json").exists():
            model = PeftModel.from_pretrained(model, str(model_path), is_trainable=False)
        model.to("cuda")
        model.eval()

        prompt_lengths = [
            len(tokenizer.apply_chat_template(row["prompt"], tokenize=True, add_generation_prompt=True))
            for row in rows
        ]
        too_long = [
            (str(row["id"]), length)
            for row, length in zip(rows, prompt_lengths)
            if length > args.max_input_tokens
        ]
        if too_long:
            preview = ", ".join(f"{record_id}:{length}" for record_id, length in too_long[:8])
            raise ValueError(
                f"{len(too_long)} prompts exceed --max_input_tokens={args.max_input_tokens} ({preview}); "
                "no automatic truncation is performed."
            )
        if context_limit and max(prompt_lengths) + args.max_new_tokens > context_limit:
            raise ValueError(
                f"Longest prompt plus generation cap exceeds model context ({context_limit}); "
                "reduce --max_new_tokens or prepare shorter inputs."
            )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = output_path.with_name(output_path.name + ".partial")
        if temporary_path.exists():
            raise FileExistsError(f"Refusing to overwrite stale partial output: {temporary_path}")
        try:
            with temporary_path.open("x", encoding="utf-8", newline="\n") as stream:
                for start in range(0, len(rows), args.batch_size):
                    batch = rows[start : start + args.batch_size]
                    rendered = [
                        tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
                        for prompt in (row["prompt"] for row in batch)
                    ]
                    tokens = tokenizer(
                        rendered,
                        return_tensors="pt",
                        padding=True,
                        add_special_tokens=False,
                    ).to("cuda")
                    with torch.inference_mode():
                        generated = model.generate(
                            **tokens,
                            do_sample=False,
                            max_new_tokens=args.max_new_tokens,
                            pad_token_id=tokenizer.pad_token_id,
                            eos_token_id=tokenizer.eos_token_id,
                        )
                    prompt_width = tokens["input_ids"].shape[1]
                    texts = tokenizer.batch_decode(generated[:, prompt_width:], skip_special_tokens=True)
                    for row, text in zip(batch, texts):
                        result = (
                            dict(existing[str(row["id"])])
                            if existing is not None
                            else {
                                "id": row["id"],
                                "source": row["source"],
                                "human_sum": row["reference"],
                            }
                        )
                        result[args.summary_col] = text.strip()
                        if "style" in row:
                            result["style"] = row["style"]
                        stream.write(json.dumps(result, ensure_ascii=False) + "\n")
                    stream.flush()
                    print(f"Generated {min(start + len(batch), len(rows))}/{len(rows)}", flush=True)
            temporary_path.replace(output_path)
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise
        manifest = {
            "experiment": "VDT summarization generation",
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "model_or_adapter_path": str(model_path),
            "base_model_path": str(base_model_path),
            "model_commit_hash_from_config": getattr(model_config, "_commit_hash", None),
            "input_jsonl": str(input_path),
            "input_sha256": _sha256(input_path),
            "existing_output_jsonl": str(existing_path) if existing_path else None,
            "existing_output_sha256": _sha256(existing_path) if existing_path else None,
            "output_jsonl": str(output_path),
            "output_sha256": _sha256(output_path),
            "records": len(rows),
            "summary_column": args.summary_col,
            "decoding": {
                "do_sample": False,
                "max_new_tokens": args.max_new_tokens,
                "batch_size": args.batch_size,
            },
            "runtime_versions": {
                package: version(package)
                for package in ("torch", "transformers", "peft")
            },
            "cuda_version": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(torch.cuda.current_device()),
            "offline_mode": True,
        }
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Saved summaries to {output_path}")
        print(f"Generation manifest: {manifest_path}")
        return 0
    except (OSError, ValueError, RuntimeError, ImportError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
