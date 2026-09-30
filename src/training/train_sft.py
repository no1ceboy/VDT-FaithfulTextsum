#!/usr/bin/env python3
"""Optional LoRA supervised warm-start from the human summaries."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.training.run_utils import (
    TRL_TARGET_VERSION,
    is_inside,
    package_version,
    resolve_path,
    write_json,
)
from src.training.grpo_data import validate_prepared_record


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Local base-model directory (not an adapter directory)")
    parser.add_argument("--train_jsonl", required=True)
    parser.add_argument("--eval_jsonl", help="Optional validation split; reference NLL only, never factuality")
    parser.add_argument("--output_dir", required=True, help="Unique run directory inside this repository")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lr_scheduler_type", choices=("linear", "cosine", "constant"), default="linear")
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--precision", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_name")
    parser.add_argument("--ablation", default="sft-warm-start")
    parser.add_argument("--report_to", choices=("tensorboard", "none"), default="tensorboard")
    parser.add_argument("--logging_dir")
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_strategy", choices=("steps", "epoch"), default="epoch")
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--max_checkpoints", type=int, default=2)
    parser.add_argument("--eval_strategy", choices=("no", "steps", "epoch"))
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--resume_from_checkpoint")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target_modules", nargs="+", default=("q_proj", "k_proj", "v_proj", "o_proj"))
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    args = _parse_args()
    manifest: dict | None = None
    manifest_path: Path | None = None
    output_path: Path | None = None
    previous_manifest: dict = {}
    try:
        model_path = resolve_path(args.model)
        train_path = resolve_path(args.train_jsonl)
        eval_path = resolve_path(args.eval_jsonl) if args.eval_jsonl else None
        output_path = resolve_path(args.output_dir)
        if not is_inside(output_path, REPO_ROOT) or output_path == REPO_ROOT:
            raise ValueError(f"--output_dir must be inside this repository: {REPO_ROOT}")
        if not model_path.is_dir() or (model_path / "adapter_config.json").exists():
            raise ValueError("--model must be a local base-model directory, not a PEFT adapter")
        if not train_path.is_file() or (eval_path is not None and not eval_path.is_file()):
            raise ValueError("Training/validation JSONL path does not exist")
        eval_strategy = args.eval_strategy or ("epoch" if eval_path else "no")
        if eval_strategy != "no" and eval_path is None:
            raise ValueError("--eval_strategy requires --eval_jsonl")
        if eval_path is not None and eval_strategy == "no":
            raise ValueError("--eval_jsonl was provided but --eval_strategy is 'no'")
        resume_path = resolve_path(args.resume_from_checkpoint) if args.resume_from_checkpoint else None
        if resume_path:
            if not resume_path.is_dir() or not is_inside(resume_path, output_path):
                raise ValueError("--resume_from_checkpoint must be inside the existing --output_dir")
            if not (resume_path / "trainer_state.json").is_file():
                raise ValueError("Resume path is missing trainer_state.json")
            manifest_path = output_path / "run_manifest.json"
            if not manifest_path.is_file():
                raise ValueError("Resuming requires the original run_manifest.json")
            previous_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if previous_manifest.get("training_data_sha256") != _sha256(train_path):
                raise ValueError("Resume checkpoint training-data hash differs from the original run")
            if previous_manifest.get("base_model_path") != str(model_path):
                raise ValueError("Resume checkpoint base-model path differs from the original run")
        elif output_path.exists() and any(output_path.iterdir()):
            raise ValueError(f"Output directory is not empty; choose a unique run directory: {output_path}")
        if (args.learning_rate <= 0 or args.max_length < 1 or args.max_grad_norm <= 0
                or args.weight_decay < 0 or not 0 <= args.warmup_ratio < 1):
            raise ValueError("learning rate/max length/max grad norm must be positive; weight decay non-negative; warmup in [0,1)")
        if args.num_train_epochs <= 0 or args.max_steps == 0 or args.max_steps < -1:
            raise ValueError("num_train_epochs must be positive; max_steps must be -1 or positive")
        if (args.per_device_train_batch_size < 1 or args.gradient_accumulation_steps < 1
                or args.logging_steps < 1 or args.save_steps < 1 or args.max_checkpoints < 1
                or args.eval_steps < 1):
            raise ValueError("batch, accumulation, logging, save, checkpoint, and eval intervals must be positive")
        if args.lora_r < 1 or args.lora_alpha < 1 or not 0 <= args.lora_dropout < 1:
            raise ValueError("LoRA rank/alpha must be positive and dropout must be in [0,1)")
        if package_version("trl") != TRL_TARGET_VERSION:
            raise RuntimeError(f"This runner targets TRL {TRL_TARGET_VERSION}; found {package_version('trl') or 'not installed'}")
        if args.report_to == "tensorboard" and importlib.util.find_spec("tensorboard") is None:
            raise RuntimeError("TensorBoard logging was requested, but tensorboard is not installed")

        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        import torch
        from datasets import load_dataset
        from peft import LoraConfig
        from transformers import AutoConfig, AutoTokenizer
        from trl import SFTConfig, SFTTrainer

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for this training script")
        if args.precision == "bf16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16 was selected but this GPU/runtime does not report BF16 support")
        model_dtype_name = "bfloat16" if args.precision == "bf16" else "float16"
        tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
        if not tokenizer.chat_template:
            raise ValueError("Local tokenizer has no chat template; add/approve one before using conversational prompts")
        if tokenizer.eos_token is None:
            raise ValueError("Local tokenizer must define EOS for padding")
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model_config = AutoConfig.from_pretrained(str(model_path), local_files_only=True)
        context_limit = getattr(model_config, "max_position_embeddings", None)
        if context_limit and args.max_length > context_limit:
            raise ValueError(f"--max_length={args.max_length} exceeds model context limit {context_limit}")

        def load_split(path: Path):
            dataset = load_dataset("json", data_files=str(path), split="train")
            required = {"id", "source", "reference", "prompt"}
            missing = required - set(dataset.column_names)
            if missing:
                raise ValueError(f"{path} is missing columns: {sorted(missing)}")
            if not len(dataset):
                raise ValueError(f"{path} contains no rows")
            seen_ids: set[str] = set()
            sources: set[str] = set()
            for row in dataset:
                validate_prepared_record(row)
                if row["id"] in seen_ids:
                    raise ValueError(f"{path} contains duplicate id {row['id']!r}")
                seen_ids.add(row["id"])
                sources.add(unicodedata.normalize("NFC", " ".join(row["source"].split())).casefold())
            mapped = dataset.map(
                lambda row: {"completion": [{"role": "assistant", "content": row["reference"]}]}
            )
            too_long: list[tuple[str, int]] = []
            for row in mapped:
                messages = row["prompt"] + row["completion"]
                token_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
                if len(token_ids) > args.max_length:
                    too_long.append((str(row["id"]), len(token_ids)))
            if too_long:
                sample = ", ".join(f"{record_id}:{length}" for record_id, length in too_long[:8])
                raise ValueError(f"{len(too_long)} prompt/reference pairs exceed --max_length ({sample}); no truncation is allowed")
            return mapped.select_columns(["prompt", "completion"]), sources

        train_dataset, train_sources = load_split(train_path)
        if eval_path:
            eval_dataset, eval_sources = load_split(eval_path)
        else:
            eval_dataset, eval_sources = None, set()
        if train_sources & eval_sources:
            raise ValueError("Training and validation contain duplicate normalized source documents")
        if len(train_dataset) < 100:
            print("WARNING: fewer than 100 SFT training rows; treat this as a pipeline/sensitivity pilot.", file=sys.stderr)

        run_name = args.run_name or output_path.name
        logging_path = Path(args.logging_dir).expanduser() if args.logging_dir else output_path / "tensorboard"
        if not logging_path.is_absolute():
            logging_path = output_path / logging_path
        logging_path = logging_path.resolve()
        if not is_inside(logging_path, output_path):
            raise ValueError("--logging_dir must remain inside --output_dir")
        config = SFTConfig(
            output_dir=str(output_path),
            run_name=run_name,
            logging_dir=str(logging_path) if args.report_to == "tensorboard" else None,
            learning_rate=args.learning_rate,
            lr_scheduler_type=args.lr_scheduler_type,
            warmup_ratio=args.warmup_ratio,
            weight_decay=args.weight_decay,
            max_grad_norm=args.max_grad_norm,
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            num_train_epochs=args.num_train_epochs,
            max_steps=args.max_steps,
            max_length=args.max_length,
            completion_only_loss=True,
            bf16=args.precision == "bf16",
            fp16=args.precision == "fp16",
            gradient_checkpointing=True,
            logging_steps=args.logging_steps,
            logging_first_step=True,
            save_strategy=args.save_strategy,
            save_steps=args.save_steps,
            save_total_limit=args.max_checkpoints,
            eval_strategy=eval_strategy,
            eval_steps=args.eval_steps if eval_strategy == "steps" else None,
            report_to="tensorboard" if args.report_to == "tensorboard" else "none",
            seed=args.seed,
            data_seed=args.seed,
            # Keep the Trainer/TRL config JSON-serializable. TRL resolves this
            # dtype name back to torch.bfloat16/torch.float16 when loading.
            model_init_kwargs={"local_files_only": True, "torch_dtype": model_dtype_name},
        )
        output_path.mkdir(parents=True, exist_ok=True)
        if args.report_to == "tensorboard":
            logging_path.mkdir(parents=True, exist_ok=True)
        trainer = SFTTrainer(
            model=str(model_path),
            args=config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            peft_config=LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=list(args.lora_target_modules),
            ),
        )
        train_hash = _sha256(train_path)
        manifest_path = output_path / "run_manifest.json"
        manifest = {
            "experiment": "VDT Vietnamese faithful summarization SFT warm-start",
            "run_name": run_name,
            "ablation": args.ablation,
            "status": "running",
            "started_utc": previous_manifest.get("started_utc") or datetime.now(timezone.utc).isoformat(),
            "resumed_utc": datetime.now(timezone.utc).isoformat() if resume_path else None,
            "trl_target_version": TRL_TARGET_VERSION,
            "base_model_path": str(model_path),
            "training_data": str(train_path),
            "training_data_sha256": train_hash,
            "training_rows": len(train_dataset),
            "evaluation_data": str(eval_path) if eval_path else None,
            "evaluation_data_sha256": _sha256(eval_path) if eval_path else None,
            "evaluation_rows": len(eval_dataset) if eval_dataset is not None else 0,
            "evaluation_metric": "reference token negative log-likelihood only; not factuality",
            "model_commit_hash_from_config": getattr(model_config, "_commit_hash", None),
            "cli_args": vars(args),
            "max_length": args.max_length,
            "model_context_limit": context_limit,
            "seed": args.seed,
            "precision": args.precision,
            "report_to": args.report_to,
            "logging_dir": str(logging_path) if args.report_to == "tensorboard" else None,
            "lora_config": {
                "r": args.lora_r,
                "lora_alpha": args.lora_alpha,
                "lora_dropout": args.lora_dropout,
                "target_modules": list(args.lora_target_modules),
            },
            "runtime_versions": {
                package: package_version(package)
                for package in ("torch", "transformers", "trl", "peft", "datasets", "accelerate", "tensorboard")
            },
            "cuda_version": torch.version.cuda,
            "gpu_devices": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
        }
        write_json(manifest_path, manifest)
        train_output = trainer.train(resume_from_checkpoint=str(resume_path) if resume_path else None)
        final_path = output_path / "final_adapter"
        trainer.save_model(str(final_path))
        tokenizer.save_pretrained(str(final_path))
        metrics = dict(getattr(train_output, "metrics", {}) or {})
        trainer.save_metrics("train", metrics)
        write_json(output_path / "training_history.json", {"log_history": trainer.state.log_history})
        manifest["status"] = "completed"
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["train_metrics"] = metrics
        write_json(manifest_path, manifest)
        print(f"SFT complete. Adapter: {final_path}")
        return 0
    except KeyboardInterrupt:
        if manifest is not None and manifest_path is not None:
            manifest["status"] = "interrupted"
            manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
            write_json(manifest_path, manifest)
        print("Training interrupted; any saved checkpoint remains in the run directory.", file=sys.stderr)
        return 130
    except Exception as exc:
        if manifest is not None and manifest_path is not None:
            manifest["status"] = "failed"
            manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
            manifest["failure"] = {"type": type(exc).__name__, "message": str(exc)}
            write_json(manifest_path, manifest)
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
