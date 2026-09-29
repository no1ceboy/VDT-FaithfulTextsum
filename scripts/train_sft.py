#!/usr/bin/env python3
"""Optional LoRA supervised warm-start from the human summaries."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.training.grpo_data import validate_prepared_record  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Local base-model directory (not an adapter directory)")
    parser.add_argument("--train_jsonl", required=True)
    parser.add_argument("--eval_jsonl", help="Optional held-out split; used for reference NLL only")
    parser.add_argument("--output_dir", required=True, help="New output directory for the SFT adapter")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    try:
        model_path = Path(args.model).expanduser().resolve()
        train_path = Path(args.train_jsonl).expanduser().resolve()
        eval_path = Path(args.eval_jsonl).expanduser().resolve() if args.eval_jsonl else None
        output_path = Path(args.output_dir).expanduser().resolve()
        if not model_path.is_dir() or (model_path / "adapter_config.json").exists():
            raise ValueError("--model must be a local base-model directory, not a PEFT adapter")
        if not train_path.is_file():
            raise ValueError(f"Training JSONL does not exist: {train_path}")
        if eval_path and not eval_path.is_file():
            raise ValueError(f"Evaluation JSONL does not exist: {eval_path}")
        if output_path.exists() and any(output_path.iterdir()):
            raise ValueError(f"Output directory is not empty; choose a new run directory: {output_path}")
        if args.learning_rate <= 0 or args.max_length < 1:
            raise ValueError("learning_rate and max_length must be positive")

        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        import torch
        from datasets import load_dataset
        from peft import LoraConfig
        from transformers import AutoConfig, AutoTokenizer
        from trl import SFTConfig, SFTTrainer
        from importlib.metadata import version

        if version("trl") != "0.29.0":
            raise RuntimeError(f"This script targets TRL 0.29.0 exactly; found TRL {version('trl')}")

        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("This pilot requires a CUDA GPU with BF16 support")
        tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
        if not tokenizer.chat_template:
            raise ValueError("Local tokenizer has no chat template; add/approve one before using conversational prompts")
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
            seen_ids: set[str] = set()
            for row in dataset:
                validate_prepared_record(row)
                if row["id"] in seen_ids:
                    raise ValueError(f"{path} contains duplicate id {row['id']!r}")
                seen_ids.add(row["id"])
            mapped = dataset.map(
                lambda row: {"completion": [{"role": "assistant", "content": row["reference"]}]}
            )
            # Validate the full prompt+target before SFTConfig can truncate it.
            too_long: list[tuple[str, int]] = []
            for row in mapped:
                messages = row["prompt"] + row["completion"]
                token_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
                if len(token_ids) > args.max_length:
                    too_long.append((str(row["id"]), len(token_ids)))
            if too_long:
                sample = ", ".join(f"{record_id}:{length}" for record_id, length in too_long[:8])
                raise ValueError(
                    f"{len(too_long)} prompt/reference pairs exceed --max_length={args.max_length} ({sample}); "
                    "no truncation is allowed in this run. Increase the limit within the model context or shorten inputs explicitly."
                )
            return mapped.select_columns(["prompt", "completion"])

        train_dataset = load_split(train_path)
        if len(train_dataset) == 0:
            raise ValueError("Training dataset is empty")
        eval_dataset = load_split(eval_path) if eval_path else None

        config = SFTConfig(
            output_dir=str(output_path),
            learning_rate=args.learning_rate,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            num_train_epochs=args.num_train_epochs,
            max_length=args.max_length,
            completion_only_loss=True,
            bf16=True,
            gradient_checkpointing=True,
            logging_steps=1,
            save_strategy="epoch",
            eval_strategy="epoch" if eval_dataset is not None else "no",
            report_to="none",
            seed=args.seed,
            data_seed=args.seed,
            model_init_kwargs={"local_files_only": True, "torch_dtype": torch.bfloat16},
        )
        output_path.mkdir(parents=True, exist_ok=True)
        trainer = SFTTrainer(
            model=str(model_path),
            args=config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            peft_config=LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            ),
        )
        digest = hashlib.sha256(train_path.read_bytes()).hexdigest()
        manifest = {
            "experiment": "VDT Vietnamese faithful summarization SFT warm-start",
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "trl_target_version": "0.29.0",
            "base_model_path": str(model_path),
            "training_data": str(train_path),
            "training_data_sha256": digest,
            "training_rows": len(train_dataset),
            "evaluation_data": str(eval_path) if eval_path else None,
            "evaluation_data_sha256": hashlib.sha256(eval_path.read_bytes()).hexdigest() if eval_path else None,
            "evaluation_metric": "reference token negative log-likelihood only; not factuality",
            "model_commit_hash_from_config": getattr(model_config, "_commit_hash", None),
            "learning_rate": args.learning_rate,
            "max_length": args.max_length,
            "model_context_limit": context_limit,
            "seed": args.seed,
            "bf16": True,
            "uses_lora": True,
            "lora_config": {
                "r": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.05,
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            },
            "offline_mode": True,
            "runtime_versions": {
                package: version(package)
                for package in ("torch", "transformers", "trl", "peft", "datasets", "accelerate")
            },
            "cuda_version": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(torch.cuda.current_device()),
        }
        (output_path / "run_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        trainer.train()
        final_path = output_path / "final_adapter"
        trainer.save_model(str(final_path))
        tokenizer.save_pretrained(str(final_path))
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        (output_path / "run_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"SFT complete. Use this directory as --model for train_grpo.py: {final_path}")
        return 0
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
