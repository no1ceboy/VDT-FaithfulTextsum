#!/usr/bin/env python3
"""Run an offline, LoRA-based GRPO experiment on prepared JSONL data."""

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
from src.training.grpo_rewards import make_metric_reward, reference_char_reward  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Local base-model directory or local PEFT adapter directory")
    parser.add_argument("--base_model", help="Local base-model directory required when --model is an adapter")
    parser.add_argument("--train_jsonl", required=True, help="Training split from prepare_grpo_data.py")
    parser.add_argument("--output_dir", required=True, help="New directory for adapter, checkpoints, and run manifest")
    parser.add_argument("--faithfulness_metrics", nargs="+", choices=("factcc", "minicheck", "alignscore"), required=True)
    parser.add_argument("--factcc_model_path", help="Local FactCC checkpoint directory")
    parser.add_argument(
        "--alignscore_ckpt",
        default="models/alignscore/AlignScore-base.ckpt",
        help="Local AlignScore-base .ckpt file",
    )
    parser.add_argument(
        "--alignscore_backbone_path",
        default="models/roberta-base",
        help="Local roberta-base model/tokenizer folder paired with AlignScore-base",
    )
    parser.add_argument("--hf_cache_dir", default="models/hf-cache", help="Local MiniCheck cache/model folder")
    parser.add_argument("--reward_device", default="cpu", help="Device for metric models (cpu recommended for GPU headroom)")
    parser.add_argument("--reference_weight", type=float, default=0.25)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--beta", type=float, default=0.0, help="GRPO KL coefficient; 0 disables a separate reference model")
    parser.add_argument("--num_generations", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max_prompt_tokens", type=int, default=4096, help="Fail if any fully rendered prompt exceeds this length")
    parser.add_argument("--max_completion_length", type=int, default=256)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--max_checkpoints", type=int, default=2)
    parser.add_argument("--allow_multiple_metrics", action="store_true", help="Acknowledge that combined metric reward weights are exploratory")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_local_inputs(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    model_path = Path(args.model).expanduser().resolve()
    base_model_path = Path(args.base_model).expanduser().resolve() if args.base_model else model_path
    train_path = Path(args.train_jsonl).expanduser().resolve()
    output_path = Path(args.output_dir).expanduser().resolve()
    if not model_path.is_dir():
        raise ValueError(f"--model must be an existing local checkpoint directory: {model_path}")
    if not base_model_path.is_dir():
        raise ValueError(f"Base model directory does not exist: {base_model_path}")
    if (model_path / "adapter_config.json").is_file() and not args.base_model:
        raise ValueError("--model is a PEFT adapter; pass --base_model with its local base checkpoint")
    if args.base_model and not (model_path / "adapter_config.json").is_file():
        raise ValueError("--base_model is only used when --model is a PEFT adapter")
    if not train_path.is_file():
        raise ValueError(f"Training JSONL does not exist: {train_path}")
    if output_path.exists() and any(output_path.iterdir()):
        raise ValueError(f"Output directory is not empty; choose a new run directory: {output_path}")
    for metric in args.faithfulness_metrics:
        if metric == "factcc" and not args.factcc_model_path:
            raise ValueError("--faithfulness_metrics factcc requires --factcc_model_path")
        if metric == "alignscore" and not args.alignscore_ckpt:
            raise ValueError("--faithfulness_metrics alignscore requires --alignscore_ckpt")
    if args.reference_weight < 0 or args.learning_rate <= 0 or args.beta < 0:
        raise ValueError("reference weight and beta must be non-negative; learning rate must be positive")
    if args.num_generations < 2 or args.gradient_accumulation_steps < 1:
        raise ValueError("num_generations must be >= 2 and gradient_accumulation_steps must be >= 1")
    if args.max_prompt_tokens < 1 or args.max_completion_length < 1 or args.num_train_epochs <= 0:
        raise ValueError("prompt/completion limits and num_train_epochs must be positive")
    if args.temperature <= 0:
        raise ValueError("temperature must be positive")
    if args.logging_steps < 1 or args.save_steps < 1 or args.max_checkpoints < 1:
        raise ValueError("logging_steps, save_steps, and max_checkpoints must be positive")
    if len(args.faithfulness_metrics) > 1 and not args.allow_multiple_metrics:
        raise ValueError("Select one metric for the first run; pass --allow_multiple_metrics to acknowledge a multi-metric pilot")
    if "minicheck" in args.faithfulness_metrics:
        cache_path = Path(args.hf_cache_dir).expanduser()
        if not cache_path.is_absolute():
            cache_path = REPO_ROOT / cache_path
        if not cache_path.is_dir():
            raise ValueError("MiniCheck requires --hf_cache_dir pointing to the local Hugging Face cache")
        args.hf_cache_dir = str(cache_path.resolve())
    if "factcc" in args.faithfulness_metrics:
        if not Path(args.factcc_model_path).expanduser().is_dir():
            raise ValueError("--factcc_model_path must be an existing local model directory")
    if "alignscore" in args.faithfulness_metrics:
        checkpoint_path = Path(args.alignscore_ckpt).expanduser()
        if not checkpoint_path.is_absolute():
            checkpoint_path = REPO_ROOT / checkpoint_path
        if not checkpoint_path.is_file():
            raise ValueError(f"AlignScore checkpoint does not exist: {checkpoint_path}")
        backbone_path = Path(args.alignscore_backbone_path).expanduser()
        if not backbone_path.is_absolute():
            backbone_path = REPO_ROOT / backbone_path
        if not backbone_path.is_dir():
            raise ValueError(f"Local roberta-base folder does not exist: {backbone_path}")
        required_assets = ("config.json",)
        missing_assets = [name for name in required_assets if not (backbone_path / name).is_file()]
        has_weights = any(
            (backbone_path / name).is_file()
            for name in (
                "model.safetensors",
                "pytorch_model.bin",
                "model.safetensors.index.json",
                "pytorch_model.bin.index.json",
            )
        )
        has_tokenizer = (backbone_path / "tokenizer.json").is_file() or (
            (backbone_path / "vocab.json").is_file() and (backbone_path / "merges.txt").is_file()
        )
        if not has_weights:
            missing_assets.append("model.safetensors or pytorch_model.bin")
        if not has_tokenizer:
            missing_assets.append("tokenizer.json or vocab.json plus merges.txt")
        if missing_assets:
            raise ValueError(
                f"Incomplete local roberta-base folder {backbone_path}; missing: "
                + ", ".join(missing_assets)
            )
        args.alignscore_ckpt = str(checkpoint_path)
        args.alignscore_backbone_path = str(backbone_path)
    return model_path, base_model_path, train_path, output_path


def _rendered_prompt_lengths(dataset: object, tokenizer: object) -> list[tuple[str, int]]:
    lengths: list[tuple[str, int]] = []
    for row in dataset:
        token_ids = tokenizer.apply_chat_template(
            row["prompt"], tokenize=True, add_generation_prompt=True
        )
        lengths.append((str(row["id"]), len(token_ids)))
    return lengths


def main() -> int:
    args = _parse_args()
    try:
        model_path, base_model_path, train_path, output_path = _check_local_inputs(args)
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        if args.hf_cache_dir:
            os.environ["HF_HUB_CACHE"] = str(Path(args.hf_cache_dir).expanduser().resolve())

        import torch
        from datasets import load_dataset
        from peft import LoraConfig
        from transformers import AutoConfig, AutoTokenizer
        from trl import GRPOConfig, GRPOTrainer
        from importlib.metadata import version

        if version("trl") != "0.29.0":
            raise RuntimeError(f"This script targets TRL 0.29.0 exactly; found TRL {version('trl')}")

        train_dataset = load_dataset("json", data_files=str(train_path), split="train")
        required_columns = {"id", "source", "reference", "prompt"}
        missing = required_columns - set(train_dataset.column_names)
        if missing:
            raise ValueError(f"Prepared training JSONL is missing columns: {sorted(missing)}")
        if len(train_dataset) == 0:
            raise ValueError("Prepared training dataset is empty")
        seen_ids: set[str] = set()
        for row in train_dataset:
            validate_prepared_record(row)
            if row["id"] in seen_ids:
                raise ValueError(f"Prepared training JSONL contains duplicate id {row['id']!r}")
            seen_ids.add(row["id"])
        if len(train_dataset) < 100:
            print(
                f"WARNING: only {len(train_dataset)} training rows; treat results as a pipeline smoke test, not evidence of generalization.",
                file=sys.stderr,
            )

        tokenizer = AutoTokenizer.from_pretrained(str(base_model_path), local_files_only=True)
        if not tokenizer.chat_template:
            raise ValueError("Local tokenizer has no chat template; add/approve one before using conversational prompts")
        if tokenizer.eos_token is None:
            raise ValueError("Local tokenizer must define an EOS token for GRPO padding")
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        lengths = _rendered_prompt_lengths(train_dataset, tokenizer)
        too_long = [(record_id, length) for record_id, length in lengths if length > args.max_prompt_tokens]
        if too_long:
            preview = ", ".join(f"{record_id}:{length}" for record_id, length in too_long[:8])
            raise ValueError(
                f"{len(too_long)} prompts exceed --max_prompt_tokens={args.max_prompt_tokens} ({preview}). "
                "No automatic source truncation is performed; increase the limit only if the model context and GPU budget allow it."
            )

        model_config = AutoConfig.from_pretrained(str(base_model_path), local_files_only=True)
        context_limit = getattr(model_config, "max_position_embeddings", None)
        if context_limit and max(length for _, length in lengths) + args.max_completion_length > context_limit:
            raise ValueError(
                f"Longest prompt plus completion cap exceeds model context ({context_limit} tokens); "
                "reduce --max_completion_length or prepare shorter source documents."
            )

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for this training script; no GPU was detected")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("This pilot is configured for BF16; use a BF16-capable GPU/runtime")

        is_adapter = (model_path / "adapter_config.json").is_file()
        if is_adapter:
            from peft import PeftModel
            from transformers import AutoModelForCausalLM

            base = AutoModelForCausalLM.from_pretrained(
                str(base_model_path), local_files_only=True, torch_dtype=torch.bfloat16
            )
            model = PeftModel.from_pretrained(base, str(model_path), is_trainable=True)
            peft_config = None
        else:
            if args.base_model:
                raise ValueError("--base_model is only used with a PEFT adapter in --model")
            model = str(model_path)
            peft_config = LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            )

        metric_rewards = [
            make_metric_reward(
                metric,
                device=args.reward_device,
                factcc_model_path=args.factcc_model_path,
                hf_cache_dir=args.hf_cache_dir,
                alignscore_ckpt=args.alignscore_ckpt,
                alignscore_backbone_path=args.alignscore_backbone_path,
                batch_size=4,
            )
            for metric in args.faithfulness_metrics
        ]
        reward_funcs = [reference_char_reward, *metric_rewards]
        reward_weights = [args.reference_weight, *([1.0] * len(metric_rewards))]
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        effective_batch = world_size * args.gradient_accumulation_steps
        if effective_batch % args.num_generations:
            raise ValueError(
                f"Effective train batch ({effective_batch}) must divide evenly by num_generations "
                f"({args.num_generations}); adjust accumulation or generation count."
            )

        output_path.mkdir(parents=True, exist_ok=True)
        training_args = GRPOConfig(
            output_dir=str(output_path),
            learning_rate=args.learning_rate,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            num_generations=args.num_generations,
            temperature=args.temperature,
            max_completion_length=args.max_completion_length,
            num_train_epochs=args.num_train_epochs,
            beta=args.beta,
            reward_weights=reward_weights,
            multi_objective_aggregation="sum_then_normalize",
            scale_rewards="none",
            remove_unused_columns=False,
            bf16=True,
            fp16=False,
            gradient_checkpointing=True,
            logging_steps=args.logging_steps,
            logging_first_step=True,
            save_strategy="steps",
            save_steps=args.save_steps,
            save_total_limit=args.max_checkpoints,
            eval_strategy="no",
            report_to="none",
            seed=args.seed,
            data_seed=args.seed,
            model_init_kwargs={"local_files_only": True, "torch_dtype": torch.bfloat16},
            log_completions=False,
        )
        # Passing the existing LoRA adapter as a model (rather than peft_config)
        # continues its trainable weights; a base checkpoint receives a fresh adapter.
        trainer = GRPOTrainer(
            model=model,
            reward_funcs=reward_funcs,
            args=training_args,
            train_dataset=train_dataset,
            processing_class=tokenizer,
            peft_config=peft_config,
        )
        manifest = {
            "experiment": "VDT Vietnamese faithful summarization GRPO pilot",
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "trl_target_version": "0.29.0",
            "base_or_adapter_path": str(model_path),
            "base_model_path": str(base_model_path),
            "training_data": str(train_path),
            "training_data_sha256": _sha256(train_path),
            "training_rows": len(train_dataset),
            "model_context_limit": context_limit,
            "max_observed_prompt_tokens": max(length for _, length in lengths),
            "metrics": args.faithfulness_metrics,
            "reference_reward": "whitespace-insensitive Unicode character n-gram F-beta (beta=2), not canonical SacreBLEU chrF",
            "reference_weight": args.reference_weight,
            "metric_weights": {metric: 1.0 for metric in args.faithfulness_metrics},
            "metric_device": args.reward_device,
            "metric_checkpoints": {
                "factcc": str(Path(args.factcc_model_path).expanduser().resolve()) if args.factcc_model_path else None,
                "minicheck_model": "flan-t5-large",
                "hf_cache_dir": str(Path(args.hf_cache_dir).expanduser().resolve())
                if args.hf_cache_dir
                else None,
                "alignscore_ckpt": str(Path(args.alignscore_ckpt).expanduser().resolve())
                if args.alignscore_ckpt
                else None,
                "alignscore_backbone": str(Path(args.alignscore_backbone_path).expanduser().resolve())
                if args.alignscore_backbone_path
                else None,
            },
            "num_generations": args.num_generations,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "effective_global_batch": effective_batch,
            "generation": {
                "temperature": args.temperature,
                "max_completion_length": args.max_completion_length,
                "vllm": False,
            },
            "learning_rate": args.learning_rate,
            "beta_kl": args.beta,
            "reward_scaling": "none",
            "seed": args.seed,
            "bf16": True,
            "uses_lora": True,
            "lora_config": {
                "r": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.05,
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            },
            "existing_adapter_continued": is_adapter,
            "offline_mode": True,
            "runtime_versions": {
                package: __import__("importlib.metadata", fromlist=["version"]).version(package)
                for package in ("torch", "transformers", "trl", "peft", "datasets", "accelerate")
            },
            "cuda_version": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(torch.cuda.current_device()),
            "gpu_free_bytes_at_manifest_time": torch.cuda.mem_get_info(torch.cuda.current_device())[0],
            "model_commit_hash_from_config": getattr(model_config, "_commit_hash", None),
            "limitations": [
                "Faithfulness metrics are not calibrated for Vietnamese; reward use is exploratory.",
                "The human reference is a reward target, not a guarantee of factual correctness.",
                "One reference-overlap reward can bias wording and coverage; inspect held-out outputs and human labels.",
            ],
        }
        (output_path / "run_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"Starting offline LoRA GRPO; run manifest: {output_path / 'run_manifest.json'}")
        trainer.train()
        trainer.save_model(str(output_path / "final_adapter"))
        tokenizer.save_pretrained(str(output_path / "final_adapter"))
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        (output_path / "run_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"Training complete. Final adapter: {output_path / 'final_adapter'}")
        return 0
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
