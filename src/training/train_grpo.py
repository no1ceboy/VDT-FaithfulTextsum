#!/usr/bin/env python3
"""Run an offline, LoRA-based GRPO experiment on prepared JSONL data."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.training.run_utils import (
    TRL_TARGET_VERSION,
    is_inside as _inside,
    package_version as _package_version,
    resolve_path as _resolve_path,
    write_json as _write_json,
)

from src.training.grpo_data import (
    read_records,
    split_records,
    validate_prepared_record,
)  # noqa: E402
from src.training.grpo_rewards import make_metric_reward, reference_char_reward  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Local base-model directory or local PEFT adapter directory")
    parser.add_argument("--base_model", help="Local base-model directory required when --model is an adapter")
    parser.add_argument(
        "--train_jsonl",
        required=True,
        help="Raw id/text/summary JSONL or a prepared source/reference/prompt training JSONL",
    )
    parser.add_argument(
        "--eval_jsonl",
        help="Optional raw or prepared validation JSONL; never pass the test split here",
    )
    parser.add_argument(
        "--internal_validation_fraction",
        type=float,
        default=0.0,
        help="Hold out this fraction from --train_jsonl by normalized source groups when --eval_jsonl is absent",
    )
    parser.add_argument("--output_dir", required=True, help="Unique run directory inside this repository")
    parser.add_argument(
        "--faithfulness_metrics",
        nargs="+",
        choices=("factcc", "minicheck", "alignscore", "mfact"),
        required=True,
    )
    parser.add_argument("--factcc_model_path", help="Local FactCC checkpoint directory")
    parser.add_argument(
        "--mfact_model_path",
        default="models/mfact-vi_VN",
        help="Local Vietnamese mFACT-vi_VN model directory",
    )
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
    parser.add_argument("--reward_batch_size", type=int, default=4)
    parser.add_argument("--reference_weight", type=float, default=0.25)
    parser.add_argument("--metric_weights", nargs="+", type=float, help="One reward weight per selected faithfulness metric")
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--lr_scheduler_type", choices=("linear", "cosine", "constant"), default="linear")
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--optim", default="adamw_torch", help="Transformers optimizer name")
    parser.add_argument("--beta", type=float, default=0.0, help="GRPO KL coefficient; 0 disables a separate reference model")
    parser.add_argument("--num_generations", type=int, default=4)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument(
        "--num_generations_eval",
        type=int,
        default=1,
        help="Validation generations per prompt; 1 keeps eval compatible with a one-example eval batch",
    )
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--max_prompt_tokens", type=int, default=4096, help="Fail if any fully rendered prompt exceeds this length")
    parser.add_argument("--max_completion_length", type=int, default=256)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--max_steps", type=int, default=-1, help="Optional step cap for reproducible smoke/ablation runs")
    parser.add_argument("--precision", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument(
        "--finetuning_method",
        choices=("lora", "qlora", "fft"),
        default="lora",
        help="Train LoRA adapters, 4-bit QLoRA adapters, or all weights (full fine-tuning)",
    )
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target_modules", nargs="+", default=("q_proj", "k_proj", "v_proj", "o_proj"))
    parser.add_argument("--qlora_quant_type", choices=("nf4", "fp4"), default="nf4")
    parser.add_argument("--qlora_double_quant", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_seed", type=int, help="Defaults to --seed")
    parser.add_argument("--eval_strategy", choices=("no", "steps", "epoch"), default="no")
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--report_to", choices=("tensorboard", "none"), default="tensorboard")
    parser.add_argument("--logging_dir", help="TensorBoard event directory; defaults to <output_dir>/tensorboard")
    parser.add_argument("--run_name", help="Tracker/TensorBoard run label; defaults to output directory name")
    parser.add_argument("--experiment_name", default="vdt-faithfultextsum-grpo")
    parser.add_argument("--ablation", default="main", help="Human-readable treatment label for this run")
    parser.add_argument("--tag", action="append", default=[], help="Repeat for searchable run tags")
    parser.add_argument("--notes", default="", help="Short research note saved to the run manifest")
    parser.add_argument("--log_completions", action="store_true", help="Log generated text locally; may expose sensitive data")
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--save_strategy", choices=("steps", "epoch"), default="steps")
    parser.add_argument("--max_checkpoints", type=int, default=2)
    parser.add_argument("--resume_from_checkpoint", help="Resume from a checkpoint inside this run directory")
    parser.add_argument("--reward_scaling", choices=("none", "group", "batch"), default="none")
    parser.add_argument("--use_vllm", action=argparse.BooleanOptionalAction, default=False,
                        help="Use TRL's vLLM rollout engine; requires a compatible preinstalled vLLM")
    parser.add_argument("--vllm_mode", choices=("colocate", "server"), default="colocate")
    parser.add_argument("--vllm_server_base_url", help="Required for --vllm_mode server")
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.25)
    parser.add_argument("--vllm_tensor_parallel_size", type=int, default=1)
    parser.add_argument("--vllm_max_model_length", type=int)
    parser.add_argument("--vllm_enable_sleep_mode", action="store_true")
    parser.add_argument("--vllm_importance_sampling_correction", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow_multiple_metrics", action="store_true", help="Acknowledge that combined metric reward weights are exploratory")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_local_inputs(args: argparse.Namespace) -> tuple[Path, Path, Path, Path | None, Path]:
    model_path = _resolve_path(args.model)
    base_model_path = _resolve_path(args.base_model) if args.base_model else model_path
    train_path = _resolve_path(args.train_jsonl)
    eval_path = _resolve_path(args.eval_jsonl) if args.eval_jsonl else None
    output_path = _resolve_path(args.output_dir)
    if not _inside(output_path, REPO_ROOT) or output_path == REPO_ROOT:
        raise ValueError(f"--output_dir must be a new run directory inside the repository: {REPO_ROOT}")
    if args.logging_dir:
        logging_path = Path(args.logging_dir).expanduser()
        if not logging_path.is_absolute():
            logging_path = output_path / logging_path
        logging_path = logging_path.resolve()
    else:
        logging_path = output_path / "tensorboard"
    if not _inside(logging_path, output_path):
        raise ValueError("--logging_dir must remain inside --output_dir so run artifacts stay together")
    if not model_path.is_dir():
        raise ValueError(f"--model must be an existing local checkpoint directory: {model_path}")
    if not base_model_path.is_dir():
        raise ValueError(f"Base model directory does not exist: {base_model_path}")
    if (model_path / "adapter_config.json").is_file() and not args.base_model:
        raise ValueError("--model is a PEFT adapter; pass --base_model with its local base checkpoint")
    if args.base_model and not (model_path / "adapter_config.json").is_file():
        raise ValueError("--base_model is only used when --model is a PEFT adapter")
    if (model_path / "adapter_config.json").is_file() and args.finetuning_method != "lora":
        raise ValueError("Continuing an existing adapter is supported only with --finetuning_method lora")
    if not train_path.is_file():
        raise ValueError(f"Training JSONL does not exist: {train_path}")
    if eval_path and not eval_path.is_file():
        raise ValueError(f"Validation JSONL does not exist: {eval_path}")
    if args.eval_strategy != "no" and not eval_path and not args.internal_validation_fraction:
        raise ValueError("--eval_strategy requires --eval_jsonl; use the validation split, never the test split")
    if eval_path and args.eval_strategy == "no":
        raise ValueError("--eval_jsonl was provided but --eval_strategy is 'no'; choose steps or epoch")
    if not 0.0 <= args.internal_validation_fraction < 1.0:
        raise ValueError("--internal_validation_fraction must be in [0, 1)")
    if eval_path and args.internal_validation_fraction:
        raise ValueError(
            "Use either --eval_jsonl or --internal_validation_fraction, not both; "
            "the external validation file takes precedence."
        )
    if args.internal_validation_fraction and args.eval_strategy == "no":
        raise ValueError(
            "--internal_validation_fraction requires --eval_strategy epoch or steps; "
            "use --internal_validation_fraction 0 to disable validation."
        )
    if args.resume_from_checkpoint:
        resume_path = _resolve_path(args.resume_from_checkpoint)
        if not resume_path.is_dir() or not _inside(resume_path, output_path):
            raise ValueError("--resume_from_checkpoint must name an existing checkpoint inside --output_dir")
        if not (resume_path / "trainer_state.json").is_file():
            raise ValueError("Resume checkpoint is missing trainer_state.json")
        args.resume_from_checkpoint = str(resume_path)
        manifest_path = output_path / "run_manifest.json"
        if not manifest_path.is_file():
            raise ValueError("Resuming requires the original run_manifest.json in --output_dir")
        previous_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous_manifest.get("training_data_sha256") != _sha256(train_path):
            raise ValueError("Resume checkpoint training-data hash differs from the original run")
        if previous_manifest.get("base_model_path") != str(base_model_path):
            raise ValueError("Resume checkpoint base-model path differs from the original run")
        if previous_manifest.get("finetuning_method", "lora") != args.finetuning_method:
            raise ValueError("Resume checkpoint fine-tuning method differs from the original run")
        args._previous_manifest = previous_manifest
    elif output_path.exists() and any(output_path.iterdir()):
        raise ValueError(f"Output directory is not empty; choose a new run directory: {output_path}")
    for metric in args.faithfulness_metrics:
        if metric == "factcc" and not args.factcc_model_path:
            raise ValueError("--faithfulness_metrics factcc requires --factcc_model_path")
        if metric == "alignscore" and not args.alignscore_ckpt:
            raise ValueError("--faithfulness_metrics alignscore requires --alignscore_ckpt")
        if metric == "mfact" and not args.mfact_model_path:
            raise ValueError("--faithfulness_metrics mfact requires --mfact_model_path")
    if len(set(args.faithfulness_metrics)) != len(args.faithfulness_metrics):
        raise ValueError("Each faithfulness metric may appear only once")
    metric_weights = args.metric_weights or [1.0] * len(args.faithfulness_metrics)
    if len(metric_weights) != len(args.faithfulness_metrics):
        raise ValueError("--metric_weights must have exactly one value per --faithfulness_metrics entry")
    if not math.isfinite(args.reference_weight) or args.reference_weight < 0 or any(
        not math.isfinite(weight) or weight < 0 for weight in metric_weights
    ):
        raise ValueError("Reward weights must be finite and non-negative")
    args.metric_weights = metric_weights
    if (not math.isfinite(args.learning_rate) or args.learning_rate <= 0
            or not math.isfinite(args.beta) or args.beta < 0
            or not math.isfinite(args.weight_decay) or args.weight_decay < 0):
        raise ValueError("learning rate must be positive; beta and weight_decay must be non-negative")
    if (args.num_generations < 2 or args.num_generations_eval < 1
            or args.gradient_accumulation_steps < 1
            or args.per_device_train_batch_size < 1
            or args.per_device_eval_batch_size < 1):
        raise ValueError("num_generations must be >= 2; eval generations, batch sizes, and accumulation must be positive")
    if (args.max_prompt_tokens < 1 or args.max_completion_length < 1
            or not math.isfinite(args.num_train_epochs) or args.num_train_epochs <= 0):
        raise ValueError("prompt/completion limits and num_train_epochs must be positive")
    if args.max_steps == 0 or args.max_steps < -1:
        raise ValueError("--max_steps must be -1 (epoch-based) or a positive integer")
    if (not math.isfinite(args.temperature) or args.temperature <= 0
            or not math.isfinite(args.top_p) or not 0 < args.top_p <= 1 or args.top_k < 0):
        raise ValueError("temperature must be positive, top_p must be in (0, 1], and top_k non-negative")
    if (not math.isfinite(args.warmup_ratio) or not 0 <= args.warmup_ratio < 1
            or not math.isfinite(args.max_grad_norm) or args.max_grad_norm <= 0):
        raise ValueError("warmup_ratio must be in [0, 1) and max_grad_norm must be positive")
    if (args.lora_r < 1 or args.lora_alpha < 1 or not math.isfinite(args.lora_dropout)
            or not 0 <= args.lora_dropout < 1):
        raise ValueError("LoRA rank/alpha must be positive and dropout must be in [0, 1)")
    if args.reward_batch_size < 1 or args.logging_steps < 1 or args.save_steps < 1 or args.max_checkpoints < 1:
        raise ValueError("reward_batch_size, logging_steps, save_steps, and max_checkpoints must be positive")
    if args.eval_steps < 1:
        raise ValueError("eval_steps must be positive")
    if (not math.isfinite(args.vllm_gpu_memory_utilization)
            or not 0 < args.vllm_gpu_memory_utilization <= 1 or args.vllm_tensor_parallel_size < 1):
        raise ValueError("vLLM GPU memory utilization must be in (0, 1] and tensor parallel size must be positive")
    if args.vllm_max_model_length is not None and args.vllm_max_model_length < 1:
        raise ValueError("vllm_max_model_length must be positive when set")
    if args.use_vllm and args.vllm_mode == "server" and not args.vllm_server_base_url:
        raise ValueError("--use_vllm --vllm_mode server requires --vllm_server_base_url")
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
        factcc_path = _resolve_path(args.factcc_model_path)
        if not factcc_path.is_dir():
            raise ValueError("--factcc_model_path must be an existing local model directory")
        args.factcc_model_path = str(factcc_path)
    if "mfact" in args.faithfulness_metrics:
        mfact_path = _resolve_path(args.mfact_model_path)
        if not mfact_path.is_dir():
            raise ValueError("--mfact_model_path must be an existing local model directory")
        missing_assets = [] if (mfact_path / "config.json").is_file() else ["config.json"]
        has_weights = any(
            (mfact_path / name).is_file()
            for name in (
                "pytorch_model.bin",
                "model.safetensors",
                "pytorch_model.bin.index.json",
                "model.safetensors.index.json",
            )
        )
        has_tokenizer = any(
            (mfact_path / name).is_file()
            for name in ("tokenizer.json", "vocab.txt", "tokenizer.model")
        )
        if not has_weights:
            missing_assets.append("model weights")
        if not has_tokenizer:
            missing_assets.append("tokenizer files")
        # Aggregate entries above are checked separately because a checkpoint
        # may use either safetensors or PyTorch weights.
        if missing_assets:
            raise ValueError(
                f"Incomplete local mFACT model directory {mfact_path}; missing: "
                + ", ".join(missing_assets)
            )
        args.mfact_model_path = str(mfact_path)
    if "alignscore" in args.faithfulness_metrics:
        checkpoint_path = _resolve_path(args.alignscore_ckpt)
        if not checkpoint_path.is_file():
            raise ValueError(f"AlignScore checkpoint does not exist: {checkpoint_path}")
        backbone_path = _resolve_path(args.alignscore_backbone_path)
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
    return model_path, base_model_path, train_path, eval_path, output_path


def _rendered_prompt_lengths(dataset: object, tokenizer: object) -> list[tuple[str, int]]:
    lengths: list[tuple[str, int]] = []
    for row in dataset:
        token_ids = tokenizer.apply_chat_template(
            row["prompt"], tokenize=True, add_generation_prompt=True
        )
        lengths.append((str(row["id"]), len(token_ids)))
    return lengths


def _validate_prepared_dataset(dataset, label: str):
    """Validate an already prepared Hugging Face dataset."""
    required_columns = {"id", "source", "reference", "prompt"}
    missing = required_columns - set(dataset.column_names)
    if missing:
        raise ValueError(f"Prepared {label} JSONL is missing columns: {sorted(missing)}")
    if len(dataset) == 0:
        raise ValueError(f"Prepared {label} dataset is empty")
    seen_ids: set[str] = set()
    for row in dataset:
        validate_prepared_record(row)
        if row["id"] in seen_ids:
            raise ValueError(f"Prepared {label} JSONL contains duplicate id {row['id']!r}")
        seen_ids.add(row["id"])
    return dataset


def _load_training_dataset(path: Path, load_dataset, dataset_type, label: str):
    """Load either canonical raw rows or the repository's prepared schema."""
    dataset = load_dataset("json", data_files=str(path), split="train")
    required_columns = {"id", "source", "reference", "prompt"}
    if required_columns.issubset(set(dataset.column_names)):
        return _validate_prepared_dataset(dataset, label), "prepared"

    try:
        raw_records = read_records(path)
    except ValueError as exc:
        missing = sorted(required_columns - set(dataset.column_names))
        raise ValueError(
            f"{path} is neither prepared ({missing} missing) nor a valid raw training file. {exc}"
        ) from exc
    prepared = dataset_type.from_list(raw_records)
    return _validate_prepared_dataset(prepared, label), "raw_auto_prepared"


def _normalized_source(source: str) -> str:
    return unicodedata.normalize("NFC", " ".join(source.split())).casefold()


def main() -> int:
    args = _parse_args()
    output_path: Path | None = None
    manifest_path: Path | None = None
    manifest: dict | None = None
    try:
        model_path, base_model_path, train_path, eval_path, output_path = _check_local_inputs(args)
        manifest_path = output_path / "run_manifest.json"
        args.data_seed = args.seed if args.data_seed is None else args.data_seed

        installed_trl = _package_version("trl")
        if installed_trl != TRL_TARGET_VERSION:
            raise RuntimeError(
                f"This runner targets TRL {TRL_TARGET_VERSION}; found {installed_trl or 'not installed'}. "
                "Do not remove the version check without validating the API."
            )
        if args.use_vllm and importlib.util.find_spec("vllm") is None:
            raise RuntimeError("--use_vllm was requested, but vLLM is not installed in this environment")
        if args.finetuning_method == "qlora" and importlib.util.find_spec("bitsandbytes") is None:
            raise RuntimeError("QLoRA was requested, but bitsandbytes is not installed in this environment")
        if args.report_to == "tensorboard" and importlib.util.find_spec("tensorboard") is None:
            raise RuntimeError("TensorBoard logging was requested, but tensorboard is not installed")

        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        if args.hf_cache_dir:
            os.environ["HF_HUB_CACHE"] = str(_resolve_path(args.hf_cache_dir))

        import torch
        from datasets import Dataset, load_dataset
        from peft import LoraConfig
        from transformers import AutoConfig, AutoTokenizer
        from trl import GRPOConfig, GRPOTrainer

        quantization_config = None
        if args.finetuning_method == "qlora":
            from transformers import BitsAndBytesConfig

            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=args.qlora_quant_type,
                bnb_4bit_compute_dtype=(torch.bfloat16 if args.precision == "bf16" else torch.float16),
                bnb_4bit_use_double_quant=args.qlora_double_quant,
            )

        train_dataset, training_data_format = _load_training_dataset(
            train_path, load_dataset, Dataset, "training"
        )
        if eval_path:
            eval_dataset, validation_data_format = _load_training_dataset(
                eval_path, load_dataset, Dataset, "validation"
            )
        else:
            eval_dataset, validation_data_format = None, None
        validation_source = "external_jsonl" if eval_dataset is not None else None
        source_rows_before_internal_validation = len(train_dataset)
        if eval_dataset is None and args.internal_validation_fraction:
            raw_train_rows = [dict(row) for row in train_dataset]
            train_rows, validation_rows, _ = split_records(
                raw_train_rows,
                validation_fraction=args.internal_validation_fraction,
                test_fraction=0.0,
                seed=args.data_seed,
            )
            train_dataset = Dataset.from_list(train_rows)
            eval_dataset = Dataset.from_list(validation_rows)
            validation_source = "internal_source_grouped_split"
            print(
                f"Using internal validation: train={len(train_dataset)}, validation={len(eval_dataset)} "
                f"from {source_rows_before_internal_validation} input rows (fraction={args.internal_validation_fraction}, seed={args.data_seed})"
            )
        if eval_dataset is not None:
            training_sources = {_normalized_source(row["source"]) for row in train_dataset}
            eval_sources = {_normalized_source(row["source"]) for row in eval_dataset}
            overlap = training_sources & eval_sources
            if overlap:
                raise ValueError(
                    f"Training and validation splits share {len(overlap)} normalized source(s); "
                    "rebuild leakage-resistant splits before training."
                )
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
        eval_lengths = _rendered_prompt_lengths(eval_dataset, tokenizer) if eval_dataset is not None else []
        all_prompt_lengths = lengths + eval_lengths
        too_long = [(record_id, length) for record_id, length in all_prompt_lengths if length > args.max_prompt_tokens]
        if too_long:
            preview = ", ".join(f"{record_id}:{length}" for record_id, length in too_long[:8])
            raise ValueError(
                f"{len(too_long)} prompts exceed --max_prompt_tokens={args.max_prompt_tokens} ({preview}). "
                "No automatic source truncation is performed; increase the limit only if the model context and GPU budget allow it."
            )

        model_config = AutoConfig.from_pretrained(str(base_model_path), local_files_only=True)
        context_limit = getattr(model_config, "max_position_embeddings", None)
        max_prompt_length = max(length for _, length in all_prompt_lengths)
        required_context = max_prompt_length + args.max_completion_length
        if context_limit and required_context > context_limit:
            raise ValueError(
                f"Longest prompt plus completion cap ({required_context}) exceeds model context ({context_limit}); "
                "reduce --max_completion_length or prepare shorter source documents."
            )
        if args.vllm_max_model_length and args.vllm_max_model_length < required_context:
            raise ValueError("--vllm_max_model_length must cover the longest prompt plus --max_completion_length")

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for this training script; no GPU was detected")
        if args.precision == "bf16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16 was selected but this GPU/runtime does not report BF16 support")
        model_dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16
        model_dtype_name = "bfloat16" if args.precision == "bf16" else "float16"

        is_adapter = (model_path / "adapter_config.json").is_file()
        if is_adapter:
            from peft import PeftModel
            from transformers import AutoModelForCausalLM

            base = AutoModelForCausalLM.from_pretrained(
                str(base_model_path), local_files_only=True, torch_dtype=model_dtype
            )
            model = PeftModel.from_pretrained(base, str(model_path), is_trainable=True)
            peft_config = None
        else:
            if args.base_model:
                raise ValueError("--base_model is only used with a PEFT adapter in --model")
            model = str(model_path)
            peft_config = None
            if args.finetuning_method in ("lora", "qlora"):
                peft_config = LoraConfig(
                    r=args.lora_r,
                    lora_alpha=args.lora_alpha,
                    lora_dropout=args.lora_dropout,
                    bias="none",
                    task_type="CAUSAL_LM",
                    target_modules=list(args.lora_target_modules),
                )

        metric_rewards = [
            make_metric_reward(
                metric,
                device=args.reward_device,
                factcc_model_path=args.factcc_model_path,
                mfact_model_path=args.mfact_model_path,
                hf_cache_dir=args.hf_cache_dir,
                alignscore_ckpt=args.alignscore_ckpt,
                alignscore_backbone_path=args.alignscore_backbone_path,
                batch_size=args.reward_batch_size,
            )
            for metric in args.faithfulness_metrics
        ]
        reward_funcs = [reference_char_reward, *metric_rewards]
        reward_weights = [args.reference_weight, *args.metric_weights]
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        effective_batch = world_size * args.per_device_train_batch_size * args.gradient_accumulation_steps
        if effective_batch % args.num_generations:
            raise ValueError(
                f"Effective global batch ({effective_batch}) must be divisible by num_generations "
                f"({args.num_generations}); adjust batch, accumulation, or generation count."
            )
        effective_eval_batch = world_size * args.per_device_eval_batch_size
        if eval_dataset is not None and effective_eval_batch % args.num_generations_eval:
            raise ValueError(
                f"Effective eval batch ({effective_eval_batch}) must be divisible by num_generations_eval "
                f"({args.num_generations_eval}); use --num_generations_eval 1 for eval batch size 1, "
                "or increase --per_device_eval_batch_size."
            )
        if args.use_vllm and args.vllm_mode == "colocate" and args.vllm_tensor_parallel_size > torch.cuda.device_count():
            raise ValueError("Colocated vLLM tensor parallel size exceeds the visible CUDA device count")

        run_name = args.run_name or output_path.name
        logging_path = Path(args.logging_dir).expanduser() if args.logging_dir else output_path / "tensorboard"
        if not logging_path.is_absolute():
            logging_path = output_path / logging_path
        logging_path = logging_path.resolve()
        output_path.mkdir(parents=True, exist_ok=True)
        if args.report_to == "tensorboard":
            logging_path.mkdir(parents=True, exist_ok=True)

        training_args = GRPOConfig(
            output_dir=str(output_path),
            run_name=run_name,
            logging_dir=str(logging_path) if args.report_to == "tensorboard" else None,
            learning_rate=args.learning_rate,
            lr_scheduler_type=args.lr_scheduler_type,
            warmup_ratio=args.warmup_ratio,
            weight_decay=args.weight_decay,
            max_grad_norm=args.max_grad_norm,
            optim=args.optim,
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            num_generations=args.num_generations,
            num_generations_eval=args.num_generations_eval,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            max_completion_length=args.max_completion_length,
            num_train_epochs=args.num_train_epochs,
            max_steps=args.max_steps,
            beta=args.beta,
            reward_weights=reward_weights,
            multi_objective_aggregation="sum_then_normalize",
            scale_rewards=args.reward_scaling,
            remove_unused_columns=False,
            bf16=args.precision == "bf16",
            fp16=args.precision == "fp16",
            gradient_checkpointing=True,
            logging_steps=args.logging_steps,
            logging_first_step=True,
            save_strategy=args.save_strategy,
            save_steps=args.save_steps,
            save_total_limit=args.max_checkpoints,
            eval_strategy=args.eval_strategy,
            eval_steps=args.eval_steps if args.eval_strategy == "steps" else None,
            report_to="tensorboard" if args.report_to == "tensorboard" else "none",
            seed=args.seed,
            data_seed=args.data_seed,
            # Keep the Trainer/TRL config JSON-serializable. TRL resolves this
            # dtype name back to torch.bfloat16/torch.float16 when loading.
            model_init_kwargs={"local_files_only": True, "torch_dtype": model_dtype_name},
            log_completions=args.log_completions,
            use_vllm=args.use_vllm,
            vllm_mode=args.vllm_mode,
            vllm_server_base_url=args.vllm_server_base_url,
            vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            vllm_tensor_parallel_size=args.vllm_tensor_parallel_size,
            vllm_max_model_length=args.vllm_max_model_length,
            vllm_enable_sleep_mode=args.vllm_enable_sleep_mode,
            vllm_importance_sampling_correction=args.vllm_importance_sampling_correction,
        )
        trainer = GRPOTrainer(
            model=model,
            reward_funcs=reward_funcs,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            peft_config=peft_config,
            quantization_config=quantization_config,
        )

        previous_manifest = getattr(args, "_previous_manifest", {})

        runtime_versions = {
            package: _package_version(package)
            for package in ("torch", "transformers", "trl", "peft", "bitsandbytes", "datasets", "accelerate", "vllm", "tensorboard")
        }
        manifest = {
            "experiment": args.experiment_name,
            "run_name": run_name,
            "ablation": args.ablation,
            "tags": args.tag,
            "notes": args.notes,
            "status": "running",
            "started_utc": previous_manifest.get("started_utc") or datetime.now(timezone.utc).isoformat(),
            "resumed_utc": datetime.now(timezone.utc).isoformat() if args.resume_from_checkpoint else None,
            "trl_target_version": TRL_TARGET_VERSION,
            "finetuning_method": args.finetuning_method,
            "base_or_adapter_path": str(model_path),
            "base_model_path": str(base_model_path),
            "training_data": str(train_path),
            "training_data_sha256": _sha256(train_path),
            "training_data_format": training_data_format,
            "training_rows": len(train_dataset),
            "training_rows_before_internal_validation": source_rows_before_internal_validation,
            "validation_data": str(eval_path) if eval_path else None,
            "validation_data_sha256": _sha256(eval_path) if eval_path else None,
            "validation_data_format": validation_data_format,
            "validation_rows": len(eval_dataset) if eval_dataset is not None else 0,
            "validation_source": validation_source,
            "internal_validation_fraction": args.internal_validation_fraction,
            "model_context_limit": context_limit,
            "max_observed_prompt_tokens": max_prompt_length,
            "metrics": args.faithfulness_metrics,
            "reference_reward": "whitespace-insensitive Unicode character n-gram F-beta (beta=2), not canonical SacreBLEU chrF",
            "reference_weight": args.reference_weight,
            "metric_weights": dict(zip(args.faithfulness_metrics, args.metric_weights)),
            "metric_device": args.reward_device,
            "metric_batch_size": args.reward_batch_size,
            "metric_checkpoints": {
                "factcc": str(Path(args.factcc_model_path).expanduser().resolve()) if args.factcc_model_path else None,
                "mfact_model": str(Path(args.mfact_model_path).expanduser().resolve()) if args.mfact_model_path else None,
                "minicheck_model": "flan-t5-large",
                "hf_cache_dir": str(Path(args.hf_cache_dir).expanduser().resolve()) if args.hf_cache_dir else None,
                "alignscore_ckpt": str(Path(args.alignscore_ckpt).expanduser().resolve()) if args.alignscore_ckpt else None,
                "alignscore_backbone": str(Path(args.alignscore_backbone_path).expanduser().resolve()) if args.alignscore_backbone_path else None,
            },
            "optimization": {
                "learning_rate": args.learning_rate,
                "scheduler": args.lr_scheduler_type,
                "warmup_ratio": args.warmup_ratio,
                "weight_decay": args.weight_decay,
                "max_grad_norm": args.max_grad_norm,
                "optimizer": args.optim,
                "beta_kl": args.beta,
                "reward_scaling": args.reward_scaling,
                "max_steps": args.max_steps,
                "num_train_epochs": args.num_train_epochs,
                "seed": args.seed,
                "data_seed": args.data_seed,
                "precision": args.precision,
                "gradient_checkpointing": True,
                "finetuning_method": args.finetuning_method,
            },
            "batching": {
                "per_device_train_batch_size": args.per_device_train_batch_size,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "world_size": world_size,
                "effective_global_batch": effective_batch,
                "num_generations": args.num_generations,
                "per_device_eval_batch_size": args.per_device_eval_batch_size,
                "effective_eval_batch": effective_eval_batch,
                "num_generations_eval": args.num_generations_eval,
            },
            "generation": {
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
                "max_prompt_tokens": args.max_prompt_tokens,
                "max_completion_length": args.max_completion_length,
            },
            "vllm": {
                "enabled": args.use_vllm,
                "mode": args.vllm_mode,
                "server_base_url": args.vllm_server_base_url,
                "gpu_memory_utilization": args.vllm_gpu_memory_utilization,
                "tensor_parallel_size": args.vllm_tensor_parallel_size,
                "max_model_length": args.vllm_max_model_length,
                "sleep_mode": args.vllm_enable_sleep_mode,
                "importance_sampling_correction": args.vllm_importance_sampling_correction,
            },
            "lora": {
                "enabled": args.finetuning_method in ("lora", "qlora"),
                "rank": args.lora_r,
                "alpha": args.lora_alpha,
                "dropout": args.lora_dropout,
                "target_modules": list(args.lora_target_modules),
                "existing_adapter_continued": is_adapter,
            },
            "quantization": ({
                "method": "bitsandbytes_4bit",
                "quant_type": args.qlora_quant_type,
                "double_quant": args.qlora_double_quant,
                "compute_dtype": args.precision,
            } if args.finetuning_method == "qlora" else None),
            "tracking": {
                "report_to": args.report_to,
                "logging_dir": str(logging_path) if args.report_to == "tensorboard" else None,
                "logging_steps": args.logging_steps,
                "save_strategy": args.save_strategy,
                "save_steps": args.save_steps,
                "max_checkpoints": args.max_checkpoints,
                "completion_logging": args.log_completions,
            },
            "cli_args": {key: value for key, value in vars(args).items() if not key.startswith("_")},
            "offline_mode": True,
            "runtime_versions": runtime_versions,
            "cuda_version": torch.version.cuda,
            "gpu_devices": [
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "total_memory_bytes": torch.cuda.get_device_properties(index).total_memory,
                    "free_memory_bytes_at_start": torch.cuda.mem_get_info(index)[0],
                }
                for index in range(torch.cuda.device_count())
            ],
            "model_commit_hash_from_config": getattr(model_config, "_commit_hash", None),
            "limitations": [
                "Faithfulness metrics are not calibrated for Vietnamese; reward use is exploratory.",
                "The human reference is a reward target, not a guarantee of factual correctness.",
                "One reference-overlap reward can bias wording and coverage; inspect held-out outputs and human labels.",
                "TensorBoard files are local to this run; completion text logging is opt-in because summaries may be sensitive.",
            ],
        }
        _write_json(manifest_path, manifest)
        print(f"Starting offline {args.finetuning_method.upper()} GRPO run {run_name!r}; manifest: {manifest_path}")
        print(f"TensorBoard directory: {logging_path if args.report_to == 'tensorboard' else 'disabled'}")
        if args.use_vllm and args.vllm_mode == "colocate":
            print("vLLM is colocated with training and shares GPU memory; monitor utilization/OOMs.")

        train_output = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
        final_model_dir = output_path / ("final_model" if args.finetuning_method == "fft" else "final_adapter")
        trainer.save_model(str(final_model_dir))
        tokenizer.save_pretrained(str(final_model_dir))
        train_metrics = dict(getattr(train_output, "metrics", {}) or {})
        trainer.save_metrics("train", train_metrics)
        _write_json(output_path / "training_history.json", {"log_history": trainer.state.log_history})
        manifest["status"] = "completed"
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["train_metrics"] = train_metrics
        manifest["global_step"] = trainer.state.global_step
        _write_json(manifest_path, manifest)
        print(f"Training complete. Model artifact: {final_model_dir}")
        return 0
    except KeyboardInterrupt:
        if manifest is not None and manifest_path is not None:
            manifest["status"] = "interrupted"
            manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
            _write_json(manifest_path, manifest)
        print("Training interrupted; any saved checkpoint remains in the run directory.", file=sys.stderr)
        return 130
    except Exception as exc:
        if manifest is not None and manifest_path is not None:
            manifest["status"] = "failed"
            manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
            manifest["failure"] = {"type": type(exc).__name__, "message": str(exc)}
            _write_json(manifest_path, manifest)
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
