"""Run factuality metrics on one or more summary columns.

Example:
    python -m src.evaluate.run_eval --data /data/vdt/summaries.jsonl \
        --summary_cols human_sum llm_sum --metrics factcc minicheck alignscore rouge \
        --factcc_model_path models/factcc \
        --minicheck_model_path models/minicheck \
        --alignscore_ckpt models/alignscore/AlignScore-large.ckpt \
        --alignscore_backbone_path models/roberta-large \
        --nltk_data_dir models/nltk_data --offline \
        --batch_size 2 --limit 10 --output results/smoke.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
from functools import partial
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _project_path(value: str) -> Path:
    """Resolve relative model paths from the repository root, not the shell cwd."""
    path = Path(value).expanduser()
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def _nltk_data_root(value: str) -> Path:
    """Accept the NLTK data root or its nested tokenizers/punkt_tab directory."""
    path = Path(value).expanduser().resolve()
    if (path / "tokenizers" / "punkt_tab" / "english").is_dir():
        return path
    if path.name == "tokenizers" and (path / "punkt_tab" / "english").is_dir():
        return path.parent
    if path.name == "punkt_tab" and path.parent.name == "tokenizers" and (path / "english").is_dir():
        return path.parent.parent
    if (
        path.name == "english"
        and path.parent.name == "punkt_tab"
        and path.parent.parent.name == "tokenizers"
    ):
        return path.parent.parent.parent
    return path


def _build_registry(args: argparse.Namespace) -> dict[str, Any]:
    """Return evaluator factories so only one model is resident at a time."""
    device = args.device
    cuda_device = int(device.split(":", 1)[1]) if device.startswith("cuda:") else 0
    registry: dict[str, Any] = {}
    if "factcc" in args.metrics:
        from .factcc_eval import FactCCEvaluator

        registry["factcc"] = partial(
            FactCCEvaluator,
            device=device,
            model_path=str(_project_path(args.factcc_model_path)) if args.factcc_model_path else None,
            batch_size=args.batch_size,
        )
    if "fenice" in args.metrics:
        from .fenice_eval import FENICEEvaluator

        registry["fenice"] = partial(FENICEEvaluator, device=device, batch_size=args.batch_size)
    if "minicheck" in args.metrics:
        from .minicheck_eval import MiniCheckEvaluator

        registry["minicheck"] = partial(
            MiniCheckEvaluator,
            device=device,
            model_path=str(_project_path(args.minicheck_model_path)) if args.minicheck_model_path else None,
            cache_dir=str(_project_path(args.hf_cache_dir)) if args.hf_cache_dir else None,
            batch_size=args.batch_size,
            chunk_size=args.minicheck_chunk_size,
        )
    if "alignscore" in args.metrics:
        from .alignscore_eval import AlignScoreEvaluator

        registry["alignscore"] = partial(
            AlignScoreEvaluator,
            device=device,
            model_path=args.alignscore_ckpt,
            backbone_path=str(_project_path(args.alignscore_backbone_path)) if args.alignscore_backbone_path else None,
            cache_dir=str(_project_path(args.hf_cache_dir)) if args.hf_cache_dir else None,
            batch_size=args.batch_size,
            evaluation_mode=args.alignscore_mode,
        )
    if "qafacteval" in args.metrics:
        from .qafacteval_eval import QAFactEvalEvaluator

        registry["qafacteval"] = partial(
            QAFactEvalEvaluator,
            device=device,
            model_path=args.qafacteval_model_path,
            batch_size=args.batch_size,
            cuda_device=cuda_device,
        )
    if "rouge" in args.metrics:
        from .rouge_eval import RougeEvaluator

        registry["rouge"] = partial(RougeEvaluator, reference_col=args.reference_col)
    if "bertscore" in args.metrics:
        from .bertscore_eval import BertScoreEvaluator

        registry["bertscore"] = partial(
            BertScoreEvaluator,
            device=device,
            model_path=args.bertscore_model_path,
            num_layers=args.bertscore_num_layers,
            reference_col=args.reference_col,
            batch_size=args.batch_size,
        )
    return registry


def load_records(path: str) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")
    with p.open(encoding="utf-8") as file:
        if p.suffix.lower() == ".json":
            data = json.load(file)
            records = data if isinstance(data, list) else [data]
        else:
            records = [json.loads(line) for line in file if line.strip()]
    if not all(isinstance(record, dict) for record in records):
        raise ValueError("Every dataset record must be a JSON object")
    return records


def _validate_records(records: list[dict[str, Any]], columns: list[str]) -> None:
    if not records:
        raise ValueError("The dataset contains no records")
    for row_index, record in enumerate(records, start=1):
        for column in columns:
            if column not in record:
                raise ValueError(f"Record {row_index} is missing column {column!r}")
            value = record[column]
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Record {row_index}, column {column!r} must be non-empty text")


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def save_results(results: list[dict[str, Any]], output_path: str) -> None:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as file:
        for record in results:
            file.write(json.dumps(_json_safe(record), ensure_ascii=False, allow_nan=False) + "\n")
    logger.info("Saved %d records to %s", len(results), out)


def save_summary(
    results: list[dict[str, Any]], summary_cols: list[str], metrics: list[str], output_path: str
) -> None:
    summary_path = Path(output_path).with_suffix(".summary.tsv")
    lines = ["summary_col\tmetric\tmean_score\tcount\tn_nan"]
    for summary_col in summary_cols:
        for metric in metrics:
            score_col = f"{summary_col}__{metric}_score"
            values: list[float] = []
            n_nan = 0
            for record in results:
                value = record.get(score_col)
                try:
                    numeric_value = float(value)
                except (TypeError, ValueError):
                    n_nan += 1
                    continue
                if math.isfinite(numeric_value):
                    values.append(numeric_value)
                else:
                    n_nan += 1
            mean = f"{sum(values) / len(values):.6f}" if values else "N/A"
            lines.append(f"{summary_col}\t{metric}\t{mean}\t{len(values)}\t{n_nan}")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))


def save_errors(errors: list[tuple[str, str, str]], output_path: str) -> None:
    if not errors:
        return
    error_path = Path(output_path).with_suffix(".errors.tsv")
    error_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["summary_col\tmetric\terror"]
    for summary_col, metric, message in errors:
        clean_message = " ".join(message.split()).replace("\t", " ")
        lines.append(f"{summary_col}\t{metric}\t{clean_message}")
    error_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.error("Some evaluations failed; details saved to %s", error_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run factuality metrics on human and model-generated summaries.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data", required=True, help="Input JSONL or JSON file")
    parser.add_argument("--source_col", default="input", help="Source document field")
    parser.add_argument("--summary_col", default=None, help="One summary field (legacy/single-column mode)")
    parser.add_argument("--summary_cols", nargs="+", default=None, help="Summary fields to score in one run")
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=["factcc", "fenice", "minicheck", "alignscore", "qafacteval", "rouge", "bertscore"],
        default=["factcc", "fenice", "minicheck", "alignscore", "qafacteval"],
    )
    parser.add_argument(
        "--reference_col",
        default="human_sum",
        help="Human/reference summary field for ROUGE and BERTScore",
    )
    parser.add_argument(
        "--bertscore_model_path",
        default=None,
        help="Local Hugging Face encoder/tokenizer directory for BERTScore (no download)",
    )
    parser.add_argument(
        "--bertscore_num_layers",
        type=int,
        default=9,
        help="Transformer layer for BERTScore; default 9 suits bert-base-multilingual-cased",
    )
    parser.add_argument("--output", default="results/scores.jsonl")
    parser.add_argument("--device", default="cuda", help="Device, for example cuda, cuda:1, or cpu")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument(
        "--factcc_model_path",
        default=None,
        help="Local FactCC checkpoint root; relative paths resolve from the project root",
    )
    parser.add_argument("--minicheck_chunk_size", type=int, default=None, help="MiniCheck source word chunk length; default 500")
    parser.add_argument(
        "--minicheck_model_path",
        default=None,
        help="Local MiniCheck model folder containing config, weights, and tokenizer files",
    )
    parser.add_argument("--alignscore_ckpt", default=None, help="Local AlignScore .ckpt file")
    parser.add_argument("--alignscore_backbone_path", default=None, help="Local RoBERTa config/tokenizer folder; no base-model weights are needed")
    parser.add_argument("--alignscore_mode", default="nli_sp", choices=["nli_sp", "nli", "bin_sp", "bin"])
    parser.add_argument("--qafacteval_model_path", default="./models", help="Local QAFactEval model folder from download_models.sh")
    parser.add_argument("--hf_cache_dir", default=None, help="Shared local Hugging Face cache for MiniCheck and AlignScore")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N records; 0 means all")
    parser.add_argument("--offline", action="store_true", help="Disable Hugging Face network downloads")
    parser.add_argument(
        "--nltk_data_dir",
        default=None,
        help="NLTK data root or its nested tokenizers/punkt_tab folder",
    )
    args = parser.parse_args()
    if args.summary_col and args.summary_cols:
        parser.error("Use either --summary_col or --summary_cols, not both")
    if args.batch_size < 1:
        parser.error("--batch_size must be positive")
    if args.minicheck_chunk_size is not None and args.minicheck_chunk_size < 1:
        parser.error("--minicheck_chunk_size must be positive")
    if args.limit is not None and args.limit < 0:
        parser.error("--limit cannot be negative")
    return args


def main() -> int:
    args = parse_args()
    if "factcc" in args.metrics:
        if not args.factcc_model_path:
            raise ValueError("FactCC requires --factcc_model_path pointing to a local checkpoint directory")
        factcc_dir = _project_path(args.factcc_model_path)
        if not factcc_dir.is_dir():
            raise FileNotFoundError(
                f"FactCC checkpoint directory not found: {factcc_dir}. Relative model paths are "
                f"resolved from the project root ({PROJECT_ROOT}); extract the checkpoint under "
                "models/factcc or pass its absolute directory."
            )
        has_config = (factcc_dir / "config.json").is_file()
        has_weights = any(
            (factcc_dir / name).is_file()
            for name in (
                "pytorch_model.bin",
                "model.safetensors",
                "pytorch_model.bin.index.json",
                "model.safetensors.index.json",
            )
        )
        has_tokenizer = any(
            (factcc_dir / name).is_file()
            for name in ("vocab.txt", "tokenizer.json", "tokenizer.model")
        )
        if not has_config or not has_weights or not has_tokenizer:
            raise FileNotFoundError(
                f"{factcc_dir} is not a complete FactCC checkpoint root. It must directly contain "
                "config.json, model weights (pytorch_model.bin or model.safetensors), and tokenizer "
                "files (usually vocab.txt). If extraction added a nested folder, point "
                "--factcc_model_path at that inner folder."
            )
    if args.nltk_data_dir:
        nltk_data_dir = _nltk_data_root(args.nltk_data_dir)
        if not nltk_data_dir.is_dir():
            raise FileNotFoundError(f"NLTK data directory not found: {nltk_data_dir}")
        import nltk

        nltk.data.path.insert(0, str(nltk_data_dir))
    if {"minicheck", "alignscore"}.intersection(args.metrics):
        import nltk

        try:
            nltk.sent_tokenize("NLTK sentence tokenizer preflight.", language="english")
        except LookupError as exc:
            expected = (
                str(nltk_data_dir / "tokenizers" / "punkt_tab" / "english")
                if args.nltk_data_dir
                else "<nltk_data_dir>/tokenizers/punkt_tab/english"
            )
            raise RuntimeError(
                "NLTK English punkt_tab data is missing. Expected English tokenizer "
                f"files under {expected}. Set --nltk_data_dir to its NLTK data root "
                "or the nested punkt_tab folder. The expected on-disk layout is "
                "tokenizers/punkt_tab/english; a punkt_tab folder placed directly "
                "in src/ must be moved under models/nltk_data/tokenizers/."
            ) from exc
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    if args.hf_cache_dir:
        cache_dir = str(Path(args.hf_cache_dir).expanduser().resolve())
        os.environ["HF_HUB_CACHE"] = cache_dir
        os.environ["HUGGINGFACE_HUB_CACHE"] = cache_dir

    records = load_records(args.data)
    if args.limit:
        records = records[: args.limit]
    summary_cols = args.summary_cols or [args.summary_col or "abstract_sum"]
    required_columns = [args.source_col, *summary_cols]
    if {"rouge", "bertscore"}.intersection(args.metrics):
        required_columns.append(args.reference_col)
    _validate_records(records, required_columns)
    results = [dict(record) for record in records]
    registry = _build_registry(args)
    errors: list[tuple[str, str, str]] = []

    for metric_name in args.metrics:
        logger.info("Running %s", metric_name)
        evaluator = registry[metric_name]()
        for summary_col in summary_cols:
            try:
                scored_records = evaluator.evaluate(
                    records,
                    source_col=args.source_col,
                    summary_col=summary_col,
                )
                if len(scored_records) != len(results):
                    raise RuntimeError("Evaluator returned an unexpected number of records")
                for target, scored in zip(results, scored_records):
                    for key, value in scored.items():
                        if key.startswith(f"{metric_name}_"):
                            target[f"{summary_col}__{key}"] = value
            except Exception as exc:  # noqa: BLE001
                logger.exception("Metric %s failed for summary column %s", metric_name, summary_col)
                errors.append((summary_col, metric_name, str(exc)))
                for target in results:
                    target[f"{summary_col}__{metric_name}_score"] = None
        del evaluator
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    save_results(results, args.output)
    save_summary(results, summary_cols, args.metrics, args.output)
    save_errors(errors, args.output)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
