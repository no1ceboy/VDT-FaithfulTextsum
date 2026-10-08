"""Run factuality metrics on one or more summary columns.

Example:
    python -m src.evaluate.run_eval --data /data/vdt/summaries.jsonl \
        --summary_cols human_sum llm_sum --metrics factcc minicheck alignscore rouge \
        --factcc_model_path models/factcc \
        --alignscore_ckpt models/alignscore/AlignScore-base.ckpt \
        --alignscore_backbone_path models/roberta-base \
        --hf_cache_dir models/hf-cache \
        --nltk_data_dir models/nltk_data --offline \
        --batch_size 2 --limit 10 --output results/smoke.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
from collections import deque
from functools import partial
from pathlib import Path
from typing import Any

from ._local_model_paths import find_model_dir

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_SUMMARY_COLUMNS = ("human_sum", "llm_sum")


def _project_path(value: str) -> Path:
    """Resolve relative model paths from the repository root, not the shell cwd."""
    path = Path(value).expanduser()
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def _hf_cache_root(value: str) -> Path:
    """Find the actual HF Hub cache root, tolerating an enclosing archive folder."""
    requested = _project_path(value)
    if not requested.is_dir():
        raise FileNotFoundError(f"Hugging Face cache directory not found: {requested}")

    queue = deque([(requested, 0)])
    visited: set[Path] = set()
    while queue:
        candidate, depth = queue.popleft()
        if candidate in visited:
            continue
        visited.add(candidate)
        try:
            children = [child for child in candidate.iterdir() if child.is_dir()]
        except OSError:
            continue
        if any(child.name.startswith("models--") for child in children):
            if candidate != requested:
                logger.info("Using nested Hugging Face cache root: %s", candidate)
            return candidate
        if depth < 3:
            queue.extend(
                (child, depth + 1)
                for child in children
                if child.name not in {".locks", "blobs", "refs", "snapshots"}
            )
    return requested


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


def _validate_roberta_base_folder(path: Path) -> None:
    """Require local config, pretrained weights, and tokenizer before loading."""
    missing: list[str] = []
    if not (path / "config.json").is_file():
        missing.append("config.json")
    if not any(
        (path / name).is_file()
        for name in (
            "model.safetensors",
            "pytorch_model.bin",
            "model.safetensors.index.json",
            "pytorch_model.bin.index.json",
        )
    ):
        missing.append("model.safetensors or pytorch_model.bin (or a shard index)")
    if not (
        (path / "tokenizer.json").is_file()
        or ((path / "vocab.json").is_file() and (path / "merges.txt").is_file())
    ):
        missing.append("tokenizer.json or vocab.json plus merges.txt")
    if missing:
        raise FileNotFoundError(
            f"{path} is not a complete local roberta-base folder; missing: "
            + ", ".join(missing)
            + ". Extract the full model folder there, or set --alignscore_backbone_path."
        )


def _cached_roberta_base(cache_dir: str | Path) -> Path | None:
    """Find a complete roberta-base snapshot inside a transferred Hub cache."""
    return _cached_roberta_variant(cache_dir, "base")


def _complete_roberta_folder(path: Path, variant: str) -> bool:
    """Validate a RoBERTa folder and, when possible, its base/large size."""
    try:
        _validate_roberta_base_folder(path)
    except FileNotFoundError:
        return False
    try:
        config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return True
    model_type = str(config.get("model_type", "")).lower()
    if model_type and model_type != "roberta":
        return False
    hidden_size = config.get("hidden_size")
    expected_hidden_size = {"base": 768, "large": 1024}[variant]
    return hidden_size is None or hidden_size == expected_hidden_size


def _cached_roberta_variant(cache_dir: str | Path, variant: str) -> Path | None:
    """Find a complete base/large RoBERTa snapshot in a transferred cache."""
    try:
        cache_root = _hf_cache_root(str(cache_dir))
    except FileNotFoundError:
        return None
    entry_names = (
        ("models--FacebookAI--roberta-base", "models--roberta-base")
        if variant == "base"
        else ("models--FacebookAI--roberta-large", "models--roberta-large")
    )
    preferred_tokens = (
        ("roberta-base", "facebookai--roberta-base")
        if variant == "base"
        else ("roberta-large", "facebookai--roberta-large")
    )
    predicate = lambda path: _complete_roberta_folder(path, variant)
    for entry_name in entry_names:
        entry = cache_root / entry_name
        candidate = find_model_dir(
            entry,
            is_complete=predicate,
            preferred_tokens=preferred_tokens,
        )
        if candidate is not None:
            return candidate
    return find_model_dir(
        cache_root,
        is_complete=predicate,
        preferred_tokens=preferred_tokens,
    )


def _complete_roberta_base_folder(path: Path) -> bool:
    """Predicate used while searching wrapped RoBERTa cache uploads."""
    return _complete_roberta_folder(path, "base")


def _resolve_roberta_variant(
    backbone_path: str,
    cache_dir: str | None,
    variant: str,
) -> Path:
    """Resolve the requested RoBERTa-base or RoBERTa-large folder."""
    requested = _project_path(backbone_path)
    predicate = lambda path: _complete_roberta_folder(path, variant)
    preferred_tokens = (f"roberta-{variant}", f"facebookai--roberta-{variant}")
    if requested.is_dir():
        try:
            _validate_roberta_base_folder(requested)
            return requested
        except FileNotFoundError:
            nested = find_model_dir(
                requested,
                is_complete=predicate,
                preferred_tokens=preferred_tokens,
            )
            if nested is not None:
                logger.info("Using nested local RoBERTa-%s folder: %s", variant, nested)
                return nested
    if cache_dir:
        cached = _cached_roberta_variant(_project_path(cache_dir), variant)
        if cached is not None:
            logger.info("Using roberta-%s from local Hugging Face snapshot: %s", variant, cached)
            return cached
    raise FileNotFoundError(
        f"Complete roberta-{variant} assets were not found at {requested} or in the local "
        f"Hugging Face cache {cache_dir!r}. The model folder must include config, "
        "pretrained weights, and tokenizer files."
    )


def _resolve_roberta_base(backbone_path: str, cache_dir: str | None) -> Path:
    """Prefer the standard extracted folder, then a matching local Hub snapshot."""
    return _resolve_roberta_variant(backbone_path, cache_dir, "base")


def _resolve_alignscore_backbone(
    backbone_path: str,
    cache_dir: str | None,
    checkpoint_path: str,
) -> Path:
    """Resolve the RoBERTa size matching an AlignScore checkpoint filename."""
    variant = "large" if "large" in Path(checkpoint_path).name.lower() else "base"
    return _resolve_roberta_variant(backbone_path, cache_dir, variant)


def _summary_columns(
    records: list[dict[str, Any]],
    single_column: str | None = None,
    multiple_columns: list[str] | None = None,
) -> list[str]:
    """Infer candidate summary fields across canonical and legacy schemas."""
    if multiple_columns:
        return multiple_columns
    if single_column:
        return [single_column]
    if all(all(column in record for record in records) for column in _DEFAULT_SUMMARY_COLUMNS):
        return list(_DEFAULT_SUMMARY_COLUMNS)
    if all("llm_sum" in record for record in records):
        return ["llm_sum"]
    if all("human_sum" in record and "summary" in record for record in records):
        return ["human_sum", "summary"]
    if all("human_sum" in record for record in records):
        return ["human_sum"]
    if all("summary" in record for record in records):
        return ["summary"]
    if all("abstract_sum" in record for record in records):
        return ["abstract_sum"]
    if all("output" in record for record in records):
        return ["output"]
    raise ValueError("Could not infer summary fields. Pass --summary_col or --summary_cols.")


def _source_column(records: list[dict[str, Any]], requested: str | None = None) -> str:
    """Infer the source field, preferring canonical ``text`` then ``source``."""
    if requested:
        return requested
    if all("text" in record for record in records):
        return "text"
    if all("source" in record for record in records):
        return "source"
    if all("input" in record for record in records):
        return "input"
    raise ValueError("Could not infer the source field. Pass --source_col.")


def _reference_column(records: list[dict[str, Any]], requested: str | None = None) -> str:
    """Infer the reference field used by ROUGE/BERTScore.

    A canonical ``summary``-only file has no separate model candidate, so the
    field is treated as its own reference and reference metrics return null for
    that same field.  A generated comparison file should include a separate
    ``human_sum`` (or explicit ``--reference_col``) field.
    """
    if requested:
        return requested
    for candidate in ("human_sum", "reference", "abstract_sum", "summary", "output"):
        if all(candidate in record for record in records):
            return candidate
    return "human_sum"


def _default_metrics(
    records: list[dict[str, Any]], reference_col: str, summary_cols: list[str]
) -> list[str]:
    """Pick bundled core metrics; add ROUGE only when a reference is present."""
    metrics = ["factcc", "minicheck", "alignscore"]
    if all(reference_col in record for record in records) and any(
        column != reference_col for column in summary_cols
    ):
        metrics.append("rouge")
    return metrics


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
            cache_dir=str(_project_path(args.hf_cache_dir)) if args.hf_cache_dir else None,
            batch_size=args.batch_size,
            chunk_size=args.minicheck_chunk_size,
        )
    if "alignscore" in args.metrics:
        from .alignscore_eval import AlignScoreEvaluator

        registry["alignscore"] = partial(
            AlignScoreEvaluator,
            device=device,
            model_path=(str(_project_path(args.alignscore_ckpt)) if args.alignscore_ckpt else None),
            backbone_path=(
                str(_project_path(args.alignscore_backbone_path))
                if args.alignscore_backbone_path
                else None
            ),
            batch_size=args.batch_size,
            evaluation_mode=args.alignscore_mode,
        )
    if "mfact" in args.metrics:
        from .mfact_eval import MFactEvaluator

        registry["mfact"] = partial(
            MFactEvaluator,
            device=device,
            model_path=str(_project_path(args.mfact_model_path))
            if args.mfact_model_path
            else None,
            batch_size=args.batch_size,
        )
    if "qafacteval" in args.metrics:
        from .qafacteval_eval import QAFactEvalEvaluator

        registry["qafacteval"] = partial(
            QAFactEvalEvaluator,
            device=device,
            model_path=str(_project_path(args.qafacteval_model_path)),
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
            model_path=(
                str(_project_path(args.bertscore_model_path))
                if args.bertscore_model_path
                else None
            ),
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
    parser.add_argument(
        "--source_col",
        default=None,
        help="Source document field; auto-detects source, then legacy input",
    )
    parser.add_argument("--summary_col", default=None, help="One summary field (legacy/single-column mode)")
    parser.add_argument("--summary_cols", nargs="+", default=None, help="Summary fields to score in one run")
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=[
            "factcc",
            "fenice",
            "minicheck",
            "alignscore",
            "mfact",
            "qafacteval",
            "rouge",
            "bertscore",
        ],
        default=None,
    )
    parser.add_argument(
        "--reference_col",
        default=None,
        help="Human/reference summary field for ROUGE and BERTScore; auto-detected when omitted",
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
        default="models/factcc",
        help="Local FactCC checkpoint root; relative paths resolve from the project root",
    )
    parser.add_argument("--minicheck_chunk_size", type=int, default=None, help="MiniCheck source word chunk length; default 500")
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
    parser.add_argument("--alignscore_mode", default="nli_sp", choices=["nli_sp", "nli", "bin_sp", "bin"])
    parser.add_argument(
        "--mfact_model_path",
        default="models/mfact-vi_VN",
        help="Local mFACT-vi_VN model directory; relative paths resolve from the project root",
    )
    parser.add_argument("--qafacteval_model_path", default="./models", help="Local QAFactEval model folder from download_models.sh")
    parser.add_argument(
        "--hf_cache_dir",
        default="models/hf-cache",
        help="Local Hugging Face cache or extracted MiniCheck model folder",
    )
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
    args.data = str(_project_path(args.data))
    records = load_records(args.data)
    if args.limit:
        records = records[: args.limit]

    args.source_col = _source_column(records, args.source_col)
    args.reference_col = _reference_column(records, args.reference_col)
    summary_cols = _summary_columns(records, args.summary_col, args.summary_cols)

    if args.metrics is None:
        args.metrics = _default_metrics(records, args.reference_col, summary_cols)

    if "factcc" in args.metrics:
        factcc_dir = _project_path(args.factcc_model_path)
        if not factcc_dir.is_dir():
            raise FileNotFoundError(
                f"FactCC checkpoint directory not found: {factcc_dir}. Extract the checkpoint "
                "under models/factcc or pass --factcc_model_path."
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
                "config.json, model weights, and tokenizer files (usually vocab.txt). If extraction "
                "added a nested folder, point --factcc_model_path at that inner folder."
            )
        args.factcc_model_path = str(factcc_dir)

    if "alignscore" in args.metrics:
        checkpoint_path = _project_path(args.alignscore_ckpt)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"AlignScore checkpoint not found: {checkpoint_path}. Expected AlignScore-base.ckpt; "
                "pass --alignscore_ckpt if it is elsewhere."
            )
        args.alignscore_ckpt = str(checkpoint_path)
        backbone_path = _resolve_alignscore_backbone(
            args.alignscore_backbone_path,
            args.hf_cache_dir,
            args.alignscore_ckpt,
        )
        args.alignscore_backbone_path = str(backbone_path)

    if {"minicheck", "fenice"}.intersection(args.metrics):
        if not args.hf_cache_dir:
            raise ValueError(
                "MiniCheck and FENICE require --hf_cache_dir with their local model/cache"
            )
        args.hf_cache_dir = str(_hf_cache_root(args.hf_cache_dir))
        os.environ["HF_HUB_CACHE"] = args.hf_cache_dir
        os.environ["HUGGINGFACE_HUB_CACHE"] = args.hf_cache_dir
        os.environ["TRANSFORMERS_CACHE"] = args.hf_cache_dir

    if "mfact" in args.metrics:
        mfact_dir = _project_path(args.mfact_model_path)
        if not mfact_dir.is_dir():
            raise FileNotFoundError(
                f"mFACT model directory not found: {mfact_dir}. Extract mFACT-vi_VN "
                "under models/mfact-vi_VN or pass --mfact_model_path."
            )
        has_config = (mfact_dir / "config.json").is_file()
        has_weights = any(
            (mfact_dir / name).is_file()
            for name in (
                "pytorch_model.bin",
                "model.safetensors",
                "pytorch_model.bin.index.json",
                "model.safetensors.index.json",
            )
        )
        has_tokenizer = any(
            (mfact_dir / name).is_file()
            for name in ("tokenizer.json", "vocab.txt", "tokenizer.model")
        )
        if not has_config or not has_weights or not has_tokenizer:
            raise FileNotFoundError(
                f"{mfact_dir} is not a complete mFACT model root. It must directly contain "
                "config.json, model weights, and tokenizer files. If extraction added a "
                "nested folder, point --mfact_model_path at that inner folder."
            )
        args.mfact_model_path = str(mfact_dir)

    if args.nltk_data_dir:
        nltk_data_dir = _nltk_data_root(str(_project_path(args.nltk_data_dir)))
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

    args.output = str(_project_path(args.output))
    save_results(results, args.output)
    save_summary(results, summary_cols, args.metrics, args.output)
    save_errors(errors, args.output)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
