"""run_all.py — Main CLI for running faithfulness evaluators.

Usage examples
--------------
# Run all metrics (needs all dependencies installed):
python evaluate/run_all.py \\
    --data data/sample.jsonl \\
    --summary_col abstract_sum \\
    --metrics factcc fenice minicheck alignscore qafacteval \\
    --output results/scores.jsonl \\
    --device cuda

# Run only two metrics with offline model paths:
python evaluate/run_all.py \\
    --data data/my_dataset.jsonl \\
    --metrics minicheck alignscore \\
    --minicheck_model_path /mnt/models/Bespoke-MiniCheck-7B \\
    --alignscore_ckpt /mnt/models/AlignScore-large.ckpt \\
    --output results/scores.jsonl

# Evaluate an LLM-generated summary column:
python evaluate/run_all.py \\
    --data data/with_llm_sums.jsonl \\
    --summary_col llm_sum \\
    --metrics minicheck \\
    --output results/llm_scores.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Registry: maps CLI metric names → (EvaluatorClass, extra_kwargs_builder)
# ---------------------------------------------------------------------------

def _build_registry(args: argparse.Namespace) -> dict[str, Any]:
    """Lazily import evaluator classes and build the evaluator registry."""
    from evaluate.factcc_eval import FactCCEvaluator
    from evaluate.fenice_eval import FENICEEvaluator
    from evaluate.minicheck_eval import MiniCheckEvaluator
    from evaluate.alignscore_eval import AlignScoreEvaluator
    from evaluate.qafacteval_eval import QAFactEvalEvaluator

    registry = {
        "factcc": FactCCEvaluator(
            device=args.device,
            model_path=getattr(args, "factcc_model_path", None),
            batch_size=args.batch_size,
        ),
        "fenice": FENICEEvaluator(
            device=args.device,
            model_path=getattr(args, "fenice_model_path", None),
            batch_size=args.batch_size,
        ),
        "minicheck": MiniCheckEvaluator(
            device=args.device,
            model_path=getattr(args, "minicheck_model_path", None),
            batch_size=args.batch_size,
            model_name=getattr(args, "minicheck_model_name", "Bespoke-MiniCheck-7B"),
        ),
        "alignscore": AlignScoreEvaluator(
            device=args.device,
            model_path=getattr(args, "alignscore_ckpt", None),
            batch_size=args.batch_size,
            evaluation_mode=getattr(args, "alignscore_mode", "nli_sp"),
        ),
        "qafacteval": QAFactEvalEvaluator(
            device=args.device,
            model_path=getattr(args, "qafacteval_model_path", None),
            batch_size=args.batch_size,
            cuda_device=int(args.device.replace("cuda:", "").replace("cuda", "0"))
            if args.device.startswith("cuda") else -1,
        ),
    }
    return registry


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_records(path: str) -> list[dict[str, Any]]:
    """Load records from a JSONL or JSON file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    records: list[dict[str, Any]] = []
    with p.open(encoding="utf-8") as f:
        if p.suffix == ".json":
            data = json.load(f)
            records = data if isinstance(data, list) else [data]
        else:  # .jsonl or anything else → treat as JSONL
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    logger.info("Loaded %d records from %s", len(records), path)
    return records


def save_results(results: list[dict[str, Any]], output_path: str) -> None:
    """Write results as JSONL, creating parent dirs if needed."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for rec in results:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info("Saved %d results to %s", len(results), output_path)


def save_summary(results: list[dict[str, Any]], metrics: list[str], output_path: str) -> None:
    """Write a human-readable per-metric mean score summary (TSV)."""
    summary_path = Path(output_path).with_suffix(".summary.tsv")
    lines = ["metric\tmean_score\tcount\tn_nan"]
    for metric in metrics:
        col = f"{metric}_score"
        vals = [r[col] for r in results if col in r and r[col] == r[col]]  # exclude NaN
        n_nan = sum(1 for r in results if col in r and r[col] != r[col])
        if vals:
            mean = sum(vals) / len(vals)
            lines.append(f"{metric}\t{mean:.6f}\t{len(vals)}\t{n_nan}")
        else:
            lines.append(f"{metric}\tN/A\t0\t{len(results)}")
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Summary written to %s", summary_path)
    print("\n" + "\n".join(lines))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

AVAILABLE_METRICS = ["factcc", "fenice", "minicheck", "alignscore", "qafacteval"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run faithfulness metric evaluators on a Vietnamese summarization dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # --- Data ---
    p.add_argument(
        "--data",
        required=True,
        help="Path to input dataset (.jsonl or .json).",
    )
    p.add_argument(
        "--source_col",
        default="input",
        help="Record field containing the source document. (default: input)",
    )
    p.add_argument(
        "--summary_col",
        default="abstract_sum",
        help="Record field containing the summary to evaluate. "
             "(default: abstract_sum; use 'llm_sum' for LLM outputs, etc.)",
    )

    # --- Metrics ---
    p.add_argument(
        "--metrics",
        nargs="+",
        choices=AVAILABLE_METRICS,
        default=AVAILABLE_METRICS,
        metavar="METRIC",
        help=f"Metrics to run. Available: {AVAILABLE_METRICS}. "
             "Defaults to all metrics.",
    )

    # --- Output ---
    p.add_argument(
        "--output",
        default="results/scores.jsonl",
        help="Output JSONL path. A .summary.tsv file will also be written. "
             "(default: results/scores.jsonl)",
    )

    # --- Runtime ---
    p.add_argument(
        "--device",
        default="cuda",
        help="PyTorch device string: 'cuda', 'cuda:1', 'cpu', etc. (default: cuda)",
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Default batch size (individual metrics may override). (default: 8)",
    )

    # --- Per-metric model path overrides (for offline use) ---
    p.add_argument(
        "--factcc_model_path",
        default=None,
        help="Local path or HF repo id for FactCC (default: manueldeprada/FactCC).",
    )
    p.add_argument(
        "--fenice_model_path",
        default=None,
        help="Local path or HF repo id for FENICE NLI model (default: roberta-large-mnli).",
    )
    p.add_argument(
        "--minicheck_model_path",
        default=None,
        help="Local directory for MiniCheck model (overrides --minicheck_model_name).",
    )
    p.add_argument(
        "--minicheck_model_name",
        default="Bespoke-MiniCheck-7B",
        help="MiniCheck model name (default: Bespoke-MiniCheck-7B). "
             "Ignored if --minicheck_model_path is set.",
    )
    p.add_argument(
        "--alignscore_ckpt",
        default=None,
        help="Local path to AlignScore .ckpt file (e.g. /mnt/models/AlignScore-large.ckpt).",
    )
    p.add_argument(
        "--alignscore_mode",
        default="nli_sp",
        choices=["nli_sp", "nli", "bin_sp", "bin"],
        help="AlignScore evaluation mode. (default: nli_sp)",
    )
    p.add_argument(
        "--qafacteval_model_path",
        default=None,
        help="Directory containing QAFactEval pretrained models (default: ./models).",
    )

    # --- Misc ---
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only evaluate the first N records (useful for smoke tests).",
    )
    p.add_argument(
        "--offline",
        action="store_true",
        help="Set HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1 before loading models.",
    )

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        logger.info("Offline mode enabled (HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1).")

    # Load dataset
    records = load_records(args.data)
    if args.limit:
        records = records[: args.limit]
        logger.info("Limiting to first %d records.", args.limit)

    # Validate columns exist
    sample = records[0] if records else {}
    if args.source_col not in sample:
        logger.error("source_col '%s' not found in records. Available keys: %s",
                     args.source_col, list(sample.keys()))
        sys.exit(1)
    if args.summary_col not in sample:
        logger.error("summary_col '%s' not found in records. Available keys: %s",
                     args.summary_col, list(sample.keys()))
        sys.exit(1)

    # Build evaluators
    registry = _build_registry(args)

    # Run each requested metric sequentially (to avoid OOM from loading all at once)
    current_records = records
    for metric_name in args.metrics:
        evaluator = registry[metric_name]
        logger.info("=" * 60)
        logger.info("Running metric: %s", metric_name.upper())
        logger.info("=" * 60)
        try:
            current_records = evaluator.evaluate(
                current_records,
                source_col=args.source_col,
                summary_col=args.summary_col,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Metric '%s' failed entirely: %s", metric_name, exc)
            logger.error("Skipping %s and continuing.", metric_name)
            # Add NaN scores so output schema is consistent
            col = f"{metric_name}_score"
            current_records = [
                {**r, col: float("nan")} for r in current_records
            ]

    # Save outputs
    save_results(current_records, args.output)
    save_summary(current_records, args.metrics, args.output)


if __name__ == "__main__":
    main()
