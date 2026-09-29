# VDT-FaithfulTextsum

This repository evaluates factual consistency in summaries and includes an experimental Vietnamese SFT/GRPO research scaffold. It is not a validated production training recipe. Start by comparing a human summary and one or more model summaries against the same source document; see [GRPO_EXPERIMENT.md](GRPO_EXPERIMENT.md) before using automatic metrics as training rewards.

## Input data

Use JSONL with one record per document. Keep the original source and summaries in separate fields:

```json
{"id":"7","input":"source document text","human_sum":"human summary","llm_sum":"LLM summary"}
```

`input` is the grounding document. Source-based factuality metrics score each summary against it. ROUGE and BERTScore instead compare generated summaries to `human_sum`; the human summary is not assumed to be factually perfect.

## Local models and offline runs

Transfer model files to the company machine before running. The required layout depends on the metric:

| Metric | Local model files expected |
|---|---|
| FactCC | A local Transformers checkpoint under `models/factcc`, passed with `--factcc_model_path models/factcc` |
| MiniCheck | Complete extracted model folder `models/minicheck/`, passed with `--minicheck_model_path` |
| AlignScore | `models/alignscore/AlignScore-large.ckpt` passed with `--alignscore_ckpt`; a local RoBERTa config/tokenizer folder at `models/roberta-large/` passed with `--alignscore_backbone_path`; plus NLTK `punkt_tab` data |
| FENICE | `Babelscape/t5-base-summarization-claim-extractor` and `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` in the Hugging Face cache; upstream FENICE uses fixed model IDs and has no direct checkpoint-path option |
| QAFactEval | The directory tree produced by its upstream `download_models.sh`, passed with `--qafacteval_model_path` |
| BERTScore | A complete local encoder/tokenizer folder passed with `--bertscore_model_path`; for Vietnamese, start with `bert-base-multilingual-cased` and layer 9 |

ROUGE-1, ROUGE-2, and ROUGE-L are included as a small standard-library implementation. BERTScore uses the existing PyTorch/Transformers stack directly, so it does not require installing the separate `bert-score` package; it does require transferring the full encoder/tokenizer checkpoint. Its output is raw, unrescaled precision/recall/F1 with uniform token weights, and may differ slightly from the upstream `bert-score` package. Record the checkpoint and layer with results.

Keep locally extracted checkpoints in `models/` inside the repository (this directory is ignored by Git and is not included in the code ZIP). Preferred layout: `models/factcc/` for FactCC, `models/minicheck/` for MiniCheck, `models/alignscore/AlignScore-large.ckpt` for AlignScore, and `models/roberta-large/` for the RoBERTa tokenizer/config files AlignScore needs. The AlignScore checkpoint already contains trained RoBERTa weights, so the `roberta-large` folder does not need a second set of weights. Pass these local folders directly to the CLI; this avoids Hub cache-layout ambiguity. Alternatively, a complete Hugging Face cache can be placed at `models/hf-cache/` and passed with `--hf_cache_dir`. Both metrics use NLTK sentence splitting. If approved, put NLTK data outside `src/`, at `models/nltk_data/tokenizers/punkt_tab/english/...`. Pass `--nltk_data_dir models/nltk_data`—the parent of `tokenizers`. The bundle does not redistribute NLTK data because the [NLTK data license inventory](https://github.com/nltk/nltk_data/blob/gh-pages/DATASET-LICENSES.md) currently lists it without a declared license. `--offline` makes missing model files fail locally instead of downloading.

The included MiniCheck FLAN-T5 adapter and AlignScore adapter can run together. They no longer import their upstream Python packages and use the shared PyTorch/Transformers stack. These are inference-only ports and should be score-checked against upstream before treating results as interchangeable. NLTK and its approved `punkt_tab` resource are still needed for sentence splitting. ROUGE and BERTScore compare a candidate summary to `human_sum`, not to the input document; the human-reference row is left null to avoid a misleading self-match of 1.0. FENICE pins Transformers ~=4.38.2 while this project targets >=4.56.2, and its upstream repository is CC BY-NC-SA 4.0, so obtain company licensing/compliance approval before using it for company work. QAFactEval has a legacy dependency/model stack; use an IT-provisioned compatible environment if it conflicts with the main one, then combine outputs by record ID.

## Package layout and CLI

Reusable code lives in `src/evaluate/` and `src/training/`. From the repository root, launch evaluation with `python -m src.evaluate.run_eval`; no project installation or custom `PYTHONPATH` is needed. The `scripts/run_eval.py` file remains a convenience wrapper. Running from source does not require installing this project.

`VDT-FaithfulTextsum-repo-v3.zip` contains the complete project source, tests, sample data, documentation, and dependency manifests. It does not include Python, installed packages, model weights, NLTK data, or private/ignored datasets. Extract your separate model archives into `models/` under the extracted repository root. The optional SFT/GRPO code also requires an IT-provisioned training environment; see `GRPO_EXPERIMENT.md`. Do not install packages on the company machine unless IT approves it.

## Run a paired baseline

From the extracted repository root, with the approved Python environment active and model files already extracted under `models/`:

```text
python -m src.evaluate.run_eval --data /data/vdt/summaries.jsonl --summary_cols human_sum llm_sum --metrics factcc minicheck alignscore rouge --factcc_model_path models/factcc --minicheck_model_path models/minicheck --alignscore_ckpt models/alignscore/AlignScore-large.ckpt --alignscore_backbone_path models/roberta-large --nltk_data_dir models/nltk_data --offline --batch_size 2 --limit 10 --output results/smoke.jsonl
```

If you have only a standard Hugging Face cache rather than extracted model folders, omit `--minicheck_model_path` and `--alignscore_backbone_path`, then pass `--hf_cache_dir models/hf-cache`; the cache must contain complete snapshots for `lytang/MiniCheck-Flan-T5-Large` and `roberta-large`. The AlignScore path points to the `.ckpt` file itself. The command adds separate score fields; `llm_sum__rouge_score` is ROUGE-L F1. ROUGE needs no checkpoint. BERTScore is not included until its separate model checkpoint has been uploaded; to enable it, extract a complete multilingual BERT folder to `models/bert-base-multilingual-cased`, add `bertscore` to `--metrics`, and add `--bertscore_model_path models/bert-base-multilingual-cased`. The human summary has null ROUGE/BERTScore values because it is the reference. The `.summary.tsv` reports each summary-column/metric mean and valid count.

Inspect the small-slice results, then rerun the same command without `--limit 10` for the full dataset. The CLI accepts `--summary_col` for a single summary field; `--summary_cols` evaluates several fields in one run while loading each metric once.

## Reading the scores

The source-based factuality models were developed mainly for English. BERTScore can use a multilingual encoder, but that does not make its Vietnamese scores calibrated or factuality-specific. Treat Vietnamese scores as exploratory, compare systems only within the same metric/configuration, and manually label a sample for factual support before claiming improved faithfulness.

The source-based metrics measure factual support; ROUGE and BERTScore instead measure similarity to the human reference. None alone establish factual faithfulness, coverage, relevance, readability, or whether important details were omitted. These scores are exploratory on Vietnamese; validate them against human labels before drawing research conclusions.

## Experimental training

The optional [SFT + GRPO pilot](GRPO_EXPERIMENT.md) uses the human summary only as a separate training target/reward reference; it is never included in the generation prompt. It creates source-grouped train/validation/test splits, uses local-only model paths, and writes run manifests. The training stack is separate from the baseline evaluator dependencies; the company environment must be provisioned by IT. `requirements-grpo.txt` is a version target, not a ready-to-install lockfile. Do not run package installers on a restricted machine without IT approval.

## Metric references

- FactCC: Kryscinski et al., 2020, “Evaluating the Factual Consistency of Abstractive Text Summarization”
- FENICE: Scirè et al., 2024, [official repository](https://github.com/Babelscape/FENICE)
- MiniCheck: Tang et al., 2024, [official repository](https://github.com/Liyan06/MiniCheck)
- AlignScore: Zha et al., 2023, [official repository](https://github.com/yuh-zha/AlignScore)
- QAFactEval: Fabbri et al., 2022, [official repository](https://github.com/salesforce/QAFactEval)
- BERTScore: Zhang et al., 2020, [official implementation](https://github.com/Tiiiger/bert_score)
