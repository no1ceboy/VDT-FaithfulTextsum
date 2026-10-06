# VDT-FaithfulTextsum

This repository evaluates factual consistency in summaries and includes an experimental Vietnamese SFT/GRPO research scaffold. It is not a validated production training recipe. Start by comparing a human summary and one or more model summaries against the same source document; see [GRPO_EXPERIMENT.md](GRPO_EXPERIMENT.md) before using automatic metrics as training rewards.

## Input data

Use UTF-8 JSONL with one record per document. The canonical human-training format is:

```json
{"id":"7","text":"source document text","summary":"human summary"}
```

Here, `summary` is the human reference used to prepare SFT/GRPO data. The original file is read-only; preparation writes separate split files under the selected output directory. `text` is the grounding document. Source-based factuality metrics score a candidate summary against it. ROUGE and BERTScore compare a generated candidate to a separate human reference. `source` is also accepted as an alias. The older paired formats remain supported:

```json
{"id":"7","input":"source document text","human_sum":"human summary","llm_sum":"LLM summary"}
```

An `id/input/output` file is also accepted when `output` is the reference;
pass explicit columns when using it in a mixed comparison file. The evaluator
and preparer auto-detect `text`/`summary`, then `source`/`summary`, then the
legacy `input`/`human_sum` or `input`/`output` names. Use explicit
`--source_col`, `--summary_col`, or `--reference_col` when a file contains
several candidate columns.

### Clean a dataset without changing the original

For web-style Vietnamese documents, run the dependency-free cleaner before
training or generation. It writes a separate JSONL, replaces opaque URLs with
`[URL]`, removes decorative icons and separator lines, and drops rows with an
empty source or the Unicode replacement character (`�`). The reference summary
is preserved exactly by default. A JSON report and per-row audit log record
every changed or dropped row.

This also supports the legacy `id/input/output` layout. For example:

```text
python -m src.data.clean_dataset \
  --input data/batch_3.jsonl \
  --output data/batch_3_cleaned.jsonl \
  --source_col input \
  --reference_col output \
  --replacement_policy drop \
  --drop_stale_token_lengths \
  --report results/batch_3_cleaning.json \
  --audit_log results/batch_3_cleaning.audit.jsonl
```

The input file is never edited in place. Use `--clean_reference` only when
you intentionally want the same formatting cleanup applied to reference
summaries. `--drop_stale_token_lengths` removes a pre-cleaning
`qwen3_token_length` field instead of leaving misleading metadata.

## Local models and offline runs

Transfer model files to the company machine before running. The required layout depends on the metric:

| Metric | Local model files expected |
|---|---|
| FactCC | A local Transformers checkpoint under `models/factcc` |
| MiniCheck | Its Hugging Face snapshot in `models/hf-cache/` (or a directly extracted MiniCheck model folder there) |
| AlignScore | `models/alignscore/AlignScore-base.ckpt` plus matching RoBERTa-base weights/tokenizer/config, either in `models/roberta-base/` or as a complete snapshot in `models/hf-cache/`; plus NLTK `punkt_tab` data |
| mFACT | The complete Vietnamese `mFACT-vi_VN` Transformers folder under `models/mfact-vi_VN` |
| FENICE | `Babelscape/t5-base-summarization-claim-extractor` and `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` in the Hugging Face cache; upstream FENICE uses fixed model IDs and has no direct checkpoint-path option |
| QAFactEval | The directory tree produced by its upstream `download_models.sh`, passed with `--qafacteval_model_path` |
| BERTScore | A complete local encoder/tokenizer folder passed with `--bertscore_model_path`; for Vietnamese, start with `bert-base-multilingual-cased` and layer 9 |

mFACT is opt-in because it requires its own approximately 715 MB checkpoint. Enable it with `--metrics ... mfact --mfact_model_path models/mfact-vi_VN`; the model folder must directly contain `config.json`, model weights, and tokenizer files. It uses the existing PyTorch/Transformers dependencies and does not use the MiniCheck cache.

ROUGE-1, ROUGE-2, and ROUGE-L are included as a small standard-library implementation. BERTScore uses the existing PyTorch/Transformers stack directly, so it does not require installing the separate `bert-score` package; it does require transferring the full encoder/tokenizer checkpoint. Its output is raw, unrescaled precision/recall/F1 with uniform token weights, and may differ slightly from the upstream `bert-score` package. Record the checkpoint and layer with results.

Keep extracted assets in `models/` inside the repository (ignored by Git). Use `models/factcc/` for FactCC, `models/hf-cache/` for MiniCheck, and `models/alignscore/AlignScore-base.ckpt` for the AlignScore checkpoint. AlignScore needs the matching pretrained RoBERTa-base weights, tokenizer, and config as well. Put them in `models/roberta-base/`, or keep a complete `FacebookAI/roberta-base` snapshot in `models/hf-cache/`; the loader detects either arrangement so the same weights need not be copied twice. MiniCheck accepts either an intact Hub cache (do not flatten `models--...`, `snapshots`, `refs`, or `blobs`) or a directly extracted model folder. The cache loader can find a common enclosing `huggingface/hub` directory. Both metrics use NLTK sentence splitting. If approved, put NLTK data outside `src/`, at `models/nltk_data/tokenizers/punkt_tab/english/...`, and pass `--nltk_data_dir models/nltk_data`. The bundle does not redistribute NLTK data because the [NLTK data license inventory](https://github.com/nltk/nltk_data/blob/gh-pages/DATASET-LICENSES.md) currently lists it without a declared license. `--offline` prevents model downloads and makes missing assets fail locally.

The included MiniCheck FLAN-T5 adapter and AlignScore adapter can run together. They no longer import their upstream Python packages and use the shared PyTorch/Transformers stack. These are inference-only ports and should be score-checked against upstream before treating results as interchangeable. NLTK and its approved `punkt_tab` resource are still needed for sentence splitting. ROUGE and BERTScore compare a candidate summary to `human_sum`, not to the input document; the human-reference row is left null to avoid a misleading self-match of 1.0. FENICE pins Transformers ~=4.38.2 while this project targets >=4.56.2, and its upstream repository is CC BY-NC-SA 4.0, so obtain company licensing/compliance approval before using it for company work. QAFactEval has a legacy dependency/model stack; use an IT-provisioned compatible environment if it conflicts with the main one, then combine outputs by record ID.

## Package layout and CLI

Reusable code lives in `src/data/`, `src/evaluate/`, and `src/training/`. From the repository root, launch evaluation with `python -m src.evaluate.run_eval`; launch cleaning with `python -m src.data.clean_dataset`. No project installation or custom `PYTHONPATH` is needed. The `scripts/run_eval.py` file remains a convenience wrapper. Running from source does not require installing this project.

The public source repository is [github.com/no1ceboy/VDT-FaithfulTextsum](https://github.com/no1ceboy/VDT-FaithfulTextsum). Model weights, the Hugging Face cache, and NLTK data are not tracked; transfer those separately and keep the archive layout described above. The optional SFT/GRPO code also requires an IT-provisioned training environment; see `GRPO_EXPERIMENT.md`. Do not install packages on the company machine unless IT approves it.

## Run a paired baseline

From the extracted repository root, with the approved Python environment active and model files already extracted under `models/`:

For a runtime-only model check before scoring your data, the repository includes
[`tests/fixtures/metric_smoke.jsonl`](tests/fixtures/metric_smoke.jsonl), a tiny
English example. It exercises the default FactCC, MiniCheck, AlignScore, and
ROUGE path; it is not research data and must not be used to infer Vietnamese
metric quality:

```text
python -m src.evaluate.run_eval --data tests/fixtures/metric_smoke.jsonl --offline --nltk_data_dir models/nltk_data --batch_size 1 --limit 1 --output results/model-smoke.jsonl
```

The CLI defaults to CUDA for the company GPU. On a CPU-only machine, add
`--device cpu` (the MiniCheck smoke can be slow on CPU).

```text
python -m src.evaluate.run_eval --data /data/vdt/summaries.jsonl --offline --nltk_data_dir models/nltk_data --batch_size 2 --limit 10 --output results/smoke.jsonl
```

The default run selects `human_sum` and `llm_sum` from the paired schema, scores them with FactCC, MiniCheck, and AlignScore, and adds ROUGE when the human reference is present. For a canonical `source`/`summary` file, the default run scores `summary` with the source-based metrics; reference metrics need a separate generated candidate. FactCC, AlignScore, RoBERTa, and MiniCheck assets are resolved from their standard `models/` folders above. The command adds separate score fields; `llm_sum__rouge_score` is ROUGE-L F1. BERTScore is not included until its separate model checkpoint has been uploaded; to enable it, extract a complete multilingual BERT folder to `models/bert-base-multilingual-cased`, add `bertscore` to `--metrics`, and add `--bertscore_model_path models/bert-base-multilingual-cased`. The human summary has null ROUGE/BERTScore values because it is the reference. The `.summary.tsv` reports each summary-column/metric mean and valid count. Legacy datasets with an `abstract_sum` column remain supported.

For Vietnamese-specific faithfulness, add mFACT explicitly: `--metrics factcc minicheck alignscore mfact` and `--mfact_model_path models/mfact-vi_VN`. It returns `mfact_score` as the released classifier's class-1 faithful probability and `mfact_pred` at a 0.5 threshold. Keep this separate from the English-oriented metrics during analysis; it is still a silver-data research metric and needs human validation.

Inspect the small-slice results, then rerun the same command without `--limit 10` for the full dataset. The CLI accepts `--summary_col` for a single summary field; `--summary_cols` evaluates several fields in one run while loading each metric once. Create a dependency-free HTML visualization and JSON aggregate from scored JSONL with:

```text
python scripts/make_report.py --input results/baseline.jsonl --output results/baseline_report.html
```

The report includes mean/median/standard deviation, valid/missing counts, inline score bars, text-length checks, and the adjacent machine-readable JSON. Add `--run_dir outputs/grpo_lora_minicheck_001` to include the training manifest and recent training history. TensorBoard remains the detailed training visualization when the company environment provides it.

### Generate on Kaggle with Hugging Face

The default generator is offline and expects a local model folder, which is
the correct mode for the restricted company machine. On Kaggle, clone this
repository, attach the private dataset, and use a Hugging Face model ID with
`--online`. Store a Hugging Face read token in Kaggle Secrets as `HF_TOKEN`; do
not put the token in the notebook or command line. The Llama model also
requires accepting Meta's model access terms on Hugging Face.

```text
python scripts/generate_summaries.py \
  --model meta-llama/Llama-3.2-3B-Instruct \
  --online --hf_token_env HF_TOKEN \
  --input_jsonl /kaggle/input/vdt-faithfultextsum-batch3-cleaned/batch_3_cleaned.jsonl \
  --output /kaggle/working/batch_3_llama32_3b_summaries.jsonl \
  --summary_col llm_sum \
  --batch_size 1 --dtype float16
```

`--online` is opt-in. It leaves the company-machine path local-only and
selects FP16 automatically on GPUs that do not support BF16, such as many
Kaggle T4/P100 instances.

## Claim-level behavior audit

The scalar evaluators are useful for comparison, but they do not show which
claim failed. `src.evaluate.fact_audit` deterministically splits summaries into
claim-like units, retrieves likely source evidence with lexical overlap, and
optionally scores each full-source/claim pair with Vietnamese mFACT. It writes
one JSONL row per claim plus `.summary.tsv` and `.summary.json` aggregates.
Retrieved evidence is a review aid, not proof of entailment; `needs_review` is
not an automatic contradiction label.

```text
python -m src.evaluate.fact_audit \
  --data data/batch_3_cleaned.jsonl \
  --source_col input \
  --summary_col output \
  --mfact_model_path models/mfact-vi_VN \
  --device cpu --batch_size 1 --top_k 3 --limit 20 \
  --offline --output results/batch_3_human_claim_audit.jsonl
```

Use `--no_mfact` for extraction/retrieval-only debugging. Compare the same
documents across systems and manually label flagged claims before treating the
aggregate rates as evidence of a real Vietnamese faithfulness gap.

## Reading the scores

The source-based factuality models were developed mainly for English. mFACT-vi_VN is the Vietnamese-specific classifier in this repository, but it was trained from translated/silver faithfulness data rather than your company’s human labels. Treat all automatic scores as exploratory, compare systems only within the same metric/configuration, and manually label a sample for factual support before claiming improved faithfulness. BERTScore can use a multilingual encoder, but that does not make its Vietnamese scores calibrated or factuality-specific.

The source-based metrics measure factual support; ROUGE and BERTScore instead measure similarity to the human reference. None alone establish factual faithfulness, coverage, relevance, readability, or whether important details were omitted. These scores are exploratory on Vietnamese; validate them against human labels before drawing research conclusions.

## Experimental training

The optional [SFT + GRPO pilot](GRPO_EXPERIMENT.md) uses the human summary only as a separate training target/reward reference; it is never included in the generation prompt. The trainer accepts either prepared rows or raw `id/text/summary` rows and can create an internal source-grouped validation split without rewriting the input. It uses local-only model paths and writes run manifests. The training stack is separate from the baseline evaluator dependencies; the company environment must be provisioned by IT. `requirements-grpo.txt` is a version target, not a ready-to-install lockfile. Do not run package installers on a restricted machine without IT approval.

## Metric references

- FactCC: Kryscinski et al., 2020, “Evaluating the Factual Consistency of Abstractive Text Summarization”
- FENICE: Scirè et al., 2024, [official repository](https://github.com/Babelscape/FENICE)
- MiniCheck: Tang et al., 2024, [official repository](https://github.com/Liyan06/MiniCheck)
- AlignScore: Zha et al., 2023, [official repository](https://github.com/yuh-zha/AlignScore)
- mFACT: Qiu et al., 2023, [official repository](https://github.com/yfqiu-nlp/mfact-summ) and [Vietnamese checkpoint](https://huggingface.co/yfqiu-nlp/mFACT-vi_VN)
- QAFactEval: Fabbri et al., 2022, [official repository](https://github.com/salesforce/QAFactEval)
- BERTScore: Zhang et al., 2020, [official implementation](https://github.com/Tiiiger/bert_score)
