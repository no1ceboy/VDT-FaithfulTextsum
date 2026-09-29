# Vietnamese faithful summarization: SFT + GRPO pilot

This is an experimental research scaffold, not a validated production training recipe. It uses the human summary as a reference signal, then optionally adds one existing factuality evaluator as a black-box reward. The reference is never placed in the model prompt. Generated summaries are still judged against the source, not against the reference alone.

## Research question and claims

The experiment asks whether a local instruction model, adapted with Vietnamese source/summary examples and an explicitly selected faithfulness reward, produces summaries with better factual support on held-out documents. It does not assume that a human summary is factually perfect or that an automatic metric is ground truth.

Report at least these comparisons on the same untouched test set:

1. The base instruction model.
2. The optional SFT warm-start, if run.
3. The GRPO adapter.
4. The human reference as a descriptive ceiling/baseline, not as perfect truth.

Report each metric separately, output length and coverage checks, and a blinded human review of source-supported claims. Do not claim improved Vietnamese faithfulness from a metric-score increase alone. The included FactCC, MiniCheck and AlignScore adapters are not validated or calibrated for Vietnamese; the metric reward is therefore an experimental treatment that needs human validation.

## Model and company-machine constraints

The suggested first *candidate* generator is `meta-llama/Llama-3.2-3B-Instruct`, transferred as a local checkpoint directory and passed by path. Vietnamese is not among the [model card's eight officially supported languages](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct), and its out-of-scope section cautions against use in unsupported languages; another section says developers may fine-tune beyond those languages under the license and acceptable-use policy. Resolve this with company legal/research approval before treating Llama as an acceptable Vietnamese candidate. I would compare it with a Vietnamese-centered checkpoint under the same split before choosing a final model. The scripts force Hugging Face/Transformers offline mode and require local model paths; they do not download a model or install packages.

`requirements-grpo.txt` targets TRL 0.29.0 and its PEFT extra, but is intentionally not presented as a complete lockfile or as compatible with every company CUDA image. The APIs follow the [official TRL 0.29.0 GRPO](https://huggingface.co/docs/trl/v0.29.0/grpo_trainer) and [SFT](https://huggingface.co/docs/trl/v0.29.0/sft_trainer) interfaces. Ask IT to provision and freeze a compatible Python/PyTorch/CUDA/Transformers/TRL/PEFT/Datasets environment. The training scripts have not been executed here because this workspace does not have TRL, PEFT, or Accelerate installed and this task must not install them. The package target was chosen to make the expected API explicit; verify the provisioned environment with `trl env` and a short approved smoke run.

The initial settings are conservative for a busy GPU: one sequence per device, four sampled completions per prompt, gradient accumulation 4, BF16, LoRA on attention projections, metric models on CPU, no vLLM, and no separate KL reference model (`beta=0`). Four is the number of sampled outputs per prompt, not a required number of distinct training documents. A B200 with 80–100 GB free is ample for a 3B LoRA pilot in principle, but available memory, source length, scorer placement and other GPU users still determine whether a run fits. Start with an approved pilot slice and watch actual memory. The CPU metric scorer may become the speed bottleneck; move it to a GPU only after checking memory headroom.

## Data contract and leakage controls

Input is UTF-8 JSONL, one object per line. By default the source is `input`, the human reference is `abstract_sum`, and `id` is used for traceability. `style` is optional. The preparer validates text and IDs, creates a Vietnamese chat prompt, and stores the reference in a separate `reference` field for reward computation. Prompt construction and the trainer both keep `reference` out of the prompt. The split groups normalized exact duplicate source documents together so they cannot cross train/validation/test.

```json
{"id":"7","style":"daily","input":"văn bản nguồn...","abstract_sum":"tóm tắt do con người viết..."}
```

Use enough distinct documents for all three splits. A one-record sample can only smoke-test parsing with both holdout fractions set to zero; it is not a training or evaluation dataset. The deterministic split is not stratified. For a research study, inspect the split counts and balance domains/styles manually before training.

## Workflow

Run commands from the repository root. Paths below are examples; on the company machine use the already provisioned environment and local paths. None of these commands installs software.

### 1. Prepare data

```bash
python scripts/prepare_grpo_data.py \
  --input /data/vdt/summaries.jsonl \
  --output_dir /data/vdt/grpo_splits \
  --source_col input --reference_col abstract_sum \
  --validation_fraction 0.1 --test_fraction 0.1 --seed 42
```

Review `data_manifest.json`, row counts, duplicate-source grouping, and several examples in `train.jsonl`, `validation.jsonl`, and `test.jsonl`. The `--overwrite` switch replaces these named outputs; otherwise the script refuses to overwrite them.

For a parser-only smoke test on the included single-record sample, use `--validation_fraction 0 --test_fraction 0` and an output directory outside the project. Do not train or report evaluation from that one record. As a rough pilot heuristic, treat fewer than 100 training rows as a pipeline/sensitivity exercise rather than evidence of generalization; source diversity matters more than the raw count.

### 2. Optional SFT warm-start

SFT is a useful control and can teach the model the target summary style before reward optimization. It computes loss on the human completion only. `validation.jsonl` is used for reference negative log-likelihood, not as a factuality measure.

```bash
python scripts/train_sft.py \
  --model models/Llama-3.2-3B-Instruct \
  --train_jsonl /data/vdt/grpo_splits/train.jsonl \
  --eval_jsonl /data/vdt/grpo_splits/validation.jsonl \
  --output_dir models/runs/sft_001
```

The resulting adapter is `.../sft_001/final_adapter`. You may also skip SFT and start GRPO directly from the instruct checkpoint. Do not use the held-out test split for SFT, hyperparameter selection, or reward tuning.

### 3. Run GRPO with one selected metric

Start with one metric. `--faithfulness_metrics` is required so metric choice is explicit. The reference-overlap reward is included at weight 0.25 by default; the selected faithfulness metric has weight 1.0. TRL sums these weighted components without reward-scale normalization (`scale_rewards=none`), so the weights are part of the scientific treatment and must be recorded. `reference_weight=0` disables the overlap contribution. Combining several metrics requires the explicit `--allow_multiple_metrics` acknowledgement and is not recommended for the first comparison.

MiniCheck example (using the already transferred cache):

```bash
python scripts/train_grpo.py \
  --model models/runs/sft_001/final_adapter \
  --base_model models/Llama-3.2-3B-Instruct \
  --train_jsonl /data/vdt/grpo_splits/train.jsonl \
  --output_dir models/runs/grpo_minicheck_001 \
  --faithfulness_metrics minicheck \
  --hf_cache_dir models/hf-cache \
  --reward_device cpu --seed 42
```

For a base-model start, pass the base checkpoint as `--model` and omit `--base_model`. For FactCC, select `factcc` and provide `--factcc_model_path`. For AlignScore, select `alignscore`, provide `--alignscore_ckpt`, and pass `--hf_cache_dir` pointing to the cache containing `roberta-large`. Keep those checkpoint/cache folders in the same local layout used by the existing evaluation commands.

The script fails early for missing local paths, an empty training set, prompts longer than the configured cap, context overflow, or a GPU without BF16. It does not crop source text silently. The GRPO run writes `run_manifest.json`, Trainer checkpoints/logs, and a final LoRA adapter. The manifest records data hash, key hyperparameters, metric/reward weights and runtime choices. The generation script also writes a sidecar manifest with input/output hashes and decoding settings. Treat run directories and generated JSONL as sensitive: they contain source text and summaries. Completion logging is disabled by default.

### 4. Generate held-out summaries and score them

Generate validation summaries first while choosing settings. Freeze choices before generating the test set; use test results once for the final report. The generator uses greedy decoding by default (same input, deterministic output), with a configurable maximum output length. The generated JSONL contains `input`, `human_sum`, and one chosen model column, so the existing multi-metric evaluation CLI can score the human summary and model output against the same source. You can generate multiple systems on identical rows and merge by ID with `--existing_jsonl`.

```bash
python scripts/generate_summaries.py \
  --model models/Llama-3.2-3B-Instruct \
  --input_jsonl /data/vdt/grpo_splits/test.jsonl \
  --output /data/vdt/results/grpo_test.jsonl \
  --summary_col base_sum

python scripts/generate_summaries.py \
  --model models/runs/grpo_minicheck_001/final_adapter \
  --base_model models/Llama-3.2-3B-Instruct \
  --input_jsonl /data/vdt/grpo_splits/test.jsonl \
  --existing_jsonl /data/vdt/results/grpo_test.jsonl \
  --output /data/vdt/results/grpo_test_compare.jsonl \
  --summary_col grpo_sum

python -m src.evaluate.run_eval \
  --data /data/vdt/results/grpo_test_compare.jsonl \
  --summary_cols human_sum base_sum grpo_sum \
  --metrics factcc minicheck alignscore \
  --factcc_model_path models/factcc \
  --alignscore_ckpt models/alignscore/AlignScore-large.ckpt \
  --hf_cache_dir models/hf-cache \
  --nltk_data_dir models/nltk_data \
  --offline --batch_size 2 \
  --output /data/vdt/results/grpo_test_scored.jsonl
```

To compare base, SFT, and GRPO fairly, generate each from the exact same validation/test rows using the same decoding settings and preserve each output column. The included generator can merge outputs by `id` after confirming source/reference equality. Do not accidentally score a training row as a test example.

## Reward definitions and limitations

- `reference_char_reward`: whitespace-insensitive, Unicode NFC-normalized character n-gram F-beta (orders 1–6, beta 2). It is an explicitly named chrF-style proxy, not the canonical SacreBLEU chrF metric. It rewards surface overlap, can discourage valid paraphrases, and can reward copying while failing to detect unsupported claims.
- FactCC, MiniCheck, or AlignScore: existing project evaluator adapter, called on `(source, generated summary)`. Non-finite scores become zero and values are clipped to `[0, 1]` before entering GRPO. Clipping is only a common numeric bound; it does not calibrate or make the different metrics comparable.
- The human reference is neither a factual oracle nor a complete set of valid summaries. A single reference encourages its wording and content selection. Use several references if available, report human review, and include coverage/omission analysis alongside factual support.
- Reward models are not differentiable through their text scores; GRPO uses sampled output rewards. Metric quality, prompt length, sampling temperature, and number of generations all affect the optimization signal.
- B200 GPU capacity does not fix metric language bias. Validate the selected metric against Vietnamese expert labels before treating it as a reward suitable for a claim about factuality.

## Reproducibility checklist

Archive the input hash, split manifest, exact local model and adapter hashes, model/license approval, Python/PyTorch/CUDA/Transformers/TRL/PEFT/Datasets versions, GPU model and free-memory observation, seeds, prompts, reward implementations and weights, generation settings, all run manifests, per-example metric outputs, and the human annotation protocol. Keep raw data and run artifacts inside approved company storage. Avoid uploading company text or checkpoints to public trackers or model hubs.
