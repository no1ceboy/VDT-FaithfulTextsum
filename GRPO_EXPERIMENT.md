# Vietnamese faithful summarization: GRPO pilot

This is an experimental research scaffold, not a validated production training recipe. It uses the human summary as a reference signal, then optionally adds one existing factuality evaluator as a black-box reward. The reference is never placed in the model prompt. Generated summaries are still judged against the source, not against the reference alone.

## Research question and claims

The experiment asks whether a local instruction model, adapted with Vietnamese source/summary examples and an explicitly selected faithfulness reward, produces summaries with better factual support on held-out documents. It does not assume that a human summary is factually perfect or that an automatic metric is ground truth.

Report at least these comparisons on the same untouched test set:

1. The base instruction model.
2. GRPO with LoRA (recommended first training treatment).
3. Optional QLoRA and full fine-tuning (FFT) ablations.
4. Optional SFT warm-start, if run.
5. The human reference as a descriptive baseline, not as perfect truth.

Report each metric separately, output length and coverage checks, and a blinded human review of source-supported claims. Do not claim improved Vietnamese faithfulness from a metric-score increase alone. The included FactCC, MiniCheck and AlignScore adapters are not validated or calibrated for Vietnamese; the metric reward is therefore an experimental treatment that needs human validation.

## Model and company-machine constraints

The suggested first *candidate* generator is `meta-llama/Llama-3.2-3B-Instruct`, transferred as a local checkpoint directory and passed by path. Vietnamese is not among the [model card's eight officially supported languages](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct), and its out-of-scope section cautions against use in unsupported languages; another section says developers may fine-tune beyond those languages under the license and acceptable-use policy. Resolve this with company legal/research approval before treating Llama as an acceptable Vietnamese candidate. I would compare it with a Vietnamese-centered checkpoint under the same split before choosing a final model. The scripts force Hugging Face/Transformers offline mode and require local model paths; they do not download a model or install packages.

`requirements-grpo.txt` targets TRL 1.9.2 with its PEFT/vLLM extras and TensorBoard. It is an environment request, not a complete lockfile: company IT must provision and freeze compatible Python/PyTorch/CUDA/Transformers/TRL/PEFT/vLLM/Datasets/TensorBoard versions. The scripts enforce TRL 1.9.2 and run locally/offline; they do not install packages. The training path has not been run in this workspace because the company CUDA/vLLM stack is not available here. Check the company environment before a GPU run and start with a short approved pilot.

The pilot uses one prompt per device, four sampled completions per prompt, gradient accumulation 4, BF16, LoRA on attention projections, CPU metric scorers, and no separate KL reference model (`beta=0`). The example explicitly enables vLLM colocated with training; it can speed rollout generation, but shares GPU memory and may contend with optimization. The CLI leaves vLLM disabled unless `--use_vllm` is passed. Four is the number of sampled outputs per prompt, not the number of training documents. A B200 with 80–100 GB free may be sufficient for this 3B LoRA configuration, but that is not a guarantee; begin with a short pilot and watch the actual GPU memory. The CPU metric scorer may become the bottleneck after generation accelerates.

## Data contract and leakage controls

Input is UTF-8 JSONL, one object per line. The canonical training schema is `id`, `text`, and `summary`, where `text` is the source document and `summary` is the human reference. `source` is accepted as an alias, and the historical `input`/`human_sum` names remain supported. Explicit column arguments still override inference. `llm_sum` is an optional existing system output for baseline evaluation; it is not treated as the human target. `style` is optional. The preparer validates text and IDs, creates a Vietnamese chat prompt, and stores the human reference in a separate `reference` field for reward computation. Prompt construction and the trainer both keep `reference` out of the prompt. The split groups normalized exact duplicate source documents together so they cannot cross train/validation/test. The included legacy sample uses `abstract_sum`, so pass `--reference_col abstract_sum` for that schema.

```json
{"id":"7","style":"daily","input":"văn bản nguồn...","human_sum":"tóm tắt do con người viết...","llm_sum":"tóm tắt do mô hình tạo..."}
```

Use enough distinct documents for all three splits. A one-record sample can only smoke-test parsing with both holdout fractions set to zero; it is not a training or evaluation dataset. The deterministic split is not stratified. For a research study, inspect the split counts and balance domains/styles manually before training.

## Workflow

Run commands from the repository root. Paths below are examples; on the company machine use the already provisioned environment and local paths. None of these commands installs software.

### 1. Prepare and combine data

```bash
python scripts/prepare_grpo_data.py \
  --input /data/vdt/summaries_part1.jsonl /data/vdt/summaries_part2.jsonl \
  --output_dir results/grpo_splits \
  --validation_fraction 0.1 --test_fraction 0.1 --seed 42
```

Pass one or more same-schema JSONL paths after `--input`; the preparer validates and combines them, namespaces IDs by input file, then performs one grouped split across the combined corpus. Review `data_manifest.json`, row counts, duplicate-source grouping, and examples in `train.jsonl`, `validation.jsonl`, and `test.jsonl`. Only `train.jsonl` is for training; held-out records intentionally remain outside it. The original input files are read-only; the `--overwrite` switch replaces only named outputs in `--output_dir`.

For a parser-only smoke test on the included legacy single-record sample, use `--reference_col abstract_sum --validation_fraction 0 --test_fraction 0 --output_dir results/parser_smoke`. Do not train or report evaluation from that one record. Outputs are restricted to this repository unless you deliberately pass `--allow_external_output`; keep normal experiments under `results/`. As a rough pilot heuristic, treat fewer than 100 training rows as a pipeline/sensitivity exercise rather than evidence of generalization; source diversity matters more than the raw count.

### 2. Optional SFT warm-start (not required)

For an already instruction-tuned base such as the proposed Llama Instruct checkpoint, SFT is not a prerequisite: the main experiment can start directly with GRPO. The human summary is already used by the reference-overlap reward, so adding SFT changes the treatment and can amplify reference wording bias. Keep SFT as a separately labeled warm-start/control, or use it if the chosen base model does not follow the summarization prompt. It computes loss on the human completion only. `validation.jsonl` is used for reference negative log-likelihood, not factuality.

```bash
python -m src.training.train_sft \
  --model models/Llama-3.2-3B-Instruct \
  --train_jsonl results/grpo_splits/train.jsonl \
  --eval_jsonl results/grpo_splits/validation.jsonl \
  --output_dir outputs/sft_001 \
  --run_name sft-001 --ablation sft-warm-start \
  --report_to tensorboard --seed 42
```

The resulting adapter is `.../sft_001/final_adapter`. You may also skip SFT and start GRPO directly from the instruct checkpoint. Do not use the held-out test split for SFT, hyperparameter selection, or reward tuning.

### 3. Run GRPO with one selected metric

Start with one metric. `--faithfulness_metrics` is required so metric choice is explicit. The reference-overlap reward is included at weight 0.25 by default; the selected faithfulness metric has weight 1.0. TRL sums these weighted components without reward-scale normalization (`scale_rewards=none`), so the weights are part of the scientific treatment and must be recorded. `reference_weight=0` disables the overlap contribution. Combining several metrics requires the explicit `--allow_multiple_metrics` acknowledgement and is not recommended for the first comparison.

Recommended first comparison: start from the original instruction checkpoint with LoRA (using the already transferred MiniCheck cache):

```bash
python -m src.training.train_grpo \
  --model models/Llama-3.2-3B-Instruct \
  --train_jsonl results/grpo_splits/train.jsonl \
  --output_dir outputs/grpo_minicheck_001 \
  --faithfulness_metrics minicheck \
  --hf_cache_dir models/hf-cache \
  --internal_validation_fraction 0.1 \
  --reward_device cpu --reward_batch_size 4 \
  --reference_weight 0.25 --metric_weights 1.0 \
  --learning_rate 1e-6 --lr_scheduler_type linear --warmup_ratio 0.03 \
  --num_generations 4 --num_generations_eval 1 \
  --per_device_train_batch_size 1 --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 4 --num_train_epochs 1 \
  --max_prompt_tokens 4096 --max_completion_length 256 \
  --eval_strategy epoch --save_strategy steps --save_steps 50 \
  --max_checkpoints 2 --logging_steps 1 \
  --run_name grpo-minicheck-001 --ablation minicheck-plus-reference \
  --finetuning_method lora --lora_r 16 --lora_alpha 32 \
  --report_to tensorboard --use_vllm --vllm_mode colocate \
  --vllm_gpu_memory_utilization 0.25 --precision bf16 --seed 42
```

The GRPO runner exposes three update modes: `--finetuning_method lora` (default; compact adapter), `qlora` (4-bit NF4 by default with double quantization; requires company-provisioned `bitsandbytes`), and `fft` (full fine-tuning; saves all model weights). QLoRA still trains LoRA adapters over a quantized base. Run each mode into a separate `outputs/<run-name>/` folder and label `--ablation` accordingly. For a cautious first QLoRA pilot, use `--no-use_vllm`; TRL documents QLoRA and vLLM separately, but this repository/company stack has not verified their combination. FFT has a much larger optimizer-memory footprint than LoRA/QLoRA and should be a separately approved pilot even on a B200. Existing adapter checkpoints may only be continued using LoRA. For a base-model start, pass the base checkpoint as `--model` and omit `--base_model`. For FactCC, select `factcc` and provide `--factcc_model_path`. For AlignScore, select `alignscore` and use the default local pair `models/alignscore/AlignScore-base.ckpt` plus `models/roberta-base/`; override `--alignscore_ckpt` and `--alignscore_backbone_path` only when those files are stored elsewhere. MiniCheck uses `--hf_cache_dir` for its own cache/model folder.

With the default `--per_device_eval_batch_size 1`, validation uses one generation per prompt (`--num_generations_eval 1`), while training still samples four completions per prompt. If you want four validation generations, use an eval batch size divisible by four, such as `--per_device_eval_batch_size 4`. When no `--eval_jsonl` is provided, `--internal_validation_fraction 0.1` deterministically holds out 10% of the training rows by normalized source group; the source file is not changed, and the split settings are recorded in `run_manifest.json`. Set it to `0` to disable internal validation. Keep the external dataset completely untouched until final testing. Each run is a self-contained folder under `outputs/<run-name>/`, matching VDT-Anonymization's run layout. It contains `run_manifest.json` (full CLI/config, data hashes, hardware and package versions), TensorBoard events under `tensorboard/`, Trainer checkpoints, `training_history.json`, and either `final_adapter/` (LoRA/QLoRA) or `final_model/` (FFT). Resume with `--resume_from_checkpoint outputs/<same-run>/checkpoint-<step>`; the runner checks the training-data hash, base-model path, and fine-tuning method, and keeps the resume inside that run folder. `models/` is reserved for imported model assets; `results/` holds prepared splits and evaluation artifacts. Both training runners reject output paths outside this repo. TensorBoard is local only (`report_to=tensorboard`); completion-text logging stays off unless explicitly requested because generated text may be sensitive. To view curves after training:

```bash
tensorboard --logdir outputs/grpo_minicheck_001/tensorboard --host 127.0.0.1 --port 6006
```

The generation script also writes a sidecar manifest with input/output hashes and decoding settings. Treat run directories and generated JSONL as sensitive: they contain source text and summaries.

### 4. Generate held-out summaries and score them

Generate validation summaries first while choosing settings. Freeze choices before generating the test set; use test results once for the final report. The generator uses greedy decoding by default (same input, deterministic output), with a configurable maximum output length. The generated JSONL contains `input`, `human_sum`, and one chosen model column, so the existing multi-metric evaluation CLI can score the human summary and model output against the same source. You can generate multiple systems on identical rows and merge by ID with `--existing_jsonl`.

```bash
python scripts/generate_summaries.py \
  --model models/Llama-3.2-3B-Instruct \
  --input_jsonl results/grpo_splits/test.jsonl \
  --output results/grpo_test.jsonl \
  --summary_col base_sum

python scripts/generate_summaries.py \
  --model outputs/grpo_minicheck_001/final_adapter \
  --base_model models/Llama-3.2-3B-Instruct \
  --input_jsonl results/grpo_splits/test.jsonl \
  --existing_jsonl results/grpo_test.jsonl \
  --output results/grpo_test_compare.jsonl \
  --summary_col grpo_sum

python -m src.evaluate.run_eval \
  --data results/grpo_test_compare.jsonl \
  --summary_cols human_sum base_sum grpo_sum \
  --metrics factcc minicheck alignscore \
  --factcc_model_path models/factcc \
  --alignscore_ckpt models/alignscore/AlignScore-base.ckpt \
  --alignscore_backbone_path models/roberta-base \
  --hf_cache_dir models/hf-cache \
  --nltk_data_dir models/nltk_data \
  --offline --batch_size 2 \
  --output results/grpo_test_scored.jsonl
```

To compare base, LoRA/QLoRA/FFT, optional SFT, and GRPO fairly, generate each from the exact same validation/test rows using the same decoding settings and preserve each output column. The included generator can merge outputs by `id` after confirming source/reference equality. Do not accidentally score a training row as a test example.

### QLoRA and full fine-tuning ablation commands

Keep the base model, splits, reward, seed, and logging settings the same; give each treatment a unique output directory and ablation label. Tune each method using validation only. These are initial pipeline pilots, not a claim of a fair hyperparameter comparison:

```bash
python -m src.training.train_grpo --model models/Llama-3.2-3B-Instruct --train_jsonl results/grpo_splits/train.jsonl --eval_jsonl results/grpo_splits/validation.jsonl --output_dir outputs/grpo_qlora_minicheck_001 --faithfulness_metrics minicheck --hf_cache_dir models/hf-cache --finetuning_method qlora --no-use_vllm --eval_strategy epoch --report_to tensorboard --run_name grpo-qlora-minicheck-001 --ablation qlora-minicheck-plus-reference --seed 42

python -m src.training.train_grpo --model models/Llama-3.2-3B-Instruct --train_jsonl results/grpo_splits/train.jsonl --eval_jsonl results/grpo_splits/validation.jsonl --output_dir outputs/grpo_fft_minicheck_001 --faithfulness_metrics minicheck --hf_cache_dir models/hf-cache --finetuning_method fft --no-use_vllm --eval_strategy epoch --report_to tensorboard --run_name grpo-fft-minicheck-001 --ablation fft-minicheck-plus-reference --seed 42
```

## Reward definitions and limitations

- `reference_char_reward`: whitespace-insensitive, Unicode NFC-normalized character n-gram F-beta (orders 1–6, beta 2). It is an explicitly named chrF-style proxy, not the canonical SacreBLEU chrF metric. It rewards surface overlap, can discourage valid paraphrases, and can reward copying while failing to detect unsupported claims.
- FactCC, MiniCheck, or AlignScore: existing project evaluator adapter, called on `(source, generated summary)`. Non-finite scores become zero and values are clipped to `[0, 1]` before entering GRPO. Clipping is only a common numeric bound; it does not calibrate or make the different metrics comparable.
- The human reference is neither a factual oracle nor a complete set of valid summaries. A single reference encourages its wording and content selection. Use several references if available, report human review, and include coverage/omission analysis alongside factual support.
- Reward models are not differentiable through their text scores; GRPO uses sampled output rewards. Metric quality, prompt length, sampling temperature, and number of generations all affect the optimization signal.
- B200 GPU capacity does not fix metric language bias. Validate the selected metric against Vietnamese expert labels before treating it as a reward suitable for a claim about factuality.

## Reproducibility checklist

Archive the input hash, split manifest, exact local model and adapter hashes, model/license approval, Python/PyTorch/CUDA/Transformers/TRL/PEFT/Datasets versions, GPU model and free-memory observation, seeds, prompts, reward implementations and weights, generation settings, all run manifests, per-example metric outputs, and the human annotation protocol. Keep raw data and run artifacts inside approved company storage. Avoid uploading company text or checkpoints to public trackers or model hubs.
