# VDT-FaithfulTextsum

**Faithful Text Summarization Evaluation on Vietnamese Documents**

This repository provides an evaluation pipeline for assessing the **factual faithfulness** of abstractive summaries produced on Vietnamese text. It wraps five state-of-the-art metrics into a unified CLI that can run on a full dataset in a single command.

---

## Metrics

| Metric | Approach | Model (default) | Reference |
|---|---|---|---|
| **FactCC** | NLI classification | `manueldeprada/FactCC` | Kryscinski et al., 2020 |
| **FENICE** | Claim extraction + NLI | `roberta-large-mnli` | Scirè et al., 2024 |
| **MiniCheck** | Sentence fact-checking | `Bespoke-MiniCheck-7B` | Tang et al., 2024 |
| **AlignScore** | Unified alignment function | `AlignScore-large` | Zha et al., 2023 |
| **QAFactEval** | QA-based | LERC-QUIP models | Fabbri et al., 2022 |

> **Note on Vietnamese:** All metrics were originally developed for English. They will still produce numeric scores on Vietnamese text, but calibration may vary. FENICE and AlignScore use English NLI/coreference models and will emit a warning.

---

## Dataset Format

Each record in the input dataset is a JSON object (JSONL file, one record per line):

```json
{
  "id": "7",
  "style": "daily",
  "input": "<source document>",
  "abstract_sum": "<human summary>",
  "domain": "tư tưởng",
  "token_length": 1135,
  "level": 3
}
```

| Field | Description |
|---|---|
| `input` | Source document (used as the grounding/context for all metrics) |
| `abstract_sum` | Summary to evaluate (default). Can be any column via `--summary_col`. |

Future summary columns (e.g. `llm_sum` for LLM-generated summaries) are supported via the `--summary_col` flag without code changes.

---

## Installation

### 1. Clone and enter the repo
```bash
git clone <this-repo>
cd VDT-FaithfulTextsum
```

### 2. Install dependencies
```bash
# Install all metrics:
bash scripts/install_deps.sh

# Or install individual metrics:
bash scripts/install_deps.sh --factcc
bash scripts/install_deps.sh --fenice
bash scripts/install_deps.sh --minicheck
bash scripts/install_deps.sh --alignscore
bash scripts/install_deps.sh --qafacteval
```

### 3. Offline / cluster use (no internet)
On a machine with internet, pre-download models:
```bash
# FactCC
python -c "from transformers import AutoModelForSequenceClassification, AutoTokenizer; \
           AutoTokenizer.from_pretrained('manueldeprada/FactCC'); \
           AutoModelForSequenceClassification.from_pretrained('manueldeprada/FactCC')"

# MiniCheck (Bespoke-MiniCheck-7B)
python -c "from minicheck.minicheck import MiniCheck; MiniCheck('Bespoke-MiniCheck-7B')"

# AlignScore-large
python -c "from huggingface_hub import hf_hub_download; \
           hf_hub_download('yzha/AlignScore', 'AlignScore-large.ckpt', local_dir='./models/alignscore')"

# QAFactEval — run the provided download script in the cloned repo
bash /tmp/QAFactEval_install/download_models.sh
```

Then transfer model files to the cluster and use `--*_model_path` flags (see below).

---

## Usage

### Run all metrics on the sample data
```bash
python evaluate/run_all.py \
    --data data/sample.jsonl \
    --output results/scores.jsonl
```

### Run specific metrics
```bash
python evaluate/run_all.py \
    --data data/my_dataset.jsonl \
    --metrics factcc minicheck alignscore \
    --output results/scores.jsonl \
    --device cuda
```

### Evaluate LLM-generated summaries
```bash
python evaluate/run_all.py \
    --data data/with_llm_sums.jsonl \
    --summary_col llm_sum \
    --metrics minicheck alignscore \
    --output results/llm_scores.jsonl
```

### Offline mode (all models from local paths)
```bash
python evaluate/run_all.py \
    --data data/my_dataset.jsonl \
    --offline \
    --metrics factcc minicheck alignscore qafacteval \
    --factcc_model_path /mnt/models/factcc \
    --minicheck_model_path /mnt/models/Bespoke-MiniCheck-7B \
    --alignscore_ckpt /mnt/models/alignscore/AlignScore-large.ckpt \
    --qafacteval_model_path /mnt/models/qafacteval \
    --output results/scores.jsonl \
    --device cuda
```

### All CLI arguments
```
python evaluate/run_all.py --help
```

| Argument | Default | Description |
|---|---|---|
| `--data` | *(required)* | Input dataset (.jsonl or .json) |
| `--source_col` | `input` | Field containing the source document |
| `--summary_col` | `abstract_sum` | Field containing the summary to evaluate |
| `--metrics` | all | Space-separated list of metrics to run |
| `--output` | `results/scores.jsonl` | Output JSONL path |
| `--device` | `cuda` | PyTorch device string |
| `--batch_size` | `8` | Default batch size |
| `--limit` | None | Evaluate only the first N records |
| `--offline` | False | Set HF offline env vars |
| `--factcc_model_path` | None | Local path for FactCC |
| `--fenice_model_path` | None | Local NLI model path for FENICE |
| `--minicheck_model_path` | None | Local path for MiniCheck |
| `--minicheck_model_name` | `Bespoke-MiniCheck-7B` | MiniCheck model name |
| `--alignscore_ckpt` | None | Local .ckpt path for AlignScore |
| `--alignscore_mode` | `nli_sp` | AlignScore evaluation mode |
| `--qafacteval_model_path` | None | Local model folder for QAFactEval |

---

## Output Format

Each output record preserves all original fields and adds:

| Field | Type | Description |
|---|---|---|
| `factcc_score` | float [0,1] | Probability of CORRECT class |
| `fenice_score` | float [0,1] | FENICE factuality score |
| `minicheck_score` | float [0,1] | Mean sentence support probability |
| `minicheck_pred` | int 0/1 | MiniCheck aggregate prediction |
| `alignscore_score` | float [0,1] | AlignScore alignment score |
| `qafacteval_score` | float [0,1] | QAFactEval normalised score |
| `qafacteval_raw_score` | float | Raw QAFactEval score |

A `.summary.tsv` file is also written alongside the output JSONL with per-metric mean scores.

---

## Project Structure

```
VDT-FaithfulTextsum/
├── README.md
├── .gitignore
├── requirements.txt
├── data/
│   └── sample.jsonl          # 1-record sample for smoke tests
├── evaluate/
│   ├── __init__.py
│   ├── base.py               # Abstract base evaluator
│   ├── run_all.py            # Main CLI entry point
│   ├── factcc_eval.py        # FactCC wrapper
│   ├── fenice_eval.py        # FENICE wrapper
│   ├── minicheck_eval.py     # MiniCheck wrapper
│   ├── alignscore_eval.py    # AlignScore wrapper
│   └── qafacteval_eval.py    # QAFactEval wrapper
├── results/                  # Evaluation outputs (gitignored)
└── scripts/
    └── install_deps.sh       # Selective dependency installer
```

---

## Citation

If you use this pipeline, please cite the individual metrics:

```bibtex
@inproceedings{kryscinski2020evaluating,
  title={Evaluating the Factual Consistency of Abstractive Text Summarization},
  author={Kryscinski, Wojciech and McCann, Bryan and Xiong, Caiming and Socher, Richard},
  booktitle={EMNLP},
  year={2020}
}
@inproceedings{scire2024fenice,
  title={FENICE: Factuality Evaluation of summarization based on Natural language Inference and Claim Extraction},
  author={Scir{\`e}, Alessandro and others},
  booktitle={ACL Findings},
  year={2024}
}
@article{tang2024minicheck,
  title={MiniCheck: Efficient Fact-Checking of LLMs on Grounding Documents},
  author={Tang, Liyan and others},
  journal={arXiv},
  year={2024}
}
@inproceedings{zha2023alignscore,
  title={AlignScore: Evaluating Factual Consistency with a Unified Alignment Function},
  author={Zha, Yuheng and others},
  booktitle={ACL},
  year={2023}
}
@inproceedings{fabbri2022qafacteval,
  title={QAFactEval: Improved QA-Based Factual Consistency Evaluation for Summarization},
  author={Fabbri, Alexander R and others},
  booktitle={NAACL},
  year={2022}
}
```
