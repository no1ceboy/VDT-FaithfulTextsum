#!/usr/bin/env bash
# install_deps.sh — Install all faithfulness metric dependencies
#
# Usage:
#   bash scripts/install_deps.sh [--all | --factcc | --fenice | --minicheck | --alignscore | --qafacteval]
#
# If no argument is given, all metrics are installed.
#
# For offline cluster use:
#   - Run this script ON A MACHINE WITH INTERNET ACCESS first.
#   - Then copy the installed packages and downloaded model weights to the cluster.
#   - Set HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1 before running run_all.py.

set -e

INSTALL_ALL=false
INSTALL_FACTCC=false
INSTALL_FENICE=false
INSTALL_MINICHECK=false
INSTALL_ALIGNSCORE=false
INSTALL_QAFACTEVAL=false

if [[ $# -eq 0 ]]; then
    INSTALL_ALL=true
fi

for arg in "$@"; do
    case $arg in
        --all)         INSTALL_ALL=true ;;
        --factcc)      INSTALL_FACTCC=true ;;
        --fenice)      INSTALL_FENICE=true ;;
        --minicheck)   INSTALL_MINICHECK=true ;;
        --alignscore)  INSTALL_ALIGNSCORE=true ;;
        --qafacteval)  INSTALL_QAFACTEVAL=true ;;
        *)
            echo "Unknown argument: $arg"
            echo "Usage: $0 [--all | --factcc | --fenice | --minicheck | --alignscore | --qafacteval]"
            exit 1
            ;;
    esac
done

# ---------------------------------------------------------------------------
# 1. FactCC — via HuggingFace transformers (no extra install needed)
# ---------------------------------------------------------------------------
if $INSTALL_ALL || $INSTALL_FACTCC; then
    echo ""
    echo "=========================================="
    echo " Installing FactCC dependencies"
    echo "=========================================="
    pip install transformers torch accelerate tqdm
    echo ""
    echo "[FactCC] To pre-download the model for offline use:"
    echo "  python -c \"from transformers import AutoModelForSequenceClassification, AutoTokenizer; \\"
    echo "              AutoTokenizer.from_pretrained('manueldeprada/FactCC'); \\"
    echo "              AutoModelForSequenceClassification.from_pretrained('manueldeprada/FactCC')\""
fi

# ---------------------------------------------------------------------------
# 2. FENICE
# ---------------------------------------------------------------------------
if $INSTALL_ALL || $INSTALL_FENICE; then
    echo ""
    echo "=========================================="
    echo " Installing FENICE"
    echo "=========================================="
    pip install FENICE
    python -m spacy download en_core_web_sm || true
    echo ""
    echo "[FENICE] NOTE: FENICE was designed for English."
    echo "         On Vietnamese text scores are computed but may be less calibrated."
fi

# ---------------------------------------------------------------------------
# 3. MiniCheck
# ---------------------------------------------------------------------------
if $INSTALL_ALL || $INSTALL_MINICHECK; then
    echo ""
    echo "=========================================="
    echo " Installing MiniCheck"
    echo "=========================================="
    pip install "minicheck @ git+https://github.com/Liyan06/MiniCheck.git@main"
    echo ""
    echo "[MiniCheck] Default model: Bespoke-MiniCheck-7B"
    echo "  To pre-download for offline use:"
    echo "  python -c \"from minicheck.minicheck import MiniCheck; MiniCheck('Bespoke-MiniCheck-7B')\""
    echo "  Then copy ~/.cache/huggingface/hub/models--Bespoke-Teknologies--Bespoke-MiniCheck-7B to the cluster."
fi

# ---------------------------------------------------------------------------
# 4. AlignScore
# ---------------------------------------------------------------------------
if $INSTALL_ALL || $INSTALL_ALIGNSCORE; then
    echo ""
    echo "=========================================="
    echo " Installing AlignScore"
    echo "=========================================="
    ALIGNSCORE_DIR="/tmp/AlignScore_install"
    if [ ! -d "$ALIGNSCORE_DIR" ]; then
        git clone https://github.com/yuh-zha/AlignScore "$ALIGNSCORE_DIR"
    fi
    pip install -e "$ALIGNSCORE_DIR"
    python -m spacy download en_core_web_sm || true
    echo ""
    echo "[AlignScore] To pre-download AlignScore-large checkpoint:"
    echo "  pip install huggingface_hub"
    echo "  python -c \"from huggingface_hub import hf_hub_download; \\"
    echo "              hf_hub_download(repo_id='yzha/AlignScore', filename='AlignScore-large.ckpt', \\"
    echo "              local_dir='./models/alignscore')\""
fi

# ---------------------------------------------------------------------------
# 5. QAFactEval
# ---------------------------------------------------------------------------
if $INSTALL_ALL || $INSTALL_QAFACTEVAL; then
    echo ""
    echo "=========================================="
    echo " Installing QAFactEval"
    echo "=========================================="
    QAFACTEVAL_DIR="/tmp/QAFactEval_install"
    if [ ! -d "$QAFACTEVAL_DIR" ]; then
        git clone https://github.com/salesforce/QAFactEval "$QAFACTEVAL_DIR"
    fi
    pip install -e "$QAFACTEVAL_DIR"
    echo ""
    echo "[QAFactEval] Downloading pretrained models (QA + QG + LERC-QUIP)…"
    bash "$QAFACTEVAL_DIR/download_models.sh" || {
        echo "[QAFactEval] WARNING: download_models.sh failed."
        echo "             Download models manually and pass --qafacteval_model_path to run_all.py."
    }
fi

echo ""
echo "=========================================="
echo " Installation complete."
echo "=========================================="
echo ""
echo "Quick smoke test:"
echo "  python evaluate/run_all.py \\"
echo "      --data data/sample.jsonl \\"
echo "      --metrics factcc minicheck \\"
echo "      --output results/smoke_test.jsonl"
