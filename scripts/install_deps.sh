#!/usr/bin/env bash
# Install dependencies for selected metric(s).
# FENICE and QAFactEval should be installed in their own environments if their
# legacy dependency pins conflict with the shared FactCC/MiniCheck/AlignScore env.

set -euo pipefail

if [[ $# -eq 0 ]]; then
    echo "Usage: bash scripts/install_deps.sh --factcc|--minicheck|--alignscore|--fenice|--qafacteval [more compatible metrics]"
    echo "Use a separate environment for --fenice and --qafacteval."
    exit 2
fi

INSTALL_FACTCC=false
INSTALL_FENICE=false
INSTALL_MINICHECK=false
INSTALL_ALIGNSCORE=false
INSTALL_QAFACTEVAL=false

for arg in "$@"; do
    case "$arg" in
        --factcc) INSTALL_FACTCC=true ;;
        --fenice) INSTALL_FENICE=true ;;
        --minicheck) INSTALL_MINICHECK=true ;;
        --alignscore) INSTALL_ALIGNSCORE=true ;;
        --qafacteval) INSTALL_QAFACTEVAL=true ;;
        --all)
            echo "Refusing to install all metrics into one environment: FENICE/QAFactEval have legacy dependency constraints."
            echo "Choose compatible metrics, or install FENICE/QAFactEval in separate virtual environments."
            exit 2
            ;;
        *) echo "Unknown argument: $arg"; exit 2 ;;
    esac
done

LEGACY_COUNT=0
$INSTALL_FENICE && ((LEGACY_COUNT+=1))
$INSTALL_QAFACTEVAL && ((LEGACY_COUNT+=1))
SELECTED_COUNT=0
$INSTALL_FACTCC && ((SELECTED_COUNT+=1))
$INSTALL_MINICHECK && ((SELECTED_COUNT+=1))
$INSTALL_ALIGNSCORE && ((SELECTED_COUNT+=1))
$INSTALL_FENICE && ((SELECTED_COUNT+=1))
$INSTALL_QAFACTEVAL && ((SELECTED_COUNT+=1))
if (( LEGACY_COUNT > 0 && SELECTED_COUNT > 1 )); then
    echo "Install FENICE and QAFactEval one at a time in separate environments."
    echo "Install them separately from FactCC, MiniCheck, and AlignScore too."
    exit 2
fi

if $INSTALL_FACTCC || $INSTALL_MINICHECK || $INSTALL_ALIGNSCORE; then
    pip install -r requirements.txt
fi
if $INSTALL_MINICHECK; then
    NLTK_DATA_DIR="${NLTK_DATA_DIR:-models/nltk_data}"
    python -m nltk.downloader -d "$NLTK_DATA_DIR" punkt_tab
    echo "MiniCheck uses the included FLAN-T5 inference adapter; no minicheck package or Accelerate install needed."
    echo "Transfer $NLTK_DATA_DIR and set NLTK_DATA to that path for offline runs."
fi
if $INSTALL_ALIGNSCORE; then
    NLTK_DATA_DIR="${NLTK_DATA_DIR:-models/nltk_data}"
    python -m nltk.downloader -d "$NLTK_DATA_DIR" punkt_tab
    echo "AlignScore uses the included inference-only adapter; no upstream alignscore or spaCy package/model needed."
    echo "Transfer $NLTK_DATA_DIR and set NLTK_DATA to that path for offline runs."
fi
if $INSTALL_FENICE; then
    pip install FENICE tqdm
    python -m spacy download en_core_web_sm
    echo "FENICE uses its fixed Hugging Face model IDs; cache both models before offline use."
fi
if $INSTALL_QAFACTEVAL; then
    QAFACTEVAL_DIR="/tmp/QAFactEval_install"
    if [[ ! -d "$QAFACTEVAL_DIR/.git" ]]; then
        git clone https://github.com/salesforce/QAFactEval "$QAFACTEVAL_DIR"
    fi
    pip install tqdm
    pip install -e "$QAFACTEVAL_DIR"
    bash "$QAFACTEVAL_DIR/download_models.sh"
    echo "Pass the downloaded models directory as --qafacteval_model_path."
fi

echo "Selected metric dependencies installed. From the project root, run python -m src.evaluate.run_eval --help."
