#!/usr/bin/env bash
# Download pre-trained Keep Calm models.
#
# Usage:
#   ./scripts/download_models.sh                # auto-detect best source
#   ./scripts/download_models.sh --hf-only      # force Hugging Face
#   ./scripts/download_models.sh --gh-release   # force GitHub Releases
#
# Models are saved to data/models/ by default.
# Set KEEP_CALM_MODELS_DIR to override.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODELS_DIR="${KEEP_CALM_MODELS_DIR:-$SCRIPT_DIR/../data/models}"
mkdir -p "$MODELS_DIR"

FILES=(
    risk_encoder.pt risk_head.pt
    tone_encoder.pt tone_head.pt
    intent_encoder.pt intent_head.pt
    tokenizer.json tokenizer_config.json
    config.json metrics.json
)

HF_REPO="elchief84/keep-calm-models"
GH_REPO="keep-calm/keep-calm"
GH_TAG="${KEEP_CALM_MODELS_TAG:-v0.1.0-models}"

already_have_models() {
    for f in "${FILES[@]}"; do
        [ -f "$MODELS_DIR/$f" ] || return 1
    done
    return 0
}

try_huggingface() {
    echo "Downloading from Hugging Face ($HF_REPO) ..."
    if command -v huggingface-cli &>/dev/null; then
        huggingface-cli download "$HF_REPO" --local-dir "$MODELS_DIR" --local-dir-use-symlinks False
    else
        pip install -q huggingface_hub && \
        huggingface-cli download "$HF_REPO" --local-dir "$MODELS_DIR" --local-dir-use-symlinks False
    fi
}

try_github_release() {
    echo "Downloading from GitHub Releases ($GH_REPO @ $GH_TAG) ..."
    local base="https://github.com/$GH_REPO/releases/download/$GH_TAG"
    for f in "${FILES[@]}"; do
        echo "  $f"
        curl -fSL --progress-bar -o "$MODELS_DIR/$f" "$base/$f"
    done
}

print_manual() {
    cat <<EOF
Keep Calm — Model Download
===========================

Could not download models automatically. Options:

  1. Hugging Face:
     pip install huggingface_hub
     huggingface-cli download $HF_REPO --local-dir $MODELS_DIR

  2. GitHub Releases:
     wget https://github.com/$GH_REPO/releases/download/$GH_TAG/{risk_encoder.pt,...}

  3. Train from scratch:
     python scripts/train_and_save.py

Or contact the project maintainer for model files.
EOF
}

# ── main ──

if already_have_models; then
    echo "Models already present in $MODELS_DIR"
    exit 0
fi

case "${1:-}" in
    --hf-only)
        try_huggingface && already_have_models && exit 0
        ;;
    --gh-release)
        try_github_release && already_have_models && exit 0
        ;;
    *)
        if try_huggingface 2>/dev/null && already_have_models; then exit 0; fi
        echo "Hugging Face unavailable, trying GitHub Releases ..."
        if try_github_release 2>/dev/null && already_have_models; then exit 0; fi
        ;;
esac

print_manual
exit 1
