#!/usr/bin/env bash
# Download pre-trained Keep Calm models.
#
# Usage:
#   ./scripts/download_models.sh
#
# Set KEEP_CALM_MODELS_URL to override the download source.
# Models are saved to data/models/ by default.
# Set KEEP_CALM_MODELS_DIR to override the output directory.

set -euo pipefail

MODELS_DIR="${KEEP_CALM_MODELS_DIR:-$(dirname "$0")/../data/models}"
mkdir -p "$MODELS_DIR"

FILES=(
    risk_encoder.pt
    risk_head.pt
    tone_encoder.pt
    tone_head.pt
    intent_encoder.pt
    intent_head.pt
    tokenizer.json
    tokenizer_config.json
    config.json
    metrics.json
)

BASE_URL="${KEEP_CALM_MODELS_URL:-}"

if [ -z "$BASE_URL" ]; then
    cat <<EOF
Keep Calm — Model Download
===========================

Models are not publicly hosted yet. To download them:

Option 1 — Hugging Face (recommended, once published):
    pip install huggingface_hub
    huggingface-cli download keep-calm/keep-calm-models --local-dir "$MODELS_DIR"

Option 2 — Manual download:
    Place the following files in $MODELS_DIR:
$(printf '    - %s\n' "${FILES[@]}")

Option 3 — Train from scratch:
    python scripts/train_and_save.py

For now, models must be trained locally or obtained from the project maintainer.
EOF
    exit 1
fi

echo "Downloading models from $BASE_URL to $MODELS_DIR ..."
for file in "${FILES[@]}"; do
    echo "  $file"
    curl -fSL --progress-bar -o "$MODELS_DIR/$file" "$BASE_URL/$file"
done

echo "Done. Models saved to $MODELS_DIR"
