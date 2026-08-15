#!/usr/bin/env bash
# Download the binary assets required by the browser extension.
#
# These are gitignored because they are large and downloadable:
#   - ONNX Runtime Web (JS + WASM) from jsdelivr
#   - the quantized model from the repo's data/models/
#
# Usage:
#   ./scripts/setup_extension.sh

set -euo pipefail

EXT_DIR="$(cd "$(dirname "$0")/../extension" && pwd)"

echo "Downloading ONNX Runtime Web ..."
curl -sL -o "$EXT_DIR/ort.min.mjs" \
  "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.mjs"
for f in \
  ort-wasm-simd-threaded.mjs ort-wasm-simd-threaded.wasm \
  ort-wasm-simd-threaded.jsep.mjs ort-wasm-simd-threaded.jsep.wasm \
  ort-wasm-simd.mjs ort-wasm-simd.wasm \
  ort-wasm-simd.jsep.mjs ort-wasm-simd.jsep.wasm \
  ort-wasm.mjs ort-wasm.wasm ort-wasm.jsep.mjs; do
  curl -sL -o "$EXT_DIR/$f" "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/$f"
done

echo "Copying quantized model ..."
if [ -f "$EXT_DIR/../data/models/keep_calm_int8.onnx" ]; then
  cp "$EXT_DIR/../data/models/keep_calm_int8.onnx" "$EXT_DIR/model/"
  echo "  copied from data/models/"
else
  echo "  model not found — run: python scripts/quantize_onnx.py first"
  exit 1
fi

echo "Copying vocab ..."
cp "$EXT_DIR/../wasm/vocab.json" "$EXT_DIR/vocab.json"

echo "Done. Load the extension from chrome://extensions -> Load unpacked -> $EXT_DIR"
