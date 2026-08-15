# Keep Calm — in-browser demo (WASM)

Runs the Keep Calm model entirely in the browser via ONNX Runtime Web.
No server, no data leaves the page.

## Prerequisites

Two files must be present in this folder before serving:

1. **`keep_calm_int8.onnx`** — the quantized model (135MB)
2. **`vocab.json`** — already present (extracted from tokenizer.json)

Copy the model from the repo:

```bash
cp ../data/models/keep_calm_int8.onnx .
```

## Run locally

```bash
cd wasm
python -m http.server 8080
# open http://localhost:8080
```

The model (~135MB) downloads into the browser on first load, then runs
locally via WebAssembly.

## How it works

- `tokenizer.js` — WordPiece tokenizer (distilbert-base-multilingual-cased)
- `app.js` — loads the ONNX model with `onnxruntime-web`, runs inference,
  and renders risk / tone / intent / explanation
- `index.html` — minimal UI

## Notes

- The model is INT8-quantized (~135MB) for browser feasibility; it trades
  ~6% intent accuracy vs the FP32 reference.
- `onnxruntime-web` is loaded from a CDN; the model itself is local.
