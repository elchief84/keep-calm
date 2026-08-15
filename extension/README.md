# Keep Calm — Browser Extension

Analyzes messages in chat/email text fields **before you send them** — running
100% locally in your browser via ONNX Runtime Web. No data leaves the page.

## How it works

- `content.js` detects `textarea`, `input`, and `contenteditable` fields and
  shows a small inline badge while you type (debounced).
- `background.js` routes analysis requests to the offscreen document.
- `offscreen.js` hosts the INT8-quantized ONNX model and runs inference.
- `tokenizer.js` is the WordPiece tokenizer (distilbert-base-multilingual-cased).

## Setup

```bash
./scripts/setup_extension.sh
```

This downloads ONNX Runtime Web (JS + WASM) and copies the quantized model
(`keep_calm_int8.onnx`, ~129MB) and the vocab.

## Install (Chrome)

1. Open `chrome://extensions`
2. Enable **Developer mode** (top right)
3. Click **Load unpacked**
4. Select the `extension/` folder

## Usage

Go to Gmail, Slack web, Teams web, or any page with a text field, start typing,
and a colored badge appears near the field:

- 🟢 green = low risk
- 🟠 orange = medium
- 🔴 red = high

Hover the badge to read the explanation.

## Notes

- The model is INT8-quantized (~129MB) — it trades ~6% intent accuracy vs FP32.
- First analysis is slow (model load), subsequent ones are fast.
- The offscreen document keeps the model in memory; the service worker reuses it.
