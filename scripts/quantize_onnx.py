"""Quantize the ONNX model to INT8 to cut size ~4x.

Uses ONNX Runtime dynamic quantization (weight-only INT8), which needs no
calibration dataset. Note: static quantization is NOT used here because it
produces broken output for this DistilBERT graph.

Known trade-off: dynamic INT8 reduces size ~4x but costs ~6% intent accuracy.
The FP32 model remains the reference.

Usage:
    python scripts/quantize_onnx.py              # quantize + report sizes
    python scripts/quantize_onnx.py --verify     # compare predictions vs FP32
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "data" / "models"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
FP32_PATH = MODELS_DIR / "keep_calm.onnx"
INT8_PATH = MODELS_DIR / "keep_calm_int8.onnx"

MODEL_NAME = "distilbert-base-multilingual-cased"


def quantize() -> Path:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    print("Quantizing to INT8 (dynamic, weight-only) ...")
    quantize_dynamic(
        str(FP32_PATH),
        str(INT8_PATH),
        weight_type=QuantType.QInt8,
    )
    print(f"  FP32: {FP32_PATH.stat().st_size / 1e6:.1f} MB")
    print(f"  INT8: {INT8_PATH.stat().st_size / 1e6:.1f} MB")
    return INT8_PATH


def verify(int8_path: Path) -> None:
    import onnxruntime as ort
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    with open(SPLITS_DIR / "test.jsonl") as f:
        test = [json.loads(line) for line in f][:500]

    fp32 = ort.InferenceSession(str(FP32_PATH), providers=["CPUExecutionProvider"])
    int8 = ort.InferenceSession(str(int8_path), providers=["CPUExecutionProvider"])

    intent_mismatch = 0
    risk_mismatch = 0
    tone_mismatch = 0
    for d in test:
        enc = tokenizer(
            d["text"], truncation=True, padding="max_length", max_length=256, return_tensors="pt"
        )
        feeds = {"input_ids": enc["input_ids"].numpy(), "attention_mask": enc["attention_mask"].numpy()}
        r1, t1, i1 = fp32.run(["risk", "tone", "intent"], feeds)
        r2, t2, i2 = int8.run(["risk", "tone", "intent"], feeds)

        if i1.argmax() != i2.argmax():
            intent_mismatch += 1
        if abs(r1[0][0] - r2[0][0]) > 0.02:
            risk_mismatch += 1
        if (t1 > 0.4).astype(int).tolist() != (t2 > 0.4).astype(int).tolist():
            tone_mismatch += 1

    n = len(test)
    print(f"  On {n} test samples:")
    print(f"    intent argmax mismatches: {intent_mismatch} ({intent_mismatch / n * 100:.1f}%)")
    print(f"    risk >0.02 diffs: {risk_mismatch} ({risk_mismatch / n * 100:.1f}%)")
    print(f"    tone threshold mismatches: {tone_mismatch} ({tone_mismatch / n * 100:.1f}%)")

    # Latency
    single = {
        "input_ids": np.zeros((1, 256), dtype=np.int64),
        "attention_mask": np.ones((1, 256), dtype=np.int64),
    }
    for label, sess in [("FP32", fp32), ("INT8", int8)]:
        times = []
        for _ in range(20):
            t0 = time.perf_counter()
            sess.run(["risk", "tone", "intent"], single)
            times.append(time.perf_counter() - t0)
        print(f"    {label} latency: {np.mean(times) * 1000:.1f} ms")


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantize Keep Calm ONNX to INT8")
    parser.add_argument("--verify", action="store_true", help="Compare vs FP32")
    args = parser.parse_args()

    int8_path = quantize()
    if args.verify:
        verify(int8_path)


if __name__ == "__main__":
    main()
