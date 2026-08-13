"""Export the multi-task Keep Calm model to ONNX.

Exports a single shared encoder with three heads (risk, tone, intent) as one
self-contained ONNX graph with three outputs. This is the prerequisite for
WASM / non-Python inference and reduces the footprint to a single encoder.

Usage:
    python scripts/export_onnx.py                     # export
    python scripts/export_onnx.py --verify            # also check vs PyTorch
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

MODEL_NAME = "distilbert-base-multilingual-cased"
MODELS_DIR = Path(__file__).resolve().parents[1] / "data" / "models"
ONNX_PATH = MODELS_DIR / "keep_calm.onnx"


class MultiTaskWrapper(nn.Module):
    def __init__(self, encoder: nn.Module, risk_head: nn.Module,
                 tone_head: nn.Module, intent_head: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.risk_head = risk_head
        self.tone_head = tone_head
        self.intent_head = intent_head

    def forward(self, input_ids, attention_mask):
        emb = self.encoder(input_ids, attention_mask).last_hidden_state[:, 0, :]
        return self.risk_head(emb), self.tone_head(emb), self.intent_head(emb)


def build_wrapper() -> MultiTaskWrapper:
    encoder = AutoModel.from_pretrained(MODEL_NAME)
    risk_head = nn.Sequential(
        nn.Linear(768, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 1), nn.Sigmoid()
    )
    tone_head = nn.Sequential(
        nn.Linear(768, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 5), nn.Sigmoid()
    )
    intent_head = nn.Sequential(
        nn.Linear(768, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 4)
    )

    encoder.load_state_dict(
        torch.load(MODELS_DIR / "multitask_encoder.pt", map_location="cpu", weights_only=True)
    )
    risk_head.load_state_dict(
        torch.load(MODELS_DIR / "multitask_risk_head.pt", map_location="cpu", weights_only=True)
    )
    tone_head.load_state_dict(
        torch.load(MODELS_DIR / "multitask_tone_head.pt", map_location="cpu", weights_only=True)
    )
    intent_head.load_state_dict(
        torch.load(MODELS_DIR / "multitask_intent_head.pt", map_location="cpu", weights_only=True)
    )
    encoder.eval()
    risk_head.eval()
    tone_head.eval()
    intent_head.eval()

    return MultiTaskWrapper(encoder, risk_head, tone_head, intent_head).eval()


def export() -> Path:
    wrapper = build_wrapper()
    dummy_ids = torch.zeros(1, 256, dtype=torch.long)
    dummy_mask = torch.ones(1, 256, dtype=torch.long)

    torch.onnx.export(
        wrapper,
        (dummy_ids, dummy_mask),
        str(ONNX_PATH),
        input_names=["input_ids", "attention_mask"],
        output_names=["risk", "tone", "intent"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "risk": {0: "batch"},
            "tone": {0: "batch"},
            "intent": {0: "batch"},
        },
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"  Exported {ONNX_PATH} ({ONNX_PATH.stat().st_size / 1e6:.1f} MB)")
    return ONNX_PATH


def verify(onnx_path: Path) -> None:
    import numpy as np
    import onnxruntime as ort

    wrapper = build_wrapper()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    texts = [
        "You are completely useless.",
        "Great work on this, thank you!",
        "Can you send me the report?",
    ]
    enc = tokenizer(texts, truncation=True, padding="max_length", max_length=256, return_tensors="pt")

    with torch.no_grad():
        r, t, i = wrapper(enc["input_ids"], enc["attention_mask"])
        torch_outs = [r.numpy(), t.numpy(), i.numpy()]

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_outs = session.run(
        ["risk", "tone", "intent"],
        {"input_ids": enc["input_ids"].numpy(), "attention_mask": enc["attention_mask"].numpy()},
    )

    for name, a, b in zip(["risk", "tone", "intent"], torch_outs, onnx_outs, strict=True):
        diff = float(np.abs(a - b).max())
        print(f"  Verify {name}: max abs diff = {diff:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export multi-task Keep Calm model to ONNX")
    parser.add_argument("--verify", action="store_true", help="Check output vs PyTorch")
    args = parser.parse_args()

    print("Exporting multi-task model ...")
    onnx_path = export()
    if args.verify:
        verify(onnx_path)

    print("\nDone. ONNX model:", ONNX_PATH)


if __name__ == "__main__":
    main()
