"""Train a multi-task Keep Calm model: one shared encoder + three heads.

Unlike the single-task approach (three separate encoders), this shares a single
DistilBERT encoder across risk / tone / intent, cutting the on-disk footprint
to ~1/3. Combined with INT8 quantization this is the prerequisite for a
browser-friendly (~135MB) model.

Losses:
    risk   — MSE on the 0-1 score
    tone   — BCE multi-label on the 5-dim soft vector
    intent — cross-entropy on the 4-class label
Total loss is a weighted sum (risk > tone > intent, per ARCHITECTURE.md).

Usage:
    python scripts/train_multitask.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
MODELS_DIR = PROJECT_ROOT / "data" / "models"
RESULTS_DIR = PROJECT_ROOT / "data" / "results"

TONE_LABELS = ["neutral", "frustrated", "hostile", "sarcastic", "positive"]
INTENT_LABELS = ["constructive", "critical", "personal", "informational"]
INTENT_MAP = {label: i for i, label in enumerate(INTENT_LABELS)}

MODEL_NAME = "distilbert-base-multilingual-cased"
BATCH_SIZE = 32
EPOCHS = 5
LR = 2e-5
LOSS_WEIGHTS = {"risk": 1.0, "tone": 1.0, "intent": 0.7}
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def _load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


class MultiTaskDataset(Dataset):
    def __init__(self, data: list[dict], tokenizer, max_len: int = 256):
        self.texts = [d["text"] for d in data]
        self.risk = [float(d["risk"]) for d in data]
        self.tone = [d["tone_vector"] for d in data]
        self.intent = [INTENT_MAP[d["intent"]] for d in data]
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        enc = self.tokenizer(
            self.texts[idx], truncation=True, padding="max_length",
            max_length=self.max_len, return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "risk": torch.tensor(self.risk[idx], dtype=torch.float),
            "tone": torch.tensor(self.tone[idx], dtype=torch.float),
            "intent": torch.tensor(self.intent[idx], dtype=torch.long),
        }


class MultiTaskModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(MODEL_NAME)
        self.risk_head = nn.Sequential(
            nn.Linear(768, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 1), nn.Sigmoid()
        )
        self.tone_head = nn.Sequential(
            nn.Linear(768, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 5), nn.Sigmoid()
        )
        self.intent_head = nn.Sequential(
            nn.Linear(768, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 4)
        )

    def forward(self, input_ids, attention_mask):
        emb = self.encoder(input_ids, attention_mask).last_hidden_state[:, 0, :]
        return self.risk_head(emb), self.tone_head(emb), self.intent_head(emb)


def evaluate(model, dataloader) -> dict:
    model.eval()
    risk_true, risk_pred = [], []
    tone_true, tone_pred = [], []
    intent_true, intent_pred = [], []

    with torch.no_grad():
        for batch in dataloader:
            ids = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            risk, tone, intent = model(ids, mask)

            risk_true.extend(batch["risk"].tolist())
            risk_pred.extend(risk.squeeze(-1).cpu().tolist())
            tone_true.append(batch["tone"].cpu().numpy())
            tone_pred.append(tone.cpu().numpy())
            intent_true.extend(batch["intent"].tolist())
            intent_pred.extend(intent.argmax(dim=-1).cpu().tolist())

    tone_true = np.vstack(tone_true)
    tone_pred = np.vstack(tone_pred)

    risk_mae = mean_absolute_error(risk_true, risk_pred)
    risk_r = pearsonr(risk_true, risk_pred)[0]

    tone_f1 = []
    for i in range(5):
        tone_f1.append(f1_score((tone_true[:, i] > 0.5).astype(int),
                                (tone_pred[:, i] > 0.5).astype(int), zero_division=0))
    tone_macro = float(np.mean(tone_f1))

    intent_acc = accuracy_score(intent_true, intent_pred)
    intent_macro = f1_score(intent_true, intent_pred, average="macro")

    return {
        "risk_mae": risk_mae,
        "risk_pearson_r": risk_r,
        "tone_macro_f1": tone_macro,
        "tone_per_label": {TONE_LABELS[i]: tone_f1[i] for i in range(5)},
        "intent_acc": intent_acc,
        "intent_macro_f1": intent_macro,
    }


def train() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_data = _load_jsonl(SPLITS_DIR / "train.jsonl")
    val_data = _load_jsonl(SPLITS_DIR / "val.jsonl")
    test_data = _load_jsonl(SPLITS_DIR / "test.jsonl")

    train_dl = DataLoader(MultiTaskDataset(train_data, tokenizer), batch_size=BATCH_SIZE, shuffle=True)
    val_dl = DataLoader(MultiTaskDataset(val_data, tokenizer), batch_size=BATCH_SIZE)
    test_dl = DataLoader(MultiTaskDataset(test_data, tokenizer), batch_size=BATCH_SIZE)

    model = MultiTaskModel().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS * len(train_dl))

    risk_loss = nn.MSELoss()
    tone_loss = nn.BCELoss()
    intent_loss = nn.CrossEntropyLoss()

    best_val = float("inf")
    t0 = time.time()

    for epoch in range(EPOCHS):
        model.train()
        total = 0.0
        for batch in train_dl:
            ids = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            risk_t = batch["risk"].to(DEVICE).unsqueeze(1)
            tone_t = batch["tone"].to(DEVICE)
            intent_t = batch["intent"].to(DEVICE)

            optimizer.zero_grad()
            risk, tone, intent = model(ids, mask)

            l_risk = risk_loss(risk, risk_t)
            l_tone = tone_loss(tone, tone_t)
            l_intent = intent_loss(intent, intent_t)
            loss = (LOSS_WEIGHTS["risk"] * l_risk
                    + LOSS_WEIGHTS["tone"] * l_tone
                    + LOSS_WEIGHTS["intent"] * l_intent)

            loss.backward()
            optimizer.step()
            scheduler.step()
            total += loss.item()

        # Validation
        val_metrics = evaluate(model, val_dl)
        val_combined = (
            val_metrics["risk_mae"]
            + (1 - val_metrics["tone_macro_f1"])
            + (1 - val_metrics["intent_acc"])
        )
        print(f"  Epoch {epoch + 1}: loss={total / len(train_dl):.4f} "
              f"val_risk_mae={val_metrics['risk_mae']:.4f} "
              f"val_tone_f1={val_metrics['tone_macro_f1']:.4f} "
              f"val_intent_acc={val_metrics['intent_acc']:.4f}")

        if val_combined < best_val:
            best_val = val_combined
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(model.encoder.state_dict(), MODELS_DIR / "multitask_encoder.pt")
            torch.save(model.risk_head.state_dict(), MODELS_DIR / "multitask_risk_head.pt")
            torch.save(model.tone_head.state_dict(), MODELS_DIR / "multitask_tone_head.pt")
            torch.save(model.intent_head.state_dict(), MODELS_DIR / "multitask_intent_head.pt")

    train_time = time.time() - t0

    # Reload best and evaluate on test
    model.encoder.load_state_dict(torch.load(MODELS_DIR / "multitask_encoder.pt", weights_only=True))
    model.risk_head.load_state_dict(torch.load(MODELS_DIR / "multitask_risk_head.pt", weights_only=True))
    model.tone_head.load_state_dict(torch.load(MODELS_DIR / "multitask_tone_head.pt", weights_only=True))
    model.intent_head.load_state_dict(torch.load(MODELS_DIR / "multitask_intent_head.pt", weights_only=True))
    model = model.to(DEVICE)

    test_metrics = evaluate(model, test_dl)
    test_metrics["train_time"] = train_time

    print("\n  === TEST RESULTS ===")
    print(f"  Risk:    MAE={test_metrics['risk_mae']:.4f} r={test_metrics['risk_pearson_r']:.4f}")
    print(f"  Tone:    macro F1={test_metrics['tone_macro_f1']:.4f}")
    for label, f1 in test_metrics["tone_per_label"].items():
        print(f"           {label}: {f1:.4f}")
    print(f"  Intent:  acc={test_metrics['intent_acc']:.4f} macro F1={test_metrics['intent_macro_f1']:.4f}")
    print(f"  Time:    {train_time:.0f}s")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "multitask.json", "w") as f:
        json.dump(test_metrics, f, indent=2)


if __name__ == "__main__":
    train()
