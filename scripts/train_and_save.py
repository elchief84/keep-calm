"""Train and save DistilBERT models for the Keep Calm MVP.

Trains risk, tone, and intent models, evaluates them, and saves to data/models/.
Also saves tokenizer and configuration.
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
from transformers import AutoTokenizer, AutoModel


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
MODELS_DIR = PROJECT_ROOT / "data" / "models"

TONE_LABELS = ["neutral", "frustrated", "hostile", "sarcastic", "positive"]
INTENT_LABELS = ["constructive", "critical", "personal", "informational"]
INTENT_MAP = {l: i for i, l in enumerate(INTENT_LABELS)}

MODEL_NAME = "distilbert-base-multilingual-cased"
BATCH_SIZE = 16
EPOCHS = 1
LR = 2e-5
DEVICE = torch.device("cpu")


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx], truncation=True, padding="max_length",
            max_length=self.max_len, return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.float),
        }


def load_splits():
    data = {}
    for split in ["train", "val", "test"]:
        texts, risks, tones, intents = [], [], [], []
        with open(SPLITS_DIR / f"{split}.jsonl") as f:
            for line in f:
                ex = json.loads(line)
                texts.append(ex["text"])
                risks.append(ex["risk"])
                tones.append(ex["tone_vector"])
                intents.append(INTENT_MAP[ex["intent"]])
        data[split] = (texts, risks, tones, intents)
    return data


def to_level(score):
    if score < 0.2: return 0
    if score < 0.4: return 1
    if score < 0.6: return 2
    if score < 0.8: return 3
    return 4


def train_risk(data, tokenizer):
    print("\n=== Training: Risk Regression ===")
    model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
    head = nn.Sequential(nn.Linear(768, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 1), nn.Sigmoid()).to(DEVICE)
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()), lr=LR)
    loss_fn = nn.MSELoss()

    train_ds = TextDataset(data["train"][0], [[r] for r in data["train"][1]], tokenizer)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    t0 = time.time()
    for epoch in range(EPOCHS):
        model.train(); head.train()
        total_loss = 0
        for batch in train_dl:
            ids, mask, labels = batch["input_ids"].to(DEVICE), batch["attention_mask"].to(DEVICE), batch["labels"].to(DEVICE)
            emb = model(ids, mask).last_hidden_state[:, 0, :]
            pred = head(emb)
            loss = loss_fn(pred, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
        print(f"  Epoch {epoch+1}: {total_loss/len(train_dl):.4f}")

    model.eval(); head.eval()
    test_ds = TextDataset(data["test"][0], [[r] for r in data["test"][1]], tokenizer)
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE)
    preds, actual = [], []
    with torch.no_grad():
        for batch in test_dl:
            ids, mask, labels = batch["input_ids"].to(DEVICE), batch["attention_mask"].to(DEVICE), batch["labels"].to(DEVICE)
            emb = model(ids, mask).last_hidden_state[:, 0, :]
            preds.extend(head(emb).squeeze(-1).cpu().numpy().tolist())
            actual.extend(labels.squeeze(-1).cpu().numpy().tolist())

    mae = mean_absolute_error(actual, preds)
    r = pearsonr(actual, preds)[0]
    la = sum(1 for a, b in zip(map(to_level, actual), map(to_level, preds)) if abs(a-b) <= 1) / len(actual)
    print(f"  Test: MAE={mae:.4f} r={r:.4f} LevelAcc={la:.2%} ({time.time()-t0:.0f}s)")
    return model, head, {"mae": mae, "pearson_r": r, "level_acc": la}


def train_tone(data, tokenizer):
    print("\n=== Training: Tone Multi-label ===")
    model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
    head = nn.Sequential(nn.Linear(768, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 5), nn.Sigmoid()).to(DEVICE)
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()), lr=LR)
    loss_fn = nn.BCELoss()

    train_ds = TextDataset(data["train"][0], data["train"][2], tokenizer)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    t0 = time.time()
    for epoch in range(EPOCHS):
        model.train(); head.train()
        total_loss = 0
        for batch in train_dl:
            ids, mask, labels = batch["input_ids"].to(DEVICE), batch["attention_mask"].to(DEVICE), batch["labels"].to(DEVICE)
            emb = model(ids, mask).last_hidden_state[:, 0, :]
            pred = head(emb)
            loss = loss_fn(pred, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
        print(f"  Epoch {epoch+1}: {total_loss/len(train_dl):.4f}")

    model.eval(); head.eval()
    test_ds = TextDataset(data["test"][0], data["test"][2], tokenizer)
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE)
    preds, actual = [], []
    with torch.no_grad():
        for batch in test_dl:
            ids, mask, labels = batch["input_ids"].to(DEVICE), batch["attention_mask"].to(DEVICE), batch["labels"].to(DEVICE)
            emb = model(ids, mask).last_hidden_state[:, 0, :]
            preds.extend(head(emb).cpu().numpy().tolist())
            actual.extend(labels.cpu().numpy().tolist())

    preds_bin = (np.array(preds) >= 0.3).astype(int)
    per_label = {}
    for i, label in enumerate(TONE_LABELS):
        per_label[label] = float(f1_score(np.array(actual)[:, i], preds_bin[:, i], zero_division=0))
        print(f"    {label:12s}: F1={per_label[label]:.3f} (s={int(np.array(actual)[:, i].sum())})")
    macro_f1 = np.mean(list(per_label.values()))
    print(f"  Macro F1: {macro_f1:.4f} ({time.time()-t0:.0f}s)")
    return model, head, {"macro_f1": macro_f1, "per_label": per_label}


def train_intent(data, tokenizer):
    print("\n=== Training: Intent Classification ===")
    model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
    head = nn.Sequential(nn.Linear(768, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 4)).to(DEVICE)
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    train_ds = TextDataset(data["train"][0], [[i] for i in data["train"][3]], tokenizer)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    t0 = time.time()
    for epoch in range(EPOCHS):
        model.train(); head.train()
        total_loss = 0
        for batch in train_dl:
            ids, mask, labels = batch["input_ids"].to(DEVICE), batch["attention_mask"].to(DEVICE), batch["labels"].to(DEVICE)
            emb = model(ids, mask).last_hidden_state[:, 0, :]
            logits = head(emb)
            loss = loss_fn(logits, labels.squeeze(1).long())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
        print(f"  Epoch {epoch+1}: {total_loss/len(train_dl):.4f}")

    model.eval(); head.eval()
    test_ds = TextDataset(data["test"][0], [[i] for i in data["test"][3]], tokenizer)
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE)
    preds, actual = [], []
    with torch.no_grad():
        for batch in test_dl:
            ids, mask, labels = batch["input_ids"].to(DEVICE), batch["attention_mask"].to(DEVICE), batch["labels"].to(DEVICE)
            emb = model(ids, mask).last_hidden_state[:, 0, :]
            preds.extend(head(emb).argmax(-1).cpu().numpy().tolist())
            actual.extend(labels.squeeze(-1).cpu().numpy().tolist())

    acc = float(accuracy_score(actual, preds))
    mf1 = float(f1_score(actual, preds, average="macro"))
    print(f"  Test: Acc={acc:.4f} Macro F1={mf1:.4f} ({time.time()-t0:.0f}s)")
    return model, head, {"accuracy": acc, "macro_f1": mf1}


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    data = load_splits()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.save_pretrained(str(MODELS_DIR))

    print(f"Training on {DEVICE} with {len(data['train'][0])} examples ({EPOCHS} epochs)")

    results = {}

    risk_model, risk_head, risk_metrics = train_risk(data, tokenizer)
    results["risk"] = risk_metrics
    torch.save(risk_model.state_dict(), MODELS_DIR / "risk_encoder.pt")
    torch.save(risk_head.state_dict(), MODELS_DIR / "risk_head.pt")

    tone_model, tone_head, tone_metrics = train_tone(data, tokenizer)
    results["tone"] = tone_metrics
    torch.save(tone_model.state_dict(), MODELS_DIR / "tone_encoder.pt")
    torch.save(tone_head.state_dict(), MODELS_DIR / "tone_head.pt")

    intent_model, intent_head, intent_metrics = train_intent(data, tokenizer)
    results["intent"] = intent_metrics
    torch.save(intent_model.state_dict(), MODELS_DIR / "intent_encoder.pt")
    torch.save(intent_head.state_dict(), MODELS_DIR / "intent_head.pt")

    config = {
        "model_name": MODEL_NAME,
        "tone_labels": TONE_LABELS,
        "intent_labels": INTENT_LABELS,
        "risk_thresholds": {"none": 0.2, "low": 0.4, "medium": 0.6, "high": 0.8, "critical": 1.0},
        "tone_threshold": 0.3,
    }
    with open(MODELS_DIR / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    with open(MODELS_DIR / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print("MODELS SAVED")
    print(f"{'='*60}")
    print(f"Risk:    MAE={risk_metrics['mae']:.4f}  r={risk_metrics['pearson_r']:.4f}")
    print(f"Tone:    Macro F1={tone_metrics['macro_f1']:.4f}")
    print(f"Intent:  Acc={intent_metrics['accuracy']:.4f}  F1={intent_metrics['macro_f1']:.4f}")
    print(f"Path:    {MODELS_DIR}")


if __name__ == "__main__":
    main()
