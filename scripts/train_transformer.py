"""Transformer model training: DistilBERT single-task models.

Trains 3 independent models for risk regression, tone detection, and intent classification.
Each uses a shared DistilBERT encoder with a task-specific head.

Evaluates on test set and benchmarks inference latency.
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
RESULTS_DIR = PROJECT_ROOT / "data" / "results"

TONE_LABELS = ["neutral", "frustrated", "hostile", "sarcastic", "positive"]
INTENT_LABELS = ["constructive", "critical", "personal", "informational"]
INTENT_MAP = {l: i for i, l in enumerate(INTENT_LABELS)}

MODEL_NAME = "distilbert-base-uncased"
BATCH_SIZE = 16
EPOCHS = 2
LR = 2e-5
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
TRAIN_DEVICE = torch.device("cpu")  # MPS doesn't support SDPA dropout; train on CPU


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
    print("\n=== Task: Risk Regression ===")
    model = AutoModel.from_pretrained(MODEL_NAME).to(TRAIN_DEVICE)
    head = nn.Sequential(nn.Linear(768, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 1), nn.Sigmoid()).to(TRAIN_DEVICE)
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()), lr=LR)
    loss_fn = nn.MSELoss()

    train_ds = TextDataset(data["train"][0], [[r] for r in data["train"][1]], tokenizer)
    val_ds = TextDataset(data["val"][0], [[r] for r in data["val"][1]], tokenizer)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE)

    t0 = time.time()
    for epoch in range(EPOCHS):
        model.train(); head.train()
        total_loss = 0
        for batch in train_dl:
            ids, mask, labels = batch["input_ids"].to(TRAIN_DEVICE), batch["attention_mask"].to(TRAIN_DEVICE), batch["labels"].to(TRAIN_DEVICE)
            emb = model(ids, mask).last_hidden_state[:, 0, :]
            pred = head(emb)
            loss = loss_fn(pred, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
        print(f"  Epoch {epoch+1}: loss={total_loss/len(train_dl):.4f}")

    train_time = time.time() - t0

    # Evaluate
    model.eval(); head.eval()
    test_ds = TextDataset(data["test"][0], [[r] for r in data["test"][1]], tokenizer)
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE)
    test_preds, test_actual = [], []
    with torch.no_grad():
        for batch in test_dl:
            ids, mask, labels = batch["input_ids"].to(TRAIN_DEVICE), batch["attention_mask"].to(TRAIN_DEVICE), batch["labels"].to(TRAIN_DEVICE)
            emb = model(ids, mask).last_hidden_state[:, 0, :]
            pred = head(emb).squeeze(-1)
            test_preds.extend(pred.cpu().numpy().tolist())
            test_actual.extend(labels.squeeze(-1).cpu().numpy().tolist())

    mae = mean_absolute_error(test_actual, test_preds)
    r = pearsonr(test_actual, test_preds)[0]
    level_acc = sum(1 for a, b in zip(map(to_level, test_actual), map(to_level, test_preds)) if abs(a-b) <= 1) / len(test_actual)

    print(f"  Test: MAE={mae:.4f} | r={r:.4f} | LevelAcc={level_acc:.2%} | Time={train_time:.0f}s")
    return {"mae": mae, "pearson_r": r, "level_acc": level_acc, "train_time": train_time}


def train_tone(data, tokenizer):
    print("\n=== Task: Tone Multi-label ===")
    model = AutoModel.from_pretrained(MODEL_NAME).to(TRAIN_DEVICE)
    head = nn.Sequential(nn.Linear(768, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 5), nn.Sigmoid()).to(TRAIN_DEVICE)
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()), lr=LR)
    loss_fn = nn.BCELoss()

    train_ds = TextDataset(data["train"][0], data["train"][2], tokenizer)
    val_ds = TextDataset(data["val"][0], data["val"][2], tokenizer)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    t0 = time.time()
    for epoch in range(EPOCHS):
        model.train(); head.train()
        total_loss = 0
        for batch in train_dl:
            ids, mask, labels = batch["input_ids"].to(TRAIN_DEVICE), batch["attention_mask"].to(TRAIN_DEVICE), batch["labels"].to(TRAIN_DEVICE)
            emb = model(ids, mask).last_hidden_state[:, 0, :]
            pred = head(emb)
            loss = loss_fn(pred, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
        print(f"  Epoch {epoch+1}: loss={total_loss/len(train_dl):.4f}")

    train_time = time.time() - t0

    model.eval(); head.eval()
    test_ds = TextDataset(data["test"][0], data["test"][2], tokenizer)
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE)
    test_preds, test_actual = [], []
    with torch.no_grad():
        for batch in test_dl:
            ids, mask, labels = batch["input_ids"].to(TRAIN_DEVICE), batch["attention_mask"].to(TRAIN_DEVICE), batch["labels"].to(TRAIN_DEVICE)
            emb = model(ids, mask).last_hidden_state[:, 0, :]
            pred = head(emb)
            test_preds.extend(pred.cpu().numpy().tolist())
            test_actual.extend(labels.cpu().numpy().tolist())

    test_preds_bin = (np.array(test_preds) >= 0.3).astype(int)
    per_label = {}
    for i, label in enumerate(TONE_LABELS):
        f1 = f1_score(np.array(test_actual)[:, i], test_preds_bin[:, i], zero_division=0)
        print(f"    {label:12s}: F1={f1:.3f} (support={int(np.array(test_actual)[:, i].sum())})")
        per_label[label] = f1
    macro_f1 = np.mean(list(per_label.values()))
    print(f"  Macro F1: {macro_f1:.4f} | Time: {train_time:.0f}s")
    return {"macro_f1": macro_f1, "per_label": per_label, "train_time": train_time}


def train_intent(data, tokenizer):
    print("\n=== Task: Intent Classification ===")
    model = AutoModel.from_pretrained(MODEL_NAME).to(TRAIN_DEVICE)
    head = nn.Sequential(nn.Linear(768, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 4)).to(TRAIN_DEVICE)
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    train_labels = [torch.tensor([i], dtype=torch.long) for i in data["train"][3]]
    val_labels = [torch.tensor([i], dtype=torch.long) for i in data["val"][3]]
    test_labels = [torch.tensor([i], dtype=torch.long) for i in data["test"][3]]

    train_ds = TextDataset(data["train"][0], [[i] for i in data["train"][3]], tokenizer)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    t0 = time.time()
    for epoch in range(EPOCHS):
        model.train(); head.train()
        total_loss = 0
        for batch in train_dl:
            ids, mask, labels = batch["input_ids"].to(TRAIN_DEVICE), batch["attention_mask"].to(TRAIN_DEVICE), batch["labels"].to(TRAIN_DEVICE)
            emb = model(ids, mask).last_hidden_state[:, 0, :]
            logits = head(emb)
            loss = loss_fn(logits, labels.squeeze(1).long())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
        print(f"  Epoch {epoch+1}: loss={total_loss/len(train_dl):.4f}")

    train_time = time.time() - t0

    model.eval(); head.eval()
    test_ds = TextDataset(data["test"][0], [[i] for i in data["test"][3]], tokenizer)
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE)
    test_preds, test_actual = [], []
    with torch.no_grad():
        for batch in test_dl:
            ids, mask, labels = batch["input_ids"].to(TRAIN_DEVICE), batch["attention_mask"].to(TRAIN_DEVICE), batch["labels"].to(TRAIN_DEVICE)
            emb = model(ids, mask).last_hidden_state[:, 0, :]
            logits = head(emb)
            test_preds.extend(logits.argmax(-1).cpu().numpy().tolist())
            test_actual.extend(labels.squeeze(-1).cpu().numpy().tolist())

    acc = accuracy_score(test_actual, test_preds)
    macro_f1 = f1_score(test_actual, test_preds, average="macro")
    print(f"  Test: Acc={acc:.4f} | Macro F1={macro_f1:.4f} | Time: {train_time:.0f}s")
    return {"accuracy": acc, "macro_f1": macro_f1, "train_time": train_time}


def benchmark_latency(model, head, tokenizer, texts, n=50):
    latencies = []
    for text in texts[:n]:
        enc = tokenizer(text, truncation=True, padding="max_length", max_length=256, return_tensors="pt")
        ids, mask = enc["input_ids"].to(DEVICE), enc["attention_mask"].to(DEVICE)
        start = time.perf_counter()
        with torch.no_grad():
            emb = model(ids, mask).last_hidden_state[:, 0, :]
            head(emb)
        latencies.append((time.perf_counter() - start) * 1000)
    return np.mean(latencies)


def main() -> None:
    print(f"Device: {DEVICE}")
    print(f"Model: {MODEL_NAME}")

    data = load_splits()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    results = {}
    results["risk"] = train_risk(data, tokenizer)
    results["tone"] = train_tone(data, tokenizer)
    results["intent"] = train_intent(data, tokenizer)

    # Latency benchmark
    print(f"\n=== Latency Benchmark ===")
    model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)  # Use MPS for inference
    head = nn.Sequential(nn.Linear(768, 1), nn.Sigmoid()).to(DEVICE)
    lat = benchmark_latency(model, head, tokenizer, data["test"][0])
    print(f"  Mean latency: {lat:.1f}ms on {DEVICE}")

    # Summary
    print(f"\n{'='*60}")
    print("TRANSFORMER RESULTS (DistilBERT single-task)")
    print(f"{'='*60}")
    print(f"Risk:   MAE={results['risk']['mae']:.4f}  r={results['risk']['pearson_r']:.4f}  LevelAcc={results['risk']['level_acc']:.2%}")
    print(f"Tone:   Macro F1={results['tone']['macro_f1']:.4f}")
    print(f"Intent: Acc={results['intent']['accuracy']:.4f}  Macro F1={results['intent']['macro_f1']:.4f}")
    print(f"Latency: {lat:.0f}ms")

    # Save
    RESULTS_DIR.mkdir(exist_ok=True)
    results["latency_ms"] = lat

    def to_serializable(obj):
        if isinstance(obj, dict):
            return {k: to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        return obj

    with open(RESULTS_DIR / "transformer_distilbert.json", "w") as f:
        json.dump(to_serializable(results), f, indent=2)
    print(f"\nSaved to {RESULTS_DIR / 'transformer_distilbert.json'}")


if __name__ == "__main__":
    main()
