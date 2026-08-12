"""Convert annotated intent samples to training format, merge,
create new splits, and retrain the intent model.

Usage:
    python scripts/retrain_intent.py --all     # full pipeline
    python scripts/retrain_intent.py --convert # just convert & merge
    python scripts/retrain_intent.py --train   # just train
    python scripts/retrain_intent.py --eval    # just evaluate
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np

random.seed(42)
np.random.seed(42)


TONE_LABELS = ["neutral", "frustrated", "hostile", "sarcastic", "positive"]
INTENT_LABELS = ["constructive", "critical", "personal", "informational"]

DATA_DIR = Path("data")
SPLITS_DIR = DATA_DIR / "splits"
MODELS_DIR = DATA_DIR / "models"
RESULTS_DIR = DATA_DIR / "results"


def convert_annotated_to_flat(input_path: Path, use_generated_for: bool = True) -> list[dict]:
    """Convert auto_annotate.py format to training format.

    When use_generated_for is True, the `generated_for` field overrides the
    OpenAI-assigned intent label. This is intentional: we designed the templates
    for a specific class, and gpt-4o-mini systematically collapses `personal`
    into `critical`, defeating the rebalancing effort.
    """
    samples = []
    skipped = 0
    with open(input_path) as f:
        for line in f:
            r = json.loads(line)
            ann = r.get("annotations", {})
            intent = r.get("generated_for") if use_generated_for else ann.get("intent")
            if not intent or intent not in INTENT_LABELS:
                skipped += 1
                continue

            tones = ann.get("tones", [])
            tone_vector: list[float] = []
            for label in TONE_LABELS:
                match = [t for t in tones if t["label"] == label]
                tone_vector.append(match[0]["confidence"] if match else 0.0)

            risk = ann.get("communication_risk", 0.0)
            risk_level = ann.get("risk_level", "none")
            risk_level_idx = {"none": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}.get(
                risk_level, 0
            )

            samples.append({
                "id": r["id"],
                "text": r["text"],
                "language": r["language"],
                "source": r.get("source", "synthetic_generated"),
                "risk": risk,
                "risk_level": risk_level,
                "risk_level_idx": risk_level_idx,
                "tone_vector": tone_vector,
                "intent": intent,
            })
    print(f"  Converted {len(samples)} samples (skipped {skipped})")
    return samples


def convert_personal_to_flat(input_path: Path) -> list[dict]:
    """Convert unannotated hyper-personal samples using generated_for as truth.

    These messages are unambiguous personal attacks, so we assign a
    hostile+frustrated tone profile and high risk by construction.
    """
    samples = []
    with open(input_path) as f:
        for line in f:
            r = json.loads(line)
            # Tones: hostile 0.8, frustrated 0.6, neutral 0.1, others 0
            tone_vector = [0.1, 0.6, 0.8, 0.1, 0.0]
            samples.append({
                "id": r["id"],
                "text": r["text"],
                "language": r["language"],
                "source": r.get("source", "synthetic_generated"),
                "risk": 0.8,
                "risk_level": "critical",
                "risk_level_idx": 4,
                "tone_vector": tone_vector,
                "intent": "personal",
            })
    print(f"  Converted {len(samples)} personal samples")
    return samples


def convert_boundary_to_flat(input_path: Path) -> list[dict]:
    """Convert boundary critical/personal samples using generated_for as truth.

    Critical = negative on the WORK, personal = negative on the PERSON.
    Tone profile: neutral-ish with frustration for critical, hostile for personal.
    """
    samples = []
    with open(input_path) as f:
        for line in f:
            r = json.loads(line)
            intent = r.get("generated_for")
            if intent not in ("critical", "personal"):
                continue
            if intent == "personal":
                tone_vector = [0.1, 0.6, 0.8, 0.1, 0.0]
                risk = 0.75
                risk_level, risk_idx = "high", 3
            else:
                tone_vector = [0.5, 0.4, 0.1, 0.0, 0.0]
                risk = 0.55
                risk_level, risk_idx = "medium", 2
            samples.append({
                "id": r["id"],
                "text": r["text"],
                "language": r["language"],
                "source": r.get("source", "synthetic_generated"),
                "risk": risk,
                "risk_level": risk_level,
                "risk_level_idx": risk_idx,
                "tone_vector": tone_vector,
                "intent": intent,
            })
    print(f"  Converted {len(samples)} boundary samples")
    return samples


def merge_and_split(new_samples: list[dict]) -> tuple[list, list, list]:
    """Merge with existing training data, create new stratified splits."""
    # Load existing train
    existing = []
    with open(SPLITS_DIR / "train.jsonl") as f:
        for line in f:
            existing.append(json.loads(line))

    combined = existing + new_samples

    # Stratified shuffle by intent
    by_intent: dict[str, list] = {i: [] for i in INTENT_LABELS}
    for s in combined:
        by_intent[s["intent"]].append(s)

    train_new, val_new, test_new = [], [], []
    for intent in INTENT_LABELS:
        pool = by_intent[intent]
        random.shuffle(pool)
        n = len(pool)
        # 70/15/15 split
        n_train = int(n * 0.70)
        n_val = int(n * 0.15)
        train_new.extend(pool[:n_train])
        val_new.extend(pool[n_train:n_train + n_val])
        test_new.extend(pool[n_train + n_val:])

    random.shuffle(train_new)
    random.shuffle(val_new)
    random.shuffle(test_new)

    # Save new splits
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    for name, data in [("train", train_new), ("val", val_new), ("test", test_new)]:
        path = SPLITS_DIR / f"{name}.jsonl"
        with open(path, "w") as f:
            for s in data:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # Update dataset_info
    from collections import Counter
    intent_dist = Counter(s["intent"] for s in combined)
    risk_dist = Counter(s["risk_level"] for s in combined)
    info = {
        "total": len(combined),
        "train": len(train_new),
        "val": len(val_new),
        "test": len(test_new),
        "risk_distribution": dict(risk_dist),
        "intent_distribution": dict(intent_dist),
        "tone_labels": TONE_LABELS,
        "intent_labels": INTENT_LABELS,
        "sources": dict(Counter(s.get("source", "unknown") for s in combined)),
        "note": "augmented with synthetic intent samples July 2026",
    }
    with open(SPLITS_DIR / "dataset_info.json", "w") as f:
        json.dump(info, f, indent=2)

    print(f"  Train: {len(train_new)} ({len(existing)} original + {len(new_samples)} new)")
    print(f"  Val:   {len(val_new)}")
    print(f"  Test:  {len(test_new)}")
    for intent in INTENT_LABELS:
        t = intent_dist[intent]
        print(f"  {intent}: {t} total ({t/len(combined)*100:.1f}%)")

    return train_new, val_new, test_new


def train_intent_model() -> float:
    """Train intent classification model on the new splits."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModel, AutoTokenizer

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"  Using device: {device}")

    MODEL_NAME = "distilbert-base-multilingual-cased"
    BATCH_SIZE = 32 if device.type == "mps" else 16
    EPOCHS = 5
    LR = 2e-5

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    class TextDataset(Dataset):
        def __init__(self, data, tokenizer):
            self.texts = [d["text"] for d in data]
            self.labels = [INTENT_LABELS.index(d["intent"]) for d in data]
            self.tokenizer = tokenizer

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            enc = self.tokenizer(
                self.texts[idx], truncation=True, padding="max_length",
                max_length=256, return_tensors="pt",
            )
            return {
                "input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "labels": torch.tensor(self.labels[idx], dtype=torch.long),
            }

    # Load splits
    def load(path: Path) -> list[dict]:
        with open(path) as f:
            return [json.loads(l) for l in f]

    train_data = load(SPLITS_DIR / "train.jsonl")
    val_data = load(SPLITS_DIR / "val.jsonl")
    test_data = load(SPLITS_DIR / "test.jsonl")

    train_ds = TextDataset(train_data, tokenizer)
    val_ds = TextDataset(val_data, tokenizer)
    test_ds = TextDataset(test_data, tokenizer)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE)

    # Model
    encoder = AutoModel.from_pretrained(MODEL_NAME).to(device)
    head = nn.Sequential(
        nn.Linear(768, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, len(INTENT_LABELS)),
    ).to(device)

    optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(head.parameters()), lr=LR)
    loss_fn = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS * len(train_dl))

    best_val_acc = 0.0
    t0 = time.time()

    for epoch in range(EPOCHS):
        encoder.train()
        head.train()
        total_loss = 0.0
        for batch in train_dl:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            emb = encoder(ids, mask).last_hidden_state[:, 0, :]
            logits = head(emb)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        # Validation
        encoder.eval()
        head.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in val_dl:
                ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                emb = encoder(ids, mask).last_hidden_state[:, 0, :]
                logits = head(emb)
                preds = logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        val_acc = correct / total
        print(f"  Epoch {epoch+1}: loss={total_loss/len(train_dl):.4f} val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # Save
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(encoder.state_dict(), MODELS_DIR / "intent_encoder.pt")
            torch.save(head.state_dict(), MODELS_DIR / "intent_head.pt")

    train_time = time.time() - t0

    # Test evaluation
    encoder.load_state_dict(torch.load(MODELS_DIR / "intent_encoder.pt", weights_only=True))
    head.load_state_dict(torch.load(MODELS_DIR / "intent_head.pt", weights_only=True))
    encoder.to(device)
    head.to(device)
    encoder.eval()
    head.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_dl:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            emb = encoder(ids, mask).last_hidden_state[:, 0, :]
            logits = head(emb)
            all_preds.extend(logits.argmax(dim=-1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    from sklearn.metrics import accuracy_score, classification_report, f1_score

    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    per_class_f1 = f1_score(all_labels, all_preds, average=None)

    print(f"\n  Test Accuracy: {acc:.4f}")
    print(f"  Test Macro F1: {macro_f1:.4f}")
    print(f"  Train time: {train_time:.0f}s")
    print("\n  Per-class F1:")
    for i, label in enumerate(INTENT_LABELS):
        print(f"    {label}: {per_class_f1[i]:.4f}")

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result = {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "per_class_f1": {INTENT_LABELS[i]: float(per_class_f1[i]) for i in range(len(INTENT_LABELS))},
        "train_time": train_time,
        "train_samples": len(train_data),
        "val_samples": len(val_data),
        "test_samples": len(test_data),
    }
    with open(RESULTS_DIR / "intent_augmented.json", "w") as f:
        json.dump(result, f, indent=2)

    # Also print classification report for full picture
    print(f"\n{classification_report(all_labels, all_preds, target_names=INTENT_LABELS)}")

    return acc


def main():
    parser = argparse.ArgumentParser(description="Retrain intent model with augmented data")
    parser.add_argument("--all", action="store_true", help="Run full pipeline")
    parser.add_argument("--convert", action="store_true", help="Convert annotated data and create splits")
    parser.add_argument("--train", action="store_true", help="Train intent model")
    args = parser.parse_args()

    run_all = args.all or not any([args.convert, args.train])

    if run_all or args.convert:
        print("=== Converting annotated data and creating splits ===")
        en_samples = convert_annotated_to_flat(DATA_DIR / "intent_augment_en_annotated.jsonl")
        it_samples = convert_annotated_to_flat(DATA_DIR / "intent_augment_it_annotated.jsonl")
        en_personal = convert_personal_to_flat(DATA_DIR / "intent_personal_en.jsonl")
        it_personal = convert_personal_to_flat(DATA_DIR / "intent_personal_it.jsonl")
        boundary = convert_boundary_to_flat(DATA_DIR / "intent_boundary_scale.jsonl")
        all_new = en_samples + it_samples + en_personal + it_personal + boundary
        print(f"  Total new: {len(all_new)}")
        merge_and_split(all_new)

    if run_all or args.train:
        print("\n=== Training intent model ===")
        train_intent_model()


if __name__ == "__main__":
    main()
