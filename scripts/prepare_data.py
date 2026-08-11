"""Merge all annotated sources into train/val/test splits for Phase 2 training.

Reads all *_annotated.jsonl files, extracts structured labels, and creates
stratified splits saved as JSONL for both classical and transformer training.

Output:
    data/splits/train.jsonl, val.jsonl, test.jsonl
    data/splits/dataset_info.json
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
SPLITS_DIR = DATA_DIR / "splits"

ANNOTATED_FILES = [
    "youtube_en_annotated.jsonl",
    "youtube_it_annotated.jsonl",
    "github_en_annotated.jsonl",
    "adversarial_en_annotated.jsonl",
    "adversarial_it_annotated.jsonl",
    "adversarial_en_r2_annotated.jsonl",
    "adversarial_it_r2_annotated.jsonl",
    "adversarial_en_r3_annotated.jsonl",
    "adversarial_it_r3_annotated.jsonl",
    "idioms_en_annotated.jsonl",
    "idioms_it_annotated.jsonl",
    "sarcasm_en_annotated.jsonl",
]

TONE_LABELS = ["neutral", "frustrated", "hostile", "sarcastic", "positive"]
INTENT_LABELS = ["constructive", "critical", "personal", "informational"]


def load_all() -> list[dict]:
    examples = []
    for filename in ANNOTATED_FILES:
        path = DATA_DIR / filename
        if not path.exists():
            print(f"  SKIP: {filename} not found")
            continue
        with open(path) as f:
            for line in f:
                ex = json.loads(line)
                ann = ex.get("annotations", {})

                risk = ann.get("communication_risk", 0.1)
                if risk is None:
                    risk = 0.1

                level = ann.get("risk_level", "none")
                if level == "low":
                    level_idx = 1
                elif level == "medium":
                    level_idx = 2
                elif level == "high":
                    level_idx = 3
                elif level == "critical":
                    level_idx = 4
                else:
                    level_idx = 0

                tones = ann.get("tones", [])
                tone_vec = [0.0] * len(TONE_LABELS)
                for t in tones:
                    if isinstance(t, dict):
                        label = t.get("label", "")
                        conf = t.get("confidence", 0.0)
                        if label in TONE_LABELS and conf >= 0.3:
                            tone_vec[TONE_LABELS.index(label)] = 1.0

                intent = ann.get("intent", "informational")
                if intent not in INTENT_LABELS:
                    intent = "informational"

                examples.append({
                    "id": ex.get("id", ""),
                    "text": ex.get("text", ""),
                    "language": ex.get("language", "en"),
                    "source": ex.get("source", "unknown"),
                    "risk": float(risk),
                    "risk_level": level,
                    "risk_level_idx": level_idx,
                    "tone_vector": tone_vec,
                    "intent": intent,
                })

        print(f"  {filename}: {len(examples)} total")

    return examples


def main() -> None:
    print("Loading annotated data...")
    examples = load_all()
    print(f"\nTotal examples: {len(examples)}")

    # Distribution check
    levels = Counter(ex["risk_level"] for ex in examples)
    intents = Counter(ex["intent"] for ex in examples)
    print(f"Risk levels: {dict(levels)}")
    print(f"Intents: {dict(intents)}")

    # Stratify by risk_level for balanced splits
    risks = np.array([ex["risk_level_idx"] for ex in examples])
    texts = [ex["text"] for ex in examples]

    # 70% train, 15% val, 15% test
    train_idx, temp_idx = train_test_split(
        range(len(examples)), test_size=0.30, stratify=risks, random_state=42
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.50, stratify=risks[train_idx[temp_idx]] if False else risks[temp_idx],
        random_state=42,
    )
    # Redo properly
    train_idx, temp_idx = train_test_split(
        range(len(examples)), test_size=0.30, stratify=risks, random_state=42
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.50, stratify=risks[temp_idx], random_state=42,
    )

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    for name, indices in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        subset = [examples[i] for i in indices]
        path = SPLITS_DIR / f"{name}.jsonl"
        with open(path, "w") as f:
            for ex in subset:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"  {name}: {len(subset)} examples -> {path}")

    with open(SPLITS_DIR / "dataset_info.json", "w") as f:
        json.dump({
            "total": len(examples),
            "train": len(train_idx),
            "val": len(val_idx),
            "test": len(test_idx),
            "risk_distribution": dict(levels),
            "intent_distribution": dict(intents),
            "tone_labels": TONE_LABELS,
            "intent_labels": INTENT_LABELS,
            "sources": dict(Counter(ex["source"] for ex in examples)),
        }, f, indent=2)

    print(f"\nSplits saved to {SPLITS_DIR}/")


if __name__ == "__main__":
    main()
