"""Classical NLP baseline: TF-IDF + XGBoost for all 3 tasks.

Trains separate models for risk regression, tone multi-label, and intent classification.
Evaluates on test set and reports metrics.

Fast to train (~30s on CPU), serves as performance floor.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from sklearn.multioutput import MultiOutputClassifier
from xgboost import XGBClassifier, XGBRegressor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"

TONE_LABELS = ["neutral", "frustrated", "hostile", "sarcastic", "positive"]
INTENT_LABELS = ["constructive", "critical", "personal", "informational"]


def load_split(name: str) -> tuple[list, list, list, list, list]:
    texts, risks, tones, intents, levels = [], [], [], [], []
    with open(SPLITS_DIR / f"{name}.jsonl") as f:
        for line in f:
            ex = json.loads(line)
            texts.append(ex["text"])
            risks.append(ex["risk"])
            tones.append(ex["tone_vector"])
            intents.append(ex["intent"])
            levels.append(ex["risk_level"])
    return texts, risks, tones, intents, levels


def main() -> None:
    print("Loading data...")
    train_texts, train_risks, train_tones, train_intents, _ = load_split("train")
    val_texts, val_risks, val_tones, val_intents, _ = load_split("val")
    test_texts, test_risks, test_tones, test_intents, test_levels = load_split("test")

    print(f"Train: {len(train_texts)}, Val: {len(val_texts)}, Test: {len(test_texts)}")

    # TF-IDF vectorization
    print("\nVectorizing with TF-IDF...")
    t0 = time.time()
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), min_df=2, max_df=0.9)
    X_train = vectorizer.fit_transform(train_texts)
    vectorizer.transform(val_texts)
    X_test = vectorizer.transform(test_texts)
    print(f"  Done in {time.time() - t0:.1f}s. Features: {X_train.shape[1]}")

    results = {}

    # ---- Task 1: Risk Regression ----
    print("\n=== Task 1: Risk Regression (XGBoost) ===")
    t0 = time.time()
    risk_model = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
    risk_model.fit(X_train, train_risks)
    train_time = time.time() - t0

    train_pred = risk_model.predict(X_train)
    test_pred = risk_model.predict(X_test)
    train_mae = mean_absolute_error(train_risks, train_pred)
    test_mae = mean_absolute_error(test_risks, test_pred)
    pearson = pearsonr(test_risks, test_pred)[0]

    # Level accuracy
    def to_level(score):
        if score < 0.2: return 0
        if score < 0.4: return 1
        if score < 0.6: return 2
        if score < 0.8: return 3
        return 4

    test_levels_idx = [to_level(r) for r in test_risks]
    pred_levels_idx = [to_level(p) for p in test_pred]
    level_acc = sum(1 for a, b in zip(test_levels_idx, pred_levels_idx, strict=False) if abs(a-b) <= 1) / len(test_risks)

    print(f"  Train MAE: {train_mae:.4f} | Test MAE: {test_mae:.4f}")
    print(f"  Pearson r: {pearson:.4f} | Level acc (+-1): {level_acc:.2%} | Time: {train_time:.1f}s")
    results["risk"] = {"mae": test_mae, "pearson_r": pearson, "level_acc": level_acc}

    # ---- Task 2: Tone Multi-label ----
    print("\n=== Task 2: Tone Multi-label (Logistic Regression) ===")
    t0 = time.time()
    tone_model = MultiOutputClassifier(LogisticRegression(max_iter=500, random_state=42))
    tone_model.fit(X_train, train_tones)
    train_time = time.time() - t0

    tone_model.predict(X_train)
    test_tone_pred = tone_model.predict(X_test)

    print("  Per-label F1 (test):")
    per_label_f1 = []
    for i, label in enumerate(TONE_LABELS):
        f1 = f1_score([t[i] for t in test_tones], [p[i] for p in test_tone_pred], zero_division=0)
        support = sum(t[i] for t in test_tones)
        print(f"    {label:12s}: F1={f1:.3f} (support={int(support)})")
        per_label_f1.append(f1)
    macro_f1 = np.mean(per_label_f1)
    print(f"  Macro F1: {macro_f1:.4f} | Time: {train_time:.1f}s")
    results["tone"] = {"macro_f1": macro_f1, "per_label": dict(zip(TONE_LABELS, per_label_f1, strict=False))}

    # ---- Task 3: Intent Classification ----
    print("\n=== Task 3: Intent Classification (XGBoost) ===")
    t0 = time.time()
    intent_map = {l: i for i, l in enumerate(INTENT_LABELS)}
    train_intent_idx = [intent_map[i] for i in train_intents]
    test_intent_idx = [intent_map[i] for i in test_intents]

    intent_model = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
    intent_model.fit(X_train, train_intent_idx)
    train_time = time.time() - t0

    train_intent_pred = intent_model.predict(X_train)
    test_intent_pred = intent_model.predict(X_test)

    train_acc = accuracy_score(train_intent_idx, train_intent_pred)
    test_acc = accuracy_score(test_intent_idx, test_intent_pred)
    macro_f1_i = f1_score(test_intent_idx, test_intent_pred, average="macro")

    print(f"  Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f} | Macro F1: {macro_f1_i:.4f} | Time: {train_time:.1f}s")
    results["intent"] = {"accuracy": test_acc, "macro_f1": macro_f1_i}

    # ---- Summary ----
    print(f"\n{'='*60}")
    print("BASELINE SUMMARY")
    print(f"{'='*60}")
    print(f"Risk:    MAE={results['risk']['mae']:.4f}  r={results['risk']['pearson_r']:.4f}  LevelAcc={results['risk']['level_acc']:.2%}")
    print(f"Tone:    Macro F1={results['tone']['macro_f1']:.4f}")
    print(f"Intent:  Acc={results['intent']['accuracy']:.4f}  Macro F1={results['intent']['macro_f1']:.4f}")

    # Save results
    results_dir = PROJECT_ROOT / "data" / "results"
    results_dir.mkdir(exist_ok=True)
    with open(results_dir / "baseline_classical.json", "w") as f:
        json.dump({k: {kk: float(vv) if isinstance(vv, (np.floating, float)) else vv for kk, vv in v.items()} for k, v in results.items()}, f, indent=2)
    print(f"\nResults saved to {results_dir / 'baseline_classical.json'}")


if __name__ == "__main__":
    main()
