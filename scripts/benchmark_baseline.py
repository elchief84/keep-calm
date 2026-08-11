"""Baseline benchmark: test existing models on Keep Calm-style examples.

Measures how well off-the-shelf models perform on our task without any fine-tuning.
This gives us a concrete performance floor and helps quantify the gap we need to fill.

Two models tested:
1. unitary/toxic-bert — Jigsaw toxicity classifier (closest to our use case)
2. j-hartmann/emotion-english-distilroberta-base — emotion detection (tone proxy)
"""
from __future__ import annotations

import json
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from transformers import pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_EN = PROJECT_ROOT / "data" / "examples_en.jsonl"
EXAMPLES_IT = PROJECT_ROOT / "data" / "examples_it.jsonl"


def load_examples(path: Path) -> list[dict]:
    examples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def benchmark_toxicity(
    examples: list[dict],
    language: str,
) -> dict:
    """Evaluate unitary/toxic-bert on Keep Calm examples."""
    print(f"\n{'='*60}")
    print(f"Benckmark: unitary/toxic-bert on {language} examples ({len(examples)} msgs)")
    print(f"{'='*60}")

    pipe = pipeline(
        "text-classification",
        model="unitary/toxic-bert",
        device=-1,  # CPU
    )

    results = []
    latencies = []
    correct_level = 0
    total = 0

    for ex in examples:
        start = time.perf_counter()
        out = pipe(ex["text"])[0]
        elapsed = (time.perf_counter() - start) * 1000
        latencies.append(elapsed)

        toxicity_score = out["score"] if out["label"].lower() == "toxic" else 1.0 - out["score"]

        expected_risk = ex["expected_risk"]
        mae = abs(toxicity_score - expected_risk)

        # Map toxicity score to risk level for level accuracy
        if expected_risk < 0.2:
            expected_level = "none"
        elif expected_risk < 0.4:
            expected_level = "low"
        elif expected_risk < 0.6:
            expected_level = "medium"
        elif expected_risk < 0.8:
            expected_level = "high"
        else:
            expected_level = "critical"

        if toxicity_score < 0.2:
            pred_level = "none"
        elif toxicity_score < 0.4:
            pred_level = "low"
        elif toxicity_score < 0.6:
            pred_level = "medium"
        elif toxicity_score < 0.8:
            pred_level = "high"
        else:
            pred_level = "critical"

        if abs(
            ["none", "low", "medium", "high", "critical"].index(pred_level)
            - ["none", "low", "medium", "high", "critical"].index(expected_level)
        ) <= 1:
            correct_level += 1
        total += 1

        results.append({
            "id": ex["id"],
            "text": ex["text"],
            "expected_risk": expected_risk,
            "toxicity_score": toxicity_score,
            "mae": mae,
            "expected_level": expected_level,
            "pred_level": pred_level,
            "latency_ms": elapsed,
        })

    mean_latency = np.mean(latencies)
    mean_mae = np.mean([r["mae"] for r in results])
    pearson_r = np.corrcoef(
        [r["expected_risk"] for r in results],
        [r["toxicity_score"] for r in results],
    )[0, 1]
    level_accuracy = correct_level / total

    print(f"\n--- Results ---")
    print(f"Mean MAE:              {mean_mae:.3f}")
    print(f"Pearson r:             {pearson_r:.3f}")
    print(f"Level accuracy (+-1):  {level_accuracy:.1%} ({correct_level}/{total})")
    print(f"Mean latency:          {mean_latency:.1f}ms")
    print(f"Latency range:         {min(latencies):.1f}ms - {max(latencies):.1f}ms")

    # Show worst offenders
    results_sorted = sorted(results, key=lambda r: r["mae"], reverse=True)
    print(f"\n--- Top 5 worst predictions by MAE ---")
    for r in results_sorted[:5]:
        print(
            f"  [{r['id']}] Expected r={r['expected_risk']:.2f} ({r['expected_level']}), "
            f"Got r={r['toxicity_score']:.2f} ({r['pred_level']}), MAE={r['mae']:.3f}"
        )
        print(f"  Text: {r['text'][:100]}")

    # Breakdown by expected level
    print(f"\n--- Per-level MAE ---")
    for level in ["none", "low", "medium", "high", "critical"]:
        level_results = [r for r in results if r["expected_level"] == level]
        if level_results:
            level_mae = np.mean([r["mae"] for r in level_results])
            print(f"  {level:8s}: MAE={level_mae:.3f} (n={len(level_results)})")

    return {
        "language": language,
        "model": "unitary/toxic-bert",
        "mean_mae": mean_mae,
        "pearson_r": pearson_r,
        "level_accuracy": level_accuracy,
        "mean_latency_ms": mean_latency,
        "n_examples": total,
    }


def main() -> None:
    results = {}

    # English
    en_examples = load_examples(EXAMPLES_EN)
    results["en"] = benchmark_toxicity(en_examples, "en")

    # Italian
    it_examples = load_examples(EXAMPLES_IT)
    results["it"] = benchmark_toxicity(it_examples, "it")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for lang, r in results.items():
        print(f"\n{lang.upper()}: {r['model']}")
        print(f"  MAE: {r['mean_mae']:.3f} | r: {r['pearson_r']:.3f} | "
              f"Level acc: {r['level_accuracy']:.1%} | "
              f"Latency: {r['mean_latency_ms']:.0f}ms")

    print(f"\n--- Key questions this benchmark helps answer ---")
    print(
        "1. Gap size: How far is an off-the-shelf toxicity model from our expected scores?"
    )
    print(
        "2. Language gap: Does the model perform worse on Italian vs English?"
    )
    print(
        "3. Blind spots: Which risk levels does the model consistently misclassify?"
    )
    print(
        "4. Latency baseline: Is a HuggingFace pipeline on CPU fast enough?"
    )


if __name__ == "__main__":
    main()
