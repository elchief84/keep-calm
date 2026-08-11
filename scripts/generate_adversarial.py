"""Generate adversarial examples targeting medium/high/critical risk.

Uses OpenAI to create examples that are hard to classify:
- Passive-aggressive professional communication
- Condescending but technically polite
- Subtle sarcasm
- Direct communication that could be perceived as hostile
- Non-native speaker communication patterns

Output: data/adversarial_en.jsonl + data/adversarial_it.jsonl with AUTO-ANNOTATED labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_EN = PROJECT_ROOT / "data" / "adversarial_en.jsonl"
OUTPUT_IT = PROJECT_ROOT / "data" / "adversarial_it.jsonl"

OPENAI_URL = "https://api.openai.com/v1/chat/completions"
MODEL = "gpt-4o-mini"


def call_openai(system: str, prompt: str, api_key: str, long: bool = False) -> str:
    resp = requests.post(
        OPENAI_URL,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.9,
            "max_tokens": 4000 if long else 500,
            "response_format": {"type": "json_object"} if not long else None,
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def generate_adversarial_batch(api_key: str, language: str, count: int, categories: dict) -> list[dict]:
    system = f"""You generate training data for workplace communication analysis. Language: {language}.

Generate {count} diverse workplace chat messages (Slack/Teams style). Categories:
{json.dumps(categories, indent=2)}

Each message: 10-80 words, realistic (slang, fragments, typos, mixed case OK).
Diverse roles: engineer, manager, designer, QA, intern.
Balance: medium risk 40%, high risk 40%, critical risk 20%.
Include edge cases: direct but constructive, frustrated but helpful, ambiguous.

Output ONLY a JSON array: [{{"text": "...", "category": "...", "risk_level": "medium|high|critical", "context": "scene description"}}]"""

    resp = requests.post(
        OPENAI_URL,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"model": MODEL, "temperature": 0.9, "max_tokens": 4000,
              "messages": [{"role": "system", "content": system},
                           {"role": "user", "content": f"Generate {count} messages."}],
              "response_format": {"type": "json_object"}},
        timeout=120,
    )
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]
    data = json.loads(content)
    if isinstance(data, dict):
        data = list(data.values())[0] if data else []
    if not isinstance(data, list):
        return []
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate adversarial examples")
    parser.add_argument("--api-key", required=True, help="OpenAI API key")
    parser.add_argument("--count", type=int, default=500, help="Examples per language")
    parser.add_argument("--batch-size", type=int, default=80, help="Per API call")
    args = parser.parse_args()

    categories = {
        "passive_aggressive": "Polite wording, hostile intent. 'as I'm sure you're aware', 'not surprised', 'some of us actually...'",
        "condescending": "Talks down while polite. 'Let me explain simply', 'for those who might not understand', 'per my last email'",
        "subtle_sarcasm": "Could read as sincere. 'Oh wonderful, another meeting', 'Sure, let's do it your way then'",
        "direct_blunt": "Direct, concise, could be aggressive. 'This is wrong', 'Fix this', no sugar-coating",
        "frustration_constructive": "Frustrated but helpful. 'I'm tired of this, here's a fix'",
        "non_native": "Grammar quirks that could be misread as abrupt. Simple structures, missing articles",
    }

    total = 0
    for lang, output in [("English", OUTPUT_EN), ("Italian", OUTPUT_IT)]:
        print(f"\nGenerating {lang} adversarial examples...")
        all_examples = []
        remaining = args.count
        batch = 0

        while remaining > 0:
            n = min(args.batch_size, remaining)
            print(f"  Batch {batch+1}: requesting {n}...", end=" ", flush=True)
            try:
                items = generate_adversarial_batch(args.api_key, lang, n, categories)
            except Exception as e:
                print(f"Error: {e}")
                time.sleep(3)
                continue

            for item in items:
                text = item.get("text", "").strip()
                if not text or len(text) < 10:
                    continue
                all_examples.append({
                    "id": f"adv-{hashlib.sha256(text.encode()).hexdigest()[:12]}",
                    "text": text,
                    "language": lang.lower()[:2],
                    "domain": f"adversarial_{item.get('category', 'unknown')}",
                    "source": "openai_generated",
                    "context": item.get("context", "Workplace chat"),
                    "metadata": {"category": item.get("category"), "target_risk": item.get("risk_level")},
                    "annotations": {"communication_risk": None, "tones": [], "intent": None, "explanation": None, "needs_attention": None},
                })

            print(f"got {len(items)} ({len(all_examples)} total)")
            remaining -= n
            batch += 1
            if remaining > 0:
                time.sleep(1)

        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            for ex in all_examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

        total += len(all_examples)
        print(f"  {len(all_examples)} adversarial examples -> {output}")

    print(f"\nAdversarial: {total} total examples")


if __name__ == "__main__":
    main()
