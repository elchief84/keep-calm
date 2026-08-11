"""Auto-annotate YouTube comments using OpenAI API.

Reads unlabeled JSONL files, sends each comment to OpenAI with the Keep Calm
annotation guidelines, and writes an annotated JSONL output.

Usage:
    python scripts/auto_annotate.py --api-key YOUR_OPENAI_KEY
    python scripts/auto_annotate.py --api-key KEY --language it --max 30
    python scripts/auto_annotate.py --api-key KEY --resume
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
YOUTUBE_EN = PROJECT_ROOT / "data" / "youtube_en.jsonl"
YOUTUBE_IT = PROJECT_ROOT / "data" / "youtube_it.jsonl"
OUTPUT_EN = PROJECT_ROOT / "data" / "youtube_en_annotated.jsonl"
OUTPUT_IT = PROJECT_ROOT / "data" / "youtube_it_annotated.jsonl"
CHECKPOINT_PATH = PROJECT_ROOT / "data" / "annotation_checkpoint.json"

OPENAI_URL = "https://api.openai.com/v1/chat/completions"
MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = """You label text for communication risk analysis. Output ONLY valid JSON matching this schema:

{
  "communication_risk": <float 0-1>,
  "risk_level": "none" | "low" | "medium" | "high" | "critical",
  "tones": [
    {"label": "neutral", "confidence": <float>},
    {"label": "frustrated", "confidence": <float>},
    {"label": "hostile", "confidence": <float>},
    {"label": "sarcastic", "confidence": <float>},
    {"label": "positive", "confidence": <float>}
  ],
  "intent": "constructive" | "critical" | "personal" | "informational",
  "intent_confidence": <float>,
  "needs_attention": <bool>,
  "explanation": <one specific sentence about what triggered the assessment>
}

Risk scale:
- 0.0-0.2: none — clearly benign, positive, or constructive
- 0.2-0.4: low — direct but respectful, unlikely to offend
- 0.4-0.6: medium — could be perceived negatively depending on context
- 0.6-0.8: high — likely to cause tension or discomfort
- 0.8-1.0: critical — aggressive, hostile, or damaging

Key rules:
- Direct ≠ hostile. "This is wrong" is direct but not hostile.
- Frustrated ≠ aggressive. "I'm annoyed by this" is frustrated, not aggressive.
- Sarcasm uses opposite-of-literal meaning. "Great job..." after failure is sarcastic.
- Personal attacks target the person ("you're incompetent"), not the work ("this code is bad").
- YouTube comments are more casual than workplace chat — adjust risk down slightly.
- For clearly positive/constructive comments, use low risk with positive tone.
- Be specific in the explanation: mention what specific words or phrases triggered the assessment."""


def call_openai(prompt: str, api_key: str, retries: int = 3) -> dict | None:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "max_tokens": 500,
        "response_format": {"type": "json_object"},
    }

    for attempt in range(retries):
        try:
            resp = requests.post(OPENAI_URL, json=payload, headers=headers, timeout=30)
            if resp.status_code == 429:
                wait = 2 ** attempt
                print(f"rate_limited(wait={wait}s)", end=" ", flush=True)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            return json.loads(content)
        except (requests.RequestException, json.JSONDecodeError, KeyError) as e:
            if attempt < retries - 1:
                time.sleep(1.0 * (attempt + 1))
            else:
                print(f"FAILED({e})", end=" ", flush=True)
                return None
    return None


def annotate_batch(comments: list[dict], api_key: str) -> list[dict]:
    annotated = []

    for i, comment in enumerate(comments):
        text = comment["text"]
        ctx = comment.get("context", "")
        prompt = f"Comment: {text}\nContext: {ctx}"

        print(f"  [{i+1}/{len(comments)}] {text[:80]}...", end=" ", flush=True)

        result = call_openai(prompt, api_key)

        if result is None:
            result = {
                "communication_risk": 0.1,
                "risk_level": "none",
                "tones": [
                    {"label": "neutral", "confidence": 0.5},
                    {"label": "frustrated", "confidence": 0.05},
                    {"label": "hostile", "confidence": 0.05},
                    {"label": "sarcastic", "confidence": 0.05},
                    {"label": "positive", "confidence": 0.5},
                ],
                "intent": "informational",
                "intent_confidence": 0.5,
                "needs_attention": False,
                "explanation": "API call failed — manually review required.",
            }
            print("→ FALLBACK")
        else:
            valid_tones = {"neutral", "frustrated", "hostile", "sarcastic", "positive"}
            valid_intents = {"constructive", "critical", "personal", "informational"}
            valid_levels = {"none", "low", "medium", "high", "critical"}

            if result.get("risk_level") not in valid_levels:
                result["risk_level"] = "none"
            if result.get("intent") not in valid_intents:
                result["intent"] = "informational"
            if isinstance(result.get("tones"), list) and result["tones"]:
                result["tones"] = [
                    t for t in result["tones"]
                    if isinstance(t, dict) and t.get("label") in valid_tones
                ]
            else:
                result["tones"] = []

            print(f"r={result.get('communication_risk', 0):.2f} {result.get('risk_level', '?')}")

        new = dict(comment)
        new["annotations"] = {
            "communication_risk": result.get("communication_risk", 0.1),
            "risk_level": result.get("risk_level", "none"),
            "tones": result.get("tones", []),
            "intent": result.get("intent", "informational"),
            "intent_confidence": result.get("intent_confidence", 0.5),
            "needs_attention": result.get("needs_attention", False),
            "explanation": result.get("explanation", "Auto-annotated — manual review required."),
            "annotator": f"openai-{MODEL}",
        }
        annotated.append(new)

    return annotated


def save_checkpoint(data: dict) -> None:
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump(data, f, indent=2)


def load_checkpoint() -> dict | None:
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH) as f:
            return json.load(f)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto-annotate YouTube comments via OpenAI API")
    parser.add_argument("--api-key", required=True, help="OpenAI API key")
    parser.add_argument("--language", choices=["en", "it"], help="Process only one language")
    parser.add_argument("--input", type=Path, help="Process a specific JSONL file (overrides youtube files)")
    parser.add_argument("--output", type=Path, help="Output file for --input mode")
    parser.add_argument("--max", type=int, default=0, help="Max comments (0 = all)")
    parser.add_argument("--batch-size", type=int, default=20, help="Comments between saves")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    args = parser.parse_args()

    cp = load_checkpoint() if args.resume else None

    # Custom file mode
    if args.input:
        infile = args.input
        if not infile.exists():
            print(f"File not found: {infile}")
            return
        outfile = args.output or infile.parent / f"{infile.stem}_annotated.jsonl"

        with open(infile) as f:
            all_comments = [json.loads(line) for line in f]

        remaining = all_comments[:args.max] if args.max else all_comments
        print(f"\nAnnotating: {infile.name} ({len(remaining)} comments)")
        print(f"{'='*60}")

        for batch_start in range(0, len(remaining), args.batch_size):
            batch = remaining[batch_start : batch_start + args.batch_size]
            annotated = annotate_batch(batch, args.api_key)

            with open(outfile, "a") as f:
                for ex in annotated:
                    f.write(json.dumps(ex, ensure_ascii=False) + "\n")

            print(f"  {batch_start + len(annotated)}/{len(remaining)} -> {outfile.name}")

        print(f"\nDone: {len(remaining)} examples -> {outfile}")
        return

    # Default: process YouTube files
    languages = [args.language] if args.language else ["en", "it"]
    files = {"en": (YOUTUBE_EN, OUTPUT_EN), "it": (YOUTUBE_IT, OUTPUT_IT)}

    total_annotated = 0

    for lang in languages:
        infile, outfile = files[lang]
        if not infile.exists():
            print(f"Skipping {lang}: {infile} not found")
            continue

        with open(infile) as f:
            all_comments = [json.loads(line) for line in f]

        start_idx = 0
        if cp and lang in cp:
            start_idx = cp[lang].get("processed", 0)
            print(f"Resuming {lang} from index {start_idx}")

        remaining = all_comments[start_idx:]
        if args.max:
            remaining = remaining[: args.max]

        print(f"\n{'='*60}")
        print(f"Annotating {lang.upper()}: {len(remaining)} comments (model: {MODEL})")
        print(f"{'='*60}")

        for batch_start in range(0, len(remaining), args.batch_size):
            batch = remaining[batch_start : batch_start + args.batch_size]
            batch_idx = start_idx + batch_start
            annotated = annotate_batch(batch, args.api_key)

            with open(outfile, "a") as f:
                for ex in annotated:
                    f.write(json.dumps(ex, ensure_ascii=False) + "\n")

            total_annotated += len(annotated)

            if cp is None:
                cp = {}
            if lang not in cp:
                cp[lang] = {}
            cp[lang]["processed"] = batch_idx + len(annotated)
            cp[lang]["total"] = len(all_comments)
            save_checkpoint(cp)

            print(f"  Checkpoint: {cp[lang]['processed']}/{cp[lang]['total']}")

    print(f"\n{'='*60}")
    print("ANNOTATION COMPLETE")
    print(f"{'='*60}")
    print(f"Total annotated: {total_annotated}")
    print(f"EN output: {OUTPUT_EN}")
    print(f"IT output: {OUTPUT_IT}")


if __name__ == "__main__":
    main()
