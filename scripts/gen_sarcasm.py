"""Generate English sarcasm examples for training."""

import json, hashlib, time, requests
from pathlib import Path

API_KEY = "YOUR_OPENAI_API_KEY"

cats = [
    ("sarcastic_praise", "Sarcasm disguised as praise: great job as always, brilliant idea, you never fail"),
    ("sarcastic_excitement", "Sarcastic excitement: oh wonderful another meeting, cant wait for the deadline"),
    ("sarcastic_agreement", "Sarcastic agreement: sure whatever you say, great plan nothing could go wrong"),
    ("sarcastic_dev", "Developer sarcasm: works on my machine, its not a bug its a feature, who wrote this"),
    ("sarcastic_colleague", "Sarcasm about colleagues: nice of you to join us, thanks for responding 3 weeks later"),
    ("sarcastic_deadline", "Sarcasm about deadlines: love unrealistic deadlines, sure we can deliver by Friday"),
]

all_ex = []
for cat, desc in cats:
    print(f"  {cat}: ", end="", flush=True)
    for batch in range(5):
        sysprompt = f"Generate 20 workplace sarcasm expressions. Category: {cat}. Style: {desc}. Short (3-12 words), realistic, use ellipsis, periods, lowercase. Output ONLY JSON array of strings."
        try:
            resp = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
                json={"model": "gpt-4o-mini", "temperature": 0.95, "max_tokens": 4000,
                    "messages": [{"role": "system", "content": sysprompt}, {"role": "user", "content": "Generate 20 short sarcasm messages."}],
                    "response_format": {"type": "json_object"}}, timeout=30)
            items = json.loads(resp.json()["choices"][0]["message"]["content"])
            if isinstance(items, dict): items = list(items.values())[0] if items else []
            for text in items:
                if isinstance(text, str) and 4 <= len(text) <= 200:
                    all_ex.append({"id": f"sarc-{hashlib.sha256(text.encode()).hexdigest()[:12]}", "text": text, "language": "en",
                        "domain": "sarcasm", "source": "openai_generated", "metadata": {"category": cat},
                        "annotations": {"communication_risk": None, "tones": [], "intent": None, "explanation": None, "needs_attention": None}})
        except: pass
        if batch < 4: time.sleep(0.2)
    cnt = len([x for x in all_ex if x["metadata"]["category"] == cat])
    print(f"{cnt}")

with open("data/sarcasm_en.jsonl", "w") as f:
    for ex in all_ex: f.write(json.dumps(ex, ensure_ascii=False) + "\n")
print(f"\nSaved {len(all_ex)} sarcasm examples")
