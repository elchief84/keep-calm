"""Generate idiomatic/vulgar expressions in IT and EN for Keep Calm training."""

import hashlib
import json
import time
from pathlib import Path

import requests

API_KEY = "YOUR_OPENAI_API_KEY"

CATEGORIES = {
    "it": {
        "insulti_diretti": "Italian insults: sei un cretino, fai schifo, sei patetico, non vali niente, sei una frana, ritardato, inutile, sei la vergogna del team",
        "volgari": "Italian vulgar: vaffanculo, che cazzo vuoi, hai rotto il cazzo, fanculo, che due palle, non me ne frega un cazzo, cazzi tuoi, sticazzi, ma vaffanculo",
        "colloquiali": "Colloquial aggressive: ma che stai a di, ma chi te se incula, ma va a cagare, vattela a pija, ma chi ti credi di essere, scendi dal pero, ma piantala",
        "passive_aggressive": "Passive-aggressive: come sicuramente saprai, mi meraviglio di te, pensavo fossi piu sveglio, lascia perdere va, fai un po come ti pare, ti sei superato stavolta",
        "sarcasmo": "Sarcastic: bravissimo complimenti, grande come al solito, che bello un'altra riunione, ma certo facciamo come dici tu, si si va bene, ah pero",
        "minacce": "Veiled threats: poi non lamentarti, vedrai dopo, io te lavevo detto, te ne pentirai, poi vediamo, fai attenzione, vedremo chi ha ragione",
        "dialetto": "Dialect expressions: ma che stai a di, daje, ammazzete, li mortacci tua, ma chi ti criri, vatinni, mangia e statti cittu, si propriu nu fissa, te se minga bon",
        "frasi_complete": "Full sentences mixing the above in workplace context, 5-15 words each",
    },
    "en": {
        "insults": "English insults: you are pathetic, you suck, get lost, you are useless, moron, idiot, what is wrong with you, you are the worst",
        "vulgar": "Vulgar: fuck off, screw you, this is bullshit, go to hell, what the hell, are you fucking kidding me, piece of shit, you are full of crap",
        "passive_aggressive": "Passive-aggressive: as I am sure you are aware, per my last email, I am not surprised, some of us actually read the docs, let me know when you are ready to take this seriously",
        "sarcasm": "Sarcastic: great job as always, oh wonderful another meeting, sure lets do it your way, what could possibly go wrong, I love how we keep having this conversation",
        "colloquial": "Colloquial aggressive: get over yourself, who do you think you are, give me a break, are you serious right now, get real, you must be joking",
        "british": "British passive-aggressive: with all due respect, I am slightly confused, that is certainly one way to do it, how interesting, I see, right then, if you say so",
        "corporate": "American corporate: let me circle back on that, I will take that under advisement, lets align on this, thanks for sharing, I appreciate your perspective",
        "dev_sarcasm": "Developer sarcasm: works on my machine, it is not a bug it is a feature, I will just rewrite this, who wrote this, this is production ready, it compiles ship it",
        "workplace_frustration": "Frustrated workplace: this keeps breaking, we discussed this already, the deadline was yesterday, nobody reads the documentation, this is the third time",
    },
}

OUTPUTS = {"en": Path("data/idioms_en.jsonl"), "it": Path("data/idioms_it.jsonl")}

for lang, output in [("en", OUTPUTS["en"]), ("it", OUTPUTS["it"])]:
    all_ex = []
    cats = CATEGORIES[lang]

    for cat, desc in cats.items():
        print(f"  {cat}: ", end="", flush=True)

        for batch in range(5):
            system = f"Generate many {lang.upper()} workplace/messaging expressions. Category: {cat}. Style: {desc}. Use real slang, fragments, varying intensity from mild to very aggressive. Some should be complete sentences."
            try:
                resp = requests.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
                    json={
                        "model": "gpt-4o-mini", "temperature": 0.9, "max_tokens": 4000,
                        "messages": [
                            {"role": "system", "content": system},
                            {"role": "user", "content": "Generate 20 expressions. Output ONLY JSON array of strings."},
                        ],
                        "response_format": {"type": "json_object"},
                    },
                    timeout=45,
                )
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                items = json.loads(content)
                if isinstance(items, dict):
                    items = list(items.values())[0] if items else []
                for text in items:
                    if isinstance(text, str) and 4 <= len(text) <= 250:
                        h = hashlib.sha256(text.encode()).hexdigest()[:12]
                        all_ex.append({
                            "id": f"idiom-{h}", "text": text, "language": lang,
                            "domain": f"idiom_{cat}", "source": "openai_generated",
                            "metadata": {"category": cat},
                            "annotations": {"communication_risk": None, "tones": [], "intent": None, "explanation": None, "needs_attention": None},
                        })
            except Exception:
                print("ERR", end=" ")
                time.sleep(2)
            if batch < 4:
                time.sleep(0.3)

        count = len([x for x in all_ex if x["metadata"]["category"] == cat])
        print(f"{count} exs")

    with open(output, "w") as f:
        for ex in all_ex:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"  -> {len(all_ex)} total\n")

print("DONE")
