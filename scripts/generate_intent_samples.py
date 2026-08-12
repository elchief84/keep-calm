"""Generate targeted intent examples for underrepresented classes.

Generates realistic workplace chat messages biased toward `personal`,
`constructive`, and `informational` intents — the categories where
the model currently lacks training data.

Output: unlabeled JSONL ready for auto_annotate.py

Usage:
    python scripts/generate_intent_samples.py --lang en --count 2000
    python scripts/generate_intent_samples.py --lang it --count 1500
    python scripts/generate_intent_samples.py --lang both --count 3500
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

SEED = 42
random.seed(SEED)

# ── Templates ──────────────────────────────────────────────────────────

# Format: (intent, template) — placeholders will be filled from VARIANTS
# Each template generates multiple samples by varying the placeholders.

PERSON_VERBS = ["you", "your team", "your department", "your approach"]
WORK_ITEMS = ["the report", "this feature", "the documentation", "the deadline", "the code"]

EN_TEMPLATES: list[tuple[str, str]] = [
    # ── personal ──
    ("personal", "You never listen to what I'm saying."),
    ("personal", "You're the reason this project is behind schedule."),
    ("personal", "Maybe someone more experienced should handle this."),
    ("personal", "You clearly don't understand how this works."),
    ("personal", "This is exactly what I expected from {person}."),
    ("personal", "I'm tired of cleaning up {person}'s mistakes."),
    ("personal", "Your attitude is the problem here."),
    ("personal", "If {person} had done their job, we wouldn't be here."),
    ("personal", "You're not qualified to make this decision."),
    ("personal", "Everyone knows you don't pull your weight."),
    ("personal", "Why am I not surprised it was {person} again?"),
    ("personal", "{person} always finds a way to mess things up."),
    ("personal", "You're being lazy and it's affecting everyone."),
    ("personal", "Don't blame the tools when the problem is {person}."),
    ("personal", "Typical {person} — all talk, no action."),
    ("personal", "You should know this by now."),
    ("personal", "I've explained this to {person} three times already."),
    ("personal", "{person} is completely out of their depth here."),
    ("personal", "You don't have the skills for this role."),
    ("personal", "Stop pretending you know what you're doing."),

    # ── constructive ──
    ("constructive", "Here's a suggestion that might help with {work_item}."),
    ("constructive", "I think we can improve {work_item} by doing X instead of Y."),
    ("constructive", "Great start — a few suggestions on {work_item}."),
    ("constructive", "Let me help you debug this — I've seen similar issues before."),
    ("constructive", "We should break {work_item} down into smaller steps."),
    ("constructive", "Here's how we fixed this last time, might help now."),
    ("constructive", "Good point — to build on that, we could also consider X."),
    ("constructive", "I appreciate the effort on {work_item}. One thing that could make it even better:"),
    ("constructive", "Let's pair on this tomorrow and I'll walk you through it."),
    ("constructive", "This is a solid approach. Want me to review the details?"),
    ("constructive", "Nice work. Have you considered adding X to {work_item}?"),
    ("constructive", "The direction is right — let's refine the execution."),
    ("constructive", "I see what you're going for. Here's an alternative that might be cleaner."),
    ("constructive", "Happy to jump on a call and walk through {work_item} together."),
    ("constructive", "This is getting there. The main gap I see is X."),

    # ── informational ──
    ("informational", "Can you send me the latest version of {work_item}?"),
    ("informational", "The meeting is at 3pm in room 4B."),
    ("informational", "Here are the updated numbers for Q3."),
    ("informational", "The {work_item} deadline has been moved to Friday."),
    ("informational", "Can you confirm if this is the final version?"),
    ("informational", "I've attached the spreadsheet with the latest data."),
    ("informational", "The server will be down for maintenance from 2am to 4am."),
    ("informational", "Here's the link to the document you asked for."),
    ("informational", "Quick update: the client approved the proposal."),
    ("informational", "Please fill out the survey by end of day."),
    ("informational", "The build passed — we're good to deploy."),
    ("informational", "Does anyone have the minutes from yesterday's meeting?"),
    ("informational", "Just a heads up: I'll be out of office next Tuesday."),
    ("informational", "The staging environment is ready for testing."),
    ("informational", "Reminder: time sheets are due by 5pm today."),
]


IT_TEMPLATES: list[tuple[str, str]] = [
    # ── personal ──
    ("personal", "Non ascolti mai quello che dico."),
    ("personal", "Sei tu il motivo per cui il progetto e in ritardo."),
    ("personal", "Magari qualcuno di piu competente dovrebbe occuparsene."),
    ("personal", "{person} non ha idea di come funzioni."),
    ("personal", "E esattamente quello che mi aspettavo da {person}."),
    ("personal", "Sono stufo di sistemare gli errori di {person}."),
    ("personal", "Il tuo atteggiamento e il problema qui."),
    ("personal", "Se {person} avesse fatto il suo lavoro non saremmo qui."),
    ("personal", "Non sei qualificato per prendere questa decisione."),
    ("personal", "Tutti sanno che non fai la tua parte."),
    ("personal", "Perche non mi sorprende che sia stato di nuovo {person}?"),
    ("personal", "{person} trova sempre il modo di rovinare tutto."),
    ("personal", "Stai facendo il minimo indispensabile e ne risente tutto il team."),
    ("personal", "Non dare la colpa agli strumenti quando il problema sei {person}."),
    ("personal", "Il solito {person}: solo chiacchiere, zero fatti."),
    ("personal", "Dovresti saperlo ormai."),
    ("personal", "L'ho gia spiegato a {person} tre volte."),
    ("personal", "{person} e completamente fuori dalla sua portata."),
    ("personal", "Non hai le competenze per questo ruolo."),
    ("personal", "Smettila di fingere di sapere quello che fai."),

    # ── constructive ──
    ("constructive", "Ecco un suggerimento che potrebbe aiutare con {work_item}."),
    ("constructive", "Penso che possiamo migliorare {work_item} facendo X invece di Y."),
    ("constructive", "Ottimo inizio — qualche suggerimento su {work_item}:"),
    ("constructive", "Lascia che ti aiuti con questo — ho visto problemi simili."),
    ("constructive", "Suddividiamo {work_item} in passaggi piu piccoli."),
    ("constructive", "Ecco come l'abbiamo risolto la scorsa volta, potrebbe aiutare."),
    ("constructive", "Buona idea — per ampliare, potremmo anche considerare X."),
    ("constructive", "Apprezzo l'impegno su {work_item}. Una cosa che potrebbe migliorarlo:"),
    ("constructive", "Lavoriamoci insieme domani, ti mostro come fare."),
    ("constructive", "Bel lavoro. Hai considerato di aggiungere X a {work_item}?"),
    ("constructive", "La direzione e giusta — perfezioniamo l'esecuzione."),
    ("constructive", "Capisco cosa vuoi fare. Ecco un'alternativa piu pulita."),
    ("constructive", "Volentieri faccio una call e ti guido su {work_item}."),
    ("constructive", "Sta venendo bene. La principale lacuna che vedo e X."),
    ("constructive", "Ottimo lavoro sul refactoring di {work_item}."),

    # ── informational ──
    ("informational", "Mi mandi l'ultima versione di {work_item}?"),
    ("informational", "La riunione e alle 15 in sala riunioni."),
    ("informational", "Ecco i numeri aggiornati per il Q3."),
    ("informational", "La scadenza di {work_item} e stata spostata a venerdi."),
    ("informational", "Puoi confermare se questa e la versione finale?"),
    ("informational", "Ho allegato il foglio con i dati aggiornati."),
    ("informational", "Il server sara in manutenzione dalle 2 alle 4."),
    ("informational", "Ecco il link al documento che avevi chiesto."),
    ("informational", "Aggiornamento rapido: il cliente ha approvato."),
    ("informational", "Per favore compila il sondaggio entro fine giornata."),
    ("informational", "La build ha passato i test — possiamo deployare."),
    ("informational", "Qualcuno ha i verbali della riunione di ieri?"),
    ("informational", "Solo un avviso: saro fuori ufficio martedi prossimo."),
    ("informational", "L'ambiente di staging e pronto per i test."),
    ("informational", "Promemoria: i timesheet vanno consegnati entro le 17."),
]

# Fillers per rendere i messaggi piu naturali
EN_PERSON_VARIANTS = ["you", "your team", "your department", "your approach",
                       "your manager", "your colleague", "the intern"]
EN_WORK_VARIANTS = ["the report", "this feature", "the documentation",
                     "the deadline", "the code", "the API", "the database",
                     "the deployment", "the pipeline", "the dashboard"]

IT_PERSON_VARIANTS = ["tu", "il tuo team", "il tuo reparto", "il tuo approccio",
                       "il tuo manager", "il tuo collega", "lo stagista"]
IT_WORK_VARIANTS = ["il report", "questa funzionalita", "la documentazione",
                     "la scadenza", "il codice", "l'API", "il database",
                     "il deploy", "la pipeline", "la dashboard"]

PREFIXES_EN = [
    "Hey, ", "Quick note: ", "Just so we're clear: ", "Honestly, ",
    "I'm going to be direct: ", "One thing I want to say: ",
    "Let me put it this way: ", "I need to point out: ", "", ""
]
PREFIXES_IT = [
    "Senti, ", "Nota rapida: ", "Solo per essere chiari: ", "Onestamente, ",
    "Vado dritto al punto: ", "Una cosa che voglio dire: ",
    "Mettiamola cosi: ", "Devo farti notare: ", "", ""
]
SUFFIXES_EN = [
    " Can we talk?", " Just think about it.", " That's my perspective.",
    "", "", "", ""
]
SUFFIXES_IT = [
    " Ci parliamo?", " Pensaci.", " Questo e il mio parere.",
    "", "", "", ""
]


def fill_template(template: str, lang: str) -> str:
    """Replace {person} and {work_item} with random variants."""
    person_variants = EN_PERSON_VARIANTS if lang == "en" else IT_PERSON_VARIANTS
    work_variants = EN_WORK_VARIANTS if lang == "en" else IT_WORK_VARIANTS
    text = template.replace("{person}", random.choice(person_variants))
    text = text.replace("{work_item}", random.choice(work_variants))
    return text


def add_fillers(text: str, intent: str, lang: str) -> str:
    """Add optional prefix/suffix for natural variation."""
    prefixes = PREFIXES_EN if lang == "en" else PREFIXES_IT
    suffixes = SUFFIXES_EN if lang == "en" else SUFFIXES_IT
    if intent in ("personal", "constructive") and random.random() < 0.4:
        text = random.choice(prefixes) + text
    if random.random() < 0.3:
        text = text + random.choice(suffixes)
    return text


def generate_samples(lang: str, count: int) -> list[dict]:
    templates = EN_TEMPLATES if lang == "en" else IT_TEMPLATES

    # Weights: heavily bias toward personal, then constructive, then informational
    personal_templates = [t for t in templates if t[0] == "personal"]
    constructive_templates = [t for t in templates if t[0] == "constructive"]
    informational_templates = [t for t in templates if t[0] == "informational"]

    pool = (personal_templates * 3) + (constructive_templates * 2) + informational_templates
    random.shuffle(pool)

    seen: set[str] = set()
    samples: list[dict] = []

    while len(samples) < count:
        intent, template = random.choice(pool)
        text = fill_template(template, lang)
        text = add_fillers(text, intent, lang)

        key = hashlib.md5(text.encode()).hexdigest()[:12]
        if key in seen:
            continue
        seen.add(key)

        samples.append({
            "id": f"gen-{lang}-{key}",
            "text": text,
            "language": lang,
            "domain": "workplace_chat",
            "source": "synthetic_generated",
            "generated_for": intent,
        })

    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate targeted intent samples")
    parser.add_argument("--lang", default="both", choices=["en", "it", "both"])
    parser.add_argument("--count", type=int, default=3500,
                        help="Total samples (split evenly if both)")
    parser.add_argument("--out", default=None, help="Output file (default: stdout)")
    args = parser.parse_args()

    langs = ["en", "it"] if args.lang == "both" else [args.lang]
    per_lang = args.count // len(langs)

    all_samples: list[dict] = []
    for lang in langs:
        samples = generate_samples(lang, per_lang)
        all_samples.extend(samples)
        # Count by intent
        from collections import Counter
        dist = Counter(s["generated_for"] for s in samples)
        print(f"{lang.upper()}: {len(samples)} samples — "
              f"personal={dist['personal']} "
              f"constructive={dist['constructive']} "
              f"informational={dist['informational']}",
              file=sys.stderr)

    out_path = args.out or f"data/intent_augment_{args.lang}.jsonl"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for s in all_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(all_samples)} samples to {out_path}", file=sys.stderr)
    print(f"Next: auto-annotate with\n"
          f"  python scripts/auto_annotate.py --input {out_path} "
          f"--lang {args.lang} --api-key YOUR_KEY", file=sys.stderr)


if __name__ == "__main__":
    main()
