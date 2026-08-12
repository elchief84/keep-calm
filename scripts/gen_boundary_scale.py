"""Generate a large volume of critical/personal boundary examples.

The critical/personal distinction follows our own annotation guidelines:
- critical  = negative feedback targeting the WORK/OUTPUT ("Your code has bugs")
- personal  = negative feedback targeting the PERSON ("You always write bugs")

We generate both sides of the boundary at scale with ground-truth labels,
so no OpenAI annotation is needed.

Usage:
    python scripts/gen_boundary_scale.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

# ── CRITICAL: "the/this/your + WORK + negative judgment" ─────────────────
EN_WORK = [
    "code", "implementation", "design", "approach", "report", "documentation",
    "solution", "architecture", "deployment", "API", "test coverage",
    "migration plan", "feature", "refactor", "estimate", "timeline",
    "proposal", "integration", "data model", "query", "UI", "error handling",
]
EN_WORK_NEG = [
    "has serious flaws", "doesn't meet requirements", "won't scale",
    "missed a critical issue", "is broken", "is too slow", "is outdated",
    "is incomplete", "introduced more bugs", "is inconsistent",
    "is too risky", "contradicts what we discussed", "is unrealistic",
    "lacks proper analysis", "is fragile", "has a failing edge case",
    "isn't ready to ship", "is way off", "breaks backward compatibility",
    "has security gaps", "will time out in production", "is confusing",
    "ignores the budget", "is half-baked", "doesn't reflect reality",
]
EN_CRITICAL_PREFIX = [
    "The", "This", "Our", "Their", "That",
]

EN_PERSON_TRAIT = [
    "don't understand the requirements", "aren't thinking this through",
    "ignored every concern I raised", "don't have the experience for this",
    "keep making the same mistake", "have poor judgment here",
    "didn't bother to check your own work", "are missing the point",
    "rushed through this", "don't know what you're talking about",
    "dropped the ball", "have no attention to detail", "ignored the spec",
    "aren't taking this seriously", "are being careless",
    "don't listen when people explain things", "were unprepared",
    "didn't test before pushing", "have a shallow understanding",
    "made assumptions without verifying", "are cutting corners",
    "don't care about quality", "missed critical requirements",
    "don't read the documentation", "aren't putting in enough effort",
    "keep ignoring review feedback", "don't understand how this works",
    "show a lack of understanding", "were careless with details",
    "didn't think this through", "don't have a clue what you're doing",
    "are way out of your depth here", "keep dropping the ball",
    "have no idea how this system works", "are guessing instead of knowing",
    "didn't do your homework", "are being sloppy and it shows",
    "refuse to listen to anyone", "have a terrible attitude about this",
    "don't own your mistakes", "keep passing the buck",
    "are the reason this keeps failing", "are impossible to work with",
    "don't respect other people's time", "are always the last one to deliver",
    "never follow through on what you say", "are coasting on others' work",
    "can't handle constructive criticism", "get defensive over everything",
    "are out of touch with the team", "don't communicate anything clearly",
    "are the bottleneck on every task", "haven't learned anything in months",
    "are unprofessional in every meeting", "make excuses for everything",
    "don't take ownership of anything", "are holding the team back",
    "have no initiative whatsoever", "wait to be told everything",
    "are never prepared for anything", "consistently miss the mark",
    "don't understand the codebase", "are careless with production data",
    "ignore security best practices", "don't know the product at all",
    "are always behind on everything", "never raise issues until it's too late",
    "don't collaborate with anyone", "work in a silo and it hurts everyone",
    "are too proud to admit mistakes", "keep repeating the same failures",
    "are not qualified to review this", "shouldn't be making this call",
    "have no attention to detail", "are sloppy with your commits",
    "don't understand the business impact", "are wasting everyone's time",
    "aren't a team player", "undermine every decision",
    "are all talk and no action", "never deliver what you promise",
    "are out of your depth", "don't pull your weight",
    "are the weak link in the chain", "set a bad example for the juniors",
    "are consistently underperforming", "don't belong on this team",
    "are a liability to this project", "have exhausted every second chance",
    "are not learning from your mistakes", "keep making rookie errors",
    "don't understand the requirements at all", "are way behind everyone else",
    "have no ownership mentality", "blame the tools instead of yourself",
    "are resistant to every new approach", "drag your feet on every task",
    "are the reason for the low morale", "don't respect deadlines",
    "are careless with confidential info", "skip the review process entirely",
    "are not ready for this role", "need constant supervision",
    "can't be trusted to deliver", "have poor communication skills",
]

IT_WORK = [
    "codice", "implementazione", "design", "approccio", "report", "documentazione",
    "soluzione", "architettura", "deploy", "API", "copertura test",
    "piano di migrazione", "funzionalita", "refactoring", "stima", "tempistica",
    "proposta", "integrazione", "modello dati", "query", "interfaccia",
    "gestione errori",
]
IT_WORK_NEG = [
    "ha difetti seri", "non rispetta i requisiti", "non scala",
    "ha un problema critico", "e rotto", "e troppo lento", "e obsoleto",
    "e incompleto", "ha introdotto piu bug", "e incoerente",
    "e troppo rischioso", "contraddice quanto discusso", "e irrealistico",
    "manca di analisi", "e fragile", "ha un caso limite che fallisce",
    "non e pronto per il rilascio", "e completamente sbagliato",
    "rompe la compatibilita", "ha lacune di sicurezza",
    "andra in timeout in produzione", "e confuso", "ignora il budget",
    "e fatto a meta", "non riflette la realta",
]
IT_CRITICAL_PREFIX = [
    "Il", "Questo", "Il nostro", "Il loro", "Quel",
]

IT_PERSON_TRAIT = [
    "non capisci i requisiti", "non stai pensando bene",
    "hai ignorato ogni mia preoccupazione", "non hai l'esperienza per questo",
    "continui a fare lo stesso errore", "hai un pessimo giudizio qui",
    "non ti sei nemmeno controllato il lavoro", "ti stai perdendo il punto",
    "hai fatto di fretta", "non sai di cosa parli", "hai toppato",
    "non hai attenzione ai dettagli", "hai ignorato le specifiche",
    "non prendi la cosa sul serio", "sei trascurato",
    "non ascolti quando ti spiegano le cose", "eri impreparato",
    "non hai testato prima di pushare", "hai una comprensione superficiale",
    "hai fatto assunzioni senza verificare", "tagli gli angoli",
    "non ti interessa la qualita", "hai mancato requisiti critici",
    "non leggi la documentazione", "non ti impegni abbastanza",
    "continui a ignorare il feedback", "non capisci come funziona",
    "mostri poca comprensione", "sei trascurato coi dettagli",
    "non ci hai pensato prima di iniziare", "non hai idea di cosa stai facendo",
    "sei completamente fuori dalla tua portata", "continui a toppare",
    "non hai idea di come funzioni questo sistema", "tiri a indovinare invece di sapere",
    "non hai fatto i compiti", "sei sciatto e si vede",
    "ti rifiuti di ascoltare chiunque", "hai un pessimo atteggiamento",
    "non ti prendi la responsabilita dei tuoi errori", "scarichi la colpa",
    "sei il motivo per cui continua a fallire", "sei impossibile con cui lavorare",
    "non rispetti il tempo degli altri", "sei sempre l'ultimo a consegnare",
    "non mantieni mai cio che dici", "ti adagi sul lavoro degli altri",
    "non accetti le critiche costruttive", "ti metti sulla difensiva su tutto",
    "sei fuori dal giro del team", "non comunichi nulla con chiarezza",
    "sei il collo di bottiglia su ogni task", "non impari nulla da mesi",
    "sei poco professionale in ogni riunione", "trovi scuse per tutto",
    "non ti prendi la responsabilita di nulla", "stai frenando il team",
    "non hai alcuna iniziativa", "aspetti che ti dicano tutto",
    "non sei mai preparato", "mancchi costantemente il bersaglio",
    "non capisci il codebase", "sei trascurato coi dati di produzione",
    "ignori le best practice di sicurezza", "non conosci affatto il prodotto",
    "sei sempre in ritardo su tutto", "segnali i problemi quando e troppo tardi",
    "non collabori con nessuno", "lavori in un silo e danneggi tutti",
    "sei troppo orgoglioso per ammettere gli errori", "ripeti sempre gli stessi fallimenti",
    "non sei qualificato per fare review", "non dovresti prendere questa decisione",
    "non hai attenzione ai dettagli", "sei sciatto coi commit",
    "non capisci l'impatto sul business", "stai facendo perdere tempo a tutti",
    "non sai lavorare in squadra", "mini ogni decisione",
    "sei solo chiacchiere e zero fatti", "non consegni mai cio che prometti",
    "sei fuori dalla tua portata", "non fai la tua parte",
    "sei l'anello debole della catena", "dai un pessimo esempio ai junior",
    "sei costantemente sotto la media", "non appartieni a questo team",
    "sei un peso per questo progetto", "hai esaurito ogni seconda possibilita",
    "non impari dai tuoi errori", "continui a fare errori da principiante",
    "non capisci affatto i requisiti", "sei molto indietro rispetto agli altri",
    "non hai mentalita da proprietario", "dai la colpa agli strumenti invece che a te",
    "sei restio a ogni nuovo approccio", "trascini i piedi su ogni task",
    "sei il motivo del morale basso", "non rispetti le scadenze",
    "sei trascurato con le info riservate", "salti completamente il processo di review",
    "non sei pronto per questo ruolo", "hai bisogno di supervisione costante",
    "non ci si puo fidare che consegni", "hai pessime capacita comunicative",
]


def build_critical(lang: str, count: int) -> list[str]:
    work = EN_WORK if lang == "en" else IT_WORK
    neg = EN_WORK_NEG if lang == "en" else IT_WORK_NEG
    prefix = EN_CRITICAL_PREFIX if lang == "en" else IT_CRITICAL_PREFIX
    out: set[str] = set()
    i = 0
    while len(out) < count and i < count * 20:
        i += 1
        # Mix: "your work neg" (borderline pattern) and "the work neg" (clear)
        if i % 3 == 0:
            text = f"Your {work[i % len(work)]} {neg[i % len(neg)]}."
        else:
            text = f"{prefix[i % len(prefix)]} {work[i % len(work)]} {neg[i % len(neg)]}."
        out.add(text)
    return list(out)


def build_personal(lang: str, count: int) -> list[str]:
    traits = EN_PERSON_TRAIT if lang == "en" else IT_PERSON_TRAIT
    prefixes = {
        "en": ["", "", "Honestly, ", "Look, ", "I'll be direct: ", "Honestly, you "],
        "it": ["", "", "Onestamente, ", "Guarda, ", "Vado dritto: ", "Onestamente "],
    }[lang]
    out: set[str] = set()
    i = 0
    while len(out) < count and i < count * 30:
        i += 1
        trait = traits[i % len(traits)]
        p = prefixes[i % len(prefixes)]
        if p == "Honestly, you ":
            text = f"Honestly, you {trait}."
        elif p == "Onestamente ":
            text = f"Onestamente, {trait}."
        elif lang == "en":
            text = f"{p}You {trait}.".replace("  ", " ")
        else:
            base = trait.capitalize()
            text = f"{p}{base}.".replace("  ", " ")
        out.add(text)
    return list(out)


def main():
    all_samples: list[dict] = []
    for lang in ("en", "it"):
        critical = build_critical(lang, 800)
        personal = build_personal(lang, 800)
        for intent, texts in [("critical", critical), ("personal", personal)]:
            for t in texts:
                key = hashlib.md5(t.encode()).hexdigest()[:10]
                all_samples.append({
                    "id": f"boundary-{lang}-{key}",
                    "text": t,
                    "language": lang,
                    "domain": "workplace_chat",
                    "source": "synthetic_generated",
                    "generated_for": intent,
                })
        print(f"{lang.upper()}: {len(critical)} critical + {len(personal)} personal")

    path = Path("data/intent_boundary_scale.jsonl")
    with open(path, "w") as f:
        for s in all_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"\nTotal: {len(all_samples)} -> {path}")


if __name__ == "__main__":
    main()
