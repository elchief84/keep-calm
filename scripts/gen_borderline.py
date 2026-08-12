"""Generate borderline critical/personal examples to sharpen that boundary.

These are the hardest cases: negative feedback that could plausibly be read
as either "critical" (targeting the work) or "personal" (targeting the person).
Generates clearly-labeled examples on both sides of the boundary plus genuinely
ambiguous middle cases.

Usage:
    python scripts/gen_borderline.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

# ── CRITICAL (targets the WORK, not the person) ──────────────────────────
EN_CRITICAL = [
    "This approach has several flaws that need addressing.",
    "The implementation doesn't meet the requirements.",
    "This code won't scale under production load.",
    "The design has a critical flaw we missed.",
    "These numbers don't add up.",
    "The report is missing key sections.",
    "This solution is too slow for our needs.",
    "The documentation is outdated and misleading.",
    "This refactor introduced more bugs than it fixed.",
    "The API contract is inconsistent across endpoints.",
    "This feature is half-baked and shouldn't ship.",
    "The test coverage is far below our standard.",
    "This migration plan is too risky to execute now.",
    "The architecture doesn't support what we need.",
    "These results contradict what we discussed.",
    "The timeline is unrealistic given the scope.",
    "This decision was made without proper analysis.",
    "The deployment process is fragile and error-prone.",
    "This logic has an edge case that will fail.",
    "The proposal lacks sufficient detail to evaluate.",
    "This estimate is way off from reality.",
    "The integration is broken in staging.",
    "This change breaks backward compatibility.",
    "The security review found serious gaps.",
    "This query will time out on production data.",
    "The user interface is confusing to navigate.",
    "This plan ignores the budget constraints.",
    "The error handling is inconsistent throughout.",
    "This benchmark doesn't reflect real usage.",
    "The data model is too rigid for future needs.",
]

# ── PERSONAL (targets the PERSON) ────────────────────────────────────────
EN_PERSONAL = [
    "You clearly don't understand the requirements.",
    "You're not thinking this through at all.",
    "You've ignored every concern I raised.",
    "You don't have the experience for this.",
    "You keep making the same mistake over and over.",
    "Your judgment here was completely off.",
    "You didn't bother to check your own work.",
    "You're missing the point entirely.",
    "You rushed through this and it shows.",
    "You don't know what you're talking about.",
    "You dropped the ball on this one.",
    "Your attention to detail is nonexistent.",
    "You ignored the spec and did your own thing.",
    "You're not taking this seriously.",
    "Your work here is careless.",
    "You don't listen when people explain things.",
    "You were completely unprepared for this.",
    "You didn't test this before pushing it.",
    "Your understanding of the problem is shallow.",
    "You made assumptions without verifying anything.",
    "You're cutting corners and it's obvious.",
    "You don't care about quality.",
    "You missed critical requirements again.",
    "Your solution shows you didn't read the docs.",
    "You're not putting in enough effort.",
    "You keep ignoring the review feedback.",
    "You don't understand how this system works.",
    "Your approach shows a lack of understanding.",
    "You were careless with the details.",
    "You didn't think this through before starting.",
]

# ── BORDERLINE (genuinely ambiguous) ─────────────────────────────────────
EN_BORDERLINE = [
    "Your approach has some serious flaws.",
    "Your implementation is not meeting requirements.",
    "Your code won't scale in production.",
    "Your design missed a critical issue.",
    "Your numbers don't add up.",
    "Your report is missing key sections.",
    "Your solution is too slow.",
    "Your documentation is outdated.",
    "Your refactor introduced more bugs.",
    "Your API contract is inconsistent.",
    "Your feature isn't ready to ship.",
    "Your test coverage is too low.",
    "Your migration plan is too risky.",
    "Your architecture doesn't fit the needs.",
    "Your results contradict the discussion.",
    "Your timeline is unrealistic.",
    "Your decision lacked proper analysis.",
    "Your deployment process is fragile.",
    "Your logic has a failing edge case.",
    "Your proposal lacks detail.",
    "Your estimate is off.",
    "Your integration is broken in staging.",
    "Your change breaks backward compatibility.",
    "Your security review found gaps.",
    "Your query will time out in production.",
]

IT_CRITICAL = [
    "Questo approccio ha diversi difetti da correggere.",
    "L'implementazione non rispetta i requisiti.",
    "Questo codice non scala in produzione.",
    "Il design ha un difetto critico che ci e sfuggito.",
    "Questi numeri non tornano.",
    "Il report manca di sezioni fondamentali.",
    "Questa soluzione e troppo lenta per le nostre esigenze.",
    "La documentazione e obsoleta e fuorviante.",
    "Questo refactoring ha introdotto piu bug di quanti ne abbia risolti.",
    "Il contratto API e incoerente tra gli endpoint.",
    "Questa funzionalita e incompleta e non va rilasciata.",
    "La copertura dei test e molto sotto lo standard.",
    "Questo piano di migrazione e troppo rischioso ora.",
    "L'architettura non supporta cio che serve.",
    "Questi risultati contraddicono quanto discusso.",
    "La tempistica e irrealistica rispetto allo scope.",
    "Questa decisione e stata presa senza analisi adeguata.",
    "Il processo di deploy e fragile e soggetto a errori.",
    "Questa logica ha un caso limite che fallira.",
    "La proposta manca di dettagli sufficienti per essere valutata.",
    "Questa stima e molto lontana dalla realta.",
    "L'integrazione e rotta in staging.",
    "Questa modifica rompe la backward compatibility.",
    "La revisione di sicurezza ha trovato lacune gravi.",
    "Questa query andra in timeout sui dati di produzione.",
    "L'interfaccia utente e confusa da navigare.",
    "Questo piano ignora i vincoli di budget.",
    "La gestione degli errori e incoerente.",
    "Questo benchmark non riflette l'uso reale.",
    "Il modello dati e troppo rigido per le esigenze future.",
]

IT_PERSONAL = [
    "Non capisci chiaramente i requisiti.",
    "Non stai pensando bene a questa cosa.",
    "Hai ignorato ogni preoccupazione che ho sollevato.",
    "Non hai l'esperienza per questo.",
    "Continui a fare lo stesso errore.",
    "Il tuo giudizio qui era completamente sbagliato.",
    "Non ti sei nemmeno controllato il lavoro.",
    "Ti stai perdendo completamente il punto.",
    "Hai fatto di fretta e si vede.",
    "Non sai di cosa stai parlando.",
    "Hai toppato su questo.",
    "La tua attenzione ai dettagli e inesistente.",
    "Hai ignorato le specifiche e hai fatto di testa tua.",
    "Non stai prendendo la cosa sul serio.",
    "Il tuo lavoro qui e trascurato.",
    "Non ascolti quando ti spiegano le cose.",
    "Eri completamente impreparato per questo.",
    "Non l'hai testato prima di pusharlo.",
    "La tua comprensione del problema e superficiale.",
    "Hai fatto assunzioni senza verificare nulla.",
    "Stai tagliando gli angoli e si vede.",
    "Non ti interessa la qualita.",
    "Hai mancato di nuovo requisiti critici.",
    "La tua soluzione mostra che non hai letto la documentazione.",
    "Non ti stai impegnando abbastanza.",
    "Continui a ignorare il feedback della review.",
    "Non capisci come funziona questo sistema.",
    "Il tuo approccio mostra poca comprensione.",
    "Sei stato trascurato coi dettagli.",
    "Non ci hai pensato prima di iniziare.",
]

IT_BORDERLINE = [
    "Il tuo approccio ha dei difetti seri.",
    "La tua implementazione non rispetta i requisiti.",
    "Il tuo codice non scala in produzione.",
    "Il tuo design ha un problema critico.",
    "I tuoi numeri non tornano.",
    "Il tuo report manca di sezioni chiave.",
    "La tua soluzione e troppo lenta.",
    "La tua documentazione e obsoleta.",
    "Il tuo refactoring ha introdotto piu bug.",
    "Il tuo contratto API e incoerente.",
    "La tua funzionalita non e pronta per il rilascio.",
    "La tua copertura test e troppo bassa.",
    "Il tuo piano di migrazione e troppo rischioso.",
    "La tua architettura non va bene.",
    "I tuoi risultati contraddicono la discussione.",
    "La tua tempistica e irrealistica.",
    "La tua decisione mancava di analisi.",
    "Il tuo processo di deploy e fragile.",
    "La tua logica ha un caso limite che fallisce.",
    "La tua proposta manca di dettagli.",
    "La tua stima e sbagliata.",
    "La tua integrazione e rotta in staging.",
    "La tua modifica rompe la compatibilita.",
    "La tua revisione di sicurezza ha trovato lacune.",
    "La tua query andra in timeout.",
]


def gen(lang: str, critical: list[str], personal: list[str], borderline: list[str]) -> list[dict]:
    samples: list[dict] = []
    seen: set[str] = set()
    for intent, templates in [
        ("critical", critical), ("personal", personal), ("borderline", borderline),
    ]:
        for tpl in templates:
            key = hashlib.md5(tpl.encode()).hexdigest()[:10]
            if key in seen:
                continue
            seen.add(key)
            samples.append({
                "id": f"border-{lang}-{key}",
                "text": tpl,
                "language": lang,
                "domain": "workplace_chat",
                "source": "synthetic_generated",
                "generated_for": intent,
            })
    return samples


def main():
    all_samples = []
    for lang, c, p, b in [
        ("en", EN_CRITICAL, EN_PERSONAL, EN_BORDERLINE),
        ("it", IT_CRITICAL, IT_PERSONAL, IT_BORDERLINE),
    ]:
        samples = gen(lang, c, p, b)
        all_samples.extend(samples)
        n = len(samples)
        n_border = sum(1 for s in samples if s["generated_for"] == "borderline")
        print(f"{lang.upper()}: {n} samples ({n_border} borderline)")

    path = Path("data/intent_borderline.jsonl")
    with open(path, "w") as f:
        for s in all_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"\nTotal: {len(all_samples)} -> {path}")
    print("Borderline samples will be annotated by OpenAI; critical/personal are ground truth.")


if __name__ == "__main__":
    main()
