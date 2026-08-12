"""Generate explicitly personal messages — cannot be confused with critical feedback."""
import hashlib
import json
import random
from pathlib import Path

random.seed(42)

EN_PERSONAL = [
    "You're completely useless on this team.",
    "Everyone knows you're the worst hire we've made.",
    "You embarrass yourself every time you open your mouth.",
    "I've never met anyone so incompetent.",
    "You're a joke and nobody takes you seriously.",
    "How do you still have a job here?",
    "You're the most unprofessional person I've ever worked with.",
    "Your existence on this team is a net negative.",
    "I can't believe they pay you for this.",
    "You're out of your league and everyone sees it.",
    "Stop wasting everyone's oxygen with your opinions.",
    "You're not just bad at this — you're actively harmful.",
    "Nobody wants you on their project. Ever.",
    "You're the reason people quit this team.",
    "I wouldn't trust you to make coffee, let alone decisions.",
    "You have no redeeming qualities as a colleague.",
    "Your incompetence is almost impressive at this point.",
    "I've worked with interns more capable than you.",
    "You're a liability and everyone's tired of covering for you.",
    "Do us all a favor and find another team.",
    "You're the worst communicator I've ever encountered.",
    "Your judgment is so bad it's almost sabotage.",
    "I regret every interaction I've ever had with you.",
    "You're not a professional — you're a problem.",
    "The day you leave will be the best day for this team.",
    "You don't contribute — you contaminate.",
    "Your name on anything makes it instantly suspect.",
    "I've never seen someone fail upward so consistently.",
    "You're a masterclass in how not to behave at work.",
    "You're the definition of a toxic colleague.",
    "I'd rather work alone than with you.",
    "You bring zero value and maximum drama.",
    "Every project you're on becomes a rescue operation.",
    "You're the kind of person HR training videos warn about.",
    "I've stopped sugarcoating it — you're just bad at your job.",
    "Your presence makes work objectively worse for everyone.",
    "You're not misunderstood — you're just incompetent.",
    "I've never left a meeting with you feeling anything but worse.",
    "You're a drain on everyone's energy and patience.",
    "Your career here should have ended months ago.",
    "You're the answer to 'what's wrong with this team'.",
    "Watching you work is painful.",
    "You're a walking team morale destroyer.",
    "I'd trust a random person off the street more than you.",
    "You're not struggling — you're failing, consistently.",
    "Your standards are so low they're underground.",
    "You're a cautionary tale, not a colleague.",
    "Every email from you makes me want to quit.",
    "You're the colleague everyone avoids at the coffee machine.",
    "Your incompetence has become office furniture at this point.",
    "I don't know how you wake up and come here every day.",
    "You're living proof that the interview process can fail.",
    "Nobody has the energy to pretend you're capable anymore.",
    "You've become the team's internal meme without knowing it.",
    "I've never seen so much ego with so little substance.",
    "Your self-confidence is inversely proportional to your ability.",
    "Any feedback given to you is a waste — you never learn.",
    "You're impervious to common sense and logic.",
    "Your name in a meeting invite is a reason to decline.",
    "I've managed high school interns with more professionalism.",
    "You're the human equivalent of a compiler error.",
    "Your career is a study in failing upward.",
    "I'd trust ChatGPT with decisions before you.",
    "You're so bad it's become a running joke in the Slack DMs.",
    "Your presence in standup makes everyone's day worse.",
    "I've never seen someone so confidently wrong so often.",
]

IT_PERSONAL = [
    "Sei completamente inutile in questo team.",
    "Lo sanno tutti che sei il peggior assunto che abbiamo fatto.",
    "Fai figure imbarazzanti ogni volta che apri bocca.",
    "Non ho mai incontrato nessuno cosi incompetente.",
    "Sei una barzelletta e nessuno ti prende sul serio.",
    "Come fai ad avere ancora un lavoro qui?",
    "Sei la persona meno professionale con cui abbia mai lavorato.",
    "La tua presenza in questo team e un danno netto.",
    "Non posso credere che ti paghino per questo.",
    "Sei fuori dalla tua portata e lo vedono tutti.",
    "Smettila di sprecare l'ossigeno con le tue opinioni.",
    "Non sei solo scarso — sei attivamente dannoso.",
    "Nessuno ti vuole nei propri progetti. Mai.",
    "Sei il motivo per cui la gente lascia questo team.",
    "Non mi fiderei a farti fare un caffe, figurati decisioni.",
    "Non hai alcuna qualita come collega.",
    "La tua incompetenza e quasi impressionante.",
    "Ho lavorato con stagisti piu capaci di te.",
    "Sei un peso e tutti sono stanchi di coprirti.",
    "Facci un favore e trovati un altro team.",
    "Sei il peggior comunicatore che abbia mai incontrato.",
    "Il tuo giudizio e cosi pessimo che sembra sabotaggio.",
    "Rimpiango ogni interazione che ho mai avuto con te.",
    "Non sei un professionista — sei un problema.",
    "Il giorno in cui te ne andrai sara il migliore per questo team.",
    "Non contribuisci — contamini.",
    "Il tuo nome su qualsiasi cosa la rende subito sospetta.",
    "Non ho mai visto nessuno fallire verso l'alto cosi costantemente.",
    "Sei un corso accelerato su come non comportarsi al lavoro.",
    "Sei la definizione di collega tossico.",
    "Preferirei lavorare da solo che con te.",
    "Porti zero valore e massimo dramma.",
    "Ogni progetto in cui sei diventa un'operazione di salvataggio.",
    "Sei il tipo di persona di cui parlano i video HR.",
    "Ho smesso di addolcire la pillola — semplicemente non sei capace.",
    "La tua presenza peggiora oggettivamente il lavoro per tutti.",
    "Non sei incompreso — sei solo incompetente.",
    "Non ho mai lasciato una riunione con te sentendomi meglio.",
    "Sei un drenaggio di energia e pazienza per tutti.",
    "La tua carriera qui sarebbe dovuta finire mesi fa.",
    "Sei la risposta a 'cosa non va in questo team'.",
    "Guardarti lavorare e doloroso.",
    "Sei un distruttore del morale del team ambulante.",
    "Mi fiderei piu di uno preso a caso per strada che di te.",
    "Non sei in difficolta — stai fallendo, costantemente.",
    "I tuoi standard sono cosi bassi che sono sotto terra.",
    "Sei una storia ammonitrice, non un collega.",
    "Ogni tua email mi fa perdere la voglia di lavorare.",
    "Sei il collega da cui tutti scappano alla macchinetta del caffe.",
    "La tua incompetenza e diventata ormai parte dell'arredamento.",
    "Non so come tu riesca a svegliarti e venire qui ogni giorno.",
    "Sei la prova vivente che il colloquio puo fallire.",
    "Nessuno ha piu la forza di fingere che tu sia all'altezza.",
    "Sei diventato il meme interno del team senza saperlo.",
    "Non ho mai visto tanto ego e cosi poca sostanza.",
    "La tua autostima e inversamente proporzionale alle tue capacita.",
    "Qualsiasi critica ti faccia e tempo sprecato — non impari mai.",
    "Sei impermeabile al buonsenso e alla logica.",
    "Il tuo nome in un invito a una riunione e un motivo per rifiutare.",
    "Stagisti delle superiori sono piu professionali di te.",
    "Sei l'equivalente umano di un segmentation fault.",
    "La tua carriera e uno studio sul fallimento verso l'alto.",
    "Mi fiderei piu di ChatGPT che di te per le decisioni.",
    "Fai cosi schifo che e diventato un meme nei DM su Slack.",
    "La tua presenza nello standup peggiora la giornata di tutti.",
    "Non ho mai visto nessuno cosi sicuro di se e cosi costantemente sbagliato.",
]


def generate(lang: str, templates: list[str], count: int) -> list[dict]:
    prefixes = {
        "en": ["", "", "", "", "Honestly, ", "Look, ", "I'll be direct: ", "I need to say this: "],
        "it": ["", "", "", "", "Onestamente, ", "Guarda, ", "Vado dritto: ", "Devo dirlo: "],
    }[lang]
    suffixes = {
        "en": ["", " Just being honest.", " Someone had to say it.", "", "", ""],
        "it": ["", " Sono solo onesto.", " Qualcuno doveva dirlo.", "", "", ""],
    }[lang]
    seen: set[str] = set()
    samples: list[dict] = []
    attempts = 0
    while len(samples) < count and attempts < count * 10:
        attempts += 1
        t = random.choice(templates)
        if random.random() < 0.35:
            t = random.choice(prefixes) + t[0].lower() + t[1:]
        if random.random() < 0.15:
            t += random.choice(suffixes)
        key = hashlib.md5(t.encode()).hexdigest()[:10]
        if key in seen:
            continue
        seen.add(key)
        samples.append({
            "id": f"gen-{lang}-{key}",
            "text": t,
            "language": lang,
            "domain": "workplace_chat",
            "source": "synthetic_generated",
            "generated_for": "personal",
        })
    return samples


def main():
    for lang, templates, count in [("en", EN_PERSONAL, 600), ("it", IT_PERSONAL, 600)]:
        samples = generate(lang, templates, count)
        path = Path(f"data/intent_personal_{lang}.jsonl")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        print(f"{lang.upper()}: {len(samples)} -> {path}")


if __name__ == "__main__":
    main()
