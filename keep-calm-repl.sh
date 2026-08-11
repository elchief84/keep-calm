#!/bin/bash
# Keep Calm - modalità interattiva (modelli caricati una volta sola)
.venv/bin/python3 -c "
from keep_calm.analyzer import KeepCalmAnalyzer
print('Keep Calm - scrivi un messaggio e premi Invio. \"quit\" per uscire.')
a = KeepCalmAnalyzer()
print('Modelli caricati. Pronto.\n')
while True:
    try:
        msg = input('> ')
    except (EOFError, KeyboardInterrupt):
        break
    if not msg.strip(): continue
    if msg.lower() == 'quit': break
    r = a.analyze(msg)
    print(f'  Rischio: {r.communication_risk:.2f} ({r.risk_level.value})')
    print(f'  Toni: {\", \".join(f\"{t.label.value}({t.confidence:.2f})\" for t in r.tones)}')
    print(f'  Intento: {r.intent.value}')
    print(f'  {r.explanation}\n')
print('Ciao!')
" 2>/dev/null