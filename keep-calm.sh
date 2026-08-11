#!/bin/bash
.venv/bin/python3 -c "
from keep_calm.analyzer import KeepCalmAnalyzer
a = KeepCalmAnalyzer()
r = a.analyze('$*')
print(f'\nRisk: {r.communication_risk:.2f} ({r.risk_level.value})')
print(f'Tones: {\", \".join(f\"{t.label.value}({t.confidence:.2f})\" for t in r.tones)}')
print(f'Intent: {r.intent.value}')
print(f'Needs attention: {\"yes\" if r.needs_attention else \"no\"}')
print(f'\n{r.explanation}')
" 2>/dev/null
