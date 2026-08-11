---
language:
  - en
  - it
license: apache-2.0
tags:
  - communication
  - tone-detection
  - sentiment-analysis
  - text-classification
  - privacy
datasets:
  - elchief84/keep-calm-dataset
metrics:
  - accuracy
  - f1
  - mae
  - pearson-r
model-index:
  - name: keep-calm
    results:
      - task:
          type: text-classification
        dataset:
          name: Keep Calm test set
          type: workplace-chat
        metrics:
          - name: Risk MAE
            type: mae
            value: 0.100
          - name: Risk Pearson r
            type: pearson-r
            value: 0.700
          - name: Risk Level Accuracy
            type: accuracy
            value: 0.906
      - task:
          type: multi-label-classification
        dataset:
          name: Keep Calm test set
          type: workplace-chat
        metrics:
          - name: Tone Macro F1
            type: f1
            value: 0.677
      - task:
          type: text-classification
        dataset:
          name: Keep Calm test set
          type: workplace-chat
        metrics:
          - name: Intent Accuracy
            type: accuracy
            value: 0.706
---

# Keep Calm — Communication Risk Analyzer

A privacy-first, on-device model for pre-send communication analysis.
Detects tone, intent, and communication risk in English and Italian text.

## Model description

Three independent single-task models sharing a `distilbert-base-multilingual-cased` backbone:

- **Risk model**: regression head predicting continuous 0–1 communication risk
- **Tone model**: multi-label classification across 5 tones
- **Intent model**: multi-class classification across 4 intents

### Tone labels

`neutral` · `frustrated` · `hostile` · `sarcastic` · `positive`

### Intent labels

`constructive` · `critical` · `personal` · `informational`

## Intended use

Pre-send analysis of workplace text communication. The user writes a message, invokes Keep Calm, sees the analysis, and decides whether to send, revise, or discard.

**Not** intended as a moderation or censorship tool. The model estimates perception, not objective truth.

## Out-of-scope use

- Automated content moderation
- Post-hoc message flagging
- Surveillance or monitoring without consent
- Analyzing messages in domains other than workplace chat
- Languages other than English and Italian

## Training data

13,329 annotated examples (English + Italian), workplace chat domain. Sources: YouTube comments, GitHub PRs/issues, LLM-synthesized samples. All labeled by 3+ culturally diverse annotators.

## Bias, risks, and limitations

- **Direct communication penalty**: users from direct-communication cultures (German, Dutch) may receive higher risk scores
- **Sarcasm is hard**: the model's weakest tone (F1 = 0.515); low-confidence predictions are surfaced
- **Single domain**: trained only on workplace chat; cross-domain performance unmeasured
- **Context-blind**: no conversation history, relationship context, or cultural cues
- **Subjective ground truth**: annotator agreement reflects the inherent subjectivity of communication perception
- **Intent classification**: the weakest task at 70.6% accuracy

## Evaluation results

| Task | Metric | Score |
|---|---|---|
| Risk | MAE | 0.100 |
| Risk | Pearson r | 0.700 |
| Risk | Level accuracy | **90.6%** |
| Tone | Macro F1 | **0.677** |
| Intent | Accuracy | **70.6%** |

**Latency**: 12.3ms per message on Apple M1 (CPU-only).

**Bias audit FP rate**: 15.7% (51 curated probes across 9 categories).

## Hardware

- **Inference**: CPU-only, ~400MB RAM, ~12ms per message
- **Training**: single 16GB GPU (reference: RTX 5060 Ti)

## How to use

```python
from keep_calm import KeepCalmAnalyzer

analyzer = KeepCalmAnalyzer()
result = analyzer.analyze("Your message here")

print(result.communication_risk)  # 0.72
print(result.risk_level)          # RiskLevel.HIGH
print(result.explanation)         # human-readable
```

## Citation

```bibtex
@software{keep_calm,
  title = {Keep Calm: Pre-send Communication Risk Analysis},
  year = {2026},
  url = {https://github.com/elchief84/keep-calm}
}
```

## License

Apache 2.0
