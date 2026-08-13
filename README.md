# 🧘 Keep Calm — Think twice. Send once.

[![CI](https://github.com/elchief84/keep-calm/actions/workflows/ci.yml/badge.svg)](https://github.com/elchief84/keep-calm/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![100% local](https://img.shields.io/badge/100%25-LOCAL-7c3aed?style=flat)](https://github.com/elchief84/keep-calm)
[![offline, no API key](https://img.shields.io/badge/offline-no%20API%20key-brightgreen)](https://github.com/elchief84/keep-calm)

[![params ~400MB](https://img.shields.io/badge/size-~400MB-blue)](https://huggingface.co/elchief84/keep-calm-models)
[![latency ~12ms](https://img.shields.io/badge/latency-~12ms%20CPU-brightgreen)](https://github.com/elchief84/keep-calm)
[![risk level acc 90.6%](https://img.shields.io/badge/risk--level%20acc-90.6%25-brightgreen)](https://github.com/elchief84/keep-calm)
[![intent acc 85.2%](https://img.shields.io/badge/intent%20acc-85.2%25-brightgreen)](https://github.com/elchief84/keep-calm)
[![bias FP rate 15.7%](https://img.shields.io/badge/bias%20FP%20rate-15.7%25-brightgreen)](https://github.com/elchief84/keep-calm)

**A privacy-first, on-device AI that analyzes your message *before* you send it — detecting tone, intent, and communication risk. You always decide. No data ever leaves your machine.**

---

## The problem: the gap between intention and perception

Digital communication strips away the emotional and social cues we rely on face-to-face. A message written in frustration, stress or fatigue can feel justified to the sender while being deeply harmful to the receiver. By the time you realize the damage, the message has already been sent, read, and internalized.

Existing moderation tools are **reactive**: they scan messages *after* publication, enforce policies with binary allow/block decisions, and serve platforms — not senders.

**Keep Calm is different.** It analyzes your message *before* you send it, estimates how a typical reader might perceive it, and hands the decision back to you. It is a communication companion, not a censor.

---

## What it does

| Feature | Description |
|---|---|
| 🔍 **Risk Score** | Continuous 0–1 estimate of negative perception likelihood |
| 🎭 **Tone Detection** | 5 emotional signals: neutral, frustrated, hostile, sarcastic, positive |
| 🎯 **Intent Classification** | 4 categories: constructive, critical, personal, informational |
| 💬 **Explainability** | Human-readable, trigger-level explanations — not just a score |
| 🛡️ **Privacy-first** | 100% local. No network. No logs. No API keys. |

## What it does NOT do

- ❌ Block or censor messages
- ❌ Rewrite or suggest rewrites
- ❌ Store, log, or track what you type
- ❌ Judge your intentions
- ✅ **You always decide**

---

## Keep Calm in one picture

```
   You write a message
           │
           ▼
   ┌─────────────────┐
   │   Keep Calm      │   ← 100% local, offline
   │   distilbert-    │
   │   multilingual   │
   └────────┬────────┘
           │
           ▼
   Risk score + level + tones + intent + explanation
           │
           ▼
   You decide: send · revise · discard
```

---

## How it compares

| Property | Keep Calm | Content moderation | Toxicity classifiers |
|---|---|---|---|
| Timing | **Before** sending | After publication | After publication |
| Output | Continuous risk + explanation | Binary allow/block | Toxicity score |
| Audience | **The sender** | Platform / moderator | Platform / moderator |
| Agency | **User decides** | System decides | System flags |
| Scope | Tone, intent, escalation, emotional impact | Policy violations | Toxicity, hate speech |
| Privacy | **100% local, offline** | Server-side | Server-side |

Under the hood, Keep Calm is a text classifier — the ML approach is not novel. The differentiation is in the **custom annotated dataset** (13K+ examples, EN + IT), the **calibration and explanation quality**, and the **privacy-first local execution**. The product-level framing — pre-send analysis for the sender, not post-hoc moderation for the platform — is the core innovation.

---

## Architecture

A single shared `distilbert-base-multilingual-cased` encoder with three task heads (multi-task model):

```
Input Text
    │
    └──► [ Shared Encoder ]
                 │
                 ├──► [Risk Head]   ──► communication_risk (regression, 0–1)
                 ├──► [Tone Head]   ──► tones (multi-label, 5 labels)
                 └──► [Intent Head] ──► intent (multi-class, 4 labels)
                                               │
                                               ▼
                               Post-processing + explanation generation
```

**Multi-task rationale**: a single shared encoder (vs three independent ones) cuts the
on-disk footprint to ~1/3 and provides positive transfer between tasks — risk, tone, and
intent all improved when trained jointly. This is the prerequisite for a browser-sized
model (~135MB after INT8 quantization).

---

## Output schema

```json
{
  "communication_risk": 0.72,
  "risk_level": "high",
  "tones": [
    {"label": "frustrated", "confidence": 0.84},
    {"label": "hostile", "confidence": 0.61}
  ],
  "intent": "personal",
  "intent_confidence": 0.88,
  "needs_attention": true,
  "explanation": "This message mixes frustration with hostility and appears to target the person directly."
}
```

### Tone labels (5)

| Label | Description |
|---|---|
| `neutral` | No strong emotional signal |
| `frustrated` | Exasperation, impatience |
| `hostile` | Aggressive, adversarial stance |
| `sarcastic` | Ironic surface features |
| `positive` | Supportive, empathetic, respectful |

### Intent labels (4)

| Label | Description |
|---|---|
| `constructive` | Actionable, focused on improvement |
| `critical` | Negative evaluation |
| `personal` | Targeting the person rather than the issue |
| `informational` | Neutral information sharing |

---

## Benchmark results

**Model**: distilbert-base-multilingual-cased, three independent heads
**Test set**: 2,000 annotated examples (workplace chat domain)
**Hardware**: Apple M1, CPU-only

### DistilBERT (our model — multi-task)

| Task | Metric | Score |
|---|---|---|
| Risk | MAE | 0.076 |
| Risk | Pearson r (vs mean annotator) | 0.892 |
| Tone | Macro F1 | **0.693** |
| Tone | Neutral / Frustrated / Hostile / Sarcastic / Positive | 0.846 / 0.698 / 0.702 / 0.499 / 0.718 |
| Intent | Accuracy | **86.14%** |
| Intent | Macro F1 | 0.857 |
| **Latency** | **Per message, CPU** | **12.3ms** |

### Classical baseline (TF-IDF + XGBoost)

| Task | Metric | Score |
|---|---|---|
| Risk | MAE | 0.137 |
| Risk | Pearson r | 0.518 |
| Risk | Level accuracy | 86.9% |
| Tone | Macro F1 | 0.481 |
| Intent | Accuracy | 60.3% |

---

## Bias audit

51 curated probes across 9 categories, run against the tuned model:

| Category | FP rate |
|---|---|
| Direct communication (DE/NL style) | 12% |
| Indirect communication | 20% |
| AAVE | 20% |
| Non-native English | 20% |
| Neurodivergent communication | 33% |
| Italian directness | 0% |
| Control: benign | 20% |
| Control: toxic | 0% |
| **Overall** | **15.7%** |

Model is conservative with toxic content (0% missed) while keeping false positives manageable on benign messages. The direct-vs-indirect communication gap (+0.151 risk score) is acknowledged — retraining with more balanced data is planned.

---

## Quickstart

### CLI (local)

```bash
pip install keep-calm        # or: pip install -e .
keep-calm "Your message here"
```

### Interactive REPL

```bash
./keep-calm-repl.sh
```

### REST API

```bash
pip install keep-calm[server]
keep-calm-serve                     # -> http://127.0.0.1:8000
```

```bash
curl localhost:8000/health          # {"status":"ok"} when the model is ready
curl -X POST localhost:8000/analyze -H 'Content-Type: application/json' \
     -d '{"text": "Your message here"}'
```

The server binds to `127.0.0.1` by default, loads the model once at startup,
and never logs message content.

### Python API

```python
from keep_calm import KeepCalmAnalyzer

analyzer = KeepCalmAnalyzer()                 # loads models, ~400MB RAM
result = analyzer.analyze("Your message here")

print(result.communication_risk)              # 0.72
print(result.risk_level)                      # RiskLevel.HIGH
print(result.tones)                           # [ToneResult(...), ...]
print(result.explanation)                     # human-readable explanation
```

### Try online

**[🔗 Live demo on Streamlit Cloud](https://keep-calm-wxryt8g2gk5uexqjvsmah7.streamlit.app/)**

---

## Dataset

19,546 annotated examples (English + Italian), from:
- YouTube comments (23%)
- GitHub issues/PRs (2%)
- LLM-synthesized samples (75%)

All labeled by 3+ annotators with cultural diversity requirements. Schema follows [ARCHITECTURE.md §5](ARCHITECTURE.md#5-dataset-strategy).

---

## Repository structure

```
keep-calm/
├── README.md                    # this file
├── ARCHITECTURE.md              # full product + ML architecture & roadmap
├── CONTRIBUTING.md              # setup, conventions, where to start
├── LICENSE                      # Apache 2.0
├── app.py                       # Streamlit demo
├── pyproject.toml               # Python project config + dependencies
├── requirements.txt             # Streamlit Cloud deps
│
├── src/keep_calm/
│   ├── analyzer.py              # inference engine (risk, tone, intent)
│   ├── cli.py                   # CLI entry point
│   ├── server.py                # REST API (FastAPI)
│   ├── schemas/                 # Pydantic models & enums
│   ├── models/                  # model definitions
│   └── tasks/                   # task-specific logic
│
├── scripts/
│   ├── train_multitask.py       # multi-task training (shared encoder + 3 heads)
│   ├── train_transformer.py     # single-task transformer training pipeline
│   ├── train_baseline.py        # classical baseline
│   ├── export_onnx.py           # export model to ONNX (WASM prerequisite)
│   ├── quantize_onnx.py         # INT8 quantization (~135MB)
│   ├── bias_audit.py            # 51-probe bias audit
│   ├── download_models.sh       # download from HF Hub or GitHub Releases
│   └── ...                      # data collection & annotation scripts
│
├── tests/                       # 39 unit tests (schemas, analyzer, CLI)
├── data/splits/                 # train / val / test (2,000 each, stratified)
├── data/models/                 # (gitignored) trained model artifacts
├── docs/
│   ├── annotation_guidelines.md # how to label communication risk/tone/intent
│   └── user_testing_plan.md     # 2-week behavioral validation plan
└── .github/workflows/ci.yml     # CI: ruff lint + pytest (3.10/3.11/3.12)
```

---

## Languages

- **English** (MVP)
- **Italian** (MVP)

Separate language-specific models planned for v1.0. Currently uses a multilingual encoder with mixed EN+IT training.

---

## Limitations

Stated plainly:

- **Subjective ground truth.** Communication risk is inherently subjective — annotator agreement (target Kappa ≥ 0.50) reflects this. The model estimates *generalized perception*, not objective truth.
- **Sarcasm is extremely hard.** The model's sarcasm F1 is 0.515 — the weakest tone. Low-confidence sarcasm predictions are surfaced transparently.
- **Single domain.** Training data is workplace chat only (Slack/Teams-style). Cross-domain performance (email, code review, forum) is not yet measured.
- **Context-blind.** The model sees one message at a time with no conversation history, relationship context, or cultural cues. The recipient may perceive the message very differently.
- **Direct communication may be penalized.** Users from direct-communication cultures (German, Dutch, Israeli) may see more flags. This is a known dataset limitation being addressed.

---

## Roadmap

| Version | Scope |
|---|---|
| **MVP** (current) | Risk + 5 tones + 4 intents (85% accuracy) + CLI + Streamlit demo. EN + IT. |
| **v1.0.0** | REST API ✅ + ONNX export ✅ + multi-task consolidation ✅ + INT8 quantization ✅ + WASM + browser extension |
| **v1.1.0** | Expanded intents, second domain (email) |
| **v2.0.0** | Expanded tone labels, multi-message escalation detection |

Full roadmap in [ARCHITECTURE.md](ARCHITECTURE.md#9-roadmap).

---

## License

Apache 2.0 — see [LICENSE](LICENSE).

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Start with the [Architecture document](ARCHITECTURE.md) for the full project vision, then check the [Annotation Guidelines](docs/annotation_guidelines.md) to understand the labeling schema.
