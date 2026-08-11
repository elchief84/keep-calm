# Keep Calm — Project Architecture

> Think twice. Send once.

This document defines the technical and product foundation of the Keep Calm project.

It is the result of a deep analysis of the initial vision, translated into a structured, actionable blueprint suitable for guiding development from research to production.

---

## Table of Contents

1. [Product Analysis](#1-product-analysis)
2. [Product Boundaries](#2-product-boundaries)
3. [MVP Definition](#3-mvp-definition)
4. [Machine Learning Problem Definition](#4-machine-learning-problem-definition)
5. [Dataset Strategy](#5-dataset-strategy)
6. [Model Strategy](#6-model-strategy)
7. [Technical Architecture](#7-technical-architecture)
8. [Risks and Challenges](#8-risks-and-challenges)
9. [Roadmap](#9-roadmap)

---

## 1. Product Analysis

### What Problem Does Keep Calm Solve?

Digital communication strips away the emotional and social cues present in face-to-face interaction. A message written in frustration, fatigue, or stress can feel justified to the sender while being deeply harmful to the receiver. By the time the sender realizes the damage, the message has already been sent, read, and internalized.

Keep Calm addresses the **gap between intention and perception** in written communication.

The core problem is not that people intend to be harmful. The core problem is that people **do not see** how their message will land before it lands.

Specific manifestations:

- **Workplace damage**: A frustrated code review comment, a terse Slack reply, or an angry email can damage professional relationships, erode psychological safety, and create lasting resentment.
- **Open source toxicity**: Maintainers and contributors operating under stress produce communication that drives away newcomers and poisons community culture.
- **Escalation spirals**: Online discussions escalate incrementally. Each message is slightly more aggressive than the last, and neither party notices the trajectory until the conversation has become hostile.
- **Personal relationships**: Messages sent in anger to friends, partners, or family members cause harm that is difficult to repair.

### Why Existing Moderation Systems Are Insufficient

Current approaches to harmful communication share a fundamental limitation: **they are reactive, not preventive**.

| Dimension | Content Moderation | Keep Calm |
|---|---|---|
| Timing | After publication | Before sending |
| Focus | Policy violations | Communication quality |
| Output | Binary (allow/block) | Continuous risk + explanation |
| Audience | Platform / moderator | The sender |
| Goal | Protect the platform | Help the communicator |
| Scope | Toxicity, hate speech, spam | Tone, intent, escalation, emotional impact |
| Agency | System decides | User decides |

Content moderation answers: *"Does this message violate a policy?"*

Keep Calm answers: *"How might this message be received?"*

These are fundamentally different questions. A message can be:

- **Non-toxic but harmful**: "I'm not surprised you couldn't figure this out." (passive-aggressive, no policy violation)
- **Technically polite but hostile**: "As I'm sure you're aware, this was addressed three months ago." (condescending, no slur)
- **Frustrated but constructive**: "I'm really frustrated with this approach — can we discuss alternatives?" (high emotion, healthy communication)

Moderation systems miss the first two. Keep Calm is designed to catch all three and let the user decide.

### Target Users

Keep Calm is designed for a behavioral profile, not a specific profession:

- Communicates primarily via text (email, chat, forums, code reviews, social platforms)
- Experiences emotional reactions while writing (frustration, anger, stress, fatigue)
- Has expressed regret about past messages ("I shouldn't have sent that")
- Is motivated to improve communication but would not want a tool that censors or rewrites

**MVP beachhead**: The initial domain is **workplace text communication** (Slack/Teams-style messaging). This domain provides available training data (public chat archives, open source project discussions) and a narrow enough scope to build a focused dataset.

**Post-MVP expansion** targets are defined by behavioral context:
- High-stress customer interactions (support agents)
- Emotionally charged personal communication
- Cross-cultural professional communication

### Use Cases

1. **Pre-send analysis**: User writes a message, Keep Calm analyzes it before sending, user decides whether to revise.
2. **Code review awareness**: A reviewer's comment is flagged as potentially dismissive; the reviewer rephrases to focus on the code, not the person.
3. **Escalation detection**: A Slack thread is getting heated; Keep Calm signals that the conversation trajectory is worsening.
4. **Email drafting**: A manager about to send frustrated feedback pauses, sees the risk score, and rewrites with more constructive framing.
5. **Self-awareness tool**: Over time, users learn their own communication patterns and triggers.

### What Makes This Different from Toxicity Classifiers?

Toxicity classifiers (Perspective API, Jigsaw models, etc.) are designed to answer one question: *Is this text toxic?*

Keep Calm differs in five ways:

1. **Spectrum, not binary**: Communication quality exists on a spectrum. Keep Calm provides a continuous risk score, not a pass/fail judgment.
2. **Multi-dimensional analysis**: Tone, intent, and emotional impact are analyzed simultaneously. Toxicity is one signal, not the only signal.
3. **Explainability**: Keep Calm explains *why* a message might be perceived negatively, with specific trigger-level explanations, not just a score.
4. **User agency**: The system never blocks, censors, or rewrites. It informs. The user retains full control.
5. **Preventive, not punitive**: The analysis happens before the message is sent. No punishment, no flag on a record, no moderation action.

**Honest note on differentiation**: Under the hood, Keep Calm is a text classifier — the ML approach is not novel. The differentiation is in the **custom annotated dataset** (hard to replicate), the **calibration and explanation quality**, and the **privacy-first local execution**. The product-level framing (pre-send analysis for the sender, not post-hoc moderation for the platform) is the core innovation.

### What the Model Actually Predicts

The model does not predict an objective property of the text. It estimates:

> How this message is likely to be perceived by a generalized observer, absent specific relational or cultural context.

The output communicates this uncertainty: *"This message may be perceived as hostile by a typical reader. Your specific recipient may perceive it differently."*

---

## 2. Product Boundaries

### What Keep Calm Does

- Analyzes the emotional tone of a text message (5 tone categories)
- Classifies communication intent (4 intent categories)
- Estimates how the message is likely to be perceived by a generalized observer
- Provides specific, trigger-level explanations of what caused the assessment
- Provides positive feedback when communication is healthy or constructive
- Delivers a communication risk score with confidence levels
- Runs entirely locally with no data leaving the user's machine
- Works offline with no external API dependencies
- Supports English and Italian from the start (separate language-specific models)

### What Keep Calm Does NOT Do

- **Does not block or prevent sending**: The user always has the final decision.
- **Does not rewrite or suggest rewrites**: Keep Calm raises awareness; it does not author text.
- **Does not judge user intentions**: It analyzes probable perception, not intent. A well-intentioned message can still land poorly.
- **Does not store or log messages**: No message content is persisted. Analysis is ephemeral.
- **Does not enforce communication policies**: It is not a compliance tool.
- **Does not detect illegal content**: It is not a legal or safety moderation system.
- **Does not replace human judgment**: It is a signal, not an authority.
- **Does not support batch analysis**: Single-message analysis only. No batch API, no aggregation, no dashboards.
- **Does not analyze images, audio, or video**: Text only.
- **Does not provide therapy or mental health advice**: It is a communication tool, not a clinical instrument.
- **Does not score the "correctness" of opinions**: It analyzes how something is said, not what is said.

### Anti-Patterns to Avoid

- **Gamification of scores**: Risk scores should not become a game to minimize. The goal is awareness, not optimization.
- **Surveillance**: Keep Calm must never be deployed as a tool for managers to monitor employee communication quality without consent. The tool provides no batch API, no reporting, and no aggregation features to add technical friction against this misuse.
- **Mandatory usage**: If an organization requires Keep Calm analysis before sending, it becomes a censor, not a companion.
- **Over-reliance**: Users should develop their own communication awareness, not become dependent on the tool.
- **Pathologizing communication styles**: The tool must not penalize direct communication or culturally different communication norms. Direct but respectful communication should receive positive or neutral feedback.

---

## 3. MVP Definition

### Scope

The Minimum Viable Product validates the core hypothesis: **pre-send analysis changes communication behavior**.

> Note: The behavioral hypothesis is validated during MVP user testing with the actual model, not with a separate non-AI prototype. This accelerates the timeline but increases the risk of discovering the hypothesis is false after ML investment. If the hypothesis fails, the project pivots or stops.

### User Interaction

```
User writes a message (any text input)
        |
        v
User invokes Keep Calm (CLI command)
        |
        v
Keep Calm analyzes text locally via single-task models
        |
        v
Returns: risk score + risk level + tones + intent + explanation
        |
        v
User decides: send as-is, revise, or discard
```

### Input

- A single text message
- Language: English or Italian (separate language-specific models)
- Length: 1 to 2,000 characters
- Format: plain text

### Output

```json
{
  "communication_risk": 0.72,
  "risk_level": "medium",
  "tones": [
    {
      "label": "frustrated",
      "confidence": 0.84
    },
    {
      "label": "hostile",
      "confidence": 0.61
    }
  ],
  "intent": "critical",
  "needs_attention": true,
  "explanation": "The phrase 'you clearly' targets the person rather than the issue. This may be perceived as dismissive."
}
```

### Output Fields

| Field | Type | Description |
|---|---|---|
| `communication_risk` | float [0, 1] | Estimated likelihood of negative perception by a generalized observer |
| `risk_level` | enum | `none`, `low`, `medium`, `high`, `critical` |
| `tones` | list | Detected tones from 5 possible labels, with confidence scores |
| `intent` | string | Primary intent from 4 possible labels |
| `needs_attention` | boolean | Whether the message warrants review before sending |
| `explanation` | string | Specific, trigger-level explanation or positive confirmation |

### Tone Labels (MVP)

| Label | Description |
|---|---|
| `neutral` | No strong emotional signal |
| `frustrated` | Exasperation, impatience |
| `hostile` | Aggressive, adversarial stance |
| `sarcastic` | Ironic surface features signaling sarcastic intent |
| `positive` | Supportive, empathetic, or respectful tone |

### Intent Labels (MVP)

| Label | Description |
|---|---|
| `constructive` | Actionable, focused on improvement |
| `critical` | Negative evaluation, may or may not be constructive |
| `personal` | Targeting the person rather than the issue |
| `informational` | Neutral information sharing |

### Expected Behavior

- Analysis completes in under 200ms on reference hardware (Intel i5-8xxx / Apple M1, 16GB RAM)
- No network requests are made
- No data is stored or logged
- Output is deterministic for the same input (at a given model version)
- The system does not refuse to analyze any input
- Low-risk messages receive positive confirmation, not just silence

### Success Criteria

| Criterion | Target |
|---|---|
| Model agreement with human annotators on `risk_level` | >= 70% (reported alongside annotator IAA) |
| Pearson correlation (model risk vs. mean annotator risk) | >= 0.65 |
| Annotator inter-annotator agreement (Kappa) | >= 0.50 categorical, >= 0.60 continuous |
| False positive rate (flagging benign messages) | <= 20% |
| User behavioral change: revise or not send flagged messages | >= 20% of flagged messages |
| User willingness to use the tool | >= 7/10 users would use at least weekly |
| Psychological safety | <= 2/10 users report feeling judged or anxious |
| Inference latency (CPU, reference hardware) | <= 200ms |
| Model size | <= 500MB |

### MVP Delivery Format

- **CLI tool only**: `keep-calm "Your message here"`
- Python library built internally but not packaged for PyPI until v1.0
- REST API, ONNX export postponed to v1.0+

### Not Included in MVP

- Multi-message escalation detection (v2.0)
- Expanded label space (v1.0+: additional tones; v1.1+: additional intents)
- Python library packaging on PyPI (v1.0)
- REST API (v1.0)
- ONNX export (v1.0)
- WASM (indefinite)

---

## 4. Machine Learning Problem Definition

### ML Task Type

Keep Calm decomposes into three independent prediction tasks. Each is trained as a single-task model initially; multi-task consolidation is considered in v1.0 if evidence supports shared representation benefits and the latency budget requires it.

This is not a single classification task. It is a structured prediction problem with three concurrent outputs from separate models.

### ML Task Formulation

The model predicts **generalized observer perception**, not objective risk. The same message can be affectionate banter between friends or deeply hostile between strangers. The model has no access to the relationship, context, or recipient. The honest formulation is:

> "How is this message likely to be perceived by a generalized observer, absent specific relational or cultural context?"

The output communicates this limitation. Rather than stating "Communication risk: 86%" as if it were objective, the output frames the result as an estimate with inherent uncertainty.

### Classification Strategy

#### Task 1: Communication Risk Estimation

- **Type**: Regression (continuous score 0.0 to 1.0)
- **Target**: Estimated likelihood of negative perception by a generalized observer
- **Post-processing**: Discretized into risk levels (`none`, `low`, `medium`, `high`, `critical`) via calibrated thresholds

#### Task 2: Tone Detection

- **Type**: Multi-label classification (independent model)
- **Labels**: 5 tones for MVP

| Tone Label | Description |
|---|---|
| `neutral` | No strong emotional signal |
| `frustrated` | Exasperation, impatience |
| `hostile` | Aggressive, adversarial stance |
| `sarcastic` | Ironic surface features signaling sarcastic intent |
| `positive` | Supportive, empathetic, or respectful tone |

> Note: `sarcasm` was originally listed in both tone and intent. It is removed from tone — sarcasm is an intent, and ironic surface features detected by the tone model are signals for it. Positive tones and explicit validation of direct communication are included to avoid a purely negative framing.

#### Task 3: Intent Classification

- **Type**: Multi-class classification (primary intent)
- **Labels**: 4 intents for MVP

| Intent | Description |
|---|---|
| `constructive` | Actionable, specific, focused on improvement |
| `critical` | Negative evaluation, may or may not be constructive |
| `personal` | Targeting the person rather than the issue |
| `informational` | Neutral information sharing |

### Training Architecture

```
Input Text
    |
[Tokenizer + Encoder A] [Tokenizer + Encoder B] [Tokenizer + Encoder C]
    |                        |                        |
[Risk Head]            [Tone Head]             [Intent Head]
(regression)           (multi-label)           (multi-class)

Three independent models, trained separately.
Consolidated into multi-task in v1.0 if evidence supports benefit.
```

**Rationale for single-task first**:

- Independent training allows debugging each task in isolation.
- If one task fails, the others are not affected.
- Multi-task may cause interference between tasks (e.g., sarcasm detection requires different features than risk regression).
- Multi-task adds complexity (loss weighting, gradient balancing) that is premature.
- Phase 2 measures task correlations to determine if shared representations would help.
- If the latency budget forces consolidation, multi-task is adopted in Phase 3.

### Post-Processing: Severity

Severity is not a learned task. It is a deterministic post-processing function:

```
severity = aggregate(risk_score, max_tone_confidence, intent_severity_weight)
```

This reduces model complexity from 4 heads to 3 and eliminates the risk of severity outputs contradicting risk and tone predictions.

### Confidence Model

All predictions include calibrated confidence scores:

- **Risk score**: Raw regression output passed through sigmoid, calibrated via Platt scaling on validation set
- **Tone confidence**: Sigmoid output per label, threshold-tuned per label to account for class imbalance
- **Intent confidence**: Softmax probability of predicted class

Confidence reflects annotator spread: high disagreement in training data produces lower confidence at inference. Low-confidence predictions are surfaced to the user.

### Explainability Requirements

The system provides:

1. **Human-readable explanation**: A specific sentence identifying the primary concern, using template-based generation from structured outputs (MVP approach). Example: "The phrase 'you clearly' targets the person rather than the issue" — not generic statements like "hostile tone detected."
2. **Positive feedback**: Low-risk messages receive explicit confirmation: "This message reads as clear and constructive."
3. **Token-level attribution** (post-MVP): Integrated Gradients / SHAP to highlight which parts of the text triggered the assessment.

---

## 5. Dataset Strategy

> The dataset is the most important asset of this project. Model architecture is secondary.

### Design Principles

1. **Quality over quantity**: 8,000 carefully annotated examples per language are more valuable than 500,000 noisy ones.
2. **Subjectivity as a feature**: Annotators report their own perception, not a hypothetical "reasonable recipient." Disagreement is signal, not noise.
3. **Cultural diversity in annotation**: Annotators from varied cultural and linguistic backgrounds label the same data.
4. **Spectrum coverage**: The dataset must include the full spectrum from clearly benign to clearly harmful, with significant representation of ambiguous cases.
5. **Ethical sourcing**: No private messages without consent. Real human communication prioritized. Synthetic data capped at 40%.

### Annotation Schema

Each example in the dataset:

```json
{
  "id": "kc-00001",
  "text": "You clearly have no idea what you're doing.",
  "language": "en",
  "domain": "workplace_chat",
  "annotations": {
    "communication_risk": 0.88,
    "tones": [
      {"label": "hostile", "present": true, "confidence": 0.92},
      {"label": "frustrated", "present": true, "confidence": 0.78},
      {"label": "sarcastic", "present": false, "confidence": 0.15},
      {"label": "neutral", "present": false, "confidence": 0.02},
      {"label": "positive", "present": false, "confidence": 0.05}
    ],
    "intent": "personal",
    "explanation": "Targets the person's competence rather than addressing a specific issue.",
    "needs_attention": true
  },
  "metadata": {
    "annotator_count": 3,
    "annotator_agreement": 0.62,
    "annotator_demographics": [
      {
        "annotator_id": "a01",
        "native_language": "en",
        "cultural_background": "british",
        "gender": "f",
        "age_range": "30-40"
      }
    ],
    "source": "natural",
    "created_at": "2026-06-15"
  }
}
```

### Data Format

- **Storage**: JSONL (one example per line) for efficient streaming and processing
- **Versioning**: Dataset versions tracked with DVC or similar
- **Splits**: Train / Validation / Test with stratification by risk level and domain

### Domain (MVP)

| Domain | Description | Priority |
|---|---|---|
| `workplace_chat` | Slack/Teams-style professional messaging | MVP (only domain) |

Single domain for the MVP ensures a focused, high-quality dataset. Additional domains (`code_review`, `email`, `forum`) are added in v1.1+.

### Labeling Guidelines

#### Communication Risk

| Score | Level | Description |
|---|---|---|
| 0.0 - 0.2 | None | Clearly benign, positive, or constructive communication |
| 0.2 - 0.4 | Low | Direct but respectful, unlikely to cause offense |
| 0.4 - 0.6 | Medium | Could be perceived negatively depending on context or recipient |
| 0.6 - 0.8 | High | Likely to cause discomfort, tension, or negative reaction |
| 0.8 - 1.0 | Critical | Almost certainly harmful, aggressive, or damaging |

#### Key Labeling Rules

1. **Annotate your own perception**: "How would YOU perceive this message if you received it in a professional context?" Not "how would a reasonable recipient perceive it?"
2. **Record your demographics**: Native language, cultural background, gender, age range. This information contextualizes the annotation.
3. **Disagreement is signal**: If annotators disagree, the message is ambiguous. Preserve it, don't discard it.
4. **Context is specified**: All examples assume a workplace chat context.
5. **Direct is not hostile**: "This approach has significant problems" is direct but respectful. Distinguish bluntness from aggression. The dataset must include explicit "direct but respectful" examples labeled as low risk.
6. **Positive tone is not always low risk**: Sarcasm can use positive words with negative intent.
7. **Cultural awareness**: Annotators note when cultural context affects their interpretation.

### Data Sources

#### Public Datasets (Pre-training / Transfer Learning)

| Dataset | Use | Size | License |
|---|---|---|---|
| Jigsaw Toxic Comment Classification | Toxicity signals | ~2M | CC0 |
| GoEmotions | Emotion detection | ~58K | Apache 2.0 |
| ISEAR | Emotion patterns | ~7.5K | Research |
| EmpatheticDialogues | Empathy detection | ~25K | CC-BY |
| SARC | Sarcasm detection | ~1M | Various |

Public datasets are re-labeled with Keep Calm's schema to augment the custom dataset.

#### Custom Annotated Data (Core Dataset)

| Source | Method | Target Size (per language) |
|---|---|---|
| Natural workplace chats | Public Slack/Teams archives, manually annotated | 3,000 - 4,000 |
| Synthetic generation | LLM-assisted + human rewriting + validation | 2,500 - 3,500 |
| Public code reviews | GitHub PR comments, manually annotated | 1,000 - 1,500 |
| Crowdsourced scenarios | Annotators write + annotate realistic scenarios | 1,500 - 1,500 |

**Total per language**: 8,000 - 10,000 examples

#### Synthetic Data Generation Strategy

- Synthetic data capped at **40%** of the training set.
- All synthetic data is **rewritten by humans** to sound like real communication before annotation.
- Must pass a naturalness check: "Does this look like something a real person would type in a real chat window?"
- Synthetic data is flagged in metadata and tracked per category.

### Dataset Size Estimation

| Phase | Custom Annotated (EN) | Custom Annotated (IT) | Public Re-labeled | Total |
|---|---|---|---|---|
| MVP | 8,000 - 10,000 | 8,000 - 10,000 | 100K (subsampled) | ~120K |
| v1.0 | 12,000 - 16,000 | 12,000 - 16,000 | 500K | ~544K |
| v2.0 | 20,000 - 25,000 | 20,000 - 25,000 | 1M+ | ~1M+ |

### Evaluation Split Strategy

- **Random split**: 70% train / 15% validation / 15% test
- **Stratification**: By risk level, tone presence, and language to ensure balanced representation
- **Adversarial test set**: 500+ examples specifically designed to be difficult (ambiguous, sarcastic, culturally nuanced)
- **Out-of-distribution test**: Communication from non-native speakers, neurodivergent writers, and varied cultural contexts — tested before MVP release

### Inter-Annotator Agreement

- Minimum 3 annotators per example in the core dataset, with cultural diversity as a requirement
- Target Cohen's Kappa >= 0.50 for categorical labels (realistic for subjective perception tasks)
- Target Pearson's r >= 0.60 for risk scores
- Soft labels used for training: model trained on the distribution of annotator ratings, not the majority vote
- Examples with very low agreement (Kappa < 0.30) preserved as ambiguity cases, not discarded
- Actual IAA reported honestly in the dataset card

### Bias Audit (Phase 1)

Bias auditing is a dataset creation task, not a post-hoc model evaluation:

- Dataset audited for demographic and cultural representation before training begins
- Annotators recruited with cultural diversity: direct-communication cultures (German, Dutch, Israeli) and indirect-communication cultures (Japanese, British, Korean) label the same data
- Dataset includes explicit examples of communication styles that differ from Anglo-American norms, labeled by annotators from those cultures
- Bias audit is a Phase 1 deliverable, not Phase 3

### Italian Dataset

Italian is developed in parallel with English:

- Same size and quality targets: 8,000-10,000 examples
- Italian annotators recruited with regional diversity (North, Center, South)
- Italian-specific annotation guidelines accounting for cultural norms (tu/Lei formality, emotional expressiveness, regional variation)
- Separate language-specific model trained from the Italian dataset

---

## 6. Model Strategy

### Constraints

| Constraint | Target | Rationale |
|---|---|---|
| Inference hardware | CPU only | Privacy-first, no GPU dependency |
| Reference hardware | Intel i5-8xxx / Apple M1, 16GB RAM | Explicit benchmark target |
| Latency | < 200ms per message on reference hardware | Real-time pre-send analysis |
| Model size | < 500MB on disk | Downloadable, embeddable |
| Memory | < 1GB RAM at inference | Runs alongside other applications |
| Offline | No network required | Privacy guarantee |
| Dependencies | Minimal system dependencies | Easy installation |

### Approach Evaluation

#### Option A: Classical NLP Baseline

- **Method**: TF-IDF / n-gram features + logistic regression / gradient boosting
- **Pros**: Extremely fast, tiny model, interpretable, strong baseline
- **Cons**: Cannot capture context, sarcasm, or nuance. Limited ceiling performance.
- **Verdict**: Implement as Phase 2 baseline. Not the final model.

#### Option B: Fine-tuned Transformer Encoder

- **Method**: Fine-tune a pre-trained small transformer with multi-task heads
- **Candidates**:

| Model | Parameters | Size | Speed | Quality |
|---|---|---|---|---|
| DistilBERT-base | 66M | ~250MB | Fast | Good |
| MiniLM-L12 | 33M | ~130MB | Very fast | Good |
| ALBERT-base | 12M | ~45MB | Fast | Moderate |
| ModernBERT-base | 149M | ~570MB | Moderate | Very good |
| DeBERTa-v3-small | 44M | ~170MB | Fast | Good |

- **Pros**: Strong contextual understanding, proven fine-tuning approach, rich ecosystem
- **Cons**: Larger than classical models, requires careful optimization for CPU
- **Verdict**: Primary approach for MVP. MiniLM or DistilBERT are the leading candidates.

#### Option C: Knowledge Distillation

- **Method**: Train a large teacher model (e.g., DeBERTa-v3-large), distill into a small student model
- **Pros**: Better performance-size tradeoff than direct training
- **Cons**: More complex training pipeline, requires Phase 2 dataset maturity
- **Verdict**: Phase 3-4 enhancement, not MVP.

#### Option D: Small Language Model (Decoder-based)

- **Method**: Fine-tune a small generative model (e.g., Phi-3-mini, Qwen2-0.5B) for structured output
- **Pros**: Richer explanations, more flexible output
- **Cons**: Much larger (1-4GB), slower inference, overkill for classification
- **Verdict**: Evaluate for post-MVP explainability features. Not suitable for core MVP.

### Recommended Strategy

```
Phase 2: Classical NLP baseline (TF-IDF + XGBoost per task)
         Establishes performance floor for each task independently
         
Phase 2: Fine-tuned MiniLM or DistilBERT per task
         Three independent single-task models (risk, tone, intent)
         Each trained, evaluated, and benchmarked separately
         
Phase 2: Task correlation analysis
         Measure whether hard examples for one task are also hard for others
         
Phase 3: Multi-task consolidation (conditional)
         Only if independent models are strong and latency budget requires it.
         Only if task correlation analysis shows shared representation benefit.
         
Phase 4: Optional SLM for explanation generation
         Separate, optional component
```

### Single-Task vs Multi-Task Decision

The initial architecture proposed multi-task from Phase 2. This was revised after critical review:

1. **Phase 2 starts with single-task models**: One for risk regression, one for tone classification, one for intent classification.
2. **Task correlation is measured**: Do the same examples that are hard for tone also hard for intent?
3. **Multi-task only if evidence supports it**: If independent models are strong and task correlations confirm shared representation benefit, multi-task is adopted in Phase 3.
4. **Fallback guaranteed**: If multi-task fails, working single-task models already exist.

### Inference Optimization

- **ONNX Runtime**: Export trained model to ONNX for optimized CPU inference
- **Quantization**: INT8 quantization to reduce model size by ~4x with minimal accuracy loss
- **Tokenizer caching**: Cache tokenizer for repeated use
- **Batching**: Support single-message and batch analysis modes

### Model Selection Decision Criteria

The final model choice will be based on:

1. **Test set performance** across all tasks (weighted: risk > tone > intent)
2. **Inference latency** on reference hardware (Intel i5-8xxx / Apple M1)
3. **Model size** after export and quantization
4. **Calibration quality**: Are confidence scores well-calibrated? Does confidence reflect annotator spread?
5. **Robustness**: Performance on adversarial test set and out-of-distribution communication

---

## 7. Technical Architecture

### Repository Structure

The repository uses a flat structure in Phase 0-2, evolving into the full production structure in Phase 3.

**Phase 0-2 (Flat)**:

```
keep-calm/
├── README.md
├── LICENSE
├── ARCHITECTURE.md
├── IDEA.md
├── REVIEW.md
├── DECISIONS.md
├── pyproject.toml
│
├── data/
│   ├── en/
│   │   ├── raw/
│   │   ├── annotated/
│   │   └── splits/
│   └── it/
│       ├── raw/
│       ├── annotated/
│       └── splits/
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   └── 02_baseline_evaluation.ipynb
│
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   └── bias_audit.py
│
├── src/
│   └── keep_calm/
│       ├── __init__.py
│       ├── analyzer.py
│       ├── models/
│       │   ├── __init__.py
│       │   ├── tokenizer.py
│       │   └── transformer.py
│       ├── tasks/
│       │   ├── __init__.py
│       │   ├── risk.py
│       │   ├── tone.py
│       │   └── intent.py
│       └── schemas/
│           └── output.py
│
└── tests/
    ├── test_analyzer.py
    └── test_models.py
```

**Phase 3+ (Production)**:

```
keep-calm/
├── ... (flat structure +)
├── training/
│   ├── configs/
│   ├── data_module.py
│   ├── model_module.py
│   └── losses.py
├── evaluation/
│   ├── metrics.py
│   └── analysis.py
├── cli/
│   └── main.py
├── api/
│   └── server.py
└── docs/
    └── api_reference.md
```

### ML Pipeline

```
┌──────────────────────────────────────────────────────────┐
│                    DATA PIPELINE                          │
│                                                          │
│  Raw Sources ──► Formatting ──► Cleaning ──► Annotation  │
│       │                                    │             │
│       ▼                                    ▼             │
│  Public Datasets                    Custom Dataset       │
│       │                                    │             │
│       └──────────────┬─────────────────────┘             │
│                      ▼                                   │
│              Train/Val/Test Split                        │
│              (stratified, versioned)                     │
└──────────────────────────────────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────────────┐
│                  TRAINING PIPELINE                        │
│                                                          │
│  Config ──► DataModule ──► ModelModule ──► Trainer       │
│                │               │              │          │
│                ▼               ▼              ▼          │
│           Tokenization   Multi-task     PyTorch          │
│           + Encoding     Loss           Lightning        │
│                              │              │            │
│                              ▼              ▼            │
│                        Checkpointing  Early Stopping     │
│                        + Logging      + LR Scheduling    │
└──────────────────────────────────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────────────┐
│                 EVALUATION PIPELINE                       │
│                                                          │
│  Best Checkpoint ──► Test Set Eval ──► Metrics Report    │
│         │                                    │           │
│         ▼                                    ▼           │
│  Calibration ──► Adversarial Test ──► Bias Audit         │
│         │                                    │           │
│         ▼                                    ▼           │
│  ONNX Export ──► Quantization ──► Benchmarking           │
└──────────────────────────────────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────────────┐
│                 INFERENCE PIPELINE                        │
│                                                          │
│  Text Input ──► Tokenizer ──► Model ──► Post-processing  │
│                                              │           │
│                                              ▼           │
│                                    Calibration +          │
│                                    Thresholding           │
│                                              │           │
│                                              ▼           │
│                                    Structured Output      │
│                                    (JSON / Pydantic)      │
└──────────────────────────────────────────────────────────┘
```

### Training Workflow (Phase 2)

1. **Configuration**: YAML-based config specifying model, dataset paths, hyperparameters, task type
2. **Data loading**: PyTorch Lightning DataModule with stratified splits
3. **Single-task training**: Three independent training runs — one per task:
   - Risk regression (MSE loss)
   - Tone classification (BCE multi-label loss)
   - Intent classification (Cross-entropy loss)
4. **Training process**:
   - Mixed precision (where supported on CPU)
   - Gradient accumulation for effective batch sizing
   - Early stopping on validation loss
   - Learning rate scheduling (cosine with warmup)
5. **Checkpointing**: Save best model per task
6. **Logging**: Weights & Biases or TensorBoard for experiment tracking
7. **Task correlation analysis**: After training, measure whether hard examples for one task are also hard for others

### Training Workflow (Phase 3+)

If task correlation analysis supports it and the latency budget requires consolidation, train a multi-task model with weighted losses. Single-task models remain available as fallback.

### Evaluation Workflow

1. **Standard metrics**: Per-task metrics on test set
   - Risk: MAE, RMSE, Pearson correlation with mean annotator score
   - Tone: Per-label F1, macro F1, AUC-ROC
   - Intent: Accuracy, macro F1
2. **Calibration**: ECE (Expected Calibration Error), reliability diagrams
3. **Adversarial evaluation**: Performance on difficult/ambiguous examples
4. **Bias audit**: Performance disaggregated by domain, message length, linguistic features, annotator demographics
5. **Out-of-distribution testing**: Communication by non-native speakers, neurodivergent writers, varied cultural contexts
6. **Latency benchmark**: Inference time distribution on reference hardware (Intel i5-8xxx / Apple M1, 16GB RAM)

### Inference Architecture

```python
# User-facing API
from keep_calm import KeepCalmAnalyzer

analyzer = KeepCalmAnalyzer()  # Loads model into memory
result = analyzer.analyze("Your message text here")

# result is a structured Pydantic model
print(result.communication_risk)  # 0.72
print(result.risk_level)          # "medium"
print(result.tones)               # [Tone(label="frustrated", confidence=0.84), ...]
print(result.explanation)         # "The message uses language that..."
```

### Deployment Options

| Option | Description | Use Case | Timeline |
|---|---|---|---|
| CLI tool (MVP) | `keep-calm "message"` | Quick analysis, scripting | MVP |
| Python library | `pip install keep-calm` | Developers, integrations | v1.0 |
| Local REST API | `keep-calm serve` | IDE plugins, browser extensions | v1.0+ |
| ONNX model | Raw model file | Non-Python integrations (Rust, Go, JS) | v1.0+ |
| WASM | WebAssembly export | Browser-based analysis | Indefinite |

**Note**: The MVP ships as a CLI tool only. The Python library is built internally but not packaged or documented as a public API until v1.0. The model processes one message at a time — no batch analysis API is exposed.

---

## 8. Risks and Challenges

### Technical Risks

| Risk | Severity | Mitigation |
|---|---|---|
| **Sarcasm detection is extremely hard** | High | Accept lower performance on sarcasm. Flag low-confidence sarcasm predictions. Invest in sarcasm-specific training data. |
| **Context window limitations** | Medium | Transformer models have token limits (~512). Long messages may lose context. Implement sliding window or summarization for long texts. |
| **CPU inference too slow** | Medium | ONNX optimization, quantization, model distillation. Fallback to classical model if latency exceeds threshold. |
| **Multi-task interference** | Medium | Tasks may conflict during training. Monitor per-task performance. Adjust loss weights. Consider task-specific fine-tuning if interference is severe. |
| **Explanation quality** | Medium | Template-based explanations may feel generic. Invest in explanation templates. Evaluate SLM-based explanations post-MVP. |

### Ethical Risks

| Risk | Severity | Mitigation |
|---|---|---|
| **Weaponization as surveillance** | Critical | Any tool that scores communication quality will be used to evaluate people. License clauses alone cannot prevent misuse. Mitigations: no batch analysis API, no reporting or aggregation features, no logging by default, single-message processing only. Honest documentation that the project cannot technically prevent surveillance use. |
| **Erosion of authentic communication** | High | Emphasize that Keep Calm is awareness, not optimization. Do not suggest "better" phrasings. Validate positive tones to avoid purely negative framing. |
| **Over-reliance and deskilling** | Medium | Design for learning: explanations help users internalize patterns. Consider a "training mode" that gradually reduces assistance. |
| **Moral licensing** | Medium | Users may feel that a "low risk" score means their message is fine. Communicate that the model is a signal, not a guarantee. |
| **Communication anxiety** | High | Users whose communication style is frequently flagged may develop self-censorship, anxiety, or identity threat. Mitigations: validate direct but respectful communication, provide positive feedback for healthy messages, frame the tool as occasional rather than constant, never suggest rewrites. |

### Bias Risks

| Risk | Severity | Mitigation |
|---|---|---|
| **Cultural communication bias** | High | Communication norms vary dramatically across cultures. Direct communication (common in German, Dutch, Israeli cultures) may be falsely flagged. Mitigations: culturally diverse annotators labeling the same data, bias audit in Phase 1 (dataset creation), out-of-distribution testing before MVP release. |
| **AAE / dialect bias** | High | African American English and other dialects are systematically penalized by NLP systems. Mitigations: audit performance across dialects during dataset creation, include diverse linguistic patterns, test on out-of-distribution data. |
| **Gender bias** | High | Women's communication is often perceived differently than men's for the same wording. Mitigations: balanced annotator demographics, audit for gendered patterns in the dataset, not just in model output. |
| **Domain bias** | Medium | Model trained on workplace chat may misclassify communication in other domains. Acknowledged: single-domain MVP. Cross-domain evaluation in v1.1+. |
| **Length bias** | Low | Longer messages may receive higher risk scores. Mitigations: normalize or account for message length. |

### False Positive Problems

False positives (flagging benign messages as risky) are the primary adoption risk.

**Consequences**:

- Users lose trust and stop using the tool
- Users feel judged or surveilled
- Legitimate direct communication is discouraged
- Marginalized voices may be disproportionately silenced

**Mitigation**:

- Conservative thresholds: require high confidence before flagging
- Always provide explanation: a false positive with an explanation is less damaging than an unexplained flag
- Positive validation: explicitly confirm when communication is healthy
- Separate "direct" from "hostile": train the model with explicit "direct but respectful" examples at low risk
- User feedback loop: allow users to mark false positives, use for continuous improvement
- Annotator diversity: culturally diverse annotators reduce systemic bias in training labels
- Out-of-distribution testing: test on non-native speakers and varied cultural contexts before release

### Cultural and Language Challenges

- **Italian vs. English communication norms**: Italian communication tends to be more expressive and emotionally direct. Separate language-specific models are trained, not a single multilingual model. Italian dataset uses regionally diverse annotators (North, Center, South) to capture variation.
- **Formality levels**: Different languages and cultures have different formality expectations (e.g., Italian tu/Lei). The annotation guidelines account for this.
- **Idioms and colloquialisms**: "That's sick" can be positive or negative. Context-dependent meaning is a core NLP challenge.
- **Code-switching**: Multilingual users may mix languages. The model should handle this gracefully.
- **Cultural specificity**: The Keep Calm concept itself may be culturally specific to Anglo-American professional norms. This is acknowledged as a risk; the Italian dataset and testing will provide evidence for or against cultural generalizability.

---

## 9. Roadmap

Each version has explicit gates. If a gate fails, the project pauses to understand why before proceeding.

### Version Staging Summary

| Version | Scope | Gate to Next |
|---|---|---|
| **MVP** | Risk score + 5 tones + 4 intents + explanation. CLI only. Workplace chat domain. English + Italian. | Behavioral hypothesis validated. Users find it useful. |
| **v1.0** | Python library. PyPI package. Expanded tones. | Tone model performance meets targets. |
| **v1.1** | Expanded intents. REST API. Second domain. | Cross-domain performance acceptable. |
| **v2.0** | Multi-message escalation detection. ONNX export. Expanded label space. | Escalation detection validated. |

---

### Phase 0: Research and Specification

**Duration**: 3-4 weeks

**Objectives**:
- Finalize product and technical specifications
- Survey existing work and available datasets
- Define annotation guidelines
- Set up development environment

**Tasks**:
- [ ] Survey existing toxicity/emotion/communication quality models and datasets
- [ ] Benchmark existing models (Perspective API, Jigsaw) on Keep Calm-style examples
- [ ] Finalize annotation schema for both English and Italian
- [ ] Define labeling guidelines v1.0 (English and Italian)
- [ ] Set up repository, tooling, CI/CD (flat structure)
- [ ] Specify annotation tooling workflow

**Deliverables**:
- Updated product specification (this document)
- Existing work survey report
- Annotation guidelines v1.0 (EN + IT)
- Repository initialized

---

### Phase 1: Dataset Creation

**Duration**: 10-14 weeks (both languages in parallel)

**Objectives**:
- Create foundational annotated datasets for English and Italian
- Establish annotation quality and consistency
- Perform bias audit
- Prepare evaluation splits

**Tasks**:
- [ ] Collect and format public datasets (Jigsaw, GoEmotions, SARC, etc.)
- [ ] Build or configure annotation tool
- [ ] Generate synthetic data using LLMs (max 40% of training set)
- [ ] Human-rewrite synthetic data for naturalness
- [ ] Recruit culturally diverse annotators (direct and indirect communication cultures)
- [ ] Create Italian-specific annotation guidelines (formality, regional variation)
- [ ] Annotate English dataset: 8,000-10,000 examples (workplace_chat domain)
- [ ] Annotate Italian dataset: 8,000-10,000 examples (workplace_chat domain)
- [ ] Re-label subsets of public datasets with Keep Calm schema
- [ ] Measure inter-annotator agreement (target: Kappa >= 0.50, Pearson r >= 0.60)
- [ ] **Perform bias audit**: demographic and cultural representation, diverse annotator coverage
- [ ] Create stratified train/val/test splits
- [ ] Build adversarial test set (500+ difficult examples)
- [ ] Build out-of-distribution test set (non-native speakers, neurodivergent, varied cultures)
- [ ] Document dataset provenance and known limitations

**Deliverables**:
- Annotated dataset v1.0: 8,000-10,000 English + 8,000-10,000 Italian examples
- Public dataset pipeline (formatted, cleaned, re-labeled)
- Annotation guidelines v2.0 (refined based on IAA results)
- Bias audit report
- Dataset card and documentation
- Adversarial and out-of-distribution test sets

---

### Phase 2: Baseline Model

**Duration**: 6-8 weeks

**Objectives**:
- Establish performance baselines per task
- Train single-task models
- Measure task correlations
- Benchmark latency

**Tasks**:
- [ ] Implement classical NLP baseline (TF-IDF + XGBoost per task)
- [ ] Train three independent transformer models (MiniLM or DistilBERT):
  - Risk regression model
  - Tone multi-label classification model
  - Intent multi-class classification model
- [ ] Evaluate on test set and adversarial set
- [ ] Perform error analysis: where does each model fail?
- [ ] Measure task correlations: do hard examples overlap across tasks?
- [ ] Implement confidence calibration (Platt scaling / temperature scaling)
- [ ] **Benchmark inference latency** on reference hardware (Intel i5-8xxx / Apple M1)
- [ ] Export models to ONNX; benchmark optimized inference
- [ ] Apply INT8 quantization; measure accuracy tradeoff
- [ ] Test on out-of-distribution communication
- [ ] Document results

**Deliverables**:
- Classical baseline models + results
- Three single-task transformer models (EN + IT = 6 models total)
- Evaluation report with per-task metrics
- Task correlation analysis
- Error analysis report
- Latency benchmarks
- Out-of-distribution test results

---

### Phase 3: MVP Release

**Duration**: 4-6 weeks

**Objectives**:
- Package the models into a usable CLI tool
- Validate the behavioral hypothesis
- Iterate based on user feedback

**Tasks**:
- [ ] Implement CLI tool (`keep-calm "message"` with language auto-detection)
- [ ] Implement output schema with Pydantic models
- [ ] Implement template-based explanations
- [ ] Implement severity post-processing
- [ ] Write unit tests and integration tests
- [ ] Write documentation: README, CLI reference, quickstart
- [ ] Conduct user testing with 10-20 target users (2-week trial)
- [ ] **Validate behavioral hypothesis**: measure voluntary usage and revision rates
- [ ] Collect feedback; iterate on thresholds, explanations, and UX
- [ ] Expand annotated dataset based on error analysis and user feedback
- [ ] Retrain models with expanded dataset if needed
- [ ] Conditional: if task correlations and latency budget support it, explore multi-task consolidation

**Deliverables**:
- `keep-calm` CLI tool (English + Italian)
- Behavioral validation report
- User testing report (including psychological safety metrics)
- Updated models (if retrained)
- Documentation

**Gate**: Behavioral hypothesis validated (>= 7/10 users would use weekly, >= 20% revision rate on flagged messages, <= 2/10 users report feeling judged).

---

### Phase 4: Integrations and Growth

**Duration**: Ongoing

**Objectives**:
- Expand ecosystem and reach
- Develop integrations
- Explore advanced features

**Tasks (v1.0)**:
- [ ] Expand tone labels based on error analysis
- [ ] Package Python library for PyPI
- [ ] ONNX export for non-Python integrations
- [ ] Build VS Code extension
- [ ] Multi-task consolidation (if supported by Phase 2 analysis)

**Tasks (v1.1)**:
- [ ] Expand intent labels based on error analysis
- [ ] Add second domain
- [ ] Local REST API
- [ ] Build Slack/Discord integration

**Tasks (v2.0)**:
- [ ] Conversation escalation detection (multi-message analysis)
- [ ] Build GitHub Action for PR comment analysis
- [ ] Build browser extension
- [ ] Knowledge distillation for performance improvements
- [ ] User feedback pipeline for continuous dataset improvement
- [ ] Publish dataset and model benchmarks
- [ ] Explore personalization ("calibrate to my style")

**Deliverables**:
- VS Code extension
- Slack integration
- Browser extension
- GitHub Action
- Conversation escalation feature
- Published benchmarks and research

---

## Appendix: Open Questions

These questions should be resolved during Phase 0:

1. **Annotation tooling**: Label Studio vs. Prodigy vs. custom tool? Prodigy is optimized for NLP annotation but is commercial. Label Studio is open source.
2. **Annotator recruitment**: Paid annotators (Prolific, MTurk) vs. volunteer community annotators? Quality vs. scale tradeoff. Cultural diversity is a recruitment requirement.
3. **Model framework**: PyTorch + PyTorch Lightning vs. Hugging Face Transformers Trainer? Lightning offers more control; HF Trainer is more opinionated but faster to set up.
4. **License**: Apache 2.0 vs. MIT vs. a custom license with anti-surveillance clauses? Note: license clauses alone cannot prevent surveillance misuse (see DEC-18).
5. **Monetization**: Is this purely open source, or is there a path to a sustainable business model (enterprise support, hosted API, premium integrations)?
6. **Multilingual architecture**: Separate per-language models confirmed (DEC-13). Shared representations vs. language-specific calibration is a research question for Phase 2.
7. **Italian-specific challenges**: Tu/Lei formality, regional variation (North/Center/South), emotional expressiveness norms, professional vs. casual register boundaries.
8. **Multi-task consolidation criteria**: Specific thresholds for task correlation, independent model performance, and latency budget that would trigger multi-task adoption in Phase 3.

---

*This document is the single source of truth for the Keep Calm project. It should be updated as the project evolves, assumptions are validated or invalidated, and new information becomes available.*
