# Keep Calm — Annotation Guidelines (v1.0)

> Guidelines for annotating communication risk, tone, and intent in workplace chat messages.

---

## 1. Overview

These guidelines define how to label messages for the Keep Calm dataset. Every example is labeled by at least 3 annotators. Disagreement is expected and valuable — it signals ambiguity.

**Core principle**: You are labeling YOUR perception, not an objective truth. There is no "correct" label. There is only your honest assessment.

---

## 2. The Annotation Task

For each message, provide:

1. **Communication risk** — a continuous score (0.0 to 1.0)
2. **Risk level** — discretized (none / low / medium / high / critical)
3. **Tones** — which of 5 tones are present (multi-label, with confidence)
4. **Intent** — the primary communication intent (single choice)
5. **Explanation** — a short sentence explaining your assessment
6. **Needs attention** — would you pause before sending this?

---

## 3. Context Assumption

All messages assume a **professional workplace chat context** (Slack/Teams/Mattermost).

The message could be:
- A reply to a colleague's message in a channel
- A direct message about work
- A comment in a shared thread

Assume the message is visible to peers and possibly managers. Do not assume a specific relationship (friend, rival, manager) unless the text strongly implies it.

---

## 4. Communication Risk

### 4.1 Risk Score

The communication risk score estimates how likely this message is to create tension, discomfort, or negative perception for a generalized reader in a professional context.

| Score | Level | Description | Examples |
|---|---|---|---|
| 0.0 - 0.2 | None | Clearly benign, positive, or constructive | "Thanks for the update!", "Good point, let me check" |
| 0.2 - 0.4 | Low | Slightly terse or direct but respectful | "Not sure this works. Thoughts?" |
| 0.4 - 0.6 | Medium | Could be perceived negatively; ambiguous tone | "This is the third time we're discussing this." |
| 0.6 - 0.8 | High | Likely to cause discomfort or tension | "You clearly don't understand this." |
| 0.8 - 1.0 | Critical | Almost certainly harmful, aggressive, or damaging | "You are incompetent and should not be on this project." |

### 4.2 Risk Level

Derived from the score. When in doubt between two levels, choose the higher one.

### 4.3 Key Distinctions

**Direct ≠ Hostile**

Direct messages are concise and clear. They can be respectful.

| Direct (low risk) | Hostile (high risk) |
|---|---|
| "This approach has problems." | "This approach is stupid." |
| "I need this by Friday." | "If you can't do this by Friday, what are you even doing here?" |
| "Let's discuss this offline." | "We shouldn't be wasting the team's time with this." |

**Frustrated ≠ Aggressive**

Frustration expresses emotion. Aggression targets a person.

| Frustrated (medium risk) | Aggressive (high risk) |
|---|---|
| "I'm really frustrated with this bug." | "Whoever wrote this code should be fired." |
| "This keeps happening, it's annoying." | "You keep messing this up." |

---

## 5. Tone Labels

Select all that apply. Rate confidence for each on a 0.0-1.0 scale.

### 5.1 neutral
No strong emotional signal. Factual, informational, professional.

**Examples**:
- "The meeting is at 3pm."
- "Here are the updated numbers for Q3."
- "Can you send me the latest version?"

### 5.2 frustrated
Exasperation, impatience, dissatisfaction. Emotion is present but not necessarily directed at the recipient.

**Examples**:
- "I'm so tired of this bug."
- "Why does this keep happening?"
- "fine. whatever."
- "This deadline is impossible."

**Edge cases**: Frustration about the situation vs. frustration directed at the recipient. Both are `frustrated`. If the frustration explicitly targets the recipient's competence or character, add `hostile`.

### 5.3 hostile
Aggressive, adversarial, confrontational. Targets the recipient or a group. Implies fault, blame, or inadequacy.

**Examples**:
- "You have no idea what you're doing."
- "This is your fault."
- "Stop wasting everyone's time."
- "Are you even qualified to comment on this?"

**Note**: `hostile` and `frustrated` can co-occur. A message can be both frustrated AND hostile.

### 5.4 sarcastic
Ironic, mocking, saying the opposite of what is meant. Often uses exaggerated praise or politeness.

**Examples**:
- "Great job as always..." (after something went wrong)
- "Oh, wonderful. Another meeting."
- "Sure, let's do it your way. What could possibly go wrong."
- "Per my last email, as previously stated multiple times..."

**Important**: Sarcasm is subtle and culturally specific. What reads as sarcastic to one annotator may read as sincere to another. This is expected. Mark it and note your confidence.

### 5.5 positive
Supportive, empathetic, encouraging, respectful, appreciative.

**Examples**:
- "Great work on this!"
- "I really appreciate the effort."
- "That's a good question, let me think about it."
- "Take your time, no rush."
- "I understand your point, and I think we can find a middle ground."

**Note**: A message can be both critical and positive — e.g., constructive feedback that acknowledges effort while suggesting improvement. Both `positive` and the relevant negative tone can be marked.

### 5.6 Multi-label Examples

| Message | Tones |
|---|---|
| "Thanks! This looks great." | positive |
| "I'm so done with this." (about a project) | frustrated |
| "Maybe if you'd read the docs first..." | sarcastic, hostile |
| "You're so talented. /s" | sarcastic, hostile |
| "I'm frustrated but I appreciate your effort." | frustrated, positive |
| "The numbers are in the attached spreadsheet." | neutral |

---

## 6. Intent Labels

Choose the single primary intent. If multiple intents exist, choose the dominant one.

### 6.1 constructive
Actionable, helpful, focused on improvement. Aims to make something better.

**Examples**:
- "Here's how we can fix this..."
- "I suggest we try approach B instead."
- "Let me help you debug this."
- "Good start! A few suggestions: ..."

### 6.2 critical
Negative evaluation or judgment. May or may not be constructive. Expresses dissatisfaction with an approach, outcome, or situation.

**Examples**:
- "This approach won't work."
- "I don't think this is ready."
- "The design has several flaws."
- "This keeps breaking." (without offering a fix)

**Edge**: Critical feedback paired with actionable suggestions is still `constructive`, not `critical`. Criticism without a path forward is `critical`.

### 6.3 personal
Targeting a person rather than the issue. Ad hominem, character judgment, personal dismissal.

**Examples**:
- "You're incompetent."
- "Typical. You always do this."
- "Maybe someone more experienced should handle this."
- "You're being lazy."

**Note**: "You wrote buggy code" targets the work, not the person — this is `critical`, not `personal`. "You always write buggy code" targets the person — this is `personal`.

### 6.4 informational
Neutral information sharing. Facts, data, status updates, requests for information.

**Examples**:
- "The deadline is Friday."
- "Here's the link to the document."
- "Can you share the latest numbers?"
- "Meeting moved to 4pm."

---

## 7. Explanation

Write a single short sentence explaining your assessment. Be specific about what triggered your judgment.

**Good explanations** (specific, actionable):
- "The phrase 'you clearly' targets the person rather than the issue."
- "Uses exaggerated politeness that reads as sarcastic after a failure."
- "Expresses frustration but also acknowledges effort — constructive overall."
- "Given the workplace context, 'lol ok' reads as dismissive."

**Bad explanations** (generic, unhelpful):
- "It's hostile."
- "Sounds aggressive."
- "Medium risk."

**For low-risk messages**, provide positive confirmation:
- "Direct but respectful technical feedback."
- "Clear and constructive, acknowledges effort."
- "Professional and factual."

---

## 8. Needs Attention

A boolean flag: would you pause before sending this message?

- `true`: I would re-read or revise this before sending.
- `false`: I would send this as-is.

This is a subjective, high-level judgment. It should correlate with risk level (high/critical → true, none/low → false) but may diverge. A medium-risk message about a sensitive topic might need attention even if the tone isn't aggressive.

---

## 9. Edge Cases and Ambiguity

### 9.1 Short / Fragmentary Messages
- "lol ok" → Could be neutral (dismissive) or sarcastic. Use context: in a workplace chat, this likely conveys dismissiveness.
- "fine." → Likely frustrated/resentful, not neutral.
- "..." → Highly ambiguous. Rate conservatively (medium risk, neutral tone, informational intent).

### 9.2 Cultural and Regional Variations
- Italian directness: "Ma che stai a dì?" is dialectal and confrontational in tone. Annotators from different Italian regions may perceive this differently. This is expected.
- Formality (tu/Lei): Italian uses tu (informal) and Lei (formal). Formality level affects perceived tone. "Lei è pregato di..." is formally polite but may be perceived as passive-aggressive depending on context.

### 9.3 Non-Native English
Messages from non-native speakers may use unconventional phrasing. Do not penalize grammar. Judge the intended meaning:
- "You no understand this" → Likely critical or personal, not a grammar error to ignore.

### 9.4 Direct Communication Styles
Cultures with direct communication norms (German, Dutch, Israeli) value conciseness. A message like "This is wrong. Fix it." may be perfectly normal in those cultures. Annotate how YOU perceive it, and record your cultural background.

---

## 10. Annotation Workflow

1. Read the message. Read it a second time.
2. Ask yourself: "How would I feel receiving this in a work chat?"
3. Assign the risk score (0.0-1.0).
4. Derive the risk level from the score range.
5. For each tone: is it present? With what confidence?
6. Choose the primary intent.
7. Write a specific explanation.
8. Decide: needs attention?

**Time target**: 30-60 seconds per message after familiarization.

---

## 11. Annotator Metadata

Please record:
- **Annotator ID**: anonymous identifier
- **Native language(s)**: what you grew up speaking
- **Cultural background**: region or culture you identify with
- **Gender**: optional
- **Age range**: optional (20-30, 30-40, 40-50, 50+)
- **Neurodivergence**: optional self-description

This metadata contextualizes your annotations. It helps us understand when and why annotators disagree.

---

## 12. Examples from Each Risk Level

### None (0.0-0.2)
- "Thanks for the heads up!" → positive, constructive
- "Here's the updated doc." → neutral, informational
- "Great point, I agree." → positive, constructive

### Low (0.2-0.4)
- "I'm not sure about this approach." → neutral, critical
- "Can we wrap this up? I have another meeting." → neutral, informational
- "This has some issues we should address." → neutral, critical

### Medium (0.4-0.6)
- "This is really frustrating." → frustrated, critical
- "Per my last email..." → sarcastic, critical
- "We keep having this conversation." → frustrated, critical

### High (0.6-0.8)
- "You clearly don't get it." → hostile, personal
- "Maybe someone competent should handle this." → sarcastic, hostile, personal
- "I'm not surprised at all." → sarcastic, personal

### Critical (0.8-1.0)
- "You're completely useless." → hostile, personal
- "Everyone knows you can't code." → hostile, personal
- "Just quit. Seriously." → hostile, personal
