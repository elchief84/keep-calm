# Keep Calm — User Testing Plan

> Validating the behavioral hypothesis: *pre-send analysis changes communication behavior.*

## 1. Hypothesis

**Primary**: Users who receive pre-send communication risk analysis will revise or not send at least 20% of flagged messages.

**Secondary**:
- At least 7/10 users would use the tool at least weekly
- No more than 2/10 users report feeling judged or anxious

## 2. Test Design

### 2.1 Duration
2 weeks per participant. The first 3 days are an adaptation period (data excluded from analysis).

### 2.2 Participants (target: 10–20)
Recruitment criteria:
- Communicates primarily via text (Slack, email, Teams, etc.)
- Has expressed regret about past messages ("I shouldn't have sent that") at least once
- Mixed professional backgrounds (engineers, managers, support, designers)
- English and/or Italian speakers (MVP languages)

**Important**: Participants must explicitly consent to use the tool. No passive deployment.

### 2.3 Setup
Each participant:
1. Installs the CLI tool (`pip install -e .`)
2. Copies the `keep-calm-repl.sh` script or uses `keep-calm "message"`
3. Uses the tool *voluntarily* before sending messages they are unsure about
4. Records a brief note after each use (see §3)

### 2.4 Interaction Flow
```
Write message → invoke keep-calm → see analysis → decide: send / revise / discard
```
The tool is never mandatory. Participants choose when to invoke it.

## 3. Data Collection

### 3.1 Per-Use Log (participant self-reports)
After each Keep Calm invocation, the participant records:

| Field | Options |
|---|---|
| What did you do? | Sent as-is / Revised then sent / Did not send |
| Was the analysis helpful? | Yes / Partially / No |
| Did you feel judged? | Yes / No |
| Any comments? | Free text |

### 3.2 Weekly Survey
At the end of each week:

1. How many times did you use Keep Calm? (approximate)
2. Did it change any message you sent? (yes / no)
3. On a scale of 1–10, how useful was it?
4. On a scale of 1–10, did you feel judged or anxious using it?
5. Would you continue using it? (yes / no / maybe)
6. What worked well? (free text)
7. What was frustrating? (free text)
8. Did you disagree with any analysis? Can you recall the message? (free text)

### 3.3 Exit Interview
At the end of the 2 weeks:

1. Overall impression (1–10)
2. Did it change how you think about your messages, even when not using the tool?
3. Would you recommend it to a colleague?
4. Any feature requests?

## 4. Success Criteria

| Metric | Target | Measurement |
|---|---|---|
| Revision rate on flagged messages | >= 20% | Per-use log: (revised + not sent) / used |
| Willingness to use weekly | >= 7/10 | Weekly survey: "Would you continue using it?" |
| Psychological safety | <= 2/10 | Weekly survey: "felt judged" rating >= 7/10 |
| Analysis agreement | >= 70% | "Did you disagree with any analysis?" |

## 5. Qualitative Analysis

Beyond metrics, look for:

- **Patterns in disagreement**: What types of messages does the model consistently misjudge?
- **False positive triggers**: Messages that were flagged but the participant found the analysis wrong
- **False negative gaps**: Situations where the participant *wished* the tool had flagged something
- **Cultural/style mismatches**: Do participants with direct communication styles report more false positives?
- **Explanation quality**: Do users understand *why* the tool flagged a message?

## 6. Ethical Considerations

- **All participation is voluntary**. No one should be required to use the tool.
- **No message content is collected**. Only the per-use decisions and survey responses.
- **Anonymity**: Results are reported in aggregate. Individual messages are never shared.
- **Opt-out at any time**: Participants can stop without explanation.
- **No surveillance**: The tool runs locally. No log files, no network requests, no reporting.

## 7. Materials

### Participant Welcome Email Template
```
Subject: Keep Calm — User Testing Invitation

Hi [name],

I'm testing a tool called Keep Calm. It analyzes messages before you send them
and gives feedback on how they might be perceived — think of it as a
"communication companion" for text messages.

What it does:
- You type or paste a message into the tool
- It returns a risk score, tone analysis, and explanation
- You decide what to do — it never blocks or censors

What it doesn't do:
- Store or log your messages
- Send data anywhere (runs entirely on your machine)
- Judge you or your intentions

The test takes 2 weeks. You use it voluntarily, whenever you want, before sending
messages you're unsure about. Each time, you jot down what you decided (takes 10
seconds). At the end of each week there's a short survey.

Interested? Reply to this and I'll help you get set up.

Thanks!
[sender]
```

## 8. Timeline

| Week | Activity |
|---|---|
| Pre-week | Recruit participants, set up tooling |
| Week 1 | Participants use the tool (3-day adaptation, then data collection) |
| Week 1 end | Weekly survey #1 |
| Week 2 | Continued usage |
| Week 2 end | Weekly survey #2 + exit interview |
| Post-week | Analyze results, write report |

## 9. Go / No-Go Decision

After user testing:

- **Go to v1.0**: All primary success criteria met AND no red flags in qualitative analysis
- **Go with fixes**: Primary criteria partially met, clear fixes identified
- **No-go / pivot**: Primary criteria not met, users don't find value
