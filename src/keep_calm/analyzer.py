"""Keep Calm Analyzer — loads trained models and analyzes text.

Usage:
    from keep_calm import KeepCalmAnalyzer
    analyzer = KeepCalmAnalyzer()
    result = analyzer.analyze("Your message here")
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from keep_calm.schemas import AnalysisResult, Intent, RiskLevel, Tone, ToneResult

MODEL_NAME = "distilbert-base-multilingual-cased"
DEFAULT_MODELS_DIR = Path(__file__).resolve().parents[2] / "data" / "models"

TONE_LABELS = ["neutral", "frustrated", "hostile", "sarcastic", "positive"]
INTENT_LABELS = ["constructive", "critical", "personal", "informational"]
RISK_THRESHOLDS = [
    (0.25, RiskLevel.NONE),
    (0.45, RiskLevel.LOW),
    (0.65, RiskLevel.MEDIUM),
    (0.85, RiskLevel.HIGH),
    (1.0, RiskLevel.CRITICAL),
]


def _risk_to_level(score: float) -> RiskLevel:
    for threshold, level in RISK_THRESHOLDS:
        if score < threshold:
            return level
    return RiskLevel.CRITICAL


def _has_tone(tone_labels: list[str], *labels: str) -> bool:
    return any(label in tone_labels for label in labels)


def _build_explanation(risk: float, tones: list[ToneResult], intent: Intent) -> str:
    tone_labels = [t.label.value for t in tones]

    def ht(*labels: str) -> bool:
        return _has_tone(tone_labels, *labels)

    # ---- Hostility ----
    if ht("hostile"):
        if ht("sarcastic"):
            return (
                "This message combines sarcasm with hostility — "
                "it may be perceived as a personal attack disguised as humor."
            )
        if ht("frustrated"):
            if intent == Intent.PERSONAL:
                return (
                    "This message mixes frustration with hostility "
                    "and appears to target the person directly. "
                    "Consider stepping back before sending."
                )
            return (
                "This message comes across as both frustrated and hostile. "
                "The combination may escalate the situation rather than resolve it."
            )
        if intent == Intent.PERSONAL:
            return (
                "This message reads as hostile and personally directed. "
                "It may damage trust and working relationships."
            )
        return (
            "This message may be perceived as hostile — "
            "it appears to target a person rather than addressing an issue."
        )

    # ---- Sarcasm without hostility ----
    if ht("sarcastic"):
        if ht("frustrated"):
            return (
                "This message uses sarcasm to express frustration. "
                "The ironic tone may be misinterpreted and create unnecessary tension."
            )
        if intent == Intent.PERSONAL:
            return (
                "The sarcastic tone here could feel like a personal dig, "
                "even if not intended. Consider saying what you mean directly."
            )
        return (
            "This message reads as sarcastic — "
            "the ironic tone may be perceived differently than intended by the recipient."
        )

    # ---- Frustration (non-hostile, non-sarcastic) ----
    if ht("frustrated"):
        if ht("positive"):
            return (
                "This message balances frustration with a positive tone — "
                "constructive overall, but the frustration may still be noticed."
            )
        if intent == Intent.PERSONAL:
            return (
                "The frustration in this message appears directed at a person "
                "rather than the situation. Consider focusing on the issue itself."
            )
        if intent == Intent.CRITICAL:
            return (
                "This message expresses frustration while being critical. "
                "Consider separating the emotion from the feedback to make it more actionable."
            )
        if intent == Intent.CONSTRUCTIVE:
            return (
                "Frustration is noticeable here, but the message remains constructive. "
                "A calmer delivery could make the feedback land better."
            )
        if risk >= 0.6:
            return (
                "Frustrated tone with elevated risk — "
                "the recipient may perceive this more negatively than intended."
            )
        return (
            "This message carries some frustration. "
            "Consider whether the tone matches your intent."
        )

    # ---- Positive tone ----
    if ht("positive") and len(tone_labels) == 1:
        if risk < 0.2:
            return (
                "This message reads as supportive and well-intentioned — "
                "it is likely to be received positively."
            )
        if risk < 0.4:
            return "A positive and respectful message. Direct but constructive."
        return (
            "The tone is positive, but the risk level suggests "
            "the recipient may still perceive an edge. "
            "Consider the overall message framing."
        )

    # ---- Risk-based fallback (no strong tone signals) ----
    if risk < 0.2:
        if intent == Intent.INFORMATIONAL:
            return (
                "This message reads as clear, factual, "
                "and appropriate for a professional context."
            )
        if intent == Intent.CONSTRUCTIVE:
            return (
                "Constructive and well-framed — "
                "this message is likely to be received as helpful."
            )
        return "This message reads as clear and constructive."
    if risk < 0.4:
        if intent == Intent.INFORMATIONAL:
            return "Direct and to the point — this is professional, concise communication."
        if intent == Intent.CRITICAL:
            return (
                "This message is direct but appears respectful. "
                "The criticism is focused on the issue, not the person."
            )
        return "This message is direct but appears respectful."
    if risk < 0.6:
        if intent == Intent.PERSONAL:
            return (
                "This message seems to address the person rather than the situation. "
                "Consider framing it around the work instead."
            )
        if intent == Intent.CRITICAL:
            return (
                "The message reads as critical. "
                "Consider adding constructive suggestions to make the feedback more actionable."
            )
        return (
            "This message may be perceived more critically than intended. "
            "A small rephrase could improve how it lands."
        )
    if risk < 0.8:
        if intent == Intent.PERSONAL:
            return (
                "This message appears to target a person rather than the issue. "
                "Consider focusing on the work or situation instead."
            )
        return (
            "This message is likely to be perceived negatively by the recipient. "
            "Consider reviewing the tone and wording."
        )
    # risk >= 0.8
    if intent == Intent.PERSONAL:
        return (
            "This message strongly targets the person "
            "and is likely to cause significant harm to the relationship. "
            "Strongly consider revising."
        )
    return (
        "This message carries a high risk of being perceived as aggressive or damaging. "
        "Review carefully before sending."
    )


class KeepCalmAnalyzer:
    def __init__(self, models_dir: str | Path | None = None):
        models_dir = Path(models_dir) if models_dir else DEFAULT_MODELS_DIR
        self.device = torch.device("cpu")
        self.threshold = 0.4

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        # Risk model
        self.risk_encoder = AutoModel.from_pretrained(MODEL_NAME)
        self.risk_head = nn.Sequential(
            nn.Linear(768, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 1), nn.Sigmoid()
        )
        self.risk_encoder.load_state_dict(
            torch.load(models_dir / "risk_encoder.pt", map_location="cpu", weights_only=True)
        )
        self.risk_head.load_state_dict(
            torch.load(models_dir / "risk_head.pt", map_location="cpu", weights_only=True)
        )
        self.risk_encoder.eval()
        self.risk_head.eval()

        # Tone model
        self.tone_encoder = AutoModel.from_pretrained(MODEL_NAME)
        self.tone_head = nn.Sequential(
            nn.Linear(768, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 5), nn.Sigmoid()
        )
        self.tone_encoder.load_state_dict(
            torch.load(models_dir / "tone_encoder.pt", map_location="cpu", weights_only=True)
        )
        self.tone_head.load_state_dict(
            torch.load(models_dir / "tone_head.pt", map_location="cpu", weights_only=True)
        )
        self.tone_encoder.eval()
        self.tone_head.eval()

        # Intent model
        self.intent_encoder = AutoModel.from_pretrained(MODEL_NAME)
        self.intent_head = nn.Sequential(
            nn.Linear(768, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 4)
        )
        self.intent_encoder.load_state_dict(
            torch.load(models_dir / "intent_encoder.pt", map_location="cpu", weights_only=True)
        )
        self.intent_head.load_state_dict(
            torch.load(models_dir / "intent_head.pt", map_location="cpu", weights_only=True)
        )
        self.intent_encoder.eval()
        self.intent_head.eval()

    def analyze(self, text: str) -> AnalysisResult:
        enc = self.tokenizer(
            text, truncation=True, padding="max_length", max_length=256, return_tensors="pt"
        )
        ids, mask = enc["input_ids"], enc["attention_mask"]

        with torch.no_grad():
            # Risk
            emb_risk = self.risk_encoder(ids, mask).last_hidden_state[:, 0, :]
            risk_score = float(self.risk_head(emb_risk).squeeze().item())

            # Tone
            emb_tone = self.tone_encoder(ids, mask).last_hidden_state[:, 0, :]
            tone_probs = self.tone_head(emb_tone).squeeze().tolist()
            tones = [
                ToneResult(label=Tone(label), confidence=round(conf, 4))
                for label, conf in zip(TONE_LABELS, tone_probs, strict=True)
                if conf >= self.threshold
            ]
            if not tones:
                tones = [ToneResult(label=Tone.NEUTRAL, confidence=0.5)]

            tones.sort(key=lambda t: t.confidence, reverse=True)

            # Boost risk when sarcasm or hostility is detected with high confidence
            top_tone = tones[0].label.value
            if top_tone in ("sarcastic", "hostile") and tones[0].confidence > 0.4:
                risk_score = max(risk_score, 0.55)
            elif "sarcastic" in [t.label.value for t in tones[:2]] and tones[1].confidence > 0.4:
                risk_score = max(risk_score, 0.45)

            # Reduce risk when the dominant tone is positive and no negative tones present
            tone_labels_set = {t.label.value for t in tones}
            negative_tones = {"hostile", "sarcastic", "frustrated"}
            if top_tone == "positive" and not (tone_labels_set & negative_tones):
                risk_score = min(risk_score, 0.35)

            # Intent
            emb_intent = self.intent_encoder(ids, mask).last_hidden_state[:, 0, :]
            intent_logits = self.intent_head(emb_intent).squeeze()
            intent_idx = int(intent_logits.argmax().item())
            intent_conf = round(float(torch.softmax(intent_logits, dim=-1).max().item()), 4)

        risk_level = _risk_to_level(risk_score)
        intent = Intent(INTENT_LABELS[intent_idx])
        needs_attention = risk_score >= 0.5
        explanation = _build_explanation(round(risk_score, 4), tones, intent)

        return AnalysisResult(
            communication_risk=round(risk_score, 4),
            risk_level=risk_level,
            tones=tones,
            intent=intent,
            intent_confidence=intent_conf,
            needs_attention=needs_attention,
            explanation=explanation,
        )
