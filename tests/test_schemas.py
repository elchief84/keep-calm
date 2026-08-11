import pytest
from pydantic import ValidationError

from keep_calm.schemas import AnalysisResult, Intent, RiskLevel, Tone, ToneResult


class TestRiskLevel:
    def test_values(self):
        assert RiskLevel.NONE.value == "none"
        assert RiskLevel.LOW.value == "low"
        assert RiskLevel.MEDIUM.value == "medium"
        assert RiskLevel.HIGH.value == "high"
        assert RiskLevel.CRITICAL.value == "critical"

    def test_is_string_enum(self):
        assert RiskLevel.NONE == "none"
        assert isinstance(RiskLevel.LOW, str)


class TestTone:
    def test_all_labels_present(self):
        expected = {"neutral", "frustrated", "hostile", "sarcastic", "positive"}
        assert {t.value for t in Tone} == expected


class TestIntent:
    def test_all_labels_present(self):
        expected = {"constructive", "critical", "personal", "informational"}
        assert {i.value for i in Intent} == expected


class TestToneResult:
    def test_valid(self):
        tr = ToneResult(label=Tone.NEUTRAL, confidence=0.75)
        assert tr.label == Tone.NEUTRAL
        assert tr.confidence == 0.75

    def test_confidence_bounds(self):
        ToneResult(label=Tone.POSITIVE, confidence=0.0)
        ToneResult(label=Tone.HOSTILE, confidence=1.0)

    def test_confidence_out_of_range_low(self):
        with pytest.raises(ValidationError):
            ToneResult(label=Tone.NEUTRAL, confidence=-0.1)

    def test_confidence_out_of_range_high(self):
        with pytest.raises(ValidationError):
            ToneResult(label=Tone.NEUTRAL, confidence=1.1)


class TestAnalysisResult:
    def test_full_constructive_message(self):
        result = AnalysisResult(
            communication_risk=0.15,
            risk_level=RiskLevel.NONE,
            tones=[ToneResult(label=Tone.POSITIVE, confidence=0.92)],
            intent=Intent.CONSTRUCTIVE,
            intent_confidence=0.88,
            needs_attention=False,
            explanation="Reads as constructive feedback.",
        )
        assert result.communication_risk == 0.15
        assert result.risk_level == RiskLevel.NONE
        assert len(result.tones) == 1
        assert result.intent == Intent.CONSTRUCTIVE
        assert result.needs_attention is False

    def test_full_hostile_message(self):
        result = AnalysisResult(
            communication_risk=0.88,
            risk_level=RiskLevel.CRITICAL,
            tones=[
                ToneResult(label=Tone.HOSTILE, confidence=0.95),
                ToneResult(label=Tone.FRUSTRATED, confidence=0.72),
            ],
            intent=Intent.PERSONAL,
            intent_confidence=0.91,
            needs_attention=True,
            explanation="Targets the person rather than the issue.",
        )
        assert result.risk_level == RiskLevel.CRITICAL
        assert result.needs_attention is True
        assert len(result.tones) == 2

    def test_risk_out_of_range(self):
        with pytest.raises(ValidationError):
            AnalysisResult(
                communication_risk=1.5,
                risk_level=RiskLevel.HIGH,
                tones=[ToneResult(label=Tone.NEUTRAL, confidence=0.5)],
                intent=Intent.INFORMATIONAL,
                intent_confidence=0.5,
                needs_attention=False,
                explanation="",
            )

    def test_negative_risk(self):
        with pytest.raises(ValidationError):
            AnalysisResult(
                communication_risk=-0.1,
                risk_level=RiskLevel.NONE,
                tones=[ToneResult(label=Tone.NEUTRAL, confidence=0.5)],
                intent=Intent.INFORMATIONAL,
                intent_confidence=0.5,
                needs_attention=False,
                explanation="",
            )

    def test_model_serialization(self):
        result = AnalysisResult(
            communication_risk=0.42,
            risk_level=RiskLevel.MEDIUM,
            tones=[ToneResult(label=Tone.SARCASTIC, confidence=0.67)],
            intent=Intent.CRITICAL,
            intent_confidence=0.78,
            needs_attention=True,
            explanation="May be perceived as sarcastic.",
        )
        data = result.model_dump()
        assert data["risk_level"] == "medium"
        assert data["intent"] == "critical"
        assert data["tones"][0]["label"] == "sarcastic"
