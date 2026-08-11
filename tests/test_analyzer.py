
from keep_calm.analyzer import (
    INTENT_LABELS,
    RISK_THRESHOLDS,
    TONE_LABELS,
    _build_explanation,
    _risk_to_level,
)
from keep_calm.schemas import Intent, RiskLevel, Tone, ToneResult


class TestConstants:
    def test_tone_labels_match_enum(self):
        assert set(TONE_LABELS) == {t.value for t in Tone}

    def test_intent_labels_match_enum(self):
        assert set(INTENT_LABELS) == {i.value for i in Intent}

    def test_risk_thresholds_cover_range(self):
        thresholds = [t for t, _ in RISK_THRESHOLDS]
        assert thresholds[0] == 0.25
        assert thresholds[-1] == 1.0
        assert thresholds == sorted(thresholds)

    def test_risk_thresholds_map_all_levels(self):
        levels = {level for _, level in RISK_THRESHOLDS}
        assert levels == {
            RiskLevel.NONE,
            RiskLevel.LOW,
            RiskLevel.MEDIUM,
            RiskLevel.HIGH,
            RiskLevel.CRITICAL,
        }


class TestRiskToLevel:
    def test_zero(self):
        assert _risk_to_level(0.0) == RiskLevel.NONE

    def test_none_boundaries(self):
        assert _risk_to_level(0.0) == RiskLevel.NONE
        assert _risk_to_level(0.24) == RiskLevel.NONE

    def test_low_boundaries(self):
        assert _risk_to_level(0.25) == RiskLevel.LOW
        assert _risk_to_level(0.44) == RiskLevel.LOW

    def test_medium_boundaries(self):
        assert _risk_to_level(0.45) == RiskLevel.MEDIUM
        assert _risk_to_level(0.64) == RiskLevel.MEDIUM

    def test_high_boundaries(self):
        assert _risk_to_level(0.65) == RiskLevel.HIGH
        assert _risk_to_level(0.84) == RiskLevel.HIGH

    def test_critical_boundaries(self):
        assert _risk_to_level(0.85) == RiskLevel.CRITICAL
        assert _risk_to_level(0.99) == RiskLevel.CRITICAL
        assert _risk_to_level(1.0) == RiskLevel.CRITICAL

    def test_interpolation(self):
        assert _risk_to_level(0.1) == RiskLevel.NONE
        assert _risk_to_level(0.35) == RiskLevel.LOW
        assert _risk_to_level(0.55) == RiskLevel.MEDIUM
        assert _risk_to_level(0.75) == RiskLevel.HIGH
        assert _risk_to_level(0.95) == RiskLevel.CRITICAL


class TestBuildExplanation:
    def _t(self, *labels: str) -> list[ToneResult]:
        return [ToneResult(label=Tone(label), confidence=0.8) for label in labels]

    # --- Hostile / Sarcastic combinations ---

    def test_sarcastic_plus_hostile(self):
        tones = self._t("sarcastic", "hostile")
        explanation = _build_explanation(0.7, tones, Intent.PERSONAL)
        assert "sarcasm" in explanation.lower()
        assert "hostility" in explanation.lower()
        assert "personal attack" in explanation.lower()

    def test_hostile_only(self):
        tones = self._t("hostile")
        explanation = _build_explanation(0.7, tones, Intent.PERSONAL)
        assert "hostile" in explanation.lower()
        assert "personally" in explanation.lower()

    def test_sarcastic_only(self):
        tones = self._t("sarcastic")
        explanation = _build_explanation(0.5, tones, Intent.CRITICAL)
        assert "sarcastic" in explanation.lower()
        assert "differently" in explanation.lower()

    # --- Frustrated patterns ---

    def test_frustrated_critical(self):
        tones = self._t("frustrated")
        explanation = _build_explanation(0.5, tones, Intent.CRITICAL)
        assert "frustration" in explanation.lower()

    def test_frustrated_personal(self):
        tones = self._t("frustrated")
        explanation = _build_explanation(0.5, tones, Intent.PERSONAL)
        assert "frustration" in explanation.lower()

    def test_frustrated_constructive(self):
        tones = self._t("frustrated")
        explanation = _build_explanation(0.5, tones, Intent.CONSTRUCTIVE)
        assert "frustration" in explanation.lower()
        assert "constructive" in explanation.lower()

    # --- Low risk ---

    def test_very_low_risk(self):
        tones = self._t("neutral")
        explanation = _build_explanation(0.05, tones, Intent.INFORMATIONAL)
        assert "clear" in explanation.lower()
        assert "professional" in explanation.lower()

    def test_low_risk(self):
        tones = self._t("neutral")
        explanation = _build_explanation(0.25, tones, Intent.INFORMATIONAL)
        assert "direct" in explanation.lower()
        assert "professional" in explanation.lower()

    # --- Personal intent ---

    def test_personal_intent_no_tones(self):
        tones = self._t("neutral")
        explanation = _build_explanation(0.45, tones, Intent.PERSONAL)
        assert "person" in explanation.lower()

    # --- Default fallback ---

    def test_default_explanation(self):
        tones = self._t("neutral")
        explanation = _build_explanation(0.5, tones, Intent.CRITICAL)
        assert "critical" in explanation.lower()
        assert "actionable" in explanation.lower()

    # --- Positive tone doesn't override risk-based fallback ---

    def test_positive_low_risk(self):
        tones = self._t("positive")
        explanation = _build_explanation(0.1, tones, Intent.CONSTRUCTIVE)
        assert "supportive" in explanation.lower()
        assert "positively" in explanation.lower()

    # --- Empty tones case is handled upstream, but test edge ---

    def test_explanation_always_returns_string(self):
        for risk in (0.0, 0.3, 0.5, 0.7, 0.9):
            for intent in Intent:
                for tones in (
                    self._t("neutral"),
                    self._t("frustrated"),
                    self._t("hostile"),
                    self._t("sarcastic"),
                    self._t("positive"),
                ):
                    result = _build_explanation(risk, tones, intent)
                    assert isinstance(result, str)
                    assert len(result) > 0
