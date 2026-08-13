import pytest
from fastapi.testclient import TestClient

from keep_calm.schemas import AnalysisResult, Intent, RiskLevel, Tone, ToneResult
from keep_calm.server import create_app, get_analyzer


@pytest.fixture
def mock_analyzer(monkeypatch):
    result = AnalysisResult(
        communication_risk=0.72,
        risk_level=RiskLevel.HIGH,
        tones=[ToneResult(label=Tone.HOSTILE, confidence=0.84)],
        intent=Intent.PERSONAL,
        intent_confidence=0.88,
        needs_attention=True,
        explanation="This message may be perceived as hostile.",
    )

    class FakeAnalyzer:
        def analyze(self, text):
            return result

    monkeypatch.setattr("keep_calm.server.get_analyzer", lambda: FakeAnalyzer())
    return FakeAnalyzer()


def test_health(mock_analyzer):
    client = TestClient(create_app())
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_analyze(mock_analyzer):
    client = TestClient(create_app())
    response = client.post("/analyze", json={"text": "You are wrong."})
    assert response.status_code == 200
    data = response.json()
    assert data["communication_risk"] == 0.72
    assert data["risk_level"] == "high"
    assert data["intent"] == "personal"
    assert data["needs_attention"] is True
    assert data["tones"][0]["label"] == "hostile"


def test_analyze_empty_text(mock_analyzer):
    client = TestClient(create_app())
    response = client.post("/analyze", json={"text": ""})
    assert response.status_code == 422


def test_analyze_too_long(mock_analyzer):
    client = TestClient(create_app())
    response = client.post("/analyze", json={"text": "x" * 2001})
    assert response.status_code == 422


def test_analyze_missing_text(mock_analyzer):
    client = TestClient(create_app())
    response = client.post("/analyze", json={})
    assert response.status_code == 422


def test_analyzer_singleton(monkeypatch):
    calls = []

    class CountingAnalyzer:
        def __init__(self):
            calls.append(1)

    monkeypatch.setattr("keep_calm.analyzer.KeepCalmAnalyzer", CountingAnalyzer)
    monkeypatch.setattr("keep_calm.server._analyzer", None)

    a1 = get_analyzer()
    a2 = get_analyzer()
    assert a1 is a2
    assert len(calls) == 1
