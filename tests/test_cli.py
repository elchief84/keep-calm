import sys
from unittest.mock import patch

import pytest

from keep_calm.schemas import AnalysisResult, Intent, RiskLevel, Tone, ToneResult


@pytest.fixture
def mock_analyzer():
    result = AnalysisResult(
        communication_risk=0.72,
        risk_level=RiskLevel.HIGH,
        tones=[
            ToneResult(label=Tone.HOSTILE, confidence=0.84),
            ToneResult(label=Tone.FRUSTRATED, confidence=0.61),
        ],
        intent=Intent.PERSONAL,
        intent_confidence=0.88,
        needs_attention=True,
        explanation="This message may be perceived as hostile.",
    )

    with patch("keep_calm.cli.KeepCalmAnalyzer") as mock_analyzer_cls:
        instance = mock_analyzer_cls.return_value
        instance.analyze.return_value = result
        yield mock_analyzer_cls


def test_cli_no_args(capsys):
    sys.argv = ["keep-calm"]
    with pytest.raises(SystemExit) as exc:
        from keep_calm import cli
        cli.main()
    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert "Usage:" in captured.out


def test_cli_with_message(mock_analyzer, capsys):
    sys.argv = ["keep-calm", "You", "are", "wrong"]
    from keep_calm import cli
    cli.main()
    captured = capsys.readouterr()
    assert "Risk:" in captured.out
    assert "0.72" in captured.out
    assert "high" in captured.out


def test_cli_empty_message_after_join(capsys):
    sys.argv = ["keep-calm", ""]
    with pytest.raises(SystemExit) as exc:
        from keep_calm import cli
        cli.main()
    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert "Error:" in captured.out
