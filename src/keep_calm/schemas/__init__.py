from enum import Enum

from pydantic import BaseModel, Field


class RiskLevel(str, Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Tone(str, Enum):
    NEUTRAL = "neutral"
    FRUSTRATED = "frustrated"
    HOSTILE = "hostile"
    SARCASTIC = "sarcastic"
    POSITIVE = "positive"


class Intent(str, Enum):
    CONSTRUCTIVE = "constructive"
    CRITICAL = "critical"
    PERSONAL = "personal"
    INFORMATIONAL = "informational"


class ToneResult(BaseModel):
    label: Tone
    confidence: float = Field(ge=0.0, le=1.0)


class AnalysisResult(BaseModel):
    communication_risk: float = Field(ge=0.0, le=1.0)
    risk_level: RiskLevel
    tones: list[ToneResult]
    intent: Intent
    intent_confidence: float = Field(ge=0.0, le=1.0)
    needs_attention: bool
    explanation: str
