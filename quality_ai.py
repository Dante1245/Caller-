from __future__ import annotations

from dataclasses import dataclass


@dataclass
class QualityAssessment:
    score: float
    label: str
    recommendation: str


def assess_call_quality(
    latency_ms: float,
    speaking_rate_wpm: float,
    mood: str,
    has_word_timestamps: bool,
) -> QualityAssessment:
    score = 100.0

    if latency_ms > 1600:
        score -= 35
    elif latency_ms > 1000:
        score -= 20
    elif latency_ms > 700:
        score -= 10

    if speaking_rate_wpm > 190:
        score -= 8
    elif speaking_rate_wpm < 70 and speaking_rate_wpm > 0:
        score -= 5

    if mood == "negative":
        score -= 6

    if not has_word_timestamps:
        score -= 4

    score = max(0.0, min(100.0, score))

    if score >= 88:
        return QualityAssessment(score, "excellent", "Keep current settings.")
    if score >= 74:
        return QualityAssessment(
            score,
            "good",
            "Consider quality mode high for more natural output.",
        )
    if score >= 58:
        return QualityAssessment(
            score,
            "fair",
            "Improve room acoustics or reduce background noise.",
        )
    return QualityAssessment(
        score,
        "poor",
        "Switch to test mode, recalibrate voice sample, and reduce latency load.",
    )
