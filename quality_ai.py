from __future__ import annotations

from dataclasses import dataclass

# Penalty constants
LATENCY_SEVERE_PENALTY = 35
LATENCY_HIGH_PENALTY = 20
LATENCY_MODERATE_PENALTY = 10
SPEAKING_RATE_FAST_PENALTY = 8
SPEAKING_RATE_SLOW_PENALTY = 5
MOOD_NEGATIVE_PENALTY = 6
NO_TIMESTAMPS_PENALTY = 4

# Threshold constants
LATENCY_SEVERE_MS = 1600
LATENCY_HIGH_MS = 1000
LATENCY_MODERATE_MS = 700
SPEAKING_RATE_FAST_WPM = 190
SPEAKING_RATE_SLOW_WPM = 70

# Quality score thresholds
QUALITY_EXCELLENT = 88
QUALITY_GOOD = 74
QUALITY_FAIR = 58

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
    """
    Assess call quality based on multiple metrics.
    
    Args:
        latency_ms: Network latency in milliseconds (must be non-negative)
        speaking_rate_wpm: Speaking rate in words per minute (must be non-negative)
        mood: Detected mood sentiment (e.g., 'positive', 'negative')
        has_word_timestamps: Whether word-level timestamps are available
    
    Returns:
        QualityAssessment with score, label, and recommendation
        
    Raises:
        ValueError: If latency_ms or speaking_rate_wpm are negative
    """
    # Input validation
    if latency_ms < 0:
        raise ValueError("latency_ms must be non-negative")
    if speaking_rate_wpm < 0:
        raise ValueError("speaking_rate_wpm must be non-negative")
    
    score = 100.0

    if latency_ms > LATENCY_SEVERE_MS:
        score -= LATENCY_SEVERE_PENALTY
    elif latency_ms > LATENCY_HIGH_MS:
        score -= LATENCY_HIGH_PENALTY
    elif latency_ms > LATENCY_MODERATE_MS:
        score -= LATENCY_MODERATE_PENALTY

    if speaking_rate_wpm > SPEAKING_RATE_FAST_WPM:
        score -= SPEAKING_RATE_FAST_PENALTY
    elif 0 < speaking_rate_wpm < SPEAKING_RATE_SLOW_WPM:
        score -= SPEAKING_RATE_SLOW_PENALTY

    if mood == "negative":
        score -= MOOD_NEGATIVE_PENALTY

    if not has_word_timestamps:
        score -= NO_TIMESTAMPS_PENALTY

    score = max(0.0, min(100.0, score))

    if score >= QUALITY_EXCELLENT:
        return QualityAssessment(score, "excellent", "Keep current settings.")
    if score >= QUALITY_GOOD:
        return QualityAssessment(
            score,
            "good",
            "Consider quality mode high for more natural output.",
        )
    if score >= 58:
    if score >= QUALITY_FAIR:
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
    )
