from __future__ import annotations

from dataclasses import dataclass
import math

# Penalty constants
LATENCY_SEVERE_PENALTY = 35
LATENCY_HIGH_PENALTY = 20
LATENCY_MODERATE_PENALTY = 10
SPEAKING_RATE_FAST_PENALTY = 8
SPEAKING_RATE_SLOW_PENALTY = 5
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
    clipping_percent: float = 0.0,
    rms_dbfs: float | None = None,
) -> QualityAssessment:
    """
    Assess call quality based on multiple metrics.

    Args:
        latency_ms: Processing latency in milliseconds (must be non-negative)
        speaking_rate_wpm: Speaking rate in words per minute (must be non-negative)
        mood: Detected mood sentiment (e.g., 'positive', 'negative')
        has_word_timestamps: Whether word-level timestamps are available

    Returns:
        QualityAssessment with score, label, and recommendation

    Raises:
        ValueError: If latency_ms or speaking_rate_wpm are negative
    """
    # Input validation
    if not math.isfinite(latency_ms) or latency_ms < 0:
        raise ValueError("latency_ms must be non-negative")
    if not math.isfinite(speaking_rate_wpm) or speaking_rate_wpm < 0:
        raise ValueError("speaking_rate_wpm must be non-negative")

    if rms_dbfs is not None and not math.isfinite(rms_dbfs):
        raise ValueError("rms_dbfs must be finite")
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

    if not math.isfinite(clipping_percent) or not 0 <= clipping_percent <= 100:
        raise ValueError("clipping_percent must be in 0–100")
    if clipping_percent > 1:
        score -= 20
    if rms_dbfs is not None and rms_dbfs < -45:
        score -= 10

    if not has_word_timestamps:
        score -= NO_TIMESTAMPS_PENALTY

    score = max(0.0, min(100.0, score))

    label = (
        "excellent"
        if score >= QUALITY_EXCELLENT
        else "good"
        if score >= QUALITY_GOOD
        else "fair"
        if score >= QUALITY_FAIR
        else "poor"
    )
    if clipping_percent > 1:
        tip = "Input is clipping. Lower microphone gain or move farther from the microphone."
    elif rms_dbfs is not None and rms_dbfs < -45:
        tip = "Input is quiet. Move closer to the microphone or raise its input gain."
    elif latency_ms > LATENCY_HIGH_MS:
        tip = "Processing is slow. Try a smaller model or the faster-whisper backend."
    elif not has_word_timestamps:
        tip = "No word timestamps available; timing-based analysis is limited."
    elif speaking_rate_wpm > SPEAKING_RATE_FAST_WPM:
        tip = "Speech is fast. Slow down slightly for easier comprehension."
    else:
        tip = "Current measurements are healthy."
    return QualityAssessment(score, label, tip)
