"""Transparent local text heuristics; these do not call a generative AI service."""

from __future__ import annotations

import math
import re
from collections import Counter


def infer_intent(text: str) -> str:
    words = set(re.findall(r"\b[\w']+\b", text.lower()))
    for intent, tokens in (
        ("support", {"help", "support", "issue", "problem", "error"}),
        ("sales", {"buy", "price", "cost", "invoice", "payment"}),
        ("planning", {"schedule", "meeting", "tomorrow", "later"}),
        ("social", {"hello", "hi", "thanks"}),
    ):
        if words & tokens:
            return intent
    return "social" if "thank you" in text.lower() else "general"


def compute_speaking_rate_wpm(text: str, duration_s: float | None) -> float:
    if not duration_s or not math.isfinite(duration_s) or duration_s <= 0:
        return 0.0
    return len(text.split()) * 60 / duration_s


def suggest_response_style(mood: str, intent: str, wpm: float) -> str:
    if mood == "negative":
        return "calm and reassuring"
    if intent == "support":
        return "clear and step-by-step"
    if intent == "sales":
        return "confident and concise"
    return "slightly slower and articulate" if wpm > 170 else "natural and friendly"


def conversation_insights(text: str):
    """Extract source sentences, questions and candidate commitments without inventing facts."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]
    words = re.findall(r"\b[\w']+\b", text.lower())
    stopwords = set(
        "the a an and or to of in on is it for that this i you we they he she with be are was will can please have has do".split()
    )
    counts = Counter(w for w in words if w not in stopwords and len(w) > 2)
    ranked = sorted(
        enumerate(sentences),
        key=lambda pair: (
            -sum(counts[w] for w in set(re.findall(r"\b\w+\b", pair[1].lower())))
            / max(1, len(pair[1].split()) ** 0.5),
            pair[0],
        ),
    )
    selected = sorted(ranked[:3])
    action_pattern = re.compile(
        r"\b(?:i will|we will|i'll|we'll|please|need to|action item|follow up)\b", re.I
    )
    return {
        "method": "extractive English heuristics",
        "summary": [sentence for _, sentence in selected],
        "questions": [sentence for sentence in sentences if sentence.endswith("?")][:20],
        "candidate_actions": [
            sentence for sentence in sentences if action_pattern.search(sentence)
        ][:20],
        "keywords": [word for word, _ in counts.most_common(8)],
        "word_count": len(words),
        "intent": infer_intent(text),
    }
