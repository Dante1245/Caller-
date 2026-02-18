from __future__ import annotations


def infer_intent(text: str) -> str:
    """Heuristic intent tagging for live-call assistant analytics."""
    lower = text.lower()
    if any(token in lower for token in ["help", "support", "issue", "problem", "error"]):
        return "support"
    if any(token in lower for token in ["buy", "price", "cost", "invoice", "payment"]):
        return "sales"
    if any(token in lower for token in ["schedule", "meeting", "tomorrow", "later"]):
        return "planning"
    if any(token in lower for token in ["hello", "hi", "thanks", "thank you"]):
        return "social"
    return "general"


def compute_speaking_rate_wpm(text: str, duration_s: float | None) -> float:
    words = len([w for w in text.split() if w.strip()])
    if words == 0 or not duration_s or duration_s <= 0:
        return 0.0
    return words / (duration_s / 60.0)


def suggest_response_style(mood: str, intent: str, wpm: float) -> str:
    if mood == "negative":
        return "calm and reassuring"
    if intent == "support":
        return "clear and step-by-step"
    if intent == "sales":
        return "confident and concise"
    if wpm > 170:
        return "slightly slower and articulate"
    return "natural and friendly"
