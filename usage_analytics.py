from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


REPORT_PATH = Path("usage_report.json")


@dataclass
class UsageReport:
    sessions: int = 0
    utterances: int = 0
    avg_latency_ms: float = 0.0
    avg_word_count: float = 0.0
    last_profile: str = "default"
    recommendations: list[str] | None = None


def load_report() -> dict[str, Any]:
    if not REPORT_PATH.exists():
        return {}
    with REPORT_PATH.open("r") as handle:
        return json.load(handle)


def save_report(report: UsageReport) -> None:
    with REPORT_PATH.open("w") as handle:
        json.dump(asdict(report), handle, indent=2)


def update_report(
    report: UsageReport,
    latency_ms: float,
    word_count: int,
    profile_name: str,
) -> UsageReport:
    report.sessions = max(report.sessions, 1)
    report.utterances += 1
    report.avg_latency_ms = _running_average(
        report.avg_latency_ms, latency_ms, report.utterances
    )
    report.avg_word_count = _running_average(
        report.avg_word_count, float(word_count), report.utterances
    )
    report.last_profile = profile_name
    report.recommendations = generate_recommendations(report)
    return report


def start_session(report: UsageReport, profile_name: str) -> UsageReport:
    report.sessions += 1
    report.last_profile = profile_name
    return report


def generate_recommendations(report: UsageReport) -> list[str]:
    recommendations: list[str] = []
    if report.avg_latency_ms > 900:
        recommendations.append(
            "Latency is high. Try --model base, disable noise reduction, or use a GPU."
        )
    if report.avg_word_count < 3:
        recommendations.append(
            "Short utterances detected. Consider lowering --min-buffer-chunks."
        )
    if report.avg_word_count > 18:
        recommendations.append(
            "Long utterances detected. Consider raising --min-buffer-chunks for stability."
        )
    if not recommendations:
        recommendations.append("Settings look healthy. Keep monitoring for drift.")
    return recommendations


def _running_average(previous: float, value: float, count: int) -> float:
    if count <= 1:
        return value
    return previous + (value - previous) / float(count)
