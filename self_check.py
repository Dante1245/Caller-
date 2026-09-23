"""Diagnostics work even when optional runtime dependencies are missing."""

from __future__ import annotations
import importlib
import shutil
import sys
from dataclasses import asdict, dataclass


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str


def _check(name, action):
    try:
        return CheckResult(name, True, str(action()))
    except Exception as exc:
        return CheckResult(name, False, f"{type(exc).__name__}: {exc}")


def list_devices():
    import sounddevice as sd

    devices = sd.query_devices()
    inputs = [
        {"index": i, "name": d["name"]}
        for i, d in enumerate(devices)
        if d["max_input_channels"] > 0
    ]
    outputs = [
        {"index": i, "name": d["name"]}
        for i, d in enumerate(devices)
        if d["max_output_channels"] > 0
    ]
    return {"inputs": inputs, "outputs": outputs}


def _audio(kind):
    devices = list_devices()[kind]
    if not devices:
        raise RuntimeError(f"No {kind} devices detected")
    return devices


def run_self_check(engine="local", run_mode="live", stt_backend="whisper"):
    modules = [
        "numpy",
        "torch",
        "faster_whisper" if stt_backend == "faster-whisper" else "whisper",
        "noisereduce",
        "vaderSentiment",
        "soundfile",
        "sounddevice",
    ]
    if run_mode == "live":
        modules.append("TTS.api" if engine == "local" else "elevenlabs")
    results = [CheckResult("python", (3, 10) <= sys.version_info[:2] < (3, 14), sys.version)]
    results += [_check(m, lambda m=m: importlib.import_module(m).__name__) for m in modules]
    results.append(
        CheckResult(
            "ffmpeg", shutil.which("ffmpeg") is not None, shutil.which("ffmpeg") or "Install ffmpeg"
        )
    )
    results.append(_check("audio.inputs", lambda: _audio("inputs")))
    if run_mode == "live":
        results.append(_check("audio.outputs", lambda: _audio("outputs")))
        if engine == "elevenlabs":
            import os

            present = bool(os.getenv("ELEVENLABS_API_KEY", "").strip())
            results.append(
                CheckResult(
                    "elevenlabs.api_key", present, "Configured" if present else "Not configured"
                )
            )
    return results


def render_report(results):
    return {
        "summary": {"passed": sum(r.ok for r in results), "failed": sum(not r.ok for r in results)},
        "results": [asdict(r) for r in results],
    }
