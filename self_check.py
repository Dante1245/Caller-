from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any

import pyaudio
import sounddevice as sd
import torch


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str


def _check_import(module: str) -> CheckResult:
    try:
        importlib.import_module(module)
        return CheckResult(module, True, "import ok")
    except Exception as exc:
        return CheckResult(module, False, f"import failed: {exc}")


def _check_audio_inputs() -> CheckResult:
    pa = pyaudio.PyAudio()
    count = pa.get_device_count()
    inputs = [
        pa.get_device_info_by_index(i)
        for i in range(count)
        if pa.get_device_info_by_index(i).get("maxInputChannels", 0) > 0
    ]
    pa.terminate()
    ok = len(inputs) > 0
    return CheckResult("pyaudio.input_devices", ok, f"inputs={len(inputs)}")


def _check_audio_outputs() -> CheckResult:
    devices = sd.query_devices()
    outputs = [d for d in devices if d.get("max_output_channels", 0) > 0]
    ok = len(outputs) > 0
    return CheckResult("sounddevice.output_devices", ok, f"outputs={len(outputs)}")


def _check_torch() -> CheckResult:
    cuda = torch.cuda.is_available()
    return CheckResult("torch.cuda", True, f"cuda_available={cuda}")


def run_self_check() -> list[CheckResult]:
    results = [
        _check_import("whisper"),
        _check_import("TTS"),
        _check_import("soundfile"),
        _check_import("vaderSentiment"),
        _check_import("elevenlabs"),
        _check_torch(),
        _check_audio_inputs(),
        _check_audio_outputs(),
    ]
    return results


def render_report(results: list[CheckResult]) -> dict[str, Any]:
    return {
        "summary": {
            "passed": sum(1 for r in results if r.ok),
            "failed": sum(1 for r in results if not r.ok),
        },
        "results": [r.__dict__ for r in results],
    }
