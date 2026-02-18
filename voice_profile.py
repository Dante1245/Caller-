from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf


PROFILE_PATH = Path("voice_profiles.json")
NORMALIZED_DIR = Path("normalized_speakers")


@dataclass
class VoiceProfile:
    sample_rate: int
    duration_s: float
    rms: float
    peak: float
    gain: float
    silence_threshold: int


def _safe_float(value: float, default: float = 0.0) -> float:
    if np.isnan(value) or np.isinf(value):
        return default
    return float(value)


def analyze_speaker_wav(path: str) -> VoiceProfile:
    audio, sample_rate = sf.read(path, always_2d=False)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if audio.size == 0:
        return VoiceProfile(sample_rate=sample_rate, duration_s=0.0, rms=0.0, peak=0.0, gain=1.0, silence_threshold=500)

    audio = audio.astype(np.float32)
    duration_s = audio.size / sample_rate
    rms = _safe_float(float(np.sqrt(np.mean(audio ** 2))))
    peak = _safe_float(float(np.max(np.abs(audio))))
    target_rms = 0.12
    gain = 1.0 if rms == 0 else min(2.5, max(0.6, target_rms / rms))
    silence_threshold = max(200, int((rms * 32768) * 1.6))
    return VoiceProfile(
        sample_rate=sample_rate,
        duration_s=duration_s,
        rms=rms,
        peak=peak,
        gain=gain,
        silence_threshold=silence_threshold,
    )


def normalize_speaker_wav(path: str, profile: VoiceProfile, name: str) -> str:
    audio, sample_rate = sf.read(path, always_2d=False)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    audio = audio.astype(np.float32) * profile.gain
    audio = np.clip(audio, -1.0, 1.0)
    NORMALIZED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = NORMALIZED_DIR / f"{name}.wav"
    sf.write(output_path, audio, sample_rate)
    return str(output_path)


def load_profiles() -> dict[str, Any]:
    if not PROFILE_PATH.exists():
        return {}
    with PROFILE_PATH.open("r") as handle:
        return json.load(handle)


def save_profile(name: str, profile: VoiceProfile) -> None:
    profiles = load_profiles()
    profiles[name] = asdict(profile)
    with PROFILE_PATH.open("w") as handle:
        json.dump(profiles, handle, indent=2)
