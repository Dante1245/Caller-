from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any
from storage import read_object, write_object

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
        raise ValueError("Reference audio is empty")

    if not np.isfinite(audio).all() or np.max(np.abs(audio)) < 0.001:
        raise ValueError("Reference audio is silent or contains invalid samples")
    audio = audio.astype(np.float32)
    duration_s = audio.size / sample_rate
    rms = _safe_float(float(np.sqrt(np.mean(audio**2))))
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
    import re

    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", name):
        raise ValueError("Invalid profile name")
    NORMALIZED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = NORMALIZED_DIR / f"{name}.wav"
    sf.write(output_path, audio, sample_rate)
    return str(output_path)


def load_profiles() -> dict[str, Any]:
    return read_object(PROFILE_PATH)


def save_profile(name: str, profile: VoiceProfile) -> None:
    profiles = load_profiles()
    profiles[name] = asdict(profile)
    write_object(PROFILE_PATH, profiles)
