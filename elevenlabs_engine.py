from __future__ import annotations

import os
from dataclasses import dataclass

from elevenlabs import ElevenLabs, play


@dataclass
class ElevenLabsConfig:
    api_key: str
    voice_id: str
    model_id: str = "eleven_turbo_v2"


class ElevenLabsEngine:
    def __init__(self, config: ElevenLabsConfig) -> None:
        self.config = config
        self.client = ElevenLabs(api_key=config.api_key)

    def synthesize(self, text: str, mood: str) -> None:
        settings = _adaptive_voice_settings(mood)
        audio = self.client.generate(
            text=text,
            voice=self.config.voice_id,
            model_id=self.config.model_id,
            voice_settings=settings,
        )
        play(audio)

    def clone_voice(self, name: str, samples: list[str]):
        voices = getattr(self.client, "voices", None)
        if voices and hasattr(voices, "add"):
            return voices.add(name=name, files=samples)
        raise RuntimeError("ElevenLabs client does not support voice cloning in this version.")


def _adaptive_voice_settings(mood: str) -> dict[str, float]:
    presets = {
        "positive": {"stability": 0.8, "similarity_boost": 0.85},
        "negative": {"stability": 0.65, "similarity_boost": 0.75},
        "neutral": {"stability": 0.72, "similarity_boost": 0.8},
    }
    return presets.get(mood, presets["neutral"])


def get_api_key() -> str:
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        raise RuntimeError("ELEVENLABS_API_KEY is required for ElevenLabs mode.")
    return api_key
