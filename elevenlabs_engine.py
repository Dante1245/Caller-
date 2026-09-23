from __future__ import annotations

import os
from contextlib import ExitStack
from dataclasses import dataclass


@dataclass
class ElevenLabsConfig:
    api_key: str
    voice_id: str
    model_id: str = "eleven_flash_v2_5"
    device: int | None = None
    block_size: int = 2048
    speed: float = 1.0


class ElevenLabsEngine:
    def __init__(self, config: ElevenLabsConfig) -> None:
        from elevenlabs.client import ElevenLabs
        from audio_playback import PlaybackConfig, PlaybackEngine

        self.config = config
        self.client = ElevenLabs(api_key=config.api_key, timeout=30)
        self.playback = PlaybackEngine(PlaybackConfig(24000, config.device, config.block_size))

    def synthesize(self, text: str, mood: str, stop_event=None, muted=None) -> None:
        if (stop_event is not None and stop_event.is_set()) or (
            muted is not None and muted.is_set()
        ):
            return
        import numpy as np
        from elevenlabs import VoiceSettings

        if not 0.7 <= self.config.speed <= 1.2:
            raise ValueError("ElevenLabs speed must be between 0.7 and 1.2")
        chunks = self.client.text_to_speech.convert(
            text=text,
            voice_id=self.config.voice_id,
            model_id=self.config.model_id,
            output_format="pcm_24000",
            voice_settings=VoiceSettings(**_adaptive_voice_settings(mood), speed=self.config.speed),
        )
        # Network chunks are not guaranteed to end on a 16-bit sample boundary.
        pending = b""
        try:
            for chunk in chunks:
                if (stop_event is not None and stop_event.is_set()) or (
                    muted is not None and muted.is_set()
                ):
                    return
                pending += chunk
                end = len(pending) // 2 * 2
                if end:
                    self.playback.play(
                        np.frombuffer(pending[:end], dtype="<i2").astype(np.float32) / 32768,
                        stop_event=stop_event,
                        muted=muted,
                    )
                    pending = pending[end:]
            if pending:
                raise RuntimeError("ElevenLabs returned incomplete PCM audio")
        finally:
            close = getattr(chunks, "close", None)
            if close:
                close()

    def clone_voice(self, name: str, samples: list[str]):
        with ExitStack() as stack:
            files = [stack.enter_context(open(path, "rb")) for path in samples]
            return self.client.voices.ivc.create(name=name, files=files)

    def close(self):
        self.playback.stop()
        close = getattr(self.client, "close", None)
        if close:
            close()


def _adaptive_voice_settings(mood: str) -> dict[str, float]:
    presets = {
        "positive": {"stability": 0.8, "similarity_boost": 0.85},
        "negative": {"stability": 0.65, "similarity_boost": 0.75},
        "neutral": {"stability": 0.72, "similarity_boost": 0.8},
    }
    return presets.get(mood, presets["neutral"])


def get_api_key() -> str:
    api_key = os.getenv("ELEVENLABS_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("ELEVENLABS_API_KEY is required for ElevenLabs mode.")
    return api_key


def list_cloud_resources(kind="voices"):
    from elevenlabs.client import ElevenLabs

    client = ElevenLabs(api_key=get_api_key(), timeout=30)
    try:
        if kind == "models":
            return [
                {"id": model.model_id, "name": model.name}
                for model in client.models.list()
                if model.can_do_text_to_speech
            ]
        result, token = [], None
        while True:
            page = client.voices.search(page_size=100, next_page_token=token)
            result.extend({"id": voice.voice_id, "name": voice.name} for voice in page.voices)
            if not page.has_more or not page.next_page_token:
                return result
            if page.next_page_token == token:
                raise RuntimeError("Voice listing returned a repeated page token")
            token = page.next_page_token
    finally:
        close = getattr(client, "close", None)
        if close:
            close()
