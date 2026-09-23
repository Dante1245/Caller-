from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class TTSResult:
    audio: np.ndarray
    sample_rate: int


class LocalTTS:
    def __init__(
        self, model_name: str | None = None, device: str | None = None, speed: float = 1.0
    ) -> None:
        import torch
        from TTS.api import TTS

        if not 0.7 <= speed <= 1.2:
            raise ValueError("Voice speed must be 0.7–1.2")
        requested_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if requested_device == "cuda" and not torch.cuda.is_available():
            requested_device = "cpu"
        self.device = requested_device
        self.speed = speed
        self.model_name = model_name or "tts_models/multilingual/multi-dataset/xtts_v2"
        self.tts = TTS(model_name=self.model_name).to(self.device)
        self.output_sample_rate = self.tts.synthesizer.output_sample_rate
        self._reference_key = None
        self._conditioning = None

    def synthesize(self, text: str, speaker_wav: str, language: str = "en") -> TTSResult:
        if not text.strip():
            return TTSResult(np.empty(0, dtype=np.float32), self.output_sample_rate)
        reference = Path(speaker_wav).resolve()
        stat = reference.stat()
        key = (str(reference), stat.st_size, stat.st_mtime_ns)
        if self.model_name.endswith("/xtts_v2"):
            # XTTS's documented low-level API lets us reuse conditioning in memory.
            # Never serialize speaker embeddings; switching/changing the file invalidates it.
            model = self.tts.synthesizer.tts_model
            if self._reference_key != key:
                conditioning = model.get_conditioning_latents(audio_path=[str(reference)])
                self._conditioning = conditioning
                self._reference_key = key
            latent, speaker = self._conditioning
            audio = model.inference(text, language, latent, speaker, speed=self.speed)["wav"]
        else:
            audio = self.tts.tts(
                text=text, speaker_wav=str(reference), language=language, speed=self.speed
            )
        audio_np = np.asarray(audio, dtype=np.float32)
        if not np.isfinite(audio_np).all():
            raise RuntimeError("Voice model returned invalid audio samples")
        return TTSResult(audio=audio_np, sample_rate=self.output_sample_rate)
