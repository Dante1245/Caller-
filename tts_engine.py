from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from TTS.api import TTS


@dataclass
class TTSResult:
    audio: np.ndarray
    sample_rate: int


class LocalTTS:
    def __init__(self, model_name: str | None = None, device: str | None = None) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name or "tts_models/multilingual/multi-dataset/xtts_v2"
        self.tts = TTS(model_name=self.model_name).to(self.device)
        self.output_sample_rate = self.tts.synthesizer.output_sample_rate

    def synthesize(
        self,
        text: str,
        speaker_wav: str,
        language: str = "en",
    ) -> TTSResult:
        audio = self.tts.tts(text=text, speaker_wav=speaker_wav, language=language)
        audio_np = np.asarray(audio, dtype=np.float32)
        return TTSResult(audio=audio_np, sample_rate=self.output_sample_rate)
