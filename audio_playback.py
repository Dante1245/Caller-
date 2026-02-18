from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import sounddevice as sd


@dataclass
class PlaybackConfig:
    sample_rate: int
    device: int | None = None
    block_size: int = 2048


class PlaybackEngine:
    def __init__(self, config: PlaybackConfig) -> None:
        self.config = config
        self.stream: sd.OutputStream | None = None

    def start(self) -> None:
        if self.stream:
            return
        self.stream = sd.OutputStream(
            samplerate=self.config.sample_rate,
            channels=1,
            blocksize=self.config.block_size,
            device=self.config.device,
        )
        self.stream.start()

    def play(self, audio: np.ndarray) -> None:
        if not self.stream:
            self.start()
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        self.stream.write(audio.astype(np.float32))

    def stop(self) -> None:
        if not self.stream:
            return
        self.stream.stop()
        self.stream.close()
        self.stream = None
