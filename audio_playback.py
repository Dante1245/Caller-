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
        try:
            self.stream.start()
        except Exception:
            self.stream.close()
            self.stream = None
            raise

    def play(self, audio: np.ndarray, stop_event=None, muted=None, gain=1.0) -> None:
        if (stop_event is not None and stop_event.is_set()) or (
            muted is not None and muted.is_set()
        ):
            return
        if audio.size == 0:
            return
        if not np.isfinite(audio).all():
            raise ValueError("Playback audio contains non-finite values")
        if not self.stream:
            self.start()
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        audio = np.clip(audio * gain, -1.0, 1.0).astype(np.float32)
        for start in range(0, len(audio), self.config.block_size):
            if (stop_event is not None and stop_event.is_set()) or (
                muted is not None and muted.is_set()
            ):
                self.stream.abort()
                self.stream.start()
                break
            self.stream.write(audio[start : start + self.config.block_size])

    def stop(self) -> None:
        if not self.stream:
            return
        stream, self.stream = self.stream, None
        try:
            stream.stop()
        finally:
            stream.close()
