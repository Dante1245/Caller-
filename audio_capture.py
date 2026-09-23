"""Bounded PCM segmentation, independent of microphones and ML dependencies."""

from array import array
from collections import deque
import math
import sys


def pcm_rms(data: bytes) -> float:
    samples = array("h")
    samples.frombytes(data)
    if sys.byteorder != "little":
        samples.byteswap()
    return math.sqrt(sum(float(x) * x for x in samples) / len(samples)) if samples else 0.0


class UtteranceBuffer:
    """Keep pre-roll, skip silence, and cap continuous speech to ~30 seconds.

    min_chunks is a minimum speech length. Silence padding never satisfies it.
    """

    def __init__(self, threshold=500, silence_chunks=8, min_chunks=3, max_chunks=469):
        if threshold <= 0 or min(silence_chunks, min_chunks, max_chunks) <= 0:
            raise ValueError("Threshold and buffer sizes must be positive")
        if min_chunks > max_chunks:
            raise ValueError("Minimum speech length exceeds maximum clip length")
        self.threshold = threshold
        self.silence_chunks = silence_chunks
        self.min_chunks = min_chunks
        self.max_chunks = max_chunks
        self.pre_roll = deque(maxlen=3)
        self.frames = []
        self.silent = 0
        self.speech = 0

    def feed(self, data: bytes) -> bytes | None:
        voiced = pcm_rms(data) >= self.threshold
        if not self.frames and not voiced:
            self.pre_roll.append(data)
            return None
        if not self.frames:
            self.frames.extend(self.pre_roll)
            self.pre_roll.clear()
        self.frames.append(data)
        self.speech += int(voiced)
        self.silent = 0 if voiced else self.silent + 1
        if self.silent < self.silence_chunks and len(self.frames) < self.max_chunks:
            return None
        clip = b"".join(self.frames) if self.speech >= self.min_chunks else None
        self.frames.clear()
        self.silent = self.speech = 0
        return clip
