from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class PerformancePreset:
    silence_chunks: int
    min_buffer_chunks: int
    use_noise_reduction: bool


PRESETS = {
    "balanced": PerformancePreset(silence_chunks=8, min_buffer_chunks=40, use_noise_reduction=True),
    "max": PerformancePreset(silence_chunks=5, min_buffer_chunks=28, use_noise_reduction=False),
}


def apply_torch_performance_settings() -> None:
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True


def select_preset(name: str) -> PerformancePreset:
    return PRESETS.get(name, PRESETS["balanced"])
