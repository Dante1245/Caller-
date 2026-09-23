from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass
class PerformancePreset:
    silence_chunks: int
    min_buffer_chunks: int
    use_noise_reduction: bool


PRESETS = {
    "balanced": PerformancePreset(silence_chunks=8, min_buffer_chunks=3, use_noise_reduction=True),
    "max": PerformancePreset(silence_chunks=5, min_buffer_chunks=2, use_noise_reduction=False),
    "cpu": PerformancePreset(silence_chunks=10, min_buffer_chunks=4, use_noise_reduction=True),
}


def apply_torch_performance_settings() -> None:
    import torch

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True


def apply_cpu_performance_settings() -> None:
    """Tune PyTorch thread usage for CPU-heavy inference."""
    import torch

    cpu_count = os.cpu_count() or 1
    worker_threads = max(1, min(8, cpu_count - 1))
    torch.set_num_threads(worker_threads)
    torch.set_num_interop_threads(max(1, min(4, worker_threads)))


def select_preset(name: str) -> PerformancePreset:
    return PRESETS.get(name, PRESETS["balanced"])
