"""Measurements of recorded audio, independent of sentiment heuristics."""

import math


def measure_audio(audio):
    import numpy as np

    values = np.asarray(audio, dtype=np.float32)
    if values.size == 0:
        return {"rms_dbfs": -120.0, "peak_dbfs": -120.0, "clipping_percent": 0.0}
    if not np.isfinite(values).all():
        raise ValueError("Audio contains non-finite samples")
    peak = float(np.max(np.abs(values)))
    rms = float(np.sqrt(np.mean(values.astype(np.float64) ** 2)))
    return {
        "rms_dbfs": max(-120.0, 20 * math.log10(max(rms, 1e-6))),
        "peak_dbfs": max(-120.0, 20 * math.log10(max(peak, 1e-6))),
        "clipping_percent": float(np.mean(np.abs(values) >= 0.999) * 100),
    }
