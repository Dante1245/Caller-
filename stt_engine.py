"""Compatible local transcription backends; no remote transcription service."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class STTConfig:
    model: str = "base.en"
    backend: str = "whisper"
    device: str = "cpu"
    compute_type: str = "int8"
    download_root: str | None = None

    def __post_init__(self):
        if self.backend not in ("whisper", "faster-whisper"):
            raise ValueError("Unsupported transcription backend")
        if self.backend == "faster-whisper":
            if self.model.endswith(".pt"):
                raise ValueError(
                    "Faster-whisper needs a model name or converted model directory, not a .pt file"
                )
            if self.device == "cpu" and self.compute_type in ("float16", "int8_float16"):
                raise ValueError("CPU faster-whisper requires int8 or float32 compute")


class FasterWhisperAdapter:
    def __init__(self, config: STTConfig):
        from faster_whisper import WhisperModel

        self.model = WhisperModel(
            config.model,
            device=config.device,
            compute_type=config.compute_type,
            download_root=config.download_root,
        )

    def transcribe(self, audio, **options):
        options.pop("fp16", None)
        options.setdefault("vad_filter", True)
        options.setdefault("condition_on_previous_text", False)
        segments, info = self.model.transcribe(audio, **options)
        # The SDK iterator performs inference lazily. Consume it before returning.
        result = []
        for segment in segments:
            result.append(
                {
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text,
                    "avg_logprob": segment.avg_logprob,
                    "no_speech_prob": segment.no_speech_prob,
                    "words": [
                        {
                            "word": w.word,
                            "start": w.start,
                            "end": w.end,
                            "probability": w.probability,
                        }
                        for w in (segment.words or [])
                    ],
                }
            )
        return {
            "text": "".join(s["text"] for s in result).strip(),
            "language": info.language,
            "language_probability": info.language_probability,
            "segments": result,
        }


def load_stt(config: STTConfig):
    if config.backend == "faster-whisper":
        return FasterWhisperAdapter(config)
    if config.backend != "whisper":
        raise ValueError(f"Unknown transcription backend: {config.backend}")
    import whisper

    return whisper.load_model(
        config.model, device=config.device, download_root=config.download_root
    )


def recognition_warnings(result):
    """Model signals indicate uncertainty, not calibrated correctness probabilities."""
    warnings = []
    segments = result.get("segments", [])
    if any(s.get("avg_logprob", 0) < -1 for s in segments):
        warnings.append("Low model confidence: review the recognized words.")
    if any(s.get("no_speech_prob", 0) > 0.6 for s in segments):
        warnings.append("Possible non-speech: check the microphone and background noise.")
    return warnings
