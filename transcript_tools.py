"""Local recording transcription and portable, explicit transcript exports."""

from __future__ import annotations

import json
import math
from pathlib import Path

from conversation_ai import conversation_insights
from stt_engine import STTConfig, load_stt, recognition_warnings


def subtitle_time(seconds):
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError("Subtitle time must be finite and non-negative")
    ms = round(seconds * 1000)
    seconds, millis = divmod(ms, 1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02},{millis:03}"


def export_transcription(result, output_prefix):
    prefix = Path(output_prefix)
    # Append, do not replace dotted names such as meeting.2026.
    paths = [Path(str(prefix) + ext) for ext in (".json", ".txt", ".srt")]
    if any(path.exists() for path in paths):
        raise FileExistsError("Export already exists; choose a different output name")
    subtitles = []
    for segment in result.get("segments", []):
        text = segment.get("text", "").strip()
        if not text:
            continue
        start, end = float(segment["start"]), float(segment["end"])
        if end < start:
            raise ValueError("Subtitle end precedes its start")
        subtitles.append(
            f"{len(subtitles) + 1}\n{subtitle_time(start)} --> {subtitle_time(end)}\n{text}\n"
        )
    contents = [
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False),
        result.get("text", "").strip() + "\n",
        "\n".join(subtitles),
    ]
    prefix.parent.mkdir(parents=True, exist_ok=True)
    created = []
    try:
        for path, content in zip(paths, contents):
            with path.open("x", encoding="utf-8") as handle:
                created.append(path)
                handle.write(content)
    except Exception:
        for path in created:
            path.unlink(missing_ok=True)
        raise
    return [str(path) for path in paths]


def transcribe_file(
    path, config: STTConfig, language="en", task="transcribe", initial_prompt=None, stop_event=None
):
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"Audio file does not exist: {path}")
    if task not in ("transcribe", "translate"):
        raise ValueError("Task must be transcribe or translate")
    if task == "translate" and config.model.endswith(".en"):
        raise ValueError("English translation requires a multilingual model such as base")
    if stop_event and stop_event.is_set():
        return None
    model = load_stt(config)
    if stop_event and stop_event.is_set():
        return None
    result = model.transcribe(
        str(path),
        language=language,
        task=task,
        initial_prompt=initial_prompt,
        word_timestamps=True,
        fp16=config.device == "cuda",
        temperature=0.0,
        beam_size=5,
        condition_on_previous_text=False,
    )
    if stop_event and stop_event.is_set():
        return None
    result["warnings"] = recognition_warnings(result)
    result["insights"] = conversation_insights(result.get("text", ""))
    result["source_name"] = path.name
    result["backend"] = config.backend
    result["task"] = task
    return result
