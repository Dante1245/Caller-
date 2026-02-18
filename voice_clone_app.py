#!/usr/bin/env python3
import argparse
import json
import os
import queue
import threading
import time
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyaudio
import torch
import whisper
import noisereduce as nr
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from audio_playback import PlaybackConfig, PlaybackEngine
from conversation_ai import compute_speaking_rate_wpm, infer_intent, suggest_response_style
from elevenlabs_engine import ElevenLabsConfig, ElevenLabsEngine, get_api_key
from tts_engine import LocalTTS
from voice_profile import analyze_speaker_wav, normalize_speaker_wav, save_profile, VoiceProfile
from usage_analytics import (
    UsageReport,
    load_report,
    save_report,
    start_session,
    update_report,
)
from performance_tuning import (
    apply_cpu_performance_settings,
    apply_torch_performance_settings,
    select_preset,
)
from quality_ai import assess_call_quality
from self_check import render_report, run_self_check
"""
Ultra-low-latency, sentiment-aware voice cloning pipeline.

Updates in this version:
- Mood tracking via VADER sentiment for richer analytics per utterance.
- Per-word timestamps from Whisper for detailed call analytics and responsive UX.
- GPU-aware Whisper loading and tuned silence detection for faster turn-taking.
- Local ultra-fast voice cloning via Coqui XTTS v2 (no external API).
"""

sentiment_analyzer = SentimentIntensityAnalyzer()

# Audio constants for predictable, low-latency behavior
SAMPLE_FORMAT = pyaudio.paInt16
CHANNELS = 1
SAMPLE_RATE = 16000
CHUNK = 1024
SILENCE_CHUNKS = 8  # shorter silence window for snappier responses
MIN_BUFFER_CHUNKS = 40  # lower minimum to ship faster to Whisper
TRANSCRIPTS_PATH = Path("transcripts.txt")


@dataclass
class AppConfig:
    speaker_wav: str
    tts_model: str | None
    model_name: str
    language: str
    use_noise_reduction: bool
    silence_chunks: int
    min_buffer_chunks: int
    silence_threshold: int
    device_index: int | None
    playback_gain: float
    profile_name: str
    auto_upgrade: bool
    performance_mode: str
    playback_device: int | None
    playback_block_size: int
    engine: str
    elevenlabs_voice_id: str | None
    force_cpu: bool
    auto_language: bool
    run_mode: str


class StatusBus:
    def __init__(self) -> None:
        self.queue: queue.Queue[str] = queue.Queue()

    def emit(self, message: str) -> None:
        print(message)
        self.queue.put(message)


class AdaptiveGain:
    def __init__(self, base_gain: float) -> None:
        self.gain = base_gain

    def update(self, observed_peak: float) -> float:
        if observed_peak <= 0:
            return self.gain
        target_peak = 0.85
        adjustment = target_peak / observed_peak
        self.gain = min(2.0, max(0.5, self.gain * adjustment))
        return self.gain


def trim_silence(audio: np.ndarray, threshold: float = 0.01) -> np.ndarray:
    if audio.size == 0:
        return audio
    mask = np.abs(audio) > threshold
    if not mask.any():
        return audio
    start = int(np.argmax(mask))
    end = int(len(mask) - np.argmax(mask[::-1]))
    return audio[start:end]

def record_sample(duration=5, filename=None, device_index=None):
    """Capture a short, high-fidelity sample for cloning."""
    if not filename:
        filename = f"sample_{time.time()}.wav"

    pa = pyaudio.PyAudio()
    print("Recording sample...")
    stream = pa.open(
        format=SAMPLE_FORMAT,
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        frames_per_buffer=CHUNK,
        input=True,
        input_device_index=device_index,
    )
    frames = []
    for _ in range(0, int(SAMPLE_RATE / CHUNK * duration)):
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)
    stream.stop_stream()
    stream.close()
    sample_width = pa.get_sample_size(SAMPLE_FORMAT)
    pa.terminate()

    wf = wave.open(filename, "wb")
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(sample_width)
    wf.setframerate(SAMPLE_RATE)
    wf.writeframes(b"".join(frames))
    wf.close()
    print("Sample recorded.")
    return filename

def is_silent(data, threshold=500):
    """Check if audio data is silent based on RMS."""
    audio_data = np.frombuffer(data, dtype=np.int16)
    rms = np.sqrt(np.mean(audio_data ** 2))
    return rms < threshold


def get_device_and_model(model_name=None, force_cpu=False):
    """Pick the fastest Whisper model based on available hardware."""
    device = "cpu" if force_cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    preferred_model = model_name or ("small.en" if device == "cuda" else "base.en")
    print(f"Loading Whisper model '{preferred_model}' on {device} for optimized quality/latency...")
    model = whisper.load_model(preferred_model, device=device)
    return model


def resolve_whisper_model(model_name=None, force_cpu=False, quality_mode="balanced"):
    """Select Whisper model with CPU/GPU-aware quality defaults."""
    if model_name:
        return model_name
    if force_cpu:
        return "small.en" if quality_mode == "high" else "base.en"
    if torch.cuda.is_available():
        return "medium.en" if quality_mode == "high" else "small.en"
    return "small.en" if quality_mode == "high" else "base.en"


def analyze_mood(text):
    """Return mood label and compound score using VADER."""
    scores = sentiment_analyzer.polarity_scores(text)
    compound = scores.get("compound", 0)
    if compound >= 0.3:
        mood = "positive"
    elif compound <= -0.3:
        mood = "negative"
    else:
        mood = "neutral"
    return mood, compound


def mood_preamble(mood):
    """Lightweight mood marker for analytics (no TTS API required)."""
    return {
        "positive": "[cheerful tone]",
        "negative": "[calm tone]",
        "neutral": "[steady tone]",
    }.get(mood, "[steady tone]")


def extract_word_timestamps(result):
    """Flatten Whisper word timestamps for analytics and per-word monitoring."""
    word_timings = []
    for segment in result.get("segments", []):
        for word in segment.get("words", []) or []:
            word_timings.append(
                {
                    "word": word.get("word", "").strip(),
                    "start": word.get("start"),
                    "end": word.get("end"),
                }
            )
    return word_timings


def calibrate_silence_threshold(device_index=None, seconds=1.5):
    """Calibrate silence threshold based on ambient noise."""
    pa = pyaudio.PyAudio()
    stream = pa.open(
        format=SAMPLE_FORMAT,
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        frames_per_buffer=CHUNK,
        input=True,
        input_device_index=device_index,
    )
    frames = []
    for _ in range(0, int(SAMPLE_RATE / CHUNK * seconds)):
        frames.append(stream.read(CHUNK, exception_on_overflow=False))
    stream.stop_stream()
    stream.close()
    pa.terminate()
    audio_data = np.frombuffer(b"".join(frames), dtype=np.int16)
    if audio_data.size == 0:
        return 500
    rms = float(np.sqrt(np.mean(audio_data ** 2)))
    return max(200, int(rms * 1.8))

def real_time_record(audio_queue, stop_event, config: AppConfig, status_bus: StatusBus):
    """Continuously record audio, detect silence, and enqueue denoised clips."""
    pa = pyaudio.PyAudio()
    stream = pa.open(
        format=SAMPLE_FORMAT,
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        frames_per_buffer=CHUNK,
        input=True,
        input_device_index=config.device_index,
    )
    status_bus.emit("Listening continuously... Speak and pause to process.")
    buffer = []
    silence_count = 0
    while not stop_event.is_set():
        data = stream.read(CHUNK, exception_on_overflow=False)
        buffer.append(data)
        if is_silent(data, threshold=config.silence_threshold):
            silence_count += 1
        else:
            silence_count = 0
        if silence_count > config.silence_chunks and len(buffer) > config.min_buffer_chunks:
            audio_data = b"".join(buffer)
            audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
            if config.use_noise_reduction:
                reduced_noise = nr.reduce_noise(y=audio_np, sr=SAMPLE_RATE)
            else:
                reduced_noise = audio_np
            filename = f"temp_{time.time()}.wav"
            wf = wave.open(filename, "wb")
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes((reduced_noise.astype(np.int16)).tobytes())
            wf.close()
            audio_queue.put(filename)
            buffer = []
            silence_count = 0
    stream.stop_stream()
    stream.close()
    pa.terminate()


def log_transcript(text, mood, compound, word_timings, intent="general", speaking_rate_wpm=0.0, detected_language="en"):
    timestamp = time.ctime()
    with TRANSCRIPTS_PATH.open("a") as f:
        f.write(
            f"{timestamp} | mood={mood} ({compound:.3f}) | "
            f"intent={intent} | wpm={speaking_rate_wpm:.1f} | lang={detected_language} | text={text}\n"
        )
        if word_timings:
            f.write(
                "  words="
                + ", ".join(
                    [
                        f"{w['word']}@{(w['start'] or 0):.2f}-{(w['end'] or 0):.2f}"
                        for w in word_timings
                    ]
                )
                + "\n"
            )


def process_audio(audio_queue, stop_event, config: AppConfig, status_bus: StatusBus):
    model = get_device_and_model(config.model_name, force_cpu=config.force_cpu)
    tts_engine = None
    gain_controller = None
    playback = None
    elevenlabs_engine = None
    if config.engine == "local":
        tts_engine = LocalTTS(model_name=config.tts_model, device="cpu" if config.force_cpu else None)
        gain_controller = AdaptiveGain(config.playback_gain)
        playback = PlaybackEngine(
            PlaybackConfig(
                sample_rate=tts_engine.output_sample_rate,
                device=config.playback_device,
                block_size=config.playback_block_size,
            )
        )
    else:
        elevenlabs_engine = ElevenLabsEngine(
            ElevenLabsConfig(api_key=get_api_key(), voice_id=config.elevenlabs_voice_id)
        )
    report_data = load_report()
    report = UsageReport(**report_data) if report_data else UsageReport()
    report = start_session(report, config.profile_name)
    while not stop_event.is_set():
        audio_file = None
        try:
            start_time = time.perf_counter()
            audio_file = audio_queue.get(timeout=1)
            transcription_language = None if config.auto_language else config.language
            result = model.transcribe(
                audio_file,
                language=transcription_language,
                word_timestamps=True,
                fp16=torch.cuda.is_available() and not config.force_cpu,
                temperature=0.0,
                best_of=1,
                beam_size=1,
            )
            text = result.get("text", "").strip()
            if not text:
                continue
            detected_language = result.get("language", config.language)
            mood, compound = analyze_mood(text)
            word_timings = extract_word_timestamps(result)
            intent = infer_intent(text)
            utterance_duration = None
            if word_timings:
                first_start = word_timings[0].get("start")
                last_end = word_timings[-1].get("end")
                if first_start is not None and last_end is not None and last_end > first_start:
                    utterance_duration = float(last_end - first_start)
            speaking_rate_wpm = compute_speaking_rate_wpm(text, utterance_duration)
            style_hint = suggest_response_style(mood, intent, speaking_rate_wpm)
            status_bus.emit(f"You said: {text}")
            status_bus.emit(f"Mood detected: {mood} (compound={compound:.3f})")
            status_bus.emit(
                f"Detected intent={intent}, language={detected_language}, "
                f"pace={speaking_rate_wpm:.1f} wpm, style={style_hint}"
            )
            if word_timings:
                status_bus.emit("Word timings:")
                for word in word_timings:
                    status_bus.emit(
                        f"  {word['word']} :: {word['start']:.2f}s -> {word['end']:.2f}s"
                    )
            if config.run_mode == "test":
                status_bus.emit("TEST mode: transcription and analytics complete; voice playback skipped.")
            elif config.engine == "local":
                text_to_cloned_voice(text, config, tts_engine, gain_controller, playback, mood)
            else:
                text_to_elevenlabs_voice(text, elevenlabs_engine, mood)
            latency_ms = (time.perf_counter() - start_time) * 1000
            status_bus.emit(f"End-to-end latency: {latency_ms:.1f} ms")
            quality = assess_call_quality(
                latency_ms=latency_ms,
                speaking_rate_wpm=speaking_rate_wpm,
                mood=mood,
                has_word_timestamps=bool(word_timings),
            )
            status_bus.emit(
                f"Quality score: {quality.score:.1f}/100 ({quality.label}) | tip: {quality.recommendation}"
            )
            report = update_report(report, latency_ms, len(text.split()), config.profile_name)
            save_report(report)
            if config.auto_upgrade and report.recommendations:
                status_bus.emit("Auto-upgrade hints:")
                for recommendation in report.recommendations:
                    status_bus.emit(f"  - {recommendation}")
            log_transcript(
                text,
                mood,
                compound,
                word_timings,
                intent=intent,
                speaking_rate_wpm=speaking_rate_wpm,
                detected_language=detected_language,
            )
        except queue.Empty:
            continue
        finally:
            if audio_file and os.path.exists(audio_file):
                os.remove(audio_file)
    if playback:
        playback.stop()


def text_to_cloned_voice(
    text,
    config: AppConfig,
    tts_engine: LocalTTS,
    gain_controller: AdaptiveGain,
    playback: PlaybackEngine,
    mood="neutral",
):
    """Synthesize and play cloned voice using local XTTS."""
    annotated_text = f"{mood_preamble(mood)} {text}"
    result = tts_engine.synthesize(
        text=annotated_text,
        speaker_wav=config.speaker_wav,
        language=config.language,
    )
    audio = result.audio * config.playback_gain
    observed_peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    config.playback_gain = gain_controller.update(observed_peak)
    audio = np.clip(audio, -1.0, 1.0)
    audio = trim_silence(audio)
    print(f"Playing cloned voice (mood={mood})")
    playback.play(audio)


def text_to_elevenlabs_voice(text: str, engine: ElevenLabsEngine, mood: str) -> None:
    annotated_text = f"{mood_preamble(mood)} {text}"
    engine.synthesize(annotated_text, mood)


def list_input_devices():
    pa = pyaudio.PyAudio()
    devices = []
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        if info.get("maxInputChannels", 0) > 0:
            devices.append((i, info.get("name", f"Device {i}")))
    pa.terminate()
    return devices


def resolve_device_index(requested):
    if requested is None:
        return None
    devices = dict(list_input_devices())
    if requested in devices:
        return requested
    raise ValueError(f"Invalid device index {requested}. Available: {list(devices)}")


def start_threads(config: AppConfig, status_bus: StatusBus):
    audio_queue = queue.Queue()
    stop_event = threading.Event()
    record_thread = threading.Thread(
        target=real_time_record, args=(audio_queue, stop_event, config, status_bus)
    )
    process_thread = threading.Thread(
        target=process_audio, args=(audio_queue, stop_event, config, status_bus)
    )
    record_thread.start()
    process_thread.start()
    return stop_event, record_thread, process_thread


def run_control_ui(config: AppConfig):
    import tkinter as tk
    from tkinter import filedialog, messagebox

    status_bus = StatusBus()
    app_state = {"running": False, "stop_event": None, "threads": None}
    speaker_var = tk.StringVar(value=config.speaker_wav or "No file loaded")
    analysis_var = tk.StringVar(value="Voice analysis: not loaded")
    mode_var = tk.StringVar(value=config.run_mode)

    def apply_voice_file(path: str):
        if not path or not os.path.exists(path):
            messagebox.showerror("Invalid file", "Please choose a valid voice audio file.")
            return
        if config.engine == "local":
            profile = analyze_speaker_wav(path)
            normalized_path = normalize_speaker_wav(path, profile, config.profile_name)
            save_profile(config.profile_name, profile)
            config.speaker_wav = normalized_path
            config.playback_gain = profile.gain
            config.silence_threshold = max(config.silence_threshold, profile.silence_threshold)
            speaker_var.set(normalized_path)
            analysis_var.set(
                "Voice analysis: "
                f"gain={profile.gain:.2f}, silence_threshold={profile.silence_threshold}, "
                f"duration={profile.duration_s:.1f}s"
            )
            status_bus.emit(f"Loaded and analyzed voice file: {normalized_path}")
        else:
            config.speaker_wav = path
            speaker_var.set(path)
            analysis_var.set("Voice analysis: loaded for ElevenLabs clone flow")
            status_bus.emit(f"Loaded voice file for ElevenLabs mode: {path}")

    def browse_voice_file():
        selected = filedialog.askopenfilename(
            title="Select voice sample",
            filetypes=[("Audio files", "*.wav *.mp3 *.m4a *.flac"), ("All files", "*.*")],
        )
        if selected:
            apply_voice_file(selected)

    def start_app():
        if app_state["running"]:
            return
        config.run_mode = mode_var.get()
        if config.engine == "local" and (not config.speaker_wav or not os.path.exists(config.speaker_wav)):
            messagebox.showwarning(
                "Voice file required",
                "Load a voice file before starting the local cloning pipeline.",
            )
            return
        status_bus.emit(f"Starting {config.run_mode.upper()} pipeline...")
        stop_event, record_thread, process_thread = start_threads(config, status_bus)
        app_state.update(
            {
                "running": True,
                "stop_event": stop_event,
                "threads": (record_thread, process_thread),
            }
        )

    def stop_app():
        if not app_state["running"]:
            return
        status_bus.emit("Stopping pipeline...")
        app_state["stop_event"].set()
        for thread in app_state["threads"]:
            thread.join()
        app_state.update({"running": False, "stop_event": None, "threads": None})
        status_bus.emit("Stopped.")

    def poll_status():
        while not status_bus.queue.empty():
            message = status_bus.queue.get_nowait()
            log.insert(tk.END, message + "\n")
            log.see(tk.END)
        root.after(200, poll_status)

    root = tk.Tk()
    root.title("Live Voice Clone Control")
    root.geometry("560x380")

    header = tk.Label(root, text="Live Voice Clone Control", font=("Arial", 14, "bold"))
    header.pack(pady=6)

    status_frame = tk.Frame(root)
    status_frame.pack(fill=tk.X, padx=10)
    tk.Label(status_frame, text="Speaker WAV:").pack(anchor="w")
    tk.Label(status_frame, textvariable=speaker_var, wraplength=520, justify="left").pack(anchor="w")
    tk.Label(status_frame, text=f"TTS model: {config.tts_model}").pack(anchor="w")
    tk.Label(status_frame, text=f"Whisper model: {config.model_name}").pack(anchor="w")
    tk.Label(status_frame, text=f"Profile: {config.profile_name}").pack(anchor="w")
    tk.Label(status_frame, text=f"Performance: {config.performance_mode}").pack(anchor="w")
    tk.Label(status_frame, text=f"Engine: {config.engine}").pack(anchor="w")
    tk.Label(
        status_frame,
        text=f"Auto-upgrade hints: {'On' if config.auto_upgrade else 'Off'}",
    ).pack(anchor="w")
    tk.Label(
        status_frame,
        text=f"Noise reduction: {'On' if config.use_noise_reduction else 'Off'}",
    ).pack(anchor="w")
    tk.Label(status_frame, textvariable=analysis_var, wraplength=520, justify="left").pack(anchor="w")

    controls = tk.Frame(root)
    controls.pack(pady=8)
    mode_controls = tk.Frame(root)
    mode_controls.pack(pady=2)
    tk.Label(mode_controls, text="Run mode:").pack(side=tk.LEFT, padx=4)
    tk.Radiobutton(mode_controls, text="Live", variable=mode_var, value="live").pack(
        side=tk.LEFT, padx=4
    )
    tk.Radiobutton(mode_controls, text="Test", variable=mode_var, value="test").pack(
        side=tk.LEFT, padx=4
    )
    tk.Button(controls, text="Load Voice File", width=14, command=browse_voice_file).pack(
        side=tk.LEFT, padx=6
    )
    tk.Button(controls, text="Start", width=12, command=start_app).pack(
        side=tk.LEFT, padx=6
    )
    tk.Button(controls, text="Stop", width=12, command=stop_app).pack(
        side=tk.LEFT, padx=6
    )

    log = tk.Text(root, height=12)
    log.pack(fill=tk.BOTH, expand=True, padx=10, pady=8)

    root.after(200, poll_status)
    def on_close():
        stop_app()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()


def parse_args():
    parser = argparse.ArgumentParser(description="Real-time voice cloning for live calls.")
    parser.add_argument(
        "--engine",
        choices=["local", "elevenlabs"],
        default="local",
        help="Choose local XTTS or ElevenLabs cloning.",
    )
    parser.add_argument("--speaker-wav", help="Path to speaker WAV for cloning.")
    parser.add_argument(
        "--record-speaker",
        action="store_true",
        help="Record a short speaker sample at startup.",
    )
    parser.add_argument("--tts-model", default=None, help="Coqui TTS model name.")
    parser.add_argument("--model", default=None, help="Whisper model name (default: auto).")
    parser.add_argument(
        "--quality-mode",
        choices=["balanced", "high"],
        default="balanced",
        help="Quality profile for model auto-selection (default: balanced).",
    )
    parser.add_argument("--language", default="en", help="Language for XTTS (default: en).")
    parser.add_argument(
        "--auto-language",
        action="store_true",
        help="Auto-detect spoken language during transcription.",
    )
    parser.add_argument("--elevenlabs-voice-id", default=None, help="ElevenLabs voice ID.")
    parser.add_argument("--elevenlabs-clone-name", default="LocalClone", help="Name for ElevenLabs cloned voice.")
    parser.add_argument("--no-noise-reduction", action="store_true")
    parser.add_argument("--silence-chunks", type=int, default=None)
    parser.add_argument("--min-buffer-chunks", type=int, default=None)
    parser.add_argument("--device-index", type=int, default=None)
    parser.add_argument("--profile-name", default="default", help="Profile name for saved tuning.")
    parser.add_argument(
        "--auto-upgrade",
        action="store_true",
        help="Show self-improvement hints based on live usage metrics.",
    )
    parser.add_argument(
        "--performance-mode",
        choices=["balanced", "max", "cpu"],
        default="balanced",
        help="Performance preset: balanced, max, or cpu (default: balanced).",
    )
    parser.add_argument("--force-cpu", action="store_true", help="Run STT/TTS on CPU even when CUDA is available.")
    parser.add_argument("--playback-device", type=int, default=None)
    parser.add_argument("--playback-block-size", type=int, default=2048)
    parser.add_argument("--self-check", action="store_true", help="Run diagnostics and exit.")
    parser.add_argument(
        "--run-mode",
        choices=["live", "test"],
        default="live",
        help="Run mode: live plays cloned output, test runs full analytics without playback.",
    )
    parser.add_argument("--ui", action="store_true", help="Launch minimal control UI.")
    return parser.parse_args()

def main():
    print("Powerful Real-Time Speech to Cloned Voice App")
    print("Whisper STT + Local XTTS or ElevenLabs cloned voice with adaptive mood and per-word tracking")
    print("GPU-aware, noise-reduced, and tuned for the lowest possible call latency")
    args = parse_args()
    if args.self_check:
        report = render_report(run_self_check())
        print(json.dumps(report, indent=2))
        return
    apply_torch_performance_settings()
    if args.performance_mode == "cpu" or args.force_cpu:
        apply_cpu_performance_settings()
    preset = select_preset(args.performance_mode)
    if args.performance_mode == "max":
        print("Performance mode: MAX (noise reduction disabled, lower buffers).")
    elif args.performance_mode == "cpu" or args.force_cpu:
        print("Performance mode: CPU-friendly settings enabled.")

    force_cpu = args.force_cpu or args.performance_mode == "cpu"
    resolved_model = resolve_whisper_model(
        model_name=args.model,
        force_cpu=force_cpu,
        quality_mode=args.quality_mode,
    )
    print(f"Using Whisper model: {resolved_model}")

    devices = list_input_devices()
    if devices:
        print("Available input devices:")
        for device_id, name in devices:
            print(f"  [{device_id}] {name}")

    try:
        device_index = resolve_device_index(args.device_index)
    except ValueError as exc:
        print(str(exc))
        exit(1)
    speaker_wav = args.speaker_wav
    normalized_wav = ""
    voice_profile = VoiceProfile(
        sample_rate=0,
        duration_s=0.0,
        rms=0.0,
        peak=0.0,
        gain=1.0,
        silence_threshold=0,
    )
    if args.record_speaker or (args.engine == "local" and not speaker_wav) or (
        args.engine == "elevenlabs" and not args.elevenlabs_voice_id and not speaker_wav
    ):
        print("Record a short sample for the cloning model.")
        filename = record_sample(device_index=device_index)
        speaker_wav = filename
    if args.engine == "local":
        if not speaker_wav or not os.path.exists(speaker_wav):
            print("A valid --speaker-wav is required for local cloning.")
            exit(1)
        voice_profile = analyze_speaker_wav(speaker_wav)
        normalized_wav = normalize_speaker_wav(speaker_wav, voice_profile, args.profile_name)
        save_profile(args.profile_name, voice_profile)
        silence_threshold = max(
            calibrate_silence_threshold(device_index=device_index),
            voice_profile.silence_threshold,
        )
        print(f"Calibrated silence threshold: {silence_threshold}")
        print(f"Auto-tuned playback gain: {voice_profile.gain:.2f}")
        elevenlabs_voice_id = None
    else:
        silence_threshold = calibrate_silence_threshold(device_index=device_index)
        if args.elevenlabs_voice_id:
            elevenlabs_voice_id = args.elevenlabs_voice_id
        else:
            if not speaker_wav or not os.path.exists(speaker_wav):
                print("Provide --speaker-wav or --record-speaker to clone with ElevenLabs.")
                exit(1)
            engine = ElevenLabsEngine(
                ElevenLabsConfig(api_key=get_api_key(), voice_id="placeholder")
            )
            cloned = engine.clone_voice(args.elevenlabs_clone_name, [speaker_wav])
            elevenlabs_voice_id = cloned.voice_id
        normalized_wav = ""
        voice_profile = VoiceProfile(
            sample_rate=0,
            duration_s=0.0,
            rms=0.0,
            peak=0.0,
            gain=1.0,
            silence_threshold=silence_threshold,
        )

    print("For Linux: Ensure virtual mic is set up with PulseAudio.")
    print("For Windows: Install VB-Audio Virtual Cable, set default output to 'CABLE Input'.")
    print("Mood tracking and per-word timestamps are enabled for every utterance.")

    config = AppConfig(
        speaker_wav=normalized_wav if args.engine == "local" else "",
        tts_model=args.tts_model,
        model_name=resolved_model,
        language=args.language,
        use_noise_reduction=not args.no_noise_reduction and preset.use_noise_reduction,
        silence_chunks=args.silence_chunks if args.silence_chunks is not None else preset.silence_chunks,
        min_buffer_chunks=args.min_buffer_chunks if args.min_buffer_chunks is not None else preset.min_buffer_chunks,
        silence_threshold=silence_threshold,
        device_index=device_index,
        playback_gain=voice_profile.gain if args.engine == "local" else 1.0,
        profile_name=args.profile_name,
        auto_upgrade=args.auto_upgrade,
        performance_mode=args.performance_mode,
        playback_device=args.playback_device,
        playback_block_size=args.playback_block_size,
        engine=args.engine,
        elevenlabs_voice_id=elevenlabs_voice_id,
        force_cpu=force_cpu,
        auto_language=args.auto_language,
        run_mode=args.run_mode,
    )

    if args.ui:
        run_control_ui(config)
        return

    status_bus = StatusBus()
    stop_event, record_thread, process_thread = start_threads(config, status_bus)
    print("Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping...")
        stop_event.set()
        record_thread.join()
        process_thread.join()
        print("Stopped.")

if __name__ == "__main__":
    main()
