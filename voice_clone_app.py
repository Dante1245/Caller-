#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import queue
import threading
import time
import wave
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tts_engine import LocalTTS
from dataclasses import dataclass, replace
from pathlib import Path

# Runtime dependencies are loaded only after CLI validation and diagnostics.
from audio_capture import UtteranceBuffer, pcm_rms

from conversation_ai import (
    compute_speaking_rate_wpm,
    infer_intent,
    suggest_response_style,
    conversation_insights,
)
from stt_engine import STTConfig, load_stt, recognition_warnings
from audio_metrics import measure_audio
from elevenlabs_engine import ElevenLabsConfig, ElevenLabsEngine, get_api_key
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


def load_runtime():
    global np, sd, torch, whisper, nr, sentiment_analyzer
    global PlaybackConfig, PlaybackEngine, VoiceProfile
    global analyze_speaker_wav, normalize_speaker_wav, save_profile
    import numpy as np
    import sounddevice as sd
    import torch
    import whisper
    import noisereduce as nr
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    from audio_playback import PlaybackConfig, PlaybackEngine
    from voice_profile import analyze_speaker_wav, normalize_speaker_wav, save_profile, VoiceProfile

    sentiment_analyzer = SentimentIntensityAnalyzer()


# Audio constants for predictable, low-latency behavior
CHANNELS = 1
SAMPLE_RATE = 16000
CHUNK = 1024
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
    log_transcripts: bool = False
    stt_backend: str = "whisper"
    compute_type: str = "int8"
    initial_prompt: str = ""
    elevenlabs_model: str = "eleven_flash_v2_5"
    voice_speed: float = 1.0
    max_clip_seconds: float = 30.0
    max_queue_seconds: float = 10.0


class StatusBus:
    def __init__(self) -> None:
        self.queue: queue.Queue[str] = queue.Queue(maxsize=500)
        self.lock = threading.Lock()
        self.metrics = {"rms_dbfs": -120.0, "clipping_percent": 0.0, "utterances": 0, "dropped": 0}
        self.transcripts = []

    def update_metrics(self, **values):
        with self.lock:
            self.metrics.update(values)

    def snapshot(self):
        with self.lock:
            return dict(self.metrics)

    def remember(self, text):
        with self.lock:
            self.transcripts.append(text)
            self.transcripts = self.transcripts[-200:]

    def insights(self):
        with self.lock:
            text = " ".join(self.transcripts)
        return conversation_insights(text)

    def emit(self, message: str) -> None:
        print(message)
        try:
            self.queue.put_nowait(message)
        except queue.Full:
            pass


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


def record_sample(duration=8, filename=None, device_index=None, stop_event=None):
    """Capture a short, high-fidelity sample for cloning."""
    if not filename:
        filename = f"sample_{time.time()}.wav"

    frames = capture_frames(duration, device_index, stop_event)
    if stop_event is not None and stop_event.is_set():
        return None
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(b"".join(frames))
    print("Sample recorded.")
    return filename


def capture_frames(seconds, device_index=None, stop_event=None):
    with sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="int16",
        blocksize=CHUNK,
        device=device_index,
    ) as stream:
        frames = []
        for _ in range(max(1, int(SAMPLE_RATE / CHUNK * seconds))):
            if stop_event is not None and stop_event.is_set():
                break
            frames.append(bytes(stream.read(CHUNK)[0]))
        return frames


def is_silent(data, threshold=500):
    """Check if audio data is silent based on RMS."""
    return pcm_rms(data) < threshold


def get_device_and_model(model_name=None, force_cpu=False, backend="whisper", compute_type="int8"):
    device = "cpu" if force_cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    preferred_model = model_name or ("small.en" if device == "cuda" else "base.en")
    print(f"Loading {backend} model '{preferred_model}' on {device}...")
    return load_stt(STTConfig(preferred_model, backend, device, compute_type))


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
    frames = capture_frames(seconds, device_index)
    rms = pcm_rms(b"".join(frames))
    return max(200, int(rms * 1.8))


def real_time_record(audio_queue, stop_event, config: AppConfig, status_bus: StatusBus):
    """Continuously record audio, detect silence, and enqueue denoised clips."""
    segmenter = UtteranceBuffer(
        config.silence_threshold,
        config.silence_chunks,
        config.min_buffer_chunks,
        max_chunks=max(1, int(getattr(config, "max_clip_seconds", 30) * SAMPLE_RATE / CHUNK)),
    )
    with sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="int16",
        blocksize=CHUNK,
        device=config.device_index,
    ) as stream:
        status_bus.emit("Listening. Speak and pause to process.")
        while not stop_event.is_set():
            data, overflowed = stream.read(CHUNK)
            if overflowed:
                status_bus.emit("Microphone overflow: some input samples were lost.")
            level_audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768
            status_bus.update_metrics(**measure_audio(level_audio))
            clip = segmenter.feed(bytes(data))
            if clip is None:
                continue
            audio = np.frombuffer(clip, dtype=np.int16).astype(np.float32) / 32768.0
            try:
                audio_queue.put_nowait((time.monotonic(), audio))
            except queue.Full:
                status_bus.update_metrics(dropped=status_bus.snapshot()["dropped"] + 1)
                status_bus.emit("Processing is behind; dropped a new clip. Try a smaller model.")


def log_transcript(
    text,
    mood,
    compound,
    word_timings,
    intent="general",
    speaking_rate_wpm=0.0,
    detected_language="en",
):
    timestamp = time.ctime()
    with TRANSCRIPTS_PATH.open("a", encoding="utf-8") as f:
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
    model = get_device_and_model(
        config.model_name,
        force_cpu=config.force_cpu,
        backend=getattr(config, "stt_backend", "whisper"),
        compute_type=getattr(config, "compute_type", "int8"),
    )
    tts_engine = None
    gain_controller = None
    playback = None
    elevenlabs_engine = None
    if config.run_mode == "test":
        pass
    elif config.engine == "local":
        from tts_engine import LocalTTS

        tts_engine = LocalTTS(
            model_name=config.tts_model,
            device="cpu" if config.force_cpu else None,
            speed=config.voice_speed,
        )
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
            ElevenLabsConfig(
                api_key=get_api_key(),
                voice_id=config.elevenlabs_voice_id,
                device=config.playback_device,
                block_size=config.playback_block_size,
                model_id=config.elevenlabs_model,
                speed=config.voice_speed,
            )
        )
    try:
        report_data = load_report()
        report = UsageReport(**report_data) if report_data else UsageReport()
        report = start_session(report, config.profile_name)
        ready_event = getattr(stop_event, "ready", None)
        if ready_event is not None:
            ready_event.set()
        while not stop_event.is_set():
            audio_file = None
            try:
                audio_file = audio_queue.get(timeout=0.2)
                if isinstance(audio_file, tuple):
                    captured_at, audio_file = audio_file
                    queue_age = time.monotonic() - captured_at
                    if queue_age > getattr(config, "max_queue_seconds", 10):
                        status_bus.update_metrics(dropped=status_bus.snapshot()["dropped"] + 1)
                        status_bus.emit("Discarded a stale phrase to avoid delayed speech.")
                        continue
                start_time = time.perf_counter()
                audio_stats = measure_audio(audio_file)
                if getattr(config, "use_noise_reduction", False):
                    audio_file = nr.reduce_noise(y=audio_file, sr=SAMPLE_RATE)
                if stop_event.is_set():
                    break
                transcription_language = None if config.auto_language else config.language
                result = model.transcribe(
                    audio_file,
                    language=transcription_language,
                    word_timestamps=True,
                    fp16=torch.cuda.is_available() and not config.force_cpu,
                    temperature=0.0,
                    best_of=1,
                    beam_size=1,
                    initial_prompt=getattr(config, "initial_prompt", "") or None,
                    condition_on_previous_text=False,
                )
                text = result.get("text", "").strip()
                if not text:
                    continue
                detected_language = result.get("language", config.language)
                mood, compound = (
                    analyze_mood(text) if detected_language == "en" else ("unavailable", 0.0)
                )
                for warning in recognition_warnings(result):
                    status_bus.emit(warning)
                status_bus.remember(text)
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
                            f"  {word['word']} :: {(word['start'] or 0):.2f}s -> {(word['end'] or 0):.2f}s"
                        )
                if stop_event.is_set():
                    break
                if config.run_mode == "test":
                    status_bus.emit(
                        "TEST mode: transcription and analytics complete; voice playback skipped."
                    )
                elif config.engine == "local":
                    text_to_cloned_voice(
                        text,
                        replace(config, language=detected_language)
                        if config.auto_language
                        else config,
                        tts_engine,
                        gain_controller,
                        playback,
                        mood,
                        stop_event=stop_event,
                    )
                else:
                    elevenlabs_engine.synthesize(
                        text, mood, stop_event=stop_event, muted=getattr(stop_event, "muted", None)
                    )
                latency_ms = (time.perf_counter() - start_time) * 1000
                status_bus.update_metrics(
                    latency_ms=latency_ms, utterances=status_bus.snapshot()["utterances"] + 1
                )
                status_bus.emit(f"Processing + playback latency: {latency_ms:.1f} ms")
                quality = assess_call_quality(
                    latency_ms=latency_ms,
                    speaking_rate_wpm=speaking_rate_wpm,
                    mood=mood,
                    has_word_timestamps=bool(word_timings),
                    **{key: audio_stats[key] for key in ("rms_dbfs", "clipping_percent")},
                )
                status_bus.emit(
                    f"Pipeline score (heuristic): {quality.score:.1f}/100 ({quality.label}) | tip: {quality.recommendation}"
                )
                report = update_report(report, latency_ms, len(text.split()), config.profile_name)
                save_report(report)
                if config.auto_upgrade and report.recommendations:
                    status_bus.emit("Auto-upgrade hints:")
                    for recommendation in report.recommendations:
                        status_bus.emit(f"  - {recommendation}")
                if config.log_transcripts:
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
                if audio_file is not None:
                    audio_queue.task_done()
    finally:
        if playback:
            playback.stop()
        if elevenlabs_engine:
            elevenlabs_engine.close()


def text_to_cloned_voice(
    text,
    config: AppConfig,
    tts_engine: LocalTTS,
    gain_controller: AdaptiveGain,
    playback: PlaybackEngine,
    mood="neutral",
    stop_event=None,
):
    """Synthesize and play cloned voice using local XTTS."""
    if stop_event is not None and (
        stop_event.is_set() or getattr(stop_event, "muted", threading.Event()).is_set()
    ):
        return
    result = tts_engine.synthesize(
        text=text,
        speaker_wav=config.speaker_wav,
        language=config.language,
    )
    audio = result.audio * config.playback_gain
    observed_peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    config.playback_gain = gain_controller.update(observed_peak)
    audio = np.clip(audio, -1.0, 1.0)
    audio = trim_silence(audio)
    print(f"Playing cloned voice (mood={mood})")
    playback.play(audio, stop_event=stop_event, muted=getattr(stop_event, "muted", None))


def text_to_elevenlabs_voice(text: str, engine: ElevenLabsEngine, mood: str) -> None:
    engine.synthesize(text, mood)


def list_input_devices():
    from self_check import list_devices

    return [(d["index"], d["name"]) for d in list_devices()["inputs"]]


def resolve_device_index(requested):
    if requested is None:
        return None
    devices = dict(list_input_devices())
    if requested in devices:
        return requested
    raise ValueError(f"Invalid device index {requested}. Available: {list(devices)}")


def start_threads(config: AppConfig, status_bus: StatusBus):
    audio_queue = queue.Queue(maxsize=3)
    stop_event = threading.Event()
    stop_event.ready = threading.Event()
    stop_event.failed = False
    stop_event.muted = threading.Event()

    def guarded(target):
        try:
            if target is real_time_record:
                while not stop_event.ready.wait(0.1):
                    if stop_event.is_set():
                        return
                if stop_event.is_set():
                    return
            target(audio_queue, stop_event, config, status_bus)
        except Exception as exc:
            stop_event.failed = True
            status_bus.emit(f"Pipeline stopped: {type(exc).__name__}: {exc}")
            stop_event.set()

    record_thread = threading.Thread(target=guarded, args=(real_time_record,), daemon=True)
    process_thread = threading.Thread(target=guarded, args=(process_audio,), daemon=True)
    process_thread.start()
    record_thread.start()
    return stop_event, record_thread, process_thread


def run_control_ui(config: AppConfig):
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
    from self_check import list_devices

    root = tk.Tk()
    root.title("Caller · Voice studio")
    root.geometry("920x780")
    root.minsize(820, 720)
    root.configure(background="#f3f5f7")
    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure("TFrame", background="#f3f5f7")
    style.configure("TLabel", background="#f3f5f7", foreground="#172a3a", font=("Helvetica", 12))
    style.configure("Title.TLabel", font=("Helvetica", 27, "bold"))
    style.configure("Muted.TLabel", foreground="#536575")
    style.configure("TButton", padding=(14, 9), font=("Helvetica", 11))
    style.configure("Accent.TButton", background="#176c60", foreground="white")
    style.configure("TRadiobutton", background="#f3f5f7", font=("Helvetica", 11))
    status_bus = StatusBus()
    state = {"threads": (), "stop": None, "closing": False, "operation": "Listening"}
    tool_results = queue.Queue()
    speaker_var = tk.StringVar(
        root, value=config.speaker_wav or "Choose a reference voice for local playback"
    )
    detail_var = tk.StringVar(root, value="Transcription-only mode works without a voice sample.")
    mode_var = tk.StringVar(root, value=config.run_mode)
    status_var = tk.StringVar(root, value="Ready")
    input_var = tk.StringVar(root, value="System default")
    output_var = tk.StringVar(root, value="System default")
    inputs, outputs = {"System default": None}, {"System default": None}
    backend_var = tk.StringVar(root, value=config.stt_backend)
    model_var = tk.StringVar(root, value=config.model_name)
    language_var = tk.StringVar(root, value="auto" if config.auto_language else config.language)
    engine_var = tk.StringVar(root, value=config.engine)
    voice_var = tk.StringVar(root, value=config.elevenlabs_voice_id or "")
    cloud_model_var = tk.StringVar(root, value=config.elevenlabs_model)
    speed_var = tk.StringVar(root, value=str(config.voice_speed))
    prompt_var = tk.StringVar(root, value=config.initial_prompt)
    threshold_var = tk.StringVar(root, value=str(config.silence_threshold))
    noise_var = tk.BooleanVar(root, value=config.use_noise_reduction)
    mute_var = tk.BooleanVar(root, value=False)
    metrics_var = tk.StringVar(root, value="Input — · 0 phrases · 0 dropped")
    settings_widgets = []

    def selected_config():
        model = model_var.get().strip()
        language = language_var.get().strip().lower()
        if not model or not language:
            raise ValueError("Choose a model and a language (or auto).")
        if language != "en" and model.endswith(".en"):
            raise ValueError("Choose a multilingual model, such as base, for this language.")
        if backend_var.get() == "faster-whisper" and model.endswith(".pt"):
            raise ValueError(
                "Faster-whisper requires a model name or a converted model directory, not a .pt file."
            )
        speed = float(speed_var.get())
        threshold = int(threshold_var.get())
        if not 0.7 <= speed <= 1.2 or not 1 <= threshold <= 32768:
            raise ValueError("Speed must be 0.7–1.2; microphone threshold must be 1–32768.")
        if len(prompt_var.get()) > 2000:
            raise ValueError("Vocabulary hint must be at most 2000 characters.")
        return replace(
            config,
            run_mode=mode_var.get(),
            device_index=inputs[input_var.get()],
            playback_device=outputs[output_var.get()],
            stt_backend=backend_var.get(),
            model_name=model,
            language="en" if language == "auto" else language,
            auto_language=language == "auto",
            engine=engine_var.get(),
            elevenlabs_voice_id=voice_var.get().strip() or None,
            elevenlabs_model=cloud_model_var.get().strip(),
            voice_speed=speed,
            initial_prompt=prompt_var.get(),
            silence_threshold=threshold,
            use_noise_reduction=noise_var.get(),
        )

    def run_tool(label, action):
        if busy():
            return
        stop = threading.Event()
        stop.failed = False
        stop.ready = threading.Event()
        stop.muted = threading.Event()

        def worker():
            try:
                action(stop)
            except Exception as exc:
                stop.failed = True
                status_bus.emit(f"{label} failed: {exc}")

        thread = threading.Thread(target=worker, daemon=True)
        state.update(stop=stop, threads=(thread,), operation=label)
        status_var.set(label)
        set_controls(True)
        thread.start()

    def calibrate_input():
        device = inputs[input_var.get()]
        status_bus.emit("Calibrating ambient sound for 1.5 seconds. Please stay quiet.")

        def action(stop):
            threshold = calibrate_silence_threshold(device)
            if not stop.is_set():
                tool_results.put(("threshold", threshold))
                status_bus.emit(f"Calibrated microphone threshold: {threshold}")

        run_tool("Calibrating…", action)

    def record_reference():
        path = filedialog.asksaveasfilename(
            parent=root,
            title="Record an 8-second reference",
            initialfile="my-reference.wav",
            defaultextension=".wav",
        )
        if not path:
            return
        device = inputs[input_var.get()]
        status_bus.emit("Recording reference for 8 seconds. Speak naturally in a quiet room.")

        def action(stop):
            recorded = record_sample(8, path, device, stop_event=stop)
            if recorded is not None:
                profile = analyze_speaker_wav(recorded)
                normalized = normalize_speaker_wav(recorded, profile, config.profile_name)
                save_profile(config.profile_name, profile)
                tool_results.put(("reference", (normalized, profile)))
                status_bus.emit(f"Reference recorded: {recorded}")

        run_tool("Recording reference…", action)

    def transcribe_recording():
        try:
            chosen = selected_config()
        except ValueError as exc:
            messagebox.showerror("Check settings", str(exc), parent=root)
            return
        path = filedialog.askopenfilename(parent=root, title="Choose a recording")
        if not path:
            return
        prefix = filedialog.asksaveasfilename(
            parent=root,
            title="Export name (creates JSON, TXT and SRT)",
            initialfile=Path(path).stem + "-transcript",
        )
        if not prefix:
            return

        def action(stop):
            from transcript_tools import transcribe_file, export_transcription

            result = transcribe_file(
                path,
                STTConfig(
                    chosen.model_name,
                    chosen.stt_backend,
                    "cuda" if torch.cuda.is_available() and not chosen.force_cpu else "cpu",
                    chosen.compute_type,
                ),
                language=None if chosen.auto_language else chosen.language,
                initial_prompt=chosen.initial_prompt or None,
                stop_event=stop,
            )
            if result is not None:
                files = export_transcription(result, prefix)
                status_bus.remember(result.get("text", ""))
                status_bus.emit(result.get("text", ""))
                status_bus.emit("Exported: " + ", ".join(files))

        run_tool("Transcribing file…", action)

    def show_insights():
        result = status_bus.insights()
        if not result["word_count"]:
            messagebox.showinfo(
                "No transcript yet", "Run a session or transcribe a recording first.", parent=root
            )
            return
        window = tk.Toplevel(root)
        window.title("Caller · Transcript insights")
        window.geometry("700x500")
        text = tk.Text(window, wrap="word", padx=18, pady=18)
        text.pack(fill="both", expand=True)
        lines = [
            "EXTRACTIVE SUMMARY",
            *result["summary"],
            "",
            "CANDIDATE ACTIONS",
            *result["candidate_actions"],
            "",
            "QUESTIONS",
            *result["questions"],
            "",
            "Keywords: " + ", ".join(result["keywords"]),
            "",
            "English text heuristics; verify candidate actions against the transcript.",
        ]
        text.insert("1.0", "\n".join(lines))
        text.configure(state="disabled")

        def export():
            path = filedialog.asksaveasfilename(
                parent=window, defaultextension=".json", initialfile="conversation-insights.json"
            )
            if path:
                from storage import write_object

                try:
                    write_object(path, result)
                except OSError as exc:
                    messagebox.showerror("Export failed", str(exc), parent=window)

        ttk.Button(window, text="Export insights…", command=export).pack(pady=10)

    def toggle_mute():
        if state["stop"] is not None:
            event = state["stop"].muted
            event.set() if mute_var.get() else event.clear()

    def run_diagnostics():
        engine, backend, mode = engine_var.get(), backend_var.get(), mode_var.get()

        def action(stop):
            report = render_report(run_self_check(engine, mode, backend))
            summary = report["summary"]
            status_bus.emit(
                f"Diagnostics: {summary['passed']} passed, {summary['failed']} need attention."
            )
            for check in report["results"]:
                status_bus.emit(
                    f"{'OK' if check['ok'] else 'CHECK'} · {check['name']}: {check['detail']}"
                )

        run_tool("Checking runtime…", action)

    def busy():
        return any(t.is_alive() for t in state["threads"])

    def refresh_devices():
        try:
            devices = list_devices()
            inputs.clear()
            inputs["System default"] = None
            outputs.clear()
            outputs["System default"] = None
            inputs.update({f"{d['index']} · {d['name']}": d["index"] for d in devices["inputs"]})
            outputs.update({f"{d['index']} · {d['name']}": d["index"] for d in devices["outputs"]})
            for widget, variable, options, selected in (
                (input_box, input_var, inputs, config.device_index),
                (output_box, output_var, outputs, config.playback_device),
            ):
                widget.configure(values=list(options))
                variable.set(
                    next((k for k, v in options.items() if v == selected), "System default")
                )
        except Exception as exc:
            status_bus.emit(f"Audio devices unavailable: {exc}")

    def browse_voice():
        path = filedialog.askopenfilename(
            parent=root,
            title="Choose a reference voice",
            filetypes=[("Audio", "*.wav *.flac *.mp3"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            profile = analyze_speaker_wav(path)
            config.speaker_wav = normalize_speaker_wav(path, profile, config.profile_name)
            save_profile(config.profile_name, profile)
            config.playback_gain = profile.gain
            speaker_var.set(config.speaker_wav)
            for warning in profile.warnings or []:
                status_bus.emit(warning)
            detail_var.set(
                f"{profile.duration_s:.1f} seconds · {profile.sample_rate:,} Hz · gain {profile.gain:.2f}"
            )
        except Exception as exc:
            messagebox.showerror("Could not load reference", str(exc), parent=root)

    def start_app():
        if busy():
            return
        try:
            run_config = selected_config()
            if run_config.run_mode == "live":
                if run_config.engine == "local" and not Path(run_config.speaker_wav).is_file():
                    raise ValueError("Choose a reference sample before local voice playback.")
                if run_config.engine == "elevenlabs":
                    get_api_key()
                    if not run_config.elevenlabs_voice_id or not run_config.elevenlabs_model:
                        raise ValueError("Enter the ElevenLabs voice ID and model in AI settings.")
        except (ValueError, RuntimeError) as exc:
            messagebox.showwarning("Check settings", str(exc), parent=root)
            return
        status_var.set("Starting…")
        status_bus.emit(
            f"Starting {run_config.run_mode} mode. Models may take time to load on first use."
        )
        stop, recorder, processor = start_threads(run_config, status_bus)
        state.update(stop=stop, threads=(recorder, processor), operation="Listening")
        toggle_mute()
        set_controls(True)

    def stop_app():
        if state["stop"]:
            state["stop"].set()
            status_var.set("Stopping…")
            status_bus.emit("Stopping after the current model or playback operation finishes.")
            stop_button.configure(state="disabled")

    def set_controls(running):
        for widget in (
            start_button,
            live_radio,
            test_radio,
            refresh_button,
            calibrate_button,
            file_button,
            diagnostics_button,
            record_button,
        ):
            widget.configure(state="disabled" if running else "normal")
        load_button.configure(state="disabled" if running else "normal")
        input_box.configure(state="disabled" if running else "readonly")
        output_box.configure(state="disabled" if running else "readonly")
        stop_button.configure(state="normal" if running else "disabled")
        for widget in settings_widgets:
            widget.configure(
                state="disabled"
                if running
                else "readonly"
                if isinstance(widget, ttk.Combobox) and widget is not model_box
                else "normal"
            )

    def poll_status():
        if state["threads"] and not busy():
            status_var.set("Stopped with an error" if state["stop"].failed else "Ready")
            state.update(threads=(), stop=None)
            set_controls(False)
        elif (
            busy() and state["stop"] and not state["stop"].is_set() and state["stop"].ready.is_set()
        ):
            status_var.set(state["operation"])
        while not tool_results.empty():
            kind, value = tool_results.get_nowait()
            if kind == "threshold":
                threshold_var.set(str(value))
            elif kind == "reference":
                path, profile = value
                config.speaker_wav = path
                config.playback_gain = profile.gain
                speaker_var.set(path)
                detail_var.set(f"{profile.duration_s:.1f} seconds · {profile.sample_rate:,} Hz")
                for warning in profile.warnings or []:
                    status_bus.emit(warning)
        metrics = status_bus.snapshot()
        meter["value"] = max(0, min(100, (metrics["rms_dbfs"] + 60) / 60 * 100))
        metrics_var.set(
            f"Input {metrics['rms_dbfs']:.0f} dBFS · {metrics['utterances']} phrases · {metrics['dropped']} dropped · {metrics.get('latency_ms', 0):.0f} ms"
        )
        log.configure(state="normal")
        while True:
            try:
                message = status_bus.queue.get_nowait()
            except queue.Empty:
                break
            log.insert(tk.END, message + "\n")
        if int(log.index("end-1c").split(".")[0]) > 1000:
            log.delete("1.0", "200.0")
        log.see(tk.END)
        log.configure(state="disabled")
        if state["closing"] and not busy():
            root.destroy()
            return
        root.after(150, poll_status)

    outer = ttk.Frame(root, padding=28)
    outer.pack(fill="both", expand=True)
    title_row = ttk.Frame(outer)
    title_row.pack(fill="x")
    ttk.Label(title_row, text="Caller", style="Title.TLabel").pack(side="left")
    ttk.Label(title_row, textvariable=status_var, style="Muted.TLabel").pack(side="right")
    ttk.Label(
        outer, text="Your speech. Your voice. One controlled audio pipeline.", style="Muted.TLabel"
    ).pack(anchor="w", pady=(4, 22))
    tabs = ttk.Notebook(outer)
    tabs.pack(fill="x", pady=(0, 14))
    audio_tab = ttk.Frame(tabs, padding=14)
    ai_tab = ttk.Frame(tabs, padding=14)
    tools_tab = ttk.Frame(tabs, padding=14)
    tabs.add(audio_tab, text="Audio & reference")
    tabs.add(ai_tab, text="AI settings")
    tabs.add(tools_tab, text="Tools")
    ttk.Label(audio_tab, text="AUDIO ROUTING", style="Muted.TLabel").pack(anchor="w")
    routing = ttk.Frame(audio_tab)
    routing.pack(fill="x", pady=(8, 18))
    routing.columnconfigure(1, weight=1)
    ttk.Label(routing, text="Microphone").grid(row=0, column=0, sticky="w", padx=(0, 16), pady=4)
    input_box = ttk.Combobox(routing, textvariable=input_var, values=list(inputs), state="readonly")
    input_box.grid(row=0, column=1, sticky="ew", pady=4)
    ttk.Label(routing, text="Playback").grid(row=1, column=0, sticky="w", pady=4)
    output_box = ttk.Combobox(
        routing, textvariable=output_var, values=list(outputs), state="readonly"
    )
    output_box.grid(row=1, column=1, sticky="ew", pady=4)
    refresh_button = ttk.Button(routing, text="Refresh", command=refresh_devices)
    refresh_button.grid(row=0, column=2, rowspan=2, padx=(12, 0))
    ttk.Label(audio_tab, text="REFERENCE VOICE", style="Muted.TLabel").pack(anchor="w")
    reference = ttk.Frame(audio_tab)
    reference.pack(fill="x", pady=8)
    load_button = ttk.Button(reference, text="Choose audio…", command=browse_voice)
    load_button.pack(side="right", padx=(12, 0))
    ttk.Label(reference, textvariable=speaker_var, wraplength=490).pack(
        side="left", fill="x", expand=True
    )
    ttk.Label(audio_tab, textvariable=detail_var, style="Muted.TLabel").pack(anchor="w")
    ai_tab.columnconfigure(1, weight=1)
    ai_tab.columnconfigure(3, weight=1)

    def field(row, column, label, variable, values=None, editable=False):
        ttk.Label(ai_tab, text=label).grid(row=row, column=column, sticky="w", padx=(0, 10), pady=4)
        widget = (
            ttk.Combobox(
                ai_tab,
                textvariable=variable,
                values=values,
                width=18,
                state="normal" if editable else "readonly",
            )
            if values
            else ttk.Entry(ai_tab, textvariable=variable, width=20)
        )
        widget.grid(row=row, column=column + 1, sticky="ew", padx=(0, 14), pady=4)
        settings_widgets.append(widget)
        return widget

    field(0, 0, "Recognition", backend_var, ["whisper", "faster-whisper"])
    model_box = field(
        0,
        2,
        "Model",
        model_var,
        ["tiny.en", "base.en", "small.en", "base", "small", "large-v3", "turbo"],
        True,
    )
    field(1, 0, "Language", language_var)
    field(1, 2, "Voice engine", engine_var, ["local", "elevenlabs"])
    field(2, 0, "Cloud voice ID", voice_var)
    field(2, 2, "Cloud model", cloud_model_var)
    field(3, 0, "Voice speed", speed_var)
    field(3, 2, "Silence RMS", threshold_var)
    field(4, 0, "Vocabulary hint", prompt_var)
    noise_box = ttk.Checkbutton(ai_tab, text="Reduce background noise", variable=noise_var)
    noise_box.grid(row=4, column=2, columnspan=2, sticky="w")
    settings_widgets.append(noise_box)
    ttk.Label(
        ai_tab,
        text="Use a language code (en, es, fr…) or auto. Faster-whisper requires requirements-fast.txt.",
        style="Muted.TLabel",
        wraplength=780,
    ).grid(row=5, column=0, columnspan=4, sticky="w", pady=8)
    ttk.Label(tools_tab, text="LOCAL AUDIO TOOLS", style="Muted.TLabel").pack(
        anchor="w", pady=(0, 10)
    )
    file_button = ttk.Button(
        tools_tab, text="Transcribe a recording → JSON / TXT / SRT", command=transcribe_recording
    )
    file_button.pack(anchor="w", pady=4)
    record_button = ttk.Button(
        tools_tab, text="Record an 8-second voice reference…", command=record_reference
    )
    record_button.pack(anchor="w", pady=4)
    calibrate_button = ttk.Button(
        tools_tab, text="Calibrate microphone (stay quiet)", command=calibrate_input
    )
    calibrate_button.pack(anchor="w", pady=4)
    diagnostics_button = ttk.Button(
        tools_tab, text="Check runtime & devices", command=run_diagnostics
    )
    diagnostics_button.pack(anchor="w", pady=4)
    ttk.Button(tools_tab, text="View conversation insights…", command=show_insights).pack(
        anchor="w", pady=4
    )
    ttk.Label(
        tools_tab,
        text="File transcription stays local. Exports are saved only when you choose a destination.",
        style="Muted.TLabel",
        wraplength=780,
    ).pack(anchor="w", pady=10)
    ttk.Separator(outer).pack(fill="x", pady=8)
    actions = ttk.Frame(outer)
    actions.pack(fill="x")
    test_radio = ttk.Radiobutton(actions, text="Transcribe only", variable=mode_var, value="test")
    test_radio.pack(side="left")
    live_radio = ttk.Radiobutton(actions, text="Live voice", variable=mode_var, value="live")
    live_radio.pack(side="left", padx=12)
    ttk.Checkbutton(actions, text="Mute voice", variable=mute_var, command=toggle_mute).pack(
        side="left"
    )
    stop_button = ttk.Button(actions, text="Stop", command=stop_app)
    stop_button.pack(side="right")
    start_button = ttk.Button(
        actions, text="Start session", command=start_app, style="Accent.TButton"
    )
    start_button.pack(side="right", padx=10)
    ttk.Label(outer, textvariable=metrics_var, style="Muted.TLabel").pack(anchor="w", pady=(14, 4))
    meter = ttk.Progressbar(outer, maximum=100)
    meter.pack(fill="x", pady=(0, 12))
    log_frame = ttk.Frame(outer)
    log_frame.pack(fill="both", expand=True)
    log = tk.Text(
        log_frame,
        height=10,
        background="#142632",
        foreground="#dbe8ed",
        borderwidth=0,
        padx=14,
        pady=12,
        font=("Menlo", 11),
        wrap="word",
        state="disabled",
    )
    scrollbar = ttk.Scrollbar(log_frame, command=log.yview)
    log.configure(yscrollcommand=scrollbar.set)
    scrollbar.pack(side="right", fill="y")
    log.pack(fill="both", expand=True)
    ttk.Label(
        outer,
        text="Route playback to a virtual audio device, then select it as your call app’s microphone.",
        style="Muted.TLabel",
        wraplength=720,
    ).pack(anchor="w", pady=(12, 0))
    set_controls(False)
    refresh_devices()
    status_bus.emit("Ready. Select audio devices and a mode, then start a session.")
    root.after(150, poll_status)

    def on_close():
        state["closing"] = True
        stop_app()
        if not busy():
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
    parser.add_argument(
        "--elevenlabs-clone-name", default="LocalClone", help="Name for ElevenLabs cloned voice."
    )
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
    parser.add_argument(
        "--force-cpu", action="store_true", help="Run STT/TTS on CPU even when CUDA is available."
    )
    parser.add_argument("--playback-device", type=int, default=None)
    parser.add_argument("--playback-block-size", type=int, default=2048)
    parser.add_argument("--self-check", action="store_true", help="Run diagnostics and exit.")
    parser.add_argument(
        "--run-mode",
        choices=["live", "test"],
        default="live",
        help="Run mode: live plays cloned output, test runs full analytics without playback.",
    )
    parser.add_argument(
        "--log-transcripts",
        action="store_true",
        help="Save recognized text to transcripts.txt (off by default).",
    )
    parser.add_argument(
        "--list-devices", action="store_true", help="List input and output device indexes and exit."
    )
    parser.add_argument("--ui", action="store_true", help="Launch the voice studio control panel.")
    parser.add_argument("--stt-backend", choices=["whisper", "faster-whisper"], default="whisper")
    parser.add_argument(
        "--compute-type", choices=["int8", "float32", "float16", "int8_float16"], default="int8"
    )
    parser.add_argument(
        "--initial-prompt", default="", help="Vocabulary/context hint for speech recognition."
    )
    parser.add_argument(
        "--input-file", help="Transcribe an existing recording without opening a microphone."
    )
    parser.add_argument(
        "--task",
        choices=["transcribe", "translate"],
        default="transcribe",
        help="File mode: translate speech into English.",
    )
    parser.add_argument(
        "--output",
        help="File mode: output prefix for JSON, TXT and SRT exports (never overwrites).",
    )
    parser.add_argument("--inspect-reference", help="Analyze a reference audio file and exit.")
    parser.add_argument(
        "--list-voices", action="store_true", help="List your ElevenLabs voices; requires API key."
    )
    parser.add_argument(
        "--list-cloud-models",
        action="store_true",
        help="List ElevenLabs speech models; requires API key.",
    )
    parser.add_argument("--elevenlabs-model", default="eleven_flash_v2_5")
    parser.add_argument(
        "--voice-speed", type=float, default=1.0, help="Voice synthesis speed: 0.7–1.2."
    )
    parser.add_argument("--max-clip-seconds", type=float, default=30.0)
    parser.add_argument(
        "--max-queue-seconds",
        type=float,
        default=10.0,
        help="Discard phrases delayed longer than this.",
    )
    parser.add_argument("--diagnostics-output", help="Write self-check JSON to this file.")
    args = parser.parse_args()
    import math

    for name, low, high in (
        ("voice_speed", 0.7, 1.2),
        ("max_clip_seconds", 1, 60),
        ("max_queue_seconds", 0.1, 120),
    ):
        value = getattr(args, name)
        if not math.isfinite(value) or not low <= value <= high:
            parser.error(f"--{name.replace('_', '-')} must be between {low} and {high}")
    if args.output and not args.input_file:
        parser.error("--output requires --input-file")
    if args.input_file and args.ui:
        parser.error("Use --input-file or --ui, not both")
    if args.task == "translate" and not args.input_file:
        parser.error("--task translate requires --input-file")
    if args.task == "translate" and args.model and args.model.endswith(".en"):
        parser.error("Translation requires a multilingual model (for example --model base)")
    if args.compute_type in ("float16", "int8_float16") and (
        args.force_cpu or args.performance_mode == "cpu"
    ):
        parser.error("CPU mode requires --compute-type int8 or float32")
    if len(args.initial_prompt) > 2000:
        parser.error("--initial-prompt must be at most 2000 characters")

    for name in ("silence_chunks", "min_buffer_chunks", "playback_block_size"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.min_buffer_chunks is not None and args.min_buffer_chunks > 469:
        parser.error("--min-buffer-chunks must be at most 469")
    if args.silence_chunks is not None and args.silence_chunks > 469:
        parser.error("--silence-chunks must be at most 469")
    import re

    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", args.profile_name):
        parser.error("--profile-name must contain 1–64 letters, digits, underscores or hyphens")
    maximum_chunks = int(args.max_clip_seconds * SAMPLE_RATE / CHUNK)
    if args.min_buffer_chunks and args.min_buffer_chunks + 3 > maximum_chunks:
        parser.error("Minimum speech buffer plus pre-roll exceeds maximum clip duration")
    return args


def main():
    args = parse_args()
    if args.self_check:
        report = render_report(run_self_check(args.engine, args.run_mode, args.stt_backend))
        if args.diagnostics_output:
            from storage import write_object

            write_object(args.diagnostics_output, report)
        print(json.dumps(report, indent=2))
        return 1 if report["summary"]["failed"] else 0
    if args.list_devices:
        from self_check import list_devices

        print(json.dumps(list_devices(), indent=2))
        return 0
    if args.list_voices or args.list_cloud_models:
        from elevenlabs_engine import list_cloud_resources

        print(
            json.dumps(
                list_cloud_resources("models" if args.list_cloud_models else "voices"), indent=2
            )
        )
        return 0
    if args.inspect_reference:
        from dataclasses import asdict
        from voice_profile import analyze_speaker_wav

        print(json.dumps(asdict(analyze_speaker_wav(args.inspect_reference)), indent=2))
        return 0
    load_runtime()
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
    if args.auto_language or args.language != "en" or args.task == "translate":
        if args.model and args.model.endswith(".en"):
            raise ValueError(
                "English-only .en models cannot be used with another/automatic language."
            )
        resolved_model = resolved_model.removesuffix(".en")
    print(f"Using Whisper model: {resolved_model}")

    if args.input_file:
        from transcript_tools import transcribe_file, export_transcription

        result = transcribe_file(
            args.input_file,
            STTConfig(
                resolved_model,
                args.stt_backend,
                "cuda" if torch.cuda.is_available() and not force_cpu else "cpu",
                args.compute_type,
            ),
            language=None if args.auto_language or args.task == "translate" else args.language,
            task=args.task,
            initial_prompt=args.initial_prompt or None,
        )
        prefix = args.output or str(
            Path("exports") / (Path(args.input_file).stem + "-" + time.strftime("%Y%m%d-%H%M%S"))
        )
        for path in export_transcription(result, prefix):
            print(f"Saved: {path}")
        print(result.get("text", ""))
        return 0

    devices = list_input_devices()
    if devices:
        print("Available input devices:")
        for device_id, name in devices:
            print(f"  [{device_id}] {name}")

    try:
        device_index = resolve_device_index(args.device_index)
    except ValueError as exc:
        print(str(exc))
        raise ValueError(str(exc)) from exc
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
    elevenlabs_voice_id = args.elevenlabs_voice_id
    silence_threshold = 500
    if args.run_mode == "live":
        if args.record_speaker:
            speaker_wav = record_sample(device_index=device_index)
        if args.engine == "local":
            if speaker_wav:
                voice_profile = analyze_speaker_wav(speaker_wav)
                normalized_wav = normalize_speaker_wav(
                    speaker_wav, voice_profile, args.profile_name
                )
                save_profile(args.profile_name, voice_profile)
            elif not args.ui:
                raise ValueError(
                    "Provide --speaker-wav or --record-speaker, or use --ui to load a reference."
                )
        else:
            get_api_key()
            if not elevenlabs_voice_id:
                if not speaker_wav:
                    raise ValueError(
                        "Provide --elevenlabs-voice-id or --speaker-wav for ElevenLabs."
                    )
                engine = ElevenLabsEngine(
                    ElevenLabsConfig(api_key=get_api_key(), voice_id="pending")
                )
                try:
                    elevenlabs_voice_id = engine.clone_voice(
                        args.elevenlabs_clone_name, [speaker_wav]
                    ).voice_id
                finally:
                    engine.close()
    if not args.ui:
        silence_threshold = calibrate_silence_threshold(device_index=device_index)

    print("For Linux: Ensure virtual mic is set up with PulseAudio.")
    print("For Windows: Install VB-Audio Virtual Cable, set default output to 'CABLE Input'.")
    print("Mood tracking and per-word timestamps are enabled for every utterance.")

    config = AppConfig(
        speaker_wav=normalized_wav if args.engine == "local" else "",
        tts_model=args.tts_model,
        model_name=resolved_model,
        language=args.language,
        use_noise_reduction=not args.no_noise_reduction and preset.use_noise_reduction,
        silence_chunks=args.silence_chunks
        if args.silence_chunks is not None
        else preset.silence_chunks,
        min_buffer_chunks=args.min_buffer_chunks
        if args.min_buffer_chunks is not None
        else preset.min_buffer_chunks,
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
        log_transcripts=args.log_transcripts,
        stt_backend=args.stt_backend,
        compute_type=args.compute_type,
        initial_prompt=args.initial_prompt,
        elevenlabs_model=args.elevenlabs_model,
        voice_speed=args.voice_speed,
        max_clip_seconds=args.max_clip_seconds,
        max_queue_seconds=args.max_queue_seconds,
    )

    if args.ui:
        run_control_ui(config)
        return

    status_bus = StatusBus()
    stop_event, record_thread, process_thread = start_threads(config, status_bus)
    print("Press Ctrl+C to stop.")
    try:
        while not stop_event.wait(0.2):
            pass
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        stop_event.set()
        record_thread.join()
        process_thread.join()
        print("Stopped.")
    return 1 if stop_event.failed else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ImportError, RuntimeError, ValueError, OSError) as exc:
        print(f"Caller could not start: {exc}")
        print("Run --self-check for dependency and device diagnostics.")
        raise SystemExit(1)
