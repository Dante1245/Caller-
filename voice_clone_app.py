#!/usr/bin/env python3
import argparse
import importlib.util
import os
import queue
import threading
import time
import wave
from pathlib import Path

import noisereduce as nr
import numpy as np
import torch
import whisper
from elevenlabs import ElevenLabs, play
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

"""
Ultra-low-latency, sentiment-aware voice cloning pipeline.

Updates in this version:
- Mood tracking via VADER sentiment to adapt ElevenLabs stability/similarity per utterance.
- Per-word timestamps from Whisper for detailed call analytics and responsive UX.
- GPU-aware Whisper loading and tuned silence detection for faster turn-taking.
- Streamed playback with ElevenLabs for minimum disk I/O and reduced latency.
"""

# Note: Set your ElevenLabs API key as environment variable ELEVENLABS_API_KEY
# For voice cloning, use a custom voice ID from ElevenLabs dashboard (upload sample for cloning).
# ElevenLabs provides state-of-the-art voice cloning with ZERO glitches or robotic sounds—ultra-natural output.
# Optimized for high-end computers like Alienware/gaming laptops; runs efficiently on CPU/GPU with minimal delay.
# After running, connect as device microphone: Set up virtual mic and route output to it for professional calls.

client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))
sentiment_analyzer = SentimentIntensityAnalyzer()

# Audio constants for predictable, low-latency behavior
SAMPLE_FORMAT = "paInt16"
CHANNELS = 1
SAMPLE_RATE = 16000
CHUNK = 1024
SILENCE_CHUNKS = 8  # shorter silence window for snappier responses
MIN_BUFFER_CHUNKS = 40  # lower minimum to ship faster to Whisper
DEFAULT_TTS_MODEL = "eleven_turbo_v2"
DEFAULT_TARGET_RMS = 0.18


def _require_pyaudio():
    """
    Ensure PyAudio is available and return the imported module.

    Raises:
        RuntimeError: If PyAudio is not installed in the current environment.
    """
    if importlib.util.find_spec("pyaudio") is None:
        raise RuntimeError(
            "PyAudio is required for microphone capture. Install system "
            "PortAudio headers and `pip install pyaudio`, or run with "
            "--audio-file to process a pre-recorded clip."
        )
    import pyaudio

    return pyaudio


def list_input_devices():
    """
    Print available audio input devices with their indices.
    """
    pyaudio = _require_pyaudio()
    pa = pyaudio.PyAudio()
    try:
        device_count = pa.get_device_count()
        print("Available input devices:")
        for idx in range(device_count):
            info = pa.get_device_info_by_index(idx)
            if int(info.get("maxInputChannels", 0)) > 0:
                print(f"  {idx}: {info.get('name')}")
    finally:
        pa.terminate()


def normalize_audio(audio_np, target_rms=DEFAULT_TARGET_RMS, eps=1e-8):
    """
    Apply light automatic gain control to stabilize loudness.
    """
    rms = np.sqrt(np.mean(audio_np ** 2))
    if rms < eps:
        return audio_np
    gain = min(target_rms / rms, 3.0)
    return np.clip(audio_np * gain, -32768, 32767)


def calibrate_noise_floor(stream, frames_per_buffer, seconds=1.5):
    """
    Estimate background noise RMS for adaptive silence thresholding.
    """
    if seconds <= 0:
        return None
    frame_count = int(SAMPLE_RATE / frames_per_buffer * seconds)
    samples = []
    for _ in range(frame_count):
        data = stream.read(frames_per_buffer, exception_on_overflow=False)
        samples.append(np.frombuffer(data, dtype=np.int16))
    if not samples:
        return None
    audio_np = np.concatenate(samples).astype(np.float32)
    return float(np.sqrt(np.mean(audio_np ** 2)))

def record_sample(duration=5, filename=None):
    """
    Record a short high-fidelity WAV sample suitable for voice cloning.
    
    Parameters:
        duration (int | float): Length of the recording in seconds (default 5).
        filename (str | None): Destination filepath for the WAV file; if None a timestamped filename is generated.
    
    Returns:
        str: Path to the written WAV file.
    """
    if not filename:
        filename = f"sample_{time.time()}.wav"
    filename = str(filename)

    pyaudio = _require_pyaudio()
    pa = pyaudio.PyAudio()
    print("Recording sample...")
    stream = pa.open(
        format=getattr(pyaudio, SAMPLE_FORMAT),
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        frames_per_buffer=CHUNK,
        input=True,
    )
    frames = []
    for _ in range(0, int(SAMPLE_RATE / CHUNK * duration)):
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)
    stream.stop_stream()
    stream.close()
    pa.terminate()

    wf = wave.open(filename, "wb")
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(2)
    wf.setframerate(SAMPLE_RATE)
    wf.writeframes(b"".join(frames))
    wf.close()
    print("Sample recorded.")
    return filename


def ensure_output_dir(path_str):
    output_dir = Path(path_str)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def launch_gui():
    """
    Lightweight GUI for common controls: record samples, process text, or process audio files.
    """
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox, ttk
    except Exception as exc:
        raise RuntimeError("Tkinter is required for --ui mode.") from exc

    root = tk.Tk()
    root.title("Voice Clone Control Panel")
    root.geometry("560x460")

    status_var = tk.StringVar(value="Ready.")
    audio_path_var = tk.StringVar(value="")
    voice_id_var = tk.StringVar(value="")
    text_var = tk.StringVar(value="")
    record_duration_var = tk.StringVar(value="5")
    output_dir_var = tk.StringVar(value="samples")
    no_tts_var = tk.BooleanVar(value=False)

    def set_status(message):
        status_var.set(message)

    def pick_audio_file():
        path = filedialog.askopenfilename(
            title="Select audio file",
            filetypes=[("Audio Files", "*.wav *.mp3 *.m4a *.flac"), ("All Files", "*.*")],
        )
        if path:
            audio_path_var.set(path)

    def run_text():
        text = text_var.get().strip()
        if not text:
            messagebox.showwarning("Missing text", "Enter text to synthesize/analyze.")
            return
        if not os.getenv("ELEVENLABS_API_KEY") and not no_tts_var.get():
            messagebox.showwarning(
                "Missing API key",
                "Set ELEVENLABS_API_KEY or enable No TTS.",
            )
            return

        def task():
            set_status("Processing text...")
            process_text(text, voice_id=voice_id_var.get().strip() or None, enable_tts=not no_tts_var.get())
            set_status("Text processing complete.")

        threading.Thread(target=task, daemon=True).start()

    def run_audio_file():
        audio_path = audio_path_var.get().strip()
        if not audio_path:
            messagebox.showwarning("Missing audio file", "Select an audio file to process.")
            return
        if not os.path.exists(audio_path):
            messagebox.showerror("Missing file", "Selected audio file does not exist.")
            return
        if not os.getenv("ELEVENLABS_API_KEY") and not no_tts_var.get():
            messagebox.showwarning(
                "Missing API key",
                "Set ELEVENLABS_API_KEY or enable No TTS.",
            )
            return

        def task():
            set_status("Processing audio file...")
            process_audio_file(
                audio_path,
                voice_id=voice_id_var.get().strip() or None,
                enable_tts=not no_tts_var.get(),
            )
            set_status("Audio file processing complete.")

        threading.Thread(target=task, daemon=True).start()

    def record_voice_sample():
        try:
            duration = float(record_duration_var.get())
        except ValueError:
            messagebox.showwarning("Invalid duration", "Enter a numeric duration in seconds.")
            return
        output_dir = ensure_output_dir(output_dir_var.get().strip() or "samples")
        filename = output_dir / f"sample_{int(time.time())}.wav"

        def task():
            set_status("Recording sample...")
            try:
                record_sample(duration=duration, filename=str(filename))
            except RuntimeError as exc:
                messagebox.showerror("Recording error", str(exc))
                set_status("Recording failed.")
                return
            set_status(f"Sample recorded: {filename}")

        threading.Thread(target=task, daemon=True).start()

    main_frame = ttk.Frame(root, padding=12)
    main_frame.pack(fill="both", expand=True)

    ttk.Label(main_frame, text="Voice ID (optional):").grid(row=0, column=0, sticky="w")
    ttk.Entry(main_frame, textvariable=voice_id_var, width=45).grid(row=0, column=1, sticky="w")

    ttk.Checkbutton(main_frame, text="No TTS", variable=no_tts_var).grid(row=0, column=2, sticky="w")

    ttk.Label(main_frame, text="Text input:").grid(row=1, column=0, sticky="w")
    ttk.Entry(main_frame, textvariable=text_var, width=45).grid(row=1, column=1, sticky="w")
    ttk.Button(main_frame, text="Run Text", command=run_text).grid(row=1, column=2, sticky="w")

    ttk.Label(main_frame, text="Audio file:").grid(row=2, column=0, sticky="w")
    ttk.Entry(main_frame, textvariable=audio_path_var, width=45).grid(row=2, column=1, sticky="w")
    ttk.Button(main_frame, text="Browse", command=pick_audio_file).grid(row=2, column=2, sticky="w")
    ttk.Button(main_frame, text="Run Audio", command=run_audio_file).grid(row=3, column=2, sticky="w")

    ttk.Separator(main_frame, orient="horizontal").grid(row=4, column=0, columnspan=3, sticky="ew", pady=10)

    ttk.Label(main_frame, text="Record sample duration (s):").grid(row=5, column=0, sticky="w")
    ttk.Entry(main_frame, textvariable=record_duration_var, width=10).grid(row=5, column=1, sticky="w")

    ttk.Label(main_frame, text="Output directory:").grid(row=6, column=0, sticky="w")
    ttk.Entry(main_frame, textvariable=output_dir_var, width=45).grid(row=6, column=1, sticky="w")
    ttk.Button(main_frame, text="Record Sample", command=record_voice_sample).grid(row=6, column=2, sticky="w")

    ttk.Label(main_frame, textvariable=status_var, foreground="blue").grid(
        row=7, column=0, columnspan=3, sticky="w", pady=12
    )

    for idx in range(3):
        main_frame.columnconfigure(idx, weight=1)

    root.mainloop()
def is_silent(data, threshold=500):
    """
    Determine whether a PCM16 audio frame is silent by comparing its root-mean-square (RMS) amplitude to a threshold.
    
    Parameters:
        data (bytes): Raw audio bytes containing 16-bit PCM samples (int16).
        threshold (float): RMS amplitude cutoff; returns `True` when computed RMS is less than this value.
    
    Returns:
        bool: `True` if the frame's RMS is less than `threshold`, `False` otherwise.
    """
    audio_data = np.frombuffer(data, dtype=np.int16)
    rms = np.sqrt(np.mean(audio_data ** 2))
    return rms < threshold


def get_device_and_model(model_name=None):
    """
    Selects and loads a Whisper transcription model appropriate for the current hardware.
    
    The function chooses a model variant based on whether a CUDA GPU is available, loads that Whisper model onto the selected device, and returns the ready-to-use model instance.
    
    Returns:
        model: A Whisper model instance loaded onto the selected device (GPU if available, otherwise CPU).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    preferred_model = model_name or ("small.en" if device == "cuda" else "base")
    download_root = os.getenv("WHISPER_MODEL_DIR")
    print(f"Loading Whisper model '{preferred_model}' on {device} for low latency...")
    try:
        model = whisper.load_model(
            preferred_model,
            device=device,
            download_root=download_root,
        )
    except Exception as exc:
        raise RuntimeError(
            "Failed to load Whisper model. Ensure network access to download model "
            "weights or set WHISPER_MODEL_DIR to a directory containing the "
            "pre-downloaded Whisper models."
        ) from exc
    return model


def analyze_mood(text):
    """
    Determine overall mood and return the VADER compound sentiment score.
    
    Computes VADER polarity scores for the supplied text and maps the compound score to a mood label using thresholds: compound >= 0.3 -> "positive", compound <= -0.3 -> "negative", otherwise "neutral".
    
    Parameters:
        text (str): Text to analyze for sentiment.
    
    Returns:
        tuple: (mood, compound) where `mood` is one of "positive", "negative", or "neutral", and `compound` is the VADER compound score as a float.
    """
    scores = sentiment_analyzer.polarity_scores(text)
    compound = scores.get("compound", 0)
    if compound >= 0.3:
        mood = "positive"
    elif compound <= -0.3:
        mood = "negative"
    else:
        mood = "neutral"
    return mood, compound


def adaptive_voice_settings(mood, style=0.15, speaker_boost=True):
    """
    Map a detected mood to ElevenLabs voice synthesis settings.
    
    Parameters:
        mood (str): Detected mood label, expected values include "positive", "negative", or "neutral".
    
    Returns:
        dict: A preset containing `stability` and `similarity_boost` values for the given mood. If `mood` is unrecognized, returns the "neutral" preset.
    """
    presets = {
        "positive": {"stability": 0.82, "similarity_boost": 0.86},
        "negative": {"stability": 0.68, "similarity_boost": 0.78},
        "neutral": {"stability": 0.74, "similarity_boost": 0.82},
    }
    settings = presets.get(mood, presets["neutral"]).copy()
    settings["style"] = style
    settings["use_speaker_boost"] = speaker_boost
    return settings


def clean_transcript_text(text):
    """
    Normalize transcript text to reduce synthesis artifacts.
    """
    return " ".join(text.split()).strip()


def extract_word_timestamps(result):
    """
    Extract per-word timing entries from a Whisper transcription result.
    
    Parameters:
        result (dict): Whisper transcription output containing a "segments" list; each segment may include a "words" list with per-word timing dictionaries.
    
    Returns:
        list: A list of dictionaries, each with keys:
            - "word" (str): The word text (trimmed).
            - "start" (float|None): Word start time in seconds, or None if unavailable.
            - "end" (float|None): Word end time in seconds, or None if unavailable.
    """
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

def real_time_record(
    audio_queue,
    stop_event,
    silence_threshold=500,
    min_buffer_chunks=MIN_BUFFER_CHUNKS,
    max_buffer_chunks=240,
    enable_noise_reduction=True,
    enable_agc=True,
    target_rms=DEFAULT_TARGET_RMS,
    input_device_index=None,
    calibrate_seconds=0.0,
):
    """
    Continuously capture live audio, split utterances by silence, apply noise reduction, save each clip to a temporary WAV file, and enqueue the filename for downstream processing.
    
    This function runs until stop_event is set. When a sustained silence follows recorded audio, the buffered audio is denoised, written to a temporary 16-bit WAV file, and the file path is put onto audio_queue.
    
    Parameters:
        audio_queue (queue.Queue): Queue to receive paths of created WAV files for downstream consumers.
        stop_event (threading.Event): Event used to signal shutdown; recording stops when this event is set.
    """
    pyaudio = _require_pyaudio()
    pa = pyaudio.PyAudio()
    stream = pa.open(
        format=getattr(pyaudio, SAMPLE_FORMAT),
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        frames_per_buffer=CHUNK,
        input=True,
        input_device_index=input_device_index,
    )
    print("Listening continuously... Speak and pause to process.")
    if calibrate_seconds > 0:
        noise_floor = calibrate_noise_floor(stream, CHUNK, seconds=calibrate_seconds)
        if noise_floor:
            silence_threshold = max(silence_threshold, int(noise_floor * 1.8))
            print(f"Calibrated silence threshold: {silence_threshold:.0f} RMS")
    buffer = []
    silence_count = 0
    while not stop_event.is_set():
        data = stream.read(CHUNK, exception_on_overflow=False)
        buffer.append(data)
        if is_silent(data, threshold=silence_threshold):
            silence_count += 1
        else:
            silence_count = 0
        if silence_count > SILENCE_CHUNKS and len(buffer) > min_buffer_chunks:
            audio_data = b"".join(buffer)
            audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
            if enable_agc:
                audio_np = normalize_audio(audio_np, target_rms=target_rms)
            if enable_noise_reduction:
                audio_np = nr.reduce_noise(y=audio_np, sr=SAMPLE_RATE)
            filename = f"temp_{time.time()}.wav"
            wf = wave.open(filename, "wb")
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes((audio_np.astype(np.int16)).tobytes())
            wf.close()
            if audio_queue.full():
                try:
                    stale = audio_queue.get_nowait()
                    if os.path.exists(stale):
                        os.remove(stale)
                except queue.Empty:
                    pass
            audio_queue.put(filename)
            buffer = []
            silence_count = 0
        if len(buffer) > max_buffer_chunks:
            buffer = buffer[-max_buffer_chunks:]
            silence_count = min(silence_count, SILENCE_CHUNKS)
    stream.stop_stream()
    stream.close()
    pa.terminate()


def log_transcript(text, mood, compound, word_timings):
    """
    Append a timestamped transcript entry to transcripts.txt containing the transcribed text, detected mood and compound score, and optional per-word timings.
    
    Parameters:
        text (str): The transcribed text to record.
        mood (str): The mood label determined for the text (e.g., "positive", "negative", "neutral").
        compound (float): The VADER compound sentiment score associated with the text.
        word_timings (list[dict] | None): Optional list of word timing dictionaries with keys 'word', 'start', and 'end'; when provided, each word is logged as `word@start-end` with times in seconds.
    """
    timestamp = time.ctime()
    with open("transcripts.txt", "a") as f:
        f.write(f"{timestamp} | mood={mood} ({compound:.3f}) | text={text}\n")
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


def process_audio(
    audio_queue,
    voice_id,
    stop_event,
    model_name=None,
    language="en",
    word_timestamps=True,
    beam_size=5,
    best_of=5,
    tts_model=DEFAULT_TTS_MODEL,
    voice_style=0.15,
    speaker_boost=True,
    optimize_streaming_latency=None,
):
    """
    Process audio files from a queue: transcribe each file, detect mood and per-word timings, synthesize and play a mood-adaptive cloned voice, and persist a transcript.
    
    Parameters:
        audio_queue (queue.Queue): Queue that yields file paths to audio files to process.
        voice_id (str): ElevenLabs voice identifier used for synthesized playback.
        stop_event (threading.Event): Event that, when set, causes the processing loop to exit.
    
    Behavior:
        - Continuously reads audio file paths from `audio_queue` until `stop_event` is set.
        - For each audio file, obtains a transcription, determines overall mood and compound score, extracts per-word timestamps, prints transcription and timing information to stdout, synthesizes and plays the cloned voice using the detected mood, and logs the transcript and timings.
        - Deletes each processed audio file after handling it.
    """
    try:
        model = get_device_and_model(model_name=model_name)
    except RuntimeError as exc:
        print(exc)
        stop_event.set()
        return
    while not stop_event.is_set():
        audio_file = None
        try:
            audio_file = audio_queue.get(timeout=1)
            result = model.transcribe(
                audio_file,
                language=language,
                word_timestamps=word_timestamps,
                fp16=torch.cuda.is_available(),
                beam_size=beam_size,
                best_of=best_of,
            )
            text = clean_transcript_text(result["text"])
            if not text:
                continue
            mood, compound = analyze_mood(text)
            word_timings = extract_word_timestamps(result)
            print(f"You said: {text}")
            print(f"Mood detected: {mood} (compound={compound:.3f})")
            if word_timings:
                print("Word timings:")
                for word in word_timings:
                    print(f"  {word['word']} :: {word['start']:.2f}s -> {word['end']:.2f}s")
            text_to_cloned_voice(
                text,
                voice_id,
                mood,
                tts_model=tts_model,
                voice_style=voice_style,
                speaker_boost=speaker_boost,
                optimize_streaming_latency=optimize_streaming_latency,
            )
            log_transcript(text, mood, compound, word_timings)
        except queue.Empty:
            continue
        finally:
            if audio_file and os.path.exists(audio_file):
                os.remove(audio_file)


def text_to_cloned_voice(
    text,
    voice_id="21m00Tcm4TlvDq8ikWAM",
    mood="neutral",
    tts_model=DEFAULT_TTS_MODEL,
    voice_style=0.15,
    speaker_boost=True,
    optimize_streaming_latency=None,
):
    """
    Generate and play cloned-voice audio using ElevenLabs with mood-adaptive voice settings.
    
    Parameters:
        text (str): The text to synthesize into speech.
        voice_id (str): ElevenLabs voice identifier to use for synthesis. Defaults to the common cloned-voice ID.
        mood (str): Mood label that adjusts voice synthesis presets; expected values include "positive", "negative", and "neutral".
    """
    settings = adaptive_voice_settings(mood, style=voice_style, speaker_boost=speaker_boost)
    generate_kwargs = {
        "text": text,
        "voice": voice_id,
        "model_id": tts_model,
        "voice_settings": settings,
    }
    if optimize_streaming_latency is not None:
        generate_kwargs["optimize_streaming_latency"] = optimize_streaming_latency
    audio = client.generate(
        **generate_kwargs,
    )
    print(
        f"Playing cloned voice with settings: stability={settings['stability']}, "
        f"similarity={settings['similarity_boost']} (mood={mood})"
    )
    # Stream without writing to disk for minimal latency; virtual mic should capture playback
    play(audio)


def process_text(
    text,
    voice_id=None,
    enable_tts=True,
    tts_model=DEFAULT_TTS_MODEL,
    voice_style=0.15,
    speaker_boost=True,
    optimize_streaming_latency=None,
):
    """
    Analyze and optionally synthesize a provided text string.

    Parameters:
        text (str): Text input to analyze and optionally synthesize.
        voice_id (str | None): ElevenLabs voice identifier.
        enable_tts (bool): When False, skips ElevenLabs synthesis/playback.
    """
    text = clean_transcript_text(text)
    mood, compound = analyze_mood(text)
    print(f"Text input: {text}")
    print(f"Mood detected: {mood} (compound={compound:.3f})")
    if enable_tts:
        text_to_cloned_voice(
            text,
            voice_id or "21m00Tcm4TlvDq8ikWAM",
            mood,
            tts_model=tts_model,
            voice_style=voice_style,
            speaker_boost=speaker_boost,
            optimize_streaming_latency=optimize_streaming_latency,
        )
    log_transcript(text, mood, compound, [])


def process_audio_file(
    audio_file,
    voice_id=None,
    enable_tts=True,
    model_name=None,
    language="en",
    word_timestamps=True,
    beam_size=5,
    best_of=5,
    tts_model=DEFAULT_TTS_MODEL,
    voice_style=0.15,
    speaker_boost=True,
    optimize_streaming_latency=None,
):
    """
    Transcribe a single audio file, optionally synthesize a cloned voice, and log the transcript.

    Parameters:
        audio_file (str): Path to a WAV/MP3/etc file supported by Whisper.
        voice_id (str | None): ElevenLabs voice identifier. Uses default if None and TTS enabled.
        enable_tts (bool): When False, skips ElevenLabs synthesis/playback.
    """
    try:
        model = get_device_and_model(model_name=model_name)
    except RuntimeError as exc:
        print(exc)
        return
    result = model.transcribe(
        audio_file,
        language=language,
        word_timestamps=word_timestamps,
        fp16=torch.cuda.is_available(),
        beam_size=beam_size,
        best_of=best_of,
    )
    text = clean_transcript_text(result["text"])
    if not text:
        print("No speech detected in the provided audio file.")
        return
    mood, compound = analyze_mood(text)
    word_timings = extract_word_timestamps(result)
    print(f"You said: {text}")
    print(f"Mood detected: {mood} (compound={compound:.3f})")
    if word_timings:
        print("Word timings:")
        for word in word_timings:
            print(f"  {word['word']} :: {word['start']:.2f}s -> {word['end']:.2f}s")
    if enable_tts:
        text_to_cloned_voice(
            text,
            voice_id or "21m00Tcm4TlvDq8ikWAM",
            mood,
            tts_model=tts_model,
            voice_style=voice_style,
            speaker_boost=speaker_boost,
            optimize_streaming_latency=optimize_streaming_latency,
        )
    log_transcript(text, mood, compound, word_timings)


def main():
    """
    Start the interactive real-time speech-to-cloned-voice application.
    
    Performs environment validation, optionally clones a new ElevenLabs voice from recorded samples or accepts an existing voice ID,
    and then runs the real-time pipeline that records audio, transcribes with Whisper, analyzes mood, synthesizes cloned voice output,
    and logs transcripts with per-word timestamps. Lists available voices when possible, instructs about virtual mic setup, launches
    recording and processing threads, and waits for Ctrl+C to stop and clean up resources.
    """
    parser = argparse.ArgumentParser(description="Real-time voice cloning app.")
    parser.add_argument("--audio-file", help="Process a single audio file instead of live mic.")
    parser.add_argument(
        "--list-input-devices",
        action="store_true",
        help="List available audio input devices and exit.",
    )
    parser.add_argument("--input-device", type=int, help="Input device index for PyAudio.")
    parser.add_argument(
        "--silence-threshold",
        type=int,
        default=500,
        help="Silence RMS threshold for utterance splitting.",
    )
    parser.add_argument(
        "--min-utterance-seconds",
        type=float,
        default=2.5,
        help="Minimum utterance length before sending to Whisper.",
    )
    parser.add_argument(
        "--max-utterance-seconds",
        type=float,
        default=15.0,
        help="Maximum buffered utterance length before trimming.",
    )
    parser.add_argument(
        "--calibrate-seconds",
        type=float,
        default=0.0,
        help="Seconds to sample background noise for adaptive silence thresholding.",
    )
    parser.add_argument(
        "--noise-reduction",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable noise reduction on captured audio.",
    )
    parser.add_argument(
        "--agc",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable automatic gain control (AGC) for stable loudness.",
    )
    parser.add_argument(
        "--target-rms",
        type=float,
        default=DEFAULT_TARGET_RMS,
        help="Target RMS level for AGC.",
    )
    parser.add_argument(
        "--max-queue",
        type=int,
        default=6,
        help="Maximum number of buffered utterances to process.",
    )
    parser.add_argument(
        "--no-tts",
        action="store_true",
        help="Skip ElevenLabs synthesis/playback (useful for offline testing).",
    )
    parser.add_argument(
        "--voice-id",
        help="ElevenLabs voice ID to use (defaults to ElevenLabs demo voice).",
    )
    parser.add_argument(
        "--text",
        help="Analyze/synthesize a provided text string without Whisper.",
    )
    parser.add_argument(
        "--whisper-model",
        default=None,
        help="Override Whisper model name (e.g., base, small.en).",
    )
    parser.add_argument(
        "--language",
        default="en",
        help="Language code for transcription.",
    )
    parser.add_argument("--beam-size", type=int, default=5, help="Whisper beam size.")
    parser.add_argument("--best-of", type=int, default=5, help="Whisper best-of samples.")
    parser.add_argument(
        "--word-timestamps",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable per-word timestamps from Whisper.",
    )
    parser.add_argument(
        "--tts-model-id",
        default=DEFAULT_TTS_MODEL,
        help="ElevenLabs model ID for synthesis.",
    )
    parser.add_argument(
        "--voice-style",
        type=float,
        default=0.15,
        help="ElevenLabs style setting (0.0-1.0).",
    )
    parser.add_argument(
        "--speaker-boost",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable ElevenLabs speaker boost.",
    )
    parser.add_argument(
        "--optimize-streaming-latency",
        type=int,
        default=None,
        help="ElevenLabs streaming latency optimization level (0-4).",
    )
    parser.add_argument(
        "--ui",
        action="store_true",
        help="Launch a lightweight GUI for common controls.",
    )
    args = parser.parse_args()

    print("Powerful Real-Time Speech to Cloned Voice App")
    print("Whisper STT + ElevenLabs cloned voice with adaptive mood and per-word tracking")
    print("GPU-aware, noise-reduced, and tuned for the lowest possible call latency")
    if args.ui:
        launch_gui()
        return
    if args.list_input_devices:
        list_input_devices()
        return
    if not os.getenv("ELEVENLABS_API_KEY") and not args.no_tts:
        print("Please set ELEVENLABS_API_KEY environment variable or run with --no-tts.")
        exit(1)
    if args.text:
        process_text(
            args.text,
            voice_id=args.voice_id,
            enable_tts=not args.no_tts,
            tts_model=args.tts_model_id,
            voice_style=args.voice_style,
            speaker_boost=args.speaker_boost,
            optimize_streaming_latency=args.optimize_streaming_latency,
        )
        return
    if args.audio_file:
        process_audio_file(
            args.audio_file,
            voice_id=args.voice_id,
            enable_tts=not args.no_tts,
            model_name=args.whisper_model,
            language=args.language,
            word_timestamps=args.word_timestamps,
            beam_size=args.beam_size,
            best_of=args.best_of,
            tts_model=args.tts_model_id,
            voice_style=args.voice_style,
            speaker_boost=args.speaker_boost,
            optimize_streaming_latency=args.optimize_streaming_latency,
        )
        return

    try:
        _require_pyaudio()
    except RuntimeError as exc:
        print(exc)
        exit(1)
    try:
        voices_response = client.voices.get_all()
        print("Available voices for cloning:")
        for voice in voices_response.voices:
            print(f"{voice.voice_id}: {voice.name}")
    except Exception as e:
        print(f"Could not fetch voices: {e}. Proceeding without list.")

    clone_new = input("Do you want to clone a new voice? (y/n): ").strip().lower()
    if clone_new == "y":
        voice_name = input("Enter name for the new voice: ").strip()
        print("Record 3-5 samples of the voice to clone (speak clearly, 5-10 seconds each).")
        samples = []
        for i in range(3):
            input(f"Press Enter to record sample {i+1}...")
            filename = record_sample()
            samples.append(filename)
        try:
            voice = client.voices.add(name=voice_name, files=samples)
            voice_id = voice.voice_id
            print(f"New voice cloned: {voice_id}")
        except Exception as e:
            print(f"Failed to clone voice: {e}")
            exit(1)
        for f in samples:
            os.remove(f)
    else:
        voice_id = input("Enter ElevenLabs voice ID for specific voice cloning: ").strip()
        if not voice_id:
            print("Voice ID is required for cloning.")
            exit(1)

    print("For Linux: Ensure virtual mic is set up with PulseAudio.")
    print("For Windows: Install VB-Audio Virtual Cable, set default output to 'CABLE Input'.")
    print("Mood tracking and per-word timestamps are enabled for every utterance.")

    min_chunks = max(1, int(SAMPLE_RATE / CHUNK * args.min_utterance_seconds))
    max_chunks = max(min_chunks + 1, int(SAMPLE_RATE / CHUNK * args.max_utterance_seconds))
    audio_queue = queue.Queue(maxsize=args.max_queue)
    stop_event = threading.Event()
    record_thread = threading.Thread(
        target=real_time_record,
        args=(audio_queue, stop_event),
        kwargs={
            "silence_threshold": args.silence_threshold,
            "min_buffer_chunks": min_chunks,
            "max_buffer_chunks": max_chunks,
            "enable_noise_reduction": args.noise_reduction,
            "enable_agc": args.agc,
            "target_rms": args.target_rms,
            "input_device_index": args.input_device,
            "calibrate_seconds": args.calibrate_seconds,
        },
    )
    process_thread = threading.Thread(
        target=process_audio,
        args=(audio_queue, voice_id, stop_event),
        kwargs={
            "model_name": args.whisper_model,
            "language": args.language,
            "word_timestamps": args.word_timestamps,
            "beam_size": args.beam_size,
            "best_of": args.best_of,
            "tts_model": args.tts_model_id,
            "voice_style": args.voice_style,
            "speaker_boost": args.speaker_boost,
            "optimize_streaming_latency": args.optimize_streaming_latency,
        },
    )
    record_thread.start()
    process_thread.start()
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
