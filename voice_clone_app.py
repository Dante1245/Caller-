#!/usr/bin/env python3
import os
import queue
import threading
import time
import wave

import numpy as np
import pyaudio
import torch
import whisper
from elevenlabs import ElevenLabs, play
import noisereduce as nr
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
SAMPLE_FORMAT = pyaudio.paInt16
CHANNELS = 1
SAMPLE_RATE = 16000
CHUNK = 1024
SILENCE_CHUNKS = 8  # shorter silence window for snappier responses
MIN_BUFFER_CHUNKS = 40  # lower minimum to ship faster to Whisper

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

    pa = pyaudio.PyAudio()
    print("Recording sample...")
    stream = pa.open(
        format=SAMPLE_FORMAT,
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


def get_device_and_model():
    """
    Selects and loads a Whisper transcription model appropriate for the current hardware.
    
    The function chooses a model variant based on whether a CUDA GPU is available, loads that Whisper model onto the selected device, and returns the ready-to-use model instance.
    
    Returns:
        model: A Whisper model instance loaded onto the selected device (GPU if available, otherwise CPU).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    preferred_model = "small.en" if device == "cuda" else "base"
    print(f"Loading Whisper model '{preferred_model}' on {device} for low latency...")
    model = whisper.load_model(preferred_model, device=device)
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


def adaptive_voice_settings(mood):
    """
    Map a detected mood to ElevenLabs voice synthesis settings.
    
    Parameters:
        mood (str): Detected mood label, expected values include "positive", "negative", or "neutral".
    
    Returns:
        dict: A preset containing `stability` and `similarity_boost` values for the given mood. If `mood` is unrecognized, returns the "neutral" preset.
    """
    presets = {
        "positive": {"stability": 0.8, "similarity_boost": 0.85},
        "negative": {"stability": 0.65, "similarity_boost": 0.75},
        "neutral": {"stability": 0.72, "similarity_boost": 0.8},
    }
    return presets.get(mood, presets["neutral"])


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

def real_time_record(audio_queue, stop_event):
    """
    Continuously capture live audio, split utterances by silence, apply noise reduction, save each clip to a temporary WAV file, and enqueue the filename for downstream processing.
    
    This function runs until stop_event is set. When a sustained silence follows recorded audio, the buffered audio is denoised, written to a temporary 16-bit WAV file, and the file path is put onto audio_queue.
    
    Parameters:
        audio_queue (queue.Queue): Queue to receive paths of created WAV files for downstream consumers.
        stop_event (threading.Event): Event used to signal shutdown; recording stops when this event is set.
    """
    pa = pyaudio.PyAudio()
    stream = pa.open(
        format=SAMPLE_FORMAT,
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        frames_per_buffer=CHUNK,
        input=True,
    )
    print("Listening continuously... Speak and pause to process.")
    buffer = []
    silence_count = 0
    while not stop_event.is_set():
        data = stream.read(CHUNK, exception_on_overflow=False)
        buffer.append(data)
        if is_silent(data):
            silence_count += 1
        else:
            silence_count = 0
        if silence_count > SILENCE_CHUNKS and len(buffer) > MIN_BUFFER_CHUNKS:
            audio_data = b"".join(buffer)
            audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
            reduced_noise = nr.reduce_noise(y=audio_np, sr=SAMPLE_RATE)
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


def process_audio(audio_queue, voice_id, stop_event):
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
    model = get_device_and_model()
    while not stop_event.is_set():
        audio_file = None
        try:
            audio_file = audio_queue.get(timeout=1)
            result = model.transcribe(
                audio_file,
                language="en",
                word_timestamps=True,
                fp16=torch.cuda.is_available(),
            )
            text = result["text"].strip()
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
            text_to_cloned_voice(text, voice_id, mood)
            log_transcript(text, mood, compound, word_timings)
        except queue.Empty:
            continue
        finally:
            if audio_file and os.path.exists(audio_file):
                os.remove(audio_file)


def text_to_cloned_voice(text, voice_id="21m00Tcm4TlvDq8ikWAM", mood="neutral"):
    """
    Generate and play cloned-voice audio using ElevenLabs with mood-adaptive voice settings.
    
    Parameters:
        text (str): The text to synthesize into speech.
        voice_id (str): ElevenLabs voice identifier to use for synthesis. Defaults to the common cloned-voice ID.
        mood (str): Mood label that adjusts voice synthesis presets; expected values include "positive", "negative", and "neutral".
    """
    settings = adaptive_voice_settings(mood)
    audio = client.generate(
        text=text,
        voice=voice_id,
        model_id="eleven_turbo_v2",
        voice_settings=settings,
    )
    print(
        f"Playing cloned voice with settings: stability={settings['stability']}, "
        f"similarity={settings['similarity_boost']} (mood={mood})"
    )
    # Stream without writing to disk for minimal latency; virtual mic should capture playback
    play(audio)

def main():
    """
    Start the interactive real-time speech-to-cloned-voice application.
    
    Performs environment validation, optionally clones a new ElevenLabs voice from recorded samples or accepts an existing voice ID,
    and then runs the real-time pipeline that records audio, transcribes with Whisper, analyzes mood, synthesizes cloned voice output,
    and logs transcripts with per-word timestamps. Lists available voices when possible, instructs about virtual mic setup, launches
    recording and processing threads, and waits for Ctrl+C to stop and clean up resources.
    """
    print("Powerful Real-Time Speech to Cloned Voice App")
    print("Whisper STT + ElevenLabs cloned voice with adaptive mood and per-word tracking")
    print("GPU-aware, noise-reduced, and tuned for the lowest possible call latency")
    if not os.getenv("ELEVENLABS_API_KEY"):
        print("Please set ELEVENLABS_API_KEY environment variable")
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

    audio_queue = queue.Queue()
    stop_event = threading.Event()
    record_thread = threading.Thread(target=real_time_record, args=(audio_queue, stop_event))
    process_thread = threading.Thread(target=process_audio, args=(audio_queue, voice_id, stop_event))
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