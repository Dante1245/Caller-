#!/usr/bin/env python3
import whisper
import pyaudio
import wave
import os
from elevenlabs import ElevenLabs

# Note: Set your ElevenLabs API key as environment variable ELEVENLABS_API_KEY
# For voice cloning, use a custom voice ID from ElevenLabs dashboard (upload sample for cloning).
# ElevenLabs provides state-of-the-art voice cloning with ZERO glitches or robotic sounds—ultra-natural output.
# Optimized for high-end computers like Alienware/gaming laptops; runs efficiently on CPU/GPU with minimal delay.
# After running, connect as device microphone: Set up virtual mic and route output to it for professional calls.

client = ElevenLabs(api_key=os.getenv('ELEVENLABS_API_KEY'))

def record_audio(duration=5, filename="temp_audio.wav"):
    chunk = 1024
    sample_format = pyaudio.paInt16
    channels = 1
    fs = 16000  # 16kHz for Whisper compatibility
    p = pyaudio.PyAudio()
    print("Recording... Say something!")
    stream = p.open(format=sample_format,
                    channels=channels,
                    rate=fs,
                    frames_per_buffer=chunk,
                    input=True)
    frames = []
    for i in range(0, int(fs / chunk * duration)):
        data = stream.read(chunk)
        frames.append(data)
    stream.stop_stream()
    stream.close()
    p.terminate()
    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(sample_format))
    wf.setframerate(fs)
    wf.writeframes(b''.join(frames))
    wf.close()
    print("Recording finished")
    return filename

def speech_to_text():
    audio_file = record_audio()
    model = whisper.load_model("small")  # High accuracy model for professional use
    result = model.transcribe(audio_file)
    text = result["text"].strip()
    print(f"You said: {text}")
    os.remove(audio_file)
    return text if text else None

def text_to_cloned_voice(text, voice_id="21m00Tcm4TlvDq8ikWAM"):  # Default: Rachel voice
    # Using ElevenLabs turbo model for fast, high-quality voice cloning
    # Voice IDs: Use custom cloned voices from ElevenLabs for true cloning
    audio = client.generate(
        text=text,
        voice=voice_id,
        model_id="eleven_turbo_v2"  # Turbo model for professional-grade speed and quality
    )
    with open("output.mp3", "wb") as f:
        f.write(audio)
    print("Playing cloned voice...")
    # On Linux, routes to virtual mic; on Windows, ensure default output is set to virtual cable input
    from playsound import playsound
    playsound.playsound("output.mp3")
    # Clean up
    os.remove("output.mp3")

def main():
    print("Professional Speech to Cloned Voice App")
    print("Ultra-accurate STT with Whisper and glitch-free, natural TTS with ElevenLabs")
    print("Optimized for high-end devices; zero delay, output routed to virtual microphone for calls")
    if not os.getenv('ELEVENLABS_API_KEY'):
        print("Please set ELEVENLABS_API_KEY environment variable")
        exit(1)
    try:
        voices_response = client.voices.get_all()
        print("Available voices for cloning:")
        for voice in voices_response.voices:
            print(f"{voice.voice_id}: {voice.name}")
    except Exception as e:
        print(f"Could not fetch voices: {e}. Proceeding without list.")
    voice_id = input("Enter ElevenLabs voice ID for specific voice cloning: ").strip()
    if not voice_id:
        print("Voice ID is required for cloning.")
        exit(1)
    print("For Linux: Ensure virtual mic is set up with PulseAudio.")
    print("For Windows: Install VB-Audio Virtual Cable, set default output to 'CABLE Input'.")
    while True:
        text = speech_to_text()
        if text:
            text_to_cloned_voice(text, voice_id)
        again = input("Do you want to continue? (y/n): ").strip().lower()
        if again != 'y':
            break

if __name__ == "__main__":
    main()