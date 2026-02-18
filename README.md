# Professional Speech-to-Cloned Voice App

This advanced Python application leverages cutting-edge AI for ultra-accurate speech-to-text transcription and fast local voice cloning, designed for professional communications on both regular CPU laptops and high-performance machines.

## Features
- **Real-Time Streaming STT**: Continuous listening with silence detection for instant processing—speak naturally without pauses.
- **Ultra-Accurate STT**: OpenAI Whisper with GPU acceleration, multi-language support, and noise reduction for precision in any environment.
- **Per-Word Insights**: Whisper word-level timestamps printed and logged for precise call analytics and debugging.
- **Mood-Adaptive Analytics**: VADER sentiment detects tone per utterance for smarter monitoring and transcription insights.
- **Local Voice Cloning**: Coqui XTTS v2 runs locally for high-quality, fast cloning without external APIs.
- **ElevenLabs Option**: Choose the ElevenLabs engine for cloud TTS if you prefer (API key required).
- **Auto-Tuned Voice Profiles**: Upload a speaker WAV and the app auto-calibrates gain + silence thresholds for the most natural match.
- **Normalized Speaker Audio**: The reference WAV is normalized and stored under `normalized_speakers/` for consistent cloning quality.
- **Voice Cloning Samples**: Record a short speaker WAV locally and use it as the cloning reference.
- **Self-Improvement Hints**: Optional `--auto-upgrade` mode emits tuning recommendations from live usage analytics.
- **Performance Presets**: Use `--performance-mode max` for lowest latency on high-end PCs, or `--performance-mode cpu` for CPU-friendly defaults tuned for stronger quality.
- **Low-Latency Playback**: Optional output streaming via `--playback-device` and `--playback-block-size`.
- **Self Check**: Run `--self-check` to validate dependencies, GPU visibility, and audio devices.
- **Virtual Microphone Routing**: Outputs to virtual mic (PulseAudio/Linux or VB-Cable/Windows) for seamless use in calls.
- **Transcript Logging**: Saves all conversations to transcripts.txt for review, including mood and word-level timing metadata.
- **Multi-Threaded Processing**: Concurrent recording and processing for ultra-low latency and efficiency.
- **Extra Powerful**: Optimized for both CPU-only machines and high-end hardware, scalable for professional or pentesting uses.
- **Lightweight Control UI**: Optional Tkinter control panel to start/stop the pipeline and monitor status.

## System Requirements
- **OS**: Linux (tested on Kali/Debian) or Windows (with VB-Audio Virtual Cable).
- **Hardware**: Works on CPU-only systems and also benefits from dedicated GPUs for faster real-time performance.
- **Python**: 3.13+ (install from python.org for Windows).
- **Models**: Coqui XTTS v2 (downloaded on first run).
- **System Tools**:
  - Linux: PulseAudio, mpg123, portaudio19-dev.
  - Windows: VB-Audio Virtual Cable (free virtual audio device).

## Installation
1. **Install System Dependencies**:
   ```
   sudo apt update
   sudo apt install pulseaudio pulseaudio-utils mpg123 portaudio19-dev
   ```

2. **Clone or Download Repository**:
   ```
   git clone https://github.com/yourusername/voice_clone_app.git
   cd voice_clone_app
   ```

3. **Install Python Dependencies**:
   ```
   pip install -r requirements.txt
   ```
   On Windows, if issues with pyaudio, install from wheel or use conda. For GPU acceleration, install PyTorch with CUDA if available.

4. **Set Up Virtual Microphone** (run once per session):
   ```
   pactl load-module module-null-sink sink_name=virtual_mic sink_properties=device.description=VirtualMic
   pactl set-default-source virtual_mic.monitor
   ```

### CPU-Only Installation Guide (Linux / Windows / macOS)
Use this path if you do not have a dedicated GPU.

1. **Create and activate a virtual environment**:
   ```
   python -m venv .venv
   ```
   Linux/macOS:
   ```
   source .venv/bin/activate
   ```
   Windows (PowerShell):
   ```
   .\.venv\Scripts\Activate.ps1
   ```

2. **Install CPU-only PyTorch first**:
   ```
   pip install --upgrade pip
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   ```

3. **Install remaining project dependencies**:
   ```
   pip install -r requirements.txt
   ```

4. **Run CPU mode for best quality/performance balance**:
   ```
   python3 voice_clone_app.py --engine local --speaker-wav ./my_voice.wav --performance-mode cpu --force-cpu
   ```

### Windows Setup
1. **Install VB-Audio Virtual Cable**:
   - Download from https://vb-audio.com/Cable/ (free).
   - Install and reboot if needed.

2. **Configure Audio**:
   - Go to Sound settings > Playback > Set "CABLE Input (VB-Audio Virtual Cable)" as default output.
   - In Recording, "CABLE Output (VB-Audio Virtual Cable)" will appear as mic input.
   - In your call app (Zoom), select "CABLE Output" as microphone.

## Usage
1. **Run the App**:
   ```
   python3 voice_clone_app.py
   ```
   Optional quick start with arguments:
   ```
   python3 voice_clone_app.py --engine local --speaker-wav ./my_voice.wav --model base --device-index 1 --profile-name my_voice
   python3 voice_clone_app.py --engine local --speaker-wav ./my_voice.wav --performance-mode cpu --force-cpu
   python3 voice_clone_app.py --engine elevenlabs --speaker-wav ./my_voice.wav --elevenlabs-clone-name MyClone
   ```

2. **Select Speaker Sample**:
   - For local XTTS, record a short sample with `--record-speaker` or pass `--speaker-wav` to point to an existing WAV.
   - For ElevenLabs, provide `--elevenlabs-voice-id` or pass `--speaker-wav` + `--elevenlabs-clone-name` to clone.
   - The app auto-tunes gain and silence thresholds and stores the profile in `voice_profiles.json` (local only).

3. **Real-Time Operation**:
   - App starts listening continuously.
   - Speak and pause; app transcribes, clones, and plays in real-time.
   - Press Ctrl+C to stop.

4. **For Calls**:
   - Select virtual mic in Zoom/Teams.
   - Cloned voice routes seamlessly.

5. **Optional Control UI**:
   ```
   python3 voice_clone_app.py --ui
   ```
   Use Start/Stop to control the pipeline while monitoring live status updates.

## Configuration
- **Whisper Model**: Pass `--model` to force a model (e.g. `--model base.en`), or let the app auto-pick. CPU mode now defaults to `base.en` for better transcription quality.
- **TTS Model**: Override XTTS with `--tts-model` if you want a different local model.
- **Noise Reduction**: Disable for speed with `--no-noise-reduction`.
- **Device Selection**: Use `--device-index` to bind to a specific input device (list indexes in app output).
- **Profiles**: Name the saved tuning profile with `--profile-name` (saved in `voice_profiles.json`).
- **Auto-Upgrade Hints**: Enable with `--auto-upgrade` to surface recommendations in `usage_report.json`.
- **Performance**: Use `--performance-mode max` for faster turn-taking on strong GPUs, or `--performance-mode cpu` / `--force-cpu` on CPU-only devices. CPU preset now keeps noise reduction enabled for cleaner call audio.
- **Playback Routing**: Use `--playback-device` to select output device, and `--playback-block-size` to tune stream buffering.
- **Engine Selection**: Use `--engine local` or `--engine elevenlabs` to pick the voice cloning backend.
- **Diagnostics**: Run `--self-check` to print a JSON readiness report.
- **Mood + Analytics**: Sentiment analysis runs automatically; transcripts include mood score and per-word timestamps for each turn.
- **Latency Tuning**: Adjust `--silence-chunks` or `--min-buffer-chunks` for faster/longer turns.

## Troubleshooting
- **No Audio**: Ensure virtual mic is set up and selected in call apps (Linux: PulseAudio; Windows: VB-Cable).
- **Model Download**: The first run may download XTTS model weights; ensure internet access.
- **Latency**: On lower-end devices, consider GPU acceleration (install CUDA PyTorch).
- **Windows pyaudio Issues**: Install from https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio or use conda.
- **Windows Audio Not Routing**: Confirm VB-Cable devices are default in Sound settings.

## License
MIT License - Free for personal/professional use.

## Contributing
Fork, improve, and submit PRs. Ensure tests on both CPU-only and GPU-enabled hardware when possible.

---

Built with HackerAI for flawless, professional voice cloning.
