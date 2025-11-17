# Professional Speech-to-Cloned Voice App

This advanced Python application leverages cutting-edge AI for ultra-accurate speech-to-text transcription and flawless voice cloning, designed for professional communications on high-performance devices like Alienware or gaming laptops.

## Features
- **Real-Time Streaming STT**: Continuous listening with silence detection for instant processing—speak naturally without pauses.
- **Ultra-Accurate STT**: OpenAI Whisper with GPU acceleration, multi-language support, and noise reduction for precision in any environment.
- **Glitch-Free TTS**: ElevenLabs (turbo model) with stability/similarity settings for 100% natural, artifact-free voice cloning.
- **Voice Cloning**: Clone new voices by recording samples and uploading to ElevenLabs for custom voices.
- **Virtual Microphone Routing**: Outputs to virtual mic (PulseAudio/Linux or VB-Cable/Windows) for seamless use in calls.
- **Transcript Logging**: Saves all conversations to transcripts.txt for review.
- **Multi-Threaded Processing**: Concurrent recording and processing for ultra-low latency and efficiency.
- **Extra Powerful**: Optimized for high-end hardware, scalable for professional or pentesting uses.

## System Requirements
- **OS**: Linux (tested on Kali/Debian) or Windows (with VB-Audio Virtual Cable).
- **Hardware**: High-end CPU/GPU (e.g., Alienware, gaming laptops) for optimal Whisper/ElevenLabs performance.
- **Python**: 3.13+ (install from python.org for Windows).
- **APIs**: ElevenLabs API key (free tier available; premium for cloned voices).
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

4. **Set Environment Variables**:
   ```
   export ELEVENLABS_API_KEY='your-elevenlabs-api-key'
   ```

5. **Set Up Virtual Microphone** (run once per session):
   ```
   pactl load-module module-null-sink sink_name=virtual_mic sink_properties=device.description=VirtualMic
   pactl set-default-source virtual_mic.monitor
   ```

### Windows Setup
1. **Install VB-Audio Virtual Cable**:
   - Download from https://vb-audio.com/Cable/ (free).
   - Install and reboot if needed.

2. **Configure Audio**:
   - Go to Sound settings > Playback > Set "CABLE Input (VB-Audio Virtual Cable)" as default output.
   - In Recording, "CABLE Output (VB-Audio Virtual Cable)" will appear as mic input.
   - In your call app (Zoom), select "CABLE Output" as microphone.

3. **Set Environment Variable**:
   ```
   set ELEVENLABS_API_KEY=your-key
   ```

## Usage
1. **Run the App**:
   ```
   python3 voice_clone_app.py
   ```

2. **Select Voice**:
   - List available voices.
   - Option to clone a new voice by recording samples.
   - Enter voice ID or use cloned one.

3. **Real-Time Operation**:
   - App starts listening continuously.
   - Speak and pause; app transcribes, clones, and plays in real-time.
   - Press Ctrl+C to stop.

4. **For Calls**:
   - Select virtual mic in Zoom/Teams.
   - Cloned voice routes seamlessly.

2. **Select Voice**: The app fetches and lists available ElevenLabs voices. Enter the ID for your desired cloned voice (upload samples to ElevenLabs for custom cloning).

3. **Operate**:
   - Speak into your microphone; the app records 5 seconds, transcribes accurately, and generates cloned audio.
   - On Linux: Cloned voice routes to virtual mic.
   - On Windows: Cloned voice plays to default output (CABLE Input), routing to CABLE Output as mic.
   - In your call app (e.g., Zoom), select the virtual mic ("VirtualMic" on Linux, "CABLE Output" on Windows).
   - Loop for continuous use; press 'n' to stop.

4. **For Professional Calls**: The virtual mic acts as your "voice" in meetings—speak naturally, output is cloned seamlessly.

## Configuration
- **Recording Duration**: Edit `duration=5` in `record_audio()` for shorter/longer clips (affects latency).
- **Whisper Model**: Change to "base" for faster but less accurate STT, or "medium" for even higher precision (requires more resources).
- **TTS Model**: ElevenLabs turbo is optimized; switch to "eleven_monolingual_v1" if needed, but turbo is best for speed.

## Troubleshooting
- **No Audio**: Ensure virtual mic is set up and selected in call apps (Linux: PulseAudio; Windows: VB-Cable).
- **API Errors**: Check internet and API key validity.
- **Latency**: On lower-end devices, consider GPU acceleration (install CUDA PyTorch).
- **Voices Not Loading**: Verify ElevenLabs account has voices/clones.
- **Windows pyaudio Issues**: Install from https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio or use conda.
- **Windows Audio Not Routing**: Confirm VB-Cable devices are default in Sound settings.

## License
MIT License - Free for personal/professional use.

## Contributing
Fork, improve, and submit PRs. Ensure tests on high-end hardware.

---

Built with HackerAI for flawless, professional voice cloning.