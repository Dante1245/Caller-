# Caller

A desktop and command-line speech-to-voice pipeline: record a phrase, transcribe it
with local Whisper, then optionally synthesize it using a reference voice with
local XTTS or an ElevenLabs voice. Windows, macOS and Linux are supported by the
code; device compatibility and latency depend on your machine.

This is phrase-based processing, not simultaneous speech conversion. Model accuracy,
voice similarity and real-time performance are not guaranteed. The intent, mood,
and quality indicators are heuristics, not measurements of a person's emotions or
of voice-cloning accuracy. Use a voice you own or have permission to use.

## Install

Use **Python 3.12** for a new environment (supported range: 3.10–3.13). Install
FFmpeg first. On macOS, use `brew install ffmpeg python-tk@3.12`;
on Debian/Ubuntu, use `sudo apt install ffmpeg libportaudio2 python3-tk python3-venv`.
On Windows, install FFmpeg on PATH. Sounddevice supplies PortAudio binaries on Windows and macOS.
Tkinter is required only for the desktop panel.

```sh
git clone https://github.com/Dante1245/Caller-.git
cd Caller-
python3.12 -m venv .venv
source .venv/bin/activate
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Choose an installation:

```sh
# Transcription only (no voice service, API key, or XTTS required)
python -m pip install -r requirements.txt

# Local voice cloning: maintained Coqui package, which exposes TTS.api
python -m pip install -r requirements-local.txt

# Or cloud voice synthesis
python -m pip install -r requirements-elevenlabs.txt
```

For CUDA, install compatible PyTorch/torchaudio builds using the official PyTorch
instructions before the local requirements. Model weights download on first use;
review and accept any model license yourself. The application does not accept
third-party model terms automatically.

## Start with transcription

```sh
python voice_clone_app.py --self-check --run-mode test
python voice_clone_app.py --list-devices
python voice_clone_app.py --ui --run-mode test --model base.en
```

The self-check prints JSON and exits nonzero if a required check fails. It works
even when dependencies are missing. It checks installed imports and available
devices, not microphone permissions, downloaded models, API account access or an
actual call. `--help` also works without runtime dependencies.

In the studio, choose the microphone and playback device, then **Start session**.
**Stop** waits for the current inference or playback operation; it does not freeze
the window. Controls remain locked until workers exit, preventing overlapping
sessions. Closing the window also waits for active work to finish.

## Use a reference voice

```sh
python voice_clone_app.py --ui --speaker-wav my_voice.wav
# Or open the panel and use Choose audio… before selecting Live voice.
python voice_clone_app.py --ui

# Command line: explicitly request a microphone reference recording
python voice_clone_app.py --record-speaker --force-cpu
```

Use a clean recording with one speaker. Silent/empty references are rejected.
Files are read with libsndfile; WAV and FLAC are recommended. Support for compressed
formats depends on the installed decoder. Reference gain does not determine the
microphone's silence threshold. CLI mode calibrates ambient noise before listening;
the studio uses a fixed RMS threshold of 500.

## ElevenLabs

Set `ELEVENLABS_API_KEY` in your local environment; never commit it. Then:

```sh
python voice_clone_app.py --engine elevenlabs --elevenlabs-voice-id YOUR_VOICE_ID --ui
# Create a voice from your authorized sample (uploads the sample to ElevenLabs):
python voice_clone_app.py --engine elevenlabs --speaker-wav my_voice.wav --elevenlabs-clone-name MyVoice
```

Cloud mode sends recognized text to ElevenLabs. Cloning requires a suitable account
plan. Both backends respect the selected playback device. ElevenLabs uses the v2
SDK's text-to-speech and instant-voice-clone endpoints with 24 kHz PCM playback.
Selecting a local reference from the panel is available only for the local engine;
configure a cloud voice using the CLI options above.

## Route to a call

1. Install a virtual audio device: BlackHole on macOS, VB-CABLE on Windows, or a
   PulseAudio/PipeWire virtual sink on Linux.
2. In Caller, choose your **physical microphone** as input and the virtual device
   as playback. Input and output share the sounddevice device list;
   use `--list-devices` or the studio selectors.
3. In your call application, choose the corresponding virtual device/monitor as its
   microphone. Use headphones to avoid feeding synthesized output back into Caller.
4. Test in the call application's microphone meter before placing a call.

Caller does not install drivers, change system defaults or configure call apps.
For CLI routing, use `--device-index N --playback-device M`.

## Useful options

- `--run-mode test`: transcription and analytics only. No synthesis, cloning, or API key required.
- `--model tiny.en|base.en|small.en`: trade recognition quality for speed.
- `--language es` or `--auto-language`: select multilingual recognition; automatic model
  selection uses a multilingual model. Explicit `.en` models are rejected in these modes.
- `--performance-mode balanced|max|cpu` and `--force-cpu`: processing presets.
- `--no-noise-reduction`: reduce processing cost.
- `--silence-chunks N`: end a phrase after N silent chunks (each chunk is about 64 ms).
- `--min-buffer-chunks N`: minimum voiced chunks; defaults to 2–4 depending on preset.
- `--log-transcripts`: explicitly enable text logging. Off by default; recognized text
  still appears in the console and studio during a session.
- `--profile-name NAME`: letters, digits, underscores and hyphens only, up to 64 characters.

Capture keeps three chunks of pre-roll, bounds phrases to approximately 30 seconds,
and limits queued phrases to three. If processing falls behind, it drops a new clip
and displays a warning. Shutdown discards queued clips and an unfinished recording.
“Processing + playback latency” excludes capture time and time spent waiting in the queue.

## Local data

Data is saved in the current working directory: `voice_profiles.json`,
`normalized_speakers/`, `usage_report.json`, explicit `sample_*.wav` recordings, and
opt-in `transcripts.txt`. These are ignored by Git. JSON updates use atomic replacement;
a corrupted file produces an actionable error rather than silently erasing data.
Run only one Caller process per working directory to avoid concurrent report updates.
Audio clips awaiting transcription stay in memory, not temporary WAV files.

## Tests

```sh
python -m pip install -r requirements-dev.txt
python -m pytest
python -m ruff check .
python -m compileall -q .
```

Regression tests cover segmentation, overflow, startup failures, cleanup, diagnostic
behavior, storage and provider contracts using simulated audio/providers. They do
not establish live voice quality, real API access, or successful routing into a call.
See `VALIDATION.md` for the checks actually performed on this revision.
