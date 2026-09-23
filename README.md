# Caller

Caller combines a desktop voice studio with three voice engines:

- **Live conversion:** continuous microphone → w-okada voice-changer → virtual microphone, preserving delivery without a transcription step.
- **Local XTTS:** phrase transcription and resynthesis from a reference sample.
- **ElevenLabs:** phrase transcription and cloud voice synthesis.

The live bridge requires a separate compatible w-okada server and a loaded voice
model. See [live-call setup and compatibility](LIVE_CALLS.md). Voice similarity and
latency depend on the model and hardware; use a voice you own or have permission to use.
Intent, mood and quality indicators are heuristics, not measures of voice similarity.

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
**Stop** waits for active inference and interrupts playback between audio blocks; it does not freeze
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
the studio starts at an RMS threshold of 500 and offers calibration and a manual threshold in AI Settings.

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
Local references are used by the local engine. Set the cloud voice ID and model in AI Settings, or use the CLI options above.

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

Capture keeps three chunks of pre-roll, bounds phrases to 30 seconds by default,
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


## Upgraded AI and audio tools

The studio has **Audio & reference**, **AI settings**, and **Tools** tabs. Change the
recognition engine, model, language (`en`, `es`, `fr`, or `auto`), vocabulary hint,
noise reduction, voice engine, cloud voice/model IDs, voice speed and silence threshold
before starting. During sessions, the input meter reports dBFS; phrase, drop and
processing-time counters show pipeline health. **Mute voice** suppresses synthesis
and interrupts output between blocks while keeping transcription active. An in-flight
cloud request may already have incurred cost before it is muted.

### Optional faster-whisper

```sh
python -m pip install -r requirements-fast.txt
python voice_clone_app.py --stt-backend faster-whisper --model base.en --force-cpu --ui --run-mode test
```

This backend uses CTranslate2, defaults to CPU int8, and enables its Silero speech
filter. `--compute-type float32` is an alternative on CPU; CUDA can also use float16
or int8_float16 when supported by its runtime. It needs separately converted model
weights and cannot load Whisper `.pt` files. Both engines download models on first use.
Speed and accuracy vary; the optional backend is not guaranteed to outperform Whisper.
See the [upstream backend documentation](https://github.com/SYSTRAN/faster-whisper).

### Recording transcription, translation and subtitles

```sh
python voice_clone_app.py --input-file meeting.wav --model base --auto-language --output exports/meeting
python voice_clone_app.py --input-file spanish.wav --model base --task translate --output exports/english
```

These commands never open a microphone or initialize a voice-synthesis engine.
They produce `.json`, `.txt` and `.srt` files. Existing exports are protected from
overwrite. Translation produces English text, requires a multilingual model, and
is available in CLI file mode only. The studio's **Transcribe a recording** tool
uses the current recognition settings and prompts for an export destination.
Use `--initial-prompt "Acme, project terminology"` as a recognition vocabulary hint;
it does not guarantee correct spelling.

Exports include source-based summaries, candidate actions, questions and keywords.
**Record an 8-second voice reference** saves a user-selected WAV and loads a normalized
local reference; Stop cancels the recording without saving a partial file. References
longer than 60 seconds are rejected before decoding.

**View conversation insights** offers the same extraction for up to the last 200
phrases in the current studio window and an explicit JSON export. These are English
heuristics, not a generative assistant, speaker diarization, or verified commitments.
Whisper uncertainty signals appear as warnings and are not calibrated confidence
scores. VADER sentiment is disabled for detected non-English text.

### Voice and diagnostic utilities

```sh
python voice_clone_app.py --inspect-reference my_voice.wav
python voice_clone_app.py --list-voices
python voice_clone_app.py --list-cloud-models
python voice_clone_app.py --self-check --stt-backend faster-whisper --run-mode test --diagnostics-output readiness.json
```

The two cloud-listing commands require an ElevenLabs key and contact that service.
They list account voices and currently available speech models without synthesizing
speech. Use `--elevenlabs-model MODEL_ID` and `--voice-speed 1.0` to select output.
Speed supports 0.7–1.2 for both voice engines.

XTTS reuses one reference's conditioning in memory per session, invalidates it when
the reference path/size/modification time changes, and does not serialize speaker
embeddings. This uses the [documented XTTS inference interface](https://coqui-tts.readthedocs.io/en/latest/models/xtts.html).
Reference checks warn about short, quiet or clipped samples. Normalization limits
gain to preserve peak headroom. Live quality hints use measured levels/clipping and
processing time; negative text sentiment no longer reduces the quality score.

`--max-clip-seconds` controls maximum phrase length (1–60 seconds).
`--max-queue-seconds` discards phrases that waited too long (10 seconds by default),
preventing old speech from playing long after it was spoken. Noise reduction runs
in the processing worker so it does not block microphone capture.
