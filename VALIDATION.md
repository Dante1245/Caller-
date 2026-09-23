# Validation of the reliability update

Validated locally on macOS / Apple Silicon with Python 3.12 on September 22, 2026.

## Passed

- 22 regression tests (`python -m pytest -q`). Tests simulate device failures,
  PCM stream chunks and cloud responses; no live cloud request is made.
- Ruff static checks, Python compilation and `git diff --check`.
- Installed shared + ElevenLabs dependencies; `pip check` found no broken requirements.
- Transcription self-check: 10 passed, 0 failed.
- Actual device enumeration: physical microphone, speakers and BlackHole were detected.
- Real CPU Whisper `tiny.en` inference on generated English speech: recognized
  “Hello, this is a test of the caller audio pipeline, the meeting is scheduled for
  tomorrow.” Standalone inference took 2.53 seconds; a subsequent full processing
  pipeline run took 1.10 seconds after model loading. These are individual smoke
  measurements, not a latency benchmark.
- Full transcription worker processed real audio, produced word timestamps and
  analytics, saved the usage report, marked its queue item complete, and did not
  create a transcript file or initialize TTS in test mode.
- ElevenLabs v2 SDK method signatures were checked against the installed 2.68.0
  package. Simulated PCM tests verify split sample boundaries and exact text.
- The control panel launched using a Python runtime with Tk 9 and entered the Tk
  event loop. UI automation could not bind to that unbundled Python process;
  visual layout and interactive Start/Stop were not independently verified.

## Still requires end-to-end validation

- Physical microphone capture and permission handling on the target device.
- A real ElevenLabs account request and voice clone, or downloaded XTTS model and
  an authorized reference sample. Neither voice backend was live-tested.
- Audible synthesis quality, output-to-virtual-microphone routing, and a real call.
- GUI interaction, Windows/Linux hardware testing, and sustained-session latency.
- Hosted CI execution; the workflow was added but a green hosted run is not assumed.

## Environment notes

The original PyAudio installation failed because native build prerequisites were
unavailable. Capture now uses sounddevice, already used by playback, removing that
build dependency. Shared and cloud requirements subsequently installed successfully.
Homebrew's Tk installation was blocked by the Mac's unaccepted Xcode license. No
license agreement was accepted; a pre-existing bundled Tk runtime was used to launch
and inspect the process. A standard installation should use Python with Tk enabled.
