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

## Feature upgrade validation (September 23, 2026 UTC)

This section supersedes the earlier UI-binding limitation and test count above.

- 49 regression tests now pass locally, plus Ruff, compilation, diff checks and
  `pip check`. New tests cover faster-whisper's lazy result adapter, uncertainty
  warnings, subtitle exports and overwrite protection, source-grounded text
  extraction, measured clipping, cancellation/mute, stale-queue drops, paginated
  voice discovery, reference conditioning-cache invalidation, and new CLI flags.
- Installed faster-whisper 1.2.1 / CTranslate2 and ran real `tiny.en` int8 CPU
  inference. It recognized the complete synthetic test sentence in 2.40 seconds
  after loading. This single sample does not establish a speed advantage.
- The CLI file-transcription path generated valid JSON, UTF-8 text and SRT files
  from real generated speech. The updated full live-processing worker was also
  exercised using prerecorded PCM input, without opening a microphone or TTS.
- Faster-whisper transcription diagnostics passed 10/10.
- Native studio UI testing succeeded using the Homebrew Python app and pre-existing
  bundled Tcl/Tk libraries. Verified tab navigation, diagnostics, file selection,
  export destination selection, busy/Ready state transitions, restored controls,
  transcript insights, and closing the window. Checked actual export files on disk.
- Reference recording, calibration, live microphone capture, acoustic echo behavior,
  XTTS synthesis, cloud synthesis/voice creation and real call routing are still
  not validated end to end. Reference cache and cancellation behavior use provider
  or device doubles in regression tests. No cloud credentials were used.
- Translation is wired to the local model's translation task and validates model
  selection, but no multilingual translation accuracy evaluation was performed.
- The 8-second reference-recording tool has regression coverage for cancellation;
  it was not used to capture the user's microphone during this upgrade.

The project still has no generative conversation-assistant service. Summaries and
candidate actions are explicitly labeled extractive English heuristics. Model
weights, capabilities, accounts and hardware impose limits that tests cannot remove.
