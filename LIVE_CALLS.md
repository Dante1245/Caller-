# Live voice conversion for calls

Caller now has a continuous audio bridge to [w-okada/voice-changer](https://github.com/w-okada/voice-changer). This is audio-to-audio conversion, separate from the existing phrase-based XTTS and ElevenLabs modes. Caller does not dial telephone numbers: it supplies converted audio to your calling application's microphone through a virtual audio cable.

## Compatibility

The adapter implements the **source-tree REST API** inspected at commit `f1caf8e7c39fd0d6866202be27bf142790191a51`:

- `GET /info`: selected model, pass-through, audio mode and sample rates.
- `POST /test`: `{timestamp, buffer}` with base64 little-endian mono int16 PCM.
- Response: `{timestamp, changedVoiceBase64}` with one matching audio block.

Sources: [server route](https://github.com/w-okada/voice-changer/blob/f1caf8e7c39fd0d6866202be27bf142790191a51/server/restapi/MMVC_Rest_VoiceChanger.py), [official client](https://github.com/w-okada/voice-changer/blob/f1caf8e7c39fd0d6866202be27bf142790191a51/client/lib/src/client/ServerRestClient.ts), [developer setup](https://github.com/w-okada/voice-changer/blob/f1caf8e7c39fd0d6866202be27bf142790191a51/README_dev_en.md).

New VCClient v2 packaged releases are **not verified compatible**. A passing `/info` and silent-block conversion check is required. The upstream source developer setup documents Linux/WSL2 and says macOS is untested. macOS source inference is therefore not claimed working here. LLVC variable-length output is not supported. RVC-style equal-length output is required.

The repositories communicate through this API; upstream dependencies and models are isolated from Caller. No upstream source or model weights are redistributed. Upstream has its own license notices; voice model terms apply separately. An XTTS reference WAV does not become an RVC model automatically.

## Set up the voice engine

1. Install a compatible upstream source server in its own environment using its developer guide. Keep it local to this computer (loopback); Caller accepts only loopback URLs. The official sample command enables HTTPS; use trusted local TLS or configure local HTTP to match the URL below. Caller does not bypass certificate validation.
2. In the upstream web interface, load/select a trained RVC voice model you are authorized to use. Set both input and output to **48000 Hz**, disable **pass-through**, and stop its server/device audio capture. Caller handles capture and playback. Do not run the upstream browser microphone stream concurrently or change models/settings during a Caller session.
3. Install the lightweight bridge:

   ```sh
   python -m pip install -r requirements-live.txt
   python voice_clone_app.py --vc-check --vc-url http://127.0.0.1:18888
   python voice_clone_app.py --list-devices
   ```

   `--vc-check` sends a silent block to test the API/model path, without opening the microphone. It does not establish voice similarity or perceptual quality. Missing servers/models produce errors; there is no pass-through fallback.

## Route audio into a call

| Setting | macOS example | Windows example |
|---|---|---|
| Caller input | Physical microphone | Physical microphone |
| Caller output | BlackHole 2ch | CABLE Input |
| Calling app microphone | BlackHole 2ch | CABLE Output |
| Calling app speaker | Headphones | Headphones |

Use the actual device indices returned by `--list-devices`; they can change. Choose different explicit input/output devices. Keep call playback out of the virtual cable to avoid feedback. Caller leaves system default devices unchanged.

```sh
# Replace 2 and 1 with YOUR input and cable-output device indices.
python voice_clone_app.py --engine voice-changer --run-mode live \
  --device-index 2 --playback-device 1 \
  --vc-url http://127.0.0.1:18888 --vc-block-size 8192
```

The lightweight CLI needs no Whisper, XTTS, FFmpeg or cloud key. Ctrl+C stops it. To use the full studio, install `requirements.txt` and Tk, then run:

```sh
python voice_clone_app.py --ui --engine voice-changer --run-mode live
```

In **AI settings**, choose `voice-changer`, set the server URL and block size, and click **Check live server & model**. In **Audio & reference**, select Live mode and the explicit microphone/cable devices. Start, mute and stop use the existing controls. Reference WAV, sentiment, vocabulary and speech-speed settings belong to the other engines; live model/pitch controls remain in the upstream interface. Transcribe-only mode still uses Whisper.

## Latency and verification

At 48 kHz, 2048 / 4096 / 8192 / 16384 samples correspond to about 43 / 85 / 171 / 341 ms per block. These are **block durations, not measured end-to-end latency**. Model inference, capture, queueing and playback add delay. Start at 8192; smaller blocks need faster inference and may reduce quality.

The bridge bounds both queues to two blocks, discards audio older than 500 ms, emits silence on output starvation, and reports conversion time, dropped blocks and output gaps. Persistent gaps mean the chosen model/settings cannot keep up. It stops on invalid responses or server failure. Muting gates the output; no transcripts or microphone recordings are saved by this engine. The server receives the audio and may have its own recording options.

Before using a real call, make a local recording from the virtual cable and listen for intelligibility, voice identity, clipping and delay. Confirm mute and stop, then use the calling application's microphone test. End-to-end inference, voice quality and live calls remain unverified until a compatible server and authorized model are supplied.
