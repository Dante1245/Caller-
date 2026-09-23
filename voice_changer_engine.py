"""Bounded, continuous PCM bridge to w-okada's source-tree REST API.

No STT, text rewriting, or fallback to the original microphone signal.
The model and inference dependencies remain in the upstream server process.
"""
from __future__ import annotations

import base64
import http.client
import json
import queue
import threading
import time
from dataclasses import dataclass
from urllib.parse import urlsplit


@dataclass(frozen=True)
class LiveConfig:
    url: str = "http://127.0.0.1:18888"
    sample_rate: int = 48000
    block_size: int = 8192
    timeout: float = 2.0

    def __post_init__(self):
        parsed = urlsplit(self.url)
        if (parsed.scheme not in ("http", "https") or
                parsed.hostname not in ("localhost", "127.0.0.1", "::1") or
                parsed.username or parsed.password or parsed.query or parsed.fragment or
                parsed.path not in ("", "/")):
            raise ValueError("Use a loopback server URL, e.g. http://127.0.0.1:18888")
        if self.sample_rate not in (24000, 48000):
            raise ValueError("Live sample rate must be 24000 or 48000 Hz")
        if self.block_size not in (2048, 4096, 8192, 16384):
            raise ValueError("Live block size must be 2048, 4096, 8192, or 16384")
        if not 0.1 <= self.timeout <= 5:
            raise ValueError("Server timeout must be between 0.1 and 5 seconds")


class VoiceChangerClient:
    def __init__(self, config: LiveConfig):
        self.config = config
        parsed = urlsplit(config.url)
        cls = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
        self.connection = cls(parsed.hostname, parsed.port, timeout=config.timeout)
        self.sequence = 0

    def close(self):
        self.connection.close()

    def request(self, path, body=None):
        payload = json.dumps(body).encode() if body is not None else None
        try:
            self.connection.request("POST" if body is not None else "GET", path, payload,
                                    {"Content-Type": "application/json"})
            response = self.connection.getresponse()
            data = response.read(2_000_001)
            if response.status != 200 or len(data) > 2_000_000:
                raise RuntimeError(f"Voice-changer returned HTTP {response.status} or an oversized response")
            result = json.loads(data)
            if not isinstance(result, dict):
                raise ValueError("Expected a JSON object")
            return result
        except (OSError, ValueError, http.client.HTTPException) as exc:
            self.close()
            raise RuntimeError(f"Voice-changer request failed at {self.config.url}{path}: {exc}") from exc

    def check(self):
        info = self.request("/info")
        if info.get("status") != "OK" or info.get("modelSlotIndex", -1) == -1:
            raise RuntimeError("Load and select a voice model in the voice-changer web UI first")
        if info.get("passThrough"):
            raise RuntimeError("Disable voice-changer pass-through before starting Caller")
        if info.get("serverAudioStated") or info.get("serverAudioStarted"):
            raise RuntimeError("Stop server audio mode; Caller will own microphone capture")
        for key in ("inputSampleRate", "outputSampleRate"):
            if info.get(key) != self.config.sample_rate:
                raise RuntimeError(f"Set voice-changer {key} to {self.config.sample_rate} Hz in its UI")
        return info

    def convert(self, pcm: bytes):
        if len(pcm) != self.config.block_size * 2:
            raise ValueError("Expected one complete mono int16 audio block")
        self.sequence += 1
        result = self.request("/test", {"timestamp": self.sequence,
                                       "buffer": base64.b64encode(pcm).decode("ascii")})
        if result.get("timestamp") != self.sequence:
            raise RuntimeError("Voice-changer returned a mismatched frame timestamp")
        try:
            output = base64.b64decode(result["changedVoiceBase64"], validate=True)
        except (KeyError, ValueError, TypeError) as exc:
            raise RuntimeError("Voice-changer returned invalid PCM data") from exc
        # The upstream unloaded-model/error sentinel is a single zero sample.
        if len(output) != len(pcm):
            raise RuntimeError("Voice-changer output length mismatch; check model and server settings")
        return output


def offer_latest(buffer, item):
    """Discard stale audio instead of accumulating call delay. Returns drop count."""
    dropped = 0
    while True:
        try:
            buffer.put_nowait(item)
            return dropped
        except queue.Full:
            try:
                buffer.get_nowait()
                dropped += 1
            except queue.Empty:
                pass


def run_live(config, stop, status, sd_module=None):
    import numpy as np
    if sd_module is None:
        import sounddevice as sd_module
    live = LiveConfig(config.vc_url, block_size=config.vc_block_size)
    if config.device_index is None or config.playback_device is None:
        raise ValueError("Choose explicit microphone and virtual-cable output devices for live conversion")
    if config.device_index == config.playback_device:
        raise ValueError("Input and output must be different devices to avoid an audio feedback loop")
    client = VoiceChangerClient(live)
    incoming, outgoing = queue.Queue(maxsize=2), queue.Queue(maxsize=2)
    counters = {"dropped": 0, "underflows": 0}
    silence = bytes(live.block_size * 2)

    def capture(indata, frames, timing, flags):
        if stop.is_set():
            return
        if flags:
            counters["dropped"] += 1
        counters["dropped"] += offer_latest(incoming, (time.monotonic(), bytes(indata)))

    def playback(outdata, frames, timing, flags):
        try:
            created, pcm = outgoing.get_nowait()
        except queue.Empty:
            created, pcm = 0, silence
            counters["underflows"] += 1
        if stop.is_set() or stop.muted.is_set() or time.monotonic() - created > 0.5:
            pcm = silence
        outdata[:] = pcm

    try:
        info = client.check()
        sd_module.check_input_settings(device=config.device_index, channels=1,
                                       dtype="int16", samplerate=live.sample_rate)
        sd_module.check_output_settings(device=config.playback_device, channels=1,
                                        dtype="int16", samplerate=live.sample_rate)
        # Establish model inference readiness before opening the microphone.
        client.convert(silence)
        if stop.is_set():
            return
        status.emit(f"Live conversion ready · model slot {info['modelSlotIndex']} · "
                    f"{live.block_size / live.sample_rate * 1000:.0f} ms blocks")
        with sd_module.RawOutputStream(device=config.playback_device, channels=1,
                samplerate=live.sample_rate, dtype="int16", blocksize=live.block_size,
                callback=playback), sd_module.RawInputStream(device=config.device_index,
                channels=1, samplerate=live.sample_rate, dtype="int16",
                blocksize=live.block_size, callback=capture):
            stop.ready.set()
            last_report = 0
            while not stop.is_set():
                try:
                    captured, pcm = incoming.get(timeout=0.1)
                except queue.Empty:
                    continue
                if time.monotonic() - captured > 0.5:
                    counters["dropped"] += 1
                    continue
                started = time.monotonic()
                converted = client.convert(pcm)
                elapsed = time.monotonic() - started
                if stop.is_set():
                    break
                if time.monotonic() - captured > 0.5:
                    counters["dropped"] += 1
                else:
                    # Suppress buffered speech while muted, including after unmute.
                    counters["dropped"] += offer_latest(
                        outgoing, (captured, silence if stop.muted.is_set() else converted))
                if time.monotonic() - last_report > 1:
                    from audio_metrics import measure_audio
                    metrics = measure_audio(np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768)
                    status.update_metrics(**metrics, dropped=counters["dropped"], conversion_ms=elapsed * 1000)
                    status.emit(f"Conversion {elapsed * 1000:.0f} ms · "
                                f"dropped {counters['dropped']} · output gaps {counters['underflows']}")
                    last_report = time.monotonic()
    finally:
        client.close()


def start_live_threads(config, status):
    stop = threading.Event()
    stop.ready, stop.muted, stop.failed = threading.Event(), threading.Event(), False

    def worker():
        try:
            run_live(config, stop, status)
        except Exception as exc:
            stop.failed = True
            status.emit(f"Live conversion stopped: {exc}")
        finally:
            stop.set()

    thread = threading.Thread(target=worker, daemon=True)
    # Preserve the existing UI/CLI lifecycle contract with an already-finished peer.
    peer = threading.Thread(target=lambda: None, daemon=True)
    peer.start()
    thread.start()
    return stop, peer, thread
