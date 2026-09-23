import json
import queue
import struct
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import soundfile as sf

import voice_clone_app as app
from audio_capture import UtteranceBuffer, pcm_rms
from quality_ai import assess_call_quality
from storage import read_object, write_object


def pcm(value, count=1024):
    return struct.pack("<" + "h" * count, *([value] * count))


def test_loud_pcm_does_not_overflow():
    assert pcm_rms(pcm(30000)) == 30000
    assert pcm_rms(pcm(-32768)) == 32768
    assert pcm_rms(b"") == 0
    assert not app.is_silent(pcm(10000))


def test_silence_never_builds_unbounded_buffer():
    segmenter = UtteranceBuffer()
    for _ in range(10000):
        assert segmenter.feed(pcm(0)) is None
    assert not segmenter.frames
    assert len(segmenter.pre_roll) == 3


def test_short_phrase_and_preroll_are_retained():
    segmenter = UtteranceBuffer(silence_chunks=2, min_chunks=2)
    segmenter.feed(pcm(0))
    segmenter.feed(pcm(1000))
    segmenter.feed(pcm(1000))
    segmenter.feed(pcm(0))
    result = segmenter.feed(pcm(0))
    assert result == pcm(0) + pcm(1000) * 2 + pcm(0) * 2


def test_continuous_speech_is_capped():
    segmenter = UtteranceBuffer(max_chunks=5, min_chunks=2)
    for _ in range(4):
        assert segmenter.feed(pcm(1000)) is None
    assert len(segmenter.feed(pcm(1000))) == 5 * 2048
    assert not segmenter.frames


def test_noise_click_is_discarded():
    segmenter = UtteranceBuffer(min_chunks=3, silence_chunks=2)
    segmenter.feed(pcm(1000))
    segmenter.feed(pcm(0))
    assert segmenter.feed(pcm(0)) is None


def test_quality_module_executes():
    assert assess_call_quality(200, 120, "neutral", True).label == "excellent"
    assert assess_call_quality(2000, 200, "negative", False).label == "poor"
    with pytest.raises(ValueError):
        assess_call_quality(-1, 100, "neutral", True)


def test_atomic_storage_preserves_previous_value_on_failure(tmp_path):
    target = tmp_path / "report.json"
    write_object(target, {"sessions": 1})
    with pytest.raises(ValueError):
        write_object(target, {"value": float("nan")})
    assert read_object(target) == {"sessions": 1}
    assert list(tmp_path.iterdir()) == [target]


def test_corrupt_storage_has_actionable_error(tmp_path):
    target = tmp_path / "report.json"
    target.write_text("{broken")
    with pytest.raises(ValueError, match="Restore or rename"):
        read_object(target)


def test_profile_rejects_silence_and_path_escape(tmp_path, monkeypatch):
    import voice_profile as vp

    monkeypatch.setattr(vp, "NORMALIZED_DIR", tmp_path / "normalized")
    wav = tmp_path / "sample.wav"
    sf.write(wav, np.zeros(16000), 16000)
    with pytest.raises(ValueError, match="silent"):
        vp.analyze_speaker_wav(str(wav))
    sf.write(wav, np.sin(np.arange(16000) * 0.1) * 0.3, 16000)
    profile = vp.analyze_speaker_wav(str(wav))
    with pytest.raises(ValueError, match="profile name"):
        vp.normalize_speaker_wav(str(wav), profile, "../escape")


def test_worker_failure_stops_recorder_before_open(monkeypatch):
    def fail(*args):
        raise RuntimeError("model unavailable")

    recorder = Mock()
    monkeypatch.setattr(app, "process_audio", fail)
    monkeypatch.setattr(app, "real_time_record", recorder)
    bus = app.StatusBus()
    stop, record, process = app.start_threads(SimpleNamespace(), bus)
    record.join(2)
    process.join(2)
    assert not record.is_alive() and not process.is_alive()
    assert stop.is_set() and stop.failed
    recorder.assert_not_called()
    assert "model unavailable" in bus.queue.get_nowait()


def config(**overrides):
    values = dict(
        model_name="tiny.en",
        force_cpu=True,
        run_mode="test",
        engine="local",
        profile_name="default",
        auto_language=False,
        language="en",
        auto_upgrade=False,
        log_transcripts=False,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_test_mode_needs_no_tts_or_credentials_and_preserves_text(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    stop = threading.Event()
    audio_queue = queue.Queue()
    audio_queue.put(np.zeros(16000, dtype=np.float32))
    model = Mock()
    model.transcribe.return_value = {"text": "Hello there", "segments": []}
    monkeypatch.setattr(app, "get_device_and_model", lambda *a, **k: model)
    monkeypatch.setattr(
        app,
        "torch",
        SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)),
        raising=False,
    )
    monkeypatch.setattr(app, "analyze_mood", lambda text: ("neutral", 0))
    monkeypatch.setattr(app, "get_api_key", Mock(side_effect=AssertionError("API used")))
    monkeypatch.setattr(app, "save_report", lambda report: stop.set())
    app.process_audio(audio_queue, stop, config(engine="elevenlabs"), app.StatusBus())
    assert model.transcribe.call_count == 1
    assert audio_queue.unfinished_tasks == 0
    assert not (tmp_path / "transcripts.txt").exists()


def test_tts_does_not_speak_stage_directions():
    engine = Mock()
    app.text_to_elevenlabs_voice("Hello there", engine, "positive")
    engine.synthesize.assert_called_once_with("Hello there", "positive")


def test_cli_help_and_diagnostics_without_dependencies():
    script = Path(app.__file__)
    result = subprocess.run(
        [sys.executable, "-S", str(script), "--help"], capture_output=True, text=True
    )
    assert result.returncode == 0
    assert "--list-devices" in result.stdout
    result = subprocess.run(
        [sys.executable, "-S", str(script), "--self-check"], capture_output=True, text=True
    )
    assert result.returncode == 1
    report = json.loads(result.stdout[result.stdout.index("{") :])
    assert report["summary"]["failed"] > 0
    assert not result.stderr


@pytest.mark.parametrize(
    "args",
    [
        ["--silence-chunks", "0"],
        ["--min-buffer-chunks", "500"],
        ["--profile-name", "../escape"],
        ["--playback-block-size", "-1"],
    ],
)
def test_invalid_cli_values_fail_before_runtime_load(args):
    result = subprocess.run(
        [sys.executable, "-S", app.__file__, *args], capture_output=True, text=True
    )
    assert result.returncode == 2
    assert "error:" in result.stderr


def test_capture_releases_device_after_read_error(monkeypatch):
    stream = Mock()
    stream.__enter__ = Mock(return_value=stream)
    stream.__exit__ = Mock(return_value=False)
    stream.read.side_effect = OSError("disconnected")
    monkeypatch.setattr(
        app, "sd", SimpleNamespace(RawInputStream=Mock(return_value=stream)), raising=False
    )
    with pytest.raises(OSError, match="disconnected"):
        app.capture_frames(1)
    stream.__exit__.assert_called_once()


def test_backpressure_keeps_queue_bounded_and_closes_input(monkeypatch):
    stop = threading.Event()
    frames = [pcm(1000), pcm(0)] * 5
    stream = Mock()
    stream.__enter__ = Mock(return_value=stream)
    stream.__exit__ = Mock(return_value=False)

    def read(count):
        data = frames.pop(0)
        if not frames:
            stop.set()
        return data, False

    stream.read.side_effect = read
    monkeypatch.setattr(
        app, "sd", SimpleNamespace(RawInputStream=Mock(return_value=stream)), raising=False
    )
    monkeypatch.setattr(app, "np", np, raising=False)
    q = queue.Queue(maxsize=1)
    bus = app.StatusBus()
    app.real_time_record(
        q,
        stop,
        config(
            silence_threshold=500,
            silence_chunks=1,
            min_buffer_chunks=1,
            use_noise_reduction=False,
            device_index=None,
        ),
        bus,
    )
    assert q.qsize() == 1
    assert any("dropped" in message for message in list(bus.queue.queue))
    stream.__exit__.assert_called_once()


def test_playback_failure_closes_output(monkeypatch):
    import audio_playback as playback

    stream = Mock()
    stream.start.side_effect = OSError("unavailable")
    monkeypatch.setattr(playback.sd, "OutputStream", Mock(return_value=stream))
    engine = playback.PlaybackEngine(playback.PlaybackConfig(24000))
    with pytest.raises(OSError):
        engine.play(np.ones(10))
    assert engine.stream is None
    stream.close.assert_called_once()


def test_elevenlabs_pcm_handles_split_sample_boundaries(monkeypatch):
    from elevenlabs_engine import ElevenLabsEngine, ElevenLabsConfig
    import types

    fake_sdk = types.ModuleType("elevenlabs")
    fake_sdk.VoiceSettings = lambda **kwargs: kwargs
    monkeypatch.setitem(sys.modules, "elevenlabs", fake_sdk)
    engine = ElevenLabsEngine.__new__(ElevenLabsEngine)
    engine.config = ElevenLabsConfig("fake-test-key", "test-voice")
    engine.client = Mock()
    raw = struct.pack("<hhh", 32767, -32768, 1024)
    engine.client.text_to_speech.convert.return_value = iter([raw[:1], raw[1:3], raw[3:]])
    engine.playback = Mock()
    engine.synthesize("Exact words", "neutral")
    played = np.concatenate([call.args[0] for call in engine.playback.play.call_args_list])
    np.testing.assert_allclose(played, np.array([32767, -32768, 1024]) / 32768)
    params = engine.client.text_to_speech.convert.call_args.kwargs
    assert params["text"] == "Exact words"
    assert params["output_format"] == "pcm_24000"


def test_local_synthesis_preserves_spoken_words(monkeypatch):
    monkeypatch.setattr(app, "np", np, raising=False)
    engine = Mock()
    engine.synthesize.return_value = SimpleNamespace(audio=np.ones(100) * 0.1)
    playback = Mock()
    app.text_to_cloned_voice(
        "Keep these words",
        config(speaker_wav="voice.wav", playback_gain=1),
        engine,
        app.AdaptiveGain(1),
        playback,
        "positive",
    )
    assert engine.synthesize.call_args.kwargs["text"] == "Keep these words"
    playback.play.assert_called_once()
