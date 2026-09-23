import json
import queue
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

import voice_clone_app as app
from audio_metrics import measure_audio
from conversation_ai import conversation_insights, infer_intent
from quality_ai import assess_call_quality
from stt_engine import FasterWhisperAdapter, recognition_warnings
from transcript_tools import export_transcription, subtitle_time


def test_intent_matches_words_not_substrings():
    assert infer_intent("This is a nice chair") == "general"
    assert infer_intent("The costume is red") == "general"
    assert infer_intent("The price is fair") == "sales"


def test_insights_are_source_sentences_and_actions_are_candidates():
    text = "We will send the invoice tomorrow. Is the price correct? Thank you."
    result = conversation_insights(text)
    assert result["candidate_actions"] == ["We will send the invoice tomorrow."]
    assert result["questions"] == ["Is the price correct?"]
    assert all(sentence in text for sentence in result["summary"])
    assert conversation_insights("")["summary"] == []


def test_measured_levels_and_clipping():
    assert measure_audio(np.zeros(100))["rms_dbfs"] == -120
    stats = measure_audio(np.array([1.0, -1.0, 0, 0]))
    assert stats["clipping_percent"] == 50
    assert stats["peak_dbfs"] == 0
    with pytest.raises(ValueError):
        measure_audio(np.array([np.nan]))


def test_sentiment_is_not_a_quality_penalty():
    assert (
        assess_call_quality(100, 120, "negative", True).score
        == assess_call_quality(100, 120, "positive", True).score
    )
    assert (
        "clipping"
        in assess_call_quality(100, 120, "neutral", True, clipping_percent=10).recommendation
    )
    assert "smaller model" in assess_call_quality(2000, 120, "neutral", True).recommendation


def test_faster_backend_normalizes_lazy_segments():
    adapter = FasterWhisperAdapter.__new__(FasterWhisperAdapter)
    word = SimpleNamespace(word=" Hello", start=0, end=0.5, probability=0.9)
    segment = SimpleNamespace(
        start=0, end=1, text=" Hello", words=[word], avg_logprob=-0.2, no_speech_prob=0.1
    )
    consumed = []

    def segments():
        consumed.append(True)
        yield segment

    adapter.model = Mock()
    adapter.model.transcribe.return_value = (
        segments(),
        SimpleNamespace(language="en", language_probability=1),
    )
    result = adapter.transcribe(np.zeros(16000), fp16=False, word_timestamps=True)
    assert consumed and result["text"] == "Hello"
    assert result["segments"][0]["words"][0]["probability"] == 0.9
    options = adapter.model.transcribe.call_args.kwargs
    assert "fp16" not in options and options["vad_filter"]
    assert options["condition_on_previous_text"] is False


def test_model_uncertainty_is_visible():
    assert (
        len(recognition_warnings({"segments": [{"avg_logprob": -2, "no_speech_prob": 0.9}]})) == 2
    )


def test_exports_preserve_unicode_timestamps_and_existing_files(tmp_path):
    prefix = tmp_path / "meeting.2026"
    result = {
        "text": "Bonjour café.",
        "segments": [{"text": "Bonjour café.", "start": 59.9996, "end": 62.5}],
    }
    paths = export_transcription(result, prefix)
    assert json.loads((tmp_path / "meeting.2026.json").read_text()) == result
    assert "00:01:00,000 --> 00:01:02,500" in (tmp_path / "meeting.2026.srt").read_text()
    assert len(paths) == 3
    with pytest.raises(FileExistsError):
        export_transcription({"text": "changed"}, prefix)
    assert (tmp_path / "meeting.2026.txt").read_text() == "Bonjour café.\n"


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf")])
def test_subtitle_time_rejects_invalid_values(value):
    with pytest.raises(ValueError):
        subtitle_time(value)


def test_invalid_exports_do_not_create_partial_files(tmp_path):
    with pytest.raises(ValueError):
        export_transcription(
            {"text": "hi", "segments": [{"text": "hi", "start": 5, "end": 2}]}, tmp_path / "bad"
        )
    assert not list(tmp_path.iterdir())


def test_output_mute_prevents_device_open(monkeypatch):
    import audio_playback

    factory = Mock()
    monkeypatch.setattr(audio_playback.sd, "OutputStream", factory)
    mute = threading.Event()
    mute.set()
    engine = audio_playback.PlaybackEngine(audio_playback.PlaybackConfig(24000))
    engine.play(np.ones(10), muted=mute)
    factory.assert_not_called()


def test_output_stop_interrupts_blocked_sequence(monkeypatch):
    import audio_playback

    stop = threading.Event()
    stream = Mock()
    stream.write.side_effect = lambda audio: stop.set()
    monkeypatch.setattr(audio_playback.sd, "OutputStream", Mock(return_value=stream))
    engine = audio_playback.PlaybackEngine(audio_playback.PlaybackConfig(24000, block_size=4))
    engine.play(np.ones(20), stop_event=stop)
    assert stream.write.call_count == 1
    stream.abort.assert_called_once()


def test_cloud_mute_avoids_billable_request():
    from elevenlabs_engine import ElevenLabsEngine

    engine = ElevenLabsEngine.__new__(ElevenLabsEngine)
    engine.client = Mock()
    muted = threading.Event()
    muted.set()
    # Return before loading SDK objects or making a request.
    engine.synthesize("Do not send", "neutral", muted=muted)
    engine.client.text_to_speech.convert.assert_not_called()


def test_session_memory_is_bounded():
    bus = app.StatusBus()
    for i in range(300):
        bus.remember(f"Phrase {i}.")
    assert len(bus.transcripts) == 200
    assert bus.transcripts[0] == "Phrase 100."


def test_stale_audio_is_discarded_before_transcription(monkeypatch):
    import time

    stop = threading.Event()
    q = queue.Queue()
    q.put((time.monotonic() - 20, np.zeros(16000)))
    model = Mock()
    monkeypatch.setattr(app, "get_device_and_model", lambda *a, **k: model)
    monkeypatch.setattr(app, "load_report", lambda: {})
    bus = app.StatusBus()
    emit = bus.emit

    def on_status(message):
        emit(message)
        if "stale" in message:
            stop.set()

    bus.emit = on_status
    config = SimpleNamespace(
        model_name="tiny.en",
        force_cpu=True,
        run_mode="test",
        engine="local",
        profile_name="test",
        max_queue_seconds=10,
    )
    app.process_audio(q, stop, config, bus)
    model.transcribe.assert_not_called()
    assert q.unfinished_tasks == 0 and bus.snapshot()["dropped"] == 1


def test_xtts_conditioning_is_reused_and_invalidated(tmp_path):
    from tts_engine import LocalTTS

    reference = tmp_path / "voice.wav"
    reference.write_bytes(b"first")
    model = Mock()
    model.get_conditioning_latents.return_value = ("latent", "speaker")
    model.inference.return_value = {"wav": [0.1, 0.2]}
    engine = LocalTTS.__new__(LocalTTS)
    engine.model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
    engine.tts = SimpleNamespace(synthesizer=SimpleNamespace(tts_model=model))
    engine.output_sample_rate = 24000
    engine.speed = 1.1
    engine._reference_key = None
    engine._conditioning = None
    engine.synthesize("First phrase", str(reference))
    engine.synthesize("Second phrase", str(reference))
    assert model.get_conditioning_latents.call_count == 1
    reference.write_bytes(b"changed reference")
    engine.synthesize("Third phrase", str(reference))
    assert model.get_conditioning_latents.call_count == 2
    assert model.inference.call_args.kwargs["speed"] == 1.1


def test_reference_normalization_does_not_introduce_clipping(tmp_path):
    import soundfile as sf
    from voice_profile import analyze_speaker_wav

    sample = np.zeros(16000)
    sample[:100] = 0.9
    path = tmp_path / "sample.wav"
    sf.write(path, sample, 16000)
    profile = analyze_speaker_wav(str(path))
    assert profile.peak * profile.gain <= 0.951
    assert any("Short" in warning for warning in profile.warnings)


def test_cloud_listing_paginates_without_synthesis(monkeypatch):
    import sys
    import types
    import elevenlabs_engine as cloud

    client = Mock()
    client.voices.search.side_effect = [
        SimpleNamespace(
            voices=[SimpleNamespace(voice_id="a", name="First")],
            has_more=True,
            next_page_token="next",
        ),
        SimpleNamespace(
            voices=[SimpleNamespace(voice_id="b", name="Second")],
            has_more=False,
            next_page_token=None,
        ),
    ]
    sdk = types.ModuleType("elevenlabs")
    submodule = types.ModuleType("elevenlabs.client")
    submodule.ElevenLabs = Mock(return_value=client)
    monkeypatch.setitem(sys.modules, "elevenlabs", sdk)
    monkeypatch.setitem(sys.modules, "elevenlabs.client", submodule)
    monkeypatch.setattr(cloud, "get_api_key", lambda: "test-only")
    assert cloud.list_cloud_resources() == [
        {"id": "a", "name": "First"},
        {"id": "b", "name": "Second"},
    ]
    assert client.voices.search.call_args.kwargs["next_page_token"] == "next"
    client.text_to_speech.convert.assert_not_called()
    client.close.assert_called_once()


@pytest.mark.parametrize(
    "arguments",
    [
        ["--voice-speed", "nan"],
        ["--max-clip-seconds", "inf"],
        ["--task", "translate"],
        ["--input-file", "audio.wav", "--task", "translate", "--model", "base.en"],
        ["--output", "exports/test"],
        ["--force-cpu", "--compute-type", "float16"],
    ],
)
def test_new_cli_flags_fail_early(monkeypatch, arguments):
    import sys

    monkeypatch.setattr(sys, "argv", ["caller", *arguments])
    with pytest.raises(SystemExit) as error:
        app.parse_args()
    assert error.value.code == 2


def test_file_cancellation_does_not_transcribe(monkeypatch, tmp_path):
    import transcript_tools
    from stt_engine import STTConfig

    path = tmp_path / "test.wav"
    path.touch()
    stop = threading.Event()
    stop.set()
    model = Mock()
    monkeypatch.setattr(transcript_tools, "load_stt", lambda config: model)
    assert transcript_tools.transcribe_file(path, STTConfig(), stop_event=stop) is None
    model.transcribe.assert_not_called()


def test_cancelled_reference_recording_does_not_save(tmp_path, monkeypatch):
    stop = threading.Event()

    def capture(*args):
        stop.set()
        return [b"\0\0"]

    monkeypatch.setattr(app, "capture_frames", capture)
    path = tmp_path / "reference.wav"
    assert app.record_sample(filename=str(path), stop_event=stop) is None
    assert not path.exists()
