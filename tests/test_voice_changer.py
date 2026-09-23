import base64
import json
import queue
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from voice_changer_engine import LiveConfig, VoiceChangerClient, offer_latest, run_live


@pytest.fixture
def server():
    state = {"info": {"status": "OK", "modelSlotIndex": 0, "passThrough": False,
                      "inputSampleRate": 48000, "outputSampleRate": 48000},
             "mode": "valid", "requests": []}

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def reply(self, data):
            payload = json.dumps(data).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            self.reply(state["info"])

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state["requests"].append(body)
            data = base64.b64decode(body["buffer"])
            if state["mode"] == "short":
                data = b"\0\0"
            result = {"timestamp": body["timestamp"], "changedVoiceBase64": base64.b64encode(data).decode()}
            if state["mode"] == "timestamp":
                result["timestamp"] += 1
            if state["mode"] == "base64":
                result["changedVoiceBase64"] = "!!!"
            self.reply(result)

        def log_message(self, *args):
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_port}", state
    httpd.shutdown()
    httpd.server_close()
    thread.join()


@pytest.mark.parametrize("url", ["https://example.com", "http://user:pass@localhost", "http://localhost/path", "file:///tmp/a", "http://localhost?x=1"])
def test_only_local_server(url):
    with pytest.raises(ValueError):
        LiveConfig(url)


def test_real_http_pcm_contract(server):
    url, state = server
    client = VoiceChangerClient(LiveConfig(url, block_size=2048))
    try:
        assert client.check()["modelSlotIndex"] == 0
        pcm = b"\x00\x80\xff\x7f" * 1024
        assert client.convert(pcm) == pcm
        assert client.convert(pcm) == pcm
        assert [x["timestamp"] for x in state["requests"]] == [1, 2]
    finally:
        client.close()


@pytest.mark.parametrize("key,value", [("passThrough", True), ("serverAudioStated", 1), ("inputSampleRate", 16000), ("outputSampleRate", 24000), ("modelSlotIndex", -1), ("status", "ERROR")])
def test_reject_unready_server(server, key, value):
    url, state = server
    state["info"][key] = value
    client = VoiceChangerClient(LiveConfig(url))
    try:
        with pytest.raises(RuntimeError):
            client.check()
    finally:
        client.close()


@pytest.mark.parametrize("mode", ["timestamp", "short", "base64"])
def test_bad_frames_fail_closed(server, mode):
    url, state = server
    state["mode"] = mode
    client = VoiceChangerClient(LiveConfig(url))
    try:
        with pytest.raises(RuntimeError):
            client.convert(bytes(16384))
    finally:
        client.close()


def test_queue_discards_oldest():
    q = queue.Queue(maxsize=2)
    offer_latest(q, 1)
    offer_latest(q, 2)
    assert offer_latest(q, 3) == 1
    assert [q.get_nowait(), q.get_nowait()] == [2, 3]


def test_no_capture_before_server_ready(server):
    url, state = server
    state["info"]["modelSlotIndex"] = -1
    stop = threading.Event()
    config = SimpleNamespace(vc_url=url, vc_block_size=8192, device_index=0, playback_device=1)
    with pytest.raises(RuntimeError, match="Load and select"):
        run_live(config, stop, None, sd_module=object())


def test_audio_callbacks_mute_and_cleanup(server):
    url, state = server
    stop = threading.Event()
    stop.ready, stop.muted = threading.Event(), threading.Event()
    stop.muted.set()
    closed, blocks = [], []
    config = SimpleNamespace(vc_url=url, vc_block_size=8192, device_index=0, playback_device=1)

    class FakeSD:
        def check_input_settings(self, **kw):
            pass
        check_output_settings = check_input_settings

        def RawOutputStream(self, **kw):
            self.output = kw["callback"]
            return Stream("out")

        def RawInputStream(self, **kw):
            self.capture = kw["callback"]
            return Stream("in")

    class Stream:
        def __init__(self, name):
            self.name = name
        def __enter__(self):
            if self.name == "in":
                sd.capture(b"\x11\x11" * 8192, 8192, None, False)
            return self
        def __exit__(self, *args):
            closed.append(self.name)

    class Status:
        def emit(self, message):
            pass
        def update_metrics(self, **kw):
            out = bytearray(16384)
            sd.output(out, 8192, None, False)
            blocks.append(bytes(out))
            stop.set()

    sd = FakeSD()
    run_live(config, stop, Status(), sd_module=sd)
    assert blocks == [bytes(16384)]
    assert closed == ["in", "out"]
    assert stop.ready.is_set()
    assert len(state["requests"]) == 2  # readiness silence + captured frame
