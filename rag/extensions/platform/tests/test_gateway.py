import json
from types import SimpleNamespace

import httpx2
import openai
import pytest

from mlsrag import gateway as gw_module
from mlsrag.gateway import ModelGateway
from mlsrag.trace import Trace

SECRET = "sk-test-SECRET-123"


def fake_response(text="hello"):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
        usage=SimpleNamespace(prompt_tokens=7, completion_tokens=3),
    )


@pytest.fixture
def gateway(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text(f"ELM_API_KEY={SECRET}\n")
    monkeypatch.setattr(gw_module.time, "sleep", lambda s: None)
    g = ModelGateway(
        trace=Trace("r", "q", "c"),
        cache_dir=tmp_path / "cache",
        log_path=tmp_path / "log.jsonl",
        env_path=tmp_path / ".env",
    )
    g.client()  # builds the real client from the fake .env; no network yet
    return g


def install(g, monkeypatch, responses):
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        r = responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r

    monkeypatch.setattr(g._client.chat.completions, "create", create)
    return calls


def status_error(code):
    response = httpx2.Response(code, request=httpx2.Request("POST", "http://test/chat"))
    return openai.APIStatusError("error", response=response, body=None)


def test_second_identical_call_is_a_cache_hit(gateway, monkeypatch):
    calls = install(gateway, monkeypatch, [fake_response("hi")])
    msgs = [{"role": "user", "content": "q"}]
    assert gateway.chat("m", msgs) == "hi"
    assert gateway.chat("m", msgs) == "hi"
    assert len(calls) == 1
    t = gateway.trace
    assert [s.name for s in t.spans] == ["gateway.chat", "gateway.chat"]
    assert [s.attrs["cached"] for s in t.spans] == [False, True]
    assert t.cost["m"] == {"calls": 1, "prompt_tokens": 7, "completion_tokens": 3, "audio_seconds": 0.0}
    assert t.counters == {"gateway.chat.calls": 1, "gateway.chat.cache_hits": 1}


def test_retries_on_429_and_5xx(gateway, monkeypatch):
    calls = install(gateway, monkeypatch, [status_error(429), status_error(503), fake_response()])
    assert gateway.chat("m", [{"role": "user", "content": "q"}]) == "hello"
    assert len(calls) == 3
    assert gateway.trace.spans[0].attrs["retries"] == 2


def test_retries_on_timeout(gateway, monkeypatch):
    timeout = openai.APITimeoutError(request=httpx2.Request("POST", "http://test/chat"))
    calls = install(gateway, monkeypatch, [timeout, fake_response()])
    assert gateway.chat("m", [{"role": "user", "content": "q"}]) == "hello"
    assert len(calls) == 2


def test_gives_up_after_three_retries_and_not_on_400(gateway, monkeypatch):
    install(gateway, monkeypatch, [status_error(429)] * 4)
    with pytest.raises(openai.APIStatusError):
        gateway.chat("m", [{"role": "user", "content": "a"}])
    calls = install(gateway, monkeypatch, [status_error(400)])
    with pytest.raises(openai.APIStatusError):
        gateway.chat("m", [{"role": "user", "content": "b"}])
    assert len(calls) == 1
    failures = [json.loads(line) for line in gateway.log_path.read_text().splitlines()]
    assert [(f["error"], f["retries"]) for f in failures] == [
        ("APIStatusError", 3),
        ("APIStatusError", 0),
    ]


def test_log_line_has_no_key(gateway, monkeypatch):
    install(gateway, monkeypatch, [fake_response()])
    gateway.chat("m", [{"role": "user", "content": "q"}])
    text = (gateway.log_path).read_text()
    assert SECRET not in text
    line = json.loads(text.splitlines()[0])
    assert line["op"] == "chat" and line["model"] == "m" and line["cached"] is False


def test_fresh_skips_cache_reads_but_writes(gateway, monkeypatch):
    calls = install(gateway, monkeypatch, [fake_response("a"), fake_response("b")])
    gateway.fresh = True
    msgs = [{"role": "user", "content": "q"}]
    assert gateway.chat("m", msgs) == "a"
    assert gateway.chat("m", msgs) == "b"
    assert len(calls) == 2
    assert [s.attrs["cached"] for s in gateway.trace.spans] == [False, False]
    assert gateway.trace.cost["m"]["calls"] == 2
    assert len(list(gateway.cache_dir.glob("*.json"))) == 1  # written (and overwritten)
    gateway.fresh = False
    assert gateway.chat("m", msgs) == "b"  # a normal gateway reads what fresh wrote
    assert len(calls) == 2


def test_fresh_embed_recomputes(gateway, monkeypatch):
    encoded = []

    class FakeEncoder:
        def encode(self, texts, **kwargs):
            encoded.append(texts)
            return [[1.0, 0.0] for _ in texts]

    monkeypatch.setattr(gateway, "encoder", lambda name: FakeEncoder())
    gateway.embed("enc", ["q"])
    gateway.embed("enc", ["q"])
    assert len(encoded) == 1
    gateway.fresh = True
    gateway.embed("enc", ["q"])
    assert len(encoded) == 2
    assert [s.attrs["cached"] for s in gateway.trace.spans] == [False, True, False]


def test_fresh_transcribe_skips_the_cache_read(gateway, monkeypatch, tmp_path):
    audio = tmp_path / "q.wav"
    audio.write_bytes(b"not a wav")  # duration unknown is fine here
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(text=f"t{len(calls)}")

    monkeypatch.setattr(gateway._client.audio.transcriptions, "create", create)
    assert gateway.transcribe(audio) == "t1"
    assert gateway.transcribe(audio) == "t1"  # cached
    assert len(calls) == 1
    assert gateway.transcribe(audio, fresh=True) == "t2"
    assert gateway.transcribe(audio, fresh=True) == "t3"
    assert len(calls) == 3
    assert gateway.transcribe(audio) == "t3"  # fresh still writes the cache
