import json
import time

from mlsrag.trace import Trace


def test_spans_nest_and_time():
    t = Trace("run", "q1", "cfg")
    with t.span("outer", a=1):
        time.sleep(0.01)
        with t.span("inner"):
            time.sleep(0.01)
    outer, inner = t.spans
    assert outer.attrs == {"a": 1}
    assert inner.duration_ms >= 10 and outer.duration_ms >= 20
    assert outer.start_ms <= inner.start_ms
    assert inner.start_ms + inner.duration_ms <= outer.start_ms + outer.duration_ms


def test_counters_and_cost_add():
    t = Trace("run", "q1", "cfg")
    t.count("x")
    t.count("x", 2)
    t.add_cost(prompt_tokens=10, completion_tokens=5, model="m")
    t.add_cost(prompt_tokens=1, model="m")
    t.add_cost(audio_seconds=2.5, model="whisper-1")
    assert t.counters == {"x": 3}
    assert t.cost["m"] == {"calls": 2, "prompt_tokens": 11, "completion_tokens": 5, "audio_seconds": 0.0}
    assert t.cost["whisper-1"]["audio_seconds"] == 2.5


def test_serialises_to_json():
    t = Trace("run", "q1", "cfg")
    with t.span("s"):
        pass
    t.artifacts["ranked_ids"] = ["r1"]
    t.finish()
    d = json.loads(json.dumps(t.to_dict()))
    assert d["spans"][0]["name"] == "s"
    assert set(d["resources"]) == {"wall_ms", "cpu_seconds", "peak_rss_mb"}
    assert d["artifacts"]["ranked_ids"] == ["r1"]
