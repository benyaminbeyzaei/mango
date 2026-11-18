import json
from types import SimpleNamespace

import numpy as np
import pytest

from mlsrag.preprocess import label
from mlsrag.preprocess.ledger import Ledger


def test_resume_skips_ok_and_retries_failed(tmp_path):
    ledger = Ledger(tmp_path / "ledger.jsonl")
    ledger.append(0, "h0", "ok", output={"topic": "a"})
    ledger.append(1, "h1", "failed", error="timeout")
    assert ledger.done_ids() == {0}
    ledger.append(1, "h1", "ok", output={"topic": "b"})
    assert ledger.done_ids() == {0, 1}
    assert len(ledger.rows()) == 3


class FlakyGateway:
    def __init__(self, fail_on):
        self.fail_on = fail_on
        self.prompts = []

    def chat(self, model, messages, temperature=0.0, max_tokens=None):
        prompt = messages[0]["content"]
        self.prompts.append(prompt)
        if self.fail_on in prompt:
            raise TimeoutError("slow")
        return 'Sure: {"topic": "T", "summary": "S."}'


def test_label_stage_finishes_only_missing_clusters(tmp_path):
    sel, clu, out = tmp_path / "sel", tmp_path / "clu", tmp_path / "out"
    for d in (sel, clu, out):
        d.mkdir()
    (sel / "records.jsonl").write_text("".join(
        json.dumps({"record_id": f"r{i}", "text": f"record number {i}"}) + "\n" for i in range(4)))
    clusters = [{"cluster": c, "top_words": [f"word{c}"], "members": [f"r{c}", f"r{c + 2}"]} for c in range(2)]
    (clu / "clusters.json").write_text(json.dumps(clusters))
    inputs, config = {"select": sel, "cluster": clu}, {"model": "m"}

    def ctx(gw):
        return SimpleNamespace(out_dir=out, gateway=gw, trace=SimpleNamespace(cost={}), log=lambda *_: None)

    gw = FlakyGateway(fail_on="word1")
    try:
        label.run(inputs, config, ctx(gw))
        raise AssertionError("expected a failure")
    except RuntimeError:
        pass
    gw = FlakyGateway(fail_on="nothing")
    counts = label.run(inputs, config, ctx(gw))
    assert len(gw.prompts) == 1 and "word1" in gw.prompts[0]
    assert counts["failed_attempts"] == 1
    labels = [json.loads(line) for line in (out / "labels.jsonl").read_text().splitlines()]
    assert labels == [{"cluster": 0, "topic": "T", "summary": "S."}, {"cluster": 1, "topic": "T", "summary": "S."}]


def test_parse_label_rejects_non_string_fields():
    assert label.parse_label('ok {"topic": " A ", "summary": "B"}') == {"topic": "A", "summary": "B"}
    for bad in ['{"topic": null, "summary": "B"}', '{"topic": ["A"], "summary": "B"}', '{"topic": "A"}', "no json"]:
        with pytest.raises(ValueError):
            label.parse_label(bad)


def test_cut_off_last_line_is_ignored_and_not_appended_onto(tmp_path):
    ledger = Ledger(tmp_path / "ledger.jsonl")
    ledger.append(0, "h0", "ok", output={"topic": "a"})
    with ledger.path.open("a") as f:
        f.write('{"item_id": 1, "sta')  # Ctrl-C in the middle of a write
    assert ledger.done_ids() == {0}
    ledger.append(1, "h1", "ok")
    assert ledger.done_ids() == {0, 1}


class CachingBadGateway:
    """Like the real gateway: the same prompt always returns the same (cached) reply."""

    def __init__(self):
        self.prompts = []

    def chat(self, model, messages, temperature=0.0, max_tokens=None):
        prompt = messages[0]["content"]
        self.prompts.append(prompt)
        return '{"topic": "T", "summary": "S."}' if "Retry" in prompt else "not json"


def test_retry_after_bad_reply_uses_a_new_prompt(tmp_path):
    sel, clu, out = tmp_path / "sel", tmp_path / "clu", tmp_path / "out"
    for d in (sel, clu, out):
        d.mkdir()
    (sel / "records.jsonl").write_text(json.dumps({"record_id": "r0", "text": "x"}) + "\n")
    (clu / "clusters.json").write_text(json.dumps([{"cluster": 0, "top_words": ["w"], "members": ["r0"]}]))
    gw = CachingBadGateway()
    ctx = SimpleNamespace(out_dir=out, gateway=gw, trace=SimpleNamespace(cost={}), log=lambda *_: None)
    inputs = {"select": sel, "cluster": clu}
    try:
        label.run(inputs, {"model": "m"}, ctx)
        raise AssertionError("expected a failure")
    except RuntimeError:
        pass
    assert label.run(inputs, {"model": "m"}, ctx)["clusters"] == 1
    assert gw.prompts[0] != gw.prompts[1]


class NamingGateway:
    """First pass: every topic is called "Same"; a rename call names the topic after its first record."""

    def __init__(self):
        self.prompts = []

    def chat(self, model, messages, temperature=0.0, max_tokens=None):
        prompt = messages[0]["content"]
        self.prompts.append(prompt)
        if "This topic:" in prompt:
            first = prompt.split("This topic:\n- ")[1].split("\n")[0]
            return json.dumps({"topic": f"About {first}", "summary": "S."})
        return '{"topic": "Same", "summary": "S."}'


def test_label_v2_uses_nearest_records_and_renames_clashes(tmp_path, monkeypatch):
    sel, clu, out = tmp_path / "sel", tmp_path / "clu", tmp_path / "out"
    for d in (sel, clu, out):
        d.mkdir()
    (sel / "records.jsonl").write_text("".join(
        json.dumps({"record_id": f"r{i}", "text": f"record {i}"}) + "\n" for i in range(4)))
    clusters = [{"cluster": c, "top_words": ["w"], "members": [f"r{c}", f"r{c + 2}"]} for c in range(2)]
    (clu / "clusters.json").write_text(json.dumps(clusters))
    np.save(clu / "centroids.npy", np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32))
    # r2 and r3 lie nearer the centroid than r0 and r1, so they come first in each prompt
    vectors = {"r0": [0.1, 1.0], "r1": [0.1, 1.0], "r2": [1.0, 0.0], "r3": [1.0, 0.0]}
    monkeypatch.setattr(label, "load_vectors", lambda ids: np.array([vectors[r] for r in ids], dtype=np.float32))
    gw = NamingGateway()
    ctx = SimpleNamespace(out_dir=out, gateway=gw, trace=SimpleNamespace(cost={}), log=lambda *_: None)
    counts = label.run({"select": sel, "cluster": clu}, {"model": "m", "prompt": "v2"}, ctx)
    assert gw.prompts[0].index("record 2") < gw.prompts[0].index("record 0")
    assert counts["rename_calls"] == 2
    names = [json.loads(line)["topic"] for line in (out / "labels.jsonl").read_text().splitlines()]
    assert names == ["About record 2", "About record 3"]
