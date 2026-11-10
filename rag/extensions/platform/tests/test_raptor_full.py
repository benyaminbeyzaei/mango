import json
from types import SimpleNamespace

import numpy as np
import pytest

from mlsrag.preprocess import raptor_full
from mlsrag.preprocess.ledger import ItemsFailed
from mlsrag.retrievers.raptor import SummaryRetriever, load_summaries
from mlsrag.trace import Trace


def blobs(centers, n, scale, seed=0):
    rng = np.random.default_rng(seed)
    return np.concatenate([np.asarray(c) + scale * rng.normal(size=(n, len(c))) for c in centers])


def test_bic_picks_the_number_of_blobs():
    x = blobs([[0, 0], [10, 0], [0, 10]], 30, 0.5)
    assert raptor_full.optimal_clusters(x) == 3


def test_soft_membership_puts_a_midway_point_in_both_clusters():
    x = np.concatenate([blobs([[0, 0], [6, 0]], 40, 1.0), [[3.0, 0.0]]])
    labels, k = raptor_full.gmm_cluster(x)
    assert k == 2
    assert len(labels[-1]) == 2  # midway: probability > 0.1 for both
    assert sum(len(lab) == 1 for lab in labels[:-1]) >= 70  # nearly every blob point has one cluster


def test_perform_clustering_separates_blobs_with_umap():
    x = blobs([np.eye(16)[0] * 5, np.eye(16)[1] * 5, np.eye(16)[2] * 5], 30, 0.3)
    clusters = raptor_full.perform_clustering(x)
    covered = {int(i) for c in clusters for i in c}
    assert covered == set(range(90))
    for c in clusters:  # no cluster mixes two blobs
        assert len({int(i) // 30 for i in c}) == 1


def fake_clustering(calls):
    """First call: the first 30 nodes and the last 10; later calls: halves."""
    def perform(x, seed):
        calls.append(len(x))
        n = np.arange(len(x))
        return [n[:30], n[30:]] if len(calls) == 1 else [n[: len(x) // 2], n[len(x) // 2:]]
    return perform


def test_a_cluster_over_3500_tokens_is_clustered_again(monkeypatch):
    calls = []
    monkeypatch.setattr(raptor_full, "perform_clustering", fake_clustering(calls))
    x = np.zeros((40, 3))
    clusters = raptor_full.raptor_clusters(x, [200] * 40)  # 30 nodes = 6000 tokens > 3500; 10 nodes = 2000
    assert calls == [40, 30]
    assert [c.tolist() for c in clusters] == [list(range(15)), list(range(15, 30)), list(range(30, 40))]


def test_small_or_unsplittable_clusters_are_kept(monkeypatch):
    monkeypatch.setattr(raptor_full, "perform_clustering", lambda x, seed: [np.arange(len(x))])
    assert [len(c) for c in raptor_full.raptor_clusters(np.zeros((40, 3)), [100] * 40)] == [40]  # GMM gives it back whole
    assert [len(c) for c in raptor_full.raptor_clusters(np.zeros((11, 3)), [400] * 11)] == [11]  # too few to cluster


def test_get_text_matches_the_reference():
    assert raptor_full.get_text(["a\nb", "c"]) == "a b\n\nc\n\n"


class FakeGateway:
    def __init__(self, empty_first=False):
        self.calls = []
        self.trace = Trace("r", "q", "c")
        self.empty_first = empty_first

    def chat(self, model, messages, temperature=0.0, max_tokens=None, tag=None):
        self.calls.append({"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens,
                           "tag": tag})
        return " " if self.empty_first and len(self.calls) == 1 else f"summary {len(self.calls)}"

    def embed(self, encoder, texts, role="query"):
        v = np.random.default_rng(len(texts)).normal(size=(len(texts), 4))
        return v / np.linalg.norm(v, axis=1, keepdims=True)


CONFIG = {"model": "gpt-4o-mini", "max_usd": 20, "usd_per_m_input": 0.15, "usd_per_m_output": 0.60}


def build(tmp_path, monkeypatch, config=CONFIG, n=40, gateway=None, logs=None):
    select = tmp_path / "select"
    select.mkdir(parents=True)
    with (select / "records.jsonl").open("w") as f:
        for i in range(n):
            f.write(json.dumps({"record_id": f"r{i}", "text": f"record\n{i}"}) + "\n")
    v = np.random.default_rng(0).normal(size=(n, 4)).astype(np.float32)
    monkeypatch.setattr(raptor_full, "load_vectors", lambda ids: v / np.linalg.norm(v, axis=1, keepdims=True))
    monkeypatch.setattr(raptor_full, "raptor_clusters",  # consecutive pairs
                        lambda x, tokens, seed: [np.arange(i, min(i + 2, len(x))) for i in range(0, len(x), 2)])
    out = tmp_path / "out"
    out.mkdir()
    g = gateway or FakeGateway()
    log = logs.append if logs is not None else lambda m: None
    ctx = SimpleNamespace(out_dir=out, gateway=g, trace=g.trace, log=log, confirm=None)
    return raptor_full.build_run({"select": select}, config, ctx), g, out


def test_build_summarises_layer_by_layer_with_the_paper_prompt(tmp_path, monkeypatch):
    counts, g, out = build(tmp_path, monkeypatch)
    assert [layer["nodes"] for layer in counts["layers"]] == [40, 20, 10]  # stops at <= 11 nodes
    first = g.calls[0]
    assert first["messages"] == [
        {"role": "system", "content": "You are a Summarizing Text Portal"},
        {"role": "user", "content": "Write a summary of the following, including as many key details as possible: "
                                    "record 0\n\nrecord 1\n\n:"},
    ]
    assert (first["model"], first["temperature"], first["max_tokens"]) == ("gpt-4o-mini", 1.0, 100)
    vectors, rows = load_summaries(out)
    assert len(rows) == len(vectors) == 30
    top = rows[20]
    assert (top["id"], top["level"], top["children"]) == ("L2.0", 2, ["L1.0", "L1.1"])
    assert top["members"] == ["r0", "r1", "r2", "r3"]  # down to the leaf records


def test_build_resumes_without_new_calls(tmp_path, monkeypatch):
    _, g, out = build(tmp_path, monkeypatch)
    calls = len(g.calls)
    g2 = FakeGateway()
    ctx = SimpleNamespace(out_dir=out, gateway=g2, trace=g2.trace, log=lambda m: None, confirm=None)
    raptor_full.build_run({"select": tmp_path / "select"}, CONFIG, ctx)
    assert calls == 30 and g2.calls == []


def test_build_stops_before_calls_over_the_budget(tmp_path, monkeypatch):
    with pytest.raises(RuntimeError, match="over max_usd"):
        build(tmp_path, monkeypatch, config={**CONFIG, "max_usd": 0.0001})


def test_build_stops_when_only_the_whole_tree_estimate_is_over_the_budget(tmp_path, monkeypatch):
    logs, pricey = [], {**CONFIG, "max_usd": 1e9, "usd_per_m_input": 1e5, "usd_per_m_output": 1e5}
    build(tmp_path / "a", monkeypatch, config=pricey, logs=logs)
    dollars = lambda text: float(text.rsplit("$", 1)[1].split()[0])  # noqa: E731
    layer1 = dollars(next(m for m in logs if m.startswith("layer 1")))
    whole = dollars(next(m for m in logs if m.startswith("upper layers")))
    assert layer1 < whole
    with pytest.raises(RuntimeError, match="whole-tree estimate"):
        build(tmp_path / "b", monkeypatch, config={**pricey, "max_usd": (layer1 + whole) / 2})


def test_a_retry_keeps_the_prompt_and_changes_the_cache_tag(tmp_path, monkeypatch):
    g = FakeGateway(empty_first=True)
    with pytest.raises(ItemsFailed):  # the empty reply fails item L1.0
        build(tmp_path, monkeypatch, gateway=g)
    g.calls.clear()
    g.empty_first = False
    ctx = SimpleNamespace(out_dir=tmp_path / "out", gateway=g, trace=g.trace, log=lambda m: None, confirm=None)
    raptor_full.build_run({"select": tmp_path / "select"}, CONFIG, ctx)
    retry = g.calls[0]
    assert retry["tag"] == "retry1" and retry["messages"][1]["content"].endswith("record 1\n\n:")


def test_summary_hit_expands_to_leaf_records_through_layers():
    ids = ["a", "b", "c", "d"]
    vectors = np.eye(4, dtype=np.float32)
    rows = [{"id": "L2.0", "level": 2, "summary": "s", "members": ["c", "d"]}]
    retriever = SummaryRetriever(np.array([[0, 0, 0.6, 0.8]]), rows, vectors, ids)
    hits = retriever.search(np.array([0, 0, 0.6, 0.8]), k=2)
    assert [r for r, _ in hits] == ["d", "c"]
