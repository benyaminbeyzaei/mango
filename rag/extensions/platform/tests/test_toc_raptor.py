import json
from types import SimpleNamespace

import numpy as np
import pytest

from mlsrag.preprocess import ledger, summarise
from mlsrag.retrievers.raptor import SummaryData, SummaryRetriever
from mlsrag.retrievers.toc import TocRetriever, parse_ids
from mlsrag.trace import Trace


def toy_corpus():
    vectors = np.array([[1, 0, 0], [0.9, 0.1, 0], [0, 1, 0], [0, 0.9, 0.1], [0, 0, 1], [0.1, 0, 0.9]], dtype=np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors, ["a", "b", "c", "d", "e", "f"]


def toy_toc():
    # three topics (x, y, z axes); section 0 = {x, y}, section 1 = {z}
    eye = np.eye(3, dtype=np.float32)
    return {
        "topic_centroids": eye,
        "section_centroids": np.array([[0.7, 0.7, 0], [0, 0, 1]], dtype=np.float32),
        "sections": [{"section": 0, "title": "XY", "summary": "x and y", "topics": [0, 1]},
                     {"section": 1, "title": "Z", "summary": "z", "topics": [2]}],
        "topics": [{"cluster": 0, "topic": "X", "summary": "x", "members": ["a", "b"]},
                   {"cluster": 1, "topic": "Y", "summary": "y", "members": ["c", "d"]},
                   {"cluster": 2, "topic": "Z", "summary": "z", "members": ["e", "f"]}],
    }


class ScriptedGateway:
    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []
        self.trace = Trace("r", "q", "c")

    def chat(self, model, messages, temperature=0.0, max_tokens=None):
        self.prompts.append(messages[0]["content"])
        return self.replies.pop(0)


def test_parse_ids_keeps_allowed_distinct_in_order():
    assert parse_ids('Sure: {"ids": [3, 1, 3, 9]}', {1, 2, 3}, 2) == [3, 1]
    assert parse_ids("no list here", {1}, 2) == []
    assert parse_ids('{"ids": [-1, 1.5, "2", 2]}', {1, 2, 5}, 4) == [2]  # only whole allowed ints


def test_toc_centroid_route_probes_nearest_section_then_topic():
    vectors, ids = toy_corpus()
    g = SimpleNamespace(trace=Trace("r", "q", "c"))
    hits = TocRetriever(toy_toc(), vectors, ids, sections=1, m=1, gateway=g).search(np.array([0.1, 1, 0]), k=10)
    assert [r for r, _ in hits] == ["c", "d"]
    assert g.trace.counters == {"toc.sections_probed": 1, "toc.topics_probed": 1, "toc.candidates_scanned": 2}


def test_toc_llm_route_follows_the_replies_and_counts_calls():
    vectors, ids = toy_corpus()
    g = ScriptedGateway(['{"ids": [1]}', '{"ids": [2]}'])
    hits = TocRetriever(toy_toc(), vectors, ids, sections=2, m=4, route="llm", gateway=g).search(
        np.array([1.0, 0, 0]), k=10, question="where is z")  # the vector points at x; the LLM's choice (z) wins
    assert [r for r, _ in hits] == ["f", "e"]
    assert "where is z" in g.prompts[0] and "2. Z: z" in g.prompts[1] and "0. X" not in g.prompts[1]
    assert g.trace.counters["toc.llm_calls"] == 2
    assert g.trace.counters["toc.topics_probed"] == 1


def test_toc_llm_route_falls_back_to_centroids_on_bad_reply():
    vectors, ids = toy_corpus()
    g = ScriptedGateway(["I don't know", '{"ids": [7]}'])
    hits = TocRetriever(toy_toc(), vectors, ids, sections=1, m=1, route="llm", gateway=g).search(
        np.array([1.0, 0, 0]), k=10, question="q")
    assert [r for r, _ in hits] == ["a", "b"]
    assert g.trace.counters["toc.llm_fallbacks"] == 2


def toy_summaries():
    # summary 0 sits near the z records; summary 1 near x
    return np.array([[0, 0.2, 1], [1, 0, 0]], dtype=np.float32), [
        {"cluster": 0, "summary": "about z", "members": ["e", "c"]},
        {"cluster": 1, "summary": "about x", "members": ["a", "b"]},
    ]


def test_summary_hit_expands_to_its_best_members():
    vectors, ids = toy_corpus()
    sv, rows = toy_summaries()
    sv /= np.linalg.norm(sv, axis=1, keepdims=True)
    g = SimpleNamespace(trace=Trace("r", "q", "c"))
    q = np.array([0, 0.2, 1.0])  # the summary scores 1.0, above every record
    hits = SummaryRetriever(sv, rows, vectors, ids, expand_top=2, gateway=g).search(q, k=4)
    # flat order would be e, f, d, c; the summary (members e, c) puts both first
    assert [r for r, _ in hits] == ["e", "c", "f", "d"]
    assert g.trace.counters["raptor.summary_hits"] >= 1


def test_summary_as_evidence_keeps_summary_id_and_context_shows_it():
    vectors, ids = toy_corpus()
    sv, rows = toy_summaries()
    sv /= np.linalg.norm(sv, axis=1, keepdims=True)
    hits = SummaryRetriever(sv, rows, vectors, ids, summary_as_evidence=True).search(np.array([0, 0.2, 1.0]), k=3)
    assert hits[0][0] == "summary:0"
    data = SummaryData(SimpleNamespace(record_text=lambda r: f"text of {r}"), rows)
    assert data.record_text("summary:0") == "about z" and data.record_text("a") == "text of a"


def test_parse_summary_cuts_to_word_limit():
    assert summarise.parse_summary("one two three", words=2) == "one two"
    with pytest.raises(ValueError):
        summarise.parse_summary("  ")


def test_run_items_resumes_and_retries(tmp_path):
    calls = []

    trace = Trace("r", "q", "c")

    class Gw:
        def chat(self, model, messages, temperature=0.0, max_tokens=None):
            calls.append(messages[0]["content"])
            trace.add_cost(prompt_tokens=10, model=model)  # what the real gateway does on a new call
            return "" if messages[0]["content"] == "b0" else "ok " + messages[0]["content"]

    def parse(reply):
        if not reply:
            raise ValueError("empty")
        return reply

    ctx = SimpleNamespace(out_dir=tmp_path, gateway=Gw(), trace=trace, log=lambda *a: None)
    items = {0: "a", 1: "b"}
    build = lambda data, attempt: f"{data}{attempt}"  # noqa: E731
    with pytest.raises(RuntimeError):
        ledger.run_items(items, ctx, "m", build, parse)
    assert ledger.run_items(items, ctx, "m", build, parse) == {0: "ok a0", 1: "ok b1"}
    assert calls == ["a0", "b0", "b1"]  # item 0 not asked again; item 1 retried with a new prompt
    assert len((tmp_path / "ledger.jsonl").read_text().splitlines()) == 3
    failed = json.loads((tmp_path / "ledger.jsonl").read_text().splitlines()[1])
    assert failed["status"] == "failed" and failed["tokens"] == 10  # the call happened, the parse failed
