import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mlsrag.preprocess import ledger, summarise
from mlsrag.retrievers.raptor import SummaryData, SummaryRetriever
from mlsrag.retrievers.toc import SECTIONS_PROMPT, TOPICS_PROMPT, TocRetriever, parse_ids
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
    trace = Trace("r", "q", "c")
    hits = TocRetriever(toy_toc(), vectors, ids, sections=1, m=1).search(np.array([0.1, 1, 0]), k=10, trace=trace)
    assert [r for r, _ in hits] == ["c", "d"]
    assert trace.counters == {"toc.sections_probed": 1, "toc.topics_probed": 1, "toc.candidates_scanned": 2}


def test_toc_llm_route_follows_the_replies_and_counts_calls():
    vectors, ids = toy_corpus()
    g = ScriptedGateway(['{"ids": [1]}', '{"ids": [2]}'])
    hits = TocRetriever(toy_toc(), vectors, ids, sections=2, m=4, route="llm", gateway=g).search(
        np.array([1.0, 0, 0]), k=10, trace=g.trace, question="where is z")  # the vector points at x; the LLM's choice (z) wins
    assert [r for r, _ in hits] == ["f", "e"]
    assert "where is z" in g.prompts[0] and "2. Z: z" in g.prompts[1] and "0. X" not in g.prompts[1]
    assert g.trace.counters["toc.llm_calls"] == 2
    assert g.trace.counters["toc.topics_probed"] == 1


def test_toc_llm_route_falls_back_to_centroids_on_bad_reply():
    vectors, ids = toy_corpus()
    g = ScriptedGateway(["I don't know", '{"ids": [7]}'])
    hits = TocRetriever(toy_toc(), vectors, ids, sections=1, m=1, route="llm", gateway=g).search(
        np.array([1.0, 0, 0]), k=10, trace=g.trace, question="q")
    assert [r for r, _ in hits] == ["a", "b"]
    assert g.trace.counters["toc.llm_fallbacks"] == 2


def test_toc_llm_prompts_unchanged():  # prompt text is part of the cache key
    assert SECTIONS_PROMPT == (
        "Question: {question}\n\nA document collection is organised into these sections:\n{options}\n\n"
        "Which sections (at most {n}) most likely contain the answer? "
        'Reply with JSON only: {{"ids": [<section numbers>]}}')
    assert TOPICS_PROMPT == (
        "Question: {question}\n\nThese are the topics of the most likely sections:\n{options}\n\n"
        "Which topics (at most {n}) most likely contain the answer? "
        'Reply with JSON only: {{"ids": [<topic numbers>]}}')


def test_toc_llm_topics_route_lists_each_sections_topic_names():
    vectors, ids = toy_corpus()
    g = ScriptedGateway(['{"ids": [1]}', '{"ids": [2]}'])
    hits = TocRetriever(toy_toc(), vectors, ids, sections=2, m=4, route="llm-topics", gateway=g).search(
        np.array([1.0, 0, 0]), k=10, trace=g.trace, question="where is z")
    assert [r for r, _ in hits] == ["f", "e"]
    assert "0. XY: x and y\n   Topics: X; Y\n1. Z: z\n   Topics: Z" in g.prompts[0]
    assert g.prompts[1].startswith("Question: where is z\n\nThese are the topics")  # topic step as in llm
    assert g.trace.counters["toc.llm_calls"] == 2


def test_toc_llm_flat_route_one_call_over_all_topics():
    vectors, ids = toy_corpus()
    g = ScriptedGateway(['{"ids": [1, 2]}'])
    hits = TocRetriever(toy_toc(), vectors, ids, m=4, route="llm-flat", gateway=g).search(
        np.array([1.0, 0, 0]), k=10, trace=g.trace, question="q")
    assert len(g.prompts) == 1 and "0. X: x\n1. Y: y\n2. Z: z" in g.prompts[0]
    assert sorted(r for r, _ in hits) == ["c", "d", "e", "f"]  # members of topics 1 and 2
    assert g.trace.counters["toc.llm_calls"] == 1
    assert g.trace.counters["toc.candidates_scanned"] == 4


def test_toc_llm_flat_route_falls_back_to_centroid_route():
    vectors, ids = toy_corpus()
    g = ScriptedGateway(["no idea"])
    hits = TocRetriever(toy_toc(), vectors, ids, sections=1, m=1, route="llm-flat", gateway=g).search(
        np.array([1.0, 0, 0]), k=10, trace=g.trace, question="q")
    assert [r for r, _ in hits] == ["a", "b"]
    assert g.trace.counters["toc.llm_fallbacks"] == 1


def toy_summaries():
    # summary 0 sits near the z records; summary 1 near x
    return np.array([[0, 0.2, 1], [1, 0, 0]], dtype=np.float32), [
        {"id": 0, "level": 1, "summary": "about z", "members": ["e", "c"]},
        {"id": 1, "level": 1, "summary": "about x", "members": ["a", "b"]},
    ]


def test_summary_hit_expands_to_its_best_members():
    vectors, ids = toy_corpus()
    sv, rows = toy_summaries()
    sv /= np.linalg.norm(sv, axis=1, keepdims=True)
    trace = Trace("r", "q", "c")
    q = np.array([0, 0.2, 1.0])  # the summary scores 1.0, above every record
    hits = SummaryRetriever(sv, rows, vectors, ids, expand_top=2).search(q, k=4, trace=trace)
    # flat order would be e, f, d, c; the summary (members e, c) puts both first
    assert [r for r, _ in hits] == ["e", "c", "f", "d"]
    assert trace.counters["raptor.summary_hits"] >= 1


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


# RAPTOR-tree

def test_raptor_lite_prompt_and_stage_hashes_unchanged():
    import hashlib

    from mlsrag.preprocess import PIPELINES, load_pipeline_config

    assert hashlib.sha256(summarise.PROMPT_V1.encode()).hexdigest()[:16] == "b106d0599128eaa1"
    config = load_pipeline_config(Path(__file__).parents[1] / "configs/pipelines/topics-k256.yaml")
    hashes = PIPELINES["raptor"]().hashes(config)
    assert (hashes["summarise"], hashes["summary_index"]) == ("182308c8a5898ec3", "ec8b28b76c03cb6d")


def write_json(path, obj):
    path.write_text(json.dumps(obj), encoding="utf-8")
    return path


def test_soft_members_join_second_centroid_within_delta(tmp_path, monkeypatch):
    from mlsrag.preprocess import tree

    vectors = {"a": [1, 0], "b": [0.8, 0.6], "c": [0, 1], "d": [0.3, 0.954]}
    monkeypatch.setattr(tree, "load_vectors", lambda rids: np.array([vectors[r] for r in rids], dtype=np.float32))
    cluster = tmp_path / "cluster"
    cluster.mkdir()
    np.save(cluster / "centroids.npy", np.eye(2, dtype=np.float32))
    write_json(cluster / "clusters.json", [{"cluster": 0, "members": ["a", "b"]}, {"cluster": 1, "members": ["c", "d"]}])
    counts = tree.soft_members_run({"cluster": cluster}, {"share": 0.25}, SimpleNamespace(out_dir=tmp_path))
    # cos1 - cos2: a 1, b .2, c 1, d .654; the 25 % quantile is .54, so only b joins its second cluster
    members = json.loads((tmp_path / "members.json").read_text())
    assert [m["members"] for m in members] == [["a", "b"], ["c", "d", "b"]]
    assert counts["records_with_second"] == 1 and abs(counts["delta"] - 0.54) < 0.01


class SummaryGateway:
    """Replies with a summary naming the prompt's first record line; embeds summaries with fixed vectors."""

    def __init__(self, vectors=None):
        self.prompts, self.trace, self.vectors = [], Trace("r", "q", "c"), vectors

    def chat(self, model, messages, temperature=0.0, max_tokens=None):
        self.prompts.append(messages[0]["content"])
        first = next(line for line in messages[0]["content"].splitlines() if line.startswith("- "))
        return "sum of " + first[2:]

    def embed(self, encoder, texts, role="query"):
        return self.vectors


def test_summarise_full_splits_a_cluster_over_the_cap_and_combines(tmp_path):
    from mlsrag.preprocess import tree

    select, soft = tmp_path / "select", tmp_path / "soft"
    select.mkdir()
    soft.mkdir()
    (select / "records.jsonl").write_text("".join(json.dumps({"record_id": r, "text": r * 40}) + "\n" for r in "abcde"))
    write_json(soft / "members.json", [{"cluster": 0, "members": ["a"]}, {"cluster": 1, "members": ["b", "c", "d", "e"]}])
    out = tmp_path / "out"
    out.mkdir()
    g = SummaryGateway()
    ctx = SimpleNamespace(out_dir=out, gateway=g, trace=g.trace, log=lambda *a: None)
    counts = tree.summarise_full_run({"select": select, "soft_members": soft}, {"model": "m", "max_input_tokens": 25}, ctx)
    # each text is 11 tokens: cluster 1 splits into [b, c] and [d, e], then one combine call
    assert counts["split_clusters"] == 1 and counts["calls_planned"] == 4 and len(g.prompts) == 4
    assert "summaries of parts of one cluster" in g.prompts[3] and "- sum of " + "b" * 40 in g.prompts[3]
    rows = [json.loads(line) for line in (out / "summaries.jsonl").read_text().splitlines()]
    assert [r["parts"] for r in rows] == [1, 2] and rows[1]["summary"] == "sum of sum of " + "b" * 40
    with pytest.raises(RuntimeError):
        tree.summarise_full_run({"select": select, "soft_members": soft}, {"model": "m", "max_input_tokens": 25,
                                                                           "max_calls": 3}, ctx)


def test_level2_hit_expands_into_its_section_records(tmp_path):
    from mlsrag.preprocess import tree

    vectors, ids = toy_corpus()
    d = {n: tmp_path / n for n in ("soft_members", "section", "summarise_full", "section_summarise", "out")}
    for p in d.values():
        p.mkdir()
    write_json(d["soft_members"] / "members.json", [{"cluster": 0, "members": ["a", "b"]},
                                                    {"cluster": 1, "members": ["c", "d", "e"]}])
    write_json(d["section"] / "sections.json", [{"section": 0, "topics": [0, 1]}])
    (d["summarise_full"] / "summaries.jsonl").write_text('{"cluster": 0, "summary": "x"}\n{"cluster": 1, "summary": "y"}\n')
    (d["section_summarise"] / "summaries.jsonl").write_text('{"section": 0, "summary": "xy"}\n')
    # level-1 summaries point nowhere useful; the level-2 summary sits on the query
    g = SummaryGateway(np.array([[0, -1, 0], [-1, 0, 0], [0, 0.6, 0.8]], dtype=np.float32))
    tree.tree_index_run({k: v for k, v in d.items() if k != "out"}, {}, SimpleNamespace(out_dir=d["out"], gateway=g))
    from mlsrag.retrievers.raptor import load_summaries

    sv, rows = load_summaries(d["out"])
    assert [r["id"] for r in rows] == ["t0", "t1", "s0"] and rows[2]["members"] == ["a", "b", "c", "d", "e"]
    trace = Trace("r", "q", "c")
    hits = SummaryRetriever(sv, rows, vectors, ids, expand_top=2).search(np.array([0, 0.6, 0.8]), k=2, trace=trace)
    assert [r for r, _ in hits] == ["e", "d"]  # a flat search gives e, f (f is not in the section)
    assert trace.counters["raptor.level2_hits"] == 1 and trace.counters["raptor.level1_hits"] == 0
    hits = SummaryRetriever(sv, rows, vectors, ids, summary_as_evidence=True).search(np.array([0, 0.6, 0.8]), k=1)
    assert hits[0][0] == "summary:s0"
    assert SummaryData(SimpleNamespace(record_text=lambda r: r), rows).record_text("summary:s0") == "xy"
