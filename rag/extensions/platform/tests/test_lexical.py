from types import SimpleNamespace

import pytest

from mlsrag.config import RunConfig
from mlsrag.flows import get_flow
from mlsrag.retrievers.bm25 import BM25Retriever, tokenise
from mlsrag.retrievers.grep import GrepIndex
from mlsrag.stages import Stages, build_stages
from mlsrag.trace import Trace

TOY = SimpleNamespace(records={
    "r1": "A Roth IRA is funded with after-tax money.",
    "r2": "Pay off your credit card debt before you invest.",
    "r3": "Roth IRA or traditional IRA? The Roth IRA wins if your tax rate rises.",
    "r4": "Index funds have low fees.",
    "r5": "My cat sleeps all day.",
})


def test_tokenise_lowercases_and_drops_stopwords():
    assert tokenise("The Roth IRA, is it good?") == ["roth", "ira", "good"]


def test_bm25_ranks_toy_corpus_and_reuses_cache(tmp_path):
    r = BM25Retriever(TOY, cache_dir=tmp_path)
    assert "bm25.build_s" in r.build_counters and r.build_counters["bm25.index_mb"] > 0
    hits = r.search(None, 5, question="roth ira tax")  # the vector is ignored
    assert [rid for rid, _ in hits] == ["r3", "r1"]  # no zero-score records
    again = BM25Retriever(TOY, cache_dir=tmp_path)
    assert set(again.build_counters) == {"bm25.build_s", "bm25.index_mb"}
    trace = Trace("r", "q", "c")
    assert again.uses_question and again.search(None, 5, trace=trace, question="roth ira tax") == hits
    assert set(trace.counters) == {"bm25.build_s", "bm25.index_mb"} and again.build_counters == {}
    assert again.search(None, 5, question="") == [] and again.search(None, 5, question="is it the") == []


def test_grep_ranks_by_patterns_then_matches_then_length():
    index = GrepIndex(TOY)
    hits, n_matched, n_invalid = index.scan(["roth", "ira", "debt", "fund"], limit=10)
    # r3: 2 patterns, 5 matches; r1: 3 patterns (roth, ira, fund); r2: debt; r4: fund
    assert [h[0] for h in hits] == ["r1", "r3", "r4", "r2"]
    assert hits[0] == ("r1", 3, 3) and hits[1] == ("r3", 2, 5)
    assert n_matched == 4 and n_invalid == 0
    assert index.scan(["roth"], limit=1)[0] == [("r3", 1, 2)]


def test_grep_drops_and_counts_invalid_patterns():
    hits, n_matched, n_invalid = GrepIndex(TOY).scan(["cat", "(unclosed", "x*", ""], limit=10)
    assert n_invalid == 3 and [h[0] for h in hits] == ["r5"] and n_matched == 1


class FakeGateway:
    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    def encoder(self, name):  # build_stages loads the query encoder
        return None

    def chat(self, model, messages, temperature=0.0, max_tokens=None, tag=None, timeout=None):
        assert tag == "grep_v1"
        self.prompts.append(messages[0]["content"])
        return self.replies.pop(0)


class FakeASR:
    def transcribe(self, audio_path):
        return "is a roth ira good"


class FakeContext:
    def build(self, question, record_ids, trace=None):
        return ",".join(record_ids)


def run(flow, replies, k=2):
    gw = FakeGateway(replies)
    stages = Stages(asr=FakeASR(), encoder=None, retriever=GrepIndex(TOY), context=FakeContext(),
                    generator=None, gateway=gw)
    cfg = RunConfig(name="t", flow=flow, retriever="grep", k=k)
    trace = Trace("r", "q", "t")
    out = get_flow(flow)("q", "q.wav", stages, cfg, trace)
    return out, trace, gw


def test_grep_static_records_patterns_and_counts():
    out, trace, gw = run("grep-static", ['Here: ["roth", "ira"]'])
    assert out["patterns"] == [["roth", "ira"]]
    assert out["ranked_ids"] == ["r3", "r1"] and out["prompt"] == "r3,r1"
    c = trace.counters
    assert c["grep.llm_calls"] == 1 and c["grep.patterns"] == 2 and c["grep.matched_records"] == 2
    assert "grep.scan" in [s.name for s in trace.spans]


def test_grep_agentic_refines_then_cleans_up_within_caps():
    replies = ['["debt"]', '{"patterns": ["roth"]}', '{"patterns": ["ira", "tax"]}', '["r1", "zz", "r1", "r3"]']
    out, trace, gw = run("grep-agentic", replies)
    assert out["patterns"] == [["debt"], ["roth"], ["ira", "tax"]]
    assert out["kept_ids"] == ["r1", "r3"]  # unknown id ignored, duplicate dropped, at most k
    assert out["ranked_ids"][:2] == ["r1", "r3"] and out["prompt"] == "r1,r3"
    assert trace.counters["grep.iterations"] == 3 and trace.counters["grep.llm_calls"] == 4
    assert "r2" in gw.prompts[1]  # the refine prompt shows the hits


def test_grep_agentic_stops_when_model_says_so():
    out, trace, gw = run("grep-agentic", ['["roth"]', '{"stop": true}', '["r3"]'])
    assert trace.counters["grep.iterations"] == 1 and trace.counters["grep.llm_calls"] == 3
    assert out["ranked_ids"] == ["r3", "r1"] and out["prompt"] == "r3"  # context: kept ids only


def test_grep_feedback_keeps_nothing_and_returns_the_static_ranking():
    static, _, _ = run("grep-static", ['["roth"]'])
    out, trace, gw = run("grep-feedback", ['["roth"]', "[]"])
    assert out["ranked_ids"] == static["ranked_ids"] == ["r3", "r1"] and out["prompt"] == "r3,r1"
    assert out["kept_ids"] == [] and trace.counters["grep.llm_calls"] == 2 and trace.counters["grep.kept"] == 0


def test_grep_feedback_expands_from_kept_records_then_reranks(monkeypatch):
    monkeypatch.setattr("mlsrag.flows.grep_feedback.POOL", 2)  # keep/rerank see the top 2 of the toy corpus
    replies = ['["ira", "fund"]', '["r3"]', '["cat", "debt", "ira"]', '["r5", "zz"]']
    out, trace, gw = run("grep-feedback", replies)
    # static: r1, r3, r4 (top 2: r1, r3); kept r3; expand scan r3, r5, r1, r2 minus seen r1, r3 -> r5, r2
    assert out["kept_ids"] == ["r3"] and out["patterns"] == [["ira", "fund"], ["cat", "debt", "ira"]]
    assert TOY.records["r3"][:30] in gw.prompts[2] and TOY.records["r1"][:30] not in gw.prompts[2]
    assert "[r5]" in gw.prompts[3] and "[r1]" not in gw.prompts[3]  # rerank pool: kept + new hits
    # reranked, rest of the pool, rest of static, rest of the expansion
    assert out["ranked_ids"] == ["r5", "r3", "r1", "r4", "r2"] and out["prompt"] == "r5"
    assert out["context_ids"] == ["r5"]  # what the candidates file reports as retrieved
    assert out["scores"] == [1.0, 1.0, 2.0, 1.0, 1.0]
    assert trace.counters["grep.llm_calls"] == 4 and trace.counters["grep.kept"] == 1


def test_grep_feedback_keep_ignores_ids_outside_the_shown_candidates(monkeypatch):
    monkeypatch.setattr("mlsrag.flows.grep_feedback.POOL", 2)
    out, trace, gw = run("grep-feedback", ['["ira", "fund"]', '["zz", "r4", "r1", "r1"]', "no idea", "no idea"])
    assert out["kept_ids"] == ["r1"]  # r4 is a hit but was not shown; zz is unknown; duplicate dropped
    assert out["prompt"] == "r1,r3"  # rerank reply unusable: context falls back to the top k


def test_grep_feedback_empty_rerank_reply_gives_an_empty_context():
    out, trace, gw = run("grep-feedback", ['["roth"]', '["r1"]', '["tax"]', "[]"])
    assert out["reranked_ids"] == [] and out["prompt"] == "" and out["ranked_ids"][0] == "r1"


def test_build_stages_registers_grep(tmp_path):
    cfg = RunConfig(name="t", retriever="grep", flow="grep-static", asr_cache=str(tmp_path / "t.jsonl"))
    stages = build_stages(cfg, FakeGateway([]), TOY)
    assert isinstance(stages.retriever, GrepIndex) and stages.gateway is not None


def test_grep_agentic_context_falls_back_to_top_k_when_cleanup_reply_is_not_a_list():
    out, trace, gw = run("grep-agentic", ['["roth"]', '{"stop": true}', "no idea"])
    assert out["kept_ids"] is None and out["prompt"] == "r3,r1"


def test_grep_retriever_only_with_grep_flows(tmp_path):
    cfg = RunConfig(name="t", retriever="grep", flow="standard", asr_cache=str(tmp_path / "t.jsonl"))
    with pytest.raises(ValueError, match="grep"):
        build_stages(cfg, FakeGateway([]), TOY)


def test_standard_flow_gives_bm25_the_question_and_reports_the_build_once(tmp_path):
    class Encoder:
        def encode(self, text):
            raise AssertionError("BM25 needs no query vector")

    bm25 = BM25Retriever(TOY, cache_dir=tmp_path)
    stages = Stages(asr=FakeASR(), encoder=Encoder(), retriever=bm25, context=FakeContext(), generator=None)
    cfg = RunConfig(name="t", retriever="bm25", k=2)
    first, second = Trace("r", "q1", "t"), Trace("r", "q2", "t")
    out = get_flow("standard")("q1", "q1.wav", stages, cfg, first)
    get_flow("standard")("q2", "q2.wav", stages, cfg, second)
    assert out["ranked_ids"] == ["r3", "r1"] and out["context_ids"] == ["r3", "r1"]  # "is a roth ira good"
    assert "bm25.build_s" in first.counters and "bm25.build_s" not in second.counters
    assert "encode" not in [s.name for s in first.spans]
