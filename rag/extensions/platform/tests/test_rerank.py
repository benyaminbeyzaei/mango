"""The rerank retriever (order, pool, tail) and gateway.rerank (cache, --fresh, truncation count)."""
import pytest

from mlsrag import gateway as gw_module
from mlsrag.gateway import ModelGateway
from mlsrag.retrievers.rerank import RerankRetriever
from mlsrag.trace import Trace


class FakeBase:
    def __init__(self, n):
        self.ids = [f"r{i}" for i in range(n)]

    def search(self, query_vec, k, trace=None):
        return [(rid, 1.0 - i / 100) for i, rid in enumerate(self.ids[:k])]


class FakeData:
    def record_text(self, rid):
        return f"text {rid}"


class FakeScorer:
    """Scores a record by the number in its id, so r9 > r8 > ... (the reverse of the base order)."""

    def __init__(self):
        self.calls = []

    def rerank(self, model, query, texts, revision=None):
        self.calls.append(texts)
        return [float(t.split("r")[-1]) for t in texts], 1


def test_pool_is_reordered_and_the_tail_keeps_the_base_order():
    trace = Trace("r", "q", "c")
    hits = RerankRetriever(FakeBase(10), FakeData(), FakeScorer(), "m", pool=4).search(None, 8, trace, "q?")
    assert [rid for rid, _ in hits] == ["r3", "r2", "r1", "r0", "r4", "r5", "r6", "r7"]
    assert [s.name for s in trace.spans] == ["rerank"]
    assert trace.counters == {"rerank.pool": 4, "rerank.truncated": 1}


def test_pool_larger_than_k_returns_k_and_asks_the_base_for_the_pool():
    scorer = FakeScorer()
    hits = RerankRetriever(FakeBase(100), FakeData(), scorer, "m", pool=20).search(None, 5, None, "q?")
    assert [rid for rid, _ in hits] == ["r19", "r18", "r17", "r16", "r15"]
    assert len(scorer.calls[0]) == 20


def test_base_shorter_than_pool():
    hits = RerankRetriever(FakeBase(3), FakeData(), FakeScorer(), "m", pool=50).search(None, 100, None, "q?")
    assert [rid for rid, _ in hits] == ["r2", "r1", "r0"]


def test_question_reaches_a_question_aware_base():
    class QuestionBase(FakeBase):
        uses_question = True

        def search(self, query_vec, k, trace=None, question=None):
            self.question = question
            return super().search(query_vec, k, trace)

    base = QuestionBase(5)
    RerankRetriever(base, FakeData(), FakeScorer(), "m", pool=5).search(None, 5, None, "q?")
    assert base.question == "q?"


class FakeTokenizer:
    def __call__(self, query, text):
        return {"input_ids": [0] * len(text)}  # one token per character


class FakeCrossEncoder:
    tokenizer = FakeTokenizer()

    def __init__(self):
        self.calls = 0

    def predict(self, pairs, show_progress_bar=False):
        self.calls += 1
        return [float(len(t)) for _, t in pairs]


@pytest.fixture
def gateway(tmp_path, monkeypatch):
    encoder = FakeCrossEncoder()
    monkeypatch.setattr(gw_module, "_rerankers", {("m", None): encoder})
    g = ModelGateway(trace=Trace("r", "q", "c"), cache_dir=tmp_path / "cache", log_path=tmp_path / "log.jsonl")
    return g, encoder


def test_rerank_cache_hit_and_miss(gateway):
    g, encoder = gateway
    long_text = "x" * (gw_module.RERANK_MAX_LENGTH + 1)
    assert g.rerank("m", "q", ["ab", long_text]) == ([2.0, 513.0], 1)
    assert g.rerank("m", "q", ["ab", long_text]) == ([2.0, 513.0], 1)  # hit: same scores and count
    assert encoder.calls == 1
    g.rerank("m", "other query", ["ab"])  # a different query is a miss
    assert encoder.calls == 2
    assert [s.attrs["cached"] for s in g.trace.spans] == [False, True, False]
    assert g.trace.cost == {}  # local model: no ELM cost


def test_rerank_fresh_recomputes_and_keeps_the_cache_file(gateway):
    g, encoder = gateway
    g.rerank("m", "q", ["ab"])
    g.fresh = True
    g.rerank("m", "q", ["ab"])
    assert encoder.calls == 2
    assert len(list(g.cache_dir.glob("*.json"))) == 1
