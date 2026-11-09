"""Plan 008: lazy graph, LLM knowledge graph and the record-subset restriction on toy data."""
import json
from types import SimpleNamespace

import numpy as np

from mlsrag.preprocess import kg, lazy_graph, phrases
from mlsrag.retrievers.graph import KGRetriever, LazyGraphRetriever
from mlsrag.config import baseline
from mlsrag.flows import get_flow
from mlsrag.runner import retrieval_metrics
from mlsrag.stages import Stages
from mlsrag.trace import Trace

TEXTS = {
    "r1": "A Roth IRA grows tax free. Vanguard offers a Roth IRA.",
    "r2": "Your Roth IRAs and your 401k are retirement accounts.",
    "r3": "The 401k plan from your employer may match contributions.",
    "r4": "Index funds at Vanguard have low fees.",
}
IDS = list(TEXTS)
VECS = np.eye(4, dtype=np.float32)  # record i points along axis i


def ctx(out_dir, gateway=None):
    trace = Trace("test", "-", "test")
    if gateway is not None:
        gateway.trace = trace
    return SimpleNamespace(out_dir=out_dir, gateway=gateway, trace=trace, log=lambda *a: None)


def write_records(d, ids=IDS):
    d.mkdir()
    (d / "records.jsonl").write_text(
        "".join(json.dumps({"record_id": r, "text": TEXTS[r]}) + "\n" for r in ids), encoding="utf-8")
    return d


def test_normalise_merges_inflections_and_drops_stop_words():
    nlp = phrases.load_nlp()
    assert phrases.normalise(nlp("the Roth IRAs")) == phrases.normalise(nlp("Roth IRA")) == "roth ira"
    assert "roth ira" in phrases.question_ngrams("What is a Roth IRA?")
    long = "Federal Deposit Insurance Corporation savings account rules"
    assert phrases.normalise(nlp(long)) in phrases.question_ngrams(f"What are the {long}?")


def test_lazy_graph_build_and_search(tmp_path):
    sel = write_records(tmp_path / "select")
    for name in ("phrases", "graph", "index"):
        (tmp_path / name).mkdir()
    phrases.run({"select": sel}, {}, ctx(tmp_path / "phrases"))
    phrases.run({"select": sel}, {}, ctx(tmp_path / "phrases"))  # resume: nothing left to do
    assert len((tmp_path / "phrases" / "phrases.jsonl").read_text().splitlines()) == 4

    stats = lazy_graph.run_graph({"phrases": tmp_path / "phrases"}, {}, ctx(tmp_path / "graph"))
    assert stats["nodes"] >= 3 and stats["edges"] >= 1  # roth ira, vanguard, 401k occur twice
    lazy_graph.run_index({"phrases": tmp_path / "phrases", "lazy_graph": tmp_path / "graph"}, {},
                         ctx(tmp_path / "index"))
    index = json.loads((tmp_path / "index" / "index.json").read_text())
    assert sorted(index["nodes"]["roth ira"]["records"]) == ["r1", "r2"]

    trace = Trace("t", "q", "c")
    hits = LazyGraphRetriever(index, VECS, IDS).search(VECS[1], 10, trace=trace, question="Should I open a Roth IRA?")
    assert [h[0] for h in hits] == ["r2", "r1"]  # equal graph score, r2 is closer to the query vector
    assert trace.counters["lazy.nodes_matched"] >= 1 and trace.counters["lazy.candidates"] == 2


class FakeGateway:
    trace = None

    def __init__(self, replies):
        self.replies = replies
        self.calls = 0

    def chat(self, model, messages, temperature=0.0, max_tokens=None):
        self.calls += 1
        self.trace.add_cost(prompt_tokens=100, completion_tokens=20, model=model)
        text = messages[0]["content"]
        return next(v for k, v in self.replies.items() if k in text)


GRAPHS = {
    "Vanguard offers": {"entities": [{"name": "Roth IRA", "type": "account"}, {"name": "Vanguard", "type": "org"}],
                        "relations": [["Vanguard", "offers", "Roth IRA"]]},
    "retirement accounts": {"entities": [{"name": "Roth IRAs", "type": "account"}, {"name": "401k", "type": "account"}],
                            "relations": [["401k", "is a", "retirement account"]]},
    "employer may match": '{"error": "cannot help"}',  # valid JSON, wrong shape: a failure
    "Index funds": {"entities": [{"name": "Vanguard", "type": "org"}], "relations": []},
}


def test_kg_extract_respects_the_call_budget_and_index(tmp_path, monkeypatch):
    sel = write_records(tmp_path / "select")
    (tmp_path / "kg").mkdir()
    (tmp_path / "index").mkdir()
    replies = {k: v if isinstance(v, str) else json.dumps(v) for k, v in GRAPHS.items()}
    gw = FakeGateway(replies)
    monkeypatch.setattr(kg, "MAX_CALLS", 4)
    counts = kg.run_extract({"select": sel}, {"model": "m"}, ctx(tmp_path / "kg", gw))
    assert counts["ok"] == 3 and counts["failed_records"] == 1 and counts["mean_tokens_per_record"] == 120
    assert counts["failed_attempt_tokens"] == 120
    kg.run_extract({"select": sel}, {"model": "m"}, ctx(tmp_path / "kg", gw))  # budget spent: no retry
    (tmp_path / "kg-other-model").mkdir()  # another config's output shares the same budget
    kg.run_extract({"select": sel}, {"model": "m2"}, ctx(tmp_path / "kg-other-model", gw))
    assert gw.calls == 4
    monkeypatch.setattr(kg, "MAX_CALLS", 3)  # 4 calls spent, budget now negative: still no call
    (tmp_path / "kg-overspent").mkdir()
    sel3 = write_records(tmp_path / "select3", ids=IDS[:3])
    kg.run_extract({"select": sel3}, {"model": "m3"}, ctx(tmp_path / "kg-overspent", gw))
    assert gw.calls == 4

    kg.run_index({"kg_extract": tmp_path / "kg"}, {}, ctx(tmp_path / "index"))
    index = json.loads((tmp_path / "index" / "index.json").read_text())
    assert index["entities"]["roth ira"]["records"] == ["r1", "r2"]  # "Roth IRAs" resolved by lemma
    assert "vanguard" in index["entities"]["roth ira"]["neighbours"]

    t = Trace("t", "q", "c")
    r = KGRetriever(index, VECS, IDS)
    hits = [h[0] for h in r.search(VECS[3], 10, trace=t, question="Which accounts does Vanguard offer?")]
    assert hits[:2] == ["r1", "r4"]  # both match "vanguard"; r1 also holds the 1-hop "roth ira"
    assert "r2" in hits  # via the 1-hop neighbour "roth ira"
    assert t.counters["kg.entities_matched"] == 1

    gw2 = FakeGateway({"Question:": '{"entities": ["401k", "nonexistent thing"]}'})
    gw2.trace = Trace("t", "q", "c")
    r = KGRetriever(index, VECS, IDS, question_entities="llm", model="m", gateway=gw2)
    assert [h[0] for h in r.search(VECS[0], 10, question="anything")][0] == "r2"
    assert gw2.calls == 1


def test_record_subset_restricts_candidates_and_qrels(tmp_path):
    class Fake:
        def transcribe(self, audio_path):
            return "q"

        def encode(self, text):
            return VECS[0]

        def search(self, query_vec, k, trace=None):  # no uses_question: must not be given the question
            self.k = k
            return [(r, 1.0 - i / 10) for i, r in enumerate(IDS)][:k]

        def build(self, question, record_ids, trace=None):
            return ""

    (tmp_path / "subset.txt").write_text("r2\nr4\n", encoding="utf-8")
    f = Fake()
    stages = Stages(asr=f, encoder=f, retriever=f, context=f, generator=None)
    config = baseline().model_copy(update={"params": {"record_subset": str(tmp_path / "subset.txt")}})
    t = Trace("t", "q", "c")
    a = get_flow("standard")("q", "q.wav", stages, config, t)
    assert a["ranked_ids"] == ["r2", "r4"] and f.k > 100  # searched every candidate, then filtered
    assert t.counters == {"subset.candidates_in": 4, "subset.candidates_kept": 2}

    rel = {"r1": 1, "r2": 1}
    assert retrieval_metrics(a["ranked_ids"], rel, frozenset({"r2", "r4"}))["Recall@5"] == 1.0
    assert retrieval_metrics(a["ranked_ids"], rel)["Recall@5"] == 0.5
