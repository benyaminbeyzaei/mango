"""final.py: the reranker choice rule, paired differences, summary counts, slim rows."""
from mlsrag.final import choose_reranker, paired, slim_row, summary_levels, top5_moves


def setting(config, ndcg, ms):
    return {"config": config, "NDCG@10": {"mean": ndcg}, "latency_ms": {"end_to_end": {"mean": ms}}}


def test_choice_takes_the_fastest_setting_within_0005_of_the_best():
    settings = [setting("a", 0.500, 900), setting("b", 0.496, 100), setting("c", 0.490, 10)]
    assert choose_reranker(settings) == "b"
    assert choose_reranker([setting("a", 0.5, 900), setting("c", 0.49, 10)]) == "a"


def q(qid, ids, score):
    return {"query_id": qid, "context_ids": ids, "correctness": score, "grounding": score,
            "completeness": score, "answer_norm": score / 4}


def test_paired_difference_on_all_and_differing_questions():
    base = [q("1", ["a", "b"], 2), q("2", ["c"], 1)]
    rows = [q("1", ["a", "b"], 3), q("2", ["d"], 3)]  # question 2 has a different context
    out = paired(rows, base)
    assert out["context_differs"] == 1
    assert out["all"]["correctness"]["mean"] == 1.5
    assert out["differing"]["correctness"] == {"mean": 2.0, "ci95": [2.0, 2.0]}


def test_paired_with_no_differing_question():
    assert paired([q("1", ["a"], 2)], [q("1", ["a"], 1)])["differing"]["grounding"] is None


def test_summary_levels():
    rows = [{"context_ids": ["r1", "summary:t3"]}, {"context_ids": ["summary:s1", "summary:t2"]},
            {"context_ids": ["r2"]}]
    assert summary_levels(rows) == {"any": 2, "topic": 2, "section": 1}


def test_slim_row_splits_rerank_out_of_retrieve():
    row = {"query_id": "q", "artifacts": {"context_ids": ["r1"], "prompt": "long"},
           "spans": [{"name": "retrieve", "duration_ms": 10.0}, {"name": "rerank", "duration_ms": 7.0},
                     {"name": "asr", "duration_ms": 1.0}],
           "cost": {"m": {"calls": 1, "prompt_tokens": 5, "completion_tokens": 2, "audio_seconds": 0.0}},
           "counters": {"rerank.pool": 20, "rerank.truncated": 3}, "resources": {"wall_ms": 12.0},
           "metrics": dict.fromkeys(["Recall@5", "Recall@10", "Recall@20", "Recall@100", "MRR@10", "NDCG@10"], 1.0)}
    s = slim_row(row)
    assert s["ms"] == {"asr": 1.0, "encode": None, "dense_retrieve": 3.0, "rerank": 7.0, "generate": None,
                       "end_to_end": 12.0}
    assert (s["llm_calls"], s["prompt_tokens"], s["rerank_truncated"]) == (1, 5, 3)
    assert "prompt" not in s


def test_top5_moves():
    base = [{"query_id": "1", "context_ids": ["a", "b"]}, {"query_id": "2", "context_ids": ["c"]}]
    rows = [{"query_id": "1", "context_ids": ["b", "x"]}, {"query_id": "2", "context_ids": ["d"]}]
    qrels = {"1": {"a": 1}, "2": {"d": 1}}
    assert top5_moves(rows, base, qrels) == {"relevant_pushed_out": 1, "questions_losing": 1,
                                             "relevant_pulled_in": 1, "questions_gaining": 1}
