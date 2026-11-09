import json

import pytest
from pydantic import ValidationError

from mlsrag.results import Metrics, bootstrap_ci, build_metrics, compare, list_runs, load_run, write_metrics

INFO = {"run_id": "r1", "config_name": "c", "subset": "practice-smoke20",
        "started": "2026-11-07T17:00:00", "finished": None, "git_commit": None}


def row(wall_ms, generate_ms=None, cost=None):
    spans = [{"name": "encode", "duration_ms": 2.0}, {"name": "gateway.embed", "duration_ms": 1.0}]
    if generate_ms is not None:
        spans.append({"name": "generate", "duration_ms": generate_ms})
    metrics = {k: 0.5 for k in ["Recall@5", "Recall@10", "Recall@20", "Recall@100", "MRR@10", "NDCG@10"]}
    return {"spans": spans, "cost": cost or {}, "metrics": metrics,
            "resources": {"wall_ms": wall_ms, "cpu_seconds": 0.5, "peak_rss_mb": 100.0}}


def test_build_metrics_fills_the_schema():
    whisper = {"whisper-1": {"calls": 1, "prompt_tokens": 0, "completion_tokens": 0, "audio_seconds": 60.0}}
    chat = {"qwen": {"calls": 1, "prompt_tokens": 10, "completion_tokens": 5, "audio_seconds": 0.0}}
    m = build_metrics([row(10.0, 8.0, whisper), row(30.0, 20.0, chat)], INFO)
    assert m["retrieval"]["Recall@10"] == 0.5
    assert m["answers"] is None
    assert m["cost"] == {"elm_calls": 2, "prompt_tokens": 10, "completion_tokens": 5,
                         "audio_seconds": 60.0, "est_usd": pytest.approx(0.006)}
    assert m["latency_ms"]["encode"] == 2.0 and m["latency_ms"]["generate"] == 14.0
    assert m["latency_ms"]["asr"] is None  # stage never ran
    assert m["latency_ms"]["end_to_end"]["mean"] == 20.0
    assert m["resources"] == {"cpu_s": 1.0, "peak_rss_mb": 100.0}
    assert m["run"]["n_queries"] == 2


def test_schema_rejects_unknown_and_missing_keys(tmp_path):
    m = build_metrics([row(10.0)], INFO)
    with pytest.raises(ValidationError):
        Metrics.model_validate({**m, "extra": 1})
    with pytest.raises(ValidationError):
        Metrics.model_validate({**m, "retrieval": {**m["retrieval"], "Recall@1": 0.5}})
    del m["retrieval"]["MRR@10"]
    with pytest.raises(ValidationError):
        write_metrics(tmp_path, m)


def test_compare_puts_runs_side_by_side(tmp_path):
    for name in ("a", "b"):
        d = tmp_path / name
        d.mkdir()
        write_metrics(d, build_metrics([row(10.0)], {**INFO, "run_id": name}))
        (d / "config.json").write_text(json.dumps({"name": "c"}))
    assert list_runs(tmp_path) == [tmp_path / "a", tmp_path / "b"]
    table = compare(list_runs(tmp_path)).splitlines()
    assert table[0] == "| metric | a | b |"
    assert "| Recall@10 | 0.500 | 0.500 |" in table
    assert "| answer_norm | - | - |" in table


@pytest.mark.parametrize("run_dir", list_runs(), ids=lambda p: p.name)
def test_every_saved_run_follows_the_schema(run_dir):
    run = load_run(run_dir)  # validates metrics.json
    assert run["metrics"]["run"]["run_id"] == run_dir.name


def test_bootstrap_ci_of_a_coin_flip_metric():
    lo, hi = bootstrap_ci([0.0, 1.0] * 50)  # mean 0.5, standard error 0.05
    assert [lo, hi] == pytest.approx([0.4, 0.6], abs=1e-9)  # mean +- 1.96 standard errors, as expected
    assert bootstrap_ci([0.0, 1.0] * 50) == [lo, hi]  # fixed seed


def test_bootstrap_ci_of_a_constant_metric_has_zero_width():
    assert bootstrap_ci([0.3] * 20) == [pytest.approx(0.3)] * 2
