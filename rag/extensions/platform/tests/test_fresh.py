"""--fresh: matrix reuse rule, hidden latency and cost for non-fresh runs, judge left cached."""
import json
from dataclasses import replace

from mlsrag import evaluate, matrix, results
from mlsrag.course import paths
from mlsrag.results import build_metrics, write_metrics

ROW = {"spans": [{"name": "generate", "duration_ms": 5.0}], "metrics": dict.fromkeys(results.METRIC_KEYS, 0.5),
       "cost": {"qwen": {"calls": 1, "prompt_tokens": 10, "completion_tokens": 5, "audio_seconds": 0.0}},
       "resources": {"wall_ms": 9.0, "cpu_seconds": 0.1, "peak_rss_mb": 10.0}}


def make_run(root, run_id, config_name, fresh):
    run_dir = root / run_id
    run_dir.mkdir(parents=True)
    config = matrix.resolve_config(config_name)
    (run_dir / "config.json").write_text(config.model_dump_json())
    info = {"run_id": run_id, "config_name": config.name, "subset": "practice-smoke20",
            "started": "x", "finished": "x", "git_commit": None, "fresh": fresh}
    write_metrics(run_dir, build_metrics([ROW], info))
    return run_dir


def setup(tmp_path, monkeypatch, configs):
    root = tmp_path / "results"
    monkeypatch.setattr(matrix, "paths", replace(paths, results_dir=root))
    est = {"config": "c", "subset": "s", "queries": 1, "elm_calls": 0, "audio_seconds": 0.0, "est_usd": 0.0,
           "calls": {"generation": 0, "judge": 0, "flow": 0, "retriever": 0, "whisper": 0}}
    monkeypatch.setattr(evaluate, "estimate", lambda *a, **k: dict(est))
    judged = []
    monkeypatch.setattr(evaluate, "judge", lambda *a, **k: judged.append((a, k)))
    spec = tmp_path / "m.yaml"
    spec.write_text(f"name: t\nsubset: practice-smoke20\nconfigs: [{', '.join(configs)}]\n")
    return root, spec, judged


def test_fresh_matrix_reruns_a_non_fresh_run_and_judge_stays_cached(tmp_path, monkeypatch):
    root, spec, judged = setup(tmp_path, monkeypatch, ["baseline"])
    make_run(root, "old", "baseline", fresh=False)
    ran = []

    def fake_run(config, subset, generate, allow_full, fresh):
        ran.append(fresh)
        return make_run(root, "new", "baseline", fresh), None

    monkeypatch.setattr(matrix.runner, "run", fake_run)
    out = matrix.run_matrix(spec, yes=True, fresh=True)
    assert ran == [True]
    assert judged == [((root / "new",), {})]  # no fresh flag reaches the judge
    row = json.loads((out / "summary.json").read_text())["rows"][0]
    assert row["run"] == "new" and row["fresh"] and row["latency_ms"] == 9.0


def test_summary_hides_latency_and_cost_of_non_fresh_runs(tmp_path, monkeypatch):
    root, spec, _ = setup(tmp_path, monkeypatch, ["baseline", "bm25"])
    make_run(root, "a", "baseline", fresh=True)
    make_run(root, "b", "bm25", fresh=False)
    monkeypatch.setattr(matrix.runner, "run", lambda *a, **k: (_ for _ in ()).throw(AssertionError("reuse")))
    out = matrix.run_matrix(spec, yes=True)  # without --fresh both runs are reused
    rows = json.loads((out / "summary.json").read_text())["rows"]
    assert rows[0]["latency_ms"] == 9.0 and rows[0]["calls/q"] == 1.0
    assert rows[1]["latency_ms"] is None and rows[1]["calls/q"] is None and rows[1]["NDCG@10"] == 0.5
    md = (out / "summary.md").read_text()
    b_line = next(line for line in md.splitlines() if line.startswith("| b "))
    assert b_line.endswith("| - | - | - | False |") and "not a fresh run" in md


def test_old_metrics_without_fresh_count_as_not_fresh(tmp_path):
    run_dir = make_run(tmp_path, "r", "baseline", fresh=False)
    m = json.loads((run_dir / "metrics.json").read_text())
    del m["run"]["fresh"]
    (run_dir / "metrics.json").write_text(json.dumps(m))
    assert results.summary_row(results.load_run(run_dir))["latency_ms"] is None
