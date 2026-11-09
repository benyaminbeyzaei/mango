import json
import shutil
from dataclasses import replace

import pytest

from mlsrag import evaluate, matrix
from mlsrag.config import baseline
from mlsrag.course import paths
from mlsrag.evaluate import estimate, read_jsonl, write_candidates
from mlsrag.results import list_runs, load_run
from mlsrag.subsets import load_subset

from conftest import needs_data


def write_matrix(tmp_path, subset):
    path = tmp_path / "m.yaml"
    path.write_text(f"name: t\nsubset: {subset}\nconfigs: [baseline]\n")
    return path


def test_matrix_refuses_full_split_without_flag(tmp_path):
    with pytest.raises(ValueError, match="--allow-full"):
        matrix.run_matrix(write_matrix(tmp_path, "practice-all"), yes=True)


@needs_data
def test_matrix_asks_before_running(tmp_path, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "n")
    monkeypatch.setattr(matrix.runner, "run", lambda *a, **k: pytest.fail("ran without a yes"))
    with pytest.raises(SystemExit, match="aborted"):
        matrix.run_matrix(write_matrix(tmp_path, "practice-smoke20"))


@needs_data
def test_matrix_override_applies_to_every_config(tmp_path, monkeypatch):
    path = tmp_path / "m.yaml"
    path.write_text("name: t\nsubset: dev-20pct\nconfigs: [baseline, bm25]\n"
                    "override: {asr_cache: rag/.cache/transcripts/whisper1-dev.jsonl, context_builder: budgeted}\n")
    seen, real = [], evaluate.estimate
    monkeypatch.setattr(evaluate, "estimate", lambda c, *a, **k: seen.append(c) or real(c, *a, **k))
    monkeypatch.setattr("builtins.input", lambda _: "n")
    with pytest.raises(SystemExit, match="aborted"):
        matrix.run_matrix(path)
    assert seen[0].retriever == "exact" and seen[0].context_builder == "budgeted"
    assert seen[0].asr_cache.endswith("whisper1-dev.jsonl")
    assert seen[0].name == "baseline+context_builder=budgeted"  # asr_cache is not in the name


def test_matrix_override_name_is_a_folder_name():
    config = matrix.with_override(baseline(), {"generator_model": "meta-llama/Llama-3.3-70B-Instruct"})
    assert config.name == "baseline+generator_model=meta-llama_Llama-3.3-70B-Instruct"


@needs_data
def test_every_flow_has_a_call_estimate():
    from mlsrag.flows import FLOWS

    assert set(evaluate.FLOW_CALLS_PER_QUERY) == set(FLOWS)


def test_estimate_counts_generation_and_judge_calls():
    e = estimate(baseline(), "practice-20pct", judge=True)
    assert e["calls"]["generation"] == 70 and e["calls"]["judge"] == 70
    assert e["calls"]["flow"] == 0


@needs_data
def test_candidates_follow_the_course_format(tmp_path):
    generated = [d for d in list_runs() if load_run(d)["metrics"]["latency_ms"]["generate"] is not None]
    if not generated:
        pytest.skip("no saved run with answers")
    run_dir = tmp_path / "run"
    shutil.copytree(generated[0], run_dir)
    rows = read_jsonl(write_candidates(run_dir))
    k = json.loads((run_dir / "config.json").read_text())["k"]
    assert set(rows[0]) == {"query_id", "split", "audio_file", "transcript", "answer",
                            "retrieved_record_ids", "timing"}
    assert all(len(r["retrieved_record_ids"]) == k and r["split"] == "practice" for r in rows)
    assert (paths.data_dir / rows[0]["audio_file"]).exists()


@needs_data
def test_candidates_list_the_ids_the_context_was_built_from(tmp_path):
    qids = load_subset("practice-smoke20")[:2]
    ranked = [f"r{i}" for i in range(10)]
    rows = [  # a grep-feedback trace whose rerank kept 2 ids, and an older trace without context_ids
        {"query_id": qids[0], "spans": [], "resources": {"wall_ms": 1},
         "artifacts": {"transcript": "q", "answer": "a", "ranked_ids": ranked, "context_ids": ["r7", "r2"]}},
        {"query_id": qids[1], "spans": [], "resources": {"wall_ms": 1},
         "artifacts": {"transcript": "q", "answer": "a", "ranked_ids": ranked}},
    ]
    (tmp_path / "config.json").write_text(json.dumps({"k": 5}))
    (tmp_path / "metrics.json").write_text(json.dumps({"run": {"subset": "practice-smoke20"}}))
    (tmp_path / "queries.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    out = read_jsonl(write_candidates(tmp_path))
    assert out[0]["retrieved_record_ids"] == ["r7", "r2"]
    assert out[1]["retrieved_record_ids"] == ranked[:5]


def generated_run(tmp_path):
    runs = [d for d in list_runs() if load_run(d)["metrics"]["latency_ms"]["generate"] is not None]
    if not runs:
        pytest.skip("no saved run with answers")
    run_dir = tmp_path / "results" / runs[0].name
    shutil.copytree(runs[0], run_dir, ignore=shutil.ignore_patterns("judge"))
    return run_dir


@needs_data
def test_judge_calls_the_course_evaluator_and_stores_means(tmp_path, monkeypatch):
    run_dir = generated_run(tmp_path)
    seen = {}

    def fake_run(cmd, env, check):  # stands in for rag/scripts/evaluate_answers.py
        seen["cmd"], seen["env"] = cmd, env
        out = run_dir / "judge"
        out.mkdir()
        (out / "judge_cache.jsonl").write_text('{"x": 1}\n{"x": 2}\n')
        (out / "answer_judge_summary.json").write_text(json.dumps({
            "judge_model": env["ELM_JUDGE_MODEL"], "queries": 2, "repeats": 1,
            "mean_correctness": 4.0, "mean_grounding": 2.0, "mean_completeness": 0.0,
            "mean_normalised_answer_score": 0.5}))

    monkeypatch.setattr(evaluate.subprocess, "run", fake_run)
    answers = evaluate.judge(run_dir, judge_model="judge-x")
    assert seen["cmd"][1].endswith("rag/scripts/evaluate_answers.py")
    assert "--allow-subset" in seen["cmd"] and seen["env"]["ELM_JUDGE_MODEL"] == "judge-x"
    assert answers["normalized"] == 0.5 and answers["judge_calls"] == 2
    assert load_run(run_dir)["metrics"]["answers"] == answers


@needs_data
def test_matrix_reuses_a_judged_run_and_writes_summary(tmp_path, monkeypatch):
    run_dir = generated_run(tmp_path)
    metrics = json.loads((run_dir / "metrics.json").read_text())
    metrics["answers"] = {"correctness": 1.0, "grounding": 1.0, "completeness": 1.0,
                          "normalized": 0.25, "judged_queries": 20, "repeats": 1, "judge_calls": 20,
                          "judge_model": baseline().judge_model}
    (run_dir / "metrics.json").write_text(json.dumps(metrics))
    monkeypatch.setattr(matrix, "paths", replace(paths, results_dir=tmp_path / "results"))
    monkeypatch.setattr(matrix.runner, "run", lambda *a, **k: pytest.fail("should reuse the run"))
    monkeypatch.setattr(evaluate, "judge", lambda *a, **k: pytest.fail("already judged"))

    out = matrix.run_matrix(write_matrix(tmp_path, "practice-smoke20"), yes=True)
    rows = json.loads((out / "summary.json").read_text())["rows"]
    assert [r["run"] for r in rows] == [run_dir.name] and rows[0]["answer_norm"] == 0.25
    assert run_dir.name in (out / "summary.md").read_text()
    lo, hi = rows[0]["ci95"]["Recall@10"]
    assert lo <= rows[0]["Recall@10"] <= hi
    assert f"[{lo:.3f}, {hi:.3f}]" in (out / "summary.md").read_text()


def test_retriever_calls_per_query_by_toc_route():
    calls = {name: evaluate.retriever_calls_per_query(matrix.resolve_config(name))
             for name in ("toc-centroid-m4", "toc-llm-m4", "toc-llm-topics-m4", "toc-llm-flat-m4", "baseline")}
    assert calls == {"toc-centroid-m4": 0, "toc-llm-m4": 2, "toc-llm-topics-m4": 2, "toc-llm-flat-m4": 1, "baseline": 0}
