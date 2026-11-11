"""Plan 032: the reranker choice and comparison.json for the final three-way comparison.

    python -m mlsrag.final

reads the matrix summaries rerank-practice-all, final-practice-all and final-dev-all, writes a slim
per_query.jsonl next to each run (ids, timings, tokens, retrieval metrics; no prompts or answers, so the
dev-all tables can be recomputed from committed files) and matrices/final/comparison.json, which also carries the
hand-written matrices/final/explanations.json (observation, likely cause, evidence, confidence).
"""
import json
from pathlib import Path

import numpy as np

from .course import CourseData, paths
from .evaluate import read_jsonl
from .results import METRIC_KEYS, bootstrap_ci

MATRICES = paths.results_dir / "matrices"
ROWS = ["baseline", "rerank-best", "raptor-tree-gpt5-evidence"]
RETRIEVAL = ["Recall@5", "Recall@10", "Recall@100", "MRR@10", "NDCG@10"]
SCORES = ["correctness", "grounding", "completeness", "answer_norm"]
STAGES = ["asr", "encode", "dense_retrieve", "rerank", "generate"]  # dense_retrieve = retrieve minus rerank
BUILD_COST = {
    "baseline": "none",
    "rerank-best": "one-time download of the local cross-encoder (ms-marco-MiniLM-L-6-v2: 88 MB on disk, "
                   "bge-reranker-base: 1.1 GB); no API calls",
    "raptor-tree-gpt5-evidence": "gpt-5-mini summaries of the two-level tree: US$1.46, 439 calls (plan 030)",
}
# How the runs were made (the laptop was in use for other work, so latencies are slower than an idle machine's).
CPU_LIMIT = {
    "reranker_selection": "6 cores (taskset -c 0-5), nice 19, OMP/MKL threads 6",
    "final_runs": "10 cores (taskset -c 0-9), nice 19, OMP/MKL threads 10; the three rows of each split under the "
                  "same limit, one row at a time",
    "dev_all_generation": "sequential, one ELM request at a time; Qwen answer timeout 420 s",
    "clean_retrieval_latency": "practice-all R1",
}
NEAR_BEST = 0.005  # the reranker choice rule: within this much NDCG@10 of the best, take the fastest


def slim_row(row) -> dict:
    """One line of per_query.jsonl from one queries.jsonl trace."""
    ms = {}
    for s in row["spans"]:
        ms[s["name"]] = ms.get(s["name"], 0.0) + s["duration_ms"]
    ms["dense_retrieve"] = ms.get("retrieve", 0.0) - ms.get("rerank", 0.0)
    cost = row["cost"].values()
    return {
        "query_id": row["query_id"],
        "context_ids": row["artifacts"]["context_ids"],
        "ms": {k: ms.get(k) for k in STAGES} | {"end_to_end": row["resources"]["wall_ms"]},
        "llm_calls": sum(c["calls"] for c in cost),
        "prompt_tokens": sum(c["prompt_tokens"] for c in cost),
        "completion_tokens": sum(c["completion_tokens"] for c in cost),
        "rerank_pool": row["counters"].get("rerank.pool", 0),
        "rerank_truncated": row["counters"].get("rerank.truncated", 0),
        **{k: row["metrics"][k] for k in METRIC_KEYS},
    }


def load_queries(run_dir) -> list[dict]:
    """per_query.jsonl of a run (written from queries.jsonl the first time), with judge scores when judged."""
    run_dir = paths.results_dir / run_dir
    slim = run_dir / "per_query.jsonl"
    if not slim.exists():
        rows = [slim_row(r) for r in read_jsonl(run_dir / "queries.jsonl")]
        slim.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    rows = read_jsonl(slim)
    judged = run_dir / "judge" / "answer_judge_per_query.jsonl"
    if judged.exists():
        scores = {j["query_id"]: j for j in read_jsonl(judged)}
        for r in rows:
            j = scores[r["query_id"]]
            r.update({k: j[k] for k in SCORES[:3]}, answer_norm=j["normalised_answer_score"])
    return rows


def mean_ci(values) -> dict:
    return {"mean": float(np.mean(values)), "ci95": bootstrap_ci(values)}


def describe(rows) -> dict:
    """Quality with intervals, latency per stage (mean, median), calls and tokens per question."""
    keys = RETRIEVAL + (SCORES if "correctness" in rows[0] else [])
    latency = {}
    for stage in STAGES + ["end_to_end"]:
        values = [r["ms"][stage] for r in rows if r["ms"][stage] is not None]
        latency[stage] = {"mean": float(np.mean(values)), "median": float(np.median(values))} if values else None
    return {
        "n": len(rows),
        **{k: mean_ci([r[k] for r in rows]) for k in keys},
        "latency_ms": latency,
        "per_question": {k: float(np.mean([r[k] for r in rows]))
                         for k in ["llm_calls", "prompt_tokens", "completion_tokens"]},
    }


def paired(rows, base) -> dict:
    """Row minus baseline per question, on all questions and on those whose context ids differ."""
    by_id = {r["query_id"]: r for r in base}
    differ = [r for r in rows if r["context_ids"] != by_id[r["query_id"]]["context_ids"]]
    out = {"questions": len(rows), "context_differs": len(differ)}
    for name, subset in [("all", rows), ("differing", differ)]:
        out[name] = {k: mean_ci([r[k] - by_id[r["query_id"]][k] for r in subset]) if subset else None
                     for k in SCORES}
    return out


def summary_levels(rows) -> dict:
    """How many questions had at least one summary in the context, by tree level (t = topic, s = section)."""
    ids = [[c.removeprefix("summary:") for c in r["context_ids"] if c.startswith("summary:")] for r in rows]
    return {"any": sum(bool(i) for i in ids),
            "topic": sum(any(c.startswith("t") for c in i) for i in ids),
            "section": sum(any(c.startswith("s") for c in i) for i in ids)}


def choose_reranker(settings) -> str:
    """Highest NDCG@10; among settings within NEAR_BEST of it, the lowest mean latency."""
    best = max(s["NDCG@10"]["mean"] for s in settings)
    near = [s for s in settings if s["NDCG@10"]["mean"] >= best - NEAR_BEST]
    return min(near, key=lambda s: s["latency_ms"]["end_to_end"]["mean"])["config"]


def matrix_runs(name) -> dict:
    """{config name: run folder} from a matrix summary."""
    summary = json.loads((MATRICES / name / "summary.json").read_text(encoding="utf-8"))
    return {r["config"]: r["run"] for r in summary["rows"]}


def judge_calls(run) -> dict:
    """Judge calls per question, from metrics.json (the course evaluator does not report its tokens)."""
    metrics = json.loads((paths.results_dir / run / "metrics.json").read_text(encoding="utf-8"))
    a = metrics["answers"]
    return {"judge_calls_per_question": a["judge_calls"] / a["judged_queries"]} if a else {}


def top5_moves(rows, base, qrels) -> dict:
    """Relevant records that left / entered the top 5 compared with the baseline, and in how many questions."""
    by_id = {r["query_id"]: set(r["context_ids"]) for r in base}
    out = {"relevant_pushed_out": 0, "questions_losing": 0, "relevant_pulled_in": 0, "questions_gaining": 0}
    for r in rows:
        rel, b5, r5 = qrels[r["query_id"]], by_id[r["query_id"]], set(r["context_ids"])
        lost, gained = len([i for i in b5 - r5 if i in rel]), len([i for i in r5 - b5 if i in rel])
        out["relevant_pushed_out"] += lost
        out["questions_losing"] += lost > 0
        out["relevant_pulled_in"] += gained
        out["questions_gaining"] += gained > 0
    return out


def selection() -> dict:
    qrels = CourseData(paths.data_dir).load_qrels("practice")
    base = load_queries(matrix_runs("rerank-practice-all")["baseline"])
    settings = []
    for config, run in matrix_runs("rerank-practice-all").items():
        rows = load_queries(run)
        pool = sum(r["rerank_pool"] for r in rows)
        settings.append({"config": config, "run": run, **describe(rows), **top5_moves(rows, base, qrels),
                         "pairs_truncated": f"{sum(r['rerank_truncated'] for r in rows)}/{pool}"})
    reranked = [s for s in settings if s["config"] != "baseline"]
    return {"rule": f"highest NDCG@10 on practice-all; within {NEAR_BEST} of the best, the lowest mean latency",
            "chosen": choose_reranker(reranked), "settings": settings}


def comparison() -> dict:
    splits = {}
    for split in ["practice-all", "dev-all"]:
        rows = {c: load_queries(run) for c, run in matrix_runs(f"final-{split}").items()}
        splits[split] = {c: {"run": run, **describe(rows[c]), **judge_calls(run)}
                         for c, run in matrix_runs(f"final-{split}").items()}
        if split == "dev-all":
            splits[split]["paired_vs_baseline"] = {c: paired(rows[c], rows["baseline"]) for c in ROWS[1:]}
            splits[split]["summary_evidence"] = summary_levels(rows["raptor-tree-gpt5-evidence"])
    explanations = MATRICES / "final" / "explanations.json"  # written by hand from the analysis in the plan
    return {"rows": ROWS, "build_cost": BUILD_COST, "splits": splits, "reranker_selection": selection(),
            "explanations": json.loads(explanations.read_text(encoding="utf-8")) if explanations.exists() else [],
            "cpu_limit": CPU_LIMIT,
            "notes": "intervals: 95 % bootstrap over questions, 1,000 resamples, seed 0; latency in ms per question"}


if __name__ == "__main__":
    out = Path(MATRICES / "final" / "comparison.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(comparison(), indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")
