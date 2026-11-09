"""Answer judging through the course evaluator, and cost estimates before a run."""
import json
import os
import subprocess
import sys
from pathlib import Path

from .course import CourseData, paths
from .results import WHISPER_USD_PER_MINUTE, write_metrics
from .stages import checkout_path, load_transcripts
from .subsets import SUBSETS, load_subset

# Upper bound on extra ELM calls per query that a flow makes besides generation.
FLOW_CALLS_PER_QUERY = {"standard": 0, "grep-static": 1, "grep-agentic": 4, "grep-feedback": 4}


def retriever_calls_per_query(config) -> int:
    """ELM calls the retriever makes per query: toc with route: llm picks sections, then topics."""
    return 2 if config.retriever == "toc" and config.params.get("route", "centroid") == "llm" else 0


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]


def write_candidates(run_dir) -> Path:
    """queries.jsonl traces -> answers.jsonl in the format rag/scripts/generate_answers.py writes."""
    run_dir = Path(run_dir)
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    subset = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))["run"]["subset"]
    split = SUBSETS[subset][0]
    manifest = {row["query_id"]: row for row in CourseData(paths.data_dir).load_audio_manifest()}
    out = run_dir / "answers.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for row in read_jsonl(run_dir / "queries.jsonl"):
            qid, a = row["query_id"], row["artifacts"]
            if a.get("answer") is None:
                raise ValueError(f"{qid} has no answer; this run was made with --no-generate")
            ms = {s["name"]: s["duration_ms"] for s in row["spans"]}
            f.write(json.dumps({
                "query_id": qid,
                "split": split,
                "audio_file": manifest[qid]["audio_file"],
                "transcript": a["transcript"],
                "answer": a["answer"],
                # the ids the answer's context was built from; older traces lack them
                "retrieved_record_ids": a.get("context_ids", a["ranked_ids"][: config["k"]]),
                "timing": {
                    "asr_ms": ms.get("asr"),
                    "embedding_ms": ms.get("encode"),
                    "retrieval_ms": ms.get("retrieve"),
                    "generation_ms": ms.get("generate"),
                    "end_to_end_ms": row["resources"]["wall_ms"],
                },
            }, ensure_ascii=False) + "\n")
    return out


def count_lines(path):
    return len(read_jsonl(path)) if path.exists() else 0


def judge(run_dir, judge_model=None, repeats=1, dry_run=False):
    """Run rag/scripts/evaluate_answers.py on the run's answers; store the means in metrics.json."""
    run_dir = Path(run_dir)
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    model = judge_model or config["judge_model"]
    candidates = write_candidates(run_dir)
    out_dir = run_dir / "judge"
    cache = out_dir / "judge_cache.jsonl"
    before = count_lines(cache)

    cmd = [sys.executable, str(paths.rag_dir / "scripts" / "evaluate_answers.py"),
           "--split", SUBSETS[metrics["run"]["subset"]][0], "--candidates", str(candidates),
           "--allow-subset", "--out-dir", str(out_dir), "--repeats", str(repeats)]
    if dry_run:
        cmd.append("--dry-run")
    subprocess.run(cmd, env={**os.environ, "ELM_JUDGE_MODEL": model}, check=True)
    if dry_run:
        return None

    s = json.loads((out_dir / "answer_judge_summary.json").read_text(encoding="utf-8"))
    metrics["answers"] = {
        "correctness": s["mean_correctness"],
        "grounding": s["mean_grounding"],
        "completeness": s["mean_completeness"],
        "normalized": s["mean_normalised_answer_score"],
        "judged_queries": s["queries"],
        "judge_model": s["judge_model"],
        "repeats": s["repeats"],
        "judge_calls": count_lines(cache) - before,
    }
    write_metrics(run_dir, metrics)
    return metrics["answers"]


def estimate(config, subset, judge=False, repeats=1, generate=True) -> dict:
    """Upper bound on the ELM calls a run makes (cached calls are free) and the whisper cost."""
    ids = load_subset(subset)
    durations = {r["query_id"]: r["duration_seconds"]
                 for r in CourseData(paths.data_dir).load_audio_manifest()}
    if config.asr == "whisper1":
        to_transcribe = ids
    elif config.asr == "cached":  # missing ids need `mlsrag transcribe` (whisper) first
        done = load_transcripts(checkout_path(config.asr_cache)) if config.asr_cache else {}
        to_transcribe = [q for q in ids if q not in done]
    else:  # course ASR runs locally on a GPU
        to_transcribe = []
    audio_seconds = sum(durations[q] for q in to_transcribe)
    calls = {
        "generation": len(ids) if generate else 0,
        "judge": len(ids) * repeats if judge else 0,
        "flow": FLOW_CALLS_PER_QUERY[config.flow] * len(ids),
        "retriever": retriever_calls_per_query(config) * len(ids),
        "whisper": len(to_transcribe),
    }
    return {
        "config": config.name,
        "subset": subset,
        "queries": len(ids),
        "calls": calls,
        "elm_calls": sum(calls.values()),
        "audio_seconds": audio_seconds,
        "est_usd": audio_seconds / 60 * WHISPER_USD_PER_MINUTE,
    }


def format_estimate(e, fresh=False) -> str:
    c = e["calls"]
    mode = "fresh: every listed call is made (judge stays cached)" if fresh else "upper bound, cached calls are free"
    return (
        f"{e['config']} on {e['subset']} ({e['queries']} queries), {mode}:\n"
        f"  generation calls: {c['generation']}\n"
        f"  judge calls:      {c['judge']}\n"
        f"  flow calls:       {c['flow']}\n"
        f"  retriever calls:  {c['retriever']}\n"
        f"  whisper calls:    {c['whisper']} ({e['audio_seconds']:.0f} audio s, ~${e['est_usd']:.3f})\n"
        f"  total ELM calls:  {e['elm_calls']} (free models count as $0)"
    )
