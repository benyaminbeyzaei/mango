"""Run a config over a subset and write config.json, queries.jsonl and metrics.json."""
import json
import time

import numpy as np

from .course import CourseData, mrr_at_k, ndcg_at_k, paths, recall_at_k
from .flows import get_flow
from .gateway import ModelGateway
from .stages import WhisperASR, CourseASR, build_stages, checkout_path, load_transcripts
from .subsets import load_subset
from .trace import Trace

METRIC_KEYS = ["Recall@5", "Recall@10", "Recall@20", "Recall@100", "MRR@10", "NDCG@10"]


def retrieval_metrics(ranked, rel) -> dict:
    """The six metrics exactly as rag/scripts/evaluate_rag.py computes them."""
    return {
        "Recall@5": recall_at_k(ranked, rel, 5),
        "Recall@10": recall_at_k(ranked, rel, 10),
        "Recall@20": recall_at_k(ranked, rel, 20),
        "Recall@100": recall_at_k(ranked, rel, 100),
        "MRR@10": mrr_at_k(ranked, rel, 10),
        "NDCG@10": ndcg_at_k(ranked, rel, 10),
    }


def summarise(rows, wall_seconds) -> dict:
    out = {"queries": len(rows)}
    for key in METRIC_KEYS:
        out[key] = float(np.mean([r["metrics"][key] for r in rows]))
    out["wall_seconds"] = wall_seconds

    stage_ms = {}  # span name -> total ms per query, averaged over queries
    for r in rows:
        for s in r["spans"]:
            stage_ms[s["name"]] = stage_ms.get(s["name"], 0.0) + s["duration_ms"]
    out["mean_stage_ms"] = {name: ms / len(rows) for name, ms in stage_ms.items()}

    totals = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "audio_seconds": 0.0}
    for r in rows:
        for c in r["cost"].values():
            for k in totals:
                totals[k] += c[k]
    out["elm_calls"] = totals["calls"]
    out["prompt_tokens"] = totals["prompt_tokens"]
    out["completion_tokens"] = totals["completion_tokens"]
    out["audio_seconds"] = totals["audio_seconds"]
    out["cache_hits"] = sum(
        v for r in rows for k, v in r["counters"].items() if k.endswith(".cache_hits")
    )
    return out


def run(config, subset, generate=True, results_dir=None):
    data = CourseData(paths.data_dir)
    manifest = {row["query_id"]: row for row in data.load_audio_manifest()}
    qrels = {**data.load_qrels("practice"), **data.load_qrels("dev")}
    gateway = ModelGateway()
    stages = build_stages(config, gateway, data)
    if not generate:
        stages.generator = None
    flow = get_flow(config.flow)

    run_id = f"{time.strftime('%Y%m%d-%H%M')}-{config.name}-{subset}"
    root = results_dir or paths.results_dir
    n = 2
    while (root / run_id).exists():  # same config and subset within the same minute
        run_id = f"{run_id.rsplit('~', 1)[0]}~{n}"
        n += 1
    out = root / run_id
    out.mkdir(parents=True)
    (out / "config.json").write_text(config.model_dump_json(indent=2) + "\n", encoding="utf-8")

    ids = load_subset(subset)
    rows = []
    start = time.perf_counter()
    with (out / "queries.jsonl").open("w", encoding="utf-8") as f:
        for i, qid in enumerate(ids, start=1):
            trace = Trace(run_id, qid, config.name)
            gateway.trace = trace
            flow(qid, paths.data_dir / manifest[qid]["audio_file"], stages, config, trace)
            trace.finish()
            row = trace.to_dict()
            row["metrics"] = retrieval_metrics(trace.artifacts["ranked_ids"], qrels[qid])
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows.append(row)
            print(f"[{i:>3}/{len(ids)}] {qid} R@10={row['metrics']['Recall@10']:.3f}")

    metrics = summarise(rows, time.perf_counter() - start)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return out, metrics


def transcribe(asr_name, subset, out_path):
    """Fill a transcript cache for a subset; ids already in the file are skipped."""
    out_path = checkout_path(out_path)
    data = CourseData(paths.data_dir)
    manifest = {row["query_id"]: row for row in data.load_audio_manifest()}
    trace = Trace("transcribe", "-", asr_name)
    asr = WhisperASR(ModelGateway(trace)) if asr_name == "whisper1" else CourseASR()

    done = load_transcripts(out_path)
    todo = [q for q in load_subset(subset) if q not in done]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for i, qid in enumerate(todo, start=1):
        text = asr.transcribe(paths.data_dir / manifest[qid]["audio_file"])
        with out_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"query_id": qid, "transcript": text}, ensure_ascii=False) + "\n")
        print(f"[{i:>3}/{len(todo)}] {qid} {text[:60]!r}")
    return {"skipped": len(done), "transcribed": len(todo), "cost": trace.cost}
