"""Run a config over a subset and write config.json, queries.jsonl and metrics.json."""
import json
import subprocess
import time
import wave
from datetime import datetime
from pathlib import Path

from . import evaluate
from .course import CourseData, mrr_at_k, ndcg_at_k, paths, recall_at_k
from .flows import get_flow
from .gateway import ModelGateway
from .results import METRIC_KEYS, build_metrics, write_metrics  # noqa: F401 (METRIC_KEYS re-exported)
from .stages import WhisperASR, CourseASR, build_stages, checkout_path, load_record_subset, load_transcripts
from .subsets import SUBSETS, check_subset, load_subset
from .trace import Trace


def retrieval_metrics(ranked, rel, record_ids=None) -> dict:
    """The six metrics exactly as rag/scripts/evaluate_rag.py computes them. With `record_ids`
    (params.record_subset) only the relevant records inside that subset count."""
    if record_ids is not None:
        rel = {r: g for r, g in rel.items() if r in record_ids}
    return {
        "Recall@5": recall_at_k(ranked, rel, 5),
        "Recall@10": recall_at_k(ranked, rel, 10),
        "Recall@20": recall_at_k(ranked, rel, 20),
        "Recall@100": recall_at_k(ranked, rel, 100),
        "MRR@10": mrr_at_k(ranked, rel, 10),
        "NDCG@10": ndcg_at_k(ranked, rel, 10),
    }


def git_commit():
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=paths.rag_dir, capture_output=True, text=True)
    except OSError:
        return None
    return out.stdout.strip() or None


def now():
    return datetime.now().isoformat(timespec="seconds")


def run(config, subset, generate=True, results_dir=None, allow_full=False, fresh=False):
    """fresh=True skips cache reads for the system's chat and embed calls, so latency and cost are real."""
    try:
        check_subset(subset, allow_full)
    except ValueError as err:  # say what the refused run would have cost
        e = evaluate.estimate(config, subset, generate=generate)
        raise ValueError(f"{err}\n{evaluate.format_estimate(e)}") from None
    data = CourseData(paths.data_dir)
    manifest = {row["query_id"]: row for row in data.load_audio_manifest()}
    qrels = {**data.load_qrels("practice"), **data.load_qrels("dev")}
    record_ids = None
    if config.params.get("record_subset"):
        if SUBSETS[subset][0] != "dev":  # the subset is built around the dev qrels
            raise ValueError("params.record_subset is meant for dev subsets only")
        record_ids = load_record_subset(config.params["record_subset"])
    gateway = ModelGateway(fresh=fresh)
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
    started = now()
    with (out / "queries.jsonl").open("w", encoding="utf-8") as f:
        for i, qid in enumerate(ids, start=1):
            trace = Trace(run_id, qid, config.name)
            gateway.trace = trace
            flow(qid, paths.data_dir / manifest[qid]["audio_file"], stages, config, trace)
            trace.finish()
            row = trace.to_dict()
            row["metrics"] = retrieval_metrics(trace.artifacts["ranked_ids"], qrels[qid], record_ids)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows.append(row)
            print(f"[{i:>3}/{len(ids)}] {qid} R@10={row['metrics']['Recall@10']:.3f}")

    info = {"run_id": run_id, "config_name": config.name, "subset": subset,
            "started": started, "finished": now(), "git_commit": git_commit(), "fresh": fresh}
    metrics = build_metrics(rows, info)
    write_metrics(out, metrics)
    return out, metrics


def wav_seconds(path):
    with wave.open(str(path)) as w:
        return w.getnframes() / w.getframerate()


def transcribe(asr_name, subset, out_path, allow_full=False, fresh=False):
    """Fill a transcript cache for a subset; ids already in the file are skipped. Each new line
    carries `seconds` (wall time of the call) and `audio_s` (WAV duration); the model load time
    is not per query, so it goes once into <out>.meta.json. fresh=True makes whisper skip its
    gateway cache read."""
    check_subset(subset, allow_full)
    out_path = checkout_path(out_path)
    data = CourseData(paths.data_dir)
    manifest = {row["query_id"]: row for row in data.load_audio_manifest()}
    trace = Trace("transcribe", "-", asr_name)

    done = load_transcripts(out_path)
    todo = [q for q in load_subset(subset) if q not in done]
    if not todo:
        return {"skipped": len(done), "transcribed": 0, "cost": trace.cost}
    start = time.perf_counter()
    asr = WhisperASR(ModelGateway(trace), fresh=fresh) if asr_name == "whisper1" else CourseASR()
    if asr_name == "course":
        asr.adapter.prepare()  # loads the model, so the first query's time is inference only
    meta = {"asr": asr_name, "load_seconds": time.perf_counter() - start, "fresh": fresh}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Path(f"{out_path}.meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    for i, qid in enumerate(todo, start=1):
        audio = paths.data_dir / manifest[qid]["audio_file"]
        start = time.perf_counter()
        text = asr.transcribe(audio)
        row = {"query_id": qid, "transcript": text,
               "seconds": time.perf_counter() - start, "audio_s": wav_seconds(audio)}
        with out_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[{i:>3}/{len(todo)}] {qid} {row['seconds']:.2f}s {text[:60]!r}")
    return {"skipped": len(done), "transcribed": len(todo), "cost": trace.cost}
