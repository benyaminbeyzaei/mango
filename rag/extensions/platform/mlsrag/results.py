"""The metrics.json schema shared by every run, and helpers to read and tabulate runs."""
import json
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict, field_validator

from .course import paths

METRIC_KEYS = ["Recall@5", "Recall@10", "Recall@20", "Recall@100", "MRR@10", "NDCG@10"]
STAGES = ["asr", "encode", "retrieve", "context", "generate"]
WHISPER_USD_PER_MINUTE = 0.006  # whisper-1 list price; the free ELM chat models count as $0


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Answers(Strict):
    correctness: float
    grounding: float
    completeness: float
    normalized: float
    judged_queries: int
    judge_model: str
    repeats: int
    judge_calls: int  # new judge requests made (cached verdicts are not counted)


class Cost(Strict):
    elm_calls: int
    prompt_tokens: int
    completion_tokens: int
    audio_seconds: float
    est_usd: float


class Percentiles(Strict):
    mean: float
    p50: float
    p95: float


class Latency(Strict):  # mean ms per query; None when the stage did not run
    asr: float | None
    encode: float | None
    retrieve: float | None
    context: float | None
    generate: float | None
    end_to_end: Percentiles


class Resources(Strict):
    cpu_s: float  # summed over queries
    peak_rss_mb: float  # whole process


class RunInfo(Strict):
    run_id: str
    config_name: str
    subset: str
    n_queries: int
    started: str
    finished: str | None
    git_commit: str | None


class Metrics(Strict):
    retrieval: dict[str, float]  # exactly METRIC_KEYS
    answers: Answers | None = None
    cost: Cost
    latency_ms: Latency
    resources: Resources
    run: RunInfo

    @field_validator("retrieval")
    @classmethod
    def has_metric_keys(cls, value):
        if sorted(value) != sorted(METRIC_KEYS):
            raise ValueError(f"retrieval keys must be {METRIC_KEYS}, got {sorted(value)}")
        return value


def build_metrics(rows, run_info) -> dict:
    """metrics.json (without answers) from the per-query traces in queries.jsonl."""
    retrieval = {k: float(np.mean([r["metrics"][k] for r in rows])) for k in METRIC_KEYS}

    cost = {"elm_calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "audio_seconds": 0.0}
    whisper_seconds = 0.0
    for r in rows:
        for model, c in r["cost"].items():
            cost["elm_calls"] += c["calls"]
            cost["prompt_tokens"] += c["prompt_tokens"]
            cost["completion_tokens"] += c["completion_tokens"]
            cost["audio_seconds"] += c["audio_seconds"]
            if model.startswith("whisper"):
                whisper_seconds += c["audio_seconds"]
    cost["est_usd"] = whisper_seconds / 60 * WHISPER_USD_PER_MINUTE

    latency = {}
    for stage in STAGES:
        per_query = [sum(s["duration_ms"] for s in r["spans"] if s["name"] == stage) for r in rows]
        ran = any(s["name"] == stage for r in rows for s in r["spans"])
        latency[stage] = float(np.mean(per_query)) if ran else None
    wall = [r["resources"]["wall_ms"] for r in rows]
    latency["end_to_end"] = {
        "mean": float(np.mean(wall)),
        "p50": float(np.percentile(wall, 50)),
        "p95": float(np.percentile(wall, 95)),
    }

    resources = {
        "cpu_s": sum(r["resources"]["cpu_seconds"] for r in rows),
        "peak_rss_mb": max(r["resources"]["peak_rss_mb"] for r in rows),
    }
    metrics = {
        "retrieval": retrieval,
        "answers": None,
        "cost": cost,
        "latency_ms": latency,
        "resources": resources,
        "run": {**run_info, "n_queries": len(rows)},
    }
    Metrics.model_validate(metrics)
    return metrics


def write_metrics(run_dir, metrics):
    Metrics.model_validate(metrics)
    Path(run_dir, "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")


def load_run(run_dir) -> dict:
    """{dir, config, metrics} of one run; metrics are checked against the schema."""
    run_dir = Path(run_dir)
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    Metrics.model_validate(metrics)
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    return {"dir": run_dir, "config": config, "metrics": metrics}


def list_runs(results_dir=None) -> list[Path]:
    """Every completed run folder (one with metrics.json), oldest first."""
    root = Path(results_dir or paths.results_dir)
    return sorted(p.parent for p in root.glob("*/metrics.json"))


def summary_row(run) -> dict:
    """The numbers one table row shows for a run (from load_run)."""
    m = run["metrics"]
    n = m["run"]["n_queries"]
    a = m["answers"] or {}
    return {
        "run": m["run"]["run_id"],
        "config": m["run"]["config_name"],
        **m["retrieval"],
        "answer_norm": a.get("normalized"),
        "correctness": a.get("correctness"),
        "grounding": a.get("grounding"),
        "completeness": a.get("completeness"),
        "latency_ms": m["latency_ms"]["end_to_end"]["mean"],
        "calls/q": m["cost"]["elm_calls"] / n,
        "tokens/q": (m["cost"]["prompt_tokens"] + m["cost"]["completion_tokens"]) / n,
    }


def fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.1f}" if abs(v) >= 100 else f"{v:.3f}"
    return str(v)


def markdown_table(header, rows) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in rows]
    return "\n".join(lines) + "\n"


def compare(run_dirs) -> str:
    """Metrics as rows, one column per run."""
    rows = [summary_row(load_run(d)) for d in run_dirs]
    keys = [k for k in rows[0] if k != "run"]
    header = ["metric"] + [r["run"] for r in rows]
    return markdown_table(header, [[k] + [r[k] for r in rows] for k in keys])
