"""Compare two transcript caches (two ASR systems): WER between them and the retrieval each one gives.

There are no reference transcripts, so WER here is disagreement with cache `a`, not accuracy.
"""
import json
import re
from pathlib import Path

import jiwer

from . import runner
from .course import paths
from .results import markdown_table
from .stages import checkout_path, load_transcripts
from .subsets import load_subset

RETRIEVAL_KEYS = ["Recall@5", "Recall@10", "MRR@10", "NDCG@10"]


def normalise(text):
    """Lowercase, drop punctuation (keep apostrophes), collapse spaces."""
    return " ".join(re.sub(r"[^\w\s']", " ", text.lower()).split())


def compare_transcripts(a, b, ids) -> dict:
    """WER of b against a as the reference, and the share of queries transcribed identically."""
    refs = [normalise(a[q]) for q in ids]
    hyps = [normalise(b[q]) for q in ids]
    return {
        "wer": jiwer.wer(refs, hyps),
        "agreement": sum(r == h for r, h in zip(refs, hyps, strict=True)) / len(ids),
    }


def asr_compare(a_path, b_path, subset, config, name=None, results_dir=None) -> tuple[Path, dict]:
    a, b = load_transcripts(checkout_path(a_path)), load_transcripts(checkout_path(b_path))
    ids = load_subset(subset)
    missing = [q for q in ids if q not in a or q not in b]
    if missing:
        raise SystemExit(f"{len(missing)} queries of {subset} missing from a cache, e.g. {missing[0]}")

    root = Path(results_dir or paths.results_dir)
    summary = {"a": str(a_path), "b": str(b_path), "subset": subset, "config": config.name,
               "n_queries": len(ids), **compare_transcripts(a, b, ids)}
    for side, path in (("a", a_path), ("b", b_path)):
        cfg = config.model_copy(update={"name": f"{config.name}-asr-{Path(path).stem}",
                                        "asr": "cached", "asr_cache": str(path)})
        run_dir, metrics = runner.run(cfg, subset, generate=False, results_dir=root)
        summary[f"retrieval_{side}"] = {k: metrics["retrieval"][k] for k in RETRIEVAL_KEYS}
        summary[f"run_{side}"] = run_dir.name

    name = name or f"{Path(a_path).stem}-vs-{Path(b_path).stem}-{subset}"
    out = root / "asr-compare" / name
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return out, summary


def format_summary(s) -> str:
    rows = [[k, s["retrieval_a"][k], s["retrieval_b"][k]] for k in RETRIEVAL_KEYS]
    return (f"WER(a -> b) = {s['wer']:.3f}, agreement = {s['agreement']:.3f} over {s['n_queries']} queries\n"
            + markdown_table(["metric", f"a: {Path(s['a']).stem}", f"b: {Path(s['b']).stem}"], rows))
