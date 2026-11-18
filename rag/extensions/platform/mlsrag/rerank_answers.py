"""Do the better rerankers give better answers on test? Each reranker's end-to-end run minus the
final-test baseline per question, split by whether the top 5 gained or lost relevant records, plus two checks that
explain the result: answer-score noise (baseline vs a row with the same contexts) and how much the answer score
depends on retrieving a relevant record at all. Run: python -m mlsrag.rerank_answers"""
import json

import numpy as np

from .final import load_queries, mean_ci, paths

BASELINE = "20261111-0115-baseline-test"
SAME_CONTEXTS = "20261111-0719-raptor-tree-gpt5-evidence-test"  # same top 5 as the baseline for 115 of 125
RUNS = {"rerank-bge-m3-p20": "20261116-2358-rerank-bge-m3-p20-test",
        "rerank-mxbai-v2-p20": "20261117-0316-rerank-mxbai-v2-p20-test"}
SCORES = ["answer_norm", "correctness", "grounding", "completeness"]


def by_id(run):
    return {r["query_id"]: r for r in load_queries(run)}


def minus(rows, base, ids, key="answer_norm"):
    return mean_ci([rows[q][key] - base[q][key] for q in ids]) if len(ids) > 1 else None


def analyse() -> dict:
    base = by_id(BASELINE)
    ids = list(base)
    out = {"baseline": BASELINE, "rows": {}}
    for config, run in RUNS.items():
        rows = by_id(run)
        groups = {"recall5_up": [q for q in ids if rows[q]["Recall@5"] > base[q]["Recall@5"]],
                  "recall5_down": [q for q in ids if rows[q]["Recall@5"] < base[q]["Recall@5"]],
                  "recall5_same": [q for q in ids if rows[q]["Recall@5"] == base[q]["Recall@5"]]}
        out["rows"][config] = {
            "run": run,
            "minus_baseline": {k: minus(rows, base, ids, k) for k in SCORES},
            "by_recall5_change": {g: {"n": len(qs), "answer_norm": minus(rows, base, qs)} for g, qs in groups.items()},
        }
    same = by_id(SAME_CONTEXTS)
    fixed = [q for q in ids if same[q]["context_ids"] == base[q]["context_ids"]]
    diffs = [same[q]["answer_norm"] - base[q]["answer_norm"] for q in fixed]
    out["noise_same_contexts"] = {"run": SAME_CONTEXTS, "questions": len(fixed),
                                  "score_changed": int(sum(d != 0 for d in diffs)), "sd": float(np.std(diffs))}
    out["answer_norm_by_recall5"] = {}
    for config, run in {"baseline": BASELINE, **RUNS}.items():
        rows = list(by_id(run).values())
        out["answer_norm_by_recall5"][config] = {
            name: {"n": len(rs), "answer_norm": float(np.mean([r["answer_norm"] for r in rs]))}
            for name, rs in [("none_relevant", [r for r in rows if r["Recall@5"] == 0]),
                             ("some_relevant", [r for r in rows if r["Recall@5"] > 0])]}
    return out


if __name__ == "__main__":
    path = paths.results_dir / "matrices" / "final-rerank-test" / "analysis.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(analyse(), indent=2) + "\n", encoding="utf-8")
    print(f"wrote {path}")
