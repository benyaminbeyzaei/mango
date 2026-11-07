"""Summarise stage (RAPTOR-lite): one LLM summary per cluster, written from the members nearest
its centroid, through the ledger."""
import json

import numpy as np

from .cluster import load_vectors
from .ledger import run_items

PROMPT_VERSION = "v1"
PROMPT_V1 = (
    "Below are records from one cluster of a document collection.\n\n{records}\n\n"
    "Summarise in at most {words} words what these records are about: the shared subject, the "
    "main names, places, dates and facts. Reply with the summary only."
)


def build_prompt(texts, attempt=0, words=120) -> str:
    prompt = PROMPT_V1.format(records="\n".join(f"- {t}" for t in texts), words=words)
    if attempt:
        prompt += f"\n\n(Retry {attempt}: the previous reply was not usable. Reply with the summary only.)"
    return prompt


def parse_summary(reply, words=120) -> str:
    """The reply, cut to `words` words; an empty reply is an error."""
    out = reply.split()
    if not out:
        raise ValueError("empty summary")
    return " ".join(out[:words])


def run(inputs, config, ctx):
    texts = {}
    for line in (inputs["select"] / "records.jsonl").open(encoding="utf-8"):
        r = json.loads(line)
        texts[r["record_id"]] = r["text"]
    clusters = json.loads((inputs["cluster"] / "clusters.json").read_text(encoding="utf-8"))
    centroids = np.load(inputs["cluster"] / "centroids.npy")
    words, chars = config.get("words", 120), config.get("chars", 400)
    items = {}
    for c in clusters:
        sims = load_vectors(c["members"]) @ centroids[c["cluster"]]
        nearest = [c["members"][i] for i in np.argsort(-sims, kind="stable")[: config.get("members", 8)]]
        items[c["cluster"]] = [texts[r][:chars] for r in nearest]
    summaries = run_items(items, ctx, config["model"], lambda t, a: build_prompt(t, a, words),
                          lambda reply: parse_summary(reply, words), max_tokens=config.get("max_tokens", 300))
    with (ctx.out_dir / "summaries.jsonl").open("w", encoding="utf-8") as f:
        for c in clusters:
            f.write(json.dumps({"cluster": c["cluster"], "summary": summaries[c["cluster"]]}, ensure_ascii=False) + "\n")
    lengths = [len(s.split()) for s in summaries.values()]
    return {"clusters": len(clusters), "mean_words": sum(lengths) / len(lengths), "max_words": max(lengths)}
