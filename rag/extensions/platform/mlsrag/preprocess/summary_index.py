"""Summary index stage: the cluster summaries embedded like passages (summaries.npy, same row
order as summaries.jsonl, which also carries each cluster's members), for SummaryRetriever."""
import json

import numpy as np


def run(inputs, config, ctx):
    clusters = json.loads((inputs["cluster"] / "clusters.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (inputs["summarise"] / "summaries.jsonl").open(encoding="utf-8")]
    vectors = ctx.gateway.embed(config.get("encoder", "gte-small"), [r["summary"] for r in rows], role="passage")
    np.save(ctx.out_dir / "summaries.npy", vectors.astype(np.float32))
    with (ctx.out_dir / "summaries.jsonl").open("w", encoding="utf-8") as f:
        for r, c in zip(rows, clusters, strict=True):
            f.write(json.dumps({**r, "members": c["members"]}, ensure_ascii=False) + "\n")
    return {"summaries": len(rows), "dim": int(vectors.shape[1])}
