"""Topic index stage: one index.json (plus centroids.npy) that TopicRetriever loads."""
import json
import shutil


def run(inputs, config, ctx):
    clusters = json.loads((inputs["cluster"] / "clusters.json").read_text(encoding="utf-8"))
    labels = [json.loads(line) for line in (inputs["label"] / "labels.jsonl").open(encoding="utf-8")]
    shutil.copy(inputs["cluster"] / "centroids.npy", ctx.out_dir / "centroids.npy")
    index = {
        "k": len(clusters),
        "centroids": "centroids.npy",  # relative to this file
        "labels": labels,
        "members": [c["members"] for c in clusters],
    }
    (ctx.out_dir / "index.json").write_text(json.dumps(index) + "\n", encoding="utf-8")
    return {"k": len(clusters), "records": sum(len(c["members"]) for c in clusters)}
