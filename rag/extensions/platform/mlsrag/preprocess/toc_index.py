"""TOC index stage: toc.json (sections -> topics -> record ids) plus both centroid files, for TocRetriever."""
import json
import shutil


def run(inputs, config, ctx):
    clusters = json.loads((inputs["cluster"] / "clusters.json").read_text(encoding="utf-8"))
    labels = [json.loads(line) for line in (inputs["label"] / "labels.jsonl").open(encoding="utf-8")]
    sections = json.loads((inputs["section"] / "sections.json").read_text(encoding="utf-8"))
    shutil.copy(inputs["cluster"] / "centroids.npy", ctx.out_dir / "topic_centroids.npy")
    shutil.copy(inputs["section"] / "section_centroids.npy", ctx.out_dir / "section_centroids.npy")
    toc = {
        "section_centroids": "section_centroids.npy",  # relative to this file
        "topic_centroids": "topic_centroids.npy",
        "sections": sections,  # [{section, title, summary, topics: [cluster ids]}]
        "topics": [{"cluster": c["cluster"], "topic": lab["topic"], "summary": lab["summary"],
                    "members": c["members"]} for c, lab in zip(clusters, labels, strict=True)],
    }
    (ctx.out_dir / "toc.json").write_text(json.dumps(toc, ensure_ascii=False) + "\n", encoding="utf-8")
    return {"sections": len(sections), "topics": len(clusters)}
