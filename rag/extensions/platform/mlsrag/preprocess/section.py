"""Section stage: k-means on the topic centroids groups the topics into sections, and one LLM
call per section names it from its topics' labels (the upper level of the table of contents)."""
import json

import numpy as np
from sklearn.cluster import KMeans

from .label import parse_label
from .ledger import run_items

PROMPT_VERSION = "v1"
PROMPT_V1 = (
    "Below are the topics of one section of a document collection.\n\n{topics}\n\n"
    "Name the section in at most six words and describe it in one sentence. "
    'Reply with JSON only: {{"topic": "...", "summary": "..."}}'
)


def build_prompt(topics, attempt=0) -> str:
    prompt = PROMPT_V1.format(topics="\n".join(f"- {t['topic']}: {t['summary']}" for t in topics))
    if attempt:
        prompt += f"\n\n(Retry {attempt}: the previous reply was not usable. Reply with the JSON object only.)"
    return prompt


def run(inputs, config, ctx):
    centroids = np.load(inputs["cluster"] / "centroids.npy")
    labels = [json.loads(line) for line in (inputs["label"] / "labels.jsonl").open(encoding="utf-8")]
    km = KMeans(n_clusters=config["n"], random_state=0, n_init=1).fit(centroids)
    section_centroids = km.cluster_centers_.astype(np.float32)
    section_centroids /= np.linalg.norm(section_centroids, axis=1, keepdims=True)
    np.save(ctx.out_dir / "section_centroids.npy", section_centroids)

    topics = {s: [c for c, lab in enumerate(km.labels_) if lab == s] for s in range(config["n"])}
    names = run_items({s: [labels[c] for c in cs] for s, cs in topics.items()}, ctx, config["model"],
                      build_prompt, parse_label)
    sections = [{"section": s, "title": names[s]["topic"], "summary": names[s]["summary"], "topics": topics[s]}
                for s in range(config["n"])]
    (ctx.out_dir / "sections.json").write_text(json.dumps(sections, indent=1, ensure_ascii=False) + "\n",
                                               encoding="utf-8")
    sizes = [len(t) for t in topics.values()]
    return {"sections": config["n"], "min_topics": min(sizes), "max_topics": max(sizes)}
