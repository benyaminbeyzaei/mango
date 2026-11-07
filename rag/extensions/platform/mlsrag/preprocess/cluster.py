"""Cluster stage: k-means over the selected records' enc_a vectors.

Plain (Euclidean) k-means on L2-normalised vectors: at this size it groups the same way as
spherical k-means would, and the centroids are re-normalised afterwards for cosine routing.
"""
import json
import re
from collections import Counter

import numpy as np
from sklearn.cluster import KMeans

from ..course import paths

STOP_WORDS = set(
    "the and for are but not you all any can had her was one our out has have his how its may "
    "new now old see two way who did get got let put say she too use with that this from they "
    "will would there their what which when where were been more also than then them these "
    "those into over such only some other about after before because between both each could "
    "should very most much many just like said does doing being here while within without "
    "your yours ours him himself herself itself whom why upon used using well even".split()
)


def top_words(texts, n=10) -> list:
    """The words that appear in the most member records (3+ letters, no stop words); ties
    alphabetical, so the result does not depend on set order (which changes between processes)."""
    counts = Counter()
    for t in texts:
        counts.update({w for w in re.findall(r"[a-z]{3,}", t.lower()) if w not in STOP_WORDS})
    return sorted(counts, key=lambda w: (-counts[w], w))[:n]


def load_vectors(record_ids, index_name="enc_a") -> np.ndarray:
    emb = paths.data_dir / "embeddings"
    all_ids = json.loads((emb / f"corpus_{index_name}_ids.json").read_text(encoding="utf-8"))
    row = {rid: i for i, rid in enumerate(all_ids)}
    vectors = np.load(emb / f"corpus_{index_name}.npy")
    return np.asarray(vectors[[row[r] for r in record_ids]], dtype=np.float32)


def run(inputs, config, ctx):
    records = [json.loads(line) for line in (inputs["select"] / "records.jsonl").open(encoding="utf-8")]
    ids = [r["record_id"] for r in records]
    x = load_vectors(ids)
    km = KMeans(n_clusters=config["k"], random_state=0, n_init=1).fit(x)
    centroids = km.cluster_centers_.astype(np.float32)
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)
    np.save(ctx.out_dir / "centroids.npy", centroids)

    labels = [int(c) for c in km.labels_]
    with (ctx.out_dir / "assignments.jsonl").open("w", encoding="utf-8") as f:
        for rid, c in zip(ids, labels, strict=True):
            f.write(json.dumps({"record_id": rid, "cluster": c}) + "\n")

    clusters = []
    for c in range(config["k"]):
        members = [r for r, lab in zip(records, labels, strict=True) if lab == c]
        clusters.append({
            "cluster": c,
            "size": len(members),
            "top_words": top_words([m["text"] for m in members]),
            "members": [m["record_id"] for m in members],
        })
    (ctx.out_dir / "clusters.json").write_text(json.dumps(clusters, indent=1) + "\n", encoding="utf-8")
    sizes = [c["size"] for c in clusters]
    return {"records": len(ids), "k": config["k"], "min_size": min(sizes), "max_size": max(sizes)}
