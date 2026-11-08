"""TopicRetriever: route the query to the m nearest cluster centroids, then search only their
members exactly (the idea behind an IVF index)."""
import json
from pathlib import Path

import numpy as np

from ..trace import Trace


def load_index(path) -> dict:
    """index.json from the topic_index stage, with its centroids loaded."""
    path = Path(path)
    index = json.loads(path.read_text(encoding="utf-8"))
    index["centroids"] = np.load(path.parent / index["centroids"])
    return index


class TopicRetriever:
    def __init__(self, index, vectors, ids, m=4):
        """`vectors`/`ids`: the corpus enc_a matrix and its record ids."""
        self.centroids = index["centroids"]
        row = {rid: i for i, rid in enumerate(ids)}
        self.members = [np.array([row[r] for r in ms], dtype=np.int64) for ms in index["members"]]
        self.vectors = np.asarray(vectors, dtype=np.float32)
        self.ids = ids
        self.m = m

    def search(self, query_vec, k, trace=None, m=None):
        m = m or self.m
        trace = trace or Trace("-", "-", "-")  # throwaway when nobody is tracing
        q = np.asarray(query_vec, dtype=np.float32).reshape(-1)
        q = q / np.linalg.norm(q)
        with trace.span("topic.route"):
            probed = np.argsort(-(self.centroids @ q), kind="stable")[:m]
        rows = np.sort(np.concatenate([self.members[c] for c in probed]))  # corpus order, so ties break as in a flat scan
        scores = self.vectors[rows] @ q
        top = np.argsort(-scores, kind="stable")[:k]
        trace.count("topic.clusters_probed", len(probed))
        trace.count("topic.candidates_scanned", len(rows))
        return [(self.ids[int(rows[i])], float(scores[i])) for i in top]
