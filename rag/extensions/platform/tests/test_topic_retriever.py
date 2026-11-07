import numpy as np

from mlsrag.retrievers.topic import TopicRetriever
from mlsrag.trace import Trace


def toy():
    vectors = np.array([[1, 0], [0.9, 0.1], [0, 1], [0.1, 0.9]], dtype=np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    ids = ["a", "b", "c", "d"]
    index = {"centroids": np.array([[1, 0], [0, 1]], dtype=np.float32), "members": [["a", "b"], ["c", "d"]]}
    return index, vectors, ids


class G:
    trace = None


def test_probes_nearest_cluster_only_and_counts():
    index, vectors, ids = toy()
    g = G()
    g.trace = Trace("r", "q", "c")
    hits = TopicRetriever(index, vectors, ids, m=1, gateway=g).search(np.array([0.2, 1.0]), k=10)
    assert [r for r, _ in hits] == ["d", "c"]
    assert g.trace.counters == {"topic.clusters_probed": 1, "topic.candidates_scanned": 2}
    assert [s.name for s in g.trace.spans] == ["topic.route"]


def test_all_clusters_equal_flat_search():
    index, vectors, ids = toy()
    q = np.array([0.6, 0.4], dtype=np.float32)
    hits = TopicRetriever(index, vectors, ids).search(q, k=3, m=2)
    flat = np.argsort(-(vectors @ (q / np.linalg.norm(q))))[:3]
    assert [r for r, _ in hits] == [ids[i] for i in flat]


def test_ties_break_in_corpus_order_like_flat():
    vectors = np.array([[1, 0], [1, 0], [0, 1]], dtype=np.float32)
    index = {"centroids": np.array([[0, 1], [1, 0]], dtype=np.float32), "members": [["b", "c"], ["a"]]}
    ids = ["a", "b", "c"]  # a and b have equal scores; b's cluster is listed first
    hits = TopicRetriever(index, vectors, ids, m=2).search(np.array([1.0, 0.0]), k=2)
    assert [r for r, _ in hits] == ["a", "b"]
