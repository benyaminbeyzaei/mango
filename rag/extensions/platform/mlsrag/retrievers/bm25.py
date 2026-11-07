"""BM25 over the corpus text (bm25s), index cached under rag/.cache/bm25/."""
import hashlib
import pickle
import re
import resource
import time

import bm25s
import numpy as np
from bm25s.stopwords import STOPWORDS_EN

from ..course import paths

TOKENIZER_VERSION = "words-lower-stopen-v1"  # change it when tokenise() changes: it enters the cache name
_STOPWORDS = set(STOPWORDS_EN)


def tokenise(text):
    return [w for w in re.findall(r"\w+", text.lower()) if w not in _STOPWORDS]


class BM25Retriever:
    uses_question = True  # the standard flow passes question=<transcript>
    uses_vector = False  # so the flow skips the encode stage and passes no vector

    def __init__(self, data, cache_dir=None):
        self.ids = list(data.records)
        texts = [data.records[rid] for rid in self.ids]
        digest = hashlib.sha256(TOKENIZER_VERSION.encode())
        for rid, text in zip(self.ids, texts, strict=True):
            digest.update(f"{rid}\t{text}\n".encode())
        path = (cache_dir or paths.cache_dir / "bm25") / f"{digest.hexdigest()[:16]}.pkl"

        start = time.perf_counter()
        rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if path.exists():
            self.index = pickle.loads(path.read_bytes())
            built = False
        else:
            self.index = bm25s.BM25()
            self.index.index([tokenise(t) for t in texts], show_progress=False)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(pickle.dumps(self.index))
            built = True
        seconds = time.perf_counter() - start
        # Reported once, on the first query's trace (see flows/standard.py).
        # rss_mb: growth of the process's peak RSS while building or loading (0 if below an earlier peak).
        rss_mb = (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - rss_before) / 1024
        self.build_counters = {
            "bm25.build_s" if built else "bm25.load_s": seconds,
            "bm25.build_rss_mb" if built else "bm25.load_rss_mb": rss_mb,
            "bm25.index_mb": path.stat().st_size / 1e6,  # size of the pickled index on disk
        }

    def search(self, query_vec, k, question):
        tokens = tokenise(question)
        if not tokens:  # empty or only stop words: bm25s cannot score an empty query
            return []
        scores = self.index.get_scores(tokens)
        top = np.argsort(-scores, kind="stable")[:k]
        return [(self.ids[i], float(scores[i])) for i in top if scores[i] > 0]
