"""BM25 over the corpus text (bm25s), index cached under rag/.cache/bm25/."""
import hashlib
import pickle
import re
import time

import bm25s
import numpy as np
from bm25s.stopwords import STOPWORDS_EN

from ..course import paths
from ..trace import Trace

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
        if path.exists():
            self.index = pickle.loads(path.read_bytes())
        else:
            self.index = bm25s.BM25()
            self.index.index([tokenise(t) for t in texts], show_progress=False)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(pickle.dumps(self.index))
        # One-off costs, added to the first query's trace: seconds to build (or load from the cache)
        # and the size of the pickled index on disk.
        self.build_counters = {"bm25.build_s": time.perf_counter() - start,
                               "bm25.index_mb": path.stat().st_size / 1e6}

    def search(self, query_vec, k, trace=None, question=None):
        trace = trace or Trace("-", "-", "-")
        for key, value in self.build_counters.items():
            trace.count(key, value)
        self.build_counters = {}
        tokens = tokenise(question)
        if not tokens:  # empty or only stop words: bm25s cannot score an empty query
            return []
        scores = self.index.get_scores(tokens)
        top = np.argsort(-scores, kind="stable")[:k]
        return [(self.ids[i], float(scores[i])) for i in top if scores[i] > 0]
