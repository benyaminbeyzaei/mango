"""SummaryRetriever (RAPTOR-lite, one level): exact search over the records and the cluster
summaries together. A summary in the ranking is either replaced by its cluster's `expand_top`
members most similar to the query (default), or kept as evidence itself (`summary:<cluster>`).
RAPTOR-tree (plan 028) uses the same rule over two levels (ids t<cluster> and s<section>);
hits are also counted per level."""
import json
from collections import Counter
from pathlib import Path

import numpy as np

from ..trace import Trace

SUMMARY_PREFIX = "summary:"


def load_summaries(directory) -> tuple:
    """(vectors, rows) from the summary_index or tree_index stage; rows are {id, level, summary, members}
    (a raptor-lite row gets id = its cluster and level 1)."""
    directory = Path(directory)
    rows = [json.loads(line) for line in (directory / "summaries.jsonl").open(encoding="utf-8")]
    rows = [{"id": r.get("cluster"), "level": 1, **r} for r in rows]
    return np.load(directory / "summaries.npy"), rows


class SummaryRetriever:
    def __init__(self, summary_vectors, summaries, vectors, ids, summary_as_evidence=False, expand_top=3):
        row = {rid: i for i, rid in enumerate(ids)}
        self.members = [np.array([row[r] for r in s["members"]], dtype=np.int64) for s in summaries]
        self.keys = [s["id"] for s in summaries]
        self.levels = [s["level"] for s in summaries]
        self.summary_vectors = np.asarray(summary_vectors, dtype=np.float32)
        self.vectors = np.asarray(vectors, dtype=np.float32)
        self.ids = ids
        self.as_evidence, self.expand_top = summary_as_evidence, expand_top

    def search(self, query_vec, k, trace=None):
        trace = trace or Trace("-", "-", "-")
        q = np.asarray(query_vec, dtype=np.float32).reshape(-1)
        q = q / np.linalg.norm(q)
        n = len(self.ids)
        with trace.span("raptor.search"):
            record_scores = self.vectors @ q
            scores = np.concatenate([record_scores, self.summary_vectors @ q])
            order = np.argsort(-scores, kind="stable")  # records before summaries on ties
        out, seen, summary_hits, level_hits = [], set(), 0, Counter()
        for i in order:
            if len(out) >= k:
                break
            i = int(i)
            if i < n:
                if i not in seen:
                    seen.add(i)
                    out.append((self.ids[i], float(scores[i])))
                continue
            summary_hits += 1
            c = i - n
            level_hits[self.levels[c]] += 1
            if self.as_evidence:
                out.append((f"{SUMMARY_PREFIX}{self.keys[c]}", float(scores[i])))
                continue
            members = self.members[c]
            for j in members[np.argsort(-record_scores[members], kind="stable")[: self.expand_top]]:
                if int(j) not in seen and len(out) < k:
                    seen.add(int(j))
                    out.append((self.ids[int(j)], float(record_scores[j])))
        trace.count("raptor.summary_hits", summary_hits)
        for level in sorted(set(self.levels)):
            trace.count(f"raptor.level{level}_hits", level_hits[level])
        trace.count("raptor.candidates_scanned", len(scores))
        return out


class SummaryData:
    """The course data, plus the summaries' text under `summary:<id>` ids (for the context builder)."""

    def __init__(self, data, summaries):
        self.data = data
        self.summaries = {f"{SUMMARY_PREFIX}{s['id']}": s["summary"] for s in summaries}

    def record_text(self, rid):
        return self.summaries[rid] if rid in self.summaries else self.data.record_text(rid)
