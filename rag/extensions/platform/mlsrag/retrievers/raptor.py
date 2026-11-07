"""SummaryRetriever (RAPTOR-lite, one level): exact search over the records and the cluster
summaries together. A summary in the ranking is either replaced by its cluster's `expand_top`
members most similar to the query (default), or kept as evidence itself (`summary:<cluster>`)."""
import json
from contextlib import nullcontext
from pathlib import Path

import numpy as np

SUMMARY_PREFIX = "summary:"


def load_summaries(directory) -> tuple:
    """(vectors, rows) from the summary_index stage; rows are {cluster, summary, members}."""
    directory = Path(directory)
    rows = [json.loads(line) for line in (directory / "summaries.jsonl").open(encoding="utf-8")]
    return np.load(directory / "summaries.npy"), rows


class SummaryRetriever:
    def __init__(self, summary_vectors, summaries, vectors, ids, summary_as_evidence=False, expand_top=3, gateway=None):
        row = {rid: i for i, rid in enumerate(ids)}
        self.members = [np.array([row[r] for r in s["members"]], dtype=np.int64) for s in summaries]
        self.summary_vectors = np.asarray(summary_vectors, dtype=np.float32)
        self.vectors = np.asarray(vectors, dtype=np.float32)
        self.ids = ids
        self.as_evidence, self.expand_top = summary_as_evidence, expand_top
        self.gateway = gateway

    def search(self, query_vec, k):
        q = np.asarray(query_vec, dtype=np.float32).reshape(-1)
        q = q / np.linalg.norm(q)
        trace = getattr(self.gateway, "trace", None)
        n = len(self.ids)
        with trace.span("raptor.search") if trace is not None else nullcontext():
            record_scores = self.vectors @ q
            scores = np.concatenate([record_scores, self.summary_vectors @ q])
            order = np.argsort(-scores, kind="stable")  # records before summaries on ties
        out, seen, summary_hits = [], set(), 0
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
            if self.as_evidence:
                out.append((f"{SUMMARY_PREFIX}{c}", float(scores[i])))
                continue
            members = self.members[c]
            for j in members[np.argsort(-record_scores[members], kind="stable")[: self.expand_top]]:
                if int(j) not in seen and len(out) < k:
                    seen.add(int(j))
                    out.append((self.ids[int(j)], float(record_scores[j])))
        if trace is not None:
            trace.count("raptor.summary_hits", summary_hits)
            trace.count("raptor.candidates_scanned", len(scores))
        return out


class SummaryData:
    """The course data, plus the summaries' text under `summary:<cluster>` ids (for the context builder)."""

    def __init__(self, data, summaries):
        self.data = data
        self.summaries = {f"{SUMMARY_PREFIX}{s['cluster']}": s["summary"] for s in summaries}

    def record_text(self, rid):
        return self.summaries[rid] if rid in self.summaries else self.data.record_text(rid)
