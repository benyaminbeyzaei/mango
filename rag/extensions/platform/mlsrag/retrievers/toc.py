"""TocRetriever: a two-level table of contents (sections -> topics). The query picks sections,
then topics inside them, then the topics' members are searched exactly.

route "centroid": nearest section centroids, then nearest topic centroids among their topics.
route "llm": one chat call picks sections from their titles, one picks topics from their labels.
The LLM needs the question text: `uses_question = True` makes the flow pass it as `question=`.
"""
import json
import re
from pathlib import Path

import numpy as np

from ..trace import Trace

SECTIONS_PROMPT = (
    "Question: {question}\n\nA document collection is organised into these sections:\n{options}\n\n"
    "Which sections (at most {n}) most likely contain the answer? "
    'Reply with JSON only: {{"ids": [<section numbers>]}}'
)
TOPICS_PROMPT = (
    "Question: {question}\n\nThese are the topics of the most likely sections:\n{options}\n\n"
    "Which topics (at most {n}) most likely contain the answer? "
    'Reply with JSON only: {{"ids": [<topic numbers>]}}'
)


def load_toc(path) -> dict:
    """toc.json from the toc_index stage, with both centroid matrices loaded."""
    path = Path(path)
    toc = json.loads(path.read_text(encoding="utf-8"))
    for key in ("section_centroids", "topic_centroids"):
        toc[key] = np.load(path.parent / toc[key])
    return toc


def parse_ids(reply, allowed, n) -> list:
    """The first n distinct allowed integers in the reply's first [...] JSON list (in reply order).
    Anything else in the list (negative, decimal, text) is dropped, never reinterpreted."""
    match = re.search(r"\[.*?\]", reply, re.DOTALL)
    try:
        values = json.loads(match.group(0)) if match else []
    except json.JSONDecodeError:
        return []
    out = []
    for x in values:
        if type(x) is int and x in allowed and x not in out:
            out.append(x)
    return out[:n]


class TocRetriever:
    uses_question = True

    def __init__(self, toc, vectors, ids, sections=2, m=4, route="centroid", gateway=None, model=None):
        if route not in ("centroid", "llm"):
            raise ValueError(f"route must be centroid or llm, not {route!r}")
        self.toc = toc
        row = {rid: i for i, rid in enumerate(ids)}
        self.members = [np.array([row[r] for r in t["members"]], dtype=np.int64) for t in toc["topics"]]
        self.vectors = np.asarray(vectors, dtype=np.float32)
        self.ids = ids
        self.n_sections, self.m, self.route = sections, m, route
        self.gateway, self.model = gateway, model

    def _nearest(self, centroids, q, allowed, n):
        allowed = np.asarray(allowed)
        return [int(i) for i in allowed[np.argsort(-(centroids[allowed] @ q), kind="stable")[:n]]]

    def _ask(self, prompt, allowed, n, trace):
        reply = self.gateway.chat(self.model, [{"role": "user", "content": prompt}], temperature=0.0, max_tokens=100)
        trace.count("toc.llm_calls")
        return parse_ids(reply, set(allowed), n)

    def search(self, query_vec, k, trace=None, question=None):
        trace = trace or Trace("-", "-", "-")
        q = np.asarray(query_vec, dtype=np.float32).reshape(-1)
        q = q / np.linalg.norm(q)
        all_sections = list(range(len(self.toc["sections"])))
        with trace.span("toc.route", route=self.route):
            if self.route == "centroid":
                sections = self._nearest(self.toc["section_centroids"], q, all_sections, self.n_sections)
            else:
                if question is None:
                    raise ValueError("route llm needs the question text")
                options = "\n".join(f"{s['section']}. {s['title']}: {s['summary']}" for s in self.toc["sections"])
                sections = self._ask(SECTIONS_PROMPT.format(question=question, options=options, n=self.n_sections),
                                     all_sections, self.n_sections, trace)
                if not sections:  # unusable reply: fall back to the centroids
                    trace.count("toc.llm_fallbacks")
                    sections = self._nearest(self.toc["section_centroids"], q, all_sections, self.n_sections)
            topics_in = [t for s in sections for t in self.toc["sections"][s]["topics"]]
            if self.route == "centroid":
                topics = self._nearest(self.toc["topic_centroids"], q, topics_in, self.m)
            else:
                options = "\n".join(f"{t}. {self.toc['topics'][t]['topic']}: {self.toc['topics'][t]['summary']}"
                                    for t in topics_in)
                topics = self._ask(TOPICS_PROMPT.format(question=question, options=options, n=self.m),
                                   topics_in, self.m, trace)
                if not topics:
                    trace.count("toc.llm_fallbacks")
                    topics = self._nearest(self.toc["topic_centroids"], q, topics_in, self.m)
        rows = np.sort(np.concatenate([self.members[t] for t in topics]))  # corpus order: ties break as in a flat scan
        scores = self.vectors[rows] @ q
        top = np.argsort(-scores, kind="stable")[:k]
        trace.count("toc.sections_probed", len(sections))
        trace.count("toc.topics_probed", len(topics))
        trace.count("toc.candidates_scanned", len(rows))
        return [(self.ids[int(rows[i])], float(scores[i])) for i in top]
