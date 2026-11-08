"""Graph retrievers. Both take the question text and the query vector; candidates come from the
graph, and dense similarity (enc_a) only orders candidates that the graph ranks equally.

LazyGraphRetriever: phrases of the question that are graph nodes -> their records, each record
scored by the sum of 1/degree of its matched nodes (a common phrase counts little).
KGRetriever: entities of the question (spaCy phrase matching or one LLM call) -> their 1-hop
neighbours -> records ranked by direct matches, then 1-hop matches.
"""
import json
import re
from pathlib import Path

import numpy as np

from ..preprocess.phrases import load_nlp, normalise, question_ngrams
from ..trace import Trace

QUESTION_ENTITIES_PROMPT = (
    "List the entities (people, organisations, products, financial instruments, accounts, taxes, "
    "laws, places, finance concepts) that this question is about.\n"
    'Reply with JSON only: {{"entities": ["...", "..."]}}\n\nQuestion: {question}'
)


class GraphRanker:
    """Shared by both retrievers: the corpus vectors and the final ranking."""

    uses_question = True  # the flow passes question=<transcript>

    def __init__(self, vectors, ids):
        self.vectors = np.asarray(vectors, dtype=np.float32)
        self.ids = ids
        self.row = {rid: i for i, rid in enumerate(ids)}

    def rank(self, keys, query_vec, k, use_dense=True):
        """keys: {record_id: tuple, larger is better}; ties broken by cosine with the query
        (if use_dense), then by corpus order. Returns [(record_id, cosine)]."""
        rids = sorted(keys, key=self.row.__getitem__)
        q = np.asarray(query_vec, dtype=np.float32).reshape(-1)
        q = q / np.linalg.norm(q)
        dense = self.vectors[[self.row[r] for r in rids]] @ q if rids else np.zeros(0)
        order = sorted(range(len(rids)), key=lambda i: (keys[rids[i]], dense[i] if use_dense else 0.0, -i),
                       reverse=True)
        return [(rids[i], float(dense[i])) for i in order[:k]]


class LazyGraphRetriever(GraphRanker):
    def __init__(self, index, vectors, ids, rerank_dense=True):
        super().__init__(vectors, ids)
        self.nodes = index["nodes"]
        self.rerank_dense = rerank_dense

    def search(self, query_vec, k, trace=None, question=None):
        if question is None:
            raise ValueError("the lazy-graph retriever needs the question text")
        trace = trace or Trace("-", "-", "-")
        with trace.span("lazy.match"):
            matched = sorted(question_ngrams(question) & self.nodes.keys())
        scores = {}
        for p in matched:
            node = self.nodes[p]
            for rid in node["records"]:
                scores[rid] = scores.get(rid, 0.0) + 1.0 / max(node["degree"], 1)
        trace.count("lazy.nodes_matched", len(matched))
        trace.count("lazy.candidates", len(scores))
        trace.count("lazy.community_hits", len({self.nodes[p]["community"] for p in matched}))
        return self.rank({r: (s,) for r, s in scores.items()}, query_vec, k, self.rerank_dense)


class KGRetriever(GraphRanker):
    def __init__(self, index, vectors, ids, question_entities="spacy", model=None, gateway=None):
        super().__init__(vectors, ids)
        if question_entities not in ("spacy", "llm"):
            raise ValueError("params.question_entities must be 'spacy' or 'llm'")
        self.entities = index["entities"]
        self.mode = question_entities
        self.model = model
        self.gateway = gateway

    def question_entities(self, question) -> set:
        if self.mode == "spacy":
            return question_ngrams(question) & self.entities.keys()
        reply = self.gateway.chat(self.model, [{"role": "user", "content": QUESTION_ENTITIES_PROMPT.format(
            question=question)}], temperature=0.0)
        match = re.search(r"\{.*\}", reply, re.DOTALL)
        names = json.loads(match.group(0)).get("entities", []) if match else []
        names = [n for n in names if isinstance(n, str)]
        return {normalise(doc) for doc in load_nlp().pipe(names)} & self.entities.keys()

    def search(self, query_vec, k, trace=None, question=None):
        if question is None:
            raise ValueError("the kg retriever needs the question text")
        trace = trace or Trace("-", "-", "-")
        with trace.span("kg.match"):
            matched = self.question_entities(question)
        hop = {n for e in matched for n in self.entities[e]["neighbours"]} - matched
        direct, near = {}, {}
        for names, counts in ((matched, direct), (hop, near)):
            for e in names:
                for rid in self.entities[e]["records"]:
                    counts[rid] = counts.get(rid, 0) + 1
        keys = {r: (direct.get(r, 0), near.get(r, 0)) for r in direct.keys() | near.keys()}
        trace.count("kg.entities_matched", len(matched))
        trace.count("kg.entities_1hop", len(hop))
        trace.count("kg.candidates", len(keys))
        return self.rank(keys, query_vec, k)


def load_index(path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
