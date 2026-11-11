"""RerankRetriever: take the base retriever's top `pool` records, re-score each (question, record text)
pair with a local cross-encoder (through the gateway, so it is timed and cached), and put them first in
the new order. The rest of the base ranking follows unchanged, so Recall@100 stays measurable."""
from ..trace import Trace


class RerankRetriever:
    uses_question = True  # the standard flow passes question=<transcript>

    def __init__(self, base, data, gateway, model, pool=50, revision=None):
        self.base, self.data, self.gateway = base, data, gateway
        self.model, self.pool, self.revision = model, pool, revision

    def search(self, query_vec, k, trace=None, question=None):
        trace = trace or Trace("-", "-", "-")
        extra = {"question": question} if getattr(self.base, "uses_question", False) else {}
        hits = self.base.search(query_vec, max(k, self.pool), trace=trace, **extra)
        head, tail = hits[: self.pool], hits[self.pool :]
        with trace.span("rerank"):
            texts = [self.data.record_text(rid) for rid, _ in head]
            scores, truncated = self.gateway.rerank(self.model, question, texts, self.revision)
        trace.count("rerank.pool", len(head))
        trace.count("rerank.truncated", truncated)
        order = sorted(range(len(head)), key=lambda i: -scores[i])  # stable: ties keep the base order
        return ([(head[i][0], scores[i]) for i in order] + tail)[:k]
