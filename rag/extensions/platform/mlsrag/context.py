"""Context builders that select evidence: a token budget, and duplicate removal before the budget.
Both write the course prompt (SimpleContextBuilder) around the evidence they keep."""
import re
from types import SimpleNamespace

import numpy as np

from .course import SimpleContextBuilder

SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def estimate_tokens(text) -> int:
    """A word-based estimate (words x 1.3); close enough to compare prompt sizes."""
    return round(len(text.split()) * 1.3)


def truncate(text, max_tokens) -> str:
    """The longest run of whole sentences from the start of `text` within `max_tokens` ('' if none fits)."""
    kept = ""
    for sentence in SENTENCE_END.split(text):
        longer = f"{kept} {sentence}" if kept else sentence
        if estimate_tokens(longer) > max_tokens:
            break
        kept = longer
    return kept


def apply_budget(texts, budget) -> tuple[list[str], bool]:
    """Texts in rank order until the budget is used; the last one cut at a sentence boundary."""
    out, used = [], 0
    for text in texts:
        n = estimate_tokens(text)
        if used + n <= budget:
            out.append(text)
            used += n
            continue
        cut = truncate(text, budget - used)
        if cut:
            out.append(cut)
        return out, True
    return out, False


class BudgetedContext:
    """`params.budget_tokens` (800): evidence tokens allowed in the prompt, question and instructions excluded."""

    def __init__(self, data, budget_tokens=800):
        self.data = data
        self.budget = budget_tokens
        self.inner = SimpleContextBuilder()

    def select(self, record_ids, trace):
        return record_ids

    def build(self, question, record_ids, trace=None) -> str:
        ids = self.select(record_ids, trace)
        texts, truncated = apply_budget([self.data.record_text(r) for r in ids], self.budget)
        if trace is not None:
            trace.count("context.records_used", len(texts))
            trace.count("context.tokens", sum(estimate_tokens(t) for t in texts))
            trace.count("context.truncated", int(truncated))
        kept = dict(zip(ids, texts, strict=False))
        return self.inner.build(question, list(kept), SimpleNamespace(record_text=kept.__getitem__))["prompt"]


class DedupeContext(BudgetedContext):
    """Drops a record whose cosine similarity to an already kept one exceeds `params.dedupe_threshold`
    (0.9), then applies the budget. Records without a vector (summaries) are always kept."""

    def __init__(self, data, vectors, ids, threshold=0.9, budget_tokens=800):
        super().__init__(data, budget_tokens)
        self.vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        self.row = {rid: i for i, rid in enumerate(ids)}
        self.threshold = threshold

    def select(self, record_ids, trace):
        kept, kept_vecs = [], []
        for rid in record_ids:
            if rid in self.row:
                v = self.vectors[self.row[rid]]
                if kept_vecs and max(float(v @ k) for k in kept_vecs) > self.threshold:
                    continue
                kept_vecs.append(v)
            kept.append(rid)
        if trace is not None:
            trace.count("context.dropped_duplicates", len(record_ids) - len(kept))
        return kept
