"""Grep feedback: static grep, the model keeps the relevant hits, one new search with words from
the kept records (relevance feedback), then one rerank. Nothing kept: the static ranking."""
import json

from ..prompts import grep_v1
from .grep_agentic import show
from .grep_static import ask, clean_patterns, scan
from .standard import RANKED_DEPTH, finish

POOL = 20  # candidates shown to the keep and rerank calls


def pick(reply, allowed, limit=None):
    """Ids from a JSON list reply that are in `allowed`, in reply order, no duplicates."""
    out = []
    for rid in reply if isinstance(reply, list) else []:
        if isinstance(rid, str) and rid in allowed and rid not in out:
            out.append(rid)
    return out[:limit]


def retrieve(stages, config, trace, question):
    """(ranked ids, scores, context ids or None). Calls: patterns, keep, expand, rerank; 2 if nothing kept."""
    a = trace.artifacts

    p0 = clean_patterns(ask(stages, config, trace, grep_v1.PATTERNS.format(question=question)))
    a["patterns"] = [p0]
    hits0 = scan(stages, trace, p0)
    score = {rid: float(n) for rid, n, _ in hits0}
    static = [rid for rid, _, _ in hits0]

    kept = []
    if hits0:
        prompt = grep_v1.KEEP.format(question=question, hits=show(stages, static[:POOL], 300))
        kept = pick(ask(stages, config, trace, prompt), set(static[:POOL]))
    a["kept_ids"] = kept
    trace.count("grep.kept", len(kept))
    if not kept:
        return static, [score[rid] for rid in static], None

    texts = stages.retriever.records
    prompt = grep_v1.EXPAND.format(
        question=question, patterns=json.dumps(p0),
        kept="\n".join(f"[{rid}] {texts[rid][:300]}" for rid in kept[:5]),
    )
    p1 = clean_patterns(ask(stages, config, trace, prompt))
    a["patterns"].append(p1)
    seen = set(static[:POOL])
    hits1 = [h for h in scan(stages, trace, p1) if h[0] not in seen]
    for rid, n, _ in hits1:
        score.setdefault(rid, float(n))
    expanded = [rid for rid, _, _ in hits1]

    pool = (kept + expanded)[:POOL]
    prompt = grep_v1.CLEANUP.format(question=question, hits=show(stages, pool, 300), k=config.k)
    reply = ask(stages, config, trace, prompt)
    reranked = pick(reply, set(pool), config.k)
    a["reranked_ids"] = reranked

    ranked = []
    for rid in reranked + pool + static + expanded:
        if rid not in ranked:
            ranked.append(rid)
    ranked = ranked[:RANKED_DEPTH]
    context = reranked if isinstance(reply, list) else None  # None: unusable reply, the context is the top k
    return ranked, [score.get(rid, 0.0) for rid in ranked], context


def run_query(query_id, audio_path, stages, config, trace) -> dict:
    a = trace.artifacts
    with trace.span("asr"):
        a["transcript"] = stages.asr.transcribe(audio_path)
    with trace.span("retrieve"):  # every LLM call and scan up to the final ranking
        ranked, scores, context = retrieve(stages, config, trace, a["transcript"])
    return finish(stages, config, trace, ranked, scores, context_ids=context)
