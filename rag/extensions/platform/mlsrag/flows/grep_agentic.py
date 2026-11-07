"""Grep agentic: grep, show the model the top hits, let it refine the patterns or stop, then clean up."""
import json

from ..prompts import grep_v1
from .grep_static import ask, clean_patterns, finish, scan

MAX_ITERATIONS = 3  # scans per query, so at most 1 + 2 + 1 = 4 LLM calls


def show(stages, hits, chars):
    texts = stages.retriever.records
    return "\n".join(f"[{rid}] {texts[rid][:chars]}" for rid, _, _ in hits)


def retrieve(stages, config, trace, question):
    """(ranked ids, scan hits, kept ids or None). Calls: 1 for patterns, up to 2 refinements, 1 clean-up."""
    a = trace.artifacts

    patterns = clean_patterns(ask(stages, config, trace, grep_v1.PATTERNS.format(question=question)))
    a["patterns"] = [patterns]
    hits = scan(stages, trace, patterns)

    for _ in range(MAX_ITERATIONS - 1):
        prompt = grep_v1.REFINE.format(
            question=question, patterns=json.dumps(patterns), hits=show(stages, hits[:10], 200) or "(none)"
        )
        reply = ask(stages, config, trace, prompt)
        new = clean_patterns(reply.get("patterns")) if isinstance(reply, dict) else []
        if not new or (isinstance(reply, dict) and reply.get("stop")):
            break
        new_hits = scan(stages, trace, new)
        a["patterns"].append(new)
        if not new_hits:  # refining found nothing: keep the previous hits
            break
        patterns, hits = new, new_hits

    kept = None  # None: no clean-up (no hits) or its reply was not a list; the context is then the top k
    if hits:
        prompt = grep_v1.CLEANUP.format(question=question, hits=show(stages, hits[:20], 300), k=config.k)
        reply = ask(stages, config, trace, prompt)
        if isinstance(reply, list):
            top = {rid for rid, _, _ in hits[:20]}
            kept = []
            for rid in reply:
                if isinstance(rid, str) and rid in top and rid not in kept:
                    kept.append(rid)
            kept = kept[: config.k]
    a["kept_ids"] = kept
    # The ranking for the metrics: kept ids first, then the rest of the scan order (so Recall@100
    # stays comparable). The context gets only the kept ids.
    ranked = (kept or []) + [rid for rid, _, _ in hits if rid not in (kept or [])]
    return ranked, hits, kept


def run_query(query_id, audio_path, stages, config, trace) -> dict:
    a = trace.artifacts
    with trace.span("asr"):
        a["transcript"] = stages.asr.transcribe(audio_path)
    question = a["transcript"]
    with trace.span("retrieve"):  # every LLM call and scan up to the final ranking
        ranked, hits, kept = retrieve(stages, config, trace, question)
    score = {rid: float(n) for rid, n, _ in hits}
    return finish(stages, config, trace, ranked, [score[rid] for rid in ranked], context_ids=kept)
