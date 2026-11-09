"""The course baseline: asr -> encode -> retrieve -> context -> generate."""
from ..stages import load_record_subset

RANKED_DEPTH = 100  # keep this many ranked ids so Recall@100 can be computed
ALL_CANDIDATES = 1_000_000  # more than the corpus: a retriever returns every candidate it has


def retrieve(stages, config, trace, question) -> list[tuple[str, float]]:
    """Encode (unless the retriever needs no vector) and search; at most RANKED_DEPTH hits."""
    vec = None
    if getattr(stages.retriever, "uses_vector", True):
        with trace.span("encode"):
            vec = stages.encoder.encode(question)
    depth = max(config.k, RANKED_DEPTH)
    subset = config.params.get("record_subset")  # compare on a record subset: keep only its ids
    extra = {"question": question} if getattr(stages.retriever, "uses_question", False) else {}
    with trace.span("retrieve"):
        hits = stages.retriever.search(vec, ALL_CANDIDATES if subset else depth, trace=trace, **extra)
        if subset:
            record_ids = load_record_subset(subset)
            trace.count("subset.candidates_in", len(hits))
            hits = [h for h in hits if h[0] in record_ids]
            trace.count("subset.candidates_kept", len(hits))
            hits = hits[:depth]
    return hits


def finish(stages, config, trace, ranked_ids, scores, context_ids=None) -> dict:
    """Context (top k, or context_ids when given) and generation."""
    a = trace.artifacts
    a["ranked_ids"] = ranked_ids
    a["scores"] = scores
    with trace.span("context"):
        ids = ranked_ids[: config.k] if context_ids is None else context_ids
        a["context_ids"] = ids
        a["prompt"] = stages.context.build(a["transcript"], ids, trace)
    a["answer"] = None
    if stages.generator is not None:
        with trace.span("generate"):
            a["answer"] = stages.generator.generate(a["prompt"])
    return a


def run_query(query_id, audio_path, stages, config, trace) -> dict:
    a = trace.artifacts
    with trace.span("asr"):
        a["transcript"] = stages.asr.transcribe(audio_path)
    hits = retrieve(stages, config, trace, a["transcript"])
    return finish(stages, config, trace, [rid for rid, _ in hits], [score for _, score in hits])
