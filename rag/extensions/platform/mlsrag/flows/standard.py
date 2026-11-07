"""The course baseline: asr -> encode -> retrieve -> context -> generate."""
from ..stages import load_record_subset

RANKED_DEPTH = 100  # keep this many ranked ids so Recall@100 can be computed
ALL_CANDIDATES = 1_000_000  # more than the corpus: a retriever returns every candidate it has


def run_query(query_id, audio_path, stages, config, trace) -> dict:
    a = trace.artifacts
    with trace.span("asr"):
        a["transcript"] = stages.asr.transcribe(audio_path)
    with trace.span("encode"):
        vec = stages.encoder.encode(a["transcript"])
    depth = max(config.k, RANKED_DEPTH)
    subset = config.params.get("record_subset")  # compare on a record subset: keep only its ids
    extra = {"question": a["transcript"]} if getattr(stages.retriever, "uses_question", False) else {}
    # One-off costs a retriever reports (BM25 index build or load), on the first query only.
    if getattr(stages.retriever, "build_counters", None):
        for key, value in stages.retriever.build_counters.items():
            trace.count(key, value)
        stages.retriever.build_counters = {}
    with trace.span("retrieve"):
        hits = stages.retriever.search(vec, ALL_CANDIDATES if subset else depth, **extra)
        if subset:
            record_ids = load_record_subset(subset)
            trace.count("subset.candidates_in", len(hits))
            hits = [h for h in hits if h[0] in record_ids]
            trace.count("subset.candidates_kept", len(hits))
            hits = hits[:depth]
    a["ranked_ids"] = [rid for rid, _ in hits]
    a["scores"] = [score for _, score in hits]
    with trace.span("context"):
        a["prompt"] = stages.context.build(a["transcript"], a["ranked_ids"][: config.k])
    a["answer"] = None
    if stages.generator is not None:
        with trace.span("generate"):
            a["answer"] = stages.generator.generate(a["prompt"])
    return a
