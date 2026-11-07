"""The course baseline: asr -> encode -> retrieve -> context -> generate."""

RANKED_DEPTH = 100  # keep this many ranked ids so Recall@100 can be computed


def run_query(query_id, audio_path, stages, config, trace) -> dict:
    a = trace.artifacts
    with trace.span("asr"):
        a["transcript"] = stages.asr.transcribe(audio_path)
    with trace.span("encode"):
        vec = stages.encoder.encode(a["transcript"])
    with trace.span("retrieve"):
        hits = stages.retriever.search(vec, max(config.k, RANKED_DEPTH))
    a["ranked_ids"] = [rid for rid, _ in hits]
    a["scores"] = [score for _, score in hits]
    with trace.span("context"):
        a["prompt"] = stages.context.build(a["transcript"], a["ranked_ids"][: config.k])
    a["answer"] = None
    if stages.generator is not None:
        with trace.span("generate"):
            a["answer"] = stages.generator.generate(a["prompt"])
    return a
