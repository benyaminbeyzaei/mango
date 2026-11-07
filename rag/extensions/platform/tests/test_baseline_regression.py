"""mlsrag's baseline must give the course evaluator's numbers on the same transcripts."""
import numpy as np
import pytest

from mlsrag.config import baseline
from mlsrag.course import (
    CourseData,
    ExactDenseRetriever,
    LocalEncoder,
    mrr_at_k,
    ndcg_at_k,
    paths,
    recall_at_k,
)
from mlsrag.flows import get_flow
from mlsrag.gateway import ModelGateway
from mlsrag.runner import METRIC_KEYS, retrieval_metrics
from mlsrag.stages import build_stages, checkout_path, load_transcripts
from mlsrag.subsets import load_subset
from mlsrag.trace import Trace


@pytest.mark.slow
def test_baseline_matches_course_code_path(tmp_path):
    config = baseline()
    transcripts = load_transcripts(checkout_path(config.asr_cache))
    ids = load_subset("practice-smoke20")
    assert all(q in transcripts for q in ids), "run `mlsrag transcribe` for the smoke subset first"

    data = CourseData(paths.data_dir)
    manifest = {row["query_id"]: row for row in data.load_audio_manifest()}
    qrels = data.load_qrels("practice")

    # mlsrag, with a fresh gateway cache so the encoder really runs
    gateway = ModelGateway(cache_dir=tmp_path / "cache", log_path=tmp_path / "log.jsonl")
    stages = build_stages(config, gateway, data)
    stages.generator = None
    flow = get_flow(config.flow)
    ours = []
    for qid in ids:
        trace = Trace("test", qid, config.name)
        gateway.trace = trace
        out = flow(qid, paths.data_dir / manifest[qid]["audio_file"], stages, config, trace)
        ours.append(retrieval_metrics(out["ranked_ids"], qrels[qid]))

    # the course path, written as rag/scripts/evaluate_rag.py does it
    encoder = LocalEncoder(paths.models_dir / "enc_a")
    emb = paths.data_dir / "embeddings"
    retriever = ExactDenseRetriever(emb / "corpus_enc_a.npy", emb / "corpus_enc_a_ids.json")
    course = []
    for qid in ids:
        ranked = retriever.retrieve(encoder.encode(transcripts[qid]), k=100)["record_ids"]
        rel = qrels[qid]
        course.append({
            "Recall@5": recall_at_k(ranked, rel, 5),
            "Recall@10": recall_at_k(ranked, rel, 10),
            "Recall@20": recall_at_k(ranked, rel, 20),
            "Recall@100": recall_at_k(ranked, rel, 100),
            "MRR@10": mrr_at_k(ranked, rel, 10),
            "NDCG@10": ndcg_at_k(ranked, rel, 10),
        })

    for key in METRIC_KEYS:
        a = np.mean([m[key] for m in ours])
        b = np.mean([m[key] for m in course])
        assert abs(a - b) <= 1e-9, (key, a, b)
