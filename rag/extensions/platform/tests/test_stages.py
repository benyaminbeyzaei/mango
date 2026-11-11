import json

import numpy as np

from mlsrag.config import baseline
from mlsrag.course import ExactDenseRetriever, paths
from mlsrag.stages import (
    COURSE_SYSTEM_PROMPT,
    CachedASR,
    ExactRetriever,
    GatewayGenerator,
    build_stages,
)

from conftest import needs_data


class FakeGateway:
    def __init__(self):
        self.chats = []

    def encoder(self, name):
        return None

    def chat(self, model, messages, temperature=0.0, max_tokens=None, timeout=None):
        self.chats.append((model, messages, temperature, timeout))
        return "answer"


class FakeData:
    def record_text(self, rid):
        return f"text of {rid}"


def test_cached_asr_looks_up_by_audio_stem(tmp_path):
    p = tmp_path / "t.jsonl"
    p.write_text(json.dumps({"query_id": "q_1", "transcript": "hello"}) + "\n")
    assert CachedASR(p).transcribe("audio/practice/q_1.wav") == "hello"


def test_generator_uses_course_prompt_and_temperature():
    g = FakeGateway()
    assert GatewayGenerator(g, "m").generate("p") == "answer"
    model, messages, temperature, timeout = g.chats[0]
    assert messages[0] == {"role": "system", "content": COURSE_SYSTEM_PROMPT}
    assert temperature == 0.1
    assert timeout == 420


@needs_data
def test_build_stages_from_baseline(tmp_path):
    cfg = baseline().model_copy(update={"asr_cache": str(tmp_path / "t.jsonl")})
    stages = build_stages(cfg, FakeGateway(), FakeData())
    assert "[Document 1]\ntext of r_a" in stages.context.build("q?", ["r_a", "r_b"])
    assert stages.generator.model == cfg.generator_model


@needs_data
def test_exact_retriever_matches_course_retriever():
    emb = paths.data_dir / "embeddings"
    course = ExactDenseRetriever(emb / "corpus_enc_a.npy", emb / "corpus_enc_a_ids.json")
    vec = np.random.default_rng(0).standard_normal(384).astype(np.float32)
    hits = ExactRetriever("enc_a").search(vec, 100)
    expected = course.retrieve(vec, k=100)
    assert [r for r, _ in hits] == expected["record_ids"]
    assert [s for _, s in hits] == expected["scores"]
