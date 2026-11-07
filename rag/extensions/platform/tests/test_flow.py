import numpy as np

from mlsrag.config import baseline
from mlsrag.flows import get_flow
from mlsrag.stages import Stages
from mlsrag.trace import Trace


class Fake:
    def transcribe(self, audio_path):
        return "what is x"

    def encode(self, text):
        return np.ones(3, dtype=np.float32)

    def search(self, vec, k):
        self.k = k
        return [(f"r{i}", 1.0 - i / 1000) for i in range(k)]

    def build(self, question, record_ids):
        self.context_ids = record_ids
        return f"{question} | {','.join(record_ids)}"

    def generate(self, prompt):
        return "x is y"


def test_standard_flow_records_artifacts_and_spans():
    f = Fake()
    stages = Stages(asr=f, encoder=f, retriever=f, context=f, generator=f)
    trace = Trace("r", "q1", "baseline")
    out = get_flow("standard")("q1", "q1.wav", stages, baseline(), trace)
    assert f.k == 100 and len(out["ranked_ids"]) == 100 and len(out["scores"]) == 100
    assert f.context_ids == ["r0", "r1", "r2", "r3", "r4"]
    assert out["transcript"] == "what is x" and out["answer"] == "x is y"
    assert out["prompt"].startswith("what is x |")
    assert [s.name for s in trace.spans] == ["asr", "encode", "retrieve", "context", "generate"]


def test_no_generator_skips_generation():
    f = Fake()
    stages = Stages(asr=f, encoder=f, retriever=f, context=f, generator=None)
    trace = Trace("r", "q1", "baseline")
    out = get_flow("standard")("q1", "q1.wav", stages, baseline(), trace)
    assert out["answer"] is None
    assert "generate" not in [s.name for s in trace.spans]
