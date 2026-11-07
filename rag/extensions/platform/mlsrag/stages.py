"""The five stage interfaces and the adapters that wrap the course code behind them."""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from .course import ExactDenseRetriever, SimpleContextBuilder, paths

# Copied from rag/src/generator.py (ELMGenerator.generate), which keeps it inline.
COURSE_SYSTEM_PROMPT = (
    "Use the supplied evidence to answer the user's question. "
    "Do not invent facts that are unsupported by the evidence."
)
COURSE_TEMPERATURE = 0.1


class ASR(Protocol):
    def transcribe(self, audio_path) -> str: ...


class Encoder(Protocol):
    def encode(self, text: str) -> np.ndarray: ...


class Retriever(Protocol):
    def search(self, query_vec: np.ndarray, k: int) -> list[tuple[str, float]]: ...


class ContextBuilder(Protocol):
    def build(self, question: str, record_ids: list[str]) -> str: ...


class Generator(Protocol):
    def generate(self, prompt: str) -> str: ...


def checkout_path(path) -> Path:
    """Resolve a path from a config relative to the checkout root (the folder with rag/)."""
    path = Path(path)
    return path if path.is_absolute() else paths.rag_dir.parent / path


def load_transcripts(path) -> dict:
    path = Path(path)
    if not path.exists():
        return {}
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {r["query_id"]: r["transcript"] for r in rows}


class CachedASR:
    """Transcripts from a JSONL cache, looked up by the audio file's name (the query id)."""

    def __init__(self, path):
        self.path = checkout_path(path)
        self.transcripts = load_transcripts(self.path)

    def transcribe(self, audio_path) -> str:
        qid = Path(audio_path).stem
        if qid not in self.transcripts:
            raise KeyError(f"{qid} not in transcript cache {self.path}; run `mlsrag transcribe` first")
        return self.transcripts[qid]


class WhisperASR:
    def __init__(self, gateway):
        self.gateway = gateway

    def transcribe(self, audio_path) -> str:
        return self.gateway.transcribe(audio_path, model="whisper-1")


class CourseASR:
    """The course's own ASR (hw1-asr kernels). Needs the GPU model, so only works on the cluster."""

    def __init__(self):
        from src.asr_adapter import ASRAdapter

        self.adapter = ASRAdapter()

    def transcribe(self, audio_path) -> str:
        try:
            return self.adapter.transcribe(audio_path)
        except Exception as err:
            raise RuntimeError(
                "course ASR is not available here (it needs the GLM-ASR model and a GPU); "
                "use asr: cached or whisper1, or run on the cluster"
            ) from err


class GatewayEncoder:
    def __init__(self, gateway, name):
        self.gateway = gateway
        self.name = name
        gateway.encoder(name)  # load now so the first query's timing is not the model load

    def encode(self, text: str) -> np.ndarray:
        return self.gateway.embed(self.name, [text], role="query")[0]


class ExactRetriever:
    def __init__(self, index_name):
        emb = paths.data_dir / "embeddings"
        self.inner = ExactDenseRetriever(
            emb / f"corpus_{index_name}.npy", emb / f"corpus_{index_name}_ids.json"
        )

    def search(self, query_vec, k):
        r = self.inner.retrieve(query_vec, k=k)
        return list(zip(r["record_ids"], r["scores"], strict=True))


class SimpleContext:
    def __init__(self, data):
        self.data = data
        self.inner = SimpleContextBuilder()

    def build(self, question, record_ids) -> str:
        return self.inner.build(question, record_ids, self.data)["prompt"]


class GatewayGenerator:
    def __init__(self, gateway, model):
        self.gateway = gateway
        self.model = model

    def generate(self, prompt) -> str:
        messages = [
            {"role": "system", "content": COURSE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        return self.gateway.chat(self.model, messages, temperature=COURSE_TEMPERATURE)


@dataclass
class Stages:
    asr: ASR
    encoder: Encoder
    retriever: Retriever
    context: ContextBuilder
    generator: Generator | None  # None for retrieval-only runs


def build_stages(config, gateway, data) -> Stages:
    if config.asr == "cached":
        if not config.asr_cache:
            raise ValueError("asr: cached needs asr_cache")
        asr = CachedASR(config.asr_cache)
    elif config.asr == "whisper1":
        asr = WhisperASR(gateway)
    else:
        asr = CourseASR()
    if config.retriever != "exact":
        raise ValueError(f"unknown retriever {config.retriever!r}")
    if config.context_builder != "simple":
        raise ValueError(f"unknown context_builder {config.context_builder!r}")
    return Stages(
        asr=asr,
        encoder=GatewayEncoder(gateway, config.query_encoder),
        retriever=ExactRetriever(config.corpus_index),
        context=SimpleContext(data),
        generator=GatewayGenerator(gateway, config.generator_model),
    )
