"""The five stage interfaces and the adapters that wrap the course code behind them."""
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Protocol

import numpy as np

from .config import ENCODER_SPACE
from .course import ExactDenseRetriever, SimpleContextBuilder, paths
from .maps import map_path

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
    # a retriever with `uses_question = True` is also given `question=<transcript>` by the flow


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


BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class ReEncode:
    """Encode the query with the encoder that produced the index (optionally with bge's query prefix)."""

    def __init__(self, gateway, name, prefix=False):
        self.inner = GatewayEncoder(gateway, name)
        self.prefix = BGE_QUERY_PREFIX if prefix else ""

    def encode(self, text: str) -> np.ndarray:
        return self.inner.encode(self.prefix + text)


class Translate:
    """Map the inner encoder's vector into the index space with a fitted linear map, then normalise."""

    def __init__(self, inner, map_path):
        map_path = Path(map_path)
        if not map_path.exists():
            raise FileNotFoundError(f"{map_path} missing; run `mlsrag fit-map` first")
        self.inner = inner
        self.w = np.load(map_path)

    def encode(self, text: str) -> np.ndarray:
        v = self.inner.encode(text) @ self.w
        return v / np.linalg.norm(v)


def build_encoder(config, gateway) -> Encoder:
    """Resolve config.mismatch_policy into the query encoder (error is enforced by RunConfig)."""
    space = ENCODER_SPACE[config.query_encoder]
    if space == config.corpus_index or config.mismatch_policy in ("error", "as-is"):
        return GatewayEncoder(gateway, config.query_encoder)
    if config.mismatch_policy == "re-encode":
        index_encoder = {v: k for k, v in ENCODER_SPACE.items()}[config.corpus_index]
        prefix = config.params.get("bge_prefix", False)
        if prefix and index_encoder != "bge-small":
            raise ValueError("params.bge_prefix is for bge-small (enc_b) only")
        return ReEncode(gateway, index_encoder, prefix)
    method = config.params.get("map", "lstsq")
    return Translate(GatewayEncoder(gateway, config.query_encoder),
                     map_path(space, config.corpus_index, method))


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


def topic_retriever(config, gateway):
    """`params.index`: the topics pipeline config (YAML) whose topic_index output to load."""
    from .preprocess import load_pipeline_config, topics_pipeline
    from .retrievers.topic import TopicRetriever, load_index

    if config.corpus_index != "enc_a":
        raise ValueError("the topic index is built from enc_a vectors")
    pipeline = topics_pipeline()
    pipeline_config = load_pipeline_config(checkout_path(config.params["index"]))
    rows = pipeline.status(pipeline_config)
    if rows[-1]["status"] != "ok":
        raise FileNotFoundError(
            f"topic index for {config.params['index']} is not built; run "
            f"`mlsrag preprocess run --pipeline topics --config {config.params['index']}`"
        )
    index = load_index(pipeline.stage_dir("topic_index", rows[-1]["hash"]) / "index.json")
    exact = ExactRetriever("enc_a").inner
    return TopicRetriever(index, exact.embeddings, exact.record_ids, m=config.params.get("m", 4), gateway=gateway)


def built_index(pipeline, config_path, file_name) -> Path:
    """The last stage's `file_name` for a pipeline config, or an error saying how to build it."""
    from .preprocess import load_pipeline_config

    rows = pipeline.status(load_pipeline_config(checkout_path(config_path)))
    if rows[-1]["status"] != "ok":
        raise FileNotFoundError(f"{pipeline.name} index for {config_path} is not built; run "
                                f"`mlsrag preprocess run --pipeline {pipeline.name} --config {config_path}`")
    return pipeline.stage_dir(rows[-1]["stage"], rows[-1]["hash"]) / file_name


def lazy_graph_retriever(config, gateway):
    """`params.index`: the lazy-graph pipeline config; `params.rerank_dense` (default true)."""
    from .preprocess import lazy_graph_pipeline
    from .retrievers.graph import LazyGraphRetriever, load_index

    if config.corpus_index != "enc_a":
        raise ValueError("the graph retrievers order candidates with enc_a vectors")
    index = load_index(built_index(lazy_graph_pipeline(), config.params["index"], "index.json"))
    exact = ExactRetriever("enc_a").inner
    return LazyGraphRetriever(index, exact.embeddings, exact.record_ids,
                              rerank_dense=config.params.get("rerank_dense", True), gateway=gateway)


def kg_retriever(config, gateway):
    """`params.index`: the kg pipeline config; `params.question_entities`: spacy | llm
    (`params.question_model`, default the generator model)."""
    from .preprocess import kg_pipeline
    from .retrievers.graph import KGRetriever, load_index

    if config.corpus_index != "enc_a":
        raise ValueError("the graph retrievers order candidates with enc_a vectors")
    index = load_index(built_index(kg_pipeline(), config.params["index"], "index.json"))
    exact = ExactRetriever("enc_a").inner
    return KGRetriever(index, exact.embeddings, exact.record_ids,
                       question_entities=config.params.get("question_entities", "spacy"),
                       model=config.params.get("question_model", config.generator_model), gateway=gateway)


@lru_cache(maxsize=8)
def load_record_subset(path) -> frozenset:
    """`params.record_subset`: a text file with one record id per line (relative to the checkout root)."""
    lines = checkout_path(path).read_text(encoding="utf-8").splitlines()
    return frozenset(line.strip() for line in lines if line.strip())


@dataclass
class Stages:
    asr: ASR
    encoder: Encoder
    retriever: Retriever
    context: ContextBuilder
    generator: Generator | None  # None for retrieval-only runs


# name -> factory(config, gateway, data). A new retriever or context builder registers here.
RETRIEVERS = {
    "exact": lambda config, gateway, data: ExactRetriever(config.corpus_index),
    "topic": lambda config, gateway, data: topic_retriever(config, gateway),
    "lazy-graph": lambda config, gateway, data: lazy_graph_retriever(config, gateway),
    "kg": lambda config, gateway, data: kg_retriever(config, gateway),
}
CONTEXT_BUILDERS = {
    "simple": lambda config, gateway, data: SimpleContext(data),
}


def build_stages(config, gateway, data) -> Stages:
    if config.asr == "cached":
        if not config.asr_cache:
            raise ValueError("asr: cached needs asr_cache")
        asr = CachedASR(config.asr_cache)
    elif config.asr == "whisper1":
        asr = WhisperASR(gateway)
    else:
        asr = CourseASR()
    if config.retriever not in RETRIEVERS:
        raise ValueError(f"unknown retriever {config.retriever!r}; known: {sorted(RETRIEVERS)}")
    if config.context_builder not in CONTEXT_BUILDERS:
        raise ValueError(f"unknown context_builder {config.context_builder!r}; known: {sorted(CONTEXT_BUILDERS)}")
    return Stages(
        asr=asr,
        encoder=build_encoder(config, gateway),
        retriever=RETRIEVERS[config.retriever](config, gateway, data),
        context=CONTEXT_BUILDERS[config.context_builder](config, gateway, data),
        generator=GatewayGenerator(gateway, config.generator_model),
    )
