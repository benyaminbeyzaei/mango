"""The five stages of a query (asr, encoder, retriever, context, generator) and the adapters that
wrap the course code behind them."""
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

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


def build_encoder(config, gateway):
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

    def search(self, query_vec, k, trace=None):
        r = self.inner.retrieve(query_vec, k=k)
        return list(zip(r["record_ids"], r["scores"], strict=True))


class SimpleContext:
    def __init__(self, data):
        self.data = data
        self.inner = SimpleContextBuilder()

    def build(self, question, record_ids, trace=None) -> str:
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


def stage_dir(pipeline_name, config, stage) -> Path:
    """The output folder of one preprocessing stage for the pipeline config in `params.index`,
    or an error saying how to build it."""
    from .preprocess import PIPELINES, load_pipeline_config

    if config.corpus_index != "enc_a":
        raise ValueError("the preprocessing indexes are built from enc_a vectors")
    pipeline = PIPELINES[pipeline_name]()
    rows = {r["stage"]: r for r in pipeline.status(load_pipeline_config(checkout_path(config.params["index"])))}
    if rows[stage]["status"] != "ok":
        raise FileNotFoundError(
            f"{stage} for {config.params['index']} is not built; run "
            f"`mlsrag preprocess run --pipeline {pipeline_name} --config {config.params['index']}`"
        )
    return pipeline.stage_dir(stage, rows[stage]["hash"])


def enc_a() -> tuple:
    """(vectors, record ids) of the corpus enc_a index, loaded as the preprocessing stages load it."""
    from .preprocess.cluster import load_vectors

    ids = json.loads((paths.data_dir / "embeddings" / "corpus_enc_a_ids.json").read_text(encoding="utf-8"))
    return load_vectors(ids), ids


def topic_retriever(config):
    """`params.index`: the topics pipeline config (YAML) whose topic_index output to load."""
    from .retrievers.topic import TopicRetriever, load_index

    index = load_index(stage_dir("topics", config, "topic_index") / "index.json")
    return TopicRetriever(index, *enc_a(), m=config.params.get("m", 4))


TOC_ROUTE_MODEL = "meta-llama/Llama-3.3-70B-Instruct"


def toc_retriever(config, gateway):
    """`params`: index (pipeline config), sections (2), m (4), route (centroid | llm), route_model."""
    from .retrievers.toc import TocRetriever, load_toc

    p = config.params
    toc = load_toc(stage_dir("toc", config, "toc_index") / "toc.json")
    return TocRetriever(toc, *enc_a(), sections=p.get("sections", 2), m=p.get("m", 4),
                        route=p.get("route", "centroid"), gateway=gateway,
                        model=p.get("route_model", TOC_ROUTE_MODEL))


def raptor_retriever(config):
    """`params`: index (pipeline config), summary_as_evidence (false), expand_top (3)."""
    from .retrievers.raptor import SummaryRetriever, load_summaries

    p = config.params
    if p.get("summary_as_evidence", False) and config.context_builder != "with-summaries":
        raise ValueError("summary_as_evidence: true needs context_builder: with-summaries")
    vectors, rows = load_summaries(stage_dir("raptor", config, "summary_index"))
    return SummaryRetriever(vectors, rows, *enc_a(), summary_as_evidence=p.get("summary_as_evidence", False),
                            expand_top=p.get("expand_top", 3))


def summary_context(config, data):
    """SimpleContext that can also show `summary:<cluster>` evidence from the raptor index."""
    from .retrievers.raptor import SummaryData, load_summaries

    _, rows = load_summaries(stage_dir("raptor", config, "summary_index"))
    return SimpleContext(SummaryData(data, rows))


def lazy_graph_retriever(config):
    """`params.index`: the lazy-graph pipeline config; `params.rerank_dense` (default true)."""
    from .retrievers.graph import LazyGraphRetriever, load_index

    index = load_index(stage_dir("lazy-graph", config, "lazy_index") / "index.json")
    return LazyGraphRetriever(index, *enc_a(), rerank_dense=config.params.get("rerank_dense", True))


def kg_retriever(config, gateway):
    """`params.index`: the kg pipeline config; `params.question_entities`: spacy | llm
    (`params.question_model`, default the generator model)."""
    from .retrievers.graph import KGRetriever, load_index

    index = load_index(stage_dir("kg", config, "kg_index") / "index.json")
    return KGRetriever(index, *enc_a(), question_entities=config.params.get("question_entities", "spacy"),
                       model=config.params.get("question_model", config.generator_model), gateway=gateway)


@lru_cache(maxsize=8)
def load_record_subset(path) -> frozenset:
    """`params.record_subset`: a text file with one record id per line (relative to the checkout root)."""
    lines = checkout_path(path).read_text(encoding="utf-8").splitlines()
    return frozenset(line.strip() for line in lines if line.strip())


@dataclass
class Stages:
    asr: object  # .transcribe(audio_path) -> str
    encoder: object  # .encode(text) -> vector
    # .search(query_vec, k, trace=None) -> [(record_id, score)]; with `uses_question = True` the flow
    # also passes question=<transcript>; with `uses_vector = False` it skips encoding and passes None
    retriever: object
    context: object  # .build(question, record_ids, trace=None) -> prompt
    generator: object | None  # .generate(prompt) -> str; None for retrieval-only runs
    gateway: object = None  # for flows that call a model themselves (grep)


def bm25_retriever(data):
    from .retrievers.bm25 import BM25Retriever

    return BM25Retriever(data)


def grep_index(data):
    from .retrievers.grep import GrepIndex

    return GrepIndex(data)


# name -> factory(config, gateway, data). A new retriever or context builder registers here.
RETRIEVERS = {
    "exact": lambda config, gateway, data: ExactRetriever(config.corpus_index),
    "topic": lambda config, gateway, data: topic_retriever(config),
    "lazy-graph": lambda config, gateway, data: lazy_graph_retriever(config),
    "kg": lambda config, gateway, data: kg_retriever(config, gateway),
    "toc": lambda config, gateway, data: toc_retriever(config, gateway),
    "raptor-lite": lambda config, gateway, data: raptor_retriever(config),
    "bm25": lambda config, gateway, data: bm25_retriever(data),
    "grep": lambda config, gateway, data: grep_index(data),  # only for the grep flows, which call scan()
}
def budgeted_context(config, data):
    from .context import BudgetedContext

    return BudgetedContext(data, config.params.get("budget_tokens", 800))


def dedupe_context(config, data):
    from .context import DedupeContext

    return DedupeContext(data, *enc_a(), threshold=config.params.get("dedupe_threshold", 0.9),
                         budget_tokens=config.params.get("budget_tokens", 800))


CONTEXT_BUILDERS = {
    "simple": lambda config, gateway, data: SimpleContext(data),
    "budgeted": lambda config, gateway, data: budgeted_context(config, data),
    "dedupe": lambda config, gateway, data: dedupe_context(config, data),
    "with-summaries": lambda config, gateway, data: summary_context(config, data),
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
    if (config.retriever == "grep") != config.flow.startswith("grep-"):
        raise ValueError("retriever grep goes with flow grep-static or grep-agentic, and only with them")
    return Stages(
        asr=asr,
        encoder=build_encoder(config, gateway),
        retriever=RETRIEVERS[config.retriever](config, gateway, data),
        context=CONTEXT_BUILDERS[config.context_builder](config, gateway, data),
        generator=GatewayGenerator(gateway, config.generator_model),
        gateway=gateway,
    )
