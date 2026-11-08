"""Request and response bodies of the API (the OpenAPI contract the web client is built against)."""
from typing import Literal

from pydantic import BaseModel, Field

from ..config import RunConfig


class HealthResponse(BaseModel):
    status: str
    git_commit: str | None


class TextInput(BaseModel):
    text: str


class VectorInput(BaseModel):
    vector: list[float] = Field(description="384 floats")
    space: Literal["enc_a", "enc_b"]
    question: str = Field("", description="question text for the prompt (the vector replaces only the encoder)")


class QueryRequest(BaseModel):
    config: str | dict = Field(description="config name from GET /configs, or a full RunConfig")
    input: TextInput | VectorInput


class Hit(BaseModel):
    record_id: str
    score: float
    text: str = Field(description="first 500 characters of the record")


class QueryResponse(BaseModel):
    answer: str | None
    transcript: str
    retrieved: list[Hit]
    prompt: str
    trace: dict = Field(description="Trace.to_dict(): spans, counters, cost, resources, artifacts")


class NamedItem(BaseModel):
    name: str
    available: bool = True
    space: str | None = None  # encoders only: the corpus index their vectors belong to


class ModelsResponse(BaseModel):
    asr: list[NamedItem]
    encoders: list[NamedItem]
    indexes: list[str]
    chat_models: list[str]
    retrievers: list[str]
    context_builders: list[str]
    flows: list[str]


class ConfigEntry(BaseModel):
    name: str
    source: str  # "preset" or the YAML path relative to the platform folder
    config: RunConfig


class RunSummary(BaseModel):
    run_id: str
    config_name: str
    subset: str
    n_queries: int
    finished: str | None
    fresh: bool = Field(False, description="latency and cost are valid only for fresh runs")
    retrieval: dict[str, float] = Field(description="metrics.json retrieval block: Recall@k, MRR@10, NDCG@10")
    cost: dict = Field(description="metrics.json cost block: elm_calls, tokens, audio_seconds, est_usd")


class ResultsResponse(BaseModel):
    runs: list[RunSummary]
    matrices: list[dict] = Field(description="rag/extensions/results/matrices/<name>/summary.json, one per matrix")
    asr_compare: list[dict] = Field(description="rag/extensions/results/asr-compare/<name>/summary.json")


class RunDetail(BaseModel):
    config: dict
    metrics: dict


class QueriesPage(BaseModel):
    total: int
    offset: int
    items: list[dict]


class RecordResponse(BaseModel):
    record_id: str
    text: str


class StartRunRequest(BaseModel):
    config: str | dict
    subset: str = "practice-smoke20"
    generate: bool = True
    allow_full: bool = False
    fresh: bool = Field(False, description="skip cache reads so latency and cost are valid")


class JobStatus(BaseModel):
    id: str
    status: Literal["running", "done", "failed"]
    estimate: dict
    run_id: str | None = None
    error: str | None = None
