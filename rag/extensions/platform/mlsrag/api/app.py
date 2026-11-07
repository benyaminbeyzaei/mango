"""The HTTP API in front of mlsrag. Nothing about users is stored; model output is only returned, never executed."""
import hmac
import json
import os
import tempfile
import threading
import uuid
from collections import OrderedDict
from dataclasses import replace
from pathlib import Path
from typing import Annotated

import numpy as np
import openai
import yaml
from fastapi import APIRouter, Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import ValidationError

from .. import evaluate, runner
from ..config import ENCODER_SPACE, PRESETS, RunConfig
from ..course import CourseData, paths
from ..flows import FLOWS, get_flow
from ..gateway import ModelGateway
from ..maps import map_path
from ..matrix import CONFIGS_DIR, FULL_SUBSETS, config_hash
from ..results import list_runs, load_run
from ..stages import CONTEXT_BUILDERS, RETRIEVERS, Translate, WhisperASR, build_stages
from ..subsets import load_subset
from ..trace import Trace
from .models import (ConfigEntry, Hit, JobStatus, ModelsResponse, NamedItem, QueriesPage, QueryRequest,
                     QueryResponse, RecordResponse, ResultsResponse, RunDetail, RunSummary,
                     StartRunRequest, TextInput)

ELM_MODELS = [
    "Qwen/Qwen3.5-397B-A17B-FP8",
    "mistralai/Mistral-Small-4-119B-2603",
    "meta-llama/Llama-3.3-70B-Instruct",
    "utter-project/EuroLLM-22B-Instruct-2512",
]
MAX_CACHED_CONFIGS = 8
VECTOR_DIM = 384


class Fixed:
    """Stands in for the ASR (a typed question) or the encoder (a given vector) so the flow is unchanged."""

    def __init__(self, value):
        self.value = value

    def transcribe(self, audio_path):
        return self.value

    def encode(self, text):
        return self.value


def course_asr_available():
    import torch

    return torch.cuda.is_available()


def named_configs() -> list[tuple[str, str, RunConfig]]:
    """(name, source, config) for every preset and every RunConfig YAML in configs/.

    configs/ also holds matrix and preprocessing files; a file counts as a RunConfig when all its keys are
    RunConfig fields."""
    out = [(name, "preset", make()) for name, make in PRESETS.items()]
    for path in sorted(CONFIGS_DIR.glob("*.yaml")):
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if isinstance(raw, dict) and set(raw) <= set(RunConfig.model_fields):
            out.append((path.stem, f"configs/{path.name}", RunConfig(**raw)))
    return out


def resolve(config) -> RunConfig:
    """A config name from GET /configs or a full RunConfig dict; 400 otherwise. Never a path from the client."""
    if isinstance(config, dict):
        try:
            return RunConfig(**config)
        except ValidationError as err:
            raise HTTPException(400, f"bad config: {err.errors()[0]['msg']}") from err
    if config in PRESETS:
        return PRESETS[config]()
    path = CONFIGS_DIR / f"{config}.yaml"
    if path.parent == CONFIGS_DIR and path.exists():
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        try:
            return RunConfig(**raw)
        except (ValidationError, TypeError) as err:
            raise HTTPException(400, f"{config} is not a RunConfig") from err
    raise HTTPException(400, f"unknown config {config!r}; see GET /api/configs")


def run_dir(root, run_id) -> Path:
    path = (root / run_id).resolve()
    if path.parent != root.resolve() or not (path / "metrics.json").exists():
        raise HTTPException(404, f"no run {run_id!r}")
    return path


def create_app(web_dir=None, results_dir=None) -> FastAPI:
    app = FastAPI(title="mlsrag", version="0.1.0",
                  description="Query the RAG pipeline with a chosen config and read the committed results.")
    api = APIRouter(prefix="/api")
    results_root = Path(results_dir or paths.results_dir)
    admin_token = os.environ.get("ADMIN_TOKEN")
    gateway = ModelGateway()
    stages_cache = OrderedDict()  # config hash -> Stages, oldest first
    lock = threading.Lock()  # one query at a time: the gateway has one current trace
    jobs = {}  # job id -> JobStatus
    data = {}

    def course_data():
        if "data" not in data:
            data["data"] = CourseData(paths.data_dir)
        return data["data"]

    def stages_for(config):
        key = config_hash(config)
        if key not in stages_cache:
            stages_cache[key] = build_stages(config, gateway, course_data())
            if len(stages_cache) > MAX_CACHED_CONFIGS:
                stages_cache.popitem(last=False)
        stages_cache.move_to_end(key)
        return stages_cache[key]

    def answer(config, audio_path="api", asr=None, encoder=None) -> QueryResponse:
        """One query through the config's flow, exactly as `mlsrag run` does it."""
        trace = Trace("api", "api", config.name)
        with lock:
            try:
                stages = stages_for(config)
                stages = replace(stages, asr=asr or stages.asr, encoder=encoder or stages.encoder)
                gateway.trace = trace
                a = get_flow(config.flow)("api", audio_path, stages, config, trace)
            except openai.OpenAIError as err:
                raise HTTPException(502, f"ELM request failed: {type(err).__name__}") from err
            except (ValueError, KeyError, FileNotFoundError, RuntimeError, ImportError) as err:
                raise HTTPException(400, str(err)) from err
            finally:
                gateway.trace = None
        trace.finish()
        records = course_data().records
        hits = [Hit(record_id=rid, score=score, text=records.get(rid, "")[:500])
                for rid, score in zip(a["ranked_ids"][: config.k], a["scores"][: config.k], strict=True)]
        return QueryResponse(answer=a["answer"], transcript=a["transcript"], retrieved=hits,
                             prompt=a["prompt"], trace=trace.to_dict())

    def vector_encoder(config, inp):
        if len(inp.vector) != VECTOR_DIM:
            raise HTTPException(400, f"vector must have {VECTOR_DIM} floats, got {len(inp.vector)}")
        vec = Fixed(np.asarray(inp.vector, dtype=np.float32))
        if inp.space == config.corpus_index or config.mismatch_policy == "as-is":
            return vec
        if config.mismatch_policy == "translate":
            path = map_path(inp.space, config.corpus_index, config.params.get("map", "lstsq"))
            try:
                return Translate(vec, path)
            except FileNotFoundError as err:
                raise HTTPException(400, str(err)) from err
        raise HTTPException(400, f"vector is in {inp.space} but corpus_index is {config.corpus_index}; "
                                 f"mismatch_policy {config.mismatch_policy!r} cannot apply to a vector "
                                 "(use as-is or translate)")

    def require_admin(x_admin_token: str | None = Header(None)):
        if not admin_token:
            raise HTTPException(403, "admin endpoints are disabled (ADMIN_TOKEN is not set)")
        if not x_admin_token or not hmac.compare_digest(x_admin_token, admin_token):
            raise HTTPException(403, "wrong or missing X-Admin-Token")

    @api.get("/health")
    def health():
        return {"status": "ok", "git_commit": runner.git_commit()}

    @api.get("/models", response_model=ModelsResponse)
    def models():
        return ModelsResponse(
            asr=[NamedItem(name="whisper1"), NamedItem(name="course", available=course_asr_available()),
                 NamedItem(name="cached")],
            encoders=[NamedItem(name=n, space=s) for n, s in ENCODER_SPACE.items()],
            indexes=sorted(set(ENCODER_SPACE.values())),
            chat_models=ELM_MODELS,
            retrievers=sorted(RETRIEVERS),
            context_builders=sorted(CONTEXT_BUILDERS),
            flows=sorted(FLOWS),
        )

    @api.get("/configs", response_model=list[ConfigEntry])
    def configs():
        return [ConfigEntry(name=n, source=s, config=c) for n, s, c in named_configs()]

    @api.post("/query", response_model=QueryResponse)
    def query(body: QueryRequest):
        config = resolve(body.config)
        if isinstance(body.input, TextInput):
            return answer(config, asr=Fixed(body.input.text))
        return answer(config, asr=Fixed(body.input.question), encoder=vector_encoder(config, body.input))

    @api.post("/query/audio", response_model=QueryResponse)
    def query_audio(config: Annotated[str, Form(description="config name or a RunConfig as a JSON string")],
                    audio: Annotated[UploadFile, File(description="WAV or WebM/Opus")]):
        try:
            config = resolve(json.loads(config) if config.lstrip().startswith("{") else config)
        except json.JSONDecodeError as err:
            raise HTTPException(400, "config is not valid JSON") from err
        # an uploaded file is never in a transcript cache, so cached configs fall back to whisper-1
        asr = WhisperASR(gateway) if config.asr == "cached" else None
        suffix = Path(audio.filename or "audio.wav").suffix or ".wav"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / f"upload{suffix}"
            path.write_bytes(audio.file.read())
            return answer(config, audio_path=path, asr=asr)

    @api.get("/results", response_model=ResultsResponse)
    def results():
        runs = []
        for d in list_runs(results_root):
            m = load_run(d)["metrics"]
            runs.append(RunSummary(**m["run"], retrieval=m["retrieval"], cost=m["cost"]))
        def summaries(folder):  # list_runs only sees run folders, not these summaries
            return [json.loads(p.read_text(encoding="utf-8"))
                    for p in sorted(results_root.glob(f"{folder}/*/summary.json"))]

        return ResultsResponse(runs=runs, matrices=summaries("matrices"), asr_compare=summaries("asr-compare"))

    @api.get("/results/{run_id}", response_model=RunDetail)
    def result(run_id: str):
        run = load_run(run_dir(results_root, run_id))
        return RunDetail(config=run["config"], metrics=run["metrics"])

    @api.get("/results/{run_id}/queries", response_model=QueriesPage)
    def result_queries(run_id: str, offset: int = 0, limit: int = 20):
        if offset < 0 or not 1 <= limit <= 100:
            raise HTTPException(400, "offset >= 0 and 1 <= limit <= 100")
        lines = (run_dir(results_root, run_id) / "queries.jsonl").read_text(encoding="utf-8").splitlines()
        return QueriesPage(total=len(lines), offset=offset,
                           items=[json.loads(line) for line in lines[offset: offset + limit]])

    @api.get("/records/{record_id}", response_model=RecordResponse)
    def record(record_id: str):
        records = course_data().records
        if record_id not in records:
            raise HTTPException(404, f"no record {record_id!r}")
        return RecordResponse(record_id=record_id, text=records[record_id])

    @api.post("/runs", response_model=JobStatus, dependencies=[Depends(require_admin)])
    def start_run(body: StartRunRequest):
        config = resolve(body.config)
        if body.subset in FULL_SUBSETS and not body.allow_full:
            raise HTTPException(400, f"{body.subset} is a full split; set allow_full: true to run it")
        try:
            load_subset(body.subset)
        except ValueError as err:
            raise HTTPException(400, str(err)) from err
        estimate = evaluate.estimate(config, body.subset)
        if not body.generate:  # estimate() always counts generation
            estimate["elm_calls"] -= estimate["calls"]["generation"]
            estimate["calls"]["generation"] = 0
        job = JobStatus(id=uuid.uuid4().hex[:12], status="running", estimate=estimate)
        jobs[job.id] = job

        def work():
            try:
                out, _ = runner.run(config, body.subset, generate=body.generate, results_dir=results_root)
                job.run_id, job.status = out.name, "done"
            except Exception as err:  # reported through GET /runs/{id}/status
                job.error, job.status = f"{type(err).__name__}: {err}", "failed"

        threading.Thread(target=work, daemon=True).start()
        return job

    @api.get("/runs/{job_id}/status", response_model=JobStatus, dependencies=[Depends(require_admin)])
    def run_status(job_id: str):
        if job_id not in jobs:
            raise HTTPException(404, f"no job {job_id!r}")
        return jobs[job_id]

    app.include_router(api)

    if web_dir:
        web_root = Path(web_dir).resolve()

        @app.get("/{path:path}", include_in_schema=False)
        def web(path: str):
            if path == "api" or path.startswith("api/"):
                raise HTTPException(404, "not found")
            file = (web_root / path).resolve()
            if file.is_file() and file.is_relative_to(web_root):
                return FileResponse(file)
            return FileResponse(web_root / "index.html")  # SPA fallback: the client routes the rest

    return app
