"""The API with fake stages and data: no network, no models, no course data."""
import json
import shutil
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from mlsrag import evaluate, runner
from mlsrag.api import app as app_module
from mlsrag.cli import OPENAPI_PATH, openapi_json
from mlsrag.course import paths
from mlsrag.stages import CONTEXT_BUILDERS, RETRIEVERS, Stages

RECORDS = {f"r{i}": f"text of record {i} " * 40 for i in range(100)}


class FakeStage:
    def __init__(self):
        self.vectors = []

    def transcribe(self, audio_path):
        return "cached transcript"

    def encode(self, text):
        return np.ones(384, dtype=np.float32)

    def search(self, vec, k, trace=None):
        self.vectors.append(vec)
        return [(f"r{i}", 1.0 - i / 1000) for i in range(k)]

    def build(self, question, record_ids):
        return f"{question} | {','.join(record_ids)}"

    def generate(self, prompt):
        return "the answer"


class FakeData:
    records = RECORDS


class FakeGateway:
    trace = None

    def transcribe(self, audio_path, model="whisper-1"):
        assert audio_path.read_bytes() == b"RIFF fake wav"
        return "spoken question"


@pytest.fixture
def stage(monkeypatch, tmp_path):
    fake = FakeStage()
    built = []

    def build(config, gateway, data):
        built.append(config.name)
        return Stages(asr=fake, encoder=fake, retriever=fake, context=fake, generator=fake)

    monkeypatch.setattr(app_module, "build_stages", build)
    monkeypatch.setattr(app_module, "CourseData", lambda _: FakeData())
    monkeypatch.setattr(app_module, "ModelGateway", FakeGateway)
    monkeypatch.setattr(app_module, "course_asr_available", lambda: False)
    fake.built = built
    return fake


@pytest.fixture
def client(stage, tmp_path):
    src = sorted(paths.results_dir.glob("*-baseline-practice-smoke20"))[0]
    shutil.copytree(src, tmp_path / src.name)
    (tmp_path / "matrices" / "m1").mkdir(parents=True)
    (tmp_path / "matrices" / "m1" / "summary.json").write_text('{"name": "m1", "subset": "s", "rows": []}')
    (tmp_path / "asr-compare" / "a-vs-b").mkdir(parents=True)
    (tmp_path / "asr-compare" / "a-vs-b" / "summary.json").write_text('{"wer": 0.1}')
    return TestClient(app_module.create_app(results_dir=tmp_path))


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200 and r.json()["status"] == "ok" and "git_commit" in r.json()


def test_models_and_configs_come_from_the_registries(client):
    m = client.get("/api/models").json()
    assert {"name": "course", "available": False, "space": None} in m["asr"]
    assert [e["name"] for e in m["encoders"]] == ["gte-small", "bge-small"]
    assert m["indexes"] == ["enc_a", "enc_b"] and len(m["chat_models"]) == 4
    assert m["retrievers"] == sorted(RETRIEVERS) and m["context_builders"] == sorted(CONTEXT_BUILDERS)
    assert m["flows"] == sorted(app_module.FLOWS)
    names = [c["name"] for c in client.get("/api/configs").json()]
    assert "baseline" in names and "enc-b-translate-lstsq" in names
    assert "topics-k256-m4" in names  # a RunConfig using plan 002's topic retriever
    assert not any(n.startswith("matrix-") for n in names) and "topics-k256" not in names  # matrix / preprocessing


def test_text_query_runs_the_flow_and_returns_the_trace(client, stage):
    r = client.post("/api/query", json={"config": "baseline", "input": {"text": "what is x"}})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["answer"] == "the answer" and body["transcript"] == "what is x"
    assert [h["record_id"] for h in body["retrieved"]] == ["r0", "r1", "r2", "r3", "r4"]
    assert len(body["retrieved"][0]["text"]) == 500
    assert body["prompt"].startswith("what is x |")
    assert [s["name"] for s in body["trace"]["spans"]] == ["asr", "encode", "retrieve", "context", "generate"]
    assert len(body["trace"]["artifacts"]["ranked_ids"]) == 100
    client.post("/api/query", json={"config": "baseline", "input": {"text": "again"}})
    assert stage.built == ["baseline"]  # stages are built once per config


def test_vector_query_uses_the_given_vector(client, stage):
    vec = [0.5] * 384
    r = client.post("/api/query", json={"config": "baseline",
                                        "input": {"vector": vec, "space": "enc_a", "question": "q"}})
    assert r.status_code == 200, r.text
    assert np.allclose(stage.vectors[-1], 0.5) and r.json()["transcript"] == "q"


def test_vector_query_errors(client):
    def post(config, space, n=384):
        return client.post("/api/query", json={"config": config, "input": {"vector": [0.1] * n, "space": space}})

    assert post("baseline", "enc_a", n=3).status_code == 400
    assert post("baseline", "enc_b").status_code == 400  # policy error
    assert post("enc-b-reencode", "enc_a").status_code == 400  # a vector cannot be re-encoded
    assert post("enc-b-asis", "enc_a").status_code == 200


def test_vector_without_question_is_400_for_a_question_retriever(client, stage):
    stage.uses_question = True  # like BM25: it searches the question text
    body = {"config": "baseline", "input": {"vector": [0.1] * 384, "space": "enc_a"}}
    r = client.post("/api/query", json=body)
    assert r.status_code == 400 and "question" in r.json()["detail"]


def test_bad_config_is_400(client):
    assert client.post("/api/query", json={"config": "nope", "input": {"text": "x"}}).status_code == 400
    assert client.post("/api/query", json={"config": "../configs/enc-b-asis", "input": {"text": "x"}}).status_code == 400
    bad = {"name": "x", "query_encoder": "bge-small", "corpus_index": "enc_a"}
    r = client.post("/api/query", json={"config": bad, "input": {"text": "x"}})
    assert r.status_code == 400 and "mismatch_policy" in r.json()["detail"]


def test_elm_failure_is_502(client, stage, monkeypatch):
    import httpx2
    import openai

    def fail(prompt):
        raise openai.APIConnectionError(request=httpx2.Request("POST", "http://x"))

    monkeypatch.setattr(stage, "generate", fail)
    assert client.post("/api/query", json={"config": "baseline", "input": {"text": "x"}}).status_code == 502


def test_audio_query_with_a_cached_config_uses_whisper(client):
    r = client.post("/api/query/audio", data={"config": "baseline"},
                    files={"audio": ("clip.wav", b"RIFF fake wav", "audio/wav")})
    assert r.status_code == 200, r.text
    assert r.json()["transcript"] == "spoken question"


def test_results_endpoints(client):
    listing = client.get("/api/results").json()
    run_id = listing["runs"][0]["run_id"]
    assert listing["runs"][0]["subset"] == "practice-smoke20" and listing["matrices"][0]["name"] == "m1"
    assert listing["asr_compare"] == [{"wer": 0.1}]
    metrics = json.loads((paths.results_dir / run_id / "metrics.json").read_text())
    assert listing["runs"][0]["retrieval"] == metrics["retrieval"] and listing["runs"][0]["cost"] == metrics["cost"]
    detail = client.get(f"/api/results/{run_id}").json()
    assert detail["metrics"]["run"]["run_id"] == run_id and detail["config"]["name"] == "baseline"
    page = client.get(f"/api/results/{run_id}/queries", params={"offset": 18, "limit": 5}).json()
    assert page["total"] == 20 and len(page["items"]) == 2 and "spans" in page["items"][0]
    assert client.get("/api/results/nope").status_code == 404
    assert client.get("/api/results/..").status_code == 404
    assert client.get("/api/records/r3").json()["text"] == RECORDS["r3"]
    assert client.get("/api/records/zz").status_code == 404


def test_admin_disabled_without_token(client):
    assert client.post("/api/runs", json={"config": "baseline"}).status_code == 403


def test_admin_runs(stage, tmp_path, monkeypatch):
    monkeypatch.setenv("ADMIN_TOKEN", "s3cret")
    monkeypatch.setattr(evaluate, "estimate", lambda config, subset: {"calls": {"generation": 20}, "elm_calls": 20})
    calls = []

    def fake_run(config, subset, generate, results_dir, allow_full, fresh):
        calls.append((config.name, subset, generate, fresh))
        return tmp_path / "20261107-0000-baseline-practice-smoke20", {}

    monkeypatch.setattr(runner, "run", fake_run)
    client = TestClient(app_module.create_app(results_dir=tmp_path))
    body = {"config": "baseline", "subset": "practice-smoke20", "generate": False, "fresh": True}
    assert client.post("/api/runs", json=body).status_code == 403
    assert client.post("/api/runs", json=body, headers={"X-Admin-Token": "wrong"}).status_code == 403
    admin = {"X-Admin-Token": "s3cret"}
    full = client.post("/api/runs", json={"config": "baseline", "subset": "practice-all"}, headers=admin)
    assert full.status_code == 400 and "full split" in full.json()["detail"]
    job = client.post("/api/runs", json=body, headers=admin).json()
    assert job["estimate"] == {"calls": {"generation": 0}, "elm_calls": 0}  # generate: false
    for _ in range(50):
        status = client.get(f"/api/runs/{job['id']}/status", headers=admin).json()
        if status["status"] != "running":
            break
        time.sleep(0.02)
    assert status["status"] == "done" and status["run_id"] == "20261107-0000-baseline-practice-smoke20"
    assert calls == [("baseline", "practice-smoke20", False, True)]


def test_web_dir_serves_the_client_with_spa_fallback(stage, tmp_path):
    (tmp_path / "index.html").write_text("<html>app</html>")
    (tmp_path / "app.js").write_text("console.log(1)")
    client = TestClient(app_module.create_app(web_dir=tmp_path, results_dir=tmp_path))
    assert client.get("/").text == "<html>app</html>"
    assert client.get("/app.js").text == "console.log(1)"
    assert client.get("/runs/some/page").text == "<html>app</html>"
    assert client.get("/api/health").json()["status"] == "ok"
    assert client.get("/api/nope").status_code == 404


def test_openapi_json_is_current():
    assert json.loads(OPENAPI_PATH.read_text(encoding="utf-8")) == json.loads(openapi_json(app_module.create_app()))
