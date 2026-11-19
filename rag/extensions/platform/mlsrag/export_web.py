"""Writes every API response the website reads as a JSON file, so the built site runs with no server.

The site's static build (VITE_STATIC=1) maps each request to the same path (web/src/api/client.ts, staticPath).
"""
import json
import shutil
import tempfile
from pathlib import Path

from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.testclient import TestClient

from .api.app import create_app
from .course import paths


# The cached questions of the static site: the report's three methods, each a full judged run on the 125 test
# questions whose complete traces (transcript, evidence, prompt, answer) are committed.
SHOWCASE_RUNS = [
    "20261114-1448-baseline-courseasr-test",
    "20261116-2358-rerank-bge-m3-p20-test",
    "20261114-1721-raptor-tree-gpt5-evidence-courseasr-test",
]


def static_path(path: str, query: dict | None = None) -> str:
    """/api/results/R/queries?offset=100&limit=100 -> /static-api/results/R/queries/limit-100_offset-100.json"""
    pairs = "_".join(f"{k}-{v}" for k, v in sorted((query or {}).items()))
    return f"/static-api{path.removeprefix('/api')}{'/' + pairs if pairs else ''}.json"


def export(out: Path, transcripts: Path | None = None):
    transcripts = transcripts or paths.rag_dir / "extensions" / "transcripts"
    with tempfile.TemporaryDirectory() as cache:
        shutil.copytree(transcripts, Path(cache) / "transcripts")
        client = TestClient(create_app(cache_dir=cache))
        _export(client, out.resolve())


def _export(client, out: Path):
    def get(path, missing_ok=False, **query):
        res = client.get(path, params=query)
        if res.status_code == 404 and missing_ok:
            return None
        if res.status_code != 200:
            raise SystemExit(f"GET {path} {query}: HTTP {res.status_code} {res.text[:200]}")
        file = out / static_path(path, query).lstrip("/")
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(res.content)
        return res.json()

    for name in ["health", "models", "configs"]:
        get(f"/api/{name}")
    record_ids, cached = set(), []
    for run in get("/api/results")["runs"]:
        run_id = run["run_id"]
        detail = get(f"/api/results/{run_id}")
        k = int(detail["config"]["k"])
        items, offset, total = [], 0, 1
        while offset < total:
            page = get(f"/api/results/{run_id}/queries", offset=offset, limit=100)
            items += page["items"]
            total, offset = page["total"], offset + 100
        for q in items:
            record_ids.update(q["artifacts"]["ranked_ids"][:k])
        if run_id in SHOWCASE_RUNS:
            if not all(q["artifacts"].get("answer") for q in items):
                raise SystemExit(f"{run_id} is a showcase run but has queries without an answer")
            m = detail["metrics"]
            cached.append({"run_id": run_id, "config_name": run["config_name"], "subset": run["subset"], "k": k,
                           "transcripts": "our ASR" if "/course-" in (detail["config"].get("asr_cache") or "")
                           else "whisper-1",
                           "ndcg10": m["retrieval"]["NDCG@10"],
                           "answer_score": (m.get("answers") or {}).get("normalized"),
                           "questions": [{"query_id": q["query_id"], "transcript": q["artifacts"]["transcript"]}
                                         for q in items]})
    # RAPTOR summary nodes (summary:...) are ranked like records but are not in the corpus: the live API 404s too
    missing = sum(get(f"/api/records/{rid}", missing_ok=True) is None for rid in sorted(record_ids))

    query_ids = set()
    for split in ["test", "practice"]:
        for match in ["all", "differ", "same"]:
            offset, total = 0, 1
            while offset < total or offset == 0:
                page = get("/api/questions", split=split, match=match, offset=offset, limit=20)
                query_ids.update(q["query_id"] for q in page["items"])
                total, offset = page["total"], offset + 20
    manifest = [json.loads(line) for line in
                (paths.data_dir / "student_audio_manifest.jsonl").read_text(encoding="utf-8").splitlines() if line]
    for entry in manifest:
        if entry["query_id"] in query_ids:
            dest = out / "static-api" / "questions" / entry["query_id"] / "audio.wav"
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(paths.data_dir / entry["audio_file"], dest)

    if len(cached) != len(SHOWCASE_RUNS):
        raise SystemExit(f"showcase runs missing from the results: {set(SHOWCASE_RUNS) - {c['run_id'] for c in cached}}")
    cached.sort(key=lambda c: SHOWCASE_RUNS.index(c["run_id"]))
    (out / "static-api" / "cached-qa.json").write_text(json.dumps(cached), encoding="utf-8")

    # The API reference, read-only: Swagger UI on the exported OpenAPI document, with "Try it out" switched off
    (out / "static-api" / "openapi.json").write_bytes(client.get("/api-docs/openapi.json").content)
    html = get_swagger_ui_html(openapi_url="/static-api/openapi.json", title="Mango API",
                               swagger_favicon_url="/favicon.ico",
                               swagger_ui_parameters={"supportedSubmitMethods": []})
    (out / "api-docs.html").write_bytes(html.body)

    files = [p for p in (out / "static-api").rglob("*") if p.is_file()]
    size = sum(p.stat().st_size for p in files)
    print(f"wrote {len(files)} files, {size / 1e6:.1f} MB, under {out / 'static-api'} "
          f"({len(record_ids) - missing} records, {missing} ids not in the corpus, {len(cached)} cached runs)")
