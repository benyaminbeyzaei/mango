# Hand-over (2026-11-08)

## What exists

- `mlsrag` (this folder): RunConfig, Trace, gateway, five stage interfaces, three flows, eight
  retrievers, four context builders, preprocessing pipelines, judging through the course evaluator,
  matrices, an HTTP API. Start at [README.md](README.md).
- `web/`: the client (Docs, Models, Results, Interactive, Guide), served by `mlsrag serve` or Docker.
- `compose.yaml` + `Dockerfile`: `docker compose up` from a clean clone (sequence in the README,
  checked on 2026-11-08: about 6 minutes including the image build).
- `rag/extensions/results/`: 26 runs, all fresh (`run.fresh: true`), and four matrix summaries:
  `a-smoke` (11 retrievers, retrieval only), `context-smoke` (judged baseline, budgeted, dedupe),
  `encoders-smoke` (encoder policies), `toc-raptor-smoke`; plus six runs on the 600-record dev subset.

## What each plan produced

- 001: the platform skeleton: config, trace, gateway, stages, dense flow, runner.
- 002: preprocessing pipelines (hashed stages, ledger, selection) and the topic routing retriever.
- 003: lexical retrieval: BM25, grep static, grep agentic.
- 004: evaluation: course judge, `metrics.json` schema, `estimate`, `matrix`, `compare`.
- 005: encoder mismatch policies (re-encode, translate with fitted maps) and the ASR comparison harness.
- 006: the FastAPI server and its OpenAPI contract.
- 007: two-level TOC and RAPTOR-lite retrievers.
- 008: lazy graph retriever, LLM knowledge-graph code, `record_subset` restriction.
- 009: web client skeleton (routes, docs, models, guide).
- 010: Docker image and compose for API, client and CLI.
- 011: web Results mode (runs, comparison charts, run and query detail).
- 012: web Interactive mode (text and voice, config picker, trace viewer).
- 013: budgeted and dedupe context builders; experiment matrices A, B, C.
- 014: this integration: clean-clone check, docs split into `docs/`, evaluator guide, stale runs
  replaced by fresh ones, matrix override names, KG budget clamp, API bound to localhost.
- 015: one full-split guard for `run`, `transcribe`, `matrix` and the API.
- 016: simplicity pass (trace argument, plain gateway cache and retry, config folders).
- 017: measurement fixes (local embeddings are not remote calls, BM25 skips encode).
- 018: KG extraction with Llama-3.3 (598 records) and the KG rows on the 600-record subset.
- 019: `--fresh`: latency and cost are valid only for fresh runs.

## Open items

- **Full runs to schedule** (each needs Benyamin's go-ahead; numbers from `mlsrag estimate`,
  wall time at one request at a time, judging not included):
  - fill the practice transcript cache for practice-20pct: 50 whisper-1 calls, ~$0.03;
  - matrix A on practice-20pct: 490 ELM calls, ~7.8 h; on dev-20pct: 175 calls, ~2.8 h;
  - matrix B (simple and budgeted) on dev-20pct: 275 + 175 judge calls each, ~4.5 h + judging each;
  - matrix C on dev-20pct: 0 ELM calls;
  - `dev-all` costs five times the dev-20pct numbers (see the Guide page for the full-split table).
- **Course ASR transcripts** from the teammate (the CUDA model on the cluster): needed for
  `mlsrag asr-compare` (no comparison is committed yet) and for matrix C's ASR row. Put them in
  `rag/.cache/transcripts/course-<split>.jsonl` (`{query_id, transcript}` per line).
- **The report** (`report/`): Benyamin writes the argument; the matrices above are its evidence.
- A run that crashes leaves a folder without `metrics.json`; the listings ignore it, delete it by hand.
