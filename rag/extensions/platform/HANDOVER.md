# Hand-over (2026-11-09)

## What exists

- `mlsrag` (this folder): RunConfig, Trace, gateway, five stage interfaces, three flows, six
  retrievers, four context builders, preprocessing pipelines, judging through the course evaluator,
  matrices, an HTTP API. Start at [README.md](README.md).
- `web/`: the client (Docs, Models, Results, Interactive, Guide), served by `mlsrag serve` or Docker.
- `compose.yaml` + `Dockerfile`: `docker compose up` from a clean clone (sequence in the README,
  checked on 2026-11-08: about 6 minutes including the image build).
- `rag/extensions/results/`: 36 smoke runs, all fresh (`run.fresh: true`), and four matrix summaries:
  `retrievers-practice-20` (20 retrievers, retrieval only), `context-practice-20` (judged baseline, budgeted, dedupe),
  `encoders-practice-20` (encoder policies), `toc-raptor-practice-20`. Plan 029 added four full-split matrix
  summaries for the no-LLM methods (`retrievers-practice`, `retrievers-test`, `encoders-practice`,
  `encoders-test`, with 95 % bootstrap intervals); their run folders are local only, not committed.
  Plan 032 added `rerank-practice-20`, `rerank-practice`, `final-practice-20`, `final-practice`, `final-test`
  and `final/comparison.json`; the 10 smoke runs are committed whole, the full-split runs as `metrics.json`,
  `config.json`, `per_query.jsonl` (plus judge scores for test), never prompts or answers, so
  `python -m mlsrag.final` rebuilds `comparison.json` from a clean clone.

## What each plan produced

- 001: the platform skeleton: config, trace, gateway, stages, dense flow, runner.
- 002: preprocessing pipelines (hashed stages, ledger, selection) and the topic routing retriever.
- 003: lexical retrieval: BM25, grep static, grep agentic.
- 004: evaluation: course judge, `metrics.json` schema, `estimate`, `matrix`, `compare`.
- 005: encoder mismatch policies (re-encode, translate with fitted maps) and the ASR comparison harness.
- 006: the FastAPI server and its OpenAPI contract.
- 007: two-level TOC and RAPTOR-lite retrievers.
- 008: two graph retrievers (removed by 026).
- 009: web client skeleton (routes, docs, models, guide).
- 010: Docker image and compose for API, client and CLI.
- 011: web Results mode (runs, comparison charts, run and query detail).
- 012: web Interactive mode (text and voice, config picker, trace viewer).
- 013: budgeted and dedupe context builders; experiment matrices A, B, C.
- 014: this integration: clean-clone check, docs split into `docs/`, evaluator guide, stale runs
  replaced by fresh ones, matrix override names, API bound to localhost.
- 015: one full-split guard for `run`, `transcribe`, `matrix` and the API.
- 016: simplicity pass (trace argument, plain gateway cache and retry, config folders).
- 017: measurement fixes (local embeddings are not remote calls, BM25 skips encode).
- 018: runs for the 008 graph retrievers (removed by 026).
- 019: `--fresh`: latency and cost are valid only for fresh runs.
- 020: second simplicity pass (stage folders, one ledger loop, `estimate` generate flag, static config picker).
- 021: grep pattern calls on Llama-3.3 (Qwen's reasoning took ~70-100 s per call); answers stay on Qwen.
- 022: `seconds`/`audio_s` in new transcript cache lines, `transcribe --fresh` for whisper-1, ASR latency and RTF in `asr-compare`; whisper-1 measured on practice-20.
- 023: the grep-feedback flow (static grep, keep, one expansion from the kept records, rerank); matrix A and B rows.
- 024: stale cost numbers fixed, `pnpm typecheck` checks the app, small simplifications, and the
  candidates file lists the ids the answer's context was built from.
- 026: the two graph retrievers removed with their configs, runs, dev record subset and dependencies.
- 027: TOC routes `llm-topics` (section prompt lists topic names) and `llm-flat` (one call over all
  256 topics) as new matrix A rows; `llm` kept.
- 028: `raptor-tree` (soft membership, whole-cluster summaries, 16 section summaries, one collapsed
  index; 439 Llama calls, 50 min build) as a new matrix A row; `raptor-lite` kept unchanged.
- 029: whisper-1 transcripts for all of practice and dev (430 calls, ~$0.23); 95 % bootstrap
  intervals for Recall@10 and NDCG@10 in matrix summaries; matrices A (10 no-LLM retrievers) and C
  on practice and test, fresh, summaries committed (results in the README).
- 030: gpt-5-mini (paid, named for these two builds only) versions of the TOC labels (label prompt
  `v2`: nearest records, distinct names) and of the RAPTOR-tree summaries, next to the Llama builds;
  smoke rows for the four TOC routes and both RAPTOR-tree modes, full-split rows for RAPTOR-tree;
  US$1.59 for 711 build calls.
- 031: `raptor-full`, a faithful RAPTOR (reference code ported: UMAP + GMM/BIC soft clustering,
  global/local, 3,500-token re-clustering, 5 layers, the paper's prompt, `max_tokens` 100) on the
  24,000 records with gpt-4o-mini: 5,243 summaries (4,547 / 567 / 107 / 18 / 4), $1.11, 4.1 h build;
  matrix A rows on smoke and both full splits (below the baseline; README). Each copied value cites
  its reference file:line in `preprocess/raptor_full.py`; deviations in `docs/pipelines.md`.
- 032: the final three-way comparison. `rerank` retriever (dense top N re-scored by a local cross-encoder,
  `gateway.rerank`), chosen on practice among five settings (`rerank-best` = MiniLM-L-6, pool 20);
  baseline, rerank-best and raptor-tree-gpt5-evidence on practice (retrieval) and test (375 Qwen
  answers, judged). Neither change improves answers (README). `results/matrices/final/comparison.json`
  holds every number plus `explanations.json`; `python -m mlsrag.final` rebuilds it. The gateway now never
  rewrites an existing cache file (`--fresh` only adds files).

## Open items

- **Full runs to schedule** (each needs Benyamin's go-ahead; numbers from `mlsrag estimate`,
  wall time at one request at a time, judging not included):
  - (done by 029: both transcript caches are full, and every no-LLM retriever has run on
    practice and test; what remains are the LLM methods below);
  - matrix A on practice-70: 980 ELM calls, ~24 min; on test-25: 350 calls, ~9 min;
  - matrix B (simple and budgeted) on test-25: 375 + 175 judge calls each, ~2.6 h + judging each;
  - `test` costs five times the test-25 numbers (see the Guide page for the full-split table).
- **Course ASR transcripts** from the teammate (the CUDA model on the cluster): needed for
  `mlsrag asr-compare` (no comparison is committed yet) and for matrix C's ASR row. Put them in
  `rag/.cache/transcripts/course-<split>.jsonl` (`{query_id, transcript}` per line).
- **Course ASR latency** (plan 022): run `mlsrag transcribe --asr course` on practice-20 on a
  cluster GPU, then `asr-compare` it against `whisper1-practice-timed.jsonl`; exact commands in
  [docs/retrievers.md, ASR latency](docs/retrievers.md#asr-latency). Nothing has been run on the cluster yet.
- **The report** (`report/`): Benyamin writes the argument; the matrices above are its evidence.
- A run that crashes leaves a folder without `metrics.json`; the listings ignore it, delete it by hand.
