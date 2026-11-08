# HTTP API (`mlsrag serve`)

Part of the [mlsrag README](../README.md).

```bash
mlsrag serve --host 127.0.0.1 --port 8000 [--web-dir web/dist]   # API under /api, the client (if given) at /
mlsrag openapi                                                    # rewrite openapi.json (a test checks it is current)
```

`openapi.json` (in `rag/extensions/platform/`) is the contract the web client is built against. Every query
goes through the same `build_stages` and flow as `mlsrag run`, with one trace per request; stages are
built once per config (at most 8 kept) and queries run one at a time. Typed text replaces the ASR
stage's output, a given vector replaces the encoder's.

| Method and path | What it does |
|---|---|
| `GET /api/health` | `{status, git_commit}` |
| `GET /api/models` | ASR (`course` is `available: false` without a GPU), encoders and their space, indexes, the four free ELM models, retrievers and context builders (`stages.RETRIEVERS`, `stages.CONTEXT_BUILDERS`) and flows (`flows.FLOWS`) |
| `GET /api/configs` | presets and the RunConfig YAMLs in `configs/` (top level; pipelines are in `configs/pipelines/`, matrices in `configs/matrices/`), with all fields |
| `POST /api/query` | JSON `{config: <name or RunConfig>, input: {text} \| {vector, space, question?}}` → `{answer, transcript, retrieved: [{record_id, score, text[:500]}], prompt, trace}` |
| `POST /api/query/audio` | `multipart/form-data`: `config` (name or RunConfig JSON), `audio` (WAV or WebM/Opus, passed to whisper-1 unconverted). A `cached` ASR config uses whisper-1, since an upload is never in a transcript cache. Non-WAV audio is costed as 0 s |
| `GET /api/results` | committed runs `{run_id, config_name, subset, n_queries, finished, fresh}`, matrix summaries (`results/matrices/`) and ASR comparisons (`results/asr-compare/`) |
| `GET /api/results/{run_id}` | its `config.json` and `metrics.json` |
| `GET /api/results/{run_id}/queries?offset&limit` | a page of `queries.jsonl` traces (limit at most 100) |
| `GET /api/records/{record_id}` | the full record text |
| `POST /api/runs` (admin) | `{config, subset, generate, allow_full, fresh}`: starts `mlsrag run` in a background thread, returns a job id and the call estimate; full splits are refused unless `allow_full: true` |
| `GET /api/runs/{id}/status` (admin) | `running`, `done` (with `run_id`) or `failed` (with `error`) |

Errors: 400 for an unknown or invalid config, a vector of the wrong size, or a vector in the wrong
space whose `mismatch_policy` cannot apply (`as-is` and `translate` can; `re-encode` needs text);
502 when an ELM request fails.

**Admin.** Set `ADMIN_TOKEN` in the environment and send it as the `X-Admin-Token` header. Without
`ADMIN_TOKEN` the admin endpoints always return 403. Jobs live in memory only.

**Privacy and safety.** Nothing about users is stored: no accounts, no logs of who asked what
(uploads go to a temp folder and are deleted after transcription). Model calls are still cached and
logged by the gateway as for any run, under `rag/.cache/`. Model output is returned as text and
never executed.
