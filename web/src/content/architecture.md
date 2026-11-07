# Architecture

## The pipeline

Each query goes through five stages, the course's standard flow:

1. **ASR** turns the audio question into a transcript (the course CUDA model, whisper-1, or a cached transcript).
2. **Encoder** turns the transcript into a 384-d vector (gte-small for the enc_a index, bge-small for enc_b).
3. **Retriever** ranks the 24,000 corpus records by exact dense similarity and keeps the top 100.
4. **Context builder** puts the top k records (5 by default) into a prompt as evidence.
5. **Generator** asks an ELM chat model (Qwen by default) for an answer grounded in that evidence.

## Where the platform sits

The course code in `rag/src/` is used unchanged. `mlsrag` (in `rag/extensions/platform/`) wraps it: a run configuration picks one implementation per stage, every model call goes through one cached, logged and costed gateway, and every query leaves a trace of spans, counts and costs. A run over a query subset writes `config.json`, `queries.jsonl` and `metrics.json` under `rag/extensions/results/`. The HTTP API (`mlsrag serve`, routes under `/api`) exposes those runs, the available models and single queries; this web client only talks to that API.

## The modes

- **Docs, Models, Guide**: reference pages; Models shows what the running API reports.
- **Results**: browse saved runs and compare their retrieval quality, answer scores and cost.
- **Interactive**: ask one question by text or audio and see every stage's output and timing.

## What is and is not stored

- Stored: run results written by `mlsrag run` and the matrix and comparison summaries; the gateway's disk cache of model calls and its call log; transcript caches.
- Not stored: anything about users. An interactive query is answered and returned, not saved; an uploaded audio file lives in a temporary folder only for the request.
- In the browser: only the theme and, if you set one, the admin token (`localStorage`).
