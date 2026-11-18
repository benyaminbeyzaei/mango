# Architecture

## The pipeline

Each query goes through five stages, the course's standard flow. The course gives one implementation of each; the platform adds alternatives, chosen per run in the config:

1. **ASR** turns the audio question into a transcript (the course CUDA model, whisper-1, or a cached transcript).
2. **Encoder** turns the transcript into a 384-d vector (gte-small for the enc_a index, bge-small for enc_b).
3. **Retriever** ranks the 24,000 corpus records and keeps the top 100. The course's is exact dense similarity (`exact`); the alternatives are topic routing (`topic`), table-of-contents routing (`toc`), summary trees (`raptor-lite`, `raptor-tree`, `raptor-full`), keyword search (`bm25`), LLM-written grep patterns (`grep`) and a cross-encoder reranker over the dense top results (`rerank`).
4. **Context builder** puts the top k records (5 by default) into a prompt as evidence: `simple` (the course's), `budgeted` (a token budget), `dedupe` or `with-summaries` (adds tree summaries).
5. **Generator** asks an ELM chat model (Qwen by default) for an answer grounded in that evidence.

## Where the platform sits

The course code in `rag/src/` is used unchanged. `mlsrag` (in `rag/extensions/platform/`) wraps it: a run configuration picks one implementation per stage, every model call goes through one cached, logged and costed gateway, and every query leaves a trace of spans, counts and costs. A run over a question set writes `config.json`, `queries.jsonl` and `metrics.json` under `rag/extensions/results/`. The HTTP API (`mlsrag serve`, routes under `/api`) exposes those runs, the available models and single queries; this web client only talks to that API.

## The modes

- **Docs, Models, Guide**: reference pages; Models shows what the running API reports.
- **Results**: browse saved runs (20-question quick checks hidden unless you pick `practice-20` in the filter) and compare their retrieval quality, answer scores and cost.
- **Questions**: listen to each benchmark question and compare the whisper-1 transcript with our own ASR's, word by word.
- **Interactive**: ask one question by text or audio and see every stage's output and timing.

## What is and is not stored

- Stored: run results written by `mlsrag run` and the matrix and comparison summaries; the gateway's disk cache of model calls and its call log; transcript caches.
- Not stored: anything about users. An interactive query is answered and returned, not saved; an uploaded audio file lives in a temporary folder only for the request.
- In the browser only: the chat history of the Interactive page (IndexedDB), and in `localStorage` the theme, the Interactive settings (default config, confirm-transcript) and, if you set one, the admin token.
