// Reads of the committed results (/api/results...) and helpers to turn metrics.json into numbers.
import { useQueries, useQuery } from "@tanstack/react-query"

import { api } from "@/api/client"

export type Metrics = {
  retrieval: Record<string, number>
  answers: Record<string, number | string> | null
  cost: Record<string, number>
  latency_ms: Record<string, number | null | Record<string, number>>
  run: {
    run_id: string
    config_name: string
    subset: string
    n_queries: number
    started?: string
    fresh?: boolean
  }
}
export type Span = {
  name: string
  start_ms: number
  duration_ms: number
  attrs: Record<string, unknown>
}
// One line of queries.jsonl. Full-split runs commit only a slim per_query.jsonl; the API turns
// its lines into this shape with `slim: true` (no transcript, prompt, answer or gateway spans).
export type QueryTrace = {
  query_id: string
  slim?: boolean
  spans: Span[]
  counters: Record<string, number>
  cost: Record<string, unknown>
  resources: { wall_ms?: number }
  artifacts: {
    transcript: string
    ranked_ids: string[]
    scores: number[]
    prompt: string
    answer: string | null
  }
  metrics: Record<string, number>
  judge?: Record<string, unknown>
}

export function useResults() {
  return useQuery({
    queryKey: ["results"],
    queryFn: async () => {
      const { data, error } = await api.GET("/api/results")
      if (error) throw new Error("could not load /api/results")
      return data
    },
  })
}

async function fetchRun(runId: string) {
  const { data, error } = await api.GET("/api/results/{run_id}", {
    params: { path: { run_id: runId } },
  })
  if (error) throw new Error(`could not load run ${runId}`)
  return data as { config: Record<string, unknown>; metrics: Metrics }
}

// Every query of a run; the API pages them 100 at a time.
async function fetchQueries(runId: string) {
  const items: QueryTrace[] = []
  for (let offset = 0; ; offset += 100) {
    const { data, error } = await api.GET("/api/results/{run_id}/queries", {
      params: { path: { run_id: runId }, query: { offset, limit: 100 } },
    })
    if (error) throw new Error(`could not load queries of ${runId}`)
    items.push(...(data.items as QueryTrace[]))
    if (items.length >= data.total) return items
  }
}

export const useRun = (runId: string) =>
  useQuery({ queryKey: ["run", runId], queryFn: () => fetchRun(runId) })

export const useRunQueries = (runId: string) =>
  useQuery({ queryKey: ["queries", runId], queryFn: () => fetchQueries(runId) })

export const useRuns = (runIds: string[]) =>
  useQueries({
    queries: runIds.map((id) => ({
      queryKey: ["run", id],
      queryFn: () => fetchRun(id),
    })),
  })

export const useRunsQueries = (runIds: string[]) =>
  useQueries({
    queries: runIds.map((id) => ({
      queryKey: ["queries", id],
      queryFn: () => fetchQueries(id),
    })),
  })

export function useRecord(recordId: string, enabled = true) {
  return useQuery({
    queryKey: ["record", recordId],
    enabled,
    queryFn: async () => {
      const { data, error } = await api.GET("/api/records/{record_id}", {
        params: { path: { record_id: recordId } },
      })
      if (error) throw new Error(`could not load record ${recordId}`)
      return data.text
    },
  })
}

// Every number in metrics.json under a dotted path, e.g. "latency_ms.end_to_end.p95".
export function flatten(obj: unknown, prefix = ""): Record<string, number> {
  const out: Record<string, number> = {}
  if (obj && typeof obj === "object")
    for (const [k, v] of Object.entries(obj)) {
      const key = prefix ? `${prefix}.${k}` : k
      if (typeof v === "number") out[key] = v
      else if (v && typeof v === "object") Object.assign(out, flatten(v, key))
    }
  return out
}

// The unit of every number in metrics.json (mlsrag/results.py, Metrics), by its flattened path.
const UNITS: Record<string, string> = {
  "retrieval.Recall@5": "0–1",
  "retrieval.Recall@10": "0–1",
  "retrieval.Recall@20": "0–1",
  "retrieval.Recall@100": "0–1",
  "retrieval.MRR@10": "0–1",
  "retrieval.NDCG@10": "0–1",
  "answers.correctness": "0–4",
  "answers.grounding": "0–4",
  "answers.completeness": "0–4",
  "answers.normalized": "0–1",
  "answers.judged_queries": "count",
  "answers.repeats": "count",
  "answers.judge_calls": "count",
  "cost.elm_calls": "calls",
  "cost.prompt_tokens": "tokens",
  "cost.completion_tokens": "tokens",
  "cost.audio_seconds": "s",
  "cost.est_usd": "USD",
  "latency_ms.asr": "ms",
  "latency_ms.encode": "ms",
  "latency_ms.retrieve": "ms",
  "latency_ms.context": "ms",
  "latency_ms.generate": "ms",
  "latency_ms.end_to_end.mean": "ms",
  "latency_ms.end_to_end.p50": "ms",
  "latency_ms.end_to_end.p95": "ms",
  "resources.cpu_s": "s",
  "resources.peak_rss_mb": "MB",
}

export const unit = (path: string) => UNITS[path] ?? ""

// Remote (ELM) calls: chat and transcribe. Embeddings are local and free, so they never count.
// Remote-call spans served from the cache take about 1 ms and cost nothing.
export const isGatewaySpan = (s: Span) =>
  s.name === "gateway.chat" || s.name === "gateway.transcribe"
export const isCached = (s: Span) => s.attrs.cached === true

// Latency and cost are valid only for runs made with --fresh (no cache reads).
export const NOT_FRESH = "not a fresh run: latency and cost invalid"
export const isTimeOrCost = (path: string) =>
  path.startsWith("latency_ms") || path.startsWith("cost")

export const MAX_COMPARE = 6

export const SLIM_NOTE =
  "This run kept only per-query metrics, stage timings and judge scores; its transcripts, prompts and answers were not committed."

// A comma-separated run list from the URL without repeats, at most MAX_COMPARE runs.
export const uniqueIds = (list: string) =>
  [...new Set(list.split(",").filter(Boolean))].slice(0, MAX_COMPARE).join(",")

export const runDate = (runId: string) =>
  runId.replace(/^(\d{4})(\d{2})(\d{2})-(\d{2})(\d{2}).*/, "$1-$2-$3 $4:$5")

// How every number is shown: decimals by size, so scores read 0.18 (not
// 0.18333333333333335), latencies 113 or 74,123 and counts 1,250. The stored
// value stays in the data and in each cell's tooltip (`title`).
//   integers           1,250
//   |v| >= 100         74,123   (no decimals)
//   10 <= |v| < 100    85.3     (one)
//   0.01 <= |v| < 10   0.18     (two)
//   0 < |v| < 0.01     0.0012   (two significant digits, so small costs do not read 0.00)
export function fmt(v: number | string | null | undefined): string {
  if (v == null) return "–"
  if (typeof v === "string") return v
  const a = Math.abs(v)
  if (Number.isInteger(v)) return v.toLocaleString("en-GB")
  if (a >= 100) return Math.round(v).toLocaleString("en-GB")
  if (a >= 10) return v.toFixed(1)
  if (a >= 0.01) return v.toFixed(2)
  return v.toPrecision(2)
}

// Subsets used for quick tests; the Results page hides their runs unless asked.
export const isSmoke = (subset: string) => subset.includes("smoke")
