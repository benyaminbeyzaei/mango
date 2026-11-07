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
  }
}
export type Span = {
  name: string
  start_ms: number
  duration_ms: number
  attrs: Record<string, unknown>
}
// One line of queries.jsonl.
export type QueryTrace = {
  query_id: string
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

export function unit(path: string) {
  if (path.startsWith("latency_ms")) return "ms"
  if (path === "resources.cpu_s") return "s"
  if (path === "resources.peak_rss_mb") return "MB"
  if (/^answers\.(judged_queries|repeats|judge_calls)$/.test(path))
    return "count"
  if (path === "cost.est_usd") return "USD"
  if (path.endsWith("tokens")) return "tokens"
  if (path === "cost.audio_seconds") return "s"
  if (path.startsWith("cost")) return "calls"
  if (path.startsWith("answers.") && path !== "answers.normalized") return "0–4"
  return "0–1"
}

// Remote-call spans served from the cache take about 1 ms and cost nothing.
export const isGatewaySpan = (s: Span) => s.name.startsWith("gateway.")
export const isCached = (s: Span) => s.attrs.cached === true

// Share of a run's remote-call spans that were cache hits (0 when there were none).
export function cachedShare(queries: QueryTrace[]) {
  const calls = queries.flatMap((q) => q.spans.filter(isGatewaySpan))
  return calls.length ? calls.filter(isCached).length / calls.length : 0
}

export const MAX_COMPARE = 6

// A comma-separated run list from the URL without repeats, at most MAX_COMPARE runs.
export const uniqueIds = (list: string) =>
  [...new Set(list.split(",").filter(Boolean))].slice(0, MAX_COMPARE).join(",")

export const runDate = (runId: string) =>
  runId.replace(/^(\d{4})(\d{2})(\d{2})-(\d{2})(\d{2}).*/, "$1-$2-$3 $4:$5")

// Short display of a stored number for cards; tables print stored values in full.
export const fmt = (v: number | null | undefined) =>
  v == null ? "–" : Number.isInteger(v) ? String(v) : v.toFixed(4)
