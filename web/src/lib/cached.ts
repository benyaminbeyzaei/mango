// The cached questions of the static build: three full evaluations on the test questions, written by
// `mlsrag export-web` as /static-api/cached-qa.json. The stored traces are shown like live answers.
import { useQuery } from "@tanstack/react-query"

import { STATIC } from "@/api/client"
import { fetchQueries, traceToResponse } from "@/lib/results"

export type CachedRun = {
  run_id: string
  config_name: string
  subset: string
  k: number
  transcripts: string // "our ASR" or "whisper-1"
  ndcg10: number
  answer_score: number | null
  questions: { query_id: string; transcript: string }[]
}

export type CachedPick = { runId: string; queryId: string; transcript: string }

export function useCachedRuns() {
  return useQuery({
    queryKey: ["cached-qa"],
    enabled: STATIC,
    queryFn: async () => {
      const res = await fetch("/static-api/cached-qa.json")
      if (!res.ok) throw new Error("could not load the cached questions")
      return (await res.json()) as CachedRun[]
    },
  })
}

export async function askCached(pick: CachedPick, k: number) {
  const q = (await fetchQueries(pick.runId)).find(
    (x) => x.query_id === pick.queryId
  )
  if (!q) throw new Error(`no question ${pick.queryId} in ${pick.runId}`)
  return traceToResponse(q, k)
}

// One of the runs that answered this question, chosen at random (undefined if none did).
export function randomRunFor(runs: CachedRun[], queryId: string) {
  const having = runs.filter((r) =>
    r.questions.some((q) => q.query_id === queryId)
  )
  return having[Math.floor(Math.random() * having.length)]
}
