import { Link, createFileRoute } from "@tanstack/react-router"

import { Markdown } from "@/components/markdown"
import { QueryDetail } from "@/components/query-detail"
import { useRun, useRunQueries } from "@/lib/results"

export const Route = createFileRoute("/results/$runId/$queryId")({
  component: StoredQuery,
})

// One stored query of a run, read-only: the answer, its scores, and the full detail panel.
function StoredQuery() {
  const { runId, queryId } = Route.useParams()
  const run = useRun(runId)
  const queries = useRunQueries(runId)

  if (run.error || queries.error)
    return <p>{(run.error ?? queries.error)!.message}</p>
  if (!run.data || !queries.data) return <p>Loading {queryId}…</p>
  const q = queries.data.find((x) => x.query_id === queryId)
  if (!q)
    return (
      <p>
        No query {queryId} in {runId}.
      </p>
    )

  const k = Number(run.data.config.k)
  const a = q.artifacts
  const response = {
    answer: a.answer,
    transcript: a.transcript,
    prompt: a.prompt,
    // the API keeps the record text out of the trace; QueryDetail loads it per record
    retrieved: a.ranked_ids
      .slice(0, k)
      .map((id, i) => ({ record_id: id, score: a.scores[i], text: "" })),
    trace: q as unknown as Record<string, unknown>,
  }
  return (
    <div className="flex flex-col gap-4">
      <Link
        to="/results/$runId"
        params={{ runId }}
        className="text-sm break-all text-muted-foreground"
      >
        ← {runId}
      </Link>
      <h1 className="font-heading text-xl font-medium break-all">{queryId}</h1>

      <section className="flex flex-col gap-1">
        <h2 className="text-sm font-medium">Answer</h2>
        {a.answer ? (
          <div className="flex flex-col gap-2 text-sm">
            <Markdown>{a.answer}</Markdown>
          </div>
        ) : (
          <p className="text-sm text-muted-foreground">
            This run did not generate answers.
          </p>
        )}
      </section>

      <section className="flex flex-col gap-1">
        <h2 className="text-sm font-medium">Retrieval metrics</h2>
        <p className="font-mono text-xs">
          {Object.entries(q.metrics)
            .map(([m, v]) => `${m} ${v}`)
            .join(" · ")}
        </p>
      </section>

      <section className="flex flex-col gap-1">
        <h2 className="text-sm font-medium">Judge</h2>
        {q.judge ? (
          <pre className="text-xs whitespace-pre-wrap">
            {JSON.stringify(q.judge, null, 2)}
          </pre>
        ) : (
          <p className="text-sm text-muted-foreground">
            No judge scores are stored with this trace.
          </p>
        )}
      </section>

      <QueryDetail response={response} open />
    </div>
  )
}
