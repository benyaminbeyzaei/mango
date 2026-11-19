import { Link, createFileRoute } from "@tanstack/react-router"

import { Markdown } from "@/components/markdown"
import { QueryDetail } from "@/components/query-detail"
import { Heading, Line, Loading } from "@/components/page-skeletons"
import { Skeleton } from "@/components/ui/skeleton"
import { configLabel, runDate } from "@/lib/labels"
import {
  SLIM_NOTE,
  fmt,
  traceToResponse,
  useRun,
  useRunQueries,
} from "@/lib/results"

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
  if (!run.data || !queries.data)
    return (
      <Loading label="Loading question">
        <Line className="w-40" />
        <Heading className="w-72" />
        <Skeleton className="h-28 w-full" />
        <Skeleton className="h-64 w-full" />
      </Loading>
    )
  const q = queries.data.find((x) => x.query_id === queryId)
  if (!q)
    return (
      <p>
        No query {queryId} in {runId}.
      </p>
    )

  const a = q.artifacts
  const response = traceToResponse(q, Number(run.data.config.k))
  return (
    <div className="flex flex-col gap-4">
      <Link
        to="/results/$runId"
        params={{ runId }}
        className="text-sm break-all text-muted-foreground"
      >
        ← {configLabel(run.data.metrics.run.config_name)}, {runDate(runId)}
      </Link>
      <h1 className="font-heading text-xl font-medium break-all">{queryId}</h1>
      {q.slim && <p className="text-sm text-muted-foreground">{SLIM_NOTE}</p>}

      <section className="flex flex-col gap-1">
        <h2 className="text-sm font-medium">Answer</h2>
        {a.answer ? (
          <div className="flex flex-col gap-2 text-sm">
            <Markdown>{a.answer}</Markdown>
          </div>
        ) : (
          <p className="text-sm text-muted-foreground">
            {q.slim
              ? "The answer was not committed for this run."
              : "This run did not generate answers."}
          </p>
        )}
      </section>

      <section className="flex flex-col gap-1">
        <h2 className="text-sm font-medium">Retrieval metrics</h2>
        <p className="font-mono text-xs">
          {Object.entries(q.metrics)
            .map(([m, v]) => `${m} ${fmt(v)}`)
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
