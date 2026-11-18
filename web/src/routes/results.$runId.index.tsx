import { Link, createFileRoute, useNavigate } from "@tanstack/react-router"
import { ArrowDownIcon, ArrowUpIcon } from "lucide-react"

import { Input } from "@/components/ui/input"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import {
  NOT_FRESH,
  SLIM_NOTE,
  fmt,
  flatten,
  isCached,
  isGatewaySpan,
  isTimeOrCost,
  unit,
  useRun,
  useRunQueries,
  type QueryTrace,
} from "@/lib/results"

type Search = { sort?: string; desc?: boolean; filter?: string }

export const Route = createFileRoute("/results/$runId/")({
  validateSearch: (s: Record<string, unknown>): Search => ({
    sort: typeof s.sort === "string" ? s.sort : undefined,
    desc: s.desc === true || s.desc === "true" ? true : undefined,
    filter: typeof s.filter === "string" && s.filter ? s.filter : undefined,
  }),
  component: RunDetail,
})

// Per-query columns: the stored retrieval metrics, then wall time and real (uncached) remote calls.
// A slim trace has no gateway spans, so its call columns would read 0: they are left out.
function row(q: QueryTrace): Record<string, number | string> {
  const calls = q.spans.filter(isGatewaySpan)
  return {
    query_id: q.query_id,
    ...q.metrics,
    wall_ms: q.resources.wall_ms ?? 0,
    ...(q.slim
      ? {}
      : {
          remote_calls: calls.filter((s) => !isCached(s)).length,
          cache_hits: calls.filter(isCached).length,
        }),
  }
}

function RunDetail() {
  const { runId } = Route.useParams()
  const { sort = "query_id", desc = false, filter = "" } = Route.useSearch()
  const navigate = useNavigate({ from: Route.fullPath })
  const run = useRun(runId)
  const queries = useRunQueries(runId)

  if (run.error || queries.error)
    return <p>{(run.error ?? queries.error)!.message}</p>
  if (!run.data || !queries.data) return <p>Loading {runId}…</p>

  const flat = flatten({ ...run.data.metrics, run: undefined })
  const fresh = run.data.metrics.run.fresh === true
  const rows = queries.data
    .map(row)
    .filter((r) => String(r.query_id).includes(filter))
  const columns = queries.data.length ? Object.keys(row(queries.data[0])) : []
  rows.sort((a, b) => {
    const c = a[sort] < b[sort] ? -1 : a[sort] > b[sort] ? 1 : 0
    return desc ? -c : c
  })
  const sortBy = (col: string) =>
    navigate({
      search: (s) => ({
        ...s,
        sort: col,
        desc: col === sort && !desc ? true : undefined,
      }),
      replace: true,
    })

  const groups = ["retrieval", "answers", "latency_ms", "cost", "resources"]
  return (
    <div className="flex flex-col gap-4">
      <Link to="/results" className="text-sm text-muted-foreground">
        ← Results
      </Link>
      <h1 className="font-heading text-xl font-medium break-all">{runId}</h1>
      <p className="text-sm text-muted-foreground">
        config {run.data.metrics.run.config_name}, subset{" "}
        {run.data.metrics.run.subset}, {run.data.metrics.run.n_queries} queries
        {!fresh && `; ${NOT_FRESH}`}
      </p>
      {queries.data[0]?.slim && (
        <p className="text-sm text-muted-foreground">{SLIM_NOTE}</p>
      )}

      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
        {groups.map((g) => {
          const keys = Object.keys(flat).filter((k) => k.startsWith(`${g}.`))
          return (
            <section key={g} className="rounded-lg border p-3">
              <h2 className="text-xs font-medium tracking-wide text-muted-foreground uppercase">
                {g}
              </h2>
              {keys.length === 0 ? (
                <p className="text-sm text-muted-foreground">not stored</p>
              ) : (
                <dl className="grid grid-cols-[1fr_auto] gap-x-3 text-sm">
                  {keys.map((k) => (
                    <div key={k} className="contents">
                      <dt className="truncate">
                        {k.slice(g.length + 1)}{" "}
                        <span className="text-xs text-muted-foreground">
                          {unit(k)}
                        </span>
                      </dt>
                      {isTimeOrCost(k) && !fresh ? (
                        <dd className="text-right font-mono" title={NOT_FRESH}>
                          –
                        </dd>
                      ) : (
                        <dd
                          className="text-right font-mono"
                          title={String(flat[k])}
                        >
                          {fmt(flat[k])}
                        </dd>
                      )}
                    </div>
                  ))}
                </dl>
              )}
            </section>
          )
        })}
      </div>

      <h2 className="font-heading text-lg font-medium">Queries</h2>
      <Input
        type="search"
        aria-label="Filter by query id"
        placeholder="Filter by query id"
        value={filter}
        onChange={(e) =>
          navigate({
            search: (s) => ({ ...s, filter: e.target.value || undefined }),
            replace: true,
          })
        }
        className="max-w-xs"
      />
      <p className="text-xs text-muted-foreground">
        A query whose remote calls were all cache hits took about 1 ms per call
        and cost nothing; its wall time is not a latency sample.
      </p>
      <Table>
        <TableHeader>
          <TableRow>
            {columns.map((c) => (
              <TableHead
                key={c}
                aria-sort={
                  c === sort ? (desc ? "descending" : "ascending") : undefined
                }
              >
                <button
                  onClick={() => sortBy(c)}
                  className="inline-flex items-center gap-1 rounded outline-none focus-visible:ring-3 focus-visible:ring-ring/50"
                >
                  {c}
                  {c === sort &&
                    (desc ? (
                      <ArrowDownIcon className="size-3" />
                    ) : (
                      <ArrowUpIcon className="size-3" />
                    ))}
                </button>
              </TableHead>
            ))}
          </TableRow>
        </TableHeader>
        <TableBody>
          {rows.map((r) => (
            <TableRow key={r.query_id}>
              {columns.map((c) => (
                <TableCell key={c} className="font-mono text-xs">
                  {c === "query_id" ? (
                    <Link
                      to="/results/$runId/$queryId"
                      params={{ runId, queryId: String(r.query_id) }}
                      className="underline-offset-2 hover:underline"
                    >
                      {r.query_id}
                    </Link>
                  ) : c === "wall_ms" &&
                    r.remote_calls === 0 &&
                    Number(r.cache_hits) > 0 ? (
                    `${fmt(r[c])} (cached)`
                  ) : (
                    <span title={String(r[c])}>{fmt(r[c])}</span>
                  )}
                </TableCell>
              ))}
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  )
}
