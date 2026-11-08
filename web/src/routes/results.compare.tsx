import { Link, createFileRoute, useNavigate } from "@tanstack/react-router"

import { GroupedBars } from "@/components/charts/grouped-bars"
import { LatencyBreakdown } from "@/components/charts/latency-breakdown"
import { PerQueryStrip } from "@/components/charts/per-query-strip"
import { QualityCostScatter } from "@/components/charts/quality-cost-scatter"
import {
  NOT_FRESH,
  isTimeOrCost,
  uniqueIds,
  flatten,
  runDate,
  unit,
  useRuns,
  useRunsQueries,
} from "@/lib/results"

const STAGES = ["asr", "encode", "retrieve", "context", "generate"]
const X_AXES = {
  calls: "mean ELM calls per query",
  latency: "mean end-to-end latency (ms)",
}

type Search = { runs: string; metric?: string; x?: keyof typeof X_AXES }

export const Route = createFileRoute("/results/compare")({
  validateSearch: (s: Record<string, unknown>): Search => ({
    runs: typeof s.runs === "string" ? uniqueIds(s.runs) : "",
    metric: typeof s.metric === "string" ? s.metric : undefined,
    x: s.x === "latency" ? "latency" : undefined,
  }),
  component: Compare,
})

// config.json as dotted field -> printed value, so nested params diff field by field.
function flatConfig(obj: object, prefix = ""): Record<string, string> {
  const out: Record<string, string> = {}
  for (const [k, v] of Object.entries(obj)) {
    const key = prefix ? `${prefix}.${k}` : k
    if (v && typeof v === "object" && !Array.isArray(v))
      Object.assign(out, flatConfig(v, key))
    else out[key] = JSON.stringify(v)
  }
  return out
}

function Compare() {
  const {
    runs: runsParam,
    metric = "retrieval.NDCG@10",
    x = "calls",
  } = Route.useSearch()
  const navigate = useNavigate({ from: Route.fullPath })
  const ids = runsParam.split(",").filter(Boolean)
  const runs = useRuns(ids)
  const traces = useRunsQueries(ids)

  if (ids.length === 0)
    return (
      <p>
        No runs selected. Pick some on <Link to="/results">Results</Link>.
      </p>
    )
  const failed = [...runs, ...traces].find((r) => r.error)
  if (failed) return <p>{failed.error!.message}</p>
  if (runs.some((r) => !r.data) || traces.some((t) => !t.data))
    return <p>Loading {ids.length} runs…</p>

  const details = runs.map((r) => r.data!)
  const queries = traces.map((t) => t.data!)
  const series = details.map((d) => ({
    id: d.metrics.run.run_id,
    label: `${d.metrics.run.config_name} ${runDate(d.metrics.run.run_id)}`,
  }))
  const flat = details.map((d) => flatten({ ...d.metrics, run: undefined }))
  const metricKeys = [...new Set(flat.flatMap(Object.keys))]
  const picked = metricKeys.includes(metric) ? metric : "retrieval.NDCG@10"
  const family = metricKeys.filter(
    (k) => k.split(".")[0] === picked.split(".")[0] && unit(k) === unit(picked)
  )
  const short = (k: string) => k.split(".").slice(1).join(".")
  const perQuery = picked.startsWith("retrieval.")
    ? picked
    : "retrieval.NDCG@10"

  const configs = details.map((d) => flatConfig(d.config))
  const fields = [...new Set(configs.flatMap(Object.keys))]
  const differs = fields.filter((f) =>
    configs.some((c) => c[f] !== configs[0][f])
  )

  // Latency and cost of a run made without --fresh are shown as "–".
  const fresh = details.map((d) => d.metrics.run.fresh === true)
  const sample = (i: number, k: string) =>
    isTimeOrCost(k) && !fresh[i] ? null : (flat[i][k] ?? null)
  const cachedNote = fresh.some((f) => !f) && (
    <span title={NOT_FRESH}>
      No latency or cost shown for{" "}
      {series
        .filter((_, i) => !fresh[i])
        .map((s) => s.label)
        .join(", ")}
      : {NOT_FRESH}.
    </span>
  )
  const xValue = (i: number) => {
    if (!fresh[i]) return null
    return x === "calls"
      ? details[i].metrics.cost.elm_calls / details[i].metrics.run.n_queries
      : (flat[i]["latency_ms.end_to_end.mean"] ?? null)
  }

  return (
    <div className="flex flex-col gap-4">
      <Link to="/results" className="text-sm text-muted-foreground">
        ← Results
      </Link>
      <h1 className="font-heading text-2xl font-medium">Compare runs</h1>

      <section className="flex flex-col gap-1">
        <h2 className="text-sm font-medium">
          What differs between the configs ({differs.length} of {fields.length}{" "}
          fields)
        </h2>
        <ConfigTable runs={series} configs={configs} fields={differs} />
        <details className="text-sm">
          <summary className="cursor-pointer text-muted-foreground">
            All config fields
          </summary>
          <ConfigTable runs={series} configs={configs} fields={fields} />
        </details>
      </section>

      <div className="flex flex-wrap items-center gap-3 text-sm">
        <label className="flex items-center gap-2">
          Metric
          <select
            value={picked}
            onChange={(e) =>
              navigate({
                search: (s) => ({ ...s, metric: e.target.value }),
                replace: true,
              })
            }
            className="h-8 rounded-lg border bg-background px-2"
          >
            {metricKeys.map((k) => (
              <option key={k} value={k}>
                {k}
              </option>
            ))}
          </select>
        </label>
        <label className="flex items-center gap-2">
          Cost axis
          <select
            value={x}
            onChange={(e) =>
              navigate({
                search: (s) => ({
                  ...s,
                  x: e.target.value as keyof typeof X_AXES,
                }),
                replace: true,
              })
            }
            className="h-8 rounded-lg border bg-background px-2"
          >
            {Object.entries(X_AXES).map(([k, label]) => (
              <option key={k} value={k}>
                {label}
              </option>
            ))}
          </select>
        </label>
      </div>

      <div className="grid min-w-0 grid-cols-1 gap-4 lg:grid-cols-2">
        <GroupedBars
          title={`${picked.split(".")[0]} (${unit(picked)}) by run`}
          runs={series}
          metrics={family.map(short)}
          value={(i, m) => sample(i, `${picked.split(".")[0]}.${m}`)}
          yLabel={unit(picked)}
          note={isTimeOrCost(picked) ? cachedNote : undefined}
        />
        <LatencyBreakdown
          title="Latency by stage (mean per query)"
          runs={series}
          stages={STAGES}
          value={(i, s) => sample(i, `latency_ms.${s}`)}
          note={cachedNote}
        />
        <QualityCostScatter
          title={`${picked} against ${X_AXES[x]}`}
          points={series.map((s, i) => ({
            label: s.label,
            x: xValue(i),
            y: sample(i, picked),
          }))}
          xLabel={X_AXES[x]}
          yLabel={picked}
          note={cachedNote}
        />
        <PerQueryStrip
          title={`${perQuery} per query`}
          runs={series}
          queryIds={[
            ...new Set(queries.flatMap((qs) => qs.map((q) => q.query_id))),
          ]}
          value={(i, id) =>
            queries[i].find((q) => q.query_id === id)?.metrics[
              short(perQuery)
            ] ?? null
          }
          yLabel={short(perQuery)}
          note={
            perQuery !== picked &&
            "Per-query values exist only for retrieval metrics."
          }
        />
      </div>
    </div>
  )
}

function ConfigTable({
  runs,
  configs,
  fields,
}: {
  runs: { id: string; label: string }[]
  configs: Record<string, string>[]
  fields: string[]
}) {
  if (fields.length === 0)
    return (
      <p className="text-sm text-muted-foreground">
        The configs are identical.
      </p>
    )
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-xs">
        <thead>
          <tr className="border-b">
            <th className="p-1 text-left font-medium">field</th>
            {runs.map((r) => (
              <th key={r.id} className="p-1 text-left font-medium">
                {r.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {fields.map((f) => (
            <tr key={f} className="border-b last:border-0">
              <td className="p-1 font-mono">{f}</td>
              {configs.map((c, i) => (
                <td key={i} className="p-1 font-mono break-all">
                  {c[f] ?? "–"}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
