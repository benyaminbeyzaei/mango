import { Link, createFileRoute, useNavigate } from "@tanstack/react-router"

import { GroupedBars } from "@/components/charts/grouped-bars"
import { LatencyBreakdown } from "@/components/charts/latency-breakdown"
import { QualityCostScatter } from "@/components/charts/quality-cost-scatter"
import {
  ChartSkeleton,
  Heading,
  Line,
  Loading,
  TableSkeleton,
} from "@/components/page-skeletons"
import { configLabel, runDate } from "@/lib/labels"
import {
  NOT_FRESH,
  isTimeOrCost,
  uniqueIds,
  flatten,
  label,
  unit,
  useRuns,
} from "@/lib/results"

const STAGES = ["asr", "encode", "retrieve", "context", "generate"]
const X_AXES = {
  calls: "mean ELM calls per query",
  latency: "mean end-to-end latency (ms)",
}

// metric: the bar chart; y and x: the quality and cost axes of the scatter.
type Search = {
  runs: string
  metric?: string
  y?: string
  x?: keyof typeof X_AXES
}

export const Route = createFileRoute("/results/compare")({
  validateSearch: (s: Record<string, unknown>): Search => ({
    runs: typeof s.runs === "string" ? uniqueIds(s.runs) : "",
    metric: typeof s.metric === "string" ? s.metric : undefined,
    y: typeof s.y === "string" ? s.y : undefined,
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
    y = "retrieval.NDCG@10",
    x = "calls",
  } = Route.useSearch()
  const navigate = useNavigate({ from: Route.fullPath })
  const ids = runsParam.split(",").filter(Boolean)
  const runs = useRuns(ids)

  if (ids.length === 0)
    return (
      <p>
        No runs selected. Pick some on <Link to="/results">Results</Link>.
      </p>
    )
  const setSearch = (next: Partial<Search>) =>
    navigate({
      search: (s) => ({ ...s, ...next }),
      replace: true,
      resetScroll: false,
    })
  const failed = runs.find((r) => r.error)
  if (failed) return <p>{failed.error!.message}</p>
  if (runs.some((r) => !r.data))
    return (
      <Loading label={`Loading ${ids.length} runs`}>
        <Line className="w-24" />
        <Heading className="w-48" />
        <TableSkeleton rows={3} cols={ids.length + 1} />
        <ChartSkeleton />
        <ChartSkeleton />
        <ChartSkeleton />
      </Loading>
    )

  const details = runs.map((r) => r.data!)
  const subsets = [...new Set(details.map((d) => d.metrics.run.subset))]
  if (subsets.length > 1)
    return (
      <div className="flex flex-col gap-2">
        <h1 className="font-heading text-2xl font-medium">Compare runs</h1>
        <p>
          These runs are on different subsets ({subsets.join(", ")}), so their
          numbers come from different questions and cannot be compared. Pick
          runs on one subset on{" "}
          <Link to="/results" className="underline underline-offset-4">
            Results
          </Link>
          .
        </p>
        <ul className="ml-6 list-disc text-sm">
          {details.map((d) => (
            <li key={d.metrics.run.run_id}>
              {configLabel(d.metrics.run.config_name)},{" "}
              {runDate(d.metrics.run.run_id)}, on {d.metrics.run.subset}
            </li>
          ))}
        </ul>
      </div>
    )
  // Series are named by experiment; the date is added only when one experiment appears twice.
  const names = details.map((d) => configLabel(d.metrics.run.config_name))
  const series = details.map((d, i) => ({
    id: d.metrics.run.run_id,
    label:
      names.filter((n) => n === names[i]).length > 1
        ? `${names[i]}, ${runDate(d.metrics.run.run_id)}`
        : names[i],
  }))
  const flat = details.map((d) => flatten({ ...d.metrics, run: undefined }))
  const metricKeys = [...new Set(flat.flatMap(Object.keys))]
  const picked = metricKeys.includes(metric) ? metric : "retrieval.NDCG@10"
  const family = metricKeys.filter(
    (k) => k.split(".")[0] === picked.split(".")[0] && unit(k) === unit(picked)
  )
  // The scatter's quality axis: retrieval metrics and answer scores, not counts.
  const qualityKeys = metricKeys.filter(
    (k) =>
      k.startsWith("retrieval.") ||
      /^answers\.(correctness|grounding|completeness|normalized)$/.test(k)
  )
  const quality = qualityKeys.includes(y) ? y : "retrieval.NDCG@10"

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

      <div className="flex min-w-0 flex-col gap-4">
        <GroupedBars
          title={`${picked.split(".")[0]} (${unit(picked)}) by run`}
          controls={
            <Picker
              label="Metric"
              value={picked}
              options={metricKeys.map((k) => [k, label(k)])}
              onChange={(v) => setSearch({ metric: v })}
            />
          }
          runs={series}
          metrics={family}
          value={sample}
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
          title={`${label(quality)} against ${X_AXES[x]}`}
          controls={
            <>
              <Picker
                label="Quality"
                value={quality}
                options={qualityKeys.map((k) => [k, label(k)])}
                onChange={(v) => setSearch({ y: v })}
              />
              <Picker
                label="Cost axis"
                value={x}
                options={Object.entries(X_AXES)}
                onChange={(v) => setSearch({ x: v as keyof typeof X_AXES })}
              />
            </>
          }
          points={series.map((s, i) => ({
            label: s.label,
            x: xValue(i),
            y: sample(i, quality),
          }))}
          xLabel={X_AXES[x]}
          yLabel={label(quality)}
          note={cachedNote}
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
      {/* Fixed layout: every value stays on one line, cut with "…"; hover shows it whole. */}
      <table className="w-full table-fixed text-xs">
        <thead>
          <tr className="border-b">
            <th className="w-36 p-1 text-left font-medium">field</th>
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
              <td className="truncate p-1 font-mono" title={f}>
                {f}
              </td>
              {configs.map((c, i) => (
                <td key={i} className="truncate p-1 font-mono" title={c[f]}>
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

function Picker({
  label,
  value,
  options,
  onChange,
}: {
  label: string
  value: string
  options: [string, string][] // [value, label]
  onChange: (value: string) => void
}) {
  return (
    <label className="flex items-center gap-2">
      {label}
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="h-8 rounded-lg border bg-background px-2"
      >
        {options.map(([v, l]) => (
          <option key={v} value={v}>
            {l}
          </option>
        ))}
      </select>
    </label>
  )
}
