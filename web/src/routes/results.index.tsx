import { Link, createFileRoute, useNavigate } from "@tanstack/react-router"

import {
  Heading,
  Line,
  Loading,
  TableSkeleton,
} from "@/components/page-skeletons"
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
  MAX_COMPARE,
  NOT_FRESH,
  fmt,
  label,
  uniqueIds,
  useResults,
} from "@/lib/results"
import {
  SUBSETS,
  configLabel,
  matrixLabel,
  runDate,
  subsetOrder,
} from "@/lib/labels"

const HEADLINE = ["Recall@10", "MRR@10", "NDCG@10"]

// hide: question sets left unticked, comma-separated; absent = the default (practice-20 hidden),
// "none" = every set shown.
type Search = { q?: string; sel?: string; hide?: string }
const DEFAULT_HIDDEN = ["practice-20"]

export const Route = createFileRoute("/results/")({
  validateSearch: (s: Record<string, unknown>): Search => ({
    q: typeof s.q === "string" && s.q ? s.q : undefined,
    sel: typeof s.sel === "string" && s.sel ? uniqueIds(s.sel) : undefined,
    hide: typeof s.hide === "string" && s.hide ? s.hide : undefined,
  }),
  component: Results,
})

type MatrixSummary = {
  name: string
  subset: string
  rows: { run: string; config: string }[]
}

function Results() {
  const { q = "", sel, hide } = Route.useSearch()
  const navigate = useNavigate({ from: Route.fullPath })
  const results = useResults()
  const selected = sel ? sel.split(",") : []

  const setSearch = (next: Search) =>
    navigate({
      search: (s) => ({ ...s, ...next }),
      replace: true,
      resetScroll: false,
    })
  const toggle = (id: string) => {
    const next = selected.includes(id)
      ? selected.filter((s) => s !== id)
      : [...selected, id]
    setSearch({ sel: next.join(",") || undefined })
  }

  if (results.isPending) return <ResultsSkeleton />
  if (results.error) return <p>{results.error.message}</p>

  // Runs are comparable only on the same subset (the same questions): once one run is
  // ticked, runs on other subsets cannot be.
  const pickedSubset = results.data.runs.find((r) =>
    selected.includes(r.run_id)
  )?.subset
  const blocked = (r: { run_id: string; subset: string }) =>
    selected.includes(r.run_id)
      ? undefined
      : selected.length >= MAX_COMPARE
        ? `at most ${MAX_COMPARE} runs`
        : pickedSubset && r.subset !== pickedSubset
          ? `only runs on ${pickedSubset} can be compared with the ticked runs`
          : undefined
  const needle = q.toLowerCase()
  const hiddenSets =
    hide === undefined ? DEFAULT_HIDDEN : hide === "none" ? [] : hide.split(",")
  const inSubset = (s: string) => !hiddenSets.includes(s)
  const toggleSubset = (s: string) => {
    const next = inSubset(s)
      ? [...hiddenSets, s]
      : hiddenSets.filter((h) => h !== s)
    const same =
      next.length === DEFAULT_HIDDEN.length &&
      DEFAULT_HIDDEN.every((h) => next.includes(h))
    setSearch({ hide: same ? undefined : next.join(",") || "none" })
  }
  const runs = results.data.runs.filter(
    (r) =>
      inSubset(r.subset) &&
      [r.run_id, r.config_name, configLabel(r.config_name), r.subset].some(
        (f) => f.toLowerCase().includes(needle)
      )
  )
  const subsets = [...new Set(results.data.runs.map((r) => r.subset))].sort(
    (a, b) => subsetOrder(a) - subsetOrder(b)
  )
  const count = (s: string) =>
    results.data.runs.filter((r) => r.subset === s).length
  const hidden = results.data.runs.filter((r) => !inSubset(r.subset)).length
  const matrices = (results.data.matrices as MatrixSummary[]).filter((m) =>
    inSubset(m.subset)
  )

  return (
    <div className="flex flex-col gap-4">
      <h1 className="font-heading text-2xl font-medium">Results</h1>
      <p className="text-sm text-muted-foreground">
        Every committed experiment run. Tick up to {MAX_COMPARE} runs on the
        same question set to compare them.
      </p>

      <div className="flex flex-wrap items-center gap-2">
        <Input
          type="search"
          aria-label="Search runs"
          placeholder="Search experiments"
          value={q}
          onChange={(e) => setSearch({ q: e.target.value || undefined })}
          className="max-w-xs"
        />
        <fieldset className="flex flex-wrap items-center gap-x-3 gap-y-1 text-sm">
          <legend className="sr-only">Question sets shown</legend>
          <span aria-hidden="true">Question sets</span>
          {subsets.map((s) => (
            <label
              key={s}
              className="flex items-center gap-1.5"
              title={`${SUBSETS.find((x) => x.id === s)?.questions ?? "?"} questions`}
            >
              <input
                type="checkbox"
                checked={inSubset(s)}
                onChange={() => toggleSubset(s)}
              />
              {s}
              <span className="text-muted-foreground">({count(s)})</span>
            </label>
          ))}
        </fieldset>
        <Link
          to="/results/compare"
          search={{ runs: selected.join(",") }}
          disabled={selected.length === 0}
          className="rounded-lg border px-3 py-1.5 text-sm font-medium outline-none hover:bg-muted focus-visible:ring-3 focus-visible:ring-ring/50 aria-disabled:pointer-events-none aria-disabled:opacity-50"
        >
          Compare {selected.length} of {MAX_COMPARE}
        </Link>
        <span className="text-sm text-muted-foreground">
          {runs.length} runs
          {hidden > 0 && `, ${hidden} hidden`}
        </span>
      </div>

      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>
              <span className="sr-only">Compare</span>
            </TableHead>
            <TableHead>Experiment</TableHead>
            <TableHead>Question set</TableHead>
            <TableHead>Date</TableHead>
            {HEADLINE.map((m) => (
              <TableHead key={m} className="text-right">
                {label(`retrieval.${m}`)}
              </TableHead>
            ))}
            <TableHead className="text-right">
              {label("cost.elm_calls")}
            </TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {runs.map((r) => (
            <TableRow key={r.run_id}>
              <TableCell>
                <input
                  type="checkbox"
                  aria-label={`Compare ${configLabel(r.config_name)}, ${runDate(r.run_id)}`}
                  checked={selected.includes(r.run_id)}
                  disabled={blocked(r) !== undefined}
                  title={blocked(r)}
                  onChange={() => toggle(r.run_id)}
                />
              </TableCell>
              <TableCell className="max-w-80 whitespace-normal">
                <Link
                  to="/results/$runId"
                  params={{ runId: r.run_id }}
                  title={r.run_id}
                  className="font-medium underline-offset-2 hover:underline"
                >
                  {configLabel(r.config_name)}
                </Link>
                <div className="font-mono text-xs text-muted-foreground">
                  {r.config_name}
                </div>
              </TableCell>
              <TableCell>
                {r.subset}
                <span className="text-muted-foreground">
                  {" "}
                  ({fmt(r.n_queries)})
                </span>
              </TableCell>
              <TableCell className="whitespace-nowrap text-muted-foreground">
                {runDate(r.run_id)}
              </TableCell>
              {HEADLINE.map((m) => (
                <TableCell
                  key={m}
                  className="text-right font-mono"
                  title={String(r.retrieval[m])}
                >
                  {fmt(r.retrieval[m])}
                </TableCell>
              ))}
              <TableCell
                className="text-right font-mono"
                title={r.fresh ? undefined : NOT_FRESH}
              >
                {r.fresh ? fmt(r.cost.elm_calls as number) : "–"}
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
      {runs.length === 0 && (
        <p className="text-sm text-muted-foreground">No run matches.</p>
      )}

      <h2 className="font-heading text-lg font-medium">Comparison groups</h2>
      {matrices.length === 0 ? (
        <p className="text-sm text-muted-foreground">
          No matrix has been committed yet (<code>mlsrag matrix</code> writes
          one).
        </p>
      ) : (
        <ul className="flex flex-col gap-1 text-sm">
          {matrices.map((m) => (
            <li key={m.name}>
              <Link
                to="/results/compare"
                search={{
                  runs: m.rows
                    .slice(0, MAX_COMPARE)
                    .map((r) => r.run)
                    .join(","),
                }}
                title={m.name}
                className="font-medium underline-offset-2 hover:underline"
              >
                {matrixLabel(m.name, m.subset)}
              </Link>{" "}
              <span className="text-muted-foreground">
                on {m.subset}:{" "}
                {m.rows.map((r) => configLabel(r.config)).join("; ")}
              </span>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}

function ResultsSkeleton() {
  return (
    <Loading label="Loading results">
      <Heading className="w-32" />
      <Line className="w-full max-w-lg" />
      <div className="flex flex-wrap items-center gap-3">
        <Line className="h-8 w-64" />
        <Line className="w-80" />
      </div>
      <TableSkeleton rows={12} cols={7} rowHeight="h-14" />
    </Loading>
  )
}
