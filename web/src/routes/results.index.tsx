import { Link, createFileRoute, useNavigate } from "@tanstack/react-router"

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
  isQuickCheck,
  label,
  runDate,
  uniqueIds,
  useResults,
} from "@/lib/results"

const HEADLINE = ["Recall@10", "MRR@10", "NDCG@10"]

// "" = every run except 20-question quick checks (the default), "*" = every run, otherwise one subset.
type Search = { q?: string; sel?: string; subset?: string }

export const Route = createFileRoute("/results/")({
  validateSearch: (s: Record<string, unknown>): Search => ({
    q: typeof s.q === "string" && s.q ? s.q : undefined,
    sel: typeof s.sel === "string" && s.sel ? uniqueIds(s.sel) : undefined,
    subset: typeof s.subset === "string" && s.subset ? s.subset : undefined,
  }),
  component: Results,
})

type MatrixSummary = {
  name: string
  subset: string
  rows: { run: string; config: string }[]
}

function Results() {
  const { q = "", sel, subset = "" } = Route.useSearch()
  const navigate = useNavigate({ from: Route.fullPath })
  const results = useResults()
  const selected = sel ? sel.split(",") : []

  const setSearch = (next: Search) =>
    navigate({ search: (s) => ({ ...s, ...next }), replace: true })
  const toggle = (id: string) => {
    const next = selected.includes(id)
      ? selected.filter((s) => s !== id)
      : [...selected, id]
    setSearch({ sel: next.join(",") || undefined })
  }

  if (results.isPending) return <p>Loading results…</p>
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
  const inSubset = (s: string) =>
    subset === "*" || (subset === "" ? !isQuickCheck(s) : s === subset)
  const runs = results.data.runs.filter(
    (r) =>
      inSubset(r.subset) &&
      [r.run_id, r.config_name, r.subset].some((f) =>
        f.toLowerCase().includes(needle)
      )
  )
  const subsets = [...new Set(results.data.runs.map((r) => r.subset))].sort()
  const hidden = results.data.runs.filter((r) => !inSubset(r.subset)).length
  const matrices = (results.data.matrices as MatrixSummary[]).filter((m) =>
    inSubset(m.subset)
  )

  return (
    <div className="flex flex-col gap-4">
      <h1 className="font-heading text-2xl font-medium">Results</h1>
      <p className="text-sm text-muted-foreground">
        Every committed run under <code>rag/extensions/results/</code>. Tick up
        to {MAX_COMPARE} runs on the same subset and compare them.
      </p>

      <div className="flex flex-wrap items-center gap-2">
        <Input
          type="search"
          aria-label="Search runs"
          placeholder="Search run, config or subset"
          value={q}
          onChange={(e) => setSearch({ q: e.target.value || undefined })}
          className="max-w-xs"
        />
        <label className="flex items-center gap-2 text-sm">
          Subset
          <select
            value={subset}
            onChange={(e) => setSearch({ subset: e.target.value || undefined })}
            className="h-8 rounded-lg border bg-background px-2"
          >
            <option value="">all but quick checks (practice-20)</option>
            <option value="*">all</option>
            {subsets.map((s) => (
              <option key={s} value={s}>
                {s}
              </option>
            ))}
          </select>
        </label>
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
          {hidden > 0 && `, ${hidden} hidden by the subset filter`}
        </span>
      </div>

      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>
              <span className="sr-only">Compare</span>
            </TableHead>
            <TableHead>Run</TableHead>
            <TableHead>Config</TableHead>
            <TableHead>Subset</TableHead>
            <TableHead className="text-right">Queries</TableHead>
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
                  aria-label={`Compare ${r.run_id}`}
                  checked={selected.includes(r.run_id)}
                  disabled={blocked(r) !== undefined}
                  title={blocked(r)}
                  onChange={() => toggle(r.run_id)}
                />
              </TableCell>
              <TableCell>
                <Link
                  to="/results/$runId"
                  params={{ runId: r.run_id }}
                  className="font-mono text-xs underline-offset-2 hover:underline"
                >
                  {r.run_id}
                </Link>
              </TableCell>
              <TableCell>{r.config_name}</TableCell>
              <TableCell>{r.subset}</TableCell>
              <TableCell className="text-right">{fmt(r.n_queries)}</TableCell>
              <TableCell>{runDate(r.run_id)}</TableCell>
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

      <h2 className="font-heading text-lg font-medium">Matrices</h2>
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
                className="font-medium underline-offset-2 hover:underline"
              >
                {m.name}
              </Link>{" "}
              <span className="text-muted-foreground">
                on {m.subset}: {m.rows.map((r) => r.config).join(", ")}
              </span>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
