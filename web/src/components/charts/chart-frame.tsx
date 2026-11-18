// Shared frame for every chart: a title, the controls that change it, a note, and a button that swaps the chart for a table of the same numbers.
import { useState, type ReactNode } from "react"

import { Button } from "@/components/ui/button"
import { fmt } from "@/lib/results"

// The preset's chart colours (greyscale), ordered so neighbours contrast, plus the text colour for a sixth series.
export const PALETTE = [
  "var(--chart-2)",
  "var(--chart-4)",
  "var(--chart-1)",
  "var(--chart-3)",
  "var(--chart-5)",
  "var(--foreground)",
]

export const AXIS = { fontSize: 11, fill: "var(--muted-foreground)" }
export const TOOLTIP = {
  contentStyle: {
    background: "var(--popover)",
    border: "1px solid var(--border)",
    fontSize: 12,
  },
  labelStyle: { color: "var(--popover-foreground)" },
  formatter: (v: unknown) => (typeof v === "number" ? fmt(v) : String(v)),
}

// Legend text in the text colour; Recharts would otherwise use each series' (possibly pale) colour.
export const LEGEND = {
  wrapperStyle: { fontSize: 11 },
  formatter: (value: string) => (
    <span style={{ color: "var(--foreground)" }}>{value}</span>
  ),
}

type TableData = {
  columns: string[]
  rows: (string | number | null)[][]
}

export function ChartFrame({
  title,
  controls,
  note,
  table,
  children,
}: {
  title: string
  controls?: ReactNode
  note?: ReactNode
  table: TableData
  children: ReactNode
}) {
  const [showTable, setShowTable] = useState(false)
  return (
    <figure className="flex min-w-0 flex-col gap-2 rounded-lg border p-3">
      <div className="flex items-start justify-between gap-2">
        <figcaption className="text-sm font-medium">{title}</figcaption>
        <Button
          variant="outline"
          size="xs"
          aria-pressed={showTable}
          onClick={() => setShowTable(!showTable)}
        >
          {showTable ? "Show chart" : "Show table"}
        </Button>
      </div>
      {controls && (
        <div className="flex flex-wrap items-center gap-3 text-sm">
          {controls}
        </div>
      )}
      {note && <p className="text-xs text-muted-foreground">{note}</p>}
      {showTable ? (
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b">
                {table.columns.map((c) => (
                  <th key={c} className="p-1 text-left font-medium">
                    {c}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {table.rows.map((row, i) => (
                <tr key={i} className="border-b last:border-0">
                  {row.map((v, j) => (
                    <td
                      key={j}
                      className="p-1 font-mono break-all"
                      title={typeof v === "number" ? String(v) : undefined}
                    >
                      {fmt(v)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <div className="h-64 w-full">{children}</div>
      )}
    </figure>
  )
}
