// One metric for every query, one line per run, queries in subset order.
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts"

import type { ReactNode } from "react"

import { AXIS, ChartFrame, LEGEND, PALETTE, TOOLTIP } from "./chart-frame"
import type { Series } from "./grouped-bars"

export function PerQueryStrip({
  title,
  note,
  runs,
  queryIds,
  value,
  yLabel,
}: {
  title: string
  note?: ReactNode
  runs: Series[]
  queryIds: string[]
  value: (run: number, queryId: string) => number | null
  yLabel: string
}) {
  const data = queryIds.map((q) =>
    Object.fromEntries([
      ["query", q],
      ...runs.map((r, i) => [r.id, value(i, q)]),
    ])
  )
  return (
    <ChartFrame
      title={title}
      note={note}
      table={{
        columns: ["query", ...runs.map((r) => r.label)],
        rows: queryIds.map((q) => [q, ...runs.map((_, i) => value(i, q))]),
      }}
    >
      <ResponsiveContainer>
        <LineChart data={data} margin={{ left: 8, bottom: 16 }}>
          <CartesianGrid vertical={false} stroke="var(--border)" />
          <XAxis
            dataKey="query"
            tick={false}
            label={{
              ...AXIS,
              value: "query (subset order)",
              position: "bottom",
            }}
          />
          <YAxis
            tick={AXIS}
            width={48}
            label={{
              ...AXIS,
              value: yLabel,
              angle: -90,
              position: "insideLeft",
            }}
          />
          <Tooltip {...TOOLTIP} />
          <Legend {...LEGEND} verticalAlign="top" />
          {runs.map((r, i) => (
            <Line
              key={r.id}
              dataKey={r.id}
              name={r.label}
              isAnimationActive={false}
              stroke={PALETTE[i % PALETTE.length]}
              strokeWidth={2}
              dot={{ r: 3 }}
              connectNulls={false}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </ChartFrame>
  )
}
