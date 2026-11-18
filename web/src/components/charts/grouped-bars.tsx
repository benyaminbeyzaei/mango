// Several metrics side by side, one bar per run.
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts"

import type { ReactNode } from "react"

import { AXIS, ChartFrame, LEGEND, PALETTE, TOOLTIP } from "./chart-frame"

export type Series = { id: string; label: string }

export function GroupedBars({
  title,
  note,
  controls,
  runs,
  metrics,
  value,
  yLabel,
}: {
  title: string
  note?: ReactNode
  controls?: ReactNode
  runs: Series[]
  metrics: string[]
  value: (run: number, metric: string) => number | null
  yLabel: string
}) {
  const data = metrics.map((metric) =>
    Object.fromEntries([
      ["metric", metric],
      ...runs.map((r, i) => [r.id, value(i, metric)]),
    ])
  )
  return (
    <ChartFrame
      title={title}
      note={note}
      controls={controls}
      table={{
        columns: ["metric", ...runs.map((r) => r.label)],
        rows: metrics.map((m) => [m, ...runs.map((_, i) => value(i, m))]),
      }}
    >
      <ResponsiveContainer>
        <BarChart data={data} margin={{ left: 8, bottom: 8 }}>
          <CartesianGrid vertical={false} stroke="var(--border)" />
          <XAxis dataKey="metric" tick={AXIS} />
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
          <Tooltip {...TOOLTIP} cursor={{ fill: "var(--muted)" }} />
          <Legend {...LEGEND} />
          {runs.map((r, i) => (
            <Bar
              key={r.id}
              dataKey={r.id}
              name={r.label}
              isAnimationActive={false}
              fill={PALETTE[i % PALETTE.length]}
              stroke="var(--border)"
            />
          ))}
        </BarChart>
      </ResponsiveContainer>
    </ChartFrame>
  )
}
