// Mean time per pipeline stage, stacked into one bar per run.
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

import { label } from "@/lib/results"

import { AXIS, ChartFrame, LEGEND, PALETTE, TOOLTIP } from "./chart-frame"
import type { Series } from "./grouped-bars"

export function LatencyBreakdown({
  title,
  note,
  runs,
  stages,
  value,
}: {
  title: string
  note?: ReactNode
  runs: Series[]
  stages: string[]
  value: (run: number, stage: string) => number | null
}) {
  const data = runs.map((r, i) =>
    Object.fromEntries([
      ["run", r.label],
      ...stages.map((s) => [s, value(i, s)]),
    ])
  )
  return (
    <ChartFrame
      title={title}
      note={note}
      table={{
        columns: ["run", ...stages.map((s) => label(`latency_ms.${s}`))],
        rows: runs.map((r, i) => [r.label, ...stages.map((s) => value(i, s))]),
      }}
    >
      <ResponsiveContainer>
        <BarChart
          data={data}
          layout="vertical"
          margin={{ left: 8, bottom: 16 }}
        >
          <CartesianGrid horizontal={false} stroke="var(--border)" />
          <XAxis
            type="number"
            tick={AXIS}
            label={{ ...AXIS, value: "mean time (ms)", position: "bottom" }}
          />
          <YAxis type="category" dataKey="run" tick={AXIS} width={96} />
          <Tooltip {...TOOLTIP} cursor={{ fill: "var(--muted)" }} />
          <Legend {...LEGEND} verticalAlign="top" />
          {stages.map((s, i) => (
            <Bar
              key={s}
              dataKey={s}
              name={label(`latency_ms.${s}`)}
              stackId="stages"
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
