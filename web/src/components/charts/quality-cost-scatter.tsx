// One point per run: a quality metric against a cost.
import {
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts"

import type { ReactNode } from "react"

import { AXIS, ChartFrame, LEGEND, PALETTE, TOOLTIP } from "./chart-frame"

type Point = { label: string; x: number | null; y: number | null }

export function QualityCostScatter({
  title,
  note,
  controls,
  points,
  xLabel,
  yLabel,
}: {
  title: string
  note?: ReactNode
  controls?: ReactNode
  points: Point[]
  xLabel: string
  yLabel: string
}) {
  return (
    <ChartFrame
      title={title}
      note={note}
      controls={controls}
      table={{
        columns: ["run", xLabel, yLabel],
        rows: points.map((p) => [p.label, p.x, p.y]),
      }}
    >
      <ResponsiveContainer>
        <ScatterChart margin={{ left: 8, bottom: 16 }}>
          <CartesianGrid stroke="var(--border)" />
          <XAxis
            type="number"
            dataKey="x"
            name={xLabel}
            padding={{ left: 12, right: 12 }}
            tick={AXIS}
            label={{ ...AXIS, value: xLabel, position: "bottom" }}
          />
          <YAxis
            type="number"
            dataKey="y"
            name={yLabel}
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
          {points.map((p, i) => (
            <Scatter
              key={p.label}
              name={p.label}
              data={p.x == null || p.y == null ? [] : [p]}
              isAnimationActive={false}
              fill={PALETTE[i % PALETTE.length]}
              stroke="var(--foreground)"
            />
          ))}
        </ScatterChart>
      </ResponsiveContainer>
    </ChartFrame>
  )
}
