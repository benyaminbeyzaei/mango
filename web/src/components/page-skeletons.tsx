// Loading placeholders shaped like the pages they stand in for, so nothing
// moves when the data arrives.
import { Skeleton } from "@/components/ui/skeleton"

export function Loading({
  label,
  children,
}: {
  label: string
  children: React.ReactNode
}) {
  return (
    <div aria-busy="true" aria-live="polite" className="flex flex-col gap-4">
      <span className="sr-only">{label}</span>
      {children}
    </div>
  )
}

// Text-height lines: h-5 matches text-sm line height, h-8 a text-2xl heading.
export const Line = ({ className = "w-48" }: { className?: string }) => (
  <Skeleton className={`h-5 ${className}`} />
)
export const Heading = ({ className = "w-56" }: { className?: string }) => (
  <Skeleton className={`h-8 ${className}`} />
)

export function TableSkeleton({
  rows = 8,
  cols = 5,
  rowHeight = "h-9",
}: {
  rows?: number
  cols?: number
  rowHeight?: string
}) {
  return (
    <div className="flex flex-col">
      <div className="flex gap-4 border-b py-2.5">
        {Array.from({ length: cols }, (_, c) => (
          <Skeleton
            key={c}
            className={`h-4 ${c === 0 ? "flex-[3]" : "flex-1"}`}
          />
        ))}
      </div>
      {Array.from({ length: rows }, (_, r) => (
        <div
          key={r}
          className={`flex items-center gap-4 border-b ${rowHeight}`}
        >
          {Array.from({ length: cols }, (_, c) => (
            <Skeleton
              key={c}
              className={`h-4 ${c === 0 ? "flex-[3]" : "flex-1"}`}
            />
          ))}
        </div>
      ))}
    </div>
  )
}

// Any page whose code is still downloading (the router's pending component).
export function PageSkeleton() {
  return (
    <Loading label="Loading page">
      <Heading />
      <Line className="w-full max-w-xl" />
      <TableSkeleton />
    </Loading>
  )
}

// A chart card: the title row, then the 16rem plot area that ChartFrame reserves.
export function ChartSkeleton() {
  return (
    <div className="flex flex-col gap-2 rounded-lg border p-3">
      <div className="flex justify-between">
        <Line className="w-56" />
        <Skeleton className="h-6 w-20" />
      </div>
      <Skeleton className="h-64 w-full" />
    </div>
  )
}

// The metric cards of a run page: a heading and a few label/value lines each.
export function CardsSkeleton({ cards = 5 }: { cards?: number }) {
  return (
    <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
      {Array.from({ length: cards }, (_, i) => (
        <div key={i} className="flex flex-col gap-1.5 rounded-lg border p-3">
          <Skeleton className="h-4 w-20" />
          {Array.from({ length: 4 }, (_, j) => (
            <div key={j} className="flex justify-between gap-3">
              <Line className="w-28" />
              <Line className="w-12" />
            </div>
          ))}
        </div>
      ))}
    </div>
  )
}
