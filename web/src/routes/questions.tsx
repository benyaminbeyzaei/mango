import { useQuery } from "@tanstack/react-query"
import { createFileRoute, useNavigate } from "@tanstack/react-router"
import { SendIcon } from "lucide-react"

import { api } from "@/api/client"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"

const PAGE = 20
type Split = "dev" | "practice"
type Search = { split?: Split; page?: number }

export const Route = createFileRoute("/questions")({
  validateSearch: (s: Record<string, unknown>): Search => ({
    split: s.split === "practice" ? "practice" : undefined,
    page: Number(s.page) > 1 ? Math.floor(Number(s.page)) : undefined,
  }),
  component: Questions,
})

const norm = (t: string) => t.trim().toLowerCase()

function Questions() {
  const { split = "dev", page = 1 } = Route.useSearch()
  const navigate = useNavigate({ from: Route.fullPath })
  const questions = useQuery({
    queryKey: ["questions", split, page],
    queryFn: async () => {
      const { data, error } = await api.GET("/api/questions", {
        params: { query: { split, offset: (page - 1) * PAGE, limit: PAGE } },
      })
      if (error) throw new Error("could not load /api/questions")
      return data
    },
  })
  const items = questions.data?.items ?? []
  const differs = (w: string | null, c: string | null) =>
    w !== null && c !== null && norm(w) !== norm(c)
  const nDiffer = items.filter((q) => differs(q.whisper1, q.course)).length
  const pages = Math.max(1, Math.ceil((questions.data?.total ?? 0) / PAGE))
  const go = (s: Search) => navigate({ search: (old) => ({ ...old, ...s }) })

  return (
    <div className="flex flex-col gap-4">
      <h1 className="font-heading text-2xl font-medium">Questions</h1>
      <p className="text-sm text-muted-foreground">
        The benchmark's spoken questions with the whisper-1 transcript and the
        transcript from our own ASR (the course model on our CUDA kernels).
        Words that only one transcript has are highlighted. A button sends that
        transcript to the Interactive page as a typed question.
      </p>
      <div className="flex flex-wrap items-center gap-2">
        {(["dev", "practice"] as const).map((s) => (
          <Button
            key={s}
            size="sm"
            variant={s === split ? "default" : "outline"}
            onClick={() =>
              go({ split: s === "dev" ? undefined : s, page: undefined })
            }
          >
            {s}
          </Button>
        ))}
        {questions.data && (
          <span className="text-sm text-muted-foreground">
            {questions.data.total} questions; {nDiffer} of {items.length} on
            this page differ
          </span>
        )}
      </div>
      {questions.isError && <p>{questions.error.message}</p>}
      {questions.isPending && <p>Loading…</p>}

      {items.map((q) => (
        <section
          key={q.query_id}
          className="flex flex-col gap-2 rounded-lg border p-3"
        >
          <div className="flex flex-wrap items-center gap-2 text-sm">
            <span className="font-mono">{q.query_id}</span>
            <span className="text-muted-foreground">
              {q.duration_seconds} s
            </span>
            {differs(q.whisper1, q.course) && (
              <Badge variant="secondary">differs</Badge>
            )}
          </div>
          <audio
            controls
            preload="none"
            src={`/api/questions/${q.query_id}/audio`}
            className="h-10 max-w-full"
            aria-label={`Audio of ${q.query_id}`}
          />
          <div className="grid gap-3 md:grid-cols-2">
            <Transcript label="whisper-1" text={q.whisper1} other={q.course} />
            <Transcript label="our ASR" text={q.course} other={q.whisper1} />
          </div>
        </section>
      ))}

      {pages > 1 && (
        <div className="flex items-center gap-2 text-sm">
          <Button
            size="sm"
            variant="outline"
            disabled={page <= 1}
            onClick={() => go({ page: page - 1 > 1 ? page - 1 : undefined })}
          >
            Previous
          </Button>
          <span>
            page {page} of {pages}
          </span>
          <Button
            size="sm"
            variant="outline"
            disabled={page >= pages}
            onClick={() => go({ page: page + 1 })}
          >
            Next
          </Button>
        </div>
      )}
    </div>
  )
}

function Transcript({
  label,
  text,
  other,
}: {
  label: string
  text: string | null
  other: string | null
}) {
  const navigate = useNavigate()
  const otherWords = new Set(norm(other ?? "").split(/\s+/))
  return (
    <div className="flex flex-col items-start gap-2">
      <h2 className="text-xs font-medium tracking-wide text-muted-foreground uppercase">
        {label}
      </h2>
      {text === null ? (
        <p className="text-sm text-muted-foreground">not transcribed yet</p>
      ) : (
        <>
          <p className="text-sm">
            {text.split(/(\s+)/).map((w, i) =>
              other !== null && w.trim() && !otherWords.has(w.toLowerCase()) ? (
                <mark key={i} className="rounded bg-primary/15 text-foreground">
                  {w}
                </mark>
              ) : (
                w
              )
            )}
          </p>
          <Button
            size="sm"
            variant="outline"
            onClick={() =>
              navigate({ to: "/interactive", search: { ask: text } })
            }
          >
            <SendIcon /> Ask with {label} transcript
          </Button>
        </>
      )}
    </div>
  )
}
