import { useQuery } from "@tanstack/react-query"
import { createFileRoute, useNavigate } from "@tanstack/react-router"
import { SendIcon } from "lucide-react"

import { api, audioUrl } from "@/api/client"
import { Line, Loading } from "@/components/page-skeletons"
import { Skeleton } from "@/components/ui/skeleton"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { fmt } from "@/lib/results"

const PAGE = 20
type Split = "test" | "practice"
type Match = "all" | "differ" | "same"
type Search = { split?: Split; match?: Match; page?: number }

export const Route = createFileRoute("/questions")({
  validateSearch: (s: Record<string, unknown>): Search => ({
    split: s.split === "practice" ? "practice" : undefined,
    match: s.match === "differ" || s.match === "same" ? s.match : undefined,
    page: Number(s.page) > 1 ? Math.floor(Number(s.page)) : undefined,
  }),
  component: Questions,
})

// The same rule as the API: lower-case words without punctuation.
const key = (w: string) => (w.toLowerCase().match(/[a-z0-9']+/g) ?? []).join("")

// Word-level diff (longest common subsequence on the keys): for each word of `a`
// and of `b`, whether the other transcript has it at the same place in the sentence.
function diffWords(a: string[], b: string[]) {
  const ka = a.map(key)
  const kb = b.map(key)
  const L = Array.from({ length: a.length + 1 }, () =>
    new Array<number>(b.length + 1).fill(0)
  )
  for (let i = a.length - 1; i >= 0; i--)
    for (let j = b.length - 1; j >= 0; j--)
      L[i][j] =
        ka[i] === kb[j]
          ? L[i + 1][j + 1] + 1
          : Math.max(L[i + 1][j], L[i][j + 1])
  // punctuation-only tokens never count as a difference
  const inA = ka.map((k) => k === "")
  const inB = kb.map((k) => k === "")
  for (let i = 0, j = 0; i < a.length && j < b.length;) {
    if (ka[i] === kb[j]) {
      inA[i++] = true
      inB[j++] = true
    } else if (L[i + 1][j] >= L[i][j + 1]) i++
    else j++
  }
  return [inA, inB]
}

const tokens = (t: string | null) => (t ?? "").split(/\s+/).filter(Boolean)

const ONLY = {
  whisper:
    "rounded-sm bg-amber-400/30 px-0.5 underline decoration-amber-600 decoration-2 underline-offset-4 dark:decoration-amber-400",
  ours: "rounded-sm bg-sky-400/30 px-0.5 underline decoration-sky-600 decoration-dotted decoration-2 underline-offset-4 dark:decoration-sky-400",
}

function Questions() {
  const { split = "test", match = "all", page = 1 } = Route.useSearch()
  const navigate = useNavigate({ from: Route.fullPath })
  const questions = useQuery({
    queryKey: ["questions", split, match, page],
    queryFn: async () => {
      const { data, error } = await api.GET("/api/questions", {
        params: {
          query: { split, match, offset: (page - 1) * PAGE, limit: PAGE },
        },
      })
      if (error) throw new Error("could not load /api/questions")
      return data
    },
  })
  const items = questions.data?.items ?? []
  const pages = Math.max(1, Math.ceil((questions.data?.total ?? 0) / PAGE))
  const go = (s: Search) => navigate({ search: (old) => ({ ...old, ...s }) })

  return (
    <div className="flex flex-col gap-4">
      <h1 className="font-heading text-2xl font-medium">Questions</h1>
      <p className="text-sm text-muted-foreground">
        The benchmark's spoken questions with the whisper-1 transcript and the
        transcript from our own ASR (the course model on our CUDA kernels). The
        two are compared word by word, ignoring case and punctuation; a button
        sends either transcript to the Interactive page as a typed question.
      </p>
      <div className="flex flex-wrap items-center gap-4 text-sm">
        <span className="flex items-center gap-1.5">
          <mark className={ONLY.whisper}>word</mark> only in whisper-1
        </span>
        <span className="flex items-center gap-1.5">
          <mark className={ONLY.ours}>word</mark> only in our ASR
        </span>
      </div>
      <div className="flex flex-wrap items-center gap-2">
        {(["test", "practice"] as const).map((s) => (
          <Button
            key={s}
            size="sm"
            variant={s === split ? "default" : "outline"}
            onClick={() =>
              go({ split: s === "test" ? undefined : s, page: undefined })
            }
          >
            {s}
          </Button>
        ))}
        <span className="mx-1 h-5 w-px bg-border" aria-hidden />
        {(["all", "differ", "same"] as const).map((m) => (
          <Button
            key={m}
            size="sm"
            variant={m === match ? "default" : "outline"}
            onClick={() =>
              go({ match: m === "all" ? undefined : m, page: undefined })
            }
          >
            {m === "all" ? "all" : m === "differ" ? "different" : "same"}
            {questions.data && m !== "all" && (
              <span className="opacity-70">{questions.data[m]}</span>
            )}
          </Button>
        ))}
        {questions.data ? (
          <span className="text-sm text-muted-foreground">
            {questions.data.total} questions
          </span>
        ) : (
          <Line className="w-24" />
        )}
      </div>
      {questions.isError && <p>{questions.error.message}</p>}
      {questions.isPending && (
        <Loading label="Loading questions">
          {Array.from({ length: 4 }, (_, i) => (
            <div key={i} className="flex flex-col gap-2 rounded-lg border p-3">
              <Line className="w-56" />
              <Skeleton className="h-10 w-72 max-w-full" />
              <div className="grid gap-3 md:grid-cols-2">
                <Skeleton className="h-16" />
                <Skeleton className="h-16" />
              </div>
            </div>
          ))}
        </Loading>
      )}
      {questions.data && items.length === 0 && (
        <p className="text-sm text-muted-foreground">No questions match.</p>
      )}

      {items.map((q) => {
        const w = tokens(q.whisper1),
          c = tokens(q.course)
        const both = q.whisper1 !== null && q.course !== null
        const [inW, inC] = both ? diffWords(w, c) : [[], []]
        const changed =
          inW.filter((x) => !x).length + inC.filter((x) => !x).length
        return (
          <section
            key={q.query_id}
            className="flex flex-col gap-2 rounded-lg border p-3"
          >
            <div className="flex flex-wrap items-center gap-2 text-sm">
              <span className="font-mono">{q.query_id}</span>
              <span className="text-muted-foreground">
                {fmt(q.duration_seconds)} s
              </span>
              {both &&
                (changed ? (
                  <Badge variant="secondary">
                    {changed} word{changed > 1 ? "s" : ""} differ
                  </Badge>
                ) : (
                  <Badge variant="outline">same</Badge>
                ))}
            </div>
            <audio
              controls
              preload="none"
              src={audioUrl(q.query_id)}
              className="h-10 max-w-full"
              aria-label={`Audio of ${q.query_id}`}
            />
            <div className="grid gap-3 md:grid-cols-2">
              <Transcript
                label="whisper-1"
                text={q.whisper1}
                words={w}
                common={inW}
                mark={ONLY.whisper}
              />
              <Transcript
                label="our ASR"
                text={q.course}
                words={c}
                common={inC}
                mark={ONLY.ours}
              />
            </div>
          </section>
        )
      })}

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
  words,
  common,
  mark,
}: {
  label: string
  text: string | null
  words: string[]
  common: boolean[] // empty when there is nothing to compare with
  mark: string
}) {
  const navigate = useNavigate()
  return (
    <div className="flex flex-col items-start gap-2">
      <h2 className="text-xs font-medium tracking-wide text-muted-foreground uppercase">
        {label}
      </h2>
      {text === null ? (
        <p className="text-sm text-muted-foreground">not transcribed yet</p>
      ) : (
        <>
          <p className="text-sm leading-7">
            {words.map((w, i) => (
              <span key={i}>
                {i > 0 && " "}
                {common.length && !common[i] ? (
                  <mark className={`${mark} text-foreground`}>{w}</mark>
                ) : (
                  w
                )}
              </span>
            ))}
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
