// Picks a cached answer in one dialog: first the experiment (one of the three full evaluations), then one
// of its questions. Native radio inputs, so arrow keys, Tab and screen readers work as usual.
import { useState } from "react"
import { SearchIcon } from "lucide-react"

import { Button } from "@/components/ui/button"
import {
  Dialog,
  DialogClose,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { Input } from "@/components/ui/input"
import type { CachedPick, CachedRun } from "@/lib/cached"
import { configLabel } from "@/lib/labels"
import { fmt } from "@/lib/results"

const OPTION =
  "flex cursor-pointer gap-3 rounded-lg border p-3 outline-none has-checked:border-primary has-checked:bg-primary/5 has-focus-visible:ring-3 has-focus-visible:ring-ring/50"

export function CachedQuestionDialog({
  runs,
  open,
  onOpenChange,
  onPick,
}: {
  runs: CachedRun[]
  open: boolean
  onOpenChange: (open: boolean) => void
  onPick: (pick: CachedPick) => void
}) {
  const [runId, setRunId] = useState(runs[0]?.run_id ?? "")
  const [queryId, setQueryId] = useState("")
  const [filter, setFilter] = useState("")
  const run = runs.find((r) => r.run_id === runId) ?? runs[0]
  const words = filter.toLowerCase().split(/\s+/).filter(Boolean)
  const shown =
    run?.questions.filter((q) =>
      words.every((w) => q.transcript.toLowerCase().includes(w))
    ) ?? []
  const question = run?.questions.find((q) => q.query_id === queryId)

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="flex max-h-[90svh] flex-col sm:max-w-2xl">
        <DialogHeader>
          <DialogTitle>Choose a cached question</DialogTitle>
          <DialogDescription>
            Answers stored from three full evaluations on the{" "}
            {run?.questions.length ?? 125} test questions. Pick an experiment,
            then a question; the stored answer opens with its evidence, prompt
            and timings.
          </DialogDescription>
        </DialogHeader>

        <form
          id="cached-question"
          className="flex min-h-0 flex-col gap-4"
          onSubmit={(e) => {
            e.preventDefault()
            if (!run || !question) return
            onPick({
              runId: run.run_id,
              queryId: question.query_id,
              transcript: question.transcript,
            })
            onOpenChange(false)
          }}
        >
          <fieldset className="flex flex-col gap-2">
            <legend className="mb-2 text-sm font-medium">1. Experiment</legend>
            {runs.map((r) => (
              <label key={r.run_id} className={OPTION}>
                <input
                  type="radio"
                  name="run"
                  value={r.run_id}
                  checked={r.run_id === run?.run_id}
                  onChange={() => setRunId(r.run_id)}
                  className="mt-0.5 accent-primary"
                />
                <span className="flex flex-col gap-0.5">
                  <span className="font-medium">
                    {configLabel(r.config_name)}
                  </span>
                  <span className="text-xs text-muted-foreground">
                    answer score {fmt(r.answer_score)} · NDCG@10 {fmt(r.ndcg10)}{" "}
                    · transcripts from {r.transcripts}
                  </span>
                </span>
              </label>
            ))}
          </fieldset>

          <fieldset className="flex min-h-0 flex-col gap-2">
            <legend className="mb-2 text-sm font-medium">2. Question</legend>
            <div className="relative">
              <SearchIcon className="pointer-events-none absolute top-1/2 left-2.5 size-4 -translate-y-1/2 text-muted-foreground" />
              <Input
                aria-label="Filter questions"
                placeholder="Filter by words…"
                value={filter}
                onChange={(e) => setFilter(e.target.value)}
                className="pl-8"
              />
            </div>
            <p className="text-xs text-muted-foreground" aria-live="polite">
              {shown.length} of {run?.questions.length ?? 0} questions
            </p>
            <div className="flex max-h-64 min-h-0 flex-col gap-1 overflow-y-auto rounded-lg border p-1">
              {shown.map((q) => (
                <label
                  key={q.query_id}
                  className="flex cursor-pointer gap-2 rounded-md px-2 py-1.5 text-sm hover:bg-muted has-checked:bg-muted has-checked:font-medium has-focus-visible:ring-3 has-focus-visible:ring-ring/50"
                >
                  <input
                    type="radio"
                    name="question"
                    value={q.query_id}
                    checked={q.query_id === queryId}
                    onChange={() => setQueryId(q.query_id)}
                    className="mt-1 accent-primary"
                  />
                  {q.transcript}
                </label>
              ))}
              {shown.length === 0 && (
                <p className="p-2 text-sm text-muted-foreground">
                  No question has all these words.
                </p>
              )}
            </div>
          </fieldset>
        </form>

        <DialogFooter>
          <DialogClose render={<Button variant="outline" />}>
            Cancel
          </DialogClose>
          <Button type="submit" form="cached-question" disabled={!question}>
            Show answer
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
