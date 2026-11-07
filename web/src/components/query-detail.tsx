// What happened during one query: the "What ran" line and the full-detail panel.
// Written for the interactive page; plan 011 (results) can reuse QueryDetail.
import type { ReactNode } from "react"

import type { QueryResponse, RunConfig } from "@/api/queries"

type Span = {
  name: string
  start_ms: number
  duration_ms: number
  attrs: Record<string, unknown>
}
type Cost = {
  calls: number
  prompt_tokens: number
  completion_tokens: number
  audio_seconds: number
}
type Trace = {
  spans?: Span[]
  counters?: Record<string, number>
  cost?: Record<string, Cost>
  resources?: { wall_ms?: number; cpu_seconds?: number; peak_rss_mb?: number }
}

const ms = (n: number) =>
  n >= 1000 ? `${(n / 1000).toFixed(1)} s` : `${Math.round(n)} ms`

// Pipeline stages in order: the implementation name from the config, the time from the trace span.
export function WhatRan({
  config,
  input,
  trace,
}: {
  config: RunConfig
  input: "text" | "voice"
  trace: Trace
}) {
  const span = (name: string) => trace.spans?.find((s) => s.name === name)
  // a typed question skips ASR; an uploaded recording never hits the transcript cache, so it uses whisper-1
  const asr =
    input === "text"
      ? "typed"
      : config.asr === "cached"
        ? "whisper1"
        : config.asr
  const encoder =
    config.mismatch_policy === "error"
      ? config.query_encoder
      : `${config.query_encoder} (${config.mismatch_policy})`
  const stages: [string, string, Span | undefined][] = [
    ["ASR", asr, span("asr")],
    ["encoder", encoder, span("encode")],
    ["index", config.corpus_index, undefined],
    ["retriever", `${config.retriever} k=${config.k}`, span("retrieve")],
    ["context", config.context_builder, span("context")],
    ["generator", config.generator_model.split("/").pop()!, span("generate")],
  ]
  return (
    <p className="text-xs text-muted-foreground">
      <span className="font-medium">
        What ran{config.flow !== "standard" && ` (${config.flow})`}:{" "}
      </span>
      {stages.map(([stage, name, s], i) => (
        <span key={stage}>
          {i > 0 && " → "}
          {stage} <span className="font-mono text-foreground">{name}</span>
          {s && ` ${ms(s.duration_ms)}`}
        </span>
      ))}
    </p>
  )
}

export function QueryDetail({ response }: { response: QueryResponse }) {
  const trace = response.trace as Trace
  const wall =
    trace.resources?.wall_ms ??
    Math.max(1, ...(trace.spans ?? []).map((s) => s.start_ms + s.duration_ms))
  return (
    <details className="rounded-lg border p-2 text-sm">
      <summary className="cursor-pointer font-medium">Full detail</summary>
      <div className="mt-2 flex flex-col gap-4">
        <Section title="Transcript">
          <p>{response.transcript || "(empty)"}</p>
        </Section>

        <Section title={`Evidence (${response.retrieved.length})`}>
          <ol className="flex flex-col gap-1">
            {response.retrieved.map((hit, i) => (
              <li key={hit.record_id}>
                <details>
                  <summary className="cursor-pointer font-mono text-xs">
                    {i + 1}. {hit.record_id}{" "}
                    <span className="text-muted-foreground">
                      score {hit.score.toFixed(4)}
                    </span>
                  </summary>
                  <p className="mt-1 text-xs whitespace-pre-wrap">{hit.text}</p>
                </details>
              </li>
            ))}
          </ol>
        </Section>

        <Section title="Prompt">
          <pre className="max-h-64 overflow-auto rounded bg-muted p-2 text-xs whitespace-pre-wrap">
            {response.prompt}
          </pre>
        </Section>

        <Section title="Timeline">
          <ul className="flex flex-col gap-1">
            {trace.spans?.map((s, i) => (
              <li
                key={i}
                className="grid grid-cols-[8rem_1fr_4rem] items-center gap-2 text-xs"
              >
                <span
                  className="truncate font-mono"
                  title={JSON.stringify(s.attrs)}
                >
                  {s.name}
                </span>
                <span className="relative h-2 rounded bg-muted">
                  <span
                    className="absolute inset-y-0 rounded bg-primary"
                    style={{
                      left: `${(100 * s.start_ms) / wall}%`,
                      width: `max(2px, ${(100 * s.duration_ms) / wall}%)`,
                    }}
                  />
                </span>
                <span className="text-right">{ms(s.duration_ms)}</span>
              </li>
            ))}
          </ul>
          {trace.resources && (
            <p className="mt-1 text-xs text-muted-foreground">
              wall {ms(trace.resources.wall_ms ?? 0)}, CPU{" "}
              {(trace.resources.cpu_seconds ?? 0).toFixed(2)} s, peak RSS{" "}
              {Math.round(trace.resources.peak_rss_mb ?? 0)} MB
            </p>
          )}
        </Section>

        <Section title="Counters">
          <KeyValues rows={Object.entries(trace.counters ?? {})} />
        </Section>

        <Section title="Cost (remote calls, cache hits are free)">
          {Object.keys(trace.cost ?? {}).length === 0 ? (
            <p className="text-xs text-muted-foreground">no remote calls</p>
          ) : (
            <KeyValues
              rows={Object.entries(trace.cost ?? {}).map(([model, c]) => [
                model,
                `${c.calls} calls, ${c.prompt_tokens} prompt + ${c.completion_tokens} completion tokens` +
                  (c.audio_seconds
                    ? `, ${c.audio_seconds.toFixed(1)} s audio`
                    : ""),
              ])}
            />
          )}
        </Section>
      </div>
    </details>
  )
}

function Section({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="flex flex-col gap-1">
      <h3 className="text-xs font-medium tracking-wide text-muted-foreground uppercase">
        {title}
      </h3>
      {children}
    </section>
  )
}

function KeyValues({ rows }: { rows: [string, unknown][] }) {
  return (
    <dl className="grid grid-cols-[minmax(0,auto)_1fr] gap-x-3 text-xs">
      {rows.map(([k, v]) => (
        <div key={k} className="contents">
          <dt className="font-mono break-all">{k}</dt>
          <dd>{String(v)}</dd>
        </div>
      ))}
    </dl>
  )
}
