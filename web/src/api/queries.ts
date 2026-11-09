// Shared API reads and the two query calls used by the interactive page.
import { useQuery } from "@tanstack/react-query"

import { api } from "./client"
import type { components } from "./schema"

export type RunConfig = components["schemas"]["RunConfig"]
export type QueryResponse = components["schemas"]["QueryResponse"]
export type ConfigEntry = components["schemas"]["ConfigEntry"]

export function useConfigs() {
  return useQuery({
    queryKey: ["configs"],
    queryFn: async () => {
      const { data, error } = await api.GET("/api/configs")
      if (error) throw new Error("could not load /api/configs")
      return data
    },
  })
}

export function useModels() {
  return useQuery({
    queryKey: ["models"],
    queryFn: async () => {
      const { data, error } = await api.GET("/api/models")
      if (error) throw new Error("could not load /api/models")
      return data
    },
  })
}

// FastAPI errors: {detail: "message"} for 400/502, {detail: [{loc, msg}]} for 422.
function errorText(status: number, body: unknown) {
  const detail = (body as { detail?: unknown } | null)?.detail
  if (typeof detail === "string") return detail
  if (Array.isArray(detail))
    return detail
      .map((d: { loc?: unknown[]; msg?: string }) =>
        [d.loc?.slice(1).join("."), d.msg].filter(Boolean).join(": ")
      )
      .join("; ")
  return `HTTP ${status}`
}

async function read(res: Response): Promise<QueryResponse> {
  const body = await res.json().catch(() => null)
  if (!res.ok) throw new Error(errorText(res.status, body))
  return body as QueryResponse
}

// No retries here: a request can take minutes, and repeating it would cost another ELM call.
export async function askText(config: RunConfig, text: string) {
  return read(
    await fetch("/api/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ config, input: { text } }),
    })
  )
}

export async function askAudio(config: RunConfig, audio: Blob) {
  const form = new FormData()
  form.append("config", JSON.stringify(config))
  const ext = audio.type.includes("mp4") ? "mp4" : "webm"
  form.append("audio", audio, `question.${ext}`)
  return read(await fetch("/api/query/audio", { method: "POST", body: form }))
}
