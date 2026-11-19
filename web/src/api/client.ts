import createClient from "openapi-fetch"

import type { paths } from "./schema"

const ADMIN_TOKEN_KEY = "adminToken"

// The static build (VITE_STATIC=1) has no API: every read is a JSON file written by `mlsrag export-web`.
export const STATIC = import.meta.env.VITE_STATIC === "1"

// /api/results/R/queries?offset=100&limit=100 -> /static-api/results/R/queries/limit-100_offset-100.json
// (static hosting ignores query strings). rag/extensions/platform/mlsrag/export_web.py writes the same paths.
export function staticPath(path: string, query: URLSearchParams) {
  const pairs = [...query.entries()]
    .sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0))
    .map(([k, v]) => `${k}-${v}`)
  const rest = path.replace(/^\/api/, "")
  return `/static-api${rest}${pairs.length ? `/${pairs.join("_")}` : ""}.json`
}

export function audioUrl(queryId: string) {
  return STATIC
    ? `/static-api/questions/${queryId}/audio.wav`
    : `/api/questions/${queryId}/audio`
}

export function getAdminToken() {
  return localStorage.getItem(ADMIN_TOKEN_KEY) ?? ""
}

export function setAdminToken(token: string) {
  if (token) localStorage.setItem(ADMIN_TOKEN_KEY, token)
  else localStorage.removeItem(ADMIN_TOKEN_KEY)
}

export const api = createClient<paths>({ baseUrl: "/" })

api.use({
  onRequest({ request }) {
    if (STATIC) {
      const url = new URL(request.url)
      return new Request(
        new URL(staticPath(url.pathname, url.searchParams), url.origin)
      )
    }
    const token = getAdminToken()
    if (token) request.headers.set("X-Admin-Token", token)
    return request
  },
})
