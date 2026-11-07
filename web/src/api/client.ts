import createClient from "openapi-fetch"

import type { paths } from "./schema"

const ADMIN_TOKEN_KEY = "adminToken"

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
    const token = getAdminToken()
    if (token) request.headers.set("X-Admin-Token", token)
    return request
  },
})
