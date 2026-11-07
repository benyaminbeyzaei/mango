// Chat history, kept only in this browser (IndexedDB). The server stores nothing about users.
import { Dexie, type EntityTable } from "dexie"

import type { QueryResponse, RunConfig } from "@/api/queries"

export type Chat = {
  id: number
  title: string
  created: number
  config: RunConfig // the config last used in this chat
}

export type Message = {
  id: number
  chatId: number
  role: "user" | "assistant"
  text: string
  created: number
  config?: RunConfig // the config the question was sent with
  audio?: Blob // on voice questions: the recording (a Blob, because object URLs die on reload)
  input?: "text" | "voice" // on assistant messages: how the question was asked
  response?: QueryResponse // the full API response, trace included, so the detail panel works offline
  error?: string // the API's message when the request failed
  confirmTranscript?: boolean // voice answer hidden until the user accepts the transcript
}

export const db = new Dexie("mlsrag") as Dexie & {
  chats: EntityTable<Chat, "id">
  messages: EntityTable<Message, "id">
}

db.version(1).stores({
  chats: "++id, created",
  messages: "++id, chatId",
})

export async function deleteChat(id: number) {
  await db.transaction("rw", db.chats, db.messages, async () => {
    await db.messages.where("chatId").equals(id).delete()
    await db.chats.delete(id)
  })
}

export async function clearAll() {
  await db.transaction("rw", db.chats, db.messages, async () => {
    await db.messages.clear()
    await db.chats.clear()
  })
}

// One chat as a JSON file. Recordings are left out (they are binary); the transcript is in the response.
export async function exportChat(id: number) {
  const chat = await db.chats.get(id)
  if (!chat) return
  const messages = await db.messages.where("chatId").equals(id).toArray()
  const data = {
    chat,
    messages: messages.map((m) => ({ ...m, audio: undefined })),
  }
  const url = URL.createObjectURL(
    new Blob([JSON.stringify(data, null, 2)], { type: "application/json" })
  )
  const a = document.createElement("a")
  a.href = url
  a.download = `chat-${id}.json`
  a.click()
  URL.revokeObjectURL(url)
}

// Reads a file written by exportChat into a new chat; returns its id.
export async function importChat(file: File) {
  const { chat, messages } = JSON.parse(await file.text()) as {
    chat: Chat
    messages: Message[]
  }
  const valid =
    typeof chat?.title === "string" &&
    typeof chat.config?.name === "string" &&
    Array.isArray(messages) &&
    messages.every((m) =>
      m.role === "user"
        ? typeof m.text === "string"
        : m.role === "assistant" &&
          (typeof m.error === "string" ||
            (typeof m.response?.transcript === "string" &&
              Array.isArray(m.response.retrieved)))
    )
  if (!valid) throw new Error("not a chat exported from this page")
  return db.transaction("rw", db.chats, db.messages, async () => {
    const chatId = await db.chats.add({ ...chat, id: undefined })
    await db.messages.bulkAdd(
      messages.map((m) => ({ ...m, id: undefined, chatId }))
    )
    return chatId
  })
}
