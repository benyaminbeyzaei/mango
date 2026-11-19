import { useEffect, useRef, useState } from "react"
import { createFileRoute, useNavigate } from "@tanstack/react-router"
import { useLiveQuery } from "dexie-react-hooks"
import {
  ListChecksIcon,
  MessagesSquareIcon,
  MicIcon,
  PlusIcon,
  SendIcon,
  SlidersHorizontalIcon,
  SquareIcon,
  XIcon,
} from "lucide-react"

import { STATIC } from "@/api/client"
import { askAudio, askText, useConfigs, type RunConfig } from "@/api/queries"
import { CachedQuestionDialog } from "@/components/cached-question-dialog"
import { ChatList } from "@/components/chat-list"
import { ConfigChips, ConfigPicker } from "@/components/config-picker"
import { Markdown } from "@/components/markdown"
import { NoticeDialog } from "@/components/notice-dialog"
import { QueryDetail, WhatRan } from "@/components/query-detail"
import { Button } from "@/components/ui/button"
import {
  Dialog,
  DialogClose,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog"
import {
  Sheet,
  SheetContent,
  SheetTitle,
  SheetTrigger,
} from "@/components/ui/sheet"
import { Skeleton } from "@/components/ui/skeleton"
import { askCached, useCachedRuns, type CachedPick } from "@/lib/cached"
import { db, type Message } from "@/lib/db"
import { fetchRun } from "@/lib/results"
import { getConfirmTranscript, getDefaultPreset } from "@/lib/settings"
import { useRecorder } from "@/lib/use-recorder"

// ?ask=<text> (from the Questions page) asks that text once, as if typed here.
// In the static build, ?run=<run id>&q=<query id> shows that cached answer once instead.
type Search = { ask?: string; run?: string; q?: string }
const text = (v: unknown) => (typeof v === "string" && v.trim() ? v : undefined)

export const Route = createFileRoute("/interactive")({
  validateSearch: (s: Record<string, unknown>): Search => ({
    ask: text(s.ask),
    run: text(s.run),
    q: text(s.q),
  }),
  component: Interactive,
})

function Interactive() {
  const configs = useConfigs()
  const chats = useLiveQuery(
    () => db.chats.orderBy("created").reverse().toArray(),
    []
  )
  const [chatId, setChatId] = useState<number | null>(null)
  const messages = useLiveQuery(
    () =>
      chatId === null
        ? []
        : db.messages.where("chatId").equals(chatId).sortBy("id"),
    [chatId]
  )
  const [picked, setPicked] = useState<RunConfig | null>(null)
  const [text, setText] = useState("")
  const [waitingSince, setWaitingSince] = useState<number | null>(null)
  const [sheetOpen, setSheetOpen] = useState(false)
  const end = useRef<HTMLDivElement>(null)
  const sending = useRef(false)

  // static build: cached answers only, picked in a dialog
  const cachedRuns = useCachedRuns()
  const [pickerOpen, setPickerOpen] = useState(false)
  const { ask, run, q } = Route.useSearch()
  const navigate = useNavigate({ from: Route.fullPath })

  const preset =
    configs.data?.find((c) => c.name === getDefaultPreset()) ??
    configs.data?.[0]
  const shownChat = chats?.find((c) => c.id === chatId)
  const config = STATIC
    ? (shownChat?.config ?? null)
    : (picked ?? preset?.config ?? null)

  useEffect(() => {
    end.current?.scrollIntoView({ block: "end" })
  }, [messages?.length, waitingSince])

  // null starts a new question and keeps the chosen config; a past question brings back its config
  function selectChat(id: number | null) {
    setChatId(id)
    setSheetOpen(false)
    const chat = chats?.find((c) => c.id === id)
    if (chat) setPicked(chat.config)
  }

  // One request at a time and never retried: the server answers one query at a time and a retry costs ELM calls.
  // Each question is its own entry in the history (no chat context).
  async function send(
    question: { text: string } | { audio: Blob } | { cached: CachedPick }
  ) {
    if (sending.current || (!config && !("cached" in question))) return
    sending.current = true // set before the first await, so a second submit cannot slip through
    setWaitingSince(Date.now())
    const voice = "audio" in question
    const question_text =
      "text" in question
        ? question.text
        : "cached" in question
          ? question.cached.transcript
          : ""
    try {
      // a cached answer is shown with the config of the run that made it
      const cfg =
        "cached" in question
          ? ((await fetchRun(question.cached.runId)).config as RunConfig)
          : config!
      const title = voice ? "Voice question" : question_text
      const id = await db.chats.add({ title, created: Date.now(), config: cfg })
      setChatId(id)
      await db.messages.add({
        chatId: id,
        role: "user",
        text: question_text,
        audio: voice ? question.audio : undefined,
        config: cfg,
        created: Date.now(),
      } as Message)
      const chatIdForReply = id
      // the chat may have been deleted while waiting: then the reply is dropped
      const addReply = (fields: Partial<Message>) =>
        db.transaction("rw", db.chats, db.messages, async () => {
          const chat = await db.chats.get(chatIdForReply)
          if (!chat) return
          await db.messages.add({
            chatId: chatIdForReply,
            role: "assistant",
            text: "",
            config: cfg,
            input: voice || "cached" in question ? "voice" : "text",
            created: Date.now(),
            ...fields,
          } as Message)
          if (fields.response && chat.title === "Voice question")
            await db.chats.update(chatIdForReply, {
              title: fields.response.transcript,
            })
        })
      try {
        const response =
          "audio" in question
            ? await askAudio(cfg, question.audio)
            : "cached" in question
              ? await askCached(question.cached, Number(cfg.k))
              : await askText(cfg, question.text)
        await addReply({
          text: response.answer ?? "",
          response,
          confirmTranscript: voice && getConfirmTranscript(),
        })
      } catch (err) {
        await addReply({
          error:
            err instanceof TypeError && !STATIC
              ? "Could not reach the API. Is `mlsrag serve` running?"
              : (err as Error).message,
        })
      }
    } finally {
      sending.current = false
      setWaitingSince(null)
    }
  }

  useEffect(() => {
    if (STATIC || !ask || !config) return
    navigate({ search: {}, replace: true }) // so a reload does not ask again
    send({ text: ask.trim() })
    // eslint-disable-next-line react-hooks/exhaustive-deps -- run once per ?ask, when a config is ready
  }, [ask, config === null])

  // static ?run=&q= (from the Questions page): show that cached answer once
  useEffect(() => {
    if (!STATIC || !run || !q || !cachedRuns.data) return
    navigate({ search: {}, replace: true }) // so a reload does not ask again
    const question = cachedRuns.data
      .find((r) => r.run_id === run)
      ?.questions.find((x) => x.query_id === q)
    if (question)
      send({
        cached: { runId: run, queryId: q, transcript: question.transcript },
      })
    // eslint-disable-next-line react-hooks/exhaustive-deps -- run once per ?run&q, when the cached list is ready
  }, [run, q, cachedRuns.data === undefined])

  const recorder = useRecorder((audio) => send({ audio }))
  const busy = waitingSince !== null || (!STATIC && !config)

  const chatList = (
    <ChatList chats={chats ?? []} activeId={chatId} onSelect={selectChat} />
  )

  return (
    <div className="grid gap-6 md:grid-cols-[15rem_minmax(0,1fr)]">
      <aside className="hidden md:block">{chatList}</aside>

      <div className="flex min-h-[70svh] min-w-0 flex-col gap-4">
        <div className="flex items-center gap-2">
          <Sheet open={sheetOpen} onOpenChange={setSheetOpen}>
            <SheetTrigger
              render={
                <Button variant="outline" size="sm" className="md:hidden" />
              }
            >
              <MessagesSquareIcon /> History
            </SheetTrigger>
            <SheetContent side="left" className="overflow-y-auto p-4">
              <SheetTitle>History</SheetTitle>
              {chatList}
            </SheetContent>
          </Sheet>
          <h1
            className="truncate font-heading text-2xl font-medium"
            title={chats?.find((c) => c.id === chatId)?.title}
          >
            {chats?.find((c) => c.id === chatId)?.title ?? "New question"}
          </h1>
        </div>

        {STATIC && (
          <>
            <NoticeDialog
              title="Cached version"
              action={{
                label: "Choose a cached question",
                onClick: () => setPickerOpen(true),
              }}
            >
              This version does not support the interactive mode, try some
              cached Q&amp;A.
            </NoticeDialog>
            {cachedRuns.data && (
              <CachedQuestionDialog
                runs={cachedRuns.data}
                open={pickerOpen}
                onOpenChange={setPickerOpen}
                onPick={(pick) => send({ cached: pick })}
              />
            )}
          </>
        )}

        <div className="flex flex-1 flex-col gap-4">
          {chatId === null && (
            <p className="text-sm text-muted-foreground">
              {STATIC
                ? "Choose an experiment and one of its questions to see the stored answer."
                : "Ask a question by typing or recording. It is answered with the chosen config."}{" "}
              Open "Full detail" under an answer to see the transcript,
              evidence, prompt and trace.
            </p>
          )}
          {messages?.map((m) => (
            <MessageView
              key={m.id}
              message={m}
              onEdit={(transcript) => {
                selectChat(null)
                setText(transcript)
              }}
            />
          ))}
          {waitingSince !== null && <Waiting since={waitingSince} />}
          <div ref={end} />
        </div>

        <div className="sticky bottom-0 flex flex-col gap-2 border-t bg-background py-3">
          <div className="flex items-start gap-2">
            {config ? (
              <ConfigChips config={config} />
            ) : STATIC ? null : (
              <span aria-busy="true" className="flex items-center gap-1">
                <span className="sr-only">Loading configs</span>
                <Skeleton className="h-6 w-24" />
                <Skeleton className="h-6 w-20" />
                <Skeleton className="h-6 w-28" />
              </span>
            )}
            {config && !STATIC && (
              <Dialog>
                <DialogTrigger
                  render={
                    <Button variant="ghost" size="sm" className="ml-auto" />
                  }
                >
                  <SlidersHorizontalIcon /> Config
                </DialogTrigger>
                <DialogContent className="max-h-[90svh] overflow-y-auto sm:max-w-2xl">
                  <DialogHeader>
                    <DialogTitle>Configuration</DialogTitle>
                  </DialogHeader>
                  <ConfigPicker config={config} onChange={setPicked} />
                  <DialogFooter>
                    <DialogClose render={<Button />}>Done</DialogClose>
                  </DialogFooter>
                </DialogContent>
              </Dialog>
            )}
          </div>
          {configs.isError && !STATIC && (
            <p className="text-sm text-destructive">{configs.error.message}</p>
          )}
          {cachedRuns.isError && (
            <p className="text-sm text-destructive">
              {cachedRuns.error.message}
            </p>
          )}
          {recorder.error && (
            <p className="text-sm text-destructive">{recorder.error}</p>
          )}

          {chatId !== null ? (
            <Button className="self-start" onClick={() => selectChat(null)}>
              <PlusIcon /> New question
            </Button>
          ) : STATIC ? (
            <Button
              className="self-start"
              disabled={busy || !cachedRuns.data}
              onClick={() => setPickerOpen(true)}
            >
              <ListChecksIcon /> Choose a cached question
            </Button>
          ) : recorder.recording ? (
            <div className="flex items-center gap-2">
              <span
                className="size-2 animate-pulse rounded-full bg-destructive"
                aria-hidden
              />
              <span className="text-sm" role="status">
                Recording {Math.floor(recorder.seconds / 60)}:
                {String(recorder.seconds % 60).padStart(2, "0")}
              </span>
              <Button className="ml-auto" onClick={recorder.stop}>
                <SquareIcon /> Stop and send
              </Button>
              <Button variant="outline" onClick={recorder.cancel}>
                <XIcon /> Cancel
              </Button>
            </div>
          ) : (
            <form
              className="flex items-end gap-2"
              onSubmit={(e) => {
                e.preventDefault()
                const q = text.trim()
                if (!q || busy) return
                setText("")
                send({ text: q })
              }}
            >
              <textarea
                aria-label="Question"
                rows={2}
                value={text}
                placeholder="Ask a question… (Enter sends, Shift+Enter for a new line)"
                onChange={(e) => setText(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault()
                    e.currentTarget.form?.requestSubmit()
                  }
                }}
                className="min-w-0 flex-1 resize-none rounded-lg border border-input bg-transparent p-2 text-sm outline-none focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/50"
              />
              <Button
                type="button"
                variant="outline"
                size="icon-lg"
                aria-label="Record a voice question"
                disabled={busy}
                onClick={recorder.start}
              >
                <MicIcon />
              </Button>
              <Button
                type="submit"
                size="icon-lg"
                aria-label="Send"
                disabled={busy || !text.trim()}
              >
                <SendIcon />
              </Button>
            </form>
          )}
        </div>
      </div>
    </div>
  )
}

// A slow answer is normal (35-90 s for Qwen, longer with retries); only the server's 502 is an error.
function Waiting({ since }: { since: number }) {
  const [now, setNow] = useState(since)
  useEffect(() => {
    const timer = setInterval(() => setNow(Date.now()), 1000)
    return () => clearInterval(timer)
  }, [])
  return (
    <div
      role="status"
      className="flex flex-col gap-1 rounded-lg border border-dashed p-3 text-sm"
    >
      <span className="flex items-center gap-2">
        <span
          className="size-2 animate-pulse rounded-full bg-primary"
          aria-hidden
        />
        Waiting for the answer… {Math.round((now - since) / 1000)} s
      </span>
      <span className="text-xs text-muted-foreground">
        A Qwen answer usually takes 35 to 90 seconds, and the server answers one
        question at a time. This is not an error; the page does not retry on its
        own.
      </span>
    </div>
  )
}

function MessageView({
  message: m,
  onEdit,
}: {
  message: Message
  onEdit: (text: string) => void
}) {
  if (m.role === "user")
    return (
      <div className="flex flex-col items-end gap-1 self-end">
        {m.audio ? (
          <AudioPlayer blob={m.audio} />
        ) : (
          <p className="rounded-lg bg-muted px-3 py-2 text-sm whitespace-pre-wrap">
            {m.text}
          </p>
        )}
      </div>
    )
  if (m.error)
    return (
      <div
        role="alert"
        className="rounded-lg border border-destructive/40 bg-destructive/10 p-3 text-sm"
      >
        <p className="font-medium text-destructive">
          The API rejected this request
        </p>
        <p className="break-words">{m.error}</p>
      </div>
    )
  const r = m.response!
  const hidden = m.confirmTranscript
  return (
    <div className="flex flex-col gap-2">
      {m.input === "voice" && (
        <p className="text-xs text-muted-foreground">
          Heard: <span className="text-foreground">“{r.transcript}”</span>
        </p>
      )}
      {hidden ? (
        <div className="flex flex-wrap items-center gap-2 text-sm">
          <span>Is the transcript right?</span>
          <Button
            size="sm"
            onClick={() =>
              db.messages.update(m.id, { confirmTranscript: false })
            }
          >
            Yes, show the answer
          </Button>
          <Button
            size="sm"
            variant="outline"
            onClick={() => {
              onEdit(r.transcript)
              db.messages.update(m.id, { confirmTranscript: false })
            }}
          >
            Edit and ask as text
          </Button>
        </div>
      ) : (
        <Markdown>{r.answer ?? "(no answer: retrieval-only config)"}</Markdown>
      )}
      {m.config && (
        <WhatRan config={m.config} input={m.input ?? "text"} trace={r.trace} />
      )}
      <QueryDetail response={r} />
    </div>
  )
}

function AudioPlayer({ blob }: { blob: Blob }) {
  const audio = useRef<HTMLAudioElement>(null)
  useEffect(() => {
    const url = URL.createObjectURL(blob)
    if (audio.current) audio.current.src = url
    return () => URL.revokeObjectURL(url)
  }, [blob])
  return (
    <audio
      ref={audio}
      controls
      className="h-10 max-w-full"
      aria-label="Your recorded question"
    />
  )
}
