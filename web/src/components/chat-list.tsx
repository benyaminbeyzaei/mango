// The chat sidebar: new, select, rename, delete, export, import, clear all. All on-device (IndexedDB).
import { useRef, useState } from "react"
import {
  DownloadIcon,
  PencilIcon,
  PlusIcon,
  Trash2Icon,
  UploadIcon,
} from "lucide-react"

import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import {
  clearAll,
  db,
  deleteChat,
  exportChat,
  importChat,
  type Chat,
} from "@/lib/db"
import { cn } from "cn"

export function ChatList({
  chats,
  activeId,
  onSelect,
}: {
  chats: Chat[]
  activeId: number | null
  onSelect: (id: number | null) => void
}) {
  const [renaming, setRenaming] = useState<number | null>(null)
  const fileInput = useRef<HTMLInputElement>(null)

  return (
    <div className="flex flex-col gap-2">
      <Button variant="outline" onClick={() => onSelect(null)}>
        <PlusIcon /> New chat
      </Button>
      <ul className="flex flex-col gap-0.5" aria-label="Chats">
        {chats.map((chat) => (
          <li key={chat.id} className="group flex items-center gap-0.5">
            {renaming === chat.id ? (
              <form
                className="flex-1"
                onSubmit={(e) => {
                  e.preventDefault()
                  const title = new FormData(e.currentTarget).get("title")
                  if (title) db.chats.update(chat.id, { title: String(title) })
                  setRenaming(null)
                }}
              >
                <Input
                  name="title"
                  aria-label="Chat title"
                  defaultValue={chat.title}
                  autoFocus
                  onBlur={(e) => e.currentTarget.form?.requestSubmit()}
                  onKeyDown={(e) => e.key === "Escape" && setRenaming(null)}
                />
              </form>
            ) : (
              <button
                type="button"
                onClick={() => onSelect(chat.id)}
                aria-current={chat.id === activeId}
                className={cn(
                  "min-w-0 flex-1 truncate rounded-md px-2 py-1.5 text-left text-sm outline-none hover:bg-muted focus-visible:ring-3 focus-visible:ring-ring/50",
                  chat.id === activeId && "bg-muted font-medium"
                )}
              >
                {chat.title}
              </button>
            )}
            <Button
              variant="ghost"
              size="icon-sm"
              aria-label={`Rename ${chat.title}`}
              onClick={() => setRenaming(chat.id)}
            >
              <PencilIcon />
            </Button>
            <Button
              variant="ghost"
              size="icon-sm"
              aria-label={`Export ${chat.title} as JSON`}
              onClick={() => exportChat(chat.id)}
            >
              <DownloadIcon />
            </Button>
            <Button
              variant="ghost"
              size="icon-sm"
              aria-label={`Delete ${chat.title}`}
              onClick={async () => {
                if (!confirm(`Delete "${chat.title}"?`)) return
                await deleteChat(chat.id)
                if (chat.id === activeId) onSelect(null)
              }}
            >
              <Trash2Icon />
            </Button>
          </li>
        ))}
      </ul>
      {chats.length === 0 && (
        <p className="px-2 text-xs text-muted-foreground">No chats yet.</p>
      )}
      <div className="mt-2 flex flex-wrap gap-1">
        <Button
          variant="ghost"
          size="sm"
          onClick={() => fileInput.current?.click()}
        >
          <UploadIcon /> Import
        </Button>
        <input
          ref={fileInput}
          type="file"
          accept="application/json,.json"
          className="hidden"
          onChange={async (e) => {
            const file = e.target.files?.[0]
            e.target.value = ""
            if (!file) return
            try {
              onSelect(await importChat(file))
            } catch {
              alert("That file is not a chat exported from this page.")
            }
          }}
        />
        {chats.length > 0 && (
          <Button
            variant="ghost"
            size="sm"
            onClick={async () => {
              if (!confirm("Delete every chat stored in this browser?")) return
              await clearAll()
              onSelect(null)
            }}
          >
            <Trash2Icon /> Clear all
          </Button>
        )}
      </div>
      <p className="px-2 text-xs text-muted-foreground">
        Chats are stored in this browser only. The server keeps nothing.
      </p>
    </div>
  )
}
