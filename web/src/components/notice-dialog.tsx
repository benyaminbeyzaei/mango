// A notice shown as a dialog: it opens by itself once per browser session (per id), and the small
// button it renders opens it again. An optional action runs after the dialog closes.
import { useState, type ReactNode } from "react"
import { InfoIcon } from "lucide-react"

import { Button } from "@/components/ui/button"
import {
  Dialog,
  DialogClose,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog"

function seen(key: string) {
  try {
    return sessionStorage.getItem(key) === "1"
  } catch {
    return false // storage blocked: show the notice
  }
}

function markSeen(key: string) {
  try {
    sessionStorage.setItem(key, "1")
  } catch {
    // storage blocked: the notice shows again next time, which is fine
  }
}

export function NoticeDialog({
  id,
  title,
  children,
  triggerLabel = "About this version",
  action,
}: {
  id: string
  title: string
  children: ReactNode
  triggerLabel?: string
  action?: { label: string; onClick: () => void }
}) {
  const key = `notice:${id}`
  const [open, setOpen] = useState(() => !seen(key))

  function change(next: boolean) {
    setOpen(next)
    if (!next) markSeen(key)
  }

  return (
    <Dialog open={open} onOpenChange={change}>
      <DialogTrigger
        render={<Button variant="ghost" size="sm" className="self-start" />}
      >
        <InfoIcon /> {triggerLabel}
      </DialogTrigger>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>{title}</DialogTitle>
          <DialogDescription render={<div />}>{children}</DialogDescription>
        </DialogHeader>
        <DialogFooter>
          <DialogClose render={<Button variant="outline" />}>Close</DialogClose>
          {action && (
            <Button
              onClick={() => {
                change(false)
                action.onClick()
              }}
            >
              {action.label}
            </Button>
          )}
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
