// A notice shown as a dialog, opened by the small button it renders. An optional action runs after it closes.
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

export function NoticeDialog({
  title,
  children,
  triggerLabel = "About this version",
  action,
}: {
  title: string
  children: ReactNode
  triggerLabel?: string
  action?: { label: string; onClick: () => void }
}) {
  const [open, setOpen] = useState(false)

  return (
    <Dialog open={open} onOpenChange={setOpen}>
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
                setOpen(false)
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
