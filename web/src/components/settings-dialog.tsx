import { useState } from "react"
import { SettingsIcon } from "lucide-react"

import { getAdminToken, setAdminToken } from "@/api/client"
import { PresetSelect } from "@/components/config-picker"
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
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import {
  getConfirmTranscript,
  getDefaultPreset,
  setConfirmTranscript,
  setDefaultPreset,
} from "@/lib/settings"

export function SettingsDialog() {
  const [token, setToken] = useState(getAdminToken)
  const [preset, setPreset] = useState(getDefaultPreset)
  const [confirm, setConfirm] = useState(getConfirmTranscript)
  const save = () => {
    setAdminToken(token.trim())
    setDefaultPreset(preset)
    setConfirmTranscript(confirm)
  }
  return (
    <Dialog
      onOpenChange={(open) => {
        if (!open) return
        setToken(getAdminToken())
        setPreset(getDefaultPreset())
        setConfirm(getConfirmTranscript())
      }}
    >
      <DialogTrigger
        render={<Button variant="ghost" size="icon" aria-label="Settings" />}
      >
        <SettingsIcon />
      </DialogTrigger>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Settings</DialogTitle>
          <DialogDescription>
            Kept in this browser only (localStorage).
          </DialogDescription>
        </DialogHeader>
        <form
          className="flex flex-col gap-4"
          onSubmit={(e) => {
            e.preventDefault()
            save()
          }}
        >
          <div className="flex flex-col gap-1">
            <Label htmlFor="default-preset">Default config for new chats</Label>
            <PresetSelect
              id="default-preset"
              value={preset}
              onChange={setPreset}
            />
          </div>
          <label className="flex items-center gap-2 text-sm">
            <input
              type="checkbox"
              checked={confirm}
              onChange={(e) => setConfirm(e.target.checked)}
            />
            Confirm the transcript of a voice question before showing its answer
          </label>
          <div className="flex flex-col gap-1">
            <Label htmlFor="admin-token">Admin token</Label>
            <Input
              id="admin-token"
              type="password"
              autoComplete="off"
              value={token}
              onChange={(e) => setToken(e.target.value)}
            />
            <p className="text-xs text-muted-foreground">
              Unlocks starting runs; sent as the X-Admin-Token header.
            </p>
          </div>
        </form>
        <DialogFooter>
          <DialogClose render={<Button onClick={save} />}>Save</DialogClose>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
