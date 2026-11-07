import { useState } from "react"
import { SettingsIcon } from "lucide-react"

import { getAdminToken, setAdminToken } from "@/api/client"
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

export function SettingsDialog() {
  const [token, setToken] = useState(getAdminToken)
  return (
    <Dialog onOpenChange={(open) => open && setToken(getAdminToken())}>
      <DialogTrigger
        render={<Button variant="ghost" size="icon" aria-label="Settings" />}
      >
        <SettingsIcon />
      </DialogTrigger>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Settings</DialogTitle>
          <DialogDescription>
            The admin token unlocks starting runs. It is kept in this browser
            only and sent as the X-Admin-Token header.
          </DialogDescription>
        </DialogHeader>
        <form
          className="flex flex-col gap-2"
          onSubmit={(e) => {
            e.preventDefault()
            setAdminToken(token.trim())
          }}
        >
          <Label htmlFor="admin-token">Admin token</Label>
          <Input
            id="admin-token"
            type="password"
            autoComplete="off"
            value={token}
            onChange={(e) => setToken(e.target.value)}
          />
        </form>
        <DialogFooter>
          <DialogClose
            render={<Button onClick={() => setAdminToken(token.trim())} />}
          >
            Save
          </DialogClose>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
