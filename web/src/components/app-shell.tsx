import { useState, type ReactNode } from "react"
import { Link } from "@tanstack/react-router"
import { MenuIcon, MoonIcon, SunIcon } from "lucide-react"

import { SettingsDialog } from "@/components/settings-dialog"
import { useTheme } from "@/components/theme-provider"
import { Button } from "@/components/ui/button"
import {
  Sheet,
  SheetContent,
  SheetTitle,
  SheetTrigger,
} from "@/components/ui/sheet"

const SECTIONS = [
  { to: "/docs", label: "Docs" },
  { to: "/models", label: "Models" },
  { to: "/results", label: "Results" },
  { to: "/questions", label: "Questions" },
  { to: "/interactive", label: "Interactive" },
  { to: "/guide", label: "Guide" },
] as const

function NavLinks({ onNavigate }: { onNavigate?: () => void }) {
  return SECTIONS.map(({ to, label }) => (
    <Link
      key={to}
      to={to}
      onClick={onNavigate}
      className="rounded-md px-3 py-2 text-sm text-muted-foreground outline-none hover:text-foreground focus-visible:ring-3 focus-visible:ring-ring/50 [&.active]:font-medium [&.active]:text-foreground"
    >
      {label}
    </Link>
  ))
}

function ThemeToggle() {
  const { theme, setTheme } = useTheme()
  return (
    <Button
      variant="ghost"
      size="icon"
      onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
      aria-label="Toggle light/dark theme"
    >
      {theme === "dark" ? <SunIcon /> : <MoonIcon />}
    </Button>
  )
}

export function AppShell({ children }: { children: ReactNode }) {
  const [menuOpen, setMenuOpen] = useState(false)
  return (
    <div className="flex min-h-svh flex-col">
      <header className="sticky top-0 z-40 border-b bg-background">
        <div className="mx-auto flex h-14 max-w-5xl items-center gap-2 px-4">
          <Sheet open={menuOpen} onOpenChange={setMenuOpen}>
            <SheetTrigger
              render={
                <Button
                  variant="ghost"
                  size="icon"
                  className="md:hidden"
                  aria-label="Open menu"
                />
              }
            >
              <MenuIcon />
            </SheetTrigger>
            <SheetContent side="left" className="p-4">
              <SheetTitle>Mango</SheetTitle>
              <nav className="flex flex-col" aria-label="Sections">
                <NavLinks onNavigate={() => setMenuOpen(false)} />
              </nav>
            </SheetContent>
          </Sheet>
          <Link
            to="/"
            className="mr-2 flex items-center gap-2 font-heading font-medium"
          >
            <img src="/logo-64.png" alt="" width={32} height={32} />
            Mango
          </Link>
          <nav className="hidden items-center md:flex" aria-label="Sections">
            <NavLinks />
          </nav>
          <div className="ml-auto flex items-center gap-1">
            <ThemeToggle />
            <SettingsDialog />
          </div>
        </div>
      </header>
      <main className="mx-auto w-full max-w-5xl min-w-0 flex-1 px-4 py-6">
        {children}
      </main>
    </div>
  )
}
