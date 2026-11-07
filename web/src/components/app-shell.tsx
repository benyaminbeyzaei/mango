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
  const { setTheme } = useTheme()
  const toggle = () =>
    setTheme(
      document.documentElement.classList.contains("dark") ? "light" : "dark"
    )
  return (
    <Button
      variant="ghost"
      size="icon"
      onClick={toggle}
      aria-label="Toggle light/dark theme"
    >
      <SunIcon className="hidden dark:block" />
      <MoonIcon className="dark:hidden" />
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
              <SheetTitle>mlsrag</SheetTitle>
              <nav className="flex flex-col" aria-label="Sections">
                <NavLinks onNavigate={() => setMenuOpen(false)} />
              </nav>
            </SheetContent>
          </Sheet>
          <Link to="/" className="mr-2 font-heading font-medium">
            mlsrag
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
