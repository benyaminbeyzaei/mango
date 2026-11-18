import { cn } from "@/lib/utils"

// A flat placeholder block. It fades in after a short delay (see .skeleton in
// index.css), so fast loads never flash it, and it is hidden from screen readers:
// the surrounding region carries aria-busy and a "Loading" label instead.
function Skeleton({ className, ...props }: React.ComponentProps<"div">) {
  return (
    <div
      aria-hidden="true"
      className={cn("skeleton rounded-md bg-muted", className)}
      {...props}
    />
  )
}

export { Skeleton }
