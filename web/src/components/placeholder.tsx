export function Placeholder({ title, plan }: { title: string; plan: string }) {
  return (
    <div className="flex flex-col gap-2">
      <h1 className="font-heading text-2xl font-medium">{title}</h1>
      <p className="text-muted-foreground">Coming in plan {plan}.</p>
    </div>
  )
}
