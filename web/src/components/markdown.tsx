import ReactMarkdown, { type Components } from "react-markdown"

// Tailwind resets element styles, so each Markdown element gets its classes here.
const components: Components = {
  h1: (p) => <h1 className="font-heading text-2xl font-medium" {...p} />,
  h2: (p) => <h2 className="mt-4 font-heading text-xl font-medium" {...p} />,
  h3: (p) => <h3 className="mt-2 font-heading text-lg font-medium" {...p} />,
  p: (p) => <p className="leading-relaxed" {...p} />,
  ul: (p) => <ul className="ml-6 list-disc space-y-1" {...p} />,
  ol: (p) => <ol className="ml-6 list-decimal space-y-1" {...p} />,
  a: (p) => <a className="underline underline-offset-4" {...p} />,
  pre: (p) => (
    <pre
      className="overflow-x-auto rounded-lg bg-muted p-3 font-mono text-xs"
      {...p}
    />
  ),
  code: (p) => <code className="font-mono text-[0.9em]" {...p} />,
  blockquote: (p) => (
    <blockquote className="border-l-2 pl-4 text-muted-foreground" {...p} />
  ),
}

export function Markdown({ children }: { children: string }) {
  return (
    <div className="flex max-w-3xl min-w-0 flex-col gap-3 text-sm">
      <ReactMarkdown components={components}>{children}</ReactMarkdown>
    </div>
  )
}
