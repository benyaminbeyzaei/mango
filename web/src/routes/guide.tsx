import { createFileRoute } from "@tanstack/react-router"

import { Markdown } from "@/components/markdown"
import guide from "@/content/evaluator-guide.md?raw"

export const Route = createFileRoute("/guide")({
  component: () => <Markdown>{guide}</Markdown>,
})
