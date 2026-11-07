import { createFileRoute } from "@tanstack/react-router"

import { Markdown } from "@/components/markdown"
import { PipelineDiagram } from "@/components/pipeline-diagram"
import architecture from "@/content/architecture.md?raw"

export const Route = createFileRoute("/docs")({ component: Docs })

function Docs() {
  return (
    <div className="flex flex-col gap-6">
      <PipelineDiagram />
      <Markdown>{architecture}</Markdown>
    </div>
  )
}
