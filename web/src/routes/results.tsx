import { createFileRoute } from "@tanstack/react-router"

import { Placeholder } from "@/components/placeholder"

export const Route = createFileRoute("/results")({
  component: () => <Placeholder title="Results" plan="011" />,
})
