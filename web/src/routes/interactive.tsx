import { createFileRoute } from "@tanstack/react-router"

import { Placeholder } from "@/components/placeholder"

export const Route = createFileRoute("/interactive")({
  component: () => <Placeholder title="Interactive" plan="012" />,
})
