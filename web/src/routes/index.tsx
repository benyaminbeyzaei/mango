import { useQuery } from "@tanstack/react-query"
import { Link, createFileRoute } from "@tanstack/react-router"

import { api } from "@/api/client"
import { Badge } from "@/components/ui/badge"

export const Route = createFileRoute("/")({ component: Home })

function Home() {
  const health = useQuery({
    queryKey: ["health"],
    queryFn: async () => {
      const { data, error } = await api.GET("/api/health")
      if (error) throw new Error("health check failed")
      return data as { status: string; git_commit: string | null }
    },
  })

  return (
    <div className="flex flex-col gap-4">
      <h1 className="font-heading text-2xl font-medium">mlsrag</h1>
      <p className="text-muted-foreground">
        A web client for the RAG experiment platform: audio query → transcript →
        retrieval → answer. Start with the{" "}
        <Link to="/docs" className="underline underline-offset-4">
          architecture
        </Link>{" "}
        or the{" "}
        <Link to="/guide" className="underline underline-offset-4">
          evaluator guide
        </Link>
        .
      </p>
      <div className="flex flex-wrap items-center gap-2 text-sm">
        <span>API:</span>
        {health.isPending && <Badge variant="outline">checking…</Badge>}
        {health.isError && <Badge variant="destructive">unreachable</Badge>}
        {health.data && (
          <>
            <Badge>{health.data.status}</Badge>
            <span className="font-mono text-xs text-muted-foreground">
              commit {health.data.git_commit ?? "unknown"}
            </span>
          </>
        )}
      </div>
    </div>
  )
}
