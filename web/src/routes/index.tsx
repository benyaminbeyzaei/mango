import { useQuery } from "@tanstack/react-query"
import { Link, createFileRoute } from "@tanstack/react-router"

import { STATIC, api } from "@/api/client"
import { Badge } from "@/components/ui/badge"
import { Skeleton } from "@/components/ui/skeleton"

export const Route = createFileRoute("/")({ component: Home })

function Home() {
  const health = useQuery({
    queryKey: ["health"],
    queryFn: async () => {
      const { data, error } = await api.GET("/api/health")
      if (error) throw new Error("health check failed")
      return data
    },
  })

  return (
    <div className="flex flex-col gap-4">
      <div className="flex items-center gap-4">
        <img src="/logo-160.png" alt="" width={80} height={80} />
        <h1 className="font-heading text-3xl font-medium">Mango</h1>
      </div>
      <p className="text-muted-foreground">
        Mango shows the RAG experiments of the mlsrag platform: audio question →
        transcript → retrieval → answer. Start with the{" "}
        <Link to="/docs" className="underline underline-offset-4">
          architecture
        </Link>
        .
      </p>
      <div className="flex flex-wrap items-center gap-2 text-sm">
        <span>API:</span>
        {health.isPending && (
          <Skeleton className="h-5 w-40">
            <span className="sr-only">checking</span>
          </Skeleton>
        )}
        {health.isError && <Badge variant="destructive">unreachable</Badge>}
        {health.data && (
          <>
            <Badge>{health.data.status}</Badge>
            {STATIC && <Badge variant="secondary">Cached version</Badge>}
            <span className="font-mono text-xs text-muted-foreground">
              commit {health.data.git_commit ?? "unknown"}
            </span>
          </>
        )}
      </div>
    </div>
  )
}
