import { createFileRoute } from "@tanstack/react-router"

import { useModels } from "@/api/queries"
import type { components } from "@/api/schema"
import { Line, Loading, TableSkeleton } from "@/components/page-skeletons"
import { Badge } from "@/components/ui/badge"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"

export const Route = createFileRoute("/models")({ component: Models })

type Row = { name: string; available: boolean; note?: string | null }

// One table per pipeline stage, in pipeline order.
function stages(
  m: components["schemas"]["ModelsResponse"]
): { stage: string; rows: Row[] }[] {
  const plain = (names: string[]): Row[] =>
    names.map((name) => ({ name, available: true }))
  return [
    {
      stage: "ASR",
      rows: m.asr.map((a) => ({
        name: a.name,
        available: a.available ?? true,
      })),
    },
    {
      stage: "Query encoders",
      rows: m.encoders.map((e) => ({
        name: e.name,
        available: e.available ?? true,
        note: e.space && `index ${e.space}`,
      })),
    },
    { stage: "Corpus indexes", rows: plain(m.indexes) },
    { stage: "Retrievers", rows: plain(m.retrievers) },
    { stage: "Context builders", rows: plain(m.context_builders) },
    { stage: "Chat models (ELM)", rows: plain(m.chat_models) },
    { stage: "Flows", rows: plain(m.flows) },
  ]
}

function Models() {
  const models = useModels()

  return (
    <div className="flex flex-col gap-6">
      <h1 className="font-heading text-2xl font-medium">Models</h1>
      {models.isPending && (
        <Loading label="Loading models">
          {[4, 3, 6].map((rows, i) => (
            <section key={i} className="flex flex-col gap-2">
              <Line className="h-7 w-32" />
              <TableSkeleton rows={rows} cols={2} />
            </section>
          ))}
        </Loading>
      )}
      {models.isError && (
        <p className="text-destructive">{models.error.message}</p>
      )}
      {models.data &&
        stages(models.data).map(({ stage, rows }) => (
          <section key={stage} className="flex flex-col gap-2">
            <h2 className="font-heading text-lg font-medium">{stage}</h2>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Name</TableHead>
                  <TableHead className="w-28">Status</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {rows.map((row) => (
                  <TableRow key={row.name}>
                    <TableCell className="font-mono text-xs break-all whitespace-normal">
                      {row.name}
                      {row.note && (
                        <span className="ml-2 text-muted-foreground">
                          ({row.note})
                        </span>
                      )}
                    </TableCell>
                    <TableCell>
                      {row.available ? (
                        <Badge>available</Badge>
                      ) : (
                        <Badge variant="outline">unavailable</Badge>
                      )}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </section>
        ))}
    </div>
  )
}
