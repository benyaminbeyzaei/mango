// Preset dropdown plus an "advanced" panel with one control per RunConfig field (built from /openapi.json).
import { useState } from "react"

import {
  useConfigs,
  useModels,
  useRunConfigSchema,
  type FieldSchema,
  type RunConfig,
} from "@/api/queries"
import { Badge } from "@/components/ui/badge"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"

export const selectClass =
  "h-8 w-full min-w-0 rounded-lg border border-input bg-background px-2 text-sm outline-none focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/50 dark:bg-input/30"

export function ConfigChips({ config }: { config: RunConfig }) {
  const chips = [
    config.name,
    `asr ${config.asr}`,
    config.query_encoder,
    config.corpus_index,
    config.mismatch_policy !== "error" && config.mismatch_policy,
    config.retriever,
    `k=${config.k}`,
    config.context_builder,
    config.generator_model.split("/").pop(),
    config.flow !== "standard" && config.flow,
    Object.keys(config.params).length > 0 && "params",
  ].filter(Boolean) as string[]
  return (
    <div className="flex flex-wrap gap-1" aria-label="Current config">
      {chips.map((c, i) => (
        <Badge key={i} variant={i === 0 ? "default" : "outline"}>
          {c}
        </Badge>
      ))}
    </div>
  )
}

export function PresetSelect({
  id,
  value,
  onChange,
}: {
  id: string
  value: string
  onChange: (name: string) => void
}) {
  const configs = useConfigs()
  const names = configs.data?.map((c) => c.name) ?? []
  return (
    <select
      id={id}
      className={selectClass}
      value={value}
      onChange={(e) => onChange(e.target.value)}
    >
      {!names.includes(value) && <option value={value}>{value}</option>}
      {configs.data?.map((c) => (
        <option key={c.name} value={c.name}>
          {c.name} ({c.source})
        </option>
      ))}
    </select>
  )
}

export function ConfigPicker({
  config,
  onChange,
}: {
  config: RunConfig
  onChange: (config: RunConfig) => void
}) {
  const configs = useConfigs()
  const schema = useRunConfigSchema()
  const models = useModels()
  const choices: Record<string, string[] | undefined> = {
    retriever: models.data?.retrievers,
    context_builder: models.data?.context_builders,
    generator_model: models.data?.chat_models,
    judge_model: models.data?.chat_models,
    flow: models.data?.flows,
  }
  const set = (field: string, value: unknown) =>
    onChange({ ...config, [field]: value })

  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-col gap-1">
        <Label htmlFor="preset">Preset</Label>
        <PresetSelect
          id="preset"
          value={config.name}
          onChange={(name) => {
            const entry = configs.data?.find((c) => c.name === name)
            if (entry) onChange(entry.config)
          }}
        />
      </div>
      <details className="rounded-lg border p-2">
        <summary className="cursor-pointer text-sm font-medium">
          Advanced: every RunConfig field
        </summary>
        {schema.isError && (
          <p className="mt-2 text-sm text-destructive">
            {schema.error.message}
          </p>
        )}
        <div className="mt-2 grid gap-2 sm:grid-cols-2">
          {schema.data &&
            Object.entries(schema.data).map(([field, s]) => (
              <Field
                key={field}
                field={field}
                schema={s}
                choices={choices[field]}
                value={config[field as keyof RunConfig]}
                onChange={(v) => set(field, v)}
              />
            ))}
        </div>
      </details>
    </div>
  )
}

function Field({
  field,
  schema,
  choices,
  value,
  onChange,
}: {
  field: string
  schema: FieldSchema
  choices?: string[]
  value: unknown
  onChange: (value: unknown) => void
}) {
  const id = `cfg-${field}`
  const label = <Label htmlFor={id}>{field}</Label>
  const options = schema.enum ?? choices

  if (field === "params")
    return (
      <div className="flex flex-col gap-1 sm:col-span-2">
        {label}
        <ParamsEditor
          key={JSON.stringify(value)}
          id={id}
          value={value as object}
          onChange={onChange}
        />
      </div>
    )
  if (options) {
    const current = String(value)
    return (
      <div className="flex flex-col gap-1">
        {label}
        <select
          id={id}
          className={selectClass}
          value={current}
          onChange={(e) => onChange(e.target.value)}
        >
          {!options.includes(current) && <option>{current}</option>}
          {options.map((o) => (
            <option key={o}>{o}</option>
          ))}
        </select>
      </div>
    )
  }
  if (schema.type === "integer")
    return (
      <div className="flex flex-col gap-1">
        {label}
        <Input
          id={id}
          type="number"
          min={1}
          value={String(value)}
          onChange={(e) => onChange(Number(e.target.value))}
        />
      </div>
    )
  const nullable = schema.anyOf?.some((t) => t.type === "null")
  return (
    <div className="flex flex-col gap-1">
      {label}
      <Input
        id={id}
        value={(value as string | null) ?? ""}
        placeholder={nullable ? "(none)" : undefined}
        onChange={(e) =>
          onChange(nullable && !e.target.value ? null : e.target.value)
        }
      />
    </div>
  )
}

function ParamsEditor({
  id,
  value,
  onChange,
}: {
  id: string
  value: object
  onChange: (value: unknown) => void
}) {
  const [text, setText] = useState(JSON.stringify(value, null, 2))
  const [error, setError] = useState("")
  const apply = () => {
    try {
      const parsed = JSON.parse(text)
      if (
        typeof parsed !== "object" ||
        parsed === null ||
        Array.isArray(parsed)
      )
        throw new Error("params must be a JSON object")
      setError("")
      onChange(parsed)
    } catch (err) {
      setError((err as Error).message)
    }
  }
  return (
    <>
      <textarea
        id={id}
        rows={3}
        spellCheck={false}
        className="w-full rounded-lg border border-input bg-transparent p-2 font-mono text-xs outline-none focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/50"
        value={text}
        onChange={(e) => setText(e.target.value)}
        onBlur={apply}
        aria-invalid={!!error}
      />
      {error && <p className="text-xs text-destructive">{error}</p>}
    </>
  )
}
