// Readable names for configs, question sets and matrices: the same names as
// rag/extensions/platform/docs/results.md. The raw ids stay in URLs and tooltips.

const CONFIGS: Record<string, string> = {
  baseline: "Dense search (baseline)",
  "baseline-courseasr": "Dense search (baseline), course ASR",
  bm25: "BM25",
  "grep-static": "Grep, static",
  "grep-agentic": "Grep, agentic",
  "grep-feedback": "Grep, feedback",
  "grep-feedback-qwen": "Grep, feedback (Qwen helper)",
  "grep-feedback-mistral": "Grep, feedback (Mistral helper)",
  "topics-k256-m2": "Topic routing, 2 topics",
  "topics-k256-m4": "Topic routing, 4 topics",
  "topics-k256-m8": "Topic routing, 8 topics",
  "toc-centroid-m4": "TOC, centroid route",
  "toc-llm-m4": "TOC, LLM route",
  "toc-llm-topics-m4": "TOC, LLM route with topic names",
  "toc-llm-flat-m4": "TOC, one LLM call over all topics",
  "toc-centroid-m4-gpt5labels": "TOC (gpt-5-mini labels), centroid route",
  "toc-llm-m4-gpt5labels": "TOC (gpt-5-mini labels), LLM route",
  "toc-llm-topics-m4-gpt5labels":
    "TOC (gpt-5-mini labels), LLM route with topic names",
  "toc-llm-flat-m4-gpt5labels":
    "TOC (gpt-5-mini labels), one LLM call over all topics",
  "raptor-lite-expand": "RAPTOR-lite, expand",
  "raptor-lite-evidence": "RAPTOR-lite, summaries as evidence",
  "raptor-tree-expand": "RAPTOR-tree, expand",
  "raptor-tree-evidence": "RAPTOR-tree, summaries as evidence",
  "raptor-tree-gpt5-expand": "RAPTOR-tree (gpt-5-mini), expand",
  "raptor-tree-gpt5-evidence":
    "RAPTOR-tree (gpt-5-mini), summaries as evidence",
  "raptor-tree-gpt5-evidence-courseasr":
    "RAPTOR-tree (gpt-5-mini), summaries as evidence, course ASR",
  "raptor-full-expand": "RAPTOR-full, expand",
  "raptor-full-evidence": "RAPTOR-full, summaries as evidence",
  "rerank-minilm-p20": "Rerank, MiniLM-L6, pool 20",
  "rerank-minilm-p50": "Rerank, MiniLM-L6, pool 50",
  "rerank-minilm-p100": "Rerank, MiniLM-L6, pool 100",
  "rerank-bge-p20": "Rerank, bge-reranker-base, pool 20",
  "rerank-bge-p50": "Rerank, bge-reranker-base, pool 50",
  "rerank-bge-p100": "Rerank, bge-reranker-base, pool 100",
  "rerank-minilm-ft-p20": "Rerank, MiniLM-L6 fine-tuned on practice, pool 20",
  "rerank-bge-m3-p20": "Rerank, bge-reranker-v2-m3, pool 20",
  "rerank-mxbai-v2-p20": "Rerank, mxbai-rerank-base-v2, pool 20",
  "enc-b-asis": "enc_b index, gte-small query as is",
  "enc-b-reencode": "enc_b index, query re-encoded with bge-small",
  "enc-b-reencode-prefix": "enc_b index, re-encoded with bge's query prefix",
  "enc-b-translate-lstsq": "enc_b index, query mapped (least squares)",
  "enc-b-translate-procrustes": "enc_b index, query mapped (Procrustes)",
  "context-budgeted": "Context: 800-token budget",
  "context-dedupe": "Context: drop near-duplicates, then budget",
}

export const configLabel = (name: string) => CONFIGS[name] ?? name

// In display order; each with the number of questions it holds.
export const SUBSETS: { id: string; questions: number }[] = [
  { id: "practice-20", questions: 20 },
  { id: "practice-70", questions: 70 },
  { id: "practice", questions: 350 },
  { id: "test-25", questions: 25 },
  { id: "test", questions: 125 },
]
export const subsetOrder = (id: string) => {
  const i = SUBSETS.findIndex((s) => s.id === id)
  return i === -1 ? SUBSETS.length : i
}

const MATRICES: Record<string, string> = {
  retrievers: "Retrievers",
  encoders: "Encoder policies",
  context: "Context builders",
  "toc-raptor": "TOC and RAPTOR",
  rerank: "Reranker settings",
  "rerank-models": "Stronger rerankers",
  "rerank-ft": "Fine-tuned reranker",
  final: "Final comparison",
  "final-bge": "Final comparison, bge-reranker-v2-m3",
  "final-mxbai": "Final comparison, mxbai-rerank-base-v2",
  courseasr: "Course ASR instead of whisper-1",
}

// "rerank-models-test" -> "Stronger rerankers" (the question set is shown next to it).
export function matrixLabel(name: string, subset: string) {
  const stem = name.endsWith(`-${subset}`)
    ? name.slice(0, -subset.length - 1)
    : name
  return MATRICES[stem] ?? name
}

// "20261110-1935-..." -> "10 Nov 2026, 19:35"
export function runDate(runId: string) {
  const m = runId.match(/^(\d{4})(\d{2})(\d{2})-(\d{2})(\d{2})/)
  if (!m) return ""
  const d = new Date(+m[1], +m[2] - 1, +m[3], +m[4], +m[5])
  return d.toLocaleString("en-GB", {
    day: "numeric",
    month: "short",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  })
}
