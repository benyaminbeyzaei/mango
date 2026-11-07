const STAGES = ["Audio", "ASR", "Encoder", "Retriever", "Context", "Generator"]

// The standard flow, left to right.
export function PipelineDiagram() {
  const w = 104
  const gap = 24
  // On a phone the diagram scrolls inside its own box, so labels stay readable.
  return (
    <div className="max-w-3xl overflow-x-auto">
      <svg
        viewBox={`0 0 ${STAGES.length * (w + gap) - gap} 56`}
        className="w-full min-w-xl text-foreground"
        role="img"
        aria-label="Pipeline: audio, ASR, encoder, retriever, context builder, generator"
      >
        <defs>
          <marker
            id="arrow"
            viewBox="0 0 8 8"
            refX="7"
            refY="4"
            markerWidth="6"
            markerHeight="6"
            orient="auto"
          >
            <path d="M0,0 L8,4 L0,8 z" fill="currentColor" />
          </marker>
        </defs>
        {STAGES.map((name, i) => {
          const x = i * (w + gap)
          return (
            <g key={name}>
              <rect
                x={x + 1}
                y={8}
                width={w - 2}
                height={40}
                rx={8}
                fill="none"
                stroke="currentColor"
              />
              <text
                x={x + w / 2}
                y={33}
                textAnchor="middle"
                fontSize={14}
                fill="currentColor"
              >
                {name}
              </text>
              {i > 0 && (
                <line
                  x1={x - gap + 2}
                  y1={28}
                  x2={x - 2}
                  y2={28}
                  stroke="currentColor"
                  markerEnd="url(#arrow)"
                />
              )}
            </g>
          )
        })}
      </svg>
    </div>
  )
}
