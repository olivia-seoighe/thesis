import type { Citation } from '../types'

interface Props {
  citations: Citation[]
}

export default function CitationCard({ citations }: Props) {
  if (citations.length === 0) return null

  return (
    <div style={{ marginTop: 12 }}>
      <p style={{ fontSize: 11, color: '#667eea', fontWeight: 600, marginBottom: 6, letterSpacing: '0.05em', textTransform: 'uppercase' }}>
        Sources
      </p>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
        {citations.map((c, i) => (
          <div
            key={i}
            style={{
              background: '#f8fafc',
              border: '1px solid #e2e8f0',
              borderLeft: '3px solid #00A7B3',
              borderRadius: 6,
              padding: '8px 12px',
              fontSize: 12,
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 4 }}>
              <a
                href={c.url}
                target="_blank"
                rel="noreferrer"
                style={{ color: '#1B2A4A', fontWeight: 600, textDecoration: 'none', fontFamily: 'monospace' }}
                title={c.url}
              >
                [{i + 1}] {c.title}
              </a>
              <span
                style={{
                  background: scoreColor(c.score),
                  color: '#fff',
                  borderRadius: 10,
                  padding: '1px 7px',
                  fontSize: 10,
                  fontWeight: 700,
                  whiteSpace: 'nowrap',
                }}
              >
                {(c.score * 100).toFixed(1)}%
              </span>
            </div>
            {c.chunk_text && (
              <p style={{ color: '#4a5568', fontSize: 11, lineHeight: 1.5, margin: 0 }}>
                {c.chunk_text.slice(0, 220)}{c.chunk_text.length > 220 ? '…' : ''}
              </p>
            )}
            {c.url && (
              <a
                href={c.url}
                target="_blank"
                rel="noreferrer"
                style={{ color: '#00A7B3', fontSize: 10, marginTop: 4, display: 'block' }}
              >
                View on GitHub →
              </a>
            )}
          </div>
        ))}
      </div>
    </div>
  )
}

function scoreColor(score: number): string {
  if (score >= 0.8) return '#38a169'
  if (score >= 0.5) return '#d69e2e'
  return '#e53e3e'
}
