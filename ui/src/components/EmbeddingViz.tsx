import { useEffect, useState } from 'react'
import {
  CartesianGrid,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { getVizEmbeddings } from '../api/client'
import type { EmbeddingPoint } from '../types'

interface ChartPoint {
  x: number
  y: number
  label: string
  score: number
  type: string
}

function scoreToColor(score: number): string {
  if (score >= 0.8) return '#38a169'
  if (score >= 0.5) return '#d69e2e'
  return '#e53e3e'
}

interface CustomDotProps {
  cx?: number
  cy?: number
  payload?: ChartPoint
}

function CustomDot({ cx = 0, cy = 0, payload }: CustomDotProps) {
  if (!payload) return null
  if (payload.type === 'query') {
    return (
      <polygon
        points={`${cx},${cy - 10} ${cx + 9},${cy + 5} ${cx - 9},${cy + 5}`}
        fill="#8ABF34"
        stroke="#1B2A4A"
        strokeWidth={1.5}
      />
    )
  }
  return (
    <circle
      cx={cx}
      cy={cy}
      r={6}
      fill={scoreToColor(payload.score)}
      fillOpacity={0.8}
      stroke="#fff"
      strokeWidth={1}
    />
  )
}

interface TooltipProps {
  active?: boolean
  payload?: Array<{ payload: ChartPoint }>
}

function CustomTooltip({ active, payload }: TooltipProps) {
  if (!active || !payload?.[0]) return null
  const p = payload[0].payload
  return (
    <div style={{
      background: '#fff',
      border: '1px solid #e2e8f0',
      borderRadius: 8,
      padding: '8px 12px',
      fontSize: 12,
      maxWidth: 260,
      boxShadow: '0 4px 12px rgba(0,0,0,0.1)',
    }}>
      <p style={{ fontWeight: 700, color: p.type === 'query' ? '#8ABF34' : '#1B2A4A', marginBottom: 4 }}>
        {p.type === 'query' ? '★ Query' : '● Chunk'}
      </p>
      <p style={{ color: '#4a5568', lineHeight: 1.4 }}>{p.label}</p>
      {p.type === 'chunk' && (
        <p style={{ color: scoreToColor(p.score), fontWeight: 600, marginTop: 4 }}>
          score: {(p.score * 100).toFixed(1)}%
        </p>
      )}
    </div>
  )
}

export default function EmbeddingViz() {
  const [points, setPoints] = useState<EmbeddingPoint[]>([])
  const [note, setNote] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const load = async () => {
    setLoading(true)
    setError('')
    try {
      const data = await getVizEmbeddings()
      setPoints(data.points)
      setNote(data.note)
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Failed to load embeddings')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { load() }, [])

  const queryPoints: ChartPoint[] = points
    .filter(p => p.type === 'query')
    .map(p => ({ x: p.x, y: p.y, label: p.label, score: p.score, type: p.type }))

  const chunkPoints: ChartPoint[] = points
    .filter(p => p.type === 'chunk')
    .map(p => ({ x: p.x, y: p.y, label: p.label, score: p.score, type: p.type }))

  return (
    <div style={{ padding: 24, height: '100%', display: 'flex', flexDirection: 'column' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
        <div>
          <h2 style={{ color: '#1B2A4A', fontSize: 20, fontWeight: 700, marginBottom: 4 }}>
            Embedding Visualisation
          </h2>
          <p style={{ color: '#64748b', fontSize: 13 }}>
            PCA 2-D projection — spatial proximity = semantic similarity
          </p>
        </div>
        <button
          onClick={load}
          disabled={loading}
          style={{
            background: '#00A7B3',
            color: '#fff',
            border: 'none',
            borderRadius: 8,
            padding: '8px 16px',
            cursor: loading ? 'not-allowed' : 'pointer',
            fontWeight: 600,
            fontSize: 13,
            opacity: loading ? 0.6 : 1,
          }}
        >
          {loading ? 'Loading…' : 'Refresh'}
        </button>
      </div>

      {error && (
        <div style={{ background: '#fff5f5', border: '1px solid #fed7d7', borderRadius: 8, padding: 12, marginBottom: 12, color: '#e53e3e', fontSize: 13 }}>
          {error}
        </div>
      )}

      {note && (
        <div style={{ background: '#ebf8ff', border: '1px solid #bee3f8', borderRadius: 8, padding: 10, marginBottom: 12, fontSize: 12, color: '#2b6cb0' }}>
          ℹ {note}
        </div>
      )}

      {points.length === 0 && !loading ? (
        <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#94a3b8', fontSize: 14 }}>
          Run some queries in the Chat tab to populate the embedding space.
        </div>
      ) : (
        <div style={{ flex: 1, minHeight: 400 }}>
          <ResponsiveContainer width="100%" height="100%">
            <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis
                dataKey="x"
                type="number"
                name="PC1"
                label={{ value: 'PC 1', position: 'insideBottom', offset: -10, fontSize: 12 }}
                tick={{ fontSize: 10 }}
              />
              <YAxis
                dataKey="y"
                type="number"
                name="PC2"
                label={{ value: 'PC 2', angle: -90, position: 'insideLeft', fontSize: 12 }}
                tick={{ fontSize: 10 }}
              />
              <Tooltip content={<CustomTooltip />} />
              <Scatter
                name="Retrieved chunks"
                data={chunkPoints}
                shape={<CustomDot />}
              />
              <Scatter
                name="Queries"
                data={queryPoints}
                shape={<CustomDot />}
              />
            </ScatterChart>
          </ResponsiveContainer>
        </div>
      )}

      <div style={{ marginTop: 12, fontSize: 11, color: '#94a3b8', display: 'flex', gap: 16 }}>
        <span>
          <span style={{ display: 'inline-block', width: 0, height: 0, borderLeft: '7px solid transparent', borderRight: '7px solid transparent', borderBottom: '12px solid #8ABF34', marginRight: 5 }} />
          query
        </span>
        <span>
          <span style={{ display: 'inline-block', width: 10, height: 10, borderRadius: '50%', background: '#38a169', marginRight: 5 }} />
          high relevance
        </span>
        <span>
          <span style={{ display: 'inline-block', width: 10, height: 10, borderRadius: '50%', background: '#d69e2e', marginRight: 5 }} />
          medium
        </span>
        <span>
          <span style={{ display: 'inline-block', width: 10, height: 10, borderRadius: '50%', background: '#e53e3e', marginRight: 5 }} />
          low
        </span>
      </div>
    </div>
  )
}
