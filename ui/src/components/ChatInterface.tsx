import { useEffect, useRef, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { getSources, sendQuery } from '../api/client'
import type { Citation, Message } from '../types'
import CitationCard from './CitationCard'

interface Props {
  conversationId: string | null
  initialMessages?: Message[]
  onConversationCreated: (id: string) => void
}

export default function ChatInterface({ conversationId, initialMessages = [], onConversationCreated }: Props) {
  const [messages, setMessages] = useState<Message[]>(initialMessages)
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [sources, setSources] = useState<string[]>([])
  const [source, setSource] = useState('')
  const [mode, setMode] = useState('hybrid')
  const [topK, setTopK] = useState(5)
  const [activeConvId, setActiveConvId] = useState<string | null>(conversationId)
  const [latencyInfo, setLatencyInfo] = useState<string | null>(null)
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    setActiveConvId(conversationId)
  }, [conversationId])

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  useEffect(() => {
    getSources()
      .then(list => {
        setSources(list)
      })
      .catch(() => {})
  }, [])

  const submit = async () => {
    const q = input.trim()
    if (!q || loading) return

    const userMsg: Message = { role: 'user', content: q, timestamp: new Date().toISOString() }
    setMessages(prev => [...prev, userMsg])
    setInput('')
    setLoading(true)
    setError('')
    setLatencyInfo(null)

    try {
      const res = await sendQuery(q, {
        source,
        mode,
        topK,
        conversationId: activeConvId ?? undefined,
      })

      const asst: Message = {
        role: 'assistant',
        content: res.answer,
        citations: res.citations,
        timestamp: new Date().toISOString(),
      }
      setMessages(prev => [...prev, asst])
      setLatencyInfo(
        `mode ${mode}  ·  total ${res.latency_ms.toFixed(0)}ms  ·  retrieval ${res.retrieval_latency_ms.toFixed(0)}ms  ·  generation ${res.generation_latency_ms.toFixed(0)}ms  ·  model ${res.model_used}`
      )

      if (!activeConvId) {
        setActiveConvId(res.conversation_id)
        onConversationCreated(res.conversation_id)
      }
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Request failed')
      setMessages(prev => prev.slice(0, -1))
      setInput(q)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      {/* Settings bar */}
      <div style={{
        display: 'flex',
        gap: 12,
        padding: '10px 20px',
        background: '#f8fafc',
        borderBottom: '1px solid #e2e8f0',
        alignItems: 'center',
        fontSize: 12,
      }}>
        <label style={{ color: '#64748b', display: 'flex', alignItems: 'center', gap: 6 }}>
          Source
          <select
            value={source}
            onChange={e => setSource(e.target.value)}
            style={{ border: '1px solid #cbd5e0', borderRadius: 6, padding: '3px 8px', fontSize: 12, background: '#fff', cursor: 'pointer' }}
          >
            <option value="">All sources</option>
            {sources.map(s => <option key={s} value={s}>{s}</option>)}
          </select>
        </label>
        <label style={{ color: '#64748b', display: 'flex', alignItems: 'center', gap: 6 }}>
          Mode
          <select
            value={mode}
            onChange={e => setMode(e.target.value)}
            style={{ border: '1px solid #cbd5e0', borderRadius: 6, padding: '3px 8px', fontSize: 12, background: '#fff', cursor: 'pointer' }}
          >
            {['hybrid', 'vector', 'keyword', 'graph'].map(m => <option key={m} value={m}>{m}</option>)}
          </select>
        </label>
        <label style={{ color: '#64748b', display: 'flex', alignItems: 'center', gap: 6 }}>
          Top-K
          <input
            type="number"
            min={1}
            max={20}
            value={topK}
            onChange={e => setTopK(Number(e.target.value))}
            style={{ border: '1px solid #cbd5e0', borderRadius: 6, padding: '3px 6px', fontSize: 12, width: 56 }}
          />
        </label>
        {latencyInfo && (
          <span style={{ color: '#94a3b8', marginLeft: 'auto', fontFamily: 'monospace', fontSize: 11 }}>
            {latencyInfo}
          </span>
        )}
      </div>

      {/* Messages */}
      <div style={{ flex: 1, overflowY: 'auto', padding: '20px 24px', display: 'flex', flexDirection: 'column', gap: 16 }}>
        {messages.length === 0 && (
          <WelcomeBanner source={source} />
        )}
        {messages.map((msg, i) => (
          <MessageBubble key={i} msg={msg} />
        ))}
        {loading && <ThinkingBubble />}
        {error && (
          <div style={{ background: '#fff5f5', border: '1px solid #fed7d7', borderRadius: 10, padding: 12, color: '#e53e3e', fontSize: 13 }}>
            {error}
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <div style={{ padding: '14px 20px', borderTop: '1px solid #e2e8f0', background: '#fff' }}>
        <div style={{ display: 'flex', gap: 10, alignItems: 'flex-end' }}>
          <textarea
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); submit() } }}
            placeholder="Ask about services, Kafka topics, business rules, service dependencies…"
            rows={2}
            disabled={loading}
            style={{
              flex: 1,
              border: '1.5px solid #e2e8f0',
              borderRadius: 10,
              padding: '10px 14px',
              fontSize: 14,
              resize: 'none',
              outline: 'none',
              fontFamily: 'inherit',
              lineHeight: 1.5,
            }}
          />
          <button
            onClick={submit}
            disabled={loading || !input.trim()}
            style={{
              background: loading || !input.trim() ? '#e2e8f0' : '#1B2A4A',
              color: loading || !input.trim() ? '#94a3b8' : '#fff',
              border: 'none',
              borderRadius: 10,
              padding: '10px 20px',
              cursor: loading || !input.trim() ? 'not-allowed' : 'pointer',
              fontWeight: 700,
              fontSize: 14,
              height: 44,
              transition: 'background 0.15s',
            }}
          >
            {loading ? '…' : 'Send'}
          </button>
        </div>
        <p style={{ color: '#94a3b8', fontSize: 10, marginTop: 6 }}>
          Enter to send · Shift+Enter for new line
        </p>
      </div>
    </div>
  )
}

function MessageBubble({ msg }: { msg: Message }) {
  const isUser = msg.role === 'user'
  return (
    <div style={{ display: 'flex', justifyContent: isUser ? 'flex-end' : 'flex-start' }}>
      <div style={{
        maxWidth: '82%',
        background: isUser ? '#1B2A4A' : '#fff',
        color: isUser ? '#fff' : '#1a202c',
        border: isUser ? 'none' : '1px solid #e2e8f0',
        borderRadius: isUser ? '16px 16px 4px 16px' : '4px 16px 16px 16px',
        padding: '12px 16px',
        boxShadow: '0 1px 3px rgba(0,0,0,0.06)',
        fontSize: 14,
        lineHeight: 1.6,
      }}>
        <ReactMarkdown remarkPlugins={[remarkGfm]}>{msg.content}</ReactMarkdown>
        {msg.citations && msg.citations.length > 0 && (
          <CitationCard citations={msg.citations} />
        )}
      </div>
    </div>
  )
}

function ThinkingBubble() {
  return (
    <div style={{ display: 'flex', justifyContent: 'flex-start' }}>
      <div style={{
        background: '#fff',
        border: '1px solid #e2e8f0',
        borderRadius: '4px 16px 16px 16px',
        padding: '12px 18px',
        fontSize: 22,
        letterSpacing: 2,
        color: '#00A7B3',
      }}>
        •••
      </div>
    </div>
  )
}

function WelcomeBanner({ source }: { source: string }) {
  const examples = [
    `What Kafka topics does ${source} consume or produce?`,
    `What thresholds are used in ${source}?`,
    `Which external APIs does ${source} depend on?`,
    `How are processing state transitions handled?`,
  ]
  return (
    <div style={{ textAlign: 'center', padding: '40px 20px' }}>
      <div style={{ fontSize: 40, marginBottom: 12 }}>💬</div>
      <h3 style={{ color: '#1B2A4A', marginBottom: 8, fontSize: 20 }}>Codebase Q&amp;A</h3>
      <p style={{ color: '#64748b', marginBottom: 24, fontSize: 14 }}>
        Ask questions about the codebase. Answers are grounded in code summaries with citations.
      </p>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8, maxWidth: 600, margin: '0 auto' }}>
        {examples.map((ex, i) => (
          <div key={i} style={{
            background: '#f8fafc',
            border: '1px solid #e2e8f0',
            borderRadius: 8,
            padding: '10px 14px',
            fontSize: 13,
            color: '#4a5568',
            textAlign: 'left',
          }}>
            "{ex}"
          </div>
        ))}
      </div>
    </div>
  )
}
