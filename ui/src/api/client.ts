import type { Conversation, QueryResponse } from '../types'

const GEN_BASE = '/api/generation'
const RET_BASE = '/api/retrieval'

export async function sendQuery(
  query: string,
  opts: { source?: string; topK?: number; conversationId?: string; model?: string; mode?: string }
): Promise<QueryResponse> {
  const res = await fetch(`${GEN_BASE}/query`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      query,
      ...(opts.source ? { source: opts.source } : {}),
      top_k: opts.topK ?? 5,
      mode: opts.mode ?? 'hybrid',
      conversation_id: opts.conversationId,
      model: opts.model,
    }),
  })
  if (!res.ok) throw new Error(`Query failed: ${res.status} ${await res.text()}`)
  return res.json()
}

export async function getSources(): Promise<string[]> {
  const res = await fetch(`${RET_BASE}/sources`)
  if (!res.ok) throw new Error(`Sources failed: ${res.status}`)
  const data = await res.json()
  return data.sources ?? []
}

export async function listConversations(): Promise<Conversation[]> {
  const res = await fetch(`${GEN_BASE}/conversations`)
  if (!res.ok) throw new Error(`List conversations failed: ${res.status}`)
  return res.json()
}

export async function getConversation(id: string): Promise<Conversation> {
  const res = await fetch(`${GEN_BASE}/conversations/${id}`)
  if (!res.ok) throw new Error(`Get conversation failed: ${res.status}`)
  return res.json()
}

export async function deleteConversation(id: string): Promise<void> {
  const res = await fetch(`${GEN_BASE}/conversations/${id}`, { method: 'DELETE' })
  if (!res.ok) throw new Error(`Delete conversation failed: ${res.status}`)
}

export async function checkHealth(): Promise<boolean> {
  try {
    const res = await fetch(`${GEN_BASE}/health`)
    return res.ok
  } catch {
    return false
  }
}
