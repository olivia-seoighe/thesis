export interface Citation {
  title: string
  url: string
  score: number
  chunk_text: string
  source_code: string
  metadata: Record<string, unknown>
}

export interface Message {
  role: 'user' | 'assistant'
  content: string
  citations?: Citation[]
  timestamp: string
}

export interface Conversation {
  id: string
  title: string
  messages: Message[]
  created_at: string
}

export interface QueryResponse {
  answer: string
  citations: Citation[]
  conversation_id: string
  model_used: string
  latency_ms: number
  retrieval_latency_ms: number
  generation_latency_ms: number
}
