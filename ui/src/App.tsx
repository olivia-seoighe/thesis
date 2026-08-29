import { useEffect, useState } from 'react'
import { deleteConversation, getConversation, listConversations } from './api/client'
import ChatInterface from './components/ChatInterface'
import ConversationSidebar from './components/ConversationSidebar'
import EmbeddingViz from './components/EmbeddingViz'
import type { Conversation, Message } from './types'

type Tab = 'chat' | 'viz'

export default function App() {
  const [conversations, setConversations] = useState<Conversation[]>([])
  const [activeConvId, setActiveConvId] = useState<string | null>(null)
  const [activeMessages, setActiveMessages] = useState<Message[]>([])
  const [activeTab, setActiveTab] = useState<Tab>('chat')
  const [mountKey, setMountKey] = useState(0)

  useEffect(() => {
    listConversations().then(setConversations).catch(() => {})
  }, [])

  const handleSelect = async (id: string) => {
    setMountKey(k => k + 1)
    setActiveConvId(id)
    setActiveTab('chat')
    try {
      const conv = await getConversation(id)
      setActiveMessages(conv.messages)
    } catch {
      setActiveMessages([])
    }
  }

  const handleNew = () => {
    setMountKey(k => k + 1)
    setActiveConvId(null)
    setActiveMessages([])
    setActiveTab('chat')
  }

  const handleDelete = async (id: string) => {
    await deleteConversation(id).catch(() => {})
    setConversations(prev => prev.filter(c => c.id !== id))
    if (activeConvId === id) handleNew()
  }

  const handleConversationCreated = async (id: string) => {
    setActiveConvId(id)
    const updated = await listConversations().catch(() => conversations)
    setConversations(updated)
  }

  return (
    <div style={{ display: 'flex', height: '100vh', overflow: 'hidden', background: '#f0f4f8' }}>
      <ConversationSidebar
        conversations={conversations}
        activeId={activeConvId}
        onSelect={handleSelect}
        onNew={handleNew}
        onDelete={handleDelete}
      />

      <main style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
        {/* Top nav */}
        <header style={{
          background: '#fff',
          borderBottom: '1px solid #e2e8f0',
          padding: '0 24px',
          display: 'flex',
          alignItems: 'center',
          gap: 0,
          height: 52,
        }}>
          <div style={{ flex: 1, display: 'flex', alignItems: 'center', gap: 12 }}>
            <span style={{ color: '#1B2A4A', fontWeight: 700, fontSize: 16 }}>
              {activeConvId ? 'Conversation' : 'New conversation'}
            </span>
          </div>
          <nav style={{ display: 'flex', gap: 0 }}>
            {(['chat', 'viz'] as Tab[]).map(tab => (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                style={{
                  border: 'none',
                  borderBottom: activeTab === tab ? '2px solid #00A7B3' : '2px solid transparent',
                  background: 'none',
                  padding: '14px 18px 12px',
                  cursor: 'pointer',
                  fontWeight: activeTab === tab ? 700 : 400,
                  color: activeTab === tab ? '#00A7B3' : '#64748b',
                  fontSize: 14,
                  transition: 'all 0.15s',
                }}
              >
                {tab === 'chat' ? '💬 Chat' : '📊 Embeddings'}
              </button>
            ))}
          </nav>
        </header>

        {/* Content */}
        <div style={{ flex: 1, overflow: 'hidden' }}>
          {activeTab === 'chat' ? (
            <ChatInterface
              key={mountKey}
              conversationId={activeConvId}
              initialMessages={activeMessages}
              onConversationCreated={handleConversationCreated}
            />
          ) : (
            <EmbeddingViz />
          )}
        </div>
      </main>
    </div>
  )
}
