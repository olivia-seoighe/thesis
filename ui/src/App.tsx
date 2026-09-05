import { useEffect, useState } from 'react'
import { deleteConversation, getConversation, listConversations } from './api/client'
import ChatInterface from './components/ChatInterface'
import ConversationSidebar from './components/ConversationSidebar'
import type { Conversation, Message } from './types'

export default function App() {
  const [conversations, setConversations] = useState<Conversation[]>([])
  const [activeConvId, setActiveConvId] = useState<string | null>(null)
  const [activeMessages, setActiveMessages] = useState<Message[]>([])
  const [mountKey, setMountKey] = useState(0)

  useEffect(() => {
    listConversations().then(setConversations).catch(console.error)
  }, [])

  const handleSelect = async (id: string) => {
    setMountKey(k => k + 1)
    setActiveConvId(id)
    try {
      const conv = await getConversation(id)
      setActiveMessages(conv.messages)
    } catch (error) {
      console.error(error)
      setActiveMessages([])
    }
  }

  const handleNew = () => {
    setMountKey(k => k + 1)
    setActiveConvId(null)
    setActiveMessages([])
  }

  const handleDelete = async (id: string) => {
    await deleteConversation(id).catch(console.error)
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
        </header>

        <div style={{ flex: 1, overflow: 'hidden' }}>
          <ChatInterface
            key={mountKey}
            conversationId={activeConvId}
            initialMessages={activeMessages}
            onConversationCreated={handleConversationCreated}
          />
        </div>
      </main>
    </div>
  )
}
