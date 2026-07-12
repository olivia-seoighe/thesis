import type { Conversation } from '../types'

interface Props {
  conversations: Conversation[]
  activeId: string | null
  onSelect: (id: string) => void
  onNew: () => void
  onDelete: (id: string) => void
}

export default function ConversationSidebar({ conversations, activeId, onSelect, onNew, onDelete }: Props) {
  return (
    <aside style={{
      width: 260,
      minWidth: 260,
      background: '#1B2A4A',
      display: 'flex',
      flexDirection: 'column',
      height: '100vh',
      overflowY: 'auto',
    }}>
      {/* Header */}
      <div style={{ padding: '20px 16px 12px', borderBottom: '1px solid rgba(255,255,255,0.08)' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 14 }}>
          <SignifyLogo />
          <div>
            <div style={{ color: '#fff', fontWeight: 700, fontSize: 14, lineHeight: 1.2 }}>Codebase Q&amp;A</div>
          </div>
        </div>
        <button
          onClick={onNew}
          style={{
            width: '100%',
            background: '#00A7B3',
            color: '#fff',
            border: 'none',
            borderRadius: 8,
            padding: '9px 0',
            cursor: 'pointer',
            fontWeight: 600,
            fontSize: 13,
          }}
        >
          + New conversation
        </button>
      </div>

      {/* Conversation list */}
      <div style={{ flex: 1, padding: '8px 0' }}>
        {conversations.length === 0 && (
          <p style={{ color: 'rgba(255,255,255,0.35)', fontSize: 12, padding: '12px 16px' }}>
            No conversations yet
          </p>
        )}
        {conversations.map((conv) => (
          <div
            key={conv.id}
            onClick={() => onSelect(conv.id)}
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              padding: '9px 16px',
              cursor: 'pointer',
              background: activeId === conv.id ? 'rgba(0,167,179,0.18)' : 'transparent',
              borderLeft: activeId === conv.id ? '3px solid #00A7B3' : '3px solid transparent',
            }}
          >
            <span style={{
              color: activeId === conv.id ? '#fff' : 'rgba(255,255,255,0.65)',
              fontSize: 13,
              overflow: 'hidden',
              textOverflow: 'ellipsis',
              whiteSpace: 'nowrap',
              flex: 1,
            }}>
              {conv.title}
            </span>
            <button
              onClick={(e) => { e.stopPropagation(); onDelete(conv.id) }}
              style={{
                background: 'none',
                border: 'none',
                color: 'rgba(255,255,255,0.3)',
                cursor: 'pointer',
                fontSize: 16,
                padding: '0 0 0 8px',
              }}
              title="Delete"
            >
              ×
            </button>
          </div>
        ))}
      </div>

    </aside>
  )
}

function SignifyLogo() {
  return (
    <svg width="32" height="32" viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg">
      {/* Signify Health 'S' leaf motif */}
      <path d="M8 22 C8 14 16 10 16 6 C16 2 12 2 10 4" stroke="#8ABF34" strokeWidth="3.5" strokeLinecap="round" fill="none"/>
      <path d="M24 10 C24 18 16 22 16 26 C16 30 20 30 22 28" stroke="#00A7B3" strokeWidth="3.5" strokeLinecap="round" fill="none"/>
    </svg>
  )
}
