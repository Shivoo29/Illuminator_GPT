import { Plus, MessageSquare, Trash2, Edit2, Check, X, MoreVertical } from "lucide-react";
import { useState } from "react";

interface Chat {
  id: string;
  title: string;
  updated_at: string;
  message_count: number;
}

interface ChatSidebarProps {
  chats: Chat[];
  currentChatId: string | null;
  onSelectChat: (id: string) => void;
  onNewChat: () => void;
  onDeleteChat: (id: string) => void;
  onRenameChat: (id: string, newTitle: string) => void;
}

export default function ChatSidebar({
  chats,
  currentChatId,
  onSelectChat,
  onNewChat,
  onDeleteChat,
  onRenameChat,
}: ChatSidebarProps) {
  const [renamingId, setRenamingId] = useState<string | null>(null);
  const [renameValue, setRenameValue] = useState("");

  const handleStartRename = (e: React.MouseEvent, chat: Chat) => {
    e.stopPropagation();
    setRenamingId(chat.id);
    setRenameValue(chat.title);
  };

  const handleConfirmRename = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (renamingId && renameValue.trim()) {
      onRenameChat(renamingId, renameValue.trim());
    }
    setRenamingId(null);
  };

  const handleCancelRename = (e: React.MouseEvent) => {
    e.stopPropagation();
    setRenamingId(null);
  };

  return (
    <div className="w-64 border-r flex flex-col shrink-0" style={{ borderColor: 'var(--color-border-primary)', background: 'var(--color-bg-secondary)' }}>
      <div className="p-4 shrink-0">
        <button
          onClick={onNewChat}
          className="w-full py-2.5 rounded-xl text-sm font-semibold flex items-center justify-center gap-2 transition-all cursor-pointer"
          style={{ 
            background: 'var(--color-bg-elevated)', 
            color: 'var(--color-text-primary)',
            border: '1px solid var(--color-border-primary)' 
          }}
          onMouseEnter={(e) => (e.currentTarget.style.background = 'var(--color-bg-hover)')}
          onMouseLeave={(e) => (e.currentTarget.style.background = 'var(--color-bg-elevated)')}
        >
          <Plus className="w-4 h-4" /> New Chat
        </button>
      </div>

      <div className="flex-1 overflow-y-auto px-3 py-2 space-y-1">
        {chats.map((chat) => (
          <div
            key={chat.id}
            onClick={() => onSelectChat(chat.id)}
            className="group relative flex items-center px-3 py-2.5 rounded-xl text-sm transition-all cursor-pointer overflow-hidden"
            style={{
              background: currentChatId === chat.id ? 'var(--color-bg-hover)' : 'transparent',
              color: currentChatId === chat.id ? 'var(--color-text-primary)' : 'var(--color-text-secondary)',
            }}
          >
            <MessageSquare className="w-4 h-4 mr-3 shrink-0 opacity-60" />
            
            {renamingId === chat.id ? (
              <div className="flex-1 flex items-center gap-1 z-10" onClick={e => e.stopPropagation()}>
                <input
                  autoFocus
                  className="w-full bg-transparent border-none outline-none text-sm p-0"
                  value={renameValue}
                  onChange={(e) => setRenameValue(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter') handleConfirmRename(e as any);
                    if (e.key === 'Escape') handleCancelRename(e as any);
                  }}
                />
                <button onClick={handleConfirmRename} className="p-0.5 hover:text-green-500"><Check className="w-3.5 h-3.5" /></button>
                <button onClick={handleCancelRename} className="p-0.5 hover:text-red-500"><X className="w-3.5 h-3.5" /></button>
              </div>
            ) : (
              <span className="flex-1 truncate mr-6">{chat.title}</span>
            )}

            {!renamingId && (
              <div className="absolute right-2 opacity-0 group-hover:opacity-100 flex items-center transition-all">
                <button
                  onClick={(e) => handleStartRename(e, chat)}
                  className="p-1 hover:text-blue-400"
                >
                  <Edit2 className="w-3.5 h-3.5" />
                </button>
                <button
                  onClick={(e) => { e.stopPropagation(); onDeleteChat(chat.id); }}
                  className="p-1 hover:text-red-500"
                >
                  <Trash2 className="w-3.5 h-3.5" />
                </button>
              </div>
            )}
          </div>
        ))}

        {chats.length === 0 && (
          <div className="text-center py-10 opacity-40">
            <p className="text-xs">No chat history</p>
          </div>
        )}
      </div>
    </div>
  );
}
