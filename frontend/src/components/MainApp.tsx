import { useState, useRef, useEffect } from "react";
import { api } from "../utils/api";
import DocumentUpload from "./DocumentUpload";
import PodcastGenerator from "./PodcastGenerator";
import ChatSidebar from "./ChatSidebar";
import {
  Send,
  Paperclip,
  FileText,
  Trash2,
  X,
  ChevronDown,
  ChevronUp,
  Sparkles,
  MessageSquare,
  BookOpen,
  Lightbulb,
  Search,
  Mic,
  PanelRightOpen,
  PanelRightClose,
  PlusCircle,
} from "lucide-react";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import "katex/dist/katex.min.css";

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: Array<{
    id: string;
    content: string;
    metadata: any;
  }>;
}

interface Document {
  id: string;
  filename: string;
  file_type: string;
  chunk_count: number;
}

interface Chat {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
  message_count: number;
}

const SUGGESTED_PROMPTS = [
  { icon: BookOpen, text: "Summarize this document" },
  { icon: Search, text: "What are the key points?" },
  { icon: Lightbulb, text: "Explain the main concepts" },
  { icon: MessageSquare, text: "Compare the documents" },
];

export default function MainApp() {
  // Chat state
  const [chats, setChats] = useState<Chat[]>([]);
  const [currentChatId, setCurrentChatId] = useState<string | null>(null);
  
  // Content state
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [documents, setDocuments] = useState<Document[]>([]);
  const [allDocuments, setAllDocuments] = useState<Document[]>([]);
  
  // UI state
  const [showUpload, setShowUpload] = useState(false);
  const [showDocLibrary, setShowDocLibrary] = useState(false);
  const [showPodcast, setShowPodcast] = useState(false);
  const [selectedDocumentId, setSelectedDocumentId] = useState<string | null>(null);
  const [docPanelOpen, setDocPanelOpen] = useState(true);
  const [expandedSources, setExpandedSources] = useState<Set<string>>(new Set());
  
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Initial load
  useEffect(() => {
    loadChats();
    loadAllDocuments();
  }, []);

  // Scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Load chat content when chat id changes
  useEffect(() => {
    if (currentChatId) {
      loadChatDetails(currentChatId);
    } else {
      setMessages([]);
      setDocuments([]);
    }
  }, [currentChatId]);

  const loadChats = async () => {
    try {
      const data = await api.get<Chat[]>("/chats");
      setChats(data);
    } catch (error) {
      console.error("Failed to load chats:", error);
    }
  };

  const loadAllDocuments = async () => {
    try {
      const docs = await api.get<Document[]>("/documents");
      setAllDocuments(docs);
    } catch (error) {
      console.error("Failed to load documents:", error);
    }
  };

  const loadChatDetails = async (chatId: string) => {
    try {
      const chat = await api.get<any>(`/chats/${chatId}`);
      setMessages(chat.messages || []);
      
      // Load docs for this chat
      const chatDocs = await api.get<Document[]>(`/documents`, { chat_id: chatId });
      setDocuments(chatDocs);
    } catch (error) {
      console.error("Failed to load chat details:", error);
    }
  };

  const handleNewChat = () => {
    setCurrentChatId(null);
    setMessages([]);
    setDocuments([]);
    setSelectedDocumentId(null);
    if (textareaRef.current) textareaRef.current.focus();
  };

  const deleteChat = async (id: string) => {
    try {
      await api.delete(`/chats/${id}`);
      if (currentChatId === id) {
        handleNewChat();
      }
      loadChats();
    } catch (error) {
      console.error("Failed to delete chat:", error);
    }
  };

  const renameChat = async (id: string, title: string) => {
    try {
      await api.put(`/chats/${id}/title`, { title });
      loadChats();
    } catch (error) {
      console.error("Failed to rename chat:", error);
    }
  };

  const handleSubmit = async (e?: React.FormEvent) => {
    e?.preventDefault();
    if (!input.trim() || loading) return;

    let chatId = currentChatId;
    const question = input.trim();
    
    // Reset textarea height
    if (textareaRef.current) textareaRef.current.style.height = "auto";
    setInput("");
    setLoading(true);

    try {
      // 1. If no active chat, create one first
      if (!chatId) {
        const newChat = await api.post<Chat>("/chats", { first_message: question });
        chatId = newChat.id;
        setCurrentChatId(chatId);
        loadChats();
      } else {
        // Add user message to history in backend
        await api.post(`/chats/${chatId}/messages`, {
          role: "user",
          content: question
        });
        
        // Update local UI
        const userMsg: Message = { id: Date.now().toString(), role: "user", content: question };
        setMessages(prev => [...prev, userMsg]);
      }

      // 2. Query RAG with current chat context
      const response = await api.post<{ answer: string; sources: any[] }>("/query", {
        question: question,
        chat_id: chatId,
        filter_document_id: selectedDocumentId,
        stream: false // Non-streaming for simplicity in this refactor
      });

      // 3. Save assistant message in backend
      await api.post(`/chats/${chatId}/messages`, {
        role: "assistant",
        content: response.answer,
        references: response.sources
      });

      // 4. Update local UI
      const assistantMsg: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: response.answer,
        sources: response.sources
      };
      setMessages(prev => [...prev, assistantMsg]);

      // 5. Trigger auto-title if it was the first message
      const chatData = chats.find(c => c.id === chatId);
      if (chatData && (chatData.title === "New Chat" || messages.length === 0)) {
        const titleData = await api.post<{title: string}>("/chats/generate_title", { content: question });
        if (titleData.title) {
          await api.put(`/chats/${chatId}/title`, { title: titleData.title });
          loadChats();
        }
      }

    } catch (error) {
      console.error("Error in submit:", error);
      setMessages(prev => [...prev, {
        id: Date.now().toString(),
        role: "assistant",
        content: "Sorry, I encountered an error. Please try again."
      }]);
    } finally {
      setLoading(false);
    }
  };

  const addDocToChat = async (docId: string) => {
    if (!currentChatId) {
      // Create chat first if needed
      const newChat = await api.post<Chat>("/chats", { first_message: "Reading document..." });
      setCurrentChatId(newChat.id);
      await api.post(`/chats/${newChat.id}/documents`, { document_id: docId });
      loadChats();
    } else {
      await api.post(`/chats/${currentChatId}/documents`, { document_id: docId });
    }
    loadChatDetails(currentChatId || ""); // chatId should exist now or will be updated by effect
    setShowDocLibrary(false);
  };

  const removeDocFromChat = async (docId: string) => {
    if (!currentChatId) return;
    try {
      await api.delete(`/chats/${currentChatId}/documents/${docId}`);
      loadChatDetails(currentChatId);
      if (selectedDocumentId === docId) setSelectedDocumentId(null);
    } catch (error) {
      console.error("Failed to remove document:", error);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const handleTextareaInput = () => {
    const el = textareaRef.current;
    if (el) {
      el.style.height = "auto";
      el.style.height = Math.min(el.scrollHeight, 150) + "px";
    }
  };

  const toggleSources = (messageId: string) => {
    setExpandedSources((prev) => {
      const next = new Set(prev);
      next.has(messageId) ? next.delete(messageId) : next.add(messageId);
      return next;
    });
  };

  const handlePromptClick = (text: string) => {
    setInput(text);
    textareaRef.current?.focus();
  };

  return (
    <div className="flex flex-1 h-full overflow-hidden">
      {/* Sidebar - Chat History */}
      <ChatSidebar 
        chats={chats}
        currentChatId={currentChatId}
        onSelectChat={setCurrentChatId}
        onNewChat={handleNewChat}
        onDeleteChat={deleteChat}
        onRenameChat={renameChat}
      />

      {/* Main chat area */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Header */}
        <div
          className="h-14 flex items-center justify-between px-5 border-b shrink-0"
          style={{ borderColor: "var(--color-border-primary)", background: "var(--color-bg-primary)" }}
        >
          <div className="flex items-center gap-2">
            <h2 className="text-sm font-semibold" style={{ color: "var(--color-text-primary)" }}>
              {currentChatId ? chats.find(c => c.id === currentChatId)?.title : "New Chat"}
            </h2>
            {selectedDocumentId && (
              <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium" style={{ background: "var(--color-accent-bg)", color: "var(--color-accent-primary)" }}>
                <FileText className="w-3 h-3" />
                <span className="max-w-[120px] truncate">{documents.find((d) => d.id === selectedDocumentId)?.filename}</span>
                <button onClick={() => setSelectedDocumentId(null)} className="hover:opacity-70 cursor-pointer">
                  <X className="w-3 h-3" />
                </button>
              </div>
            )}
          </div>
          <button
            onClick={() => setDocPanelOpen(!docPanelOpen)}
            className="p-2 rounded-lg transition-colors cursor-pointer"
            style={{ color: "var(--color-text-tertiary)" }}
            onMouseEnter={(e) => (e.currentTarget.style.background = "var(--color-bg-hover)")}
            onMouseLeave={(e) => (e.currentTarget.style.background = "transparent")}
          >
            {docPanelOpen ? <PanelRightClose className="w-5 h-5" /> : <PanelRightOpen className="w-5 h-5" />}
          </button>
        </div>

        {/* Messages / Empty state */}
        <div className="flex-1 overflow-y-auto">
          {messages.length === 0 ? (
            <div className="h-full flex flex-col items-center justify-center px-6 animate-fade-in">
              <div className="w-16 h-16 rounded-2xl flex items-center justify-center mb-6" style={{ background: "var(--color-accent-bg)" }}>
                <Sparkles className="w-8 h-8" style={{ color: "var(--color-accent-primary)" }} />
              </div>
              <h2 className="text-xl font-bold mb-2" style={{ color: "var(--color-text-primary)" }}>
                How can I help you today?
              </h2>
              <p className="text-sm mb-8 text-center max-w-sm" style={{ color: "var(--color-text-tertiary)" }}>
                {currentChatId ? "Ask questions about documents attached to this chat." : "Start a new conversation or upload a document."}
              </p>
              <div className="grid grid-cols-2 gap-2.5 w-full max-w-md">
                {SUGGESTED_PROMPTS.map((prompt) => (
                  <button
                    key={prompt.text}
                    onClick={() => handlePromptClick(prompt.text)}
                    className="flex items-center gap-2.5 p-3.5 rounded-xl text-left text-sm transition-all duration-200 border cursor-pointer"
                    style={{
                      background: "var(--color-bg-surface)",
                      borderColor: "var(--color-border-primary)",
                      color: "var(--color-text-secondary)",
                    }}
                    onMouseEnter={(e) => {
                      e.currentTarget.style.borderColor = "var(--color-border-hover)";
                      e.currentTarget.style.background = "var(--color-bg-hover)";
                    }}
                    onMouseLeave={(e) => {
                      e.currentTarget.style.borderColor = "var(--color-border-primary)";
                      e.currentTarget.style.background = "var(--color-bg-surface)";
                    }}
                  >
                    <prompt.icon className="w-4 h-4 shrink-0" style={{ color: "var(--color-text-muted)" }} />
                    <span>{prompt.text}</span>
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <div className="max-w-3xl mx-auto px-5 py-6 space-y-5">
              {messages.map((message) => (
                <div key={message.id} className={`flex ${message.role === "user" ? "justify-end" : "justify-start"} animate-fade-in-up`}>
                  <div
                    className={`max-w-[85%] rounded-2xl px-4 py-3 ${message.role === "user" ? "user-message-bg text-white rounded-br-md" : "rounded-bl-md"}`}
                    style={
                      message.role === "assistant"
                        ? { background: "var(--color-assistant-bubble)", border: `1px solid var(--color-assistant-bubble-border)` }
                        : {}
                    }
                  >
                    {message.role === "assistant" ? (
                      <div className="markdown-prose">
                        <ReactMarkdown
                          remarkPlugins={[remarkGfm, remarkMath]}
                          rehypePlugins={[rehypeKatex]}
                        >
                          {message.content}
                        </ReactMarkdown>
                      </div>
                    ) : (
                      <p className="text-sm whitespace-pre-wrap leading-relaxed">
                        {message.content}
                      </p>
                    )}

                    {/* Sources */}
                    {message.sources && message.sources.length > 0 && (
                      <div className="mt-3 pt-3" style={{ borderTop: `1px solid var(--color-border-primary)` }}>
                        <button
                          onClick={() => toggleSources(message.id)}
                          className="flex items-center gap-1.5 text-xs font-medium cursor-pointer"
                          style={{ color: "var(--color-text-muted)" }}
                        >
                          <BookOpen className="w-3 h-3" />
                          {message.sources.length} sources
                          {expandedSources.has(message.id) ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
                        </button>

                        {expandedSources.has(message.id) && (
                          <div className="mt-2 space-y-2 animate-fade-in">
                            {message.sources.slice(0, 3).map((source, i) => (
                              <div key={i} className="p-2.5 rounded-lg text-xs" style={{ background: "var(--color-bg-tertiary)" }}>
                                <p className="line-clamp-2" style={{ color: "var(--color-text-secondary)" }}>{source.content}</p>
                                <p className="mt-1 font-medium" style={{ color: "var(--color-text-muted)" }}>{source.metadata?.filename}</p>
                              </div>
                            ))}
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                </div>
              ))}

              {/* Loading */}
              {loading && (
                <div className="flex justify-start animate-fade-in">
                  <div className="rounded-2xl rounded-bl-md px-4 py-3" style={{ background: "var(--color-assistant-bubble)", border: `1px solid var(--color-assistant-bubble-border)` }}>
                    <div className="flex gap-1.5">
                      <div className="w-2 h-2 rounded-full typing-dot" style={{ background: "var(--color-text-muted)" }} />
                      <div className="w-2 h-2 rounded-full typing-dot" style={{ background: "var(--color-text-muted)" }} />
                      <div className="w-2 h-2 rounded-full typing-dot" style={{ background: "var(--color-text-muted)" }} />
                    </div>
                  </div>
                </div>
              )}

              <div ref={messagesEndRef} />
            </div>
          )}
        </div>

        {/* Input bar */}
        <div className="px-5 pb-5 pt-2 shrink-0" style={{ background: "var(--color-bg-primary)" }}>
          <div
            className="max-w-3xl mx-auto flex items-end gap-2 p-2 rounded-2xl border transition-all"
            style={{
              background: "var(--color-bg-input)",
              borderColor: "var(--color-border-secondary)",
            }}
            onFocus={(e) => (e.currentTarget.style.borderColor = "var(--color-border-focus)")}
            onBlur={(e) => (e.currentTarget.style.borderColor = "var(--color-border-secondary)")}
          >
            <button
              onClick={() => setShowDocLibrary(true)}
              className="p-2 rounded-xl transition-colors shrink-0 cursor-pointer"
              style={{ color: "var(--color-text-muted)" }}
              onMouseEnter={(e) => (e.currentTarget.style.background = "var(--color-bg-hover)")}
              onMouseLeave={(e) => (e.currentTarget.style.background = "transparent")}
            >
              <Paperclip className="w-5 h-5" />
            </button>

            <textarea
              ref={textareaRef}
              rows={1}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onInput={handleTextareaInput}
              onKeyDown={handleKeyDown}
              placeholder={selectedDocumentId ? "Ask about this document..." : (documents.length > 0 ? "Ask about chat documents..." : "Ask a question...")}
              disabled={loading}
              className="flex-1 bg-transparent border-none outline-none resize-none text-sm py-2 px-1 max-h-[150px] disabled:opacity-50"
              style={{ color: "var(--color-text-primary)" }}
            />

            <button
              onClick={() => handleSubmit()}
              disabled={loading || !input.trim()}
              className="p-2 rounded-xl transition-all shrink-0 disabled:opacity-30 cursor-pointer"
              style={{
                background: input.trim() ? "linear-gradient(135deg, #6366f1, #8b5cf6)" : "var(--color-bg-hover)",
                color: input.trim() ? "white" : "var(--color-text-muted)",
              }}
            >
              <Send className="w-5 h-5" />
            </button>
          </div>
          <p className="text-center text-[11px] mt-2 font-medium" style={{ color: "var(--color-text-muted)" }}>
            Everything runs locally. Your data never leaves this device.
          </p>
        </div>
      </div>

      {/* Document panel (right side) */}
      {docPanelOpen && (
        <div
          className="w-72 border-l flex flex-col shrink-0 animate-slide-in-right"
          style={{
            borderColor: "var(--color-border-primary)",
            background: "var(--color-bg-secondary)",
          }}
        >
          <div className="h-14 flex items-center justify-between px-4 border-b shrink-0" style={{ borderColor: "var(--color-border-primary)" }}>
            <h3 className="text-sm font-semibold" style={{ color: "var(--color-text-primary)" }}>Chat Context</h3>
            <button
              onClick={() => setShowDocLibrary(true)}
              className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-xs font-semibold transition-colors cursor-pointer"
              style={{ background: "var(--color-accent-bg)", color: "var(--color-accent-primary)" }}
            >
              <PlusCircle className="w-3.5 h-3.5" /> Attach
            </button>
          </div>

          <div className="flex-1 overflow-y-auto p-3 space-y-1">
            <button
              onClick={() => setSelectedDocumentId(null)}
              className="w-full text-left px-3 py-2 rounded-lg text-sm transition-colors cursor-pointer"
              style={{
                background: selectedDocumentId === null ? "var(--color-accent-bg)" : "transparent",
                color: selectedDocumentId === null ? "var(--color-accent-primary)" : "var(--color-text-secondary)",
              }}
            >
              All Chat Context
            </button>

            {documents.map((doc) => (
              <div
                key={doc.id}
                className="group flex items-center px-3 py-2 rounded-lg text-sm transition-colors cursor-pointer"
                style={{
                  background: selectedDocumentId === doc.id ? "var(--color-accent-bg)" : "transparent",
                  color: selectedDocumentId === doc.id ? "var(--color-accent-primary)" : "var(--color-text-secondary)",
                }}
                onClick={() => setSelectedDocumentId(doc.id)}
              >
                <FileText className="w-4 h-4 mr-2 shrink-0" />
                <span className="flex-1 truncate">{doc.filename}</span>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    removeDocFromChat(doc.id);
                  }}
                  className="opacity-0 group-hover:opacity-100 transition-opacity cursor-pointer p-1 rounded hover:bg-red-500/10"
                  style={{ color: "var(--color-error)" }}
                >
                  <Trash2 className="w-3.5 h-3.5" />
                </button>
              </div>
            ))}

            {documents.length === 0 && (
              <div className="text-center py-12 px-4 opacity-40">
                <FileText className="w-8 h-8 mx-auto mb-2" />
                <p className="text-xs">No documents attached to this chat yet.</p>
              </div>
            )}
          </div>

          {/* Podcast button */}
          <div className="p-3 border-t" style={{ borderColor: "var(--color-border-primary)" }}>
            <button
              onClick={() => setShowPodcast(true)}
              disabled={!selectedDocumentId}
              className="w-full py-2.5 rounded-xl text-xs font-semibold flex items-center justify-center gap-2 transition-all disabled:opacity-40 cursor-pointer"
              style={{ background: "var(--color-bg-tertiary)", color: "var(--color-text-secondary)" }}
            >
              <Mic className="w-4 h-4" /> Generate Podcast
            </button>
          </div>
        </div>
      )}

      {/* Doc Library Modal */}
      {showDocLibrary && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4" style={{ background: "var(--color-overlay)" }} onClick={() => setShowDocLibrary(false)}>
          <div
            className="w-full max-w-xl bg-elevated rounded-2xl p-6 animate-scale-in flex flex-col max-h-[80vh]"
            style={{ background: "var(--color-bg-elevated)", boxShadow: "var(--shadow-lg)" }}
            onClick={(e) => e.stopPropagation()}
          >
             <div className="flex items-center justify-between mb-5 shrink-0">
               <div>
                 <h2 className="text-lg font-bold" style={{ color: "var(--color-text-primary)" }}>Document Library</h2>
                 <p className="text-xs mt-1" style={{ color: "var(--color-text-muted)" }}>Select a document to add it to your current chat context.</p>
               </div>
               <div className="flex items-center gap-2">
                 <button 
                  onClick={() => { setShowDocLibrary(false); setShowUpload(true); }}
                  className="px-3 py-1.5 rounded-xl text-xs font-semibold transition-all cursor-pointer border"
                  style={{ borderColor: 'var(--color-border-primary)', color: 'var(--color-text-secondary)' }}
                 >
                   + Upload New
                 </button>
                 <button onClick={() => setShowDocLibrary(false)} className="p-1.5 rounded-lg transition-colors cursor-pointer" style={{ color: "var(--color-text-tertiary)" }}>
                  <X className="w-5 h-5" />
                </button>
               </div>
             </div>

             <div className="flex-1 overflow-y-auto space-y-2 pr-2">
                {allDocuments.map(doc => (
                  <div 
                    key={doc.id} 
                    className="flex items-center justify-between p-3 rounded-xl border transition-all hover:border-accent-primary group"
                    style={{ background: 'var(--color-bg-surface)', borderColor: 'var(--color-border-primary)' }}
                  >
                    <div className="flex items-center gap-3">
                      <div className="w-8 h-8 rounded-lg flex items-center justify-center bg-accent-bg text-accent-primary">
                        <FileText className="w-4 h-4" />
                      </div>
                      <div>
                        <p className="text-sm font-medium leading-none mb-1" style={{ color: 'var(--color-text-primary)' }}>{doc.filename}</p>
                        <p className="text-[10px]" style={{ color: 'var(--color-text-muted)' }}>{doc.file_type.toUpperCase()} • {doc.chunk_count} chunks</p>
                      </div>
                    </div>
                    
                    {documents.some(d => d.id === doc.id) ? (
                      <span className="text-[10px] font-bold uppercase tracking-wider px-2 py-1 rounded bg-green-500/10 text-green-500">Attached</span>
                    ) : (
                      <button 
                        onClick={() => addDocToChat(doc.id)}
                        className="p-1.5 rounded-lg opacity-0 group-hover:opacity-100 transition-all hover:bg-accent-bg text-accent-primary cursor-pointer"
                      >
                        <PlusCircle className="w-5 h-5" />
                      </button>
                    )}
                  </div>
                ))}

                {allDocuments.length === 0 && (
                  <div className="text-center py-12 opacity-40">
                    <p className="text-sm">No documents found in library.</p>
                  </div>
                )}
             </div>
          </div>
        </div>
      )}

      {/* Upload Modal */}
      {showUpload && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4" style={{ background: "var(--color-overlay)" }} onClick={() => setShowUpload(false)}>
          <div
            className="w-full max-w-md rounded-2xl p-6 animate-scale-in"
            style={{ background: "var(--color-bg-elevated)", boxShadow: "var(--shadow-lg)" }}
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between mb-5">
              <h2 className="text-lg font-bold" style={{ color: "var(--color-text-primary)" }}>Upload Document</h2>
              <button onClick={() => setShowUpload(false)} className="p-1.5 rounded-lg transition-colors cursor-pointer" style={{ color: "var(--color-text-tertiary)" }}>
                <X className="w-5 h-5" />
              </button>
            </div>
            <DocumentUpload 
              chatId={currentChatId}
              onComplete={() => { loadAllDocuments(); loadChatDetails(currentChatId || ""); setShowUpload(false); }} 
            />
          </div>
        </div>
      )}

      {/* Podcast Modal */}
      {showPodcast && selectedDocumentId && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4" style={{ background: "var(--color-overlay)" }} onClick={() => setShowPodcast(false)}>
          <div
            className="w-full max-w-md rounded-2xl p-6 animate-scale-in"
            style={{ background: "var(--color-bg-elevated)", boxShadow: "var(--shadow-lg)" }}
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between mb-5">
              <h2 className="text-lg font-bold" style={{ color: "var(--color-text-primary)" }}>Generate Podcast</h2>
              <button onClick={() => setShowPodcast(false)} className="p-1.5 rounded-lg transition-colors cursor-pointer" style={{ color: "var(--color-text-tertiary)" }}>
                <X className="w-5 h-5" />
              </button>
            </div>
            <PodcastGenerator
              documentId={selectedDocumentId}
              documentName={documents.find((d) => d.id === selectedDocumentId)?.filename || "Document"}
            />
          </div>
        </div>
      )}
    </div>
  );
}