import { useState, useRef, useEffect } from "react";
import { api } from "../utils/api";
import DocumentUpload from "./DocumentUpload";
import PodcastGenerator from "./PodcastGenerator";
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
} from "lucide-react";

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

const SUGGESTED_PROMPTS = [
  { icon: BookOpen, text: "Summarize this document" },
  { icon: Search, text: "What are the key points?" },
  { icon: Lightbulb, text: "Explain the main concepts" },
  { icon: MessageSquare, text: "Compare the documents" },
];

export default function MainApp() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [documents, setDocuments] = useState<Document[]>([]);
  const [selectedDocument, setSelectedDocument] = useState<string | null>(null);
  const [showUpload, setShowUpload] = useState(false);
  const [showPodcast, setShowPodcast] = useState(false);
  const [docPanelOpen, setDocPanelOpen] = useState(false);
  const [expandedSources, setExpandedSources] = useState<Set<string>>(new Set());
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    loadDocuments();
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const loadDocuments = async () => {
    try {
      const docs = await api.get<Document[]>("/documents");
      setDocuments(docs);
    } catch (error) {
      console.error("Failed to load documents:", error);
    }
  };

  const handleSubmit = async (e?: React.FormEvent) => {
    e?.preventDefault();
    if (!input.trim() || loading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: "user",
      content: input,
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setLoading(true);

    // Reset textarea height
    if (textareaRef.current) textareaRef.current.style.height = "auto";

    try {
      const response = await api.post<{ answer: string; sources: any[] }>("/query", {
        question: input,
        filter_document_id: selectedDocument,
        n_results: 5,
      });

      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: response.answer,
        sources: response.sources,
      };

      setMessages((prev) => [...prev, assistantMessage]);
    } catch {
      setMessages((prev) => [
        ...prev,
        {
          id: (Date.now() + 1).toString(),
          role: "assistant",
          content: "Sorry, I encountered an error. Please try again.",
        },
      ]);
    } finally {
      setLoading(false);
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

  const deleteDocument = async (docId: string) => {
    try {
      await api.delete(`/documents/${docId}`);
      loadDocuments();
      if (selectedDocument === docId) setSelectedDocument(null);
    } catch (error) {
      console.error("Failed to delete document:", error);
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
      {/* Main chat area */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Header */}
        <div
          className="h-14 flex items-center justify-between px-5 border-b shrink-0"
          style={{ borderColor: "var(--color-border-primary)", background: "var(--color-bg-primary)" }}
        >
          <div className="flex items-center gap-2">
            <h2 className="text-sm font-semibold" style={{ color: "var(--color-text-primary)" }}>Chat</h2>
            {selectedDocument && (
              <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium" style={{ background: "var(--color-accent-bg)", color: "var(--color-accent-primary)" }}>
                <FileText className="w-3 h-3" />
                <span className="max-w-[120px] truncate">{documents.find((d) => d.id === selectedDocument)?.filename}</span>
                <button onClick={() => setSelectedDocument(null)} className="hover:opacity-70 cursor-pointer">
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
                Upload documents and ask questions. Everything runs locally on your device.
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
                    <p className="text-sm whitespace-pre-wrap leading-relaxed" style={message.role === "assistant" ? { color: "var(--color-text-primary)" } : {}}>
                      {message.content}
                    </p>

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
              onClick={() => setShowUpload(true)}
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
              placeholder={selectedDocument ? "Ask about this document..." : "Ask about your documents..."}
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
            <h3 className="text-sm font-semibold" style={{ color: "var(--color-text-primary)" }}>Documents</h3>
            <button
              onClick={() => setShowUpload(true)}
              className="px-2.5 py-1 rounded-lg text-xs font-semibold transition-colors cursor-pointer"
              style={{ background: "var(--color-accent-bg)", color: "var(--color-accent-primary)" }}
            >
              + Add
            </button>
          </div>

          <div className="flex-1 overflow-y-auto p-3 space-y-1">
            <button
              onClick={() => setSelectedDocument(null)}
              className="w-full text-left px-3 py-2 rounded-lg text-sm transition-colors cursor-pointer"
              style={{
                background: selectedDocument === null ? "var(--color-accent-bg)" : "transparent",
                color: selectedDocument === null ? "var(--color-accent-primary)" : "var(--color-text-secondary)",
              }}
            >
              All Documents
            </button>

            {documents.map((doc) => (
              <div
                key={doc.id}
                className="group flex items-center px-3 py-2 rounded-lg text-sm transition-colors cursor-pointer"
                style={{
                  background: selectedDocument === doc.id ? "var(--color-accent-bg)" : "transparent",
                  color: selectedDocument === doc.id ? "var(--color-accent-primary)" : "var(--color-text-secondary)",
                }}
                onClick={() => setSelectedDocument(doc.id)}
                onMouseEnter={(e) => {
                  if (selectedDocument !== doc.id) e.currentTarget.style.background = "var(--color-bg-hover)";
                }}
                onMouseLeave={(e) => {
                  if (selectedDocument !== doc.id) e.currentTarget.style.background = "transparent";
                }}
              >
                <FileText className="w-4 h-4 mr-2 shrink-0" />
                <span className="flex-1 truncate">{doc.filename}</span>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    deleteDocument(doc.id);
                  }}
                  className="opacity-0 group-hover:opacity-100 transition-opacity cursor-pointer"
                  style={{ color: "var(--color-error)" }}
                >
                  <Trash2 className="w-3.5 h-3.5" />
                </button>
              </div>
            ))}

            {documents.length === 0 && (
              <div className="text-center py-8">
                <FileText className="w-8 h-8 mx-auto mb-2" style={{ color: "var(--color-text-muted)" }} />
                <p className="text-xs" style={{ color: "var(--color-text-muted)" }}>No documents yet</p>
              </div>
            )}
          </div>

          {/* Podcast button */}
          <div className="p-3 border-t" style={{ borderColor: "var(--color-border-primary)" }}>
            <button
              onClick={() => setShowPodcast(true)}
              disabled={!selectedDocument}
              className="w-full py-2.5 rounded-xl text-xs font-semibold flex items-center justify-center gap-2 transition-all disabled:opacity-40 cursor-pointer"
              style={{ background: "var(--color-bg-tertiary)", color: "var(--color-text-secondary)" }}
            >
              <Mic className="w-4 h-4" /> Generate Podcast
            </button>
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
            <DocumentUpload onComplete={() => { loadDocuments(); setShowUpload(false); }} />
          </div>
        </div>
      )}

      {/* Podcast Modal */}
      {showPodcast && selectedDocument && (
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
              documentId={selectedDocument}
              documentName={documents.find((d) => d.id === selectedDocument)?.filename || "Document"}
            />
          </div>
        </div>
      )}
    </div>
  );
}