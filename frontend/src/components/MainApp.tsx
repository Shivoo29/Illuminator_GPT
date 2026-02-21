import { useState, useRef, useEffect } from "react";
import { Link } from "react-router-dom";
import { api } from "../utils/api.ts";
import DocumentUpload from "./DocumentUpload";
import PodcastGenerator from "./PodcastGenerator.tsx";

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

export default function MainApp() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [documents, setDocuments] = useState<Document[]>([]);
  const [selectedDocument, setSelectedDocument] = useState<string | null>(null);
  const [showUpload, setShowUpload] = useState(false);
  const [showPodcast, setShowPodcast] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const messagesEndRef = useRef<HTMLDivElement>(null);

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

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || loading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: "user",
      content: input,
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setLoading(true);

    try {
      const response = await api.post<{
        answer: string;
        sources: any[];
      }>("/query", {
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
    } catch (error) {
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: "Sorry, I encountered an error processing your request. Please try again.",
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  };

  const handleDocumentUploaded = () => {
    loadDocuments();
    setShowUpload(false);
  };

  const deleteDocument = async (docId: string) => {
    if (!confirm("Are you sure you want to delete this document?")) return;

    try {
      await api.delete(`/documents/${docId}`);
      loadDocuments();
      if (selectedDocument === docId) {
        setSelectedDocument(null);
      }
    } catch (error) {
      console.error("Failed to delete document:", error);
    }
  };

  return (
    <div className="h-screen flex bg-gray-100">
      {/* Sidebar */}
      <div
        className={`${
          sidebarOpen ? "w-72" : "w-0"
        } bg-white border-r border-gray-200 flex flex-col transition-all duration-300 overflow-hidden`}
      >
        <div className="p-4 border-b border-gray-200">
          <h1 className="text-xl font-bold text-gray-800">RAG Assistant</h1>
          <p className="text-sm text-gray-500">100% Offline</p>
        </div>

        {/* Documents List */}
        <div className="flex-1 overflow-y-auto p-4">
          <div className="flex items-center justify-between mb-3">
            <h2 className="text-sm font-semibold text-gray-600 uppercase tracking-wide">
              Documents
            </h2>
            <button
              onClick={() => setShowUpload(true)}
              className="text-primary-600 hover:text-primary-700 text-sm font-medium"
            >
              + Add
            </button>
          </div>

          <div className="space-y-2">
            <button
              onClick={() => setSelectedDocument(null)}
              className={`w-full text-left px-3 py-2 rounded-lg text-sm transition-colors ${
                selectedDocument === null
                  ? "bg-primary-100 text-primary-700"
                  : "hover:bg-gray-100 text-gray-700"
              }`}
            >
              All Documents
            </button>

            {documents.map((doc) => (
              <div
                key={doc.id}
                className={`group flex items-center px-3 py-2 rounded-lg text-sm transition-colors cursor-pointer ${
                  selectedDocument === doc.id
                    ? "bg-primary-100 text-primary-700"
                    : "hover:bg-gray-100 text-gray-700"
                }`}
                onClick={() => setSelectedDocument(doc.id)}
              >
                <span className="flex-1 truncate">{doc.filename}</span>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    deleteDocument(doc.id);
                  }}
                  className="opacity-0 group-hover:opacity-100 text-red-500 hover:text-red-600 ml-2"
                >
                  ×
                </button>
              </div>
            ))}

            {documents.length === 0 && (
              <p className="text-sm text-gray-400 text-center py-4">
                No documents yet
              </p>
            )}
          </div>
        </div>

        {/* Sidebar Actions */}
        <div className="p-4 border-t border-gray-200 space-y-2">
          <button
            onClick={() => setShowPodcast(true)}
            disabled={!selectedDocument}
            className="w-full py-2 px-3 bg-purple-100 text-purple-700 rounded-lg text-sm font-medium hover:bg-purple-200 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            🎙️ Generate Podcast
          </button>
          <Link
            to="/settings"
            className="block w-full py-2 px-3 bg-gray-100 text-gray-700 rounded-lg text-sm font-medium hover:bg-gray-200 text-center"
          >
            ⚙️ Settings
          </Link>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col">
        {/* Toggle Sidebar Button */}
        <button
          onClick={() => setSidebarOpen(!sidebarOpen)}
          className="absolute top-4 left-4 z-10 p-2 bg-white rounded-lg shadow-md hover:bg-gray-50"
          style={{ left: sidebarOpen ? "280px" : "16px" }}
        >
          {sidebarOpen ? "←" : "→"}
        </button>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto p-6">
          {messages.length === 0 ? (
            <div className="h-full flex items-center justify-center">
              <div className="text-center">
                <h2 className="text-2xl font-bold text-gray-800 mb-2">
                  Ask anything about your documents
                </h2>
                <p className="text-gray-600 mb-6">
                  Upload documents and ask questions. Everything runs locally on your device.
                </p>
                <div className="flex gap-3 justify-center">
                  <button
                    onClick={() => setShowUpload(true)}
                    className="px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700"
                  >
                    Upload Document
                  </button>
                </div>
              </div>
            </div>
          ) : (
            <div className="max-w-3xl mx-auto space-y-6">
              {messages.map((message) => (
                <div
                  key={message.id}
                  className={`flex ${
                    message.role === "user" ? "justify-end" : "justify-start"
                  }`}
                >
                  <div
                    className={`max-w-[80%] rounded-2xl px-4 py-3 ${
                      message.role === "user"
                        ? "bg-primary-600 text-white"
                        : "bg-white shadow-md"
                    }`}
                  >
                    <p className="whitespace-pre-wrap">{message.content}</p>

                    {/* Sources */}
                    {message.sources && message.sources.length > 0 && (
                      <div className="mt-3 pt-3 border-t border-gray-200">
                        <p className="text-xs text-gray-500 mb-2">Sources:</p>
                        <div className="space-y-2">
                          {message.sources.slice(0, 3).map((source, i) => (
                            <div
                              key={i}
                              className="text-xs bg-gray-50 p-2 rounded"
                            >
                              <p className="text-gray-600 line-clamp-2">
                                {source.content}
                              </p>
                              <p className="text-gray-400 mt-1">
                                {source.metadata?.filename}
                              </p>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              ))}

              {loading && (
                <div className="flex justify-start">
                  <div className="bg-white shadow-md rounded-2xl px-4 py-3">
                    <div className="flex space-x-2">
                      <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
                      <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce delay-100"></div>
                      <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce delay-200"></div>
                    </div>
                  </div>
                </div>
              )}

              <div ref={messagesEndRef} />
            </div>
          )}
        </div>

        {/* Input */}
        <div className="p-4 bg-white border-t border-gray-200">
          <form onSubmit={handleSubmit} className="max-w-3xl mx-auto">
            <div className="flex gap-3">
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder={
                  selectedDocument
                    ? "Ask about this document..."
                    : "Ask about your documents..."
                }
                disabled={loading}
                className="flex-1 px-4 py-3 border border-gray-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent disabled:bg-gray-100"
              />
              <button
                type="submit"
                disabled={loading || !input.trim()}
                className="px-6 py-3 bg-primary-600 text-white rounded-xl hover:bg-primary-700 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                Send
              </button>
            </div>
            {selectedDocument && (
              <p className="text-xs text-gray-500 mt-2 text-center">
                Searching in: {documents.find((d) => d.id === selectedDocument)?.filename}
              </p>
            )}
          </form>
        </div>
      </div>

      {/* Document Upload Modal */}
      {showUpload && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-2xl p-6 max-w-lg w-full mx-4">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-xl font-bold">Upload Document</h2>
              <button
                onClick={() => setShowUpload(false)}
                className="text-gray-500 hover:text-gray-700"
              >
                ×
              </button>
            </div>
            <DocumentUpload onComplete={handleDocumentUploaded} />
          </div>
        </div>
      )}

      {/* Podcast Generator Modal */}
      {showPodcast && selectedDocument && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-2xl p-6 max-w-lg w-full mx-4">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-xl font-bold">Generate Podcast</h2>
              <button
                onClick={() => setShowPodcast(false)}
                className="text-gray-500 hover:text-gray-700"
              >
                ×
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