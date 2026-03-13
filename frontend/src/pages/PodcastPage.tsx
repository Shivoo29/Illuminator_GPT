import { useState, useEffect } from "react";
import { api } from "../utils/api";
import {
  Mic,
  FileText,
  Loader2,
  X,
  Plus
} from "lucide-react";
import PodcastGenerator from "../components/PodcastGenerator";
import DocumentUpload from "../components/DocumentUpload";

interface Document {
  id: string;
  filename: string;
  file_type: string;
  vector_count: number;
  created_at: string;
}

export default function PodcastPage() {
  const [documents, setDocuments] = useState<Document[]>([]);
  const [selectedDocument, setSelectedDocument] = useState<Document | null>(null);
  const [loading, setLoading] = useState(true);
  const [showUpload, setShowUpload] = useState(false);

  useEffect(() => {
    loadDocuments();
  }, []);

  const loadDocuments = async () => {
    try {
      setLoading(true);
      const docs = await api.get<Document[]>("/documents");
      setDocuments(docs);
    } catch (error) {
      console.error("Failed to load documents:", error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex-1 flex flex-col overflow-hidden">
      {/* Header */}
      <div className="h-14 flex items-center px-6 border-b shrink-0" style={{ borderColor: "var(--color-border-primary)" }}>
        <Mic className="w-5 h-5 mr-2" style={{ color: "var(--color-accent-primary)" }} />
        <h2 className="text-sm font-semibold" style={{ color: "var(--color-text-primary)" }}>Podcast Generation</h2>
      </div>

      <div className="flex-1 overflow-y-auto p-6">
        <div className="max-w-3xl mx-auto space-y-6">
          <div className="rounded-xl border p-5" style={{ borderColor: "var(--color-border-primary)", background: "var(--color-bg-surface)" }}>
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-sm font-semibold" style={{ color: "var(--color-text-primary)" }}>Select a Document</h3>
              <button
                onClick={() => setShowUpload(true)}
                className="text-xs transition-colors cursor-pointer hover:opacity-80 flex items-center gap-1"
                style={{ color: "var(--color-accent-primary)" }}
              >
                <Plus className="w-3.5 h-3.5" /> Upload Document
              </button>
            </div>
            
            {loading ? (
              <div className="flex items-center justify-center p-8">
                <Loader2 className="w-6 h-6 animate-spin" style={{ color: "var(--color-text-muted)" }} />
              </div>
            ) : documents.length > 0 ? (
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 max-h-[300px] overflow-y-auto pr-2 custom-scrollbar">
                {documents.map((doc) => (
                  <button
                    key={doc.id}
                    onClick={() => setSelectedDocument(doc)}
                    className="flex flex-col text-left p-3.5 rounded-xl border transition-all cursor-pointer"
                    style={{
                      background: selectedDocument?.id === doc.id ? "var(--color-accent-bg)" : "var(--color-bg-tertiary)",
                      borderColor: selectedDocument?.id === doc.id ? "var(--color-accent-primary)" : "transparent",
                    }}
                  >
                    <div className="flex items-center gap-2 w-full mb-1">
                      <FileText className="w-4 h-4 shrink-0" style={{ color: selectedDocument?.id === doc.id ? "var(--color-accent-primary)" : "var(--color-text-secondary)" }} />
                      <span className="text-sm font-medium truncate w-full" style={{ color: selectedDocument?.id === doc.id ? "var(--color-accent-primary)" : "var(--color-text-primary)" }}>
                        {doc.filename}
                      </span>
                    </div>
                    <span className="text-xs" style={{ color: "var(--color-text-muted)" }}>
                      {new Date(doc.created_at).toLocaleDateString()}
                    </span>
                  </button>
                ))}
              </div>
            ) : (
              <div className="text-center py-8">
                <FileText className="w-8 h-8 mx-auto mb-2" style={{ color: "var(--color-text-muted)" }} />
                <p className="text-xs mb-4" style={{ color: "var(--color-text-muted)" }}>No documents found</p>
                <button
                  onClick={() => setShowUpload(true)}
                  className="px-4 py-2 rounded-lg text-xs font-semibold"
                  style={{ background: "var(--color-accent-bg)", color: "var(--color-accent-primary)" }}
                >
                  Upload First Document
                </button>
              </div>
            )}
          </div>

          {selectedDocument && (
            <div className="animate-fade-in-up">
              <PodcastGenerator documentId={selectedDocument.id} documentName={selectedDocument.filename} />
            </div>
          )}

        </div>
      </div>

      {/* Upload Modal */}
      {showUpload && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/50" onClick={() => setShowUpload(false)}>
          <div
            className="w-full max-w-md rounded-2xl p-6 shadow-2xl animate-scale-in"
            style={{ background: "var(--color-bg-elevated)", borderColor: "var(--color-border-primary)", borderWidth: 1 }}
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
    </div>
  );
}
