import { useState, useEffect } from "react"
import { api } from "../utils/api"
import {
  FileText,
  Loader2,
  Trash2,
  FolderOpen,
} from "lucide-react"

interface Document {
  id: string
  filename: string
  file_type: string
  chunk_count: number
}

export default function DocumentLibraryPage() {
  const [documents, setDocuments] = useState<Document[]>([])
  const [loading, setLoading] = useState(true)
  const [deletingId, setDeletingId] = useState<string | null>(null)

  useEffect(() => {
    loadDocuments()
  }, [])

  const loadDocuments = async () => {
    try {
      setLoading(true)
      const docs = await api.get<Document[]>("/documents")
      setDocuments(docs)
    } catch (error) {
      console.error("Failed to load documents:", error)
    } finally {
      setLoading(false)
    }
  }

  const deleteDocument = async (id: string, e: React.MouseEvent) => {
    e.stopPropagation()
    if (!window.confirm("Are you sure you want to delete this document?"))
      return

    try {
      setDeletingId(id)
      await api.delete(`/documents/${id}`)
      setDocuments((docs) => docs.filter((d) => d.id !== id))
    } catch (error) {
      console.error("Failed to delete document:", error)
    } finally {
      setDeletingId(null)
    }
  }

  return (
    <div className="flex-1 flex flex-col overflow-hidden">
      {/* Header */}
      <div
        className="h-14 flex items-center px-6 border-b shrink-0"
        style={{ borderColor: "var(--color-border-primary)" }}
      >
        <FolderOpen
          className="w-5 h-5 mr-2"
          style={{ color: "var(--color-accent-primary)" }}
        />
        <h2
          className="text-sm font-semibold"
          style={{ color: "var(--color-text-primary)" }}
        >
          Document Library
        </h2>
      </div>

      <div className="flex-1 overflow-y-auto p-6">
        <div className="max-w-4xl mx-auto space-y-6">
          <div
            className="rounded-xl border p-5"
            style={{
              borderColor: "var(--color-border-primary)",
              background: "var(--color-bg-surface)",
            }}
          >
            <div className="flex items-center justify-between mb-4">
              <h3
                className="text-sm font-semibold"
                style={{ color: "var(--color-text-primary)" }}
              >
                Manage Documents
              </h3>
            </div>

            {loading ? (
              <div className="flex items-center justify-center p-8">
                <Loader2
                  className="w-6 h-6 animate-spin"
                  style={{ color: "var(--color-text-muted)" }}
                />
              </div>
            ) : documents.length > 0 ? (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {documents.map((doc) => (
                  <div
                    key={doc.id}
                    className="flex flex-col p-4 rounded-xl border"
                    style={{
                      background: "var(--color-bg-tertiary)",
                      borderColor: "var(--color-border-primary)",
                    }}
                  >
                    <div className="flex items-start justify-between mb-2">
                      <div className="flex items-center gap-3 overflow-hidden">
                        <div
                          className="p-2 rounded-lg"
                          style={{ background: "var(--color-accent-bg)" }}
                        >
                          <FileText
                            className="w-5 h-5 shrink-0"
                            style={{ color: "var(--color-accent-primary)" }}
                          />
                        </div>
                        <span
                          className="text-sm font-medium truncate"
                          style={{ color: "var(--color-text-primary)" }}
                        >
                          {doc.filename}
                        </span>
                      </div>
                      <button
                        onClick={(e) => deleteDocument(doc.id, e)}
                        disabled={deletingId === doc.id}
                        className="p-1.5 rounded-lg transition-colors hover:bg-neutral-500/20"
                      >
                        {deletingId === doc.id ? (
                          <Loader2 className="w-4 h-4 animate-spin text-red-500" />
                        ) : (
                          <Trash2 className="w-4 h-4 text-red-500 hover:text-red-400" />
                        )}
                      </button>
                    </div>

                    <div
                      className="flex items-center gap-4 mt-2 text-xs"
                      style={{ color: "var(--color-text-muted)" }}
                    >
                      <span className="bg-neutral-500/10 px-2 py-1 rounded">
                        {doc.file_type.toUpperCase()}
                      </span>
                      <span>Chunks: {doc.chunk_count}</span>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-center py-8">
                <FolderOpen
                  className="w-8 h-8 mx-auto mb-2"
                  style={{ color: "var(--color-text-muted)" }}
                />
                <p
                  className="text-sm mb-4"
                  style={{ color: "var(--color-text-muted)" }}
                >
                  No documents in your library.
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
