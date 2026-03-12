import { useState, useCallback } from "react";
import { api } from "../utils/api";
import {
  Upload,
  FileText,
  Image,
  Music,
  Video,
  X,
  CheckCircle,
  Loader2,
} from "lucide-react";

interface DocumentUploadProps {
  onComplete: () => void;
}

const FILE_ICONS: Record<string, typeof FileText> = {
  pdf: FileText,
  image: Image,
  audio: Music,
  video: Video,
};

function getFileIcon(file: File) {
  if (file.type.includes("pdf")) return FILE_ICONS.pdf;
  if (file.type.includes("image")) return FILE_ICONS.image;
  if (file.type.includes("audio")) return FILE_ICONS.audio;
  if (file.type.includes("video")) return FILE_ICONS.video;
  return FileText;
}

export default function DocumentUpload({ onComplete }: DocumentUploadProps) {
  const [file, setFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [dragActive, setDragActive] = useState(false);
  const [complete, setComplete] = useState(false);

  const supportedTypes = [".pdf", ".docx", ".pptx", ".txt", ".md", ".mp3", ".wav", ".mp4", ".jpg", ".png"];

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(e.type === "dragenter" || e.type === "dragover");
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files?.[0]) {
      setFile(e.dataTransfer.files[0]);
      setError(null);
    }
  }, []);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.[0]) {
      setFile(e.target.files[0]);
      setError(null);
    }
  };

  const handleUpload = async () => {
    if (!file) return;
    setUploading(true);
    setProgress(0);
    setError(null);

    try {
      const interval = setInterval(() => setProgress((p) => Math.min(p + 10, 90)), 200);
      await api.upload("/documents/upload", file);
      clearInterval(interval);
      setProgress(100);
      setComplete(true);
      setTimeout(onComplete, 800);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed");
      setUploading(false);
    }
  };

  const formatSize = (bytes: number) => {
    if (bytes < 1024) return bytes + " B";
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
    return (bytes / (1024 * 1024)).toFixed(1) + " MB";
  };

  const Icon = file ? getFileIcon(file) : Upload;

  return (
    <div className="space-y-4">
      {/* Drop zone */}
      <div
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
        className={`border-2 border-dashed rounded-xl p-8 text-center transition-all duration-200 ${dragActive ? "scale-[1.02]" : ""}`}
        style={{
          borderColor: dragActive ? "var(--color-accent-primary)" : "var(--color-border-secondary)",
          background: dragActive ? "var(--color-accent-bg)" : "var(--color-bg-tertiary)",
        }}
      >
        <input
          type="file"
          id="file-upload"
          className="hidden"
          accept={supportedTypes.join(",")}
          onChange={handleFileChange}
        />

        {!file ? (
          <label htmlFor="file-upload" className="cursor-pointer block">
            <Upload className="w-10 h-10 mx-auto mb-3" style={{ color: "var(--color-text-muted)" }} />
            <p className="text-sm font-medium mb-1" style={{ color: "var(--color-text-secondary)" }}>
              Drag and drop a file, or click to browse
            </p>
            <p className="text-xs" style={{ color: "var(--color-text-muted)" }}>
              PDF, DOCX, PPTX, TXT, MD, MP3, WAV, MP4, JPG, PNG
            </p>
          </label>
        ) : (
          <div>
            <Icon className="w-10 h-10 mx-auto mb-3" style={{ color: "var(--color-accent-primary)" }} />
            <p className="text-sm font-semibold" style={{ color: "var(--color-text-primary)" }}>{file.name}</p>
            <p className="text-xs mt-0.5" style={{ color: "var(--color-text-tertiary)" }}>{formatSize(file.size)}</p>
            {!uploading && (
              <button
                onClick={() => setFile(null)}
                className="mt-2 text-xs font-medium cursor-pointer flex items-center gap-1 mx-auto"
                style={{ color: "var(--color-error)" }}
              >
                <X className="w-3 h-3" /> Remove
              </button>
            )}
          </div>
        )}
      </div>

      {/* Error */}
      {error && (
        <div className="p-3 rounded-xl text-sm flex items-center gap-2" style={{ background: "var(--color-error-bg)", color: "var(--color-error)" }}>
          {error}
        </div>
      )}

      {/* Progress */}
      {uploading && (
        <div className="space-y-2">
          <div className="h-1.5 rounded-full overflow-hidden" style={{ background: "var(--color-bg-tertiary)" }}>
            <div
              className="h-full rounded-full transition-all duration-300"
              style={{ width: `${progress}%`, background: complete ? "var(--color-success)" : "linear-gradient(90deg, #6366f1, #8b5cf6)" }}
            />
          </div>
          <p className="text-xs text-center flex items-center justify-center gap-1.5" style={{ color: "var(--color-text-tertiary)" }}>
            {complete ? (
              <><CheckCircle className="w-3.5 h-3.5" style={{ color: "var(--color-success)" }} /> Complete!</>
            ) : (
              <><Loader2 className="w-3.5 h-3.5 animate-spin" /> Processing document...</>
            )}
          </p>
        </div>
      )}

      {/* Upload button */}
      <button
        onClick={handleUpload}
        disabled={!file || uploading}
        className="w-full py-3 rounded-xl text-sm font-semibold text-white flex items-center justify-center gap-2 transition-all hover:opacity-90 disabled:opacity-40 cursor-pointer"
        style={{ background: "linear-gradient(135deg, #6366f1, #8b5cf6)" }}
      >
        {uploading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Upload className="w-4 h-4" />}
        {uploading ? "Processing..." : "Upload & Process"}
      </button>

      <p className="text-[11px] text-center" style={{ color: "var(--color-text-muted)" }}>
        Documents are processed locally and stored on your device.
      </p>
    </div>
  );
}