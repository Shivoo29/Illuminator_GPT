import { useState, useCallback } from "react";
import { api } from "../utils/api.ts";

interface DocumentUploadProps {
  onComplete: () => void;
}

export default function DocumentUpload({ onComplete }: DocumentUploadProps) {
  const [file, setFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [dragActive, setDragActive] = useState(false);

  const supportedTypes = [
    ".pdf",
    ".docx",
    ".pptx",
    ".txt",
    ".md",
    ".mp3",
    ".wav",
    ".mp4",
    ".jpg",
    ".png",
  ];

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      setFile(e.dataTransfer.files[0]);
      setError(null);
    }
  }, []);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
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
      // Simulate progress while uploading
      const progressInterval = setInterval(() => {
        setProgress((prev) => Math.min(prev + 10, 90));
      }, 200);

      await api.upload("/documents/upload", file);

      clearInterval(progressInterval);
      setProgress(100);

      setTimeout(() => {
        onComplete();
      }, 500);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed");
      setUploading(false);
    }
  };

  const formatFileSize = (bytes: number): string => {
    if (bytes < 1024) return bytes + " B";
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
    return (bytes / (1024 * 1024)).toFixed(1) + " MB";
  };

  return (
    <div className="space-y-4">
      {/* Drop Zone */}
      <div
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
        className={`border-2 border-dashed rounded-xl p-8 text-center transition-colors ${
          dragActive
            ? "border-primary-500 bg-primary-50"
            : "border-gray-300 hover:border-gray-400"
        }`}
      >
        <input
          type="file"
          id="file-upload"
          className="hidden"
          accept={supportedTypes.join(",")}
          onChange={handleFileChange}
        />

        {!file ? (
          <label htmlFor="file-upload" className="cursor-pointer">
            <div className="text-4xl mb-2">📄</div>
            <p className="text-gray-600 mb-2">
              Drag and drop a file here, or click to browse
            </p>
            <p className="text-sm text-gray-400">
              Supports: PDF, DOCX, PPTX, TXT, MD, MP3, WAV, MP4, JPG, PNG
            </p>
          </label>
        ) : (
          <div>
            <div className="text-4xl mb-2">
              {file.type.includes("pdf")
                ? "📕"
                : file.type.includes("image")
                ? "🖼️"
                : file.type.includes("audio")
                ? "🎵"
                : file.type.includes("video")
                ? "🎬"
                : "📄"}
            </div>
            <p className="font-medium text-gray-800">{file.name}</p>
            <p className="text-sm text-gray-500">{formatFileSize(file.size)}</p>
            <button
              onClick={() => setFile(null)}
              className="mt-2 text-sm text-red-500 hover:text-red-600"
            >
              Remove
            </button>
          </div>
        )}
      </div>

      {/* Error Message */}
      {error && (
        <div className="p-3 bg-red-50 border border-red-200 rounded-lg text-red-700 text-sm">
          {error}
        </div>
      )}

      {/* Progress Bar */}
      {uploading && (
        <div className="space-y-2">
          <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
            <div
              className="h-full bg-primary-600 transition-all duration-300"
              style={{ width: `${progress}%` }}
            />
          </div>
          <p className="text-sm text-gray-600 text-center">
            {progress < 100 ? "Processing document..." : "Complete!"}
          </p>
        </div>
      )}

      {/* Upload Button */}
      <button
        onClick={handleUpload}
        disabled={!file || uploading}
        className="w-full py-3 bg-primary-600 text-white rounded-lg font-medium hover:bg-primary-700 disabled:opacity-50 disabled:cursor-not-allowed"
      >
        {uploading ? "Processing..." : "Upload & Process"}
      </button>

      {/* Info */}
      <p className="text-xs text-gray-500 text-center">
        Documents are processed locally and stored on your device.
      </p>
    </div>
  );
}