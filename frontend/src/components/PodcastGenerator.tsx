import { useState } from "react";
import { api } from "../utils/api.ts";

interface PodcastGeneratorProps {
  documentId: string;
  documentName: string;
}

export default function PodcastGenerator({
  documentId,
  documentName,
}: PodcastGeneratorProps) {
  const [duration, setDuration] = useState(10);
  const [generating, setGenerating] = useState(false);
  const [progress, setProgress] = useState(0);
  const [progressMessage, setProgressMessage] = useState("");
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [script, setScript] = useState<Array<{ speaker: string; text: string }>>([]);
  const [error, setError] = useState<string | null>(null);

  const generatePodcast = async () => {
    setGenerating(true);
    setProgress(0);
    setError(null);
    setAudioUrl(null);
    setScript([]);

    await api.stream(
      "/generate/podcast",
      {
        document_id: documentId,
        duration_minutes: duration,
      },
      {
        onProgress: (data) => {
          setProgress(data.progress || 0);
          setProgressMessage(data.message || "Generating...");
        },
        onComplete: (data) => {
          setGenerating(false);
          setProgress(100);
          if (data.audio_url) {
            setAudioUrl(data.audio_url);
          }
          if (data.script) {
            setScript(data.script);
          }
        },
        onError: (err) => {
          setGenerating(false);
          setError(err.message);
        },
      }
    );
  };

  const downloadAudio = () => {
    if (audioUrl) {
      const link = document.createElement("a");
      link.href = audioUrl;
      link.download = `${documentName}-podcast.mp3`;
      link.click();
    }
  };

  return (
    <div className="space-y-4">
      <div className="p-4 bg-purple-50 rounded-lg">
        <p className="text-purple-800">
          Generate an AI podcast discussion about:{" "}
          <strong>{documentName}</strong>
        </p>
      </div>

      {/* Duration Selection */}
      {!generating && !audioUrl && (
        <div className="space-y-3">
          <label className="block">
            <span className="text-sm font-medium text-gray-700">
              Podcast Duration
            </span>
            <select
              value={duration}
              onChange={(e) => setDuration(parseInt(e.target.value))}
              className="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-primary-500 focus:border-primary-500"
            >
              <option value={5}>5 minutes (Quick overview)</option>
              <option value={10}>10 minutes (Standard)</option>
              <option value={15}>15 minutes (In-depth)</option>
              <option value={30}>30 minutes (Complete analysis)</option>
            </select>
          </label>

          <button
            onClick={generatePodcast}
            className="w-full py-3 bg-purple-600 text-white rounded-lg font-medium hover:bg-purple-700"
          >
            🎙️ Generate Podcast
          </button>
        </div>
      )}

      {/* Progress */}
      {generating && (
        <div className="space-y-3">
          <div className="h-3 bg-gray-200 rounded-full overflow-hidden">
            <div
              className="h-full bg-purple-600 transition-all duration-300"
              style={{ width: `${progress}%` }}
            />
          </div>
          <p className="text-sm text-gray-600 text-center">
            {progressMessage} ({progress}%)
          </p>
          <div className="text-center text-gray-500">
            <div className="inline-flex items-center space-x-1">
              <span className="animate-pulse">🎙️</span>
              <span>Recording your podcast...</span>
            </div>
          </div>
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="p-3 bg-red-50 border border-red-200 rounded-lg text-red-700">
          {error}
        </div>
      )}

      {/* Result */}
      {audioUrl && (
        <div className="space-y-4">
          <div className="p-4 bg-green-50 border border-green-200 rounded-lg">
            <p className="text-green-700 font-medium mb-2">
              ✓ Podcast generated successfully!
            </p>
            <audio controls className="w-full">
              <source src={audioUrl} type="audio/mpeg" />
              Your browser does not support the audio element.
            </audio>
          </div>

          <button
            onClick={downloadAudio}
            className="w-full py-2 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200"
          >
            📥 Download MP3
          </button>
        </div>
      )}

      {/* Script Preview */}
      {script.length > 0 && (
        <div className="space-y-2">
          <h3 className="text-sm font-medium text-gray-700">Script Preview</h3>
          <div className="max-h-60 overflow-y-auto p-3 bg-gray-50 rounded-lg text-sm">
            {script.slice(0, 10).map((line, i) => (
              <p key={i} className="mb-2">
                <strong className="text-purple-700">{line.speaker}:</strong>{" "}
                {line.text}
              </p>
            ))}
            {script.length > 10 && (
              <p className="text-gray-400">... and {script.length - 10} more lines</p>
            )}
          </div>
        </div>
      )}

      {/* Info */}
      <p className="text-xs text-gray-500 text-center">
        Audio is generated using local text-to-speech. No internet required.
      </p>
    </div>
  );
}