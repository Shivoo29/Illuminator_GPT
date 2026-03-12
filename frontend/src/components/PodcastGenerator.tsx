import { useState } from "react";
import { api } from "../utils/api";
import {
  Mic,
  Download,
  Loader2,
  CheckCircle,
  AlertTriangle,
} from "lucide-react";

interface PodcastGeneratorProps {
  documentId: string;
  documentName: string;
}

const DURATIONS = [
  { value: 5, label: "5 min", desc: "Quick" },
  { value: 10, label: "10 min", desc: "Standard" },
  { value: 15, label: "15 min", desc: "In-depth" },
  { value: 30, label: "30 min", desc: "Complete" },
];

export default function PodcastGenerator({ documentId, documentName }: PodcastGeneratorProps) {
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

    await api.stream("/generate/podcast", {
      document_id: documentId,
      duration_minutes: duration,
    }, {
      onProgress: (data) => {
        setProgress(data.progress || 0);
        setProgressMessage(data.message || "Generating...");
      },
      onComplete: (data) => {
        setGenerating(false);
        setProgress(100);
        if (data.audio_url) setAudioUrl(data.audio_url);
        if (data.script) setScript(data.script);
      },
      onError: (err) => {
        setGenerating(false);
        setError(err.message);
      },
    });
  };

  const downloadAudio = () => {
    if (!audioUrl) return;
    const link = document.createElement("a");
    link.href = audioUrl;
    link.download = `${documentName}-podcast.mp3`;
    link.click();
  };

  return (
    <div className="space-y-4">
      {/* Document info */}
      <div className="p-3.5 rounded-xl flex items-center gap-3" style={{ background: "var(--color-accent-bg)" }}>
        <Mic className="w-5 h-5 shrink-0" style={{ color: "var(--color-accent-primary)" }} />
        <p className="text-sm" style={{ color: "var(--color-text-secondary)" }}>
          Generate a podcast about <strong style={{ color: "var(--color-text-primary)" }}>{documentName}</strong>
        </p>
      </div>

      {/* Duration chips */}
      {!generating && !audioUrl && (
        <div className="space-y-4">
          <div>
            <p className="text-xs font-medium mb-2" style={{ color: "var(--color-text-tertiary)" }}>Duration</p>
            <div className="flex gap-2">
              {DURATIONS.map((d) => (
                <button
                  key={d.value}
                  onClick={() => setDuration(d.value)}
                  className="flex-1 py-2.5 rounded-xl text-center transition-all duration-200 border cursor-pointer"
                  style={{
                    background: duration === d.value ? "var(--color-accent-bg)" : "var(--color-bg-tertiary)",
                    borderColor: duration === d.value ? "var(--color-accent-primary)" : "transparent",
                    color: duration === d.value ? "var(--color-accent-primary)" : "var(--color-text-secondary)",
                  }}
                >
                  <span className="text-sm font-semibold block">{d.label}</span>
                  <span className="text-[10px]" style={{ color: "var(--color-text-muted)" }}>{d.desc}</span>
                </button>
              ))}
            </div>
          </div>

          <button
            onClick={generatePodcast}
            className="w-full py-3 rounded-xl text-sm font-semibold text-white flex items-center justify-center gap-2 transition-all hover:opacity-90 cursor-pointer"
            style={{ background: "linear-gradient(135deg, #6366f1, #8b5cf6)" }}
          >
            <Mic className="w-4 h-4" /> Generate Podcast
          </button>
        </div>
      )}

      {/* Progress */}
      {generating && (
        <div className="space-y-3">
          <div className="h-2 rounded-full overflow-hidden" style={{ background: "var(--color-bg-tertiary)" }}>
            <div className="h-full rounded-full transition-all duration-300" style={{ width: `${progress}%`, background: "linear-gradient(90deg, #6366f1, #8b5cf6)" }} />
          </div>
          <p className="text-xs text-center" style={{ color: "var(--color-text-tertiary)" }}>
            {progressMessage} ({progress}%)
          </p>
          <div className="flex items-center justify-center gap-2" style={{ color: "var(--color-text-muted)" }}>
            <Loader2 className="w-4 h-4 animate-spin" />
            <span className="text-xs">Recording your podcast...</span>
          </div>
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="p-3 rounded-xl text-sm flex items-center gap-2" style={{ background: "var(--color-error-bg)", color: "var(--color-error)" }}>
          <AlertTriangle className="w-4 h-4 shrink-0" />
          {error}
        </div>
      )}

      {/* Result */}
      {audioUrl && (
        <div className="space-y-3">
          <div className="p-4 rounded-xl" style={{ background: "var(--color-success-bg)" }}>
            <div className="flex items-center gap-2 mb-3">
              <CheckCircle className="w-4 h-4" style={{ color: "var(--color-success)" }} />
              <p className="text-sm font-semibold" style={{ color: "var(--color-success)" }}>Podcast generated!</p>
            </div>
            <audio controls className="w-full" style={{ filter: "invert(0)" }}>
              <source src={audioUrl} type="audio/mpeg" />
            </audio>
          </div>
          <button
            onClick={downloadAudio}
            className="w-full py-2.5 rounded-xl text-sm font-medium flex items-center justify-center gap-2 transition-colors cursor-pointer"
            style={{ background: "var(--color-bg-tertiary)", color: "var(--color-text-secondary)" }}
          >
            <Download className="w-4 h-4" /> Download MP3
          </button>
        </div>
      )}

      {/* Script */}
      {script.length > 0 && (
        <div>
          <p className="text-xs font-medium mb-2" style={{ color: "var(--color-text-tertiary)" }}>Script Preview</p>
          <div className="max-h-48 overflow-y-auto p-3 rounded-xl text-sm space-y-2" style={{ background: "var(--color-bg-tertiary)" }}>
            {script.slice(0, 10).map((line, i) => (
              <p key={i}>
                <strong style={{ color: "var(--color-accent-primary)" }}>{line.speaker}:</strong>{" "}
                <span style={{ color: "var(--color-text-secondary)" }}>{line.text}</span>
              </p>
            ))}
            {script.length > 10 && (
              <p className="text-xs" style={{ color: "var(--color-text-muted)" }}>... and {script.length - 10} more lines</p>
            )}
          </div>
        </div>
      )}

      <p className="text-[11px] text-center" style={{ color: "var(--color-text-muted)" }}>
        Audio generated using local text-to-speech. No internet required.
      </p>
    </div>
  );
}