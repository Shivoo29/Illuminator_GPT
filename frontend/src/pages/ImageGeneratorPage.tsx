import { useState } from "react";
import { api } from "../utils/api";
import {
  ImageIcon,
  Download,
  Loader2,
  AlertTriangle,
  Sparkles,
  SlidersHorizontal,
  ChevronDown,
  ChevronUp,
} from "lucide-react";

const SIZE_PRESETS = [
  { w: 512, h: 512, label: "512×512" },
  { w: 768, h: 768, label: "768×768" },
  { w: 512, h: 768, label: "512×768" },
  { w: 768, h: 512, label: "768×512" },
];

export default function ImageGeneratorPage() {
  const [prompt, setPrompt] = useState("");
  const [negativePrompt, setNegativePrompt] = useState("");
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [selectedSize, setSelectedSize] = useState(0);
  const [steps, setSteps] = useState(20);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [generatedImages, setGeneratedImages] = useState<Array<{ url: string; prompt: string }>>([]);

  const handleGenerate = async () => {
    if (!prompt.trim()) return;
    setLoading(true);
    setError(null);

    try {
      const size = SIZE_PRESETS[selectedSize];
      const result = await api.post<{ success: boolean; image_url: string }>("/generate/image", {
        prompt,
        negative_prompt: negativePrompt,
        width: size.w,
        height: size.h,
        num_steps: steps,
      });

      if (result.success && result.image_url) {
        setGeneratedImages((prev) => [{ url: result.image_url, prompt }, ...prev]);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Generation failed");
    } finally {
      setLoading(false);
    }
  };

  const downloadImage = (url: string, index: number) => {
    const link = document.createElement("a");
    link.href = url;
    link.download = `illuminator-image-${index}.png`;
    link.click();
  };

  return (
    <div className="flex-1 flex flex-col overflow-hidden">
      {/* Header */}
      <div className="h-14 flex items-center px-6 border-b shrink-0" style={{ borderColor: "var(--color-border-primary)" }}>
        <ImageIcon className="w-5 h-5 mr-2" style={{ color: "var(--color-accent-primary)" }} />
        <h2 className="text-sm font-semibold" style={{ color: "var(--color-text-primary)" }}>Image Generation</h2>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto p-6">
        <div className="max-w-3xl mx-auto space-y-5">
          {/* Prompt */}
          <div>
            <label className="text-xs font-medium mb-1.5 block" style={{ color: "var(--color-text-tertiary)" }}>Prompt</label>
            <textarea
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder="Describe the image you want to create..."
              rows={3}
              className="w-full p-4 rounded-xl text-sm outline-none border resize-none transition-all"
              style={{
                background: "var(--color-bg-input)",
                borderColor: "var(--color-border-secondary)",
                color: "var(--color-text-primary)",
              }}
              onFocus={(e) => (e.currentTarget.style.borderColor = "var(--color-border-focus)")}
              onBlur={(e) => (e.currentTarget.style.borderColor = "var(--color-border-secondary)")}
            />
          </div>

          {/* Advanced toggle */}
          <button
            onClick={() => setShowAdvanced(!showAdvanced)}
            className="flex items-center gap-2 text-xs font-medium cursor-pointer"
            style={{ color: "var(--color-text-tertiary)" }}
          >
            <SlidersHorizontal className="w-3.5 h-3.5" />
            Advanced options
            {showAdvanced ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
          </button>

          {showAdvanced && (
            <div className="space-y-4 animate-fade-in-down">
              {/* Negative prompt */}
              <div>
                <label className="text-xs font-medium mb-1.5 block" style={{ color: "var(--color-text-tertiary)" }}>Negative Prompt</label>
                <input
                  type="text"
                  value={negativePrompt}
                  onChange={(e) => setNegativePrompt(e.target.value)}
                  placeholder="What to avoid..."
                  className="w-full px-4 py-2.5 rounded-xl text-sm outline-none border"
                  style={{
                    background: "var(--color-bg-input)",
                    borderColor: "var(--color-border-primary)",
                    color: "var(--color-text-primary)",
                  }}
                />
              </div>

              {/* Size presets */}
              <div>
                <label className="text-xs font-medium mb-1.5 block" style={{ color: "var(--color-text-tertiary)" }}>Size</label>
                <div className="flex gap-2">
                  {SIZE_PRESETS.map((size, i) => (
                    <button
                      key={size.label}
                      onClick={() => setSelectedSize(i)}
                      className="flex-1 py-2.5 rounded-xl text-xs font-medium border transition-all cursor-pointer"
                      style={{
                        background: selectedSize === i ? "var(--color-accent-bg)" : "var(--color-bg-tertiary)",
                        borderColor: selectedSize === i ? "var(--color-accent-primary)" : "transparent",
                        color: selectedSize === i ? "var(--color-accent-primary)" : "var(--color-text-secondary)",
                      }}
                    >
                      {size.label}
                    </button>
                  ))}
                </div>
              </div>

              {/* Steps */}
              <div>
                <label className="text-xs font-medium mb-1.5 flex items-center justify-between" style={{ color: "var(--color-text-tertiary)" }}>
                  <span>Inference Steps</span>
                  <span style={{ color: "var(--color-text-primary)" }}>{steps}</span>
                </label>
                <input
                  type="range"
                  min={5}
                  max={50}
                  value={steps}
                  onChange={(e) => setSteps(Number(e.target.value))}
                  className="w-full accent-indigo-500 cursor-pointer"
                />
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

          {/* Generate button */}
          <button
            onClick={handleGenerate}
            disabled={!prompt.trim() || loading}
            className="w-full py-3 rounded-xl text-sm font-semibold text-white flex items-center justify-center gap-2 transition-all hover:opacity-90 disabled:opacity-40 cursor-pointer"
            style={{ background: "linear-gradient(135deg, #6366f1, #8b5cf6)" }}
          >
            {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Sparkles className="w-4 h-4" />}
            {loading ? "Generating..." : "Generate Image"}
          </button>

          {/* Loading placeholder */}
          {loading && (
            <div className="rounded-xl aspect-square max-w-sm mx-auto animate-shimmer" />
          )}

          {/* Generated images */}
          {generatedImages.length > 0 && (
            <div>
              <p className="text-xs font-medium mb-3" style={{ color: "var(--color-text-tertiary)" }}>Generated Images</p>
              <div className="grid grid-cols-2 gap-3">
                {generatedImages.map((img, i) => (
                  <div key={i} className="group relative rounded-xl overflow-hidden border animate-fade-in-up" style={{ borderColor: "var(--color-border-primary)" }}>
                    <img src={img.url} alt={img.prompt} className="w-full aspect-square object-cover" />
                    <div className="absolute inset-0 bg-black/50 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
                      <button
                        onClick={() => downloadImage(img.url, i)}
                        className="p-2.5 rounded-xl bg-white/20 backdrop-blur-sm text-white cursor-pointer hover:bg-white/30 transition-colors"
                      >
                        <Download className="w-5 h-5" />
                      </button>
                    </div>
                    <p className="absolute bottom-0 left-0 right-0 p-2 text-[10px] text-white bg-black/60 truncate">
                      {img.prompt}
                    </p>
                  </div>
                ))}
              </div>
            </div>
          )}

          <p className="text-[11px] text-center" style={{ color: "var(--color-text-muted)" }}>
            Images generated locally using Stable Diffusion. No internet required.
          </p>
        </div>
      </div>
    </div>
  );
}
