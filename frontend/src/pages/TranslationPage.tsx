import { useState } from "react";
import { api } from "../utils/api";
import {
  ArrowLeftRight,
  Languages,
  Copy,
  Check,
  Loader2,
  AlertTriangle,
  Wand2,
} from "lucide-react";

const LANGUAGES = [
  { code: "en", name: "English" },
  { code: "hi", name: "Hindi" },
  { code: "es", name: "Spanish" },
  { code: "fr", name: "French" },
  { code: "de", name: "German" },
  { code: "ja", name: "Japanese" },
  { code: "zh", name: "Chinese" },
  { code: "ko", name: "Korean" },
  { code: "ru", name: "Russian" },
  { code: "ar", name: "Arabic" },
];

export default function TranslationPage() {
  const [sourceText, setSourceText] = useState("");
  const [translatedText, setTranslatedText] = useState("");
  const [sourceLang, setSourceLang] = useState("en");
  const [targetLang, setTargetLang] = useState("hi");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const [detecting, setDetecting] = useState(false);

  const handleTranslate = async () => {
    if (!sourceText.trim()) return;
    setLoading(true);
    setError(null);

    try {
      const result = await api.post<{ translated: string; success: boolean }>("/translate", {
        text: sourceText,
        source_lang: sourceLang,
        target_lang: targetLang,
      });
      setTranslatedText(result.translated);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Translation failed");
    } finally {
      setLoading(false);
    }
  };

  const swapLanguages = () => {
    setSourceLang(targetLang);
    setTargetLang(sourceLang);
    setSourceText(translatedText);
    setTranslatedText(sourceText);
  };

  const detectLanguage = async () => {
    if (!sourceText.trim()) return;
    setDetecting(true);
    try {
      const result = await api.post<{ language: string }>("/translate/detect", null);
      if (result.language) setSourceLang(result.language);
    } catch {
      // silent fail
    } finally {
      setDetecting(false);
    }
  };

  const copyTranslation = async () => {
    if (!translatedText) return;
    await navigator.clipboard.writeText(translatedText);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="flex-1 flex flex-col overflow-hidden">
      {/* Header */}
      <div className="h-14 flex items-center px-6 border-b shrink-0" style={{ borderColor: "var(--color-border-primary)" }}>
        <Languages className="w-5 h-5 mr-2" style={{ color: "var(--color-accent-primary)" }} />
        <h2 className="text-sm font-semibold" style={{ color: "var(--color-text-primary)" }}>Translation</h2>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto p-6">
        <div className="max-w-4xl mx-auto">
          {/* Language selectors */}
          <div className="flex items-center gap-3 mb-4">
            <select
              value={sourceLang}
              onChange={(e) => setSourceLang(e.target.value)}
              className="flex-1 px-4 py-2.5 rounded-xl text-sm font-medium outline-none border cursor-pointer"
              style={{
                background: "var(--color-bg-tertiary)",
                borderColor: "var(--color-border-primary)",
                color: "var(--color-text-primary)",
              }}
            >
              {LANGUAGES.map((l) => <option key={l.code} value={l.code}>{l.name}</option>)}
            </select>

            <button
              onClick={swapLanguages}
              className="p-2.5 rounded-xl transition-all duration-300 hover:rotate-180 border cursor-pointer shrink-0"
              style={{
                background: "var(--color-bg-tertiary)",
                borderColor: "var(--color-border-primary)",
                color: "var(--color-text-secondary)",
              }}
            >
              <ArrowLeftRight className="w-4 h-4" />
            </button>

            <select
              value={targetLang}
              onChange={(e) => setTargetLang(e.target.value)}
              className="flex-1 px-4 py-2.5 rounded-xl text-sm font-medium outline-none border cursor-pointer"
              style={{
                background: "var(--color-bg-tertiary)",
                borderColor: "var(--color-border-primary)",
                color: "var(--color-text-primary)",
              }}
            >
              {LANGUAGES.map((l) => <option key={l.code} value={l.code}>{l.name}</option>)}
            </select>
          </div>

          {/* Panels */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
            {/* Source */}
            <div className="rounded-xl border overflow-hidden" style={{ borderColor: "var(--color-border-primary)", background: "var(--color-bg-surface)" }}>
              <textarea
                value={sourceText}
                onChange={(e) => setSourceText(e.target.value)}
                placeholder="Enter text to translate..."
                rows={8}
                className="w-full p-4 text-sm bg-transparent outline-none resize-none"
                style={{ color: "var(--color-text-primary)" }}
              />
              <div className="flex items-center justify-between px-4 py-2 border-t" style={{ borderColor: "var(--color-border-primary)" }}>
                <span className="text-xs" style={{ color: "var(--color-text-muted)" }}>
                  {sourceText.length} characters
                </span>
                <button
                  onClick={detectLanguage}
                  disabled={detecting || !sourceText.trim()}
                  className="text-xs font-medium flex items-center gap-1 disabled:opacity-40 cursor-pointer"
                  style={{ color: "var(--color-accent-primary)" }}
                >
                  {detecting ? <Loader2 className="w-3 h-3 animate-spin" /> : <Wand2 className="w-3 h-3" />}
                  Detect language
                </button>
              </div>
            </div>

            {/* Target */}
            <div className="rounded-xl border overflow-hidden" style={{ borderColor: "var(--color-border-primary)", background: "var(--color-bg-surface)" }}>
              <div className="w-full p-4 text-sm min-h-[200px]" style={{ color: translatedText ? "var(--color-text-primary)" : "var(--color-text-muted)" }}>
                {loading ? (
                  <div className="flex items-center gap-2">
                    <Loader2 className="w-4 h-4 animate-spin" style={{ color: "var(--color-accent-primary)" }} />
                    <span>Translating...</span>
                  </div>
                ) : translatedText || "Translation will appear here..."}
              </div>
              <div className="flex items-center justify-end px-4 py-2 border-t" style={{ borderColor: "var(--color-border-primary)" }}>
                <button
                  onClick={copyTranslation}
                  disabled={!translatedText}
                  className="text-xs font-medium flex items-center gap-1 disabled:opacity-40 cursor-pointer"
                  style={{ color: "var(--color-accent-primary)" }}
                >
                  {copied ? <Check className="w-3 h-3" /> : <Copy className="w-3 h-3" />}
                  {copied ? "Copied!" : "Copy"}
                </button>
              </div>
            </div>
          </div>

          {/* Error */}
          {error && (
            <div className="p-3 rounded-xl text-sm flex items-center gap-2 mb-4" style={{ background: "var(--color-error-bg)", color: "var(--color-error)" }}>
              <AlertTriangle className="w-4 h-4 shrink-0" />
              {error}
            </div>
          )}

          {/* Translate button */}
          <button
            onClick={handleTranslate}
            disabled={!sourceText.trim() || loading}
            className="w-full py-3 rounded-xl text-sm font-semibold text-white flex items-center justify-center gap-2 transition-all hover:opacity-90 disabled:opacity-40 cursor-pointer"
            style={{ background: "linear-gradient(135deg, #6366f1, #8b5cf6)" }}
          >
            {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Languages className="w-4 h-4" />}
            {loading ? "Translating..." : "Translate"}
          </button>

          <p className="text-[11px] text-center mt-3" style={{ color: "var(--color-text-muted)" }}>
            Translation runs offline using local models. No internet required.
          </p>
        </div>
      </div>
    </div>
  );
}
