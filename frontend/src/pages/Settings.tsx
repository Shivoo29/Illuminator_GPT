import { useState, useEffect } from "react";
import { api } from "../utils/api";
import { useTheme } from "../contexts/ThemeContext";
import {
  Settings as SettingsIcon,
  Bot,
  Puzzle,
  HardDrive,
  Activity,
  Info,
  Loader2,
  Sun,
  Moon,
  AlertTriangle,
  Mic,
  Palette,
  Globe,
  Download,
  Cpu,
  MemoryStick,
  Monitor,
  X,
} from "lucide-react";

interface ModelInfo {
  name: string;
  path: string;
  size_gb: number;
  type: string;
}

interface FeatureStatus {
  installed: boolean;
  size_gb: number;
}

interface StorageInfo {
  total_used_gb: number;
  breakdown: {
    models: { gb: number };
    vector_database: { gb: number };
    documents: { gb: number };
    cache: { gb: number };
    outputs: { gb: number };
  };
  disk: {
    total_gb: number;
    free_gb: number;
    used_percent: number;
  };
}

interface PerformanceStats {
  cpu: { percent: number; cores: number };
  memory: { total_gb: number; used_gb: number; available_gb: number; percent: number };
  gpu: { utilization_percent: number; memory_used_mb: number; memory_total_mb: number } | null;
}

export default function Settings() {
  const { theme, toggleTheme } = useTheme();
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [features, setFeatures] = useState<Record<string, FeatureStatus>>({});
  const [storage, setStorage] = useState<StorageInfo | null>(null);
  const [performance, setPerformance] = useState<PerformanceStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [clearingCache, setClearingCache] = useState(false);
  const [resetting, setResetting] = useState(false);

  // Download states
  const [downloadingFeature, setDownloadingFeature] = useState<string | null>(null);
  const [featureProgress, setFeatureProgress] = useState<{ message: string; percent: number } | null>(null);

  const [showModelModal, setShowModelModal] = useState(false);
  const [availableModels, setAvailableModels] = useState<ModelInfo[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>("");
  const [downloadingModel, setDownloadingModel] = useState(false);
  const [modelProgress, setModelProgress] = useState<{ message: string; percent: number } | null>(null);
  const [ollamaInstalled, setOllamaInstalled] = useState(false);

  useEffect(() => {
    loadSettings();
  }, []);

  const loadSettings = async () => {
    try {
      const [modelsData, featuresData, storageData, perfData] = await Promise.all([
        api.get<{ installed: ModelInfo[] }>("/setup/available_models"),
        api.get<Record<string, FeatureStatus>>("/setup/feature_status"),
        api.get<StorageInfo>("/system/storage"),
        api.get<PerformanceStats>("/system/performance").catch(() => null),
      ]);
      setModels(modelsData.installed || []);
      setFeatures(featuresData);
      setStorage(storageData);
      if (perfData) setPerformance(perfData);
    } catch (error) {
      console.error("Failed to load settings:", error);
    } finally {
      setLoading(false);
    }
  };

  const clearCache = async () => {
    setClearingCache(true);
    try {
      await api.post("/system/clear_cache", {});
      await loadSettings();
    } catch (error) {
      console.error("Failed to clear cache:", error);
    } finally {
      setClearingCache(false);
    }
  };

  const resetApp = async () => {
    if (!confirm("This will delete ALL documents, vectors, and settings. Are you sure?")) return;
    setResetting(true);
    try {
      await api.post("/system/reset?confirm=true", null);
      window.location.reload();
    } catch (error) {
      console.error("Failed to reset:", error);
      setResetting(false);
    }
  };

  const downloadFeature = async (featureKey: string) => {
    setDownloadingFeature(featureKey);
    setFeatureProgress({ message: "Starting...", percent: 0 });
    try {
      const apiKey = featureKey === 'image_generation' ? 'image_generation' : featureKey;
      await new Promise<void>((resolve, reject) => {
        api.stream("/setup/download_feature", { feature: apiKey }, {
          onProgress: (data) => setFeatureProgress({
            message: `Downloading...`,
            percent: data.progress_percent || 0,
          }),
          onComplete: () => resolve(),
          onError: (err) => reject(err),
        });
      });
      await loadSettings();
    } catch (error) {
      console.error("Failed to download feature:", error);
    } finally {
      setDownloadingFeature(null);
      setFeatureProgress(null);
    }
  };

  const openModelModal = async () => {
    setShowModelModal(true);
    try {
      const [modelsData, ollamaData] = await Promise.all([
        api.get<{ models: ModelInfo[] }>("/setup/available_models/offline"),
        api.get<{ installed: boolean }>("/setup/check_ollama")
      ]);
      setAvailableModels(modelsData.models || []);
      setOllamaInstalled(ollamaData.installed);
    } catch (error) {
      console.error("Failed to load available models:", error);
    }
  };

  const downloadModel = async () => {
    if (!selectedModel) return;
    setDownloadingModel(true);
    setModelProgress({ message: "Starting...", percent: 0 });
    try {
      await new Promise<void>((resolve, reject) => {
        api.stream("/setup/download_model", { model_name: selectedModel, use_ollama: ollamaInstalled }, {
          onProgress: (data) => setModelProgress({
            message: data.message || "Downloading...",
            percent: data.progress_percent || 0,
          }),
          onComplete: () => resolve(),
          onError: (err) => reject(err),
        });
      });
      setShowModelModal(false);
      await loadSettings();
    } catch (error) {
      console.error("Failed to download model:", error);
    } finally {
      setDownloadingModel(false);
      setModelProgress(null);
    }
  };

  if (loading) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <Loader2 className="w-6 h-6 animate-spin" style={{ color: "var(--color-accent-primary)" }} />
      </div>
    );
  }

  const storageSegments = storage ? [
    { label: "Models", gb: storage.breakdown.models.gb, color: "#6366f1" },
    { label: "Vectors", gb: storage.breakdown.vector_database.gb, color: "#8b5cf6" },
    { label: "Documents", gb: storage.breakdown.documents.gb, color: "#a78bfa" },
    { label: "Cache", gb: storage.breakdown.cache.gb, color: "#c4b5fd" },
    { label: "Outputs", gb: storage.breakdown.outputs.gb, color: "#ddd6fe" },
  ] : [];

  return (
    <div className="flex-1 flex flex-col overflow-hidden">
      {/* Header */}
      <div className="h-14 flex items-center px-6 border-b shrink-0" style={{ borderColor: "var(--color-border-primary)" }}>
        <SettingsIcon className="w-5 h-5 mr-2" style={{ color: "var(--color-accent-primary)" }} />
        <h2 className="text-sm font-semibold" style={{ color: "var(--color-text-primary)" }}>Settings</h2>
      </div>

      <div className="flex-1 overflow-y-auto p-6">
        <div className="max-w-2xl mx-auto space-y-6">

          {/* Appearance */}
          <section className="rounded-xl border p-5" style={{ borderColor: "var(--color-border-primary)", background: "var(--color-bg-surface)" }}>
            <div className="flex items-center gap-2 mb-4">
              {theme === "dark" ? <Moon className="w-4 h-4" /> : <Sun className="w-4 h-4" />}
              <h3 className="text-sm font-semibold" style={{ color: "var(--color-text-primary)" }}>Appearance</h3>
            </div>
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium" style={{ color: "var(--color-text-primary)" }}>Theme</p>
                <p className="text-xs" style={{ color: "var(--color-text-tertiary)" }}>
                  Currently using {theme} mode
                </p>
              </div>
              <button
                onClick={toggleTheme}
                className="relative w-12 h-7 rounded-full transition-all duration-300 cursor-pointer"
                style={{ background: theme === "dark" ? "linear-gradient(135deg, #6366f1, #8b5cf6)" : "var(--color-bg-tertiary)" }}
              >
                <div
                  className="absolute top-1 w-5 h-5 rounded-full bg-white transition-all duration-300 flex items-center justify-center"
                  style={{ left: theme === "dark" ? "calc(100% - 24px)" : "4px" }}
                >
                  {theme === "dark" ? <Moon className="w-3 h-3 text-indigo-600" /> : <Sun className="w-3 h-3 text-amber-500" />}
                </div>
              </button>
            </div>
          </section>

          {/* Models */}
          <section className="rounded-xl border p-5" style={{ borderColor: "var(--color-border-primary)", background: "var(--color-bg-surface)" }}>
            <div className="flex items-center gap-2 mb-4">
              <Bot className="w-4 h-4" style={{ color: "var(--color-accent-primary)" }} />
              <h3 className="text-sm font-semibold" style={{ color: "var(--color-text-primary)" }}>Language Models</h3>
            </div>
            <div className="space-y-2">
              {models.length > 0 ? (
                models.map((model) => (
                  <div key={model.name} className="flex items-center justify-between p-3.5 rounded-xl" style={{ background: "var(--color-bg-tertiary)" }}>
                    <div>
                      <p className="text-sm font-medium" style={{ color: "var(--color-text-primary)" }}>{model.name}</p>
                      <p className="text-xs" style={{ color: "var(--color-text-muted)" }}>{model.size_gb} GB · {model.type?.toUpperCase()}</p>
                    </div>
                    <span className="px-2.5 py-1 rounded-full text-[10px] font-bold" style={{ background: "var(--color-success-bg)", color: "var(--color-success)" }}>
                      ACTIVE
                    </span>
                  </div>
                ))
              ) : (
                <p className="text-sm" style={{ color: "var(--color-text-muted)" }}>No models installed</p>
              )}
            </div>
            <button
              onClick={openModelModal}
              className="mt-3 text-xs font-semibold flex items-center gap-1.5 cursor-pointer" style={{ color: "var(--color-accent-primary)" }}
            >
              <Download className="w-3.5 h-3.5" /> Download New Model
            </button>
          </section>

          {/* Features */}
          <section className="rounded-xl border p-5" style={{ borderColor: "var(--color-border-primary)", background: "var(--color-bg-surface)" }}>
            <div className="flex items-center gap-2 mb-4">
              <Puzzle className="w-4 h-4" style={{ color: "var(--color-accent-primary)" }} />
              <h3 className="text-sm font-semibold" style={{ color: "var(--color-text-primary)" }}>Features</h3>
            </div>
            <div className="space-y-2">
              {[
                { key: "tts", icon: Mic, label: "Podcast Generation", size: "0.1" },
                { key: "image_generation", icon: Palette, label: "Image Generation", size: "2.0" },
                { key: "translation", icon: Globe, label: "Translation", size: "1.5" },
              ].map((feature) => (
                <div key={feature.key} className="flex items-center justify-between p-3.5 rounded-xl" style={{ background: "var(--color-bg-tertiary)" }}>
                  <div className="flex items-center gap-3">
                    <div className="w-8 h-8 rounded-lg flex items-center justify-center" style={{
                      background: features[feature.key]?.installed ? "var(--color-success-bg)" : "var(--color-bg-hover)"
                    }}>
                      <feature.icon className="w-4 h-4" style={{
                        color: features[feature.key]?.installed ? "var(--color-success)" : "var(--color-text-muted)"
                      }} />
                    </div>
                    <div>
                      <p className="text-sm font-medium" style={{ color: "var(--color-text-primary)" }}>{feature.label}</p>
                      <p className="text-xs" style={{ color: "var(--color-text-muted)" }}>{feature.size} GB</p>
                    </div>
                  </div>
                  {features[feature.key]?.installed ? (
                    <span
                      className="px-2.5 py-1 rounded-full text-[10px] font-bold"
                      style={{
                        background: "var(--color-success-bg)",
                        color: "var(--color-success)",
                      }}
                    >
                      INSTALLED
                    </span>
                  ) : downloadingFeature === feature.key ? (
                    <div className="flex items-center gap-2 w-32">
                      <div className="flex-1 h-1.5 rounded-full overflow-hidden" style={{ background: "var(--color-bg-hover)" }}>
                        <div className="h-full transition-all duration-300" style={{ width: `${featureProgress?.percent || 0}%`, background: "var(--color-accent-primary)" }} />
                      </div>
                      <span className="text-[10px] tabular-nums" style={{ color: "var(--color-text-muted)" }}>
                        {featureProgress?.percent.toFixed(0)}%
                      </span>
                    </div>
                  ) : (
                    <button
                      onClick={() => downloadFeature(feature.key)}
                      className="px-3 py-1.5 rounded-lg text-[10px] font-bold flex items-center gap-1 transition-all hover:opacity-90 cursor-pointer"
                      style={{ background: "var(--color-accent-bg)", color: "var(--color-accent-primary)" }}
                    >
                      <Download className="w-3 h-3" /> DOWNLOAD
                    </button>
                  )}
                </div>
              ))}
            </div>
          </section>

          {/* Storage */}
          {storage && (
            <section className="rounded-xl border p-5" style={{ borderColor: "var(--color-border-primary)", background: "var(--color-bg-surface)" }}>
              <div className="flex items-center gap-2 mb-4">
                <HardDrive className="w-4 h-4" style={{ color: "var(--color-accent-primary)" }} />
                <h3 className="text-sm font-semibold" style={{ color: "var(--color-text-primary)" }}>Storage</h3>
              </div>

              {/* Totals */}
              <div className="flex justify-between text-xs mb-2">
                <span style={{ color: "var(--color-text-secondary)" }}>Used: {storage.total_used_gb} GB</span>
                <span style={{ color: "var(--color-text-muted)" }}>Free: {storage.disk.free_gb} GB</span>
              </div>

              {/* Segmented bar */}
              <div className="h-2.5 rounded-full overflow-hidden flex mb-4" style={{ background: "var(--color-bg-tertiary)" }}>
                {storageSegments.map((seg) =>
                  seg.gb > 0 ? (
                    <div
                      key={seg.label}
                      className="h-full transition-all"
                      style={{
                        width: `${Math.max((seg.gb / storage.disk.total_gb) * 100, 1)}%`,
                        background: seg.color,
                      }}
                    />
                  ) : null
                )}
              </div>

              {/* Breakdown grid */}
              <div className="grid grid-cols-3 gap-2">
                {storageSegments.map((seg) => (
                  <div key={seg.label} className="flex items-center gap-2 text-xs">
                    <div className="w-2.5 h-2.5 rounded-full shrink-0" style={{ background: seg.color }} />
                    <span style={{ color: "var(--color-text-tertiary)" }}>{seg.label}</span>
                    <span className="font-medium" style={{ color: "var(--color-text-primary)" }}>{seg.gb} GB</span>
                  </div>
                ))}
              </div>

              <button
                onClick={clearCache}
                disabled={clearingCache}
                className="mt-4 px-4 py-2 rounded-xl text-xs font-medium border transition-colors disabled:opacity-50 cursor-pointer"
                style={{
                  borderColor: "var(--color-border-primary)",
                  color: "var(--color-text-secondary)",
                  background: "var(--color-bg-tertiary)",
                }}
              >
                {clearingCache ? "Clearing..." : "Clear Cache"}
              </button>
            </section>
          )}

          {/* Performance */}
          {performance && (
            <section className="rounded-xl border p-5" style={{ borderColor: "var(--color-border-primary)", background: "var(--color-bg-surface)" }}>
              <div className="flex items-center gap-2 mb-4">
                <Activity className="w-4 h-4" style={{ color: "var(--color-accent-primary)" }} />
                <h3 className="text-sm font-semibold" style={{ color: "var(--color-text-primary)" }}>Performance</h3>
              </div>
              <div className="grid grid-cols-2 gap-3">
                {[
                  { icon: Cpu, label: "CPU", value: `${performance.cpu.percent}%`, sub: `${performance.cpu.cores} cores` },
                  { icon: MemoryStick, label: "RAM", value: `${performance.memory.percent}%`, sub: `${performance.memory.used_gb}/${performance.memory.total_gb} GB` },
                  ...(performance.gpu ? [{
                    icon: Monitor, label: "GPU", value: `${performance.gpu.utilization_percent}%`, sub: `${performance.gpu.memory_used_mb}/${performance.gpu.memory_total_mb} MB`
                  }] : []),
                ].map((stat) => (
                  <div key={stat.label} className="p-3.5 rounded-xl" style={{ background: "var(--color-bg-tertiary)" }}>
                    <div className="flex items-center gap-2 mb-2">
                      <stat.icon className="w-4 h-4" style={{ color: "var(--color-text-muted)" }} />
                      <span className="text-xs font-medium" style={{ color: "var(--color-text-tertiary)" }}>{stat.label}</span>
                    </div>
                    <p className="text-lg font-bold" style={{ color: "var(--color-text-primary)" }}>{stat.value}</p>
                    <p className="text-[10px]" style={{ color: "var(--color-text-muted)" }}>{stat.sub}</p>
                  </div>
                ))}
              </div>
            </section>
          )}

          {/* About */}
          <section className="rounded-xl border p-5" style={{ borderColor: "var(--color-border-primary)", background: "var(--color-bg-surface)" }}>
            <div className="flex items-center gap-2 mb-4">
              <Info className="w-4 h-4" style={{ color: "var(--color-accent-primary)" }} />
              <h3 className="text-sm font-semibold" style={{ color: "var(--color-text-primary)" }}>About</h3>
            </div>
            <div className="space-y-1.5 text-sm" style={{ color: "var(--color-text-secondary)" }}>
              <p><strong style={{ color: "var(--color-text-primary)" }}>Illuminator GPT</strong> v1.0.0</p>
              <p>A fully offline, privacy-focused AI document assistant.</p>
              <p className="text-xs" style={{ color: "var(--color-text-muted)" }}>All processing happens locally. No data leaves your device.</p>
            </div>
          </section>

          {/* Danger zone */}
          <section className="rounded-xl border p-5" style={{ borderColor: "var(--color-error)", background: "var(--color-error-bg)" }}>
            <div className="flex items-center gap-2 mb-3">
              <AlertTriangle className="w-4 h-4" style={{ color: "var(--color-error)" }} />
              <h3 className="text-sm font-semibold" style={{ color: "var(--color-error)" }}>Danger Zone</h3>
            </div>
            <p className="text-xs mb-3" style={{ color: "var(--color-text-secondary)" }}>
              Reset will delete all documents, vectors, and settings.
            </p>
            <button
              onClick={resetApp}
              disabled={resetting}
              className="px-4 py-2 rounded-xl text-xs font-semibold text-white transition-all hover:opacity-90 disabled:opacity-50 cursor-pointer"
              style={{ background: "var(--color-error)" }}
            >
              {resetting ? "Resetting..." : "Reset Application"}
            </button>
          </section>

        </div>
      </div>

      {/* Download Model Modal */}
      {showModelModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 animate-fade-in" style={{ background: "var(--color-overlay)", backdropFilter: "blur(4px)" }}>
          <div className="w-full max-w-md rounded-2xl p-6 shadow-2xl animate-scale-in" style={{ background: "var(--color-bg-elevated)", border: "1px solid var(--color-border-primary)" }}>
            <div className="flex justify-between items-center mb-4">
              <h3 className="text-lg font-semibold" style={{ color: "var(--color-text-primary)" }}>Download New Model</h3>
              <button 
                onClick={() => !downloadingModel && setShowModelModal(false)}
                disabled={downloadingModel}
                className="p-1 rounded-lg transition-colors cursor-pointer" style={{ color: "var(--color-text-tertiary)" }}
              >
                <X className="w-5 h-5" />
              </button>
            </div>
            
            <div className="space-y-3 mb-6 max-h-[40vh] overflow-y-auto pr-2 custom-scrollbar">
              {availableModels.length > 0 ? availableModels.map((model) => {
                const isInstalled = models.some(m => m.name === model.name);
                return (
                  <button
                    key={model.name}
                    onClick={() => !isInstalled && !downloadingModel && setSelectedModel(model.name)}
                    disabled={isInstalled || downloadingModel}
                    className="w-full text-left p-3 rounded-xl transition-all border"
                    style={{
                      background: selectedModel === model.name ? "var(--color-accent-bg)" : "var(--color-bg-tertiary)",
                      borderColor: selectedModel === model.name ? "var(--color-accent-primary)" : "transparent",
                      opacity: isInstalled ? 0.5 : 1,
                      cursor: isInstalled || downloadingModel ? "default" : "pointer"
                    }}
                  >
                    <div className="flex justify-between items-center mb-1">
                      <span className="text-sm font-semibold" style={{ color: "var(--color-text-primary)" }}>{model.name}</span>
                      {isInstalled && <span className="text-[10px] font-bold" style={{ color: "var(--color-success)" }}>INSTALLED</span>}
                    </div>
                    <div className="flex gap-2 text-[10px]" style={{ color: "var(--color-text-muted)" }}>
                      <span>{model.size_gb} GB</span>
                    </div>
                  </button>
                );
              }) : (
                <div className="py-8 flex justify-center">
                  <Loader2 className="w-6 h-6 animate-spin" style={{ color: "var(--color-text-muted)" }} />
                </div>
              )}
            </div>

            {downloadingModel ? (
              <div className="space-y-2">
                <div className="h-2 rounded-full overflow-hidden" style={{ background: "var(--color-bg-tertiary)" }}>
                  <div className="h-full transition-all duration-300" style={{ width: `${modelProgress?.percent || 0}%`, background: "var(--color-accent-primary)" }} />
                </div>
                <p className="text-xs text-center" style={{ color: "var(--color-text-tertiary)" }}>
                  {modelProgress?.message} ({modelProgress?.percent.toFixed(0)}%)
                </p>
              </div>
            ) : (
              <button
                onClick={downloadModel}
                disabled={!selectedModel || downloadingModel}
                className="w-full py-2.5 rounded-xl text-sm font-semibold text-white transition-all hover:opacity-90 disabled:opacity-50 cursor-pointer flex items-center justify-center gap-2"
                style={{ background: "linear-gradient(135deg, #6366f1, #8b5cf6)" }}
              >
                <Download className="w-4 h-4" /> Download
              </button>
            )}
          </div>
        </div>
      )}
    </div>
  );
}