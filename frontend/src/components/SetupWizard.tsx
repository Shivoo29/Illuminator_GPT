import { useState, useEffect } from "react";
import { api } from "../utils/api";
import {
  HardDrive,
  Cpu,
  MonitorCheck,
  CircuitBoard,
  Download,
  Check,
  AlertTriangle,
  Sparkles,
  Loader2,
  ArrowRight,
  Mic,
  Palette,
  Globe,
  Zap,
} from "lucide-react";

interface SystemInfo {
  platform: string;
  cpu_count: number;
  ram_gb: number;
  disk_free_gb: number;
  gpu_available: boolean;
  gpu_name: string | null;
  sufficient: boolean;
}

interface ModelInfo {
  name: string;
  display_name: string;
  size_gb: number;
  description: string;
  recommended: boolean;
  speed: string;
  quality: string;
  requirements: string;
}

interface SetupWizardProps {
  onComplete: () => void;
}

const STEPS = ["System Check", "LLM Engine", "Model", "Features"];

export default function SetupWizard({ onComplete }: SetupWizardProps) {
  const [step, setStep] = useState(1);
  const [systemInfo, setSystemInfo] = useState<SystemInfo | null>(null);
  const [ollamaStatus, setOllamaStatus] = useState<{ installed: boolean; models: any[] } | null>(null);
  const [availableModels, setAvailableModels] = useState<ModelInfo[]>([]);
  const [selectedModel, setSelectedModel] = useState("llama3.2-7b");
  const [selectedFeatures, setSelectedFeatures] = useState({
    tts: false,
    imageGen: false,
    translation: false,
  });
  const [downloading, setDownloading] = useState(false);
  const [downloadProgress, setDownloadProgress] = useState({
    progress_percent: 0,
    message: "",
    status: "",
  });
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (step === 1) checkSystem();
  }, [step]);

  const checkSystem = async () => {
    try {
      const [system, ollama, modelsData] = await Promise.all([
        api.get<SystemInfo>("/setup/check_system"),
        api.get<{ installed: boolean; models: any[] }>("/setup/check_ollama"),
        api.get<{ models: ModelInfo[]; installed: any[]; recommended: string }>("/setup/available_models/offline"),
      ]);
      setSystemInfo(system);
      setOllamaStatus(ollama);
      setAvailableModels(modelsData.models || []);
      if (modelsData.recommended) setSelectedModel(modelsData.recommended);
    } catch {
      setError("Failed to check system requirements");
    }
  };

  const installOllama = async () => {
    setDownloading(true);
    setError(null);
    await api.stream("/setup/install_ollama", {}, {
      onProgress: (data) => setDownloadProgress({
        progress_percent: data.progress_percent || 0,
        message: data.message || "Installing...",
        status: data.status || "downloading",
      }),
      onComplete: async () => {
        setDownloading(false);
        await checkSystem();
      },
      onError: (err) => {
        setDownloading(false);
        setError(err.message);
      },
    });
  };

  const downloadModel = async () => {
    setDownloading(true);
    setError(null);
    await api.stream("/setup/download_model", {
      model_name: selectedModel,
      use_ollama: ollamaStatus?.installed || false,
    }, {
      onProgress: (data) => setDownloadProgress({
        progress_percent: data.progress_percent || 0,
        message: data.message || "Downloading...",
        status: data.status || "downloading",
      }),
      onComplete: () => {
        setDownloading(false);
        setStep(4);
      },
      onError: (err) => {
        setDownloading(false);
        setError(err.message);
      },
    });
  };

  const completeSetup = async () => {
    setDownloading(true);
    try {
      // Setup features to download
      const featuresToDownload = [];
      if (selectedFeatures.tts) featuresToDownload.push("tts");
      if (selectedFeatures.imageGen) featuresToDownload.push("image_generation");
      if (selectedFeatures.translation) featuresToDownload.push("translation");

      // Download each feature sequentially
      for (const feature of featuresToDownload) {
        await new Promise<void>((resolve, reject) => {
          api.stream("/setup/download_feature", { feature }, {
            onProgress: (data) => setDownloadProgress({
              progress_percent: data.progress_percent || 0,
              message: `Downloading ${feature.replace('_', ' ')}...`,
              status: data.status || "downloading",
            }),
            onComplete: () => resolve(),
            onError: (err) => reject(err),
          });
        });
      }

      await api.post("/setup/complete_setup", {});
      onComplete();
    } catch (err: any) {
      setError(err.message || "Failed to complete setup");
      setDownloading(false);
    }
  };

  return (
    <div
      className="min-h-screen flex items-center justify-center p-6"
      style={{ background: "var(--color-bg-primary)" }}
    >
      <div
        className="w-full max-w-[560px] rounded-2xl p-8 animate-fade-in-up glass"
        style={{ boxShadow: "var(--shadow-lg)" }}
      >
        {/* Header */}
        <div className="text-center mb-8">
          <div className="w-14 h-14 rounded-2xl mx-auto mb-4 flex items-center justify-center" style={{ background: "linear-gradient(135deg, #6366f1, #8b5cf6)" }}>
            <Sparkles className="w-7 h-7 text-white" />
          </div>
          <h1 className="text-2xl font-bold mb-1" style={{ color: "var(--color-text-primary)" }}>
            Welcome to Illuminator
          </h1>
          <p className="text-sm" style={{ color: "var(--color-text-tertiary)" }}>
            Let's set up your offline AI assistant
          </p>
        </div>

        {/* Step Progress */}
        <div className="flex gap-2 mb-8">
          {STEPS.map((label, i) => (
            <div key={label} className="flex-1">
              <div
                className="h-1 rounded-full transition-all duration-500"
                style={{
                  background: i + 1 <= step
                    ? "linear-gradient(90deg, #6366f1, #8b5cf6)"
                    : "var(--color-bg-tertiary)",
                }}
              />
              <p className="text-[10px] mt-1.5 text-center font-medium" style={{
                color: i + 1 === step ? "var(--color-accent-primary)" : "var(--color-text-muted)"
              }}>
                {label}
              </p>
            </div>
          ))}
        </div>

        {/* Error */}
        {error && (
          <div className="mb-6 p-3 rounded-xl flex items-center gap-2 text-sm" style={{ background: "var(--color-error-bg)", color: "var(--color-error)" }}>
            <AlertTriangle className="w-4 h-4 shrink-0" />
            {error}
          </div>
        )}

        {/* Step 1: System Check */}
        {step === 1 && (
          <div className="space-y-5 animate-fade-in">
            <h2 className="text-lg font-semibold" style={{ color: "var(--color-text-primary)" }}>System Check</h2>

            {!systemInfo ? (
              <div className="flex items-center justify-center py-10 gap-3" style={{ color: "var(--color-text-secondary)" }}>
                <Loader2 className="w-5 h-5 animate-spin" />
                <span className="text-sm">Checking system requirements...</span>
              </div>
            ) : (
              <>
                <div className="grid grid-cols-2 gap-3">
                  {[
                    { icon: HardDrive, label: "Disk Space", value: `${systemInfo.disk_free_gb.toFixed(1)} GB free`, ok: systemInfo.disk_free_gb > 20 },
                    { icon: CircuitBoard, label: "RAM", value: `${systemInfo.ram_gb.toFixed(1)} GB`, ok: systemInfo.ram_gb >= 8 },
                    { icon: Cpu, label: "GPU", value: systemInfo.gpu_available ? (systemInfo.gpu_name || "Available") : "CPU mode", ok: systemInfo.gpu_available },
                    { icon: MonitorCheck, label: "Platform", value: systemInfo.platform, ok: true },
                  ].map((item) => (
                    <div key={item.label} className="p-3.5 rounded-xl flex items-start gap-3" style={{ background: "var(--color-bg-tertiary)" }}>
                      <div className="w-8 h-8 rounded-lg flex items-center justify-center shrink-0" style={{
                        background: item.ok ? "var(--color-success-bg)" : "var(--color-warning-bg)"
                      }}>
                        <item.icon className="w-4 h-4" style={{ color: item.ok ? "var(--color-success)" : "var(--color-warning)" }} />
                      </div>
                      <div>
                        <p className="text-xs font-medium" style={{ color: "var(--color-text-tertiary)" }}>{item.label}</p>
                        <p className="text-sm font-semibold" style={{ color: "var(--color-text-primary)" }}>{item.value}</p>
                      </div>
                    </div>
                  ))}
                </div>

                {!systemInfo.sufficient && (
                  <div className="p-3 rounded-xl text-sm" style={{ background: "var(--color-warning-bg)", color: "var(--color-warning)" }}>
                    System doesn't meet minimum requirements. You may experience issues.
                  </div>
                )}

                <button
                  onClick={() => setStep(2)}
                  className="w-full py-3 rounded-xl text-sm font-semibold text-white flex items-center justify-center gap-2 transition-all duration-200 hover:opacity-90 cursor-pointer"
                  style={{ background: "linear-gradient(135deg, #6366f1, #8b5cf6)" }}
                >
                  Continue <ArrowRight className="w-4 h-4" />
                </button>
              </>
            )}
          </div>
        )}

        {/* Step 2: Ollama */}
        {step === 2 && (
          <div className="space-y-5 animate-fade-in">
            <h2 className="text-lg font-semibold" style={{ color: "var(--color-text-primary)" }}>LLM Engine Setup</h2>

            {ollamaStatus?.installed ? (
              <div className="space-y-4">
                <div className="p-4 rounded-xl flex items-center gap-3" style={{ background: "var(--color-success-bg)" }}>
                  <Check className="w-5 h-5" style={{ color: "var(--color-success)" }} />
                  <div>
                    <p className="text-sm font-semibold" style={{ color: "var(--color-success)" }}>Ollama is installed</p>
                    {ollamaStatus.models.length > 0 && (
                      <p className="text-xs mt-0.5" style={{ color: "var(--color-text-tertiary)" }}>
                        {ollamaStatus.models.length} model(s) available
                      </p>
                    )}
                  </div>
                </div>
                <button onClick={() => setStep(3)} className="w-full py-3 rounded-xl text-sm font-semibold text-white flex items-center justify-center gap-2 transition-all hover:opacity-90 cursor-pointer" style={{ background: "linear-gradient(135deg, #6366f1, #8b5cf6)" }}>
                  Continue <ArrowRight className="w-4 h-4" />
                </button>
              </div>
            ) : (
              <div className="space-y-4">
                <div className="p-4 rounded-xl" style={{ background: "var(--color-warning-bg)" }}>
                  <p className="text-sm font-medium" style={{ color: "var(--color-warning)" }}>Ollama is required to run AI models locally.</p>
                  <p className="text-xs mt-1" style={{ color: "var(--color-text-tertiary)" }}>Free, open-source, and keeps everything offline.</p>
                </div>

                {!downloading ? (
                  <button onClick={installOllama} className="w-full py-3 rounded-xl text-sm font-semibold text-white flex items-center justify-center gap-2 transition-all hover:opacity-90 cursor-pointer" style={{ background: "linear-gradient(135deg, #6366f1, #8b5cf6)" }}>
                    <Download className="w-4 h-4" /> Install Ollama
                  </button>
                ) : (
                  <div className="space-y-2">
                    <div className="h-2 rounded-full overflow-hidden" style={{ background: "var(--color-bg-tertiary)" }}>
                      <div className="h-full rounded-full transition-all duration-300" style={{ width: `${downloadProgress.progress_percent}%`, background: "linear-gradient(90deg, #6366f1, #8b5cf6)" }} />
                    </div>
                    <p className="text-xs text-center" style={{ color: "var(--color-text-tertiary)" }}>{downloadProgress.message}</p>
                  </div>
                )}

                <button onClick={() => setStep(3)} className="w-full py-2 text-sm cursor-pointer" style={{ color: "var(--color-text-muted)" }}>
                  Skip (use direct model loading)
                </button>
              </div>
            )}
          </div>
        )}

        {/* Step 3: Model Selection */}
        {step === 3 && (
          <div className="space-y-5 animate-fade-in">
            <h2 className="text-lg font-semibold" style={{ color: "var(--color-text-primary)" }}>Select Language Model</h2>

            <div className="space-y-2.5 max-h-[320px] overflow-y-auto pr-1">
              {availableModels.map((model) => (
                <button
                  key={model.name}
                  onClick={() => setSelectedModel(model.name)}
                  className="w-full text-left p-4 rounded-xl transition-all duration-200 cursor-pointer border"
                  style={{
                    background: selectedModel === model.name ? "var(--color-accent-bg)" : "var(--color-bg-tertiary)",
                    borderColor: selectedModel === model.name ? "var(--color-accent-primary)" : "transparent",
                  }}
                >
                  <div className="flex items-center gap-2 mb-1">
                    <span className="text-sm font-semibold" style={{ color: "var(--color-text-primary)" }}>{model.display_name}</span>
                    {model.recommended && (
                      <span className="px-2 py-0.5 rounded-full text-[10px] font-bold" style={{ background: "var(--color-success-bg)", color: "var(--color-success)" }}>
                        RECOMMENDED
                      </span>
                    )}
                  </div>
                  <p className="text-xs mb-2" style={{ color: "var(--color-text-tertiary)" }}>{model.description}</p>
                  <div className="flex gap-3 text-[11px] font-medium" style={{ color: "var(--color-text-muted)" }}>
                    <span>{model.size_gb} GB</span>
                    <span>Speed: {model.speed}</span>
                    <span>Quality: {model.quality}</span>
                  </div>
                </button>
              ))}
            </div>

            {!downloading ? (
              <button onClick={downloadModel} className="w-full py-3 rounded-xl text-sm font-semibold text-white flex items-center justify-center gap-2 transition-all hover:opacity-90 cursor-pointer" style={{ background: "linear-gradient(135deg, #6366f1, #8b5cf6)" }}>
                <Download className="w-4 h-4" /> Download & Install
              </button>
            ) : (
              <div className="space-y-2">
                <div className="h-2.5 rounded-full overflow-hidden" style={{ background: "var(--color-bg-tertiary)" }}>
                  <div className="h-full rounded-full transition-all duration-300" style={{ width: `${downloadProgress.progress_percent}%`, background: "linear-gradient(90deg, #6366f1, #8b5cf6)" }} />
                </div>
                <p className="text-xs text-center" style={{ color: "var(--color-text-tertiary)" }}>
                  {downloadProgress.message} ({downloadProgress.progress_percent.toFixed(0)}%)
                </p>
              </div>
            )}
          </div>
        )}

        {/* Step 4: Features */}
        {step === 4 && (
          <div className="space-y-5 animate-fade-in">
            <div>
              <h2 className="text-lg font-semibold" style={{ color: "var(--color-text-primary)" }}>Optional Features</h2>
              <p className="text-xs mt-1" style={{ color: "var(--color-text-tertiary)" }}>You can add these later from Settings</p>
            </div>

            <div className="space-y-2.5">
              {[
                { key: "tts", icon: Mic, label: "Podcast Generation", desc: "Generate audio discussions from documents", size: "100 MB" },
                { key: "imageGen", icon: Palette, label: "Image Generation", desc: "Create images from text descriptions", size: "2 GB" },
                { key: "translation", icon: Globe, label: "Translation", desc: "Translate between languages offline", size: "1.5 GB" },
              ].map((feature) => (
                <button
                  key={feature.key}
                  onClick={() => setSelectedFeatures((prev) => ({ ...prev, [feature.key]: !prev[feature.key as keyof typeof prev] }))}
                  className="w-full flex items-center gap-3.5 p-4 rounded-xl transition-all duration-200 cursor-pointer border text-left"
                  style={{
                    background: selectedFeatures[feature.key as keyof typeof selectedFeatures] ? "var(--color-accent-bg)" : "var(--color-bg-tertiary)",
                    borderColor: selectedFeatures[feature.key as keyof typeof selectedFeatures] ? "var(--color-accent-primary)" : "transparent",
                  }}
                >
                  <div className="w-10 h-10 rounded-xl flex items-center justify-center shrink-0" style={{
                    background: selectedFeatures[feature.key as keyof typeof selectedFeatures] ? "linear-gradient(135deg, #6366f1, #8b5cf6)" : "var(--color-bg-hover)"
                  }}>
                    <feature.icon className="w-5 h-5" style={{
                      color: selectedFeatures[feature.key as keyof typeof selectedFeatures] ? "white" : "var(--color-text-tertiary)"
                    }} />
                  </div>
                  <div className="flex-1">
                    <p className="text-sm font-semibold" style={{ color: "var(--color-text-primary)" }}>{feature.label}</p>
                    <p className="text-xs" style={{ color: "var(--color-text-tertiary)" }}>{feature.desc} · {feature.size}</p>
                  </div>
                  <div
                    className="w-5 h-5 rounded-md border-2 flex items-center justify-center shrink-0 transition-all"
                    style={{
                      borderColor: selectedFeatures[feature.key as keyof typeof selectedFeatures] ? "var(--color-accent-primary)" : "var(--color-border-secondary)",
                      background: selectedFeatures[feature.key as keyof typeof selectedFeatures] ? "var(--color-accent-primary)" : "transparent",
                    }}
                  >
                    {selectedFeatures[feature.key as keyof typeof selectedFeatures] && <Check className="w-3 h-3 text-white" />}
                  </div>
                </button>
              ))}
            </div>

            <button
              onClick={completeSetup}
              disabled={downloading}
              className="w-full py-3 rounded-xl text-sm font-semibold text-white flex items-center justify-center gap-2 transition-all hover:opacity-90 disabled:opacity-50 cursor-pointer"
              style={{ background: "linear-gradient(135deg, #6366f1, #8b5cf6)" }}
            >
              {downloading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Zap className="w-4 h-4" />}
              {downloading ? "Setting up..." : "Complete Setup"}
            </button>

            <button onClick={completeSetup} className="w-full py-2 text-sm cursor-pointer" style={{ color: "var(--color-text-muted)" }}>
              Skip optional features
            </button>
          </div>
        )}

        {/* Footer */}
        <div className="mt-8 text-center">
          <p className="text-[11px] font-medium" style={{ color: "var(--color-text-muted)" }}>
            100% Offline · No API Keys · Your Data Stays Private
          </p>
        </div>
      </div>
    </div>
  );
}