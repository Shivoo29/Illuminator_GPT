import { useState, useEffect } from "react";
import { api } from "../utils/api.ts";

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
  const [downloadProgress, setDownloadProgress] = useState<{
    progress_percent: number;
    message: string;
    status: string;
  }>({ progress_percent: 0, message: "", status: "" });
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (step === 1) {
      checkSystem();
    }
  }, [step]);

  const checkSystem = async () => {
    try {
      const [system, ollama, models] = await Promise.all([
        api.get<SystemInfo>("/setup/check_system"),
        api.get<{ installed: boolean; models: any[] }>("/setup/check_ollama"),
        api.get<{ models: ModelInfo[] }>("/setup/available_models"),
      ]);
      setSystemInfo(system);
      setOllamaStatus(ollama);
      setAvailableModels(models.models);
    } catch (err) {
      setError("Failed to check system requirements");
    }
  };

  const installOllama = async () => {
    setDownloading(true);
    setError(null);

    await api.stream("/setup/install_ollama", {}, {
      onProgress: (data) => {
        setDownloadProgress({
          progress_percent: data.progress_percent || 0,
          message: data.message || "Installing...",
          status: data.status || "downloading",
        });
      },
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
      onProgress: (data) => {
        setDownloadProgress({
          progress_percent: data.progress_percent || 0,
          message: data.message || "Downloading...",
          status: data.status || "downloading",
        });
      },
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
      await api.post("/setup/complete_setup", {});
      onComplete();
    } catch (err) {
      setError("Failed to complete setup");
      setDownloading(false);
    }
  };

  const calculateTotalSize = () => {
    let total = 0;
    const model = availableModels.find((m) => m.name === selectedModel);
    if (model) total += model.size_gb;
    if (selectedFeatures.tts) total += 0.1;
    if (selectedFeatures.imageGen) total += 2.0;
    if (selectedFeatures.translation) total += 1.5;
    return total.toFixed(1);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-primary-50 to-blue-100 flex items-center justify-center p-4">
      <div className="bg-white rounded-2xl shadow-xl max-w-2xl w-full p-8">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">
            Welcome to RAG Assistant
          </h1>
          <p className="text-gray-600">
            Let's get your offline AI assistant set up
          </p>
          {/* Progress indicators */}
          <div className="flex justify-center mt-6 space-x-2">
            {[1, 2, 3, 4].map((s) => (
              <div
                key={s}
                className={`w-3 h-3 rounded-full transition-colors ${
                  s === step
                    ? "bg-primary-600"
                    : s < step
                    ? "bg-primary-400"
                    : "bg-gray-300"
                }`}
              />
            ))}
          </div>
        </div>

        {error && (
          <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg text-red-700">
            {error}
          </div>
        )}

        {/* Step 1: System Check */}
        {step === 1 && (
          <div className="space-y-6">
            <h2 className="text-xl font-semibold text-gray-800">
              Step 1: System Check
            </h2>

            {!systemInfo ? (
              <div className="flex items-center justify-center py-8">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600"></div>
                <span className="ml-3 text-gray-600">Checking system...</span>
              </div>
            ) : (
              <div className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div className="p-4 bg-gray-50 rounded-lg">
                    <div className="flex items-center">
                      <span className={systemInfo.disk_free_gb > 20 ? "text-green-500" : "text-red-500"}>
                        {systemInfo.disk_free_gb > 20 ? "✓" : "✗"}
                      </span>
                      <span className="ml-2 font-medium">Disk Space</span>
                    </div>
                    <p className="text-gray-600 mt-1">
                      {systemInfo.disk_free_gb.toFixed(1)} GB free
                    </p>
                  </div>

                  <div className="p-4 bg-gray-50 rounded-lg">
                    <div className="flex items-center">
                      <span className={systemInfo.ram_gb >= 8 ? "text-green-500" : "text-yellow-500"}>
                        {systemInfo.ram_gb >= 8 ? "✓" : "!"}
                      </span>
                      <span className="ml-2 font-medium">RAM</span>
                    </div>
                    <p className="text-gray-600 mt-1">
                      {systemInfo.ram_gb.toFixed(1)} GB
                    </p>
                  </div>

                  <div className="p-4 bg-gray-50 rounded-lg">
                    <div className="flex items-center">
                      <span className={systemInfo.gpu_available ? "text-green-500" : "text-gray-400"}>
                        {systemInfo.gpu_available ? "✓" : "○"}
                      </span>
                      <span className="ml-2 font-medium">GPU</span>
                    </div>
                    <p className="text-gray-600 mt-1">
                      {systemInfo.gpu_available
                        ? systemInfo.gpu_name || "Available"
                        : "Not detected (CPU mode)"}
                    </p>
                  </div>

                  <div className="p-4 bg-gray-50 rounded-lg">
                    <div className="flex items-center">
                      <span className="text-blue-500">i</span>
                      <span className="ml-2 font-medium">Platform</span>
                    </div>
                    <p className="text-gray-600 mt-1">{systemInfo.platform}</p>
                  </div>
                </div>

                {systemInfo.sufficient ? (
                  <button
                    onClick={() => setStep(2)}
                    className="w-full py-3 bg-primary-600 text-white rounded-lg font-medium hover:bg-primary-700 transition-colors"
                  >
                    Continue Setup →
                  </button>
                ) : (
                  <div className="p-4 bg-yellow-50 border border-yellow-200 rounded-lg text-yellow-800">
                    Your system doesn't meet minimum requirements. You may experience issues.
                    <button
                      onClick={() => setStep(2)}
                      className="mt-2 text-yellow-600 underline"
                    >
                      Continue anyway
                    </button>
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {/* Step 2: Ollama Setup */}
        {step === 2 && (
          <div className="space-y-6">
            <h2 className="text-xl font-semibold text-gray-800">
              Step 2: LLM Engine Setup
            </h2>

            {ollamaStatus?.installed ? (
              <div className="space-y-4">
                <div className="p-4 bg-green-50 border border-green-200 rounded-lg">
                  <div className="flex items-center text-green-700">
                    <span className="text-xl mr-2">✓</span>
                    <span className="font-medium">Ollama is installed</span>
                  </div>
                  {ollamaStatus.models.length > 0 && (
                    <div className="mt-2 text-green-600">
                      {ollamaStatus.models.length} model(s) available
                    </div>
                  )}
                </div>
                <button
                  onClick={() => setStep(3)}
                  className="w-full py-3 bg-primary-600 text-white rounded-lg font-medium hover:bg-primary-700 transition-colors"
                >
                  Continue →
                </button>
              </div>
            ) : (
              <div className="space-y-4">
                <div className="p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
                  <p className="text-yellow-800 mb-2">
                    Ollama is required to run AI models locally.
                  </p>
                  <p className="text-yellow-700 text-sm">
                    It's free, open-source, and keeps everything offline.
                  </p>
                </div>

                {!downloading ? (
                  <button
                    onClick={installOllama}
                    className="w-full py-3 bg-primary-600 text-white rounded-lg font-medium hover:bg-primary-700 transition-colors"
                  >
                    Install Ollama
                  </button>
                ) : (
                  <div className="space-y-2">
                    <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-primary-600 transition-all duration-300"
                        style={{ width: `${downloadProgress.progress_percent}%` }}
                      />
                    </div>
                    <p className="text-sm text-gray-600 text-center">
                      {downloadProgress.message}
                    </p>
                  </div>
                )}

                <button
                  onClick={() => setStep(3)}
                  className="w-full py-2 text-gray-600 hover:text-gray-800"
                >
                  Skip (use direct model loading)
                </button>
              </div>
            )}
          </div>
        )}

        {/* Step 3: Model Selection */}
        {step === 3 && (
          <div className="space-y-6">
            <h2 className="text-xl font-semibold text-gray-800">
              Step 3: Select Language Model
            </h2>

            <div className="space-y-3">
              {availableModels.map((model) => (
                <label
                  key={model.name}
                  className={`block p-4 border-2 rounded-lg cursor-pointer transition-colors ${
                    selectedModel === model.name
                      ? "border-primary-500 bg-primary-50"
                      : "border-gray-200 hover:border-gray-300"
                  }`}
                >
                  <div className="flex items-start">
                    <input
                      type="radio"
                      name="model"
                      value={model.name}
                      checked={selectedModel === model.name}
                      onChange={(e) => setSelectedModel(e.target.value)}
                      className="mt-1"
                    />
                    <div className="ml-3 flex-1">
                      <div className="flex items-center">
                        <span className="font-medium">{model.display_name}</span>
                        {model.recommended && (
                          <span className="ml-2 px-2 py-0.5 bg-green-100 text-green-700 text-xs rounded-full">
                            Recommended
                          </span>
                        )}
                      </div>
                      <p className="text-sm text-gray-600 mt-1">{model.description}</p>
                      <div className="flex gap-4 mt-2 text-xs text-gray-500">
                        <span>Size: {model.size_gb} GB</span>
                        <span>Speed: {model.speed}</span>
                        <span>Quality: {model.quality}</span>
                      </div>
                    </div>
                  </div>
                </label>
              ))}
            </div>

            {!downloading ? (
              <button
                onClick={downloadModel}
                className="w-full py-3 bg-primary-600 text-white rounded-lg font-medium hover:bg-primary-700 transition-colors"
              >
                Download & Install Model
              </button>
            ) : (
              <div className="space-y-2">
                <div className="h-3 bg-gray-200 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-primary-600 transition-all duration-300"
                    style={{ width: `${downloadProgress.progress_percent}%` }}
                  />
                </div>
                <p className="text-sm text-gray-600 text-center">
                  {downloadProgress.message} ({downloadProgress.progress_percent.toFixed(0)}%)
                </p>
              </div>
            )}
          </div>
        )}

        {/* Step 4: Optional Features */}
        {step === 4 && (
          <div className="space-y-6">
            <h2 className="text-xl font-semibold text-gray-800">
              Step 4: Optional Features
            </h2>
            <p className="text-gray-600">
              Select additional features to install (you can add these later):
            </p>

            <div className="space-y-3">
              <label className="flex items-start p-4 border border-gray-200 rounded-lg cursor-pointer hover:bg-gray-50">
                <input
                  type="checkbox"
                  checked={selectedFeatures.tts}
                  onChange={(e) =>
                    setSelectedFeatures({ ...selectedFeatures, tts: e.target.checked })
                  }
                  className="mt-1"
                />
                <div className="ml-3">
                  <div className="font-medium">🎙️ Podcast Generation</div>
                  <p className="text-sm text-gray-600">
                    Generate audio discussions from documents (100 MB)
                  </p>
                </div>
              </label>

              <label className="flex items-start p-4 border border-gray-200 rounded-lg cursor-pointer hover:bg-gray-50">
                <input
                  type="checkbox"
                  checked={selectedFeatures.imageGen}
                  onChange={(e) =>
                    setSelectedFeatures({ ...selectedFeatures, imageGen: e.target.checked })
                  }
                  className="mt-1"
                />
                <div className="ml-3">
                  <div className="font-medium">🎨 Image Generation</div>
                  <p className="text-sm text-gray-600">
                    Create images from text descriptions (2 GB)
                  </p>
                </div>
              </label>

              <label className="flex items-start p-4 border border-gray-200 rounded-lg cursor-pointer hover:bg-gray-50">
                <input
                  type="checkbox"
                  checked={selectedFeatures.translation}
                  onChange={(e) =>
                    setSelectedFeatures({ ...selectedFeatures, translation: e.target.checked })
                  }
                  className="mt-1"
                />
                <div className="ml-3">
                  <div className="font-medium">🌐 Translation</div>
                  <p className="text-sm text-gray-600">
                    Translate between languages (1.5 GB for 5 language pairs)
                  </p>
                </div>
              </label>
            </div>

            <div className="p-4 bg-gray-50 rounded-lg">
              <p className="text-sm text-gray-600">
                Total additional download: <strong>{calculateTotalSize()} GB</strong>
              </p>
            </div>

            <button
              onClick={completeSetup}
              disabled={downloading}
              className="w-full py-3 bg-primary-600 text-white rounded-lg font-medium hover:bg-primary-700 transition-colors disabled:opacity-50"
            >
              {downloading ? "Setting up..." : "Complete Setup"}
            </button>

            <button
              onClick={completeSetup}
              className="w-full py-2 text-gray-500 hover:text-gray-700"
            >
              Skip optional features
            </button>
          </div>
        )}

        {/* Footer */}
        <div className="mt-8 text-center text-sm text-gray-500">
          <p>100% Offline • No API Keys Required • Your Data Stays Private</p>
        </div>
      </div>
    </div>
  );
}