import { useState, useEffect } from "react";
import { Link } from "react-router-dom";
import { api } from "../utils/api.ts";

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

export default function Settings() {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [features, setFeatures] = useState<Record<string, FeatureStatus>>({});
  const [storage, setStorage] = useState<StorageInfo | null>(null);
  const [loading, setLoading] = useState(true);
  const [clearingCache, setClearingCache] = useState(false);

  useEffect(() => {
    loadSettings();
  }, []);

  const loadSettings = async () => {
    try {
      const [modelsData, featuresData, storageData] = await Promise.all([
        api.get<{ installed: ModelInfo[] }>("/setup/available_models"),
        api.get<Record<string, FeatureStatus>>("/setup/feature_status"),
        api.get<StorageInfo>("/system/storage"),
      ]);

      setModels(modelsData.installed || []);
      setFeatures(featuresData);
      setStorage(storageData);
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

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-100 flex items-center justify-center">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600"></div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-100">
      {/* Header */}
      <header className="bg-white shadow-sm">
        <div className="max-w-4xl mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center">
            <Link
              to="/"
              className="text-gray-600 hover:text-gray-800 mr-4"
            >
              ← Back
            </Link>
            <h1 className="text-xl font-bold text-gray-800">Settings</h1>
          </div>
        </div>
      </header>

      <main className="max-w-4xl mx-auto px-4 py-8 space-y-8">
        {/* Language Models */}
        <section className="bg-white rounded-xl shadow-sm p-6">
          <h2 className="text-lg font-semibold text-gray-800 mb-4">
            🤖 Language Models
          </h2>

          <div className="space-y-3">
            {models.length > 0 ? (
              models.map((model) => (
                <div
                  key={model.name}
                  className="flex items-center justify-between p-4 bg-gray-50 rounded-lg"
                >
                  <div>
                    <p className="font-medium text-gray-800">{model.name}</p>
                    <p className="text-sm text-gray-500">
                      {model.size_gb} GB • {model.type.toUpperCase()}
                    </p>
                  </div>
                  <span className="px-3 py-1 bg-green-100 text-green-700 rounded-full text-sm">
                    Active
                  </span>
                </div>
              ))
            ) : (
              <p className="text-gray-500">No models installed</p>
            )}
          </div>

          <button className="mt-4 px-4 py-2 text-primary-600 hover:text-primary-700 font-medium">
            + Download New Model
          </button>
        </section>

        {/* Features */}
        <section className="bg-white rounded-xl shadow-sm p-6">
          <h2 className="text-lg font-semibold text-gray-800 mb-4">
            🎯 Features
          </h2>

          <div className="space-y-3">
            <div className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
              <div>
                <p className="font-medium text-gray-800">🎙️ Podcast Generation</p>
                <p className="text-sm text-gray-500">
                  Generate audio discussions ({features.tts?.size_gb || 0.1} GB)
                </p>
              </div>
              <span
                className={`px-3 py-1 rounded-full text-sm ${
                  features.tts?.installed
                    ? "bg-green-100 text-green-700"
                    : "bg-gray-100 text-gray-600"
                }`}
              >
                {features.tts?.installed ? "Installed" : "Not Installed"}
              </span>
            </div>

            <div className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
              <div>
                <p className="font-medium text-gray-800">🎨 Image Generation</p>
                <p className="text-sm text-gray-500">
                  Create images from text ({features.image_generation?.size_gb || 2.0} GB)
                </p>
              </div>
              <span
                className={`px-3 py-1 rounded-full text-sm ${
                  features.image_generation?.installed
                    ? "bg-green-100 text-green-700"
                    : "bg-gray-100 text-gray-600"
                }`}
              >
                {features.image_generation?.installed ? "Installed" : "Not Installed"}
              </span>
            </div>

            <div className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
              <div>
                <p className="font-medium text-gray-800">🌐 Translation</p>
                <p className="text-sm text-gray-500">
                  Translate between languages ({features.translation?.size_gb || 1.5} GB)
                </p>
              </div>
              <span
                className={`px-3 py-1 rounded-full text-sm ${
                  features.translation?.installed
                    ? "bg-green-100 text-green-700"
                    : "bg-gray-100 text-gray-600"
                }`}
              >
                {features.translation?.installed ? "Installed" : "Not Installed"}
              </span>
            </div>
          </div>
        </section>

        {/* Storage */}
        {storage && (
          <section className="bg-white rounded-xl shadow-sm p-6">
            <h2 className="text-lg font-semibold text-gray-800 mb-4">
              💾 Storage
            </h2>

            <div className="mb-4">
              <div className="flex justify-between text-sm mb-1">
                <span className="text-gray-600">
                  Used: {storage.total_used_gb} GB
                </span>
                <span className="text-gray-600">
                  Free: {storage.disk.free_gb} GB
                </span>
              </div>
              <div className="h-3 bg-gray-200 rounded-full overflow-hidden">
                <div
                  className="h-full bg-primary-600"
                  style={{ width: `${storage.disk.used_percent}%` }}
                />
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4 text-sm">
              <div className="p-3 bg-gray-50 rounded-lg">
                <p className="text-gray-500">Models</p>
                <p className="font-medium">{storage.breakdown.models.gb} GB</p>
              </div>
              <div className="p-3 bg-gray-50 rounded-lg">
                <p className="text-gray-500">Vector Database</p>
                <p className="font-medium">{storage.breakdown.vector_database.gb} GB</p>
              </div>
              <div className="p-3 bg-gray-50 rounded-lg">
                <p className="text-gray-500">Documents</p>
                <p className="font-medium">{storage.breakdown.documents.gb} GB</p>
              </div>
              <div className="p-3 bg-gray-50 rounded-lg">
                <p className="text-gray-500">Cache</p>
                <p className="font-medium">{storage.breakdown.cache.gb} GB</p>
              </div>
            </div>

            <button
              onClick={clearCache}
              disabled={clearingCache}
              className="mt-4 px-4 py-2 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 disabled:opacity-50"
            >
              {clearingCache ? "Clearing..." : "Clear Cache"}
            </button>
          </section>
        )}

        {/* About */}
        <section className="bg-white rounded-xl shadow-sm p-6">
          <h2 className="text-lg font-semibold text-gray-800 mb-4">
            ℹ️ About
          </h2>
          <div className="space-y-2 text-sm text-gray-600">
            <p><strong>Offline RAG Assistant</strong> v1.0.0</p>
            <p>A fully offline, privacy-focused document assistant.</p>
            <p className="pt-2">
              All processing happens locally on your device. No data is sent to external servers.
            </p>
          </div>
        </section>
      </main>
    </div>
  );
}