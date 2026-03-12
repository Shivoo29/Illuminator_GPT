import { useState, useEffect } from "react";
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { ThemeProvider } from "./contexts/ThemeContext";
import { api } from "./utils/api";

import SetupWizard from "./components/SetupWizard";
import Layout from "./components/Layout";
import MainApp from "./components/MainApp";
import TranslationPage from "./pages/TranslationPage";
import ImageGeneratorPage from "./pages/ImageGeneratorPage";
import Settings from "./pages/Settings";

import { Sparkles, Loader2 } from "lucide-react";

function AppContent() {
  const [setupComplete, setSetupComplete] = useState<boolean | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    checkSetupStatus();
  }, []);

  const checkSetupStatus = async () => {
    try {
      const status = await api.get<{ setup_complete: boolean }>("/setup/setup_status");
      setSetupComplete(status.setup_complete);
    } catch (error) {
      console.error("Failed to check setup status:", error);
      setSetupComplete(false);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div
        className="h-screen flex flex-col items-center justify-center gap-4"
        style={{ background: "var(--color-bg-primary)" }}
      >
        <div className="w-14 h-14 rounded-2xl flex items-center justify-center animate-pulse-glow" style={{ background: "linear-gradient(135deg, #6366f1, #8b5cf6)" }}>
          <Sparkles className="w-7 h-7 text-white" />
        </div>
        <div className="flex items-center gap-2" style={{ color: "var(--color-text-secondary)" }}>
          <Loader2 className="w-4 h-4 animate-spin" />
          <span className="text-sm font-medium">Starting Illuminator GPT...</span>
        </div>
      </div>
    );
  }

  if (!setupComplete) {
    return <SetupWizard onComplete={() => setSetupComplete(true)} />;
  }

  return (
    <BrowserRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route path="/" element={<MainApp />} />
          <Route path="/translate" element={<TranslationPage />} />
          <Route path="/image-gen" element={<ImageGeneratorPage />} />
          <Route path="/settings" element={<Settings />} />
        </Route>
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  );
}

export default function App() {
  return (
    <ThemeProvider>
      <AppContent />
    </ThemeProvider>
  );
}