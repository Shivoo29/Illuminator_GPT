import { useState } from "react";
import { NavLink, Outlet, useNavigate } from "react-router-dom";
import { useTheme } from "../contexts/ThemeContext";
import {
  MessageSquare,
  Languages,
  ImageIcon,
  Settings,
  PanelLeftClose,
  PanelLeft,
  Sun,
  Moon,
  Sparkles,
  Zap,
} from "lucide-react";

const navItems = [
  { to: "/", icon: MessageSquare, label: "Chat" },
  { to: "/translate", icon: Languages, label: "Translate" },
  { to: "/image-gen", icon: ImageIcon, label: "Image Gen" },
  { to: "/settings", icon: Settings, label: "Settings" },
];

export default function Layout() {
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const { theme, toggleTheme } = useTheme();
  const navigate = useNavigate();

  return (
    <div className="flex h-screen overflow-hidden" style={{ background: "var(--color-bg-primary)" }}>
      {/* Sidebar */}
      <aside
        className="flex flex-col transition-all duration-300 ease-in-out border-r shrink-0"
        style={{
          width: sidebarOpen ? "260px" : "68px",
          background: "var(--color-sidebar-bg)",
          borderColor: "var(--color-sidebar-border)",
        }}
      >
        {/* Logo */}
        <div
          className="flex items-center gap-3 px-4 h-16 border-b shrink-0 cursor-pointer"
          style={{ borderColor: "var(--color-border-primary)" }}
          onClick={() => navigate("/")}
        >
          <div className="w-9 h-9 rounded-xl flex items-center justify-center shrink-0" style={{ background: "linear-gradient(135deg, #6366f1, #8b5cf6)" }}>
            <Sparkles className="w-5 h-5 text-white" />
          </div>
          {sidebarOpen && (
            <div className="animate-fade-in overflow-hidden">
              <h1 className="text-sm font-bold tracking-tight whitespace-nowrap" style={{ color: "var(--color-text-primary)" }}>
                Illuminator GPT
              </h1>
              <p className="text-[10px] font-medium flex items-center gap-1 whitespace-nowrap" style={{ color: "var(--color-text-muted)" }}>
                <Zap className="w-3 h-3" style={{ color: "var(--color-success)" }} />
                100% Offline
              </p>
            </div>
          )}
        </div>

        {/* Navigation */}
        <nav className="flex-1 flex flex-col gap-1 p-3 overflow-y-auto">
          {navItems.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.to === "/"}
              className={({ isActive }) =>
                `flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm font-medium transition-all duration-200 group ${
                  isActive ? "animate-fade-in" : ""
                }`
              }
              style={({ isActive }) => ({
                background: isActive ? "var(--color-accent-bg)" : "transparent",
                color: isActive ? "var(--color-accent-primary)" : "var(--color-text-secondary)",
              })}
              onMouseEnter={(e) => {
                const el = e.currentTarget;
                if (!el.classList.contains("active")) {
                  el.style.background = "var(--color-bg-hover)";
                }
              }}
              onMouseLeave={(e) => {
                const el = e.currentTarget;
                // NavLink re-renders on route change, so check aria-current
                if (!el.getAttribute("aria-current")) {
                  el.style.background = "transparent";
                }
              }}
            >
              <item.icon className="w-5 h-5 shrink-0" />
              {sidebarOpen && <span className="truncate">{item.label}</span>}
            </NavLink>
          ))}
        </nav>

        {/* Bottom actions */}
        <div className="p-3 border-t space-y-1" style={{ borderColor: "var(--color-border-primary)" }}>
          {/* Theme toggle */}
          <button
            onClick={toggleTheme}
            className="flex items-center gap-3 w-full px-3 py-2.5 rounded-xl text-sm font-medium transition-all duration-200"
            style={{ color: "var(--color-text-secondary)" }}
            onMouseEnter={(e) => (e.currentTarget.style.background = "var(--color-bg-hover)")}
            onMouseLeave={(e) => (e.currentTarget.style.background = "transparent")}
          >
            {theme === "dark" ? (
              <Sun className="w-5 h-5 shrink-0" />
            ) : (
              <Moon className="w-5 h-5 shrink-0" />
            )}
            {sidebarOpen && <span>{theme === "dark" ? "Light Mode" : "Dark Mode"}</span>}
          </button>

          {/* Collapse toggle */}
          <button
            onClick={() => setSidebarOpen(!sidebarOpen)}
            className="flex items-center gap-3 w-full px-3 py-2.5 rounded-xl text-sm font-medium transition-all duration-200"
            style={{ color: "var(--color-text-tertiary)" }}
            onMouseEnter={(e) => (e.currentTarget.style.background = "var(--color-bg-hover)")}
            onMouseLeave={(e) => (e.currentTarget.style.background = "transparent")}
          >
            {sidebarOpen ? (
              <>
                <PanelLeftClose className="w-5 h-5 shrink-0" />
                <span>Collapse</span>
              </>
            ) : (
              <PanelLeft className="w-5 h-5 shrink-0" />
            )}
          </button>
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 flex flex-col overflow-hidden" style={{ background: "var(--color-bg-primary)" }}>
        <Outlet />
      </main>
    </div>
  );
}
