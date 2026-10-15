import type { ReactNode } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import WallClock from "./WallClock";

export interface SinceLine {
  clock: string;
  hours: number;
  minutes: number;
}

interface LayoutProps {
  usingTestVideo: boolean;
  statusMode: "editing" | "monitoring" | "idle";
  statusText: string;
  sinceLine: SinceLine | null;
  recentAlertCount: number;
  toast: { text: string; tone: "notice" | "error" } | null;
  confirmingSwitch: boolean;
  busy: boolean;
  onDiscardSwitch: () => void;
  onCancelSwitch: () => void;
  children: ReactNode;
}

const NAV: { path: string; label: string; icon: string }[] = [
  { path: "/", label: "Live Monitor", icon: "▣" },
  { path: "/alerts", label: "Alerts", icon: "⚠" },
  { path: "/history", label: "History", icon: "◷" },
  { path: "/settings", label: "Settings", icon: "⚙" },
];

export default function Layout({
  usingTestVideo,
  statusMode,
  statusText,
  sinceLine,
  recentAlertCount,
  toast,
  confirmingSwitch,
  busy,
  onDiscardSwitch,
  onCancelSwitch,
  children,
}: LayoutProps) {
  const location = useLocation();
  const navigate = useNavigate();
  const isActive = (path: string) =>
    path === "/" ? location.pathname === "/" : location.pathname.startsWith(path);

  return (
    <div className="app-shell">
      <nav className="sidebar rail" aria-label="Sections">
        <div className="rail-brand">
          <span className="brand-mark" aria-hidden="true">
            <svg viewBox="0 0 24 24" width="18" height="18" fill="currentColor" aria-hidden="true">
              <path d="M4 21V10l5 3V10l5 3V7l6-3v17H4zm6-4h4v-2H10v2zm0-3h4V9l-4-2.4V14z" />
            </svg>
          </span>
          <span className="rail-brand-text">Trinetra AI</span>
        </div>
        {NAV.map((item) => (
          <button
            key={item.path}
            type="button"
            className={`side-item ${isActive(item.path) ? "active" : ""}`}
            aria-current={isActive(item.path) ? "page" : undefined}
            onClick={() => navigate(item.path)}
          >
            <span className="side-icon" aria-hidden="true">
              {item.icon}
            </span>
            {item.label}
            {item.path === "/alerts" && recentAlertCount > 0 && (
              <span className="side-badge" data-testid="alerts-badge">
                {recentAlertCount > 99 ? "99+" : recentAlertCount}
              </span>
            )}
          </button>
        ))}
      </nav>

      <div className="page">
        <header className="topbar">
          <div className="brand">
            <div className="brand-text">
              <h1>Trinetra AI</h1>
              <p className="brand-sub">Cement Plant Safety</p>
            </div>
            <span className="demo-tag">PILOT / DEMO</span>
            {usingTestVideo && (
              <span className="test-video-tag" data-testid="test-video-tag">
                TEST VIDEO
              </span>
            )}
          </div>
          <div className="topbar-status">
            <WallClock />
            {sinceLine && (
              <span className="uptime" data-testid="uptime">
                Monitoring since {sinceLine.clock}, uptime {sinceLine.hours}h {sinceLine.minutes}m
              </span>
            )}
            <div className={`status-pill ${statusMode}`} data-testid="status-pill">
              {statusText}
            </div>
          </div>
        </header>

        <main className="main">
          {toast && (
            <div className={`inline-note ${toast.tone}`} role="status">
              {toast.text}
            </div>
          )}

          {confirmingSwitch && (
            <div className="confirm-row" role="alertdialog" aria-label="Confirm source switch">
              <span className="muted small">You have unsaved zone changes. Discard them?</span>
              <div className="button-row">
                <button type="button" onClick={onDiscardSwitch} disabled={busy}>
                  Discard
                </button>
                <button type="button" onClick={onCancelSwitch} disabled={busy}>
                  Cancel
                </button>
              </div>
            </div>
          )}

          {children}
        </main>
      </div>
    </div>
  );
}
