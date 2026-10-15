import type { ReactNode } from "react";

export interface SystemStatus {
  /** inference rate, "-" when unknown */
  fps: string;
  /** average ms per frame, "-" when unknown */
  ms: string;
  /** "5h 22m 18s" of monitoring, null while the camera is stopped */
  uptime: string | null;
  running: boolean;
}

export interface SettingsModel {
  defaultsFields: ReactNode;
  machineField: ReactNode;
  cameraCard: ReactNode;
  videosCard: ReactNode;
  dangerCard: ReactNode;
  /** side detection paths (hazard, fire, smoke, crowd, PPE). Optional,
   *  because callers from before the Detection paths card do not send it. */
  pathsCard?: ReactNode;
  system: SystemStatus;
}

function StatusDot({ on, label }: { on: boolean; label: string }): ReactNode {
  return (
    <span
      className={`status-dot ${on ? "on" : "off"}`}
      role="img"
      aria-label={label}
    />
  );
}

/** Settings in two columns of cards, the same theme as the other pages.
 *  The left column holds the detection defaults, the live system status and
 *  the danger zone; the right column holds the camera, the detection paths,
 *  the uploaded clips and the machine name. All state and saving live in
 *  App - this page only decides where each section sits. */
export default function SettingsPage({ model }: { model: SettingsModel }): ReactNode {
  const live = model.system.running;
  return (
    <div className="settings-page" data-testid="settings-page">
      <div className="settings-grid">
        <div className="settings-col">
          <section className="card defaults-card" data-testid="defaults-card">
            <h2>Detection Defaults</h2>
            {model.defaultsFields}
          </section>
          <section className="card system-card" data-testid="system-card">
            <h2>System Status</h2>
            <dl className="system-status" data-testid="system-status">
              <div className="system-status-row">
                <dt>
                  <StatusDot on={live} label={live ? "live" : "idle"} />
                  Inference FPS
                </dt>
                <dd data-testid="system-fps">{model.system.fps}</dd>
              </div>
              <div className="system-status-row">
                <dt>
                  <StatusDot on={live} label={live ? "live" : "idle"} />
                  Average time per frame
                </dt>
                <dd data-testid="system-ms">{model.system.ms}</dd>
              </div>
              <div className="system-status-row">
                <dt>
                  <StatusDot on={model.system.uptime !== null} label={model.system.uptime !== null ? "live" : "idle"} />
                  Uptime
                </dt>
                <dd data-testid="system-uptime">{model.system.uptime ?? "—"}</dd>
              </div>
              <div className="system-status-row">
                <dt>
                  <StatusDot on={live} label={live ? "running" : "stopped"} />
                  System status
                </dt>
                <dd data-testid="system-state">{live ? "Running" : "Stopped"}</dd>
              </div>
            </dl>
          </section>
          <section className="card danger-card" data-testid="danger-card">
            <h2>Danger</h2>
            {model.dangerCard}
          </section>
        </div>
        <div className="settings-col">
          <section className="card camera-card" data-testid="camera-card">
            <h2>Camera</h2>
            {model.cameraCard}
          </section>
          {model.pathsCard !== undefined && (
            <section className="card paths-card" data-testid="paths-card">
              <h2>Detection Paths</h2>
              {model.pathsCard}
            </section>
          )}
          <section className="card videos-card" data-testid="videos-card">
            <h2>Videos (test inputs)</h2>
            {model.videosCard}
          </section>
          <section className="card machine-card" data-testid="machine-card">
            <h2>Machine</h2>
            {model.machineField}
          </section>
        </div>
      </div>
    </div>
  );
}
