import type { ReactNode } from "react";

export interface DetectionPathRow {
  key: "hazard" | "fire" | "smoke" | "crowd" | "ppe";
  title: string;
  blurb: string;
  enabled: boolean;
  /** analyses per second. Absent when the row has no cadence of its own:
   *  Hazard has none, and Smoke shares Fire's single pass. */
  fps?: number;
  /** extra line under the fps field, e.g. the shared-pass note. */
  fpsCaption?: string;
  /** gate persistence windows behind the plain-words confirm hint
   *  (fire row only). Absent when the backend predates them. */
  gateSeconds?: { fire: number; smoke: number };
}

export interface DetectionPathsModel {
  rows: DetectionPathRow[];
  /** the card is usable only after /config has answered */
  ready: boolean;
  onToggle: (key: DetectionPathRow["key"], enabled: boolean) => void;
  onFps: (key: DetectionPathRow["key"], fps: number) => void;
  /** restore every system to its shipped default (hazard on, side models off
   *  at their operating cadences). Absent on callers from before the button. */
  onReset?: () => void;
}

/** minimum analyses/second the fire/smoke pass needs: below this the gate
 *  needs longer than a short clip to confirm (see below). Lower stored
 *  values are shown as-is with a warning, never silently rewritten. */
const FIRE_FPS_MIN = 0.5;
/** the short-clip reference for the gate warning: uploaded test clips run
 *  about 8 s, so a gate that needs longer may never confirm on one. */
const SHORT_CLIP_SECONDS = 8;

/** seconds of continuous presence the gate needs at this cadence:
 *  ceil(window * fps) consecutive analyses, one every 1/fps seconds. */
function confirmSeconds(windowSeconds: number, fps: number): number {
  if (!(fps > 0)) return Number.POSITIVE_INFINITY;
  return Math.ceil(windowSeconds * fps) / fps;
}

function formatSeconds(value: number): string {
  if (!Number.isFinite(value)) return "never";
  return `${Number.isInteger(value) ? value : value.toFixed(1)} s`;
}

/** the glyph per system, matching the Live Monitor key (box colours live on
 *  the video, so these are plain markers, never verdict colours). */
const ICONS: Record<DetectionPathRow["key"], string> = {
  hazard: "⚠️",
  fire: "🔥",
  smoke: "💨",
  crowd: "👥",
  ppe: "🛡️",
};

/** Settings "Detection paths" card: one row per system with a proper toggle
 *  switch and, where the system has a cadence of its own, a target FPS
 *  field. Fire and Smoke are separate switches on one shared pass, so the
 *  fps field sits on the Fire row and says so. Presentation only — all state
 *  and saving live in App, on the same stored settings the Live Monitor
 *  "Detection systems" panel writes. */
export default function DetectionPathsCard({ model }: { model: DetectionPathsModel }): ReactNode {
  return (
    <>
      {model.rows.map((row) => (
        <div key={row.key} className="path-row" data-testid={`path-row-${row.key}`}>
          <div className="path-row-head">
            <span className="path-icon" aria-hidden="true">
              {ICONS[row.key]}
            </span>
            <span className="path-title">{row.title}</span>
            <label className="switch" title={`${row.title} enabled`}>
              <input
                type="checkbox"
                aria-label={`${row.title} enabled`}
                checked={row.enabled}
                disabled={!model.ready}
                onChange={(event) => model.onToggle(row.key, event.target.checked)}
              />
            </label>
          </div>
          <p className="muted small">{row.blurb}</p>
          {row.fps !== undefined && (
            <label className="field path-fps">
              <span>Target FPS{row.fpsCaption ? ` (${row.fpsCaption})` : ""}</span>
              <input
                type="number"
                min={row.key === "fire" ? FIRE_FPS_MIN : 0.1}
                max={30}
                step={0.5}
                aria-label={`${row.title} target FPS`}
                disabled={!model.ready || !row.enabled}
                value={row.fps}
                onChange={(event) =>
                  model.onFps(
                    row.key,
                    row.key === "fire"
                      // a stored value below the minimum is shown as-is
                      // with a warning, never clamped up on edit
                      ? Math.min(30, Number(event.target.value) || 0)
                      : Math.min(30, Math.max(0.1, Number(event.target.value) || 0))
                  )
                }
              />
              {row.key === "fire" && row.gateSeconds !== undefined && row.fps > 0 && (
                <span className="muted small" data-testid="path-gate-time">
                  {`At ${row.fps} FPS: fire confirms in about ${formatSeconds(
                    confirmSeconds(row.gateSeconds.fire, row.fps)
                  )}, smoke in about ${formatSeconds(
                    confirmSeconds(row.gateSeconds.smoke, row.fps)
                  )}`}
                </span>
              )}
              {row.key === "fire" && row.fps < FIRE_FPS_MIN && (
                <span className="tile-warn" data-testid="path-fps-low-warning">
                  {`Below the ${FIRE_FPS_MIN} FPS minimum — the stored value is kept, but
                  analyses run so rarely the gate may never confirm. Raise it to
                  ${FIRE_FPS_MIN} or more.`}
                </span>
              )}
              {row.key === "fire" &&
                row.gateSeconds !== undefined &&
                row.fps >= FIRE_FPS_MIN &&
                (confirmSeconds(row.gateSeconds.fire, row.fps) > SHORT_CLIP_SECONDS ||
                  confirmSeconds(row.gateSeconds.smoke, row.fps) > SHORT_CLIP_SECONDS) && (
                  <span className="tile-warn" data-testid="path-gate-warning">
                    {`At ${row.fps} FPS the gate needs longer than an ${SHORT_CLIP_SECONDS} s
                    test clip to confirm, so it may never alert on a short clip.`}
                  </span>
                )}
            </label>
          )}
        </div>
      ))}
      <p className="muted small" data-testid="paths-cpu-note">
        Each path adds load on top of person detection (about 40 ms/frame on its own): fire/smoke
        about 3 ms, crowd about 5 ms, PPE about 10 ms per frame on average at these cadences, more
        on the analysed frames themselves. If the ms / frame readout climbs past the frame budget,
        switch a path off or lower its cadence.
      </p>
      <div className="path-reset-row">
        <button
          type="button"
          data-testid="paths-reset"
          disabled={!model.ready}
          title="Hazard zone on; Fire, Smoke, Crowd and PPE off at their default cadences"
          onClick={() => model.onReset?.()}
        >
          Reset to defaults
        </button>
      </div>
    </>
  );
}
