import { useEffect, useRef, useState } from "react";
import type { ReactNode, RefObject } from "react";
import { alertSnapshotUrl, fireAlertSnapshotUrl } from "../api";
import EvidenceModal from "../components/EvidenceModal";
import { LEGEND, MIN_POINTS, TOOLS } from "../components/monitorMeta";
import VideoSourcePicker from "../components/VideoSourcePicker";
import type { VideoSourcePickerModel } from "../components/VideoSourcePicker";
import type { AlertEvent, DrawTool, FireAlert, Point } from "../types";
import { alertLabel, alertTime, alertWho, badgeText, badgeTone, clockTime } from "../utils/format";

export type ScreenMode = "stopped" | "edit" | "live";
export type RubberBand =
  | { kind: "rectangle"; x: number; y: number; width: number; height: number }
  | { kind: "circle"; cx: number; cy: number; rx: number; ry: number }
  | null;

export interface ZoneTag {
  text: string;
  x: number;
  y: number;
  width: number;
  height: number;
}

/** one legend row; the page falls back to LEGEND when the model is older. */
export interface LegendItem {
  label: string;
  tone: string;
}

export interface ZoneCard {
  name: string;
  dot: string;
  countText: string;
  showFigures: boolean;
  peopleText: string;
  peopleOver: boolean;
  dwellText: string;
  dwellOver: boolean;
  open: boolean;
  corners: number;
  usingDefaults: boolean;
  enabled: boolean;
  rules: ReactNode;
}

export interface TileFigures {
  peopleLabel: string;
  people: number;
  peopleLimit: number | null;
  peopleTotal: boolean;
  peopleTone: "idle" | "occupy" | "over";
  dwellLabel: string;
  dwell: number;
  dwellLimit: number | null;
  dwellTotal: boolean;
  dwellTone: "idle" | "occupy" | "over";
}

export interface BannerWorker {
  id: number | null;
  zone: string;
  dwell: number;
}

/** somebody inside but still within the limit: the orange banner names them
 *  with "Xs of Ys" against their zone's own dwell limit. */
export interface BannerInside {
  id: number | null;
  zone: string;
  dwell: number;
  limit: number;
}

/** one zone's overlay tone, read off its card dot: red only while alerting. */
function overlayTone(dot: string): string {
  if (dot === "danger") return "alert";
  if (dot === "warn") return "occupy";
  return "";
}

/** one Recent-card row: a zone alert or a fire/smoke alert, newest first. */
export interface RecentItem {
  key: string;
  /** epoch ms for the newest-first merge; 0 when the row carries no time. */
  time: number;
  zone?: AlertEvent;
  fire?: FireAlert;
}

/** Merge zone and fire rows newest-first, capped to the small card list.
 *  Both inputs arrive newest-first; the sort is by wall time, so a fire row
 *  reported between two zone rows sits between them. */
export function mergeRecent(
  zones: AlertEvent[],
  fires: FireAlert[],
  limit: number
): RecentItem[] {
  const items: RecentItem[] = [
    ...zones.map((event) => ({
      key: `zone-${event.id}`,
      time: Date.parse(event.alert_at ?? event.entered_at ?? "") || 0,
      zone: event,
    })),
    ...fires.map((alert) => ({
      key: `fire-${alert.id}`,
      time: Date.parse(alert.reported_at ?? "") || 0,
      fire: alert,
    })),
  ];
  items.sort((a, b) => b.time - a.time);
  return items.slice(0, Math.max(0, limit));
}

function fireBadgeTone(alert: FireAlert): string {
  return alert.alert_type === "fire" ? "danger" : "warn";
}

/** card-compact fire line: severity, capture point and wall clock. */
function fireCardMeta(alert: FireAlert): string {
  const parts = [alert.severity];
  if (alert.captured_at !== null && alert.captured_at !== undefined) {
    parts.push(`video ${alert.captured_at.toFixed(1)}s`);
  }
  const clock = clockTime(alert.reported_at);
  if (clock) parts.push(clock);
  return parts.join(" · ");
}

/** modal meta line, identical to the Alerts page fire modal. */
function fireModalMeta(alert: FireAlert): string {
  const parts = [
    `${alert.alert_type.toUpperCase()} · ${alert.severity}`,
    `conf ${alert.confidence.toFixed(2)}`,
  ];
  if (alert.captured_at !== null && alert.captured_at !== undefined) {
    parts.push(`video ${alert.captured_at.toFixed(1)}s`);
  }
  const clock = clockTime(alert.reported_at);
  if (clock) parts.push(clock);
  return parts.join(" · ");
}

export interface LiveMonitorModel {
  machineName: string;
  screenMode: ScreenMode;
  running: boolean;  /** "Webcam 0" or the test video name: what Start will open. Picked in Settings. */
  stoppedSourceLabel: string;
  /** Live Monitor source picker (dropdown + upload + running confirm).
   *  Optional, because older model objects do not send it. */
  sourcePicker?: VideoSourcePickerModel | null;
  busy: boolean;
  startZoneBlock: string;
  drawing: boolean;
  /** the editor can open: live with a frame, or stopped with a saved size */
  canEdit: boolean;
  /** the video column shows a panel: the stream, or the dark edit canvas */
  stageVisible: boolean;
  tool: DrawTool | null;
  frameKnown: boolean;
  frameWidth: number;
  frameHeight: number;
  /** frame pixels per screen pixel: handles keep their on-screen size */
  unit: number;
  selectedShape: number | null;
  pictureUrl: string;
  imageRef: RefObject<HTMLElement>;
  /** reconnect a broken MJPEG stream; optional so older model objects still render */
  onStreamError?: () => void;
  /** why the saved source cannot be opened while stopped (a missing test
   *  video), shown in the stopped panel instead of a blank camera. */
  stoppedHint?: string | null;
  /** a single-play clip ended by itself (and the panel is still on that
   *  source): the stopped panel says "Clip finished" instead of "Camera
   *  stopped". Neutral, and Start works again. Optional, because older model
   *  objects do not send it. */
  finishedClip?: boolean;
  overlayVisible: boolean;
  shownZones: Point[][];
  rubberBand: RubberBand;
  polygonPreview: Point[] | null;
  /** the freehand stroke mid-drag, painted live in the preview colour.
   *  Optional, because older model objects do not send it. */
  strokePreview?: Point[] | null;
  draftPoints: Point[] | null;
  tags: (ZoneTag | null)[];
  tagFontSize: number;
  tagPad: number;
  handles: { point: Point; invalid: boolean; active: boolean }[] | null;
  handleZone: number | null;
  midpoints: { index: number; point: Point }[];
  cursorHint: "pick-tool" | "waiting-frame" | null;
  alert: boolean;
  bannerWorkers: BannerWorker[];
  bannerInside: BannerInside[];
  bannerMessage: string;
  muted: boolean;
  tiles: TileFigures;
  /** annotation-only side paths readout (crowd numbers, PPE grade), or null
   *  when every side path is off. Optional, because older model objects do
   *  not send it. Rendered under the zone tiles, never as an alert. */
  sidePaths?: ReactNode;
  /** per-zone overlay shapes. Falls back to shownZones when absent, so older
   *  model objects draw exactly as before. */
  overlayZones?: Point[][];
  /** legend rows for the ticked systems. Falls back to LEGEND when absent. */
  legendItems?: LegendItem[];
  /** Detection systems panel node (Live Monitor right column). */
  detectionSystems?: ReactNode;
  /** hazard-zone system switch, off. Optional, because older model objects
   *  do not send it. Renders a neutral banner instead of the green SAFE one:
   *  green must not claim a clear zone when nothing is being evaluated. */
  hazardOff?: boolean;
  /** gate progress while fire/smoke is being confirmed ("Confirming possible
   *  smoke — 2 of 5 samples"), or null when nothing is confirming. Rendered
   *  in neutral blue, never in an alert tone. */
  confirming?: string | null;
  anchorHint: { count: number; suggestBodyCenter: boolean } | null;
  legendVisible: boolean;
  zoneCards: ZoneCard[];
  renaming: number | null;
  renameDraft: string;
  confirmingCancel: boolean;
  unsavedGeometry: boolean;
  zoneValid: boolean;
  coordRows: { x: number; y: number; invalid: boolean }[] | null;
  coordTitle: string;
  coordZone: number | null;
  canAddPoint: boolean;
  recentAlerts: AlertEvent[];
  /** unacknowledged fire/smoke rows for the Recent card, newest first.
   *  Optional, because older model objects do not send them. */
  recentFireAlerts?: FireAlert[];
  /** merged-list cap (RECENT_ALERT_COUNT in App). Optional; the card falls
   *  back to 5, the same small list as before. */
  recentLimit?: number;
  /** acknowledge one fire/smoke row from the card, through the fire route. */
  onAcknowledgeFire?: (id: number) => void;
  /** open the Evidence modal for one fire/smoke row. */
  onOpenFireEvidence?: (alert: FireAlert) => void;
  /** the open fire evidence with the Alerts-page overrides. All optional,
   *  like the rest of the fire block: without them no modal opens. */
  fireEvidence?: FireAlert | null;
  fireEvidenceDetails?: { label: string; value: string }[] | null;
  onAcknowledgeFireEvidence?: (() => void) | null;
  onCloseFireEvidence?: () => void;
  onFireEvidenceStep?: (direction: 1 | -1) => void;
  fireEvidenceHasPrev?: boolean;
  fireEvidenceHasNext?: boolean;
  onFireFalseAlarm?: (id: number) => void;
  fireFalseAlarming?: number | null;
  perfFps: string;
  historyEmpty: boolean;
  draftEmpty: boolean;
  clearDisabled: boolean;
  deleteDisabled: boolean;
  // callbacks
  onStart: () => void;
  onStop: () => void;
  onEnterEdit: () => void;
  onSave: () => void;
  onCancelEdit: () => void;
  onConfirmDiscard: () => void;
  onKeepEditing: () => void;
  onPickTool: (tool: DrawTool) => void;
  onToolbarUndo: () => void;
  onToolbarDelete: () => void;
  onClearZones: () => void;
  onPictureDown: (event: React.MouseEvent) => void;
  onPictureClick: (event: React.MouseEvent) => void;
  /** double-click closes an open Polygon draft. Optional, so older model
   *  objects still render. */
  onPictureDoubleClick?: (event: React.MouseEvent) => void;
  /** Finish button for the Polygon tool. Optional, like the double-click. */
  onFinishPolygon?: () => void;
  onPictureHover: (event: React.MouseEvent) => void;
  onPictureLeave: () => void;
  onHandleDown: (zone: number, index: number) => (event: React.MouseEvent) => void;
  onEdgeClick: (zone: number, edge: number) => (event: React.MouseEvent) => void;
  onHandleDoubleClick: (zone: number, index: number) => void;
  onEditCoordinate: (zone: number, index: number, axis: 0 | 1, raw: string) => void;
  onRemoveCoordinate: (zone: number, index: number) => void;
  onAddPoint: () => void;
  onOpenZoneCard: (index: number) => void;
  onRenameStart: (index: number) => void;
  onRenameDraft: (text: string) => void;
  onCommitRename: () => void;
  onRenameCancel: () => void;
  onToggleZoneEnabled: (index: number, enabled: boolean) => void;
  onDeleteZone: (index: number) => void;
  onAddZone: () => void;
  onToggleMute: () => void;
  onAcknowledge: (id: number) => void;
  onViewAllAlerts: () => void;
}

function bannerText(
  alert: boolean,
  workers: BannerWorker[],
  fallback: string,
  dangerBanner: string
): string {
  if (!alert) return fallback || "Nobody in a danger zone";
  const first = workers[0];
  if (first) {
    const who = first.id !== null ? `Worker #${first.id}` : "A worker";
    const where = first.zone ? ` in ${first.zone}` : "";
    return `${who}${where} (${first.dwell.toFixed(1)}s)`;
  }
  return fallback || dangerBanner;
}

/** orange banner line: who is inside, where, and how far into their limit. */
function occupyText(first: BannerInside): string {
  const who = first.id !== null ? `Worker #${first.id}` : "A worker";
  const where = first.zone ? ` in ${first.zone}` : "";
  return `${who}${where} (${first.dwell.toFixed(1)}s of ${first.limit.toFixed(1)}s)`;
}

export default function LiveMonitorPage({ model: m }: { model: LiveMonitorModel }) {
  const selectedTone =
    m.handleZone !== null ? overlayTone(m.zoneCards[m.handleZone]?.dot ?? "") : "";
  /** Fullscreen lives here, not in App: no settings, no backend, just the
   *  browser Fullscreen API on the stage element. Coordinates stay correct
   *  because every pointer position is mapped through the live bounding
   *  rect. Esc exits natively; the listener below resyncs the icon. */
  const stageWrapRef = useRef<HTMLDivElement | null>(null);
  const [isFullscreen, setIsFullscreen] = useState(false);
  useEffect(() => {
    const onChange = () => setIsFullscreen(document.fullscreenElement != null);
    document.addEventListener("fullscreenchange", onChange);
    return () => document.removeEventListener("fullscreenchange", onChange);
  }, []);
  const toggleFullscreen = () => {
    if (document.fullscreenElement != null) {
      void document.exitFullscreen?.().catch(() => {});
      return;
    }
    try {
      const request = stageWrapRef.current?.requestFullscreen?.bind(stageWrapRef.current);
      const result = request?.() as Promise<void> | undefined;
      if (result && typeof result.catch === "function") result.catch(() => {});
    } catch {
      /* older browsers without the API: the button simply does nothing */
    }
  };
  return (
    <div className="layout" data-testid="live-page">
      <section className="video-col">
        <div className="camera-card-head" data-testid="camera-head">
          <span className="camera-icon" aria-hidden="true">
            <svg viewBox="0 0 24 24" width="18" height="18" fill="currentColor" aria-hidden="true">
              <path d="M4 8h3l2-2h6l2 2h3v11H4V8zm8 2.5A3.5 3.5 0 1 0 12 17a3.5 3.5 0 0 0 0-6.5z" />
            </svg>
          </span>
          <span className="camera-name">CAMERA 01</span>
          <span className="machine-name" data-testid="machine-name">
            {m.machineName || "Machine"}
          </span>
          {m.running && (
            <span className="live-chip" data-testid="live-chip">
              Live
            </span>
          )}
          {m.sourcePicker && <VideoSourcePicker model={m.sourcePicker} />}
        </div>

        {m.screenMode === "stopped" && (
          <div className="stopped-wrap" data-testid="start-panel">
            <div className="stage stopped-stage">
              <span className="stopped-text">
                {m.finishedClip === true ? "Clip finished" : "Camera stopped"}
              </span>
            </div>
            <div className="stopped-actions">
              <button
                type="button"
                className="primary start-button"
                onClick={m.onStart}
                disabled={m.busy || m.startZoneBlock !== ""}
              >
                Start camera
              </button>
              <button
                type="button"
                className="edit-stopped-button"
                onClick={m.onEnterEdit}
                disabled={m.busy || !m.canEdit}
                title={
                  m.canEdit
                    ? "Draw zones before starting"
                    : "Start the camera once so the editor knows the picture size"
                }
              >
                Edit zones
              </button>
              {m.startZoneBlock && (
                <p className="primary-hint" role="status">
                  {m.startZoneBlock}
                </p>
              )}
              {m.stoppedHint && (
                <p className="primary-hint" role="status" data-testid="stopped-hint">
                  {m.stoppedHint}
                </p>
              )}
            </div>
          </div>
        )}

        {m.drawing && (
          <div className="toolbar" role="toolbar" aria-label="Zone drawing tools" data-testid="toolbar">
            <div className="tool-group">
              {TOOLS.map((item) => (
                <button
                  key={item.id}
                  type="button"
                  className={`tool ${m.tool === item.id ? "active" : ""}`}
                  aria-pressed={m.tool === item.id}
                  aria-label={item.label}
                  onClick={() => m.onPickTool(item.id)}
                >
                  <span className="tool-icon" aria-hidden="true">
                    {item.icon}
                  </span>
                  <span className="tool-label">{item.label}</span>
                </button>
              ))}
            </div>

            <div className="tool-group">
              <button
                type="button"
                className="tool"
                aria-label="Undo"
                disabled={m.historyEmpty && m.draftEmpty}
                onClick={m.onToolbarUndo}
              >
                <span className="tool-icon" aria-hidden="true">
                  ↺
                </span>
                <span className="tool-label">Undo</span>
              </button>
              <button
                type="button"
                className="tool"
                aria-label="Delete"
                disabled={m.deleteDisabled}
                onClick={m.onToolbarDelete}
              >
                <span className="tool-icon" aria-hidden="true">
                  ⌫
                </span>
                <span className="tool-label">Delete</span>
              </button>
              <button
                type="button"
                className="tool"
                aria-label="Clear"
                disabled={m.clearDisabled}
                onClick={m.onClearZones}
              >
                <span className="tool-icon" aria-hidden="true">
                  🗑
                </span>
                <span className="tool-label">Clear</span>
              </button>
            </div>
          </div>
        )}

        {m.drawing && m.tool === "quad" && (
          <p className="quad-hint" data-testid="quad-hint">
            Click 4 corners around the area.
          </p>
        )}

        {m.drawing && m.tool === "polygon" && (
          <div className="polygon-hint" data-testid="polygon-hint">
            <p>
              Click to add corners (3–32). Click near the first point, double-click,
              or press Enter to finish.
            </p>
            <button
              type="button"
              aria-label="Finish zone"
              disabled={m.draftEmpty}
              onClick={() => m.onFinishPolygon?.()}
            >
              Finish zone
            </button>
          </div>
        )}

        {m.stageVisible && (
        <div
          ref={stageWrapRef}
          className={`stage ${m.drawing ? "drawing" : ""} ${m.tool ? `tool-${m.tool}` : ""}`}
          style={
            m.frameKnown ? { aspectRatio: `${m.frameWidth} / ${m.frameHeight}` } : undefined
          }
        >
          {m.running ? (
            <img
              ref={m.imageRef as unknown as RefObject<HTMLImageElement>}
              key={m.pictureUrl}
              className={`video ${m.drawing ? "editable" : "readonly"}`}
              src={m.pictureUrl}
              alt="camera stream"
              onClick={m.onPictureClick}
              onDoubleClick={m.onPictureDoubleClick}
              onMouseDown={m.onPictureDown}
              onMouseMove={m.onPictureHover}
              onMouseLeave={m.onPictureLeave}
              onError={m.onStreamError}
              draggable={false}
            />
          ) : (
            <div
              ref={m.imageRef as unknown as RefObject<HTMLDivElement>}
              className={`video blank-canvas ${m.drawing ? "editable" : ""}`}
              data-testid="blank-canvas"
              onClick={m.onPictureClick}
              onDoubleClick={m.onPictureDoubleClick}
              onMouseDown={m.onPictureDown}
              onMouseMove={m.onPictureHover}
              onMouseLeave={m.onPictureLeave}
            >
              <span className="blank-canvas-note">
                No video - drawing on the saved {m.frameWidth} × {m.frameHeight} layout
              </span>
            </div>
          )}

          {m.running && m.frameKnown && (
            <>
              <span className="live-badge" data-testid="live-badge">
                <span className="live-dot" aria-hidden="true" />
                LIVE
              </span>
              <span className="fps-badge" data-testid="fps-badge">
                {m.perfFps} FPS
              </span>
              <span
                className={`status-badge ${m.hazardOff ? "off" : m.alert ? "alert" : m.bannerInside.length > 0 ? "occupy" : "safe"}`}
                data-testid="status-badge"
              >
                {m.hazardOff ? "OFF" : m.alert ? "ALERT" : m.bannerInside.length > 0 ? "INSIDE" : "SAFE"}
              </span>
              {m.confirming !== null && m.confirming !== undefined && m.confirming !== "" && (
                <span className="confirming-badge" data-testid="confirming">
                  {m.confirming}
                </span>
              )}
            </>
          )}

          <button
            type="button"
            className="fullscreen-btn"
            data-testid="fullscreen-btn"
            aria-label={isFullscreen ? "Exit full screen" : "Full screen"}
            title={isFullscreen ? "Exit full screen (Esc)" : "Full screen"}
            onClick={toggleFullscreen}
          >
            {isFullscreen ? "⤡" : "⤢"}
          </button>

          {m.overlayVisible && (
            <svg
              className={`overlay ${m.drawing ? "" : "locked"}`}
              data-testid="zone-stage"
              viewBox={`0 0 ${m.frameWidth} ${m.frameHeight}`}
              preserveAspectRatio="none"
              aria-label="Danger zone video overlay"
            >
              {(m.overlayZones ?? m.shownZones).map((zone, zoneIndex) =>
                zone.length >= MIN_POINTS ? (
                  <polygon
                    key={`zone-${zoneIndex}`}
                    className={`zone ${overlayTone(m.zoneCards[zoneIndex]?.dot ?? "")} ${
                      m.selectedShape === zoneIndex ? "selected" : ""
                    }`}
                    data-testid={`zone-shape-${zoneIndex}`}
                    points={zone.map((point) => point.join(",")).join(" ")}
                  />
                ) : null
              )}

              {m.tags.map((tag, zoneIndex) =>
                tag ? (
                    <g
                      key={`tag-${zoneIndex}`}
                      transform={`translate(${tag.x} ${tag.y})`}
                      className={`zone-tag ${overlayTone(m.zoneCards[zoneIndex]?.dot ?? "")}`}
                      data-testid={`zone-tag-${zoneIndex}`}
                    >
                    <rect
                      className="zone-tag-bg"
                      x={0}
                      y={0}
                      width={tag.width}
                      height={tag.height}
                      rx={4 * m.unit}
                      ry={4 * m.unit}
                    />
                    <text className="zone-tag-text" x={m.tagPad} y={m.tagFontSize} fontSize={m.tagFontSize}>
                      {tag.text}
                    </text>
                  </g>
                ) : null
              )}

              {m.rubberBand?.kind === "rectangle" && (
                <rect
                  className="rubber-band"
                  x={m.rubberBand.x}
                  y={m.rubberBand.y}
                  width={m.rubberBand.width}
                  height={m.rubberBand.height}
                />
              )}
              {m.rubberBand?.kind === "circle" && (
                <ellipse
                  className="rubber-band"
                  cx={m.rubberBand.cx}
                  cy={m.rubberBand.cy}
                  rx={m.rubberBand.rx}
                  ry={m.rubberBand.ry}
                />
              )}

              {m.polygonPreview && (
                <polyline
                  className="zone-preview"
                  points={m.polygonPreview.map((point) => point.join(",")).join(" ")}
                />
              )}

              {m.strokePreview && m.strokePreview.length > 1 && (
                <polyline
                  className="zone-preview"
                  data-testid="stroke-preview"
                  points={m.strokePreview.map((point) => point.join(",")).join(" ")}
                />
              )}

              {m.draftPoints?.map((point, index) => (
                <g key={`draft-${index}`} className="draft-point-group">
                  <circle
                    className={`draft-point ${index === 0 ? "first" : ""}`}
                    data-testid={index === 0 ? "draft-first-point" : `draft-point-${index}`}
                    cx={point[0]}
                    cy={point[1]}
                    r={(index === 0 ? 9 : 6) * m.unit}
                  />
                </g>
              ))}

              {m.handleZone !== null &&
                m.midpoints.map(({ index, point }) => (
                  <g
                    key={`edge-${index}`}
                    className="midpoint"
                    aria-label={`Add point on edge ${index + 1}`}
                    onClick={m.onEdgeClick(m.handleZone as number, index)}
                    onMouseDown={(event) => event.stopPropagation()}
                  >
                    <circle className="midpoint-hit" cx={point[0]} cy={point[1]} r={20 * m.unit} />
                    <circle className="midpoint-dot" cx={point[0]} cy={point[1]} r={7 * m.unit} />
                    <text
                      className="midpoint-plus"
                      x={point[0]}
                      y={point[1] + 5 * m.unit}
                      fontSize={16 * m.unit}
                    >
                      +
                    </text>
                  </g>
                ))}

              {m.handles &&
                m.handleZone !== null &&
                m.handles.map((handle, index) => (
                  <g
                    key={`point-${m.handleZone}-${index}`}
                    className="handle-group"
                    onMouseDown={m.onHandleDown(m.handleZone as number, index)}
                    onDoubleClick={() => m.onHandleDoubleClick(m.handleZone as number, index)}
                  >
                    <circle
                      className="handle-hit"
                      cx={handle.point[0]}
                      cy={handle.point[1]}
                      r={20 * m.unit}
                    />
                    <circle
                      className={`handle ${selectedTone} ${handle.invalid ? "invalid" : ""} ${
                        handle.active ? "active" : ""
                      }`}
                      cx={handle.point[0]}
                      cy={handle.point[1]}
                      r={(handle.active ? 13 : 12) * m.unit}
                    />
                    <title>{`Point ${index + 1}: ${handle.point[0]}, ${handle.point[1]} - drag to move, double click to remove`}</title>
                  </g>
                ))}
            </svg>
          )}

          {m.cursorHint === "pick-tool" && (
            <div className="hint">Pick a tool above, then press and drag on the video</div>
          )}

          {m.cursorHint === "waiting-frame" && <div className="hint">Waiting for the first video frame</div>}
        </div>
        )}

        {/* The verdict belongs under the picture it is a verdict about. Red is
            an active alert, orange is somebody inside but still within the
            limit, green is nobody inside. The banner renders only when it has
            something to say: a plain SAFE feed shows the overlay chip alone,
            so the verdict word never appears twice at the same time. */}
        {(m.screenMode === "live" || m.alert) &&
          (m.hazardOff ? (
            <div className="state-banner idle" data-testid="state-banner">
              <span className="state-word">OFF</span>
              <span className="state-message">Hazard monitoring off</span>
            </div>
          ) : m.alert ? (
            <div className="state-banner alert" data-testid="state-banner">
              <span className="state-word">ALERT</span>
              <span className="state-message">
                {bannerText(m.alert, m.bannerWorkers, m.bannerMessage, "DANGER - Person in danger zone")}
              </span>
              <button
                type="button"
                className="mute-toggle"
                aria-label={m.muted ? "Unmute alert sound" : "Mute alert sound"}
                aria-pressed={m.muted}
                onClick={m.onToggleMute}
              >
                {m.muted ? "🔇" : "🔊"}
              </button>
            </div>
          ) : m.bannerInside.length > 0 ? (
            <div className="state-banner occupy" data-testid="state-banner">
              <span className="state-word">INSIDE</span>
              <span className="state-message">{occupyText(m.bannerInside[0])}</span>
            </div>
          ) : null)}

        {m.screenMode === "edit" && (
          <div className="state-banner editing" data-testid="edit-banner">
            <span className="state-word">EDITING</span>
            <span className="state-message">Changes apply after Save.</span>
          </div>
        )}

        {m.running && (
        <div className="tiles" data-testid="status-tiles">
            <div className="tile">
              <span className="tile-label" data-testid="tile-inside-label">
                {m.tiles.peopleLabel}
              </span>
                <span
                  className={`tile-value ${m.tiles.peopleTone === "idle" ? "" : m.tiles.peopleTone}`}
                  data-testid="tile-inside"
                >
                {m.tiles.people}
                {m.tiles.peopleLimit === null ? (
                  <span className="tile-of"> in total</span>
                ) : (
                  <span className="tile-of"> / {m.tiles.peopleLimit}</span>
                )}
              </span>
            </div>
            <div className="tile">
              <span className="tile-label" data-testid="tile-dwell-label">
                {m.tiles.dwellLabel}
              </span>
                <span
                  className={`tile-value ${m.tiles.dwellTone === "idle" ? "" : m.tiles.dwellTone}`}
                  data-testid="tile-dwell"
                >
                {m.tiles.dwell.toFixed(1)}s
                {m.tiles.dwellLimit === null ? (
                  <span className="tile-of"> in total</span>
                ) : (
                  <span className="tile-of"> / {m.tiles.dwellLimit.toFixed(1)}s</span>
                )}
              </span>
            </div>
          </div>
        )}

        {m.anchorHint && (
          <p className="zone-warning anchor-hint" role="status" data-testid="anchor-hint">
            <span className="zone-warning-mark" aria-hidden="true">
              !
            </span>
            {m.anchorHint.count === 1
              ? "One person's anchor point is outside the frame"
              : `${m.anchorHint.count} people's anchor points are outside the frame`}
            {", so they cannot be counted as inside a zone."}
            {m.anchorHint.suggestBodyCenter ? " Try Body center instead." : ""}
          </p>
        )}

        {m.legendVisible && (
          <ul className="legend" aria-label="Video key" data-testid="legend">
            {(m.legendItems ?? LEGEND).map((item) => (
              <li key={item.label} className={`legend-item ${item.tone}`}>
                <span className="legend-swatch" aria-hidden="true" />
                {item.label}
              </li>
            ))}
          </ul>
        )}

        {m.screenMode === "live" && (
          <div className="card-primary">
            <button type="button" className="primary" onClick={m.onEnterEdit} disabled={m.busy || !m.running}>
              Edit zones
            </button>
            <button type="button" className="stop-button" onClick={m.onStop} disabled={m.busy}>
              Stop camera
            </button>
          </div>
        )}

        {m.screenMode === "edit" && (
          <div className="card-primary" data-testid="edit-actions">
            <button
              type="button"
              className="primary"
              onClick={m.onSave}
              disabled={m.busy || !m.frameKnown || !m.zoneValid}
            >
              Save
            </button>
            {m.confirmingCancel ? (
              <div className="confirm-row" role="alertdialog" aria-label="Confirm cancel">
                <span className="muted small">
                  Discard {m.unsavedGeometry ? "your changes to the zones" : "editing"} and go back to
                  monitoring? The zones now in use stay in use.
                </span>
                <div className="button-row">
                  <button type="button" onClick={m.onConfirmDiscard} disabled={m.busy}>
                    Discard changes
                  </button>
                  <button type="button" onClick={m.onKeepEditing} disabled={m.busy}>
                    Keep editing
                  </button>
                </div>
              </div>
            ) : (
              <button type="button" className="stop-button" onClick={m.onCancelEdit} disabled={m.busy}>
                Cancel
              </button>
            )}
            {!m.zoneValid && (
              <p className="primary-hint" role="status">
                {!m.frameKnown ? "Waiting for the first video frame" : "Draw a valid zone first"}
              </p>
            )}
          </div>
        )}

        {m.screenMode === "edit" && m.coordRows && m.coordZone !== null && (
          <details className="coord-details" data-testid="coord-details">
            <summary>{m.coordTitle} coordinates</summary>
            <table className="coord-table">
              <thead>
                <tr>
                  <th scope="col">#</th>
                  <th scope="col">x</th>
                  <th scope="col">y</th>
                  <th scope="col" aria-label="Remove" />
                </tr>
              </thead>
              <tbody>
                {m.coordRows.map((point, index) => (
                  <tr key={index} className={point.invalid ? "invalid" : ""}>
                    <td className="muted small">{index + 1}</td>
                    <td>
                      <input
                        type="number"
                        aria-label={`Point ${index + 1} x`}
                        value={point.x}
                        onChange={(e) =>
                          m.onEditCoordinate(m.coordZone as number, index, 0, e.target.value)
                        }
                      />
                    </td>
                    <td>
                      <input
                        type="number"
                        aria-label={`Point ${index + 1} y`}
                        value={point.y}
                        onChange={(e) =>
                          m.onEditCoordinate(m.coordZone as number, index, 1, e.target.value)
                        }
                      />
                    </td>
                    <td>
                      <button
                        type="button"
                        className="row-remove"
                        aria-label={`Remove point ${index + 1}`}
                        onClick={() => m.onRemoveCoordinate(m.coordZone as number, index)}
                        disabled={m.busy || m.coordRows!.length <= MIN_POINTS}
                      >
                        &times;
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            <div className="button-row">
              <button type="button" onClick={m.onAddPoint} disabled={m.busy || !m.canAddPoint}>
                Add point
              </button>
            </div>
          </details>
        )}

      </section>

      <aside className="side">
        {m.screenMode !== "edit" && m.detectionSystems !== undefined && m.detectionSystems !== null && (
          <section className="card systems-card" data-testid="systems-card">
            <h2>Detection systems</h2>
            {m.detectionSystems}
          </section>
        )}
        <section className="card zone-panel">
          <div className="zone-panel-head">
            <h2>Danger zones</h2>
            <button
              type="button"
              className="add-zone-btn"
              onClick={m.onAddZone}
              disabled={m.busy || !m.canEdit}
              title={!m.canEdit ? "Start the camera first" : "Draw a new zone"}
            >
              + Add zone
            </button>
          </div>
          {m.zoneCards.length === 0 ? (
            <p className="muted small">No danger zone yet. Use Edit zones to draw one.</p>
          ) : (
            <ul className="zone-cards" data-testid="zone-cards">
              {m.zoneCards.map((card, zoneIndex) => (
                <li key={zoneIndex} className={`zone-card ${card.open ? "open" : ""}`}>
                  <div className="zone-card-head">
                    <span className={`zone-dot ${card.dot}`} aria-hidden="true" />
                    <button
                      type="button"
                      className="zone-card-open"
                      aria-expanded={m.screenMode === "edit" ? card.open : undefined}
                      aria-label={`Edit ${card.name}`}
                      onClick={() => m.onOpenZoneCard(zoneIndex)}
                    >
                      <span className="zone-card-name">{card.name}</span>
                    </button>
                    {m.renaming === zoneIndex ? (
                      <input
                        className="zone-card-rename"
                        value={m.renameDraft}
                        aria-label={`New name for ${card.name}`}
                        autoFocus
                        onClick={(event) => event.stopPropagation()}
                        onChange={(event) => m.onRenameDraft(event.target.value)}
                        onBlur={m.onCommitRename}
                        onKeyDown={(event) => {
                          if (event.key === "Enter") {
                            event.preventDefault();
                            m.onCommitRename();
                          } else if (event.key === "Escape") {
                            event.preventDefault();
                            m.onRenameCancel();
                          }
                        }}
                      />
                    ) : (
                      <button
                        type="button"
                        className="zone-card-rename-start"
                        aria-label={`Rename ${card.name}`}
                        onClick={() => m.onRenameStart(zoneIndex)}
                      >
                        Rename
                      </button>
                    )}
                    <label
                      className="zone-enable"
                      title={card.enabled ? "Disable zone" : "Enable zone"}
                    >
                      <input
                        type="checkbox"
                        aria-label={`${card.enabled ? "Disable" : "Enable"} ${card.name}`}
                        checked={card.enabled}
                        onChange={(event) => m.onToggleZoneEnabled(zoneIndex, event.target.checked)}
                      />
                      <span className="zone-enable-slider" aria-hidden="true" />
                    </label>
                    <span className="zone-card-count">{card.countText}</span>
                    {!card.enabled && <span className="zone-disabled">Disabled</span>}
                  </div>

                  {card.showFigures && (
                    <dl className="zone-card-figures">
                      <div className="zone-card-figure">
                        <dt>People</dt>
                        <dd
                          data-testid={`zone-people-${zoneIndex}`}
                          className={card.peopleOver ? "over" : undefined}
                        >
                          {card.peopleText}
                        </dd>
                      </div>
                      <div className="zone-card-figure">
                        <dt>Longest dwell</dt>
                        <dd
                          data-testid={`zone-dwell-${zoneIndex}`}
                          className={card.dwellOver ? "over" : undefined}
                        >
                          {card.dwellText}
                        </dd>
                      </div>
                    </dl>
                  )}

                  {card.open && (
                    <div className="zone-card-body">
                      <p className="muted small">
                        {card.corners} corners
                        {card.usingDefaults ? " - using the default rules" : ""}
                      </p>
                      {card.rules}
                      <div className="button-row">
                        <button
                          type="button"
                          className="danger"
                          onClick={() => m.onDeleteZone(zoneIndex)}
                          disabled={m.busy}
                        >
                          Delete zone
                        </button>
            </div>
          </div>
        )}
                </li>
              ))}
              </ul>
            )}
          </section>

          {m.screenMode !== "edit" && m.sidePaths}

          {m.screenMode !== "edit" && (
          <section className="card recent-card" data-testid="recent-alerts">
            <div className="recent-alerts-head">
              <h2 className="recent-card-h2">Recent alerts</h2>
              {mergeRecent(m.recentAlerts, m.recentFireAlerts ?? [], m.recentLimit ?? 5).length > 0 && (
                <button type="button" className="link-button" onClick={m.onViewAllAlerts}>
                  View all
                </button>
              )}
            </div>
            {(() => {
              const items = mergeRecent(m.recentAlerts, m.recentFireAlerts ?? [], m.recentLimit ?? 5);
              if (items.length === 0) {
                return (
                  <p className="muted small" data-testid="recent-alerts-empty">
                    No alerts to acknowledge.
                  </p>
                );
              }
              return (
              <ul className="alert-list compact">
                {items.map((item) => {
                  const fire = item.fire;
                  const zone = item.zone;
                  if (fire !== undefined) {
                    return (
                  <li
                    key={item.key}
                    className="alert-row"
                    data-testid={`recent-alert-fire-${fire.id}`}
                    onClick={() => m.onOpenFireEvidence?.(fire)}
                  >
                    {fire.has_snapshot ? (
                      <img
                        className="alert-thumb"
                        src={fireAlertSnapshotUrl(fire.id)}
                        alt={`${fire.alert_type} alert ${fire.id}`}
                      />
                    ) : (
                      <span className="alert-thumb empty" aria-hidden="true" />
                    )}
                    <div className="alert-body">
                      <span className={`event-badge ${fireBadgeTone(fire)}`}>
                        {fire.alert_type === "fire" ? "FIRE" : "SMOKE"} {fire.confidence.toFixed(2)}
                      </span>
                      <span className="alert-kind">{fire.severity}</span>
                      <span className="alert-meta">
                        {fire.source_label ?? "Unknown source"} · {fireCardMeta(fire)}
                      </span>
                    </div>
                    <button
                      type="button"
                      className="alert-ack"
                      onClick={(event) => {
                        event.stopPropagation();
                        m.onAcknowledgeFire?.(fire.id);
                      }}
                    >
                      Acknowledge
                    </button>
                  </li>
                    );
                  }
                  if (zone === undefined) return null;
                  return (
                  <li key={item.key} className="alert-row" data-testid={`recent-alert-zone-${zone.id}`}>
                    {zone.has_snapshot ? (
                      <img
                        className="alert-thumb"
                        src={alertSnapshotUrl(zone.id)}
                        alt={`Alert ${zone.id}`}
                      />
                    ) : (
                      <span className="alert-thumb empty" aria-hidden="true" />
                    )}
                    <div className="alert-body">
                      <span className={`event-badge ${badgeTone(zone.event_type)}`}>
                        {badgeText(zone.event_type)}
                      </span>
                      <span className="alert-kind">{alertLabel(zone)}</span>
                      <span className="alert-meta">
                        {alertWho(zone)}
                        {zone.event_type !== "CAPACITY_ALERT" && zone.zone_name
                          ? ` · ${zone.zone_name}`
                          : ""}{" "}
                        · {alertTime(zone)}
                      </span>
                    </div>
                    <button
                      type="button"
                      className="alert-ack"
                      onClick={() => m.onAcknowledge(zone.id)}
                    >
                      Acknowledge
                    </button>
                  </li>
                  );
                })}
              </ul>
              );
            })()}
          </section>
          )}
          {(m.fireEvidence ?? null) !== null && (
            <EvidenceModal
              model={{
                event: {
                  id: (m.fireEvidence as FireAlert).id,
                  worker_id: null,
                  machine_id: null,
                  event_type:
                    (m.fireEvidence as FireAlert).alert_type === "fire" ? "FIRE" : "SMOKE",
                  reason: null,
                  inside: null,
                  entered_at: null,
                  alert_at: (m.fireEvidence as FireAlert).reported_at,
                  dwell_seconds: null,
                  zone_index: null,
                  zone_name: null,
                  has_snapshot: (m.fireEvidence as FireAlert).has_snapshot,
                  acknowledged: (m.fireEvidence as FireAlert).acknowledged,
                  acknowledged_at: (m.fireEvidence as FireAlert).acknowledged_at,
                },
                hasPrev: m.fireEvidenceHasPrev ?? false,
                hasNext: m.fireEvidenceHasNext ?? false,
                onPrev: () => m.onFireEvidenceStep?.(-1),
                onNext: () => m.onFireEvidenceStep?.(1),
                onClose: () => m.onCloseFireEvidence?.(),
                imageUrl: (m.fireEvidence as FireAlert).has_snapshot
                  ? fireAlertSnapshotUrl((m.fireEvidence as FireAlert).id)
                  : null,
                titleOverride:
                  `${(m.fireEvidence as FireAlert).alert_type === "fire" ? "FIRE" : "SMOKE"} · ` +
                  (m.fireEvidence as FireAlert).severity,
                metaOverride: fireModalMeta(m.fireEvidence as FireAlert),
                onFalseAlarm: m.onFireFalseAlarm
                  ? () => m.onFireFalseAlarm?.((m.fireEvidence as FireAlert).id)
                  : null,
                falseAlarmBusy:
                  (m.fireFalseAlarming ?? null) === (m.fireEvidence as FireAlert).id,
                falseAlarmMarked: (m.fireEvidence as FireAlert).false_alarm === true,
                details: m.fireEvidenceDetails ?? null,
                onAcknowledge: m.onAcknowledgeFireEvidence ?? null,
                acknowledged: (m.fireEvidence as FireAlert).acknowledged,
              }}
            />
          )}
        </aside>
    </div>
  );
}
