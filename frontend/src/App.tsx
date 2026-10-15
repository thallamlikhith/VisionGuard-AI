import { useCallback, useEffect, useRef, useState } from "react";
import type { ChangeEvent, DragEvent } from "react";
import { HashRouter, Navigate, Route, Routes, useNavigate } from "react-router-dom";
import {
  acknowledgeAlert,
  acknowledgeAllAlerts,
  acknowledgeFireAlert,
  deleteVideo,
  getAlerts,
  getConfig,
  getFireAlerts,
  getState,
  getVideos,
  markFireFalseAlarm,
  saveConfig,
  startMonitoring,
  stopMonitoring,
  switchSource,
  uploadVideo,
} from "./api";
import Layout from "./components/Layout";
import DetectionPathsCard from "./components/DetectionPathsCard";
import type { DetectionPathRow } from "./components/DetectionPathsCard";
import DetectionSystemsPanel from "./components/DetectionSystemsPanel";
import type { DetectionSystemRow } from "./components/DetectionSystemsPanel";
import SidePathsReadout from "./components/SidePathsReadout";
import { clockTime } from "./utils/format";
import AlertsPage from "./pages/AlertsPage";
import HistoryPage from "./pages/HistoryPage";
import LiveMonitorPage from "./pages/LiveMonitorPage";
import type { BannerInside, BannerWorker, LiveMonitorModel } from "./pages/LiveMonitorPage";
import SettingsPage from "./pages/SettingsPage";
import type {
  AlertEvent,
  AnchorPoint,
  AppConfig,
  AppState,
  DrawTool,
  FireAlert,
  Point,
  RuleMode,
  TestVideo,
  ZonePayload,
  ZoneReport,
  ZoneSpec,
} from "./types";

/** /state is polled four times a second so the dwell timers feel live */
const POLL_MS = 200;
/** alerts change on the scale of a dwell timer, so they are read far less often */
const ALERTS_POLL_MS = 3000;
/** how many recent alerts sit under the video before the Alerts page is needed */
const RECENT_ALERT_COUNT = 5;
/** how long a confirmation stays in the blue bar before it clears itself */
const NOTICE_MS = 4000;
/** settings are saved without being pressed: a change waits this long for the
 *  rest of the typing, then goes to the backend on its own */
const SETTINGS_APPLY_MS = 500;
/** Paint-style shapes need many more corners than a hand-placed polygon: a
 *  circle is 24 and freehand is simplified down to 24, so the ceiling is 32. */
const MAX_POINTS = 32;
const MIN_POINTS = 3;
/** the 4-point tool places exactly this many corners, then the zone is made */
const QUAD_POINTS = 4;
/** a circle becomes this many points, so it stays cheap to test every frame */
const CIRCLE_POINTS = 24;
/** a freehand stroke is simplified to at most this many points */
const FREEHAND_POINTS = 24;
/** a sliver of the frame is never a real danger area */
const MIN_ZONE_AREA_RATIO = 0.005;
/** how close (screen px) two consecutive hand-placed corners may be before the
 *  second click counts as the same corner. A shaky hand lands its presses a
 *  few pixels apart; without this the near-duplicate spike trips the
 *  self-crossing check on close. Intentional corners are far apart, so they
 *  never merge. */
const DUP_RADIUS = 5;
/** how close (screen px) a Polygon-tool click must land to the first corner
 *  to close the ring instead of adding another corner. Wider than DUP_RADIUS
 *  on purpose: closing is deliberate, so it should be easy to hit - the same
 *  30-frame-px spirit as the DMS PolygonEditor's close snap. */
const CLOSE_RADIUS = 12;
/** how far a click can miss a shape and still select it, in screen pixels */
const SELECT_RADIUS = 16;
/** how far a pointer may move before a click counts as a drag instead */
const CLICK_SLOP = 4;

/** Shipped detection-system defaults, mirroring config.yaml: the hazard zone
 *  on, the three AI side models off, each side cadence at its operating point.
 *  The Detection Paths card's "Reset to defaults" button restores these. */
const DETECTION_DEFAULTS = {
  hazard: true,
  fire: false,
  smoke: false,
  fireFps: 2.0,
  crowd: false,
  crowdFps: 2.0,
  ppe: false,
  ppeFps: 1.0,
} as const;

const ZONE_SAVED = "Zone saved - monitoring is live";
const ZONES_REMOVED = "All zones removed - monitoring is live";
/** shown when a source is picked that has no zones of its own yet */
const ZONES_FOR_SOURCE = "Draw zones for this source";

/** Windows and POSIX spell the same file differently. Comparing two sources is
 *  about the file, not the spelling. */
const samePath = (a: string, b: string) =>
  a.replace(/\\/g, "/").replace(/\/+$/, "").toLowerCase() ===
  b.replace(/\\/g, "/").replace(/\/+$/, "").toLowerCase();

/** the upload date of a clip as YYYY-MM-DD. The backend stores an ISO stamp;
 *  an unreadable one is shown raw rather than hidden. */
function formatVideoDate(created: string): string {
  const when = new Date(created);
  if (Number.isNaN(when.getTime())) return created;
  return when.toLocaleDateString("en-CA");
}

/** Everything the user is told goes through one of these, so there is a single
 *  place a message can appear. A notice clears itself; an error stays, because
 *  it explains why nothing happened. */
type Toast = { text: string; tone: "notice" | "error" } | null;

/** A short two tone beep for the alert. Browsers only let an AudioContext start
 *  from a real click, which is why the context is opened on the Start camera
 *  press instead of when the first alert arrives. */
function beep(context: AudioContext | null): void {
  if (!context) return;
  const start = context.currentTime;
  const gain = context.createGain();
  gain.gain.setValueAtTime(0.0001, start);
  gain.gain.exponentialRampToValueAtTime(0.2, start + 0.01);
  gain.gain.exponentialRampToValueAtTime(0.0001, start + 0.4);
  gain.connect(context.destination);
  const tone = context.createOscillator();
  tone.type = "square";
  tone.frequency.setValueAtTime(880, start);
  tone.connect(gain);
  tone.start(start);
  tone.stop(start + 0.42);
}

/** Every point that comes off the pointer is clamped, so a shape can never be
 *  drawn (or dragged) outside the real frame. */
function clampPoint(point: Point, width: number, height: number): Point {
  // a frame size of 0 means it is not known yet, so there is nothing to clamp
  // against. Clamping anyway pins every corner to (0, 0), which does not fit a
  // zone onto a frame, it destroys it: a three corner zone becomes one point
  // and the API answers a start with "needs at least 3 points, got 1".
  if (width <= 0 || height <= 0) return [Math.round(point[0]), Math.round(point[1])];
  return [
    Math.min(Math.max(Math.round(point[0]), 0), width),
    Math.min(Math.max(Math.round(point[1]), 0), height),
  ];
}

/** Drops corners that sit on the same pixel as the one before them.
 *
 *  Clamping a shape against the edge of the frame lands several corners on the
 *  same point, and a zero length edge between two identical corners is not a
 *  polygon. Only neighbours are merged: a shape that folds back onto an earlier
 *  corner is a different mistake and is still reported. */
function dropRepeats(points: Point[]): Point[] {
  const kept: Point[] = [];
  for (const point of points) {
    const previous = kept[kept.length - 1];
    if (previous && previous[0] === point[0] && previous[1] === point[1]) continue;
    kept.push(point);
  }
  // the shape closes from the last corner back to the first, so those two have
  // to be different points too
  const first = kept[0];
  const last = kept[kept.length - 1];
  if (kept.length > 1 && first && last && first[0] === last[0] && first[1] === last[1]) {
    kept.pop();
  }
  return kept;
}

/** Drops consecutive corners closer together than a double-click's jitter,
 *  so every finish path (double-click, Enter, Finish button) sees the same
 *  shape. Only neighbours merge: a shape that folds back onto an earlier
 *  corner is a different mistake and is still reported. */
function dropNearRepeats(points: Point[], tolerance: number): Point[] {
  if (tolerance <= 0) return points;
  const kept: Point[] = [];
  for (const point of points) {
    const previous = kept[kept.length - 1];
    if (
      previous &&
      Math.hypot(previous[0] - point[0], previous[1] - point[1]) <= tolerance
    ) {
      continue;
    }
    kept.push(point);
  }
  return kept;
}

/** Clamps a whole polygon to the frame and merges corners that clamping pushed
 *  together, so the result is still a shape that can be closed and saved. */
function clampPolygon(points: Point[], width: number, height: number): Point[] {
  return dropRepeats(points.map((point) => clampPoint(point, width, height)));
}

/** A circle/ellipse is stored as a polygon so every zone is just points:
 *  the drag is from the centre outwards, which gives an ellipse when the two
 *  radii differ and a circle when they do not.
 *
 *  A circle dragged past the edge of the frame is moved and shrunk to fit
 *  instead of being flattened against it: clamping corner by corner would pile
 *  up a dozen points on the border and the circle would come out a sliver. */
function ellipseFromDrag(center: Point, edge: Point, width: number, height: number): Point[] {
  const rx = Math.min(Math.abs(edge[0] - center[0]), width / 2);
  const ry = Math.min(Math.abs(edge[1] - center[1]), height / 2);
  // the centre has to leave room for the radius on both sides
  const cx = Math.min(Math.max(center[0], rx), Math.max(rx, width - rx));
  const cy = Math.min(Math.max(center[1], ry), Math.max(ry, height - ry));

  const points: Point[] = [];
  for (let i = 0; i < CIRCLE_POINTS; i += 1) {
    const angle = (2 * Math.PI * i) / CIRCLE_POINTS;
    points.push(
      clampPoint([cx + rx * Math.cos(angle), cy + ry * Math.sin(angle)], width, height)
    );
  }
  return dropRepeats(points);
}

/** Perpendicular distance from a point to the segment a-b. */
function pointSegmentDistance(point: Point, a: Point, b: Point): number {
  const dx = b[0] - a[0];
  const dy = b[1] - a[1];
  const lengthSquared = dx * dx + dy * dy;
  if (lengthSquared === 0) return Math.hypot(point[0] - a[0], point[1] - a[1]);
  let t = ((point[0] - a[0]) * dx + (point[1] - a[1]) * dy) / lengthSquared;
  t = Math.max(0, Math.min(1, t));
  return Math.hypot(point[0] - (a[0] + t * dx), point[1] - (a[1] + t * dy));
}

/** Douglas-Peucker: drops the points along a wobbly freehand stroke that do
 *  not change its shape, so a scribble of 400 samples becomes a clean shape. */
function simplifyPath(points: Point[], tolerance: number): Point[] {
  if (points.length < 3) return points;

  const keep = new Array<boolean>(points.length).fill(false);
  keep[0] = true;
  keep[points.length - 1] = true;

  const stack: [number, number][] = [[0, points.length - 1]];
  while (stack.length > 0) {
    const [first, last] = stack.pop() as [number, number];
    let worst = 0;
    let worstIndex = -1;
    for (let i = first + 1; i < last; i += 1) {
      const distance = pointSegmentDistance(points[i], points[first], points[last]);
      if (distance > worst) {
        worst = distance;
        worstIndex = i;
      }
    }
    if (worstIndex !== -1 && worst > tolerance) {
      keep[worstIndex] = true;
      stack.push([first, worstIndex], [worstIndex, last]);
    }
  }

  const simplified = points.filter((_, index) => keep[index]);
  return simplified.length >= MIN_POINTS ? simplified : points;
}

/** A freehand stroke: simplified to shed the jitter, then capped so a very
 *  long scribble cannot blow past the point limit.
 *
 *  Returns the polygon plus why it failed when it cannot become a zone. A
 *  stroke that folds back on itself or collapses to nothing used to vanish
 *  silently, leaving the operator with no zone and no explanation. An empty
 *  stroke (a plain click, not a shape at all) is neither: both fields are
 *  empty and the caller treats it like the click it was. */
function freehandZone(
  points: Point[],
  width: number,
  height: number,
): { polygon: Point[]; problem: string | null } {
  if (points.length < MIN_POINTS) return { polygon: [], problem: null };
  let simplified = simplifyPath(points, 2);

  if (simplified.length > FREEHAND_POINTS) {
    // keep an even spread so the shape does not change as points are dropped
    const step = (simplified.length - 1) / (FREEHAND_POINTS - 1);
    const thinned: Point[] = [];
    for (let i = 0; i < FREEHAND_POINTS; i += 1) {
      thinned.push(simplified[Math.round(i * step)]);
    }
    simplified = thinned;
  }

  const closed = clampPolygon(simplified, width, height);
  if (closed.length < MIN_POINTS) {
    return {
      polygon: [],
      problem: `That stroke collapses to ${closed.length === 1 ? "one corner" : `${closed.length} corners`} - a danger zone needs at least ${MIN_POINTS}`,
    };
  }
  // a stroke that folds back on itself cannot be a danger zone
  if (!polygonIsSimple(closed)) {
    return {
      polygon: [],
      problem:
        "That stroke crosses over itself - draw the outline in one clean loop without doubling back",
    };
  }
  return { polygon: closed, problem: null };
}

/** Shoelace area in frame pixels. */
function polygonArea(points: Point[]): number {
  let total = 0;
  for (let i = 0; i < points.length; i += 1) {
    const [x1, y1] = points[i];
    const [x2, y2] = points[(i + 1) % points.length];
    total += x1 * y2 - x2 * y1;
  }
  return Math.abs(total) / 2;
}

function orientation(a: Point, b: Point, c: Point): number {
  return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
}

/** True only for a proper crossing, so two edges meeting at a corner do not count. */
function segmentsCross(a: Point, b: Point, c: Point, d: Point): boolean {
  const d1 = orientation(c, d, a);
  const d2 = orientation(c, d, b);
  const d3 = orientation(a, b, c);
  const d4 = orientation(a, b, d);
  return d1 * d2 < 0 && d3 * d4 < 0;
}

function polygonIsSimple(points: Point[]): boolean {
  const count = points.length;
  for (let i = 0; i < count; i += 1) {
    for (let j = i + 1; j < count; j += 1) {
      const sharesCorner =
        j === i ||
        (j + 1) % count === i ||
        (i + 1) % count === j;
      if (sharesCorner) continue;
      if (segmentsCross(points[i], points[(i + 1) % count], points[j], points[(j + 1) % count])) {
        return false;
      }
    }
  }
  return true;
}

/** Why this polygon cannot be used as a danger zone, ignoring the video size.
 *
 *  These are the checks that stand on their own, so they still run before the
 *  first frame has arrived. That matters because the API refuses any zone it
 *  cannot use, no matter what the camera is doing, and /start is what sends it
 *  there: a shape whose corners are all in the same place has three points and
 *  then one, which is the difference between passing a count and starting up. */
function shapeProblem(points: Point[]): string | null {
  if (points.length === 0) return null;

  // A closed GeoJSON-style ring repeats the first corner at the end to close
  // the shape explicitly. The backend (drop_repeats) and DMS
  // (validate_polygon_against_frame) both strip that closing duplicate before
  // counting, so the UI must too - otherwise a valid closed triangle reads as
  // "Point 4 and point 1 are at the same position" here while saving fine.
  let ring = points;
  if (
    ring.length > 1 &&
    ring[0][0] === ring[ring.length - 1][0] &&
    ring[0][1] === ring[ring.length - 1][1]
  ) {
    ring = ring.slice(0, -1);
  }
  if (ring.length === 0) return null;

  const count = ring.length;
  if (count < MIN_POINTS) {
    return `A danger zone needs at least ${MIN_POINTS} points, got ${count}.`;
  }
  if (count > MAX_POINTS) {
    return `A danger zone can have at most ${MAX_POINTS} points, got ${count}.`;
  }

  // the API merges corners that land on the same spot before it counts them,
  // so the count it sees is the one that matters, not the number of clicks
  const corners = dropRepeats(ring).length;
  if (corners < MIN_POINTS) {
    return `A danger zone needs ${MIN_POINTS} different corners, and those ${count} points make ${corners} once the repeats are dropped.`;
  }

  const seen = new Map<string, number>();
  for (let index = 0; index < count; index += 1) {
    const key = ring[index].join(",");
    const first = seen.get(key);
    if (first !== undefined) {
      return `Point ${index + 1} and point ${first + 1} are at the same position - the corners must all be different.`;
    }
    seen.set(key, index);
  }

  if (!polygonIsSimple(ring)) {
    return "The zone crosses over itself - the corners have to go around in order.";
  }

  return null;
}

/** Why this polygon cannot be used as a danger zone, or null when it is fine.
 *
 *  An empty list is not a problem: it is how the UI says "no zone yet". A frame
 *  size of 0 means it is not known yet, so the checks that need one are skipped
 *  instead of being run against a made-up picture. The API applies exactly the
 *  same rules, so anything that reaches save has already been through here. */
function zoneProblem(points: Point[], width: number, height: number): string | null {
  const shape = shapeProblem(points);
  if (shape) return shape;
  if (points.length === 0) return null;

  for (let index = 0; index < points.length; index += 1) {
    const [x, y] = points[index];
    if (x < 0 || y < 0) {
      return `Point ${index + 1} (${x}, ${y}) has a negative coordinate.`;
    }
  }

  if (width > 0 && height > 0) {
    const ratio = polygonArea(points) / (width * height);
    if (ratio < MIN_ZONE_AREA_RATIO) {
      const percent = (ratio * 100).toFixed(1);
      const minPercent = (MIN_ZONE_AREA_RATIO * 100).toFixed(1);
      return `The zone covers only ${percent}% of the frame - it has to cover at least ${minPercent}%.`;
    }
  }

  return null;
}

/** Brings a saved zone onto the frame that is on screen now.
 *
 *  A zone drawn on a 640x480 camera means nothing on a 1920x1080 one, so the
 *  saved reference size is used to scale it. Every corner is then clamped onto
 *  the frame: a zone that hangs off the edge is pulled to the edge rather than
 *  refused, so a camera that came back at another resolution is corrected
 *  instead of dropping the zone and asking for it again. */
function fitZonePoints(
  points: Point[],
  reference: [number, number] | undefined,
  width: number,
  height: number,
): { points: Point[]; adjusted: boolean } {
  if (points.length === 0) return { points: [], adjusted: false };

  const onFrame = clampPolygon(points, width, height);
  if (JSON.stringify(onFrame) === JSON.stringify(points) && !zoneProblem(points, width, height)) {
    return { points: onFrame, adjusted: false };
  }

  // it does not fit as drawn: scale it from the saved reference size, then
  // clamp what is left onto the frame
  const [refWidth, refHeight] = reference ?? [];
  let fitted = onFrame;
  if (refWidth && refHeight) {
    fitted = clampPolygon(
      points.map(
        ([x, y]) =>
          [Math.round((x * width) / refWidth), Math.round((y * height) / refHeight)] as Point,
      ),
      width,
      height,
    );
  }
  return { points: fitted, adjusted: JSON.stringify(fitted) !== JSON.stringify(points) };
}

/** Moves the whole zone, then nudges it back inside the frame. */
function shiftZone(
  points: Point[],
  dx: number,
  dy: number,
  width: number,
  height: number,
): Point[] {
  const xs = points.map((point) => point[0]);
  const ys = points.map((point) => point[1]);
  const minX = Math.min(...xs) + dx;
  const maxX = Math.max(...xs) + dx;
  const minY = Math.min(...ys) + dy;
  const maxY = Math.max(...ys) + dy;

  let fixX = 0;
  let fixY = 0;
  if (minX < 0) fixX = -minX;
  else if (maxX > width) fixX = width - maxX;
  if (minY < 0) fixY = -minY;
  else if (maxY > height) fixY = height - maxY;

  return points.map(([x, y]) => clampPoint([x + dx + fixX, y + dy + fixY], width, height));
}

/** even-odd test, so a click inside a shape can be told apart from empty frame */
function pointInZone(point: Point, zone: Point[]): boolean {
  let inside = false;
  for (let i = 0, j = zone.length - 1; i < zone.length; j = i, i += 1) {
    const [xi, yi] = zone[i];
    const [xj, yj] = zone[j];
    if (yi > point[1] !== yj > point[1]) {
      const crossing = ((xj - xi) * (point[1] - yi)) / (yj - yi) + xi;
      if (point[0] < crossing) inside = !inside;
    }
  }
  return inside;
}

/** How close a point is to a zone's outline, in frame pixels. A click near an
 *  edge still selects the shape, which is what makes a thin zone usable. */
function distanceToZone(point: Point, zone: Point[]): number {
  let best = Infinity;
  for (let i = 0; i < zone.length; i += 1) {
    best = Math.min(best, pointSegmentDistance(point, zone[i], zone[(i + 1) % zone.length]));
  }
  return best;
}

function midpoint(a: Point, b: Point): Point {
  return [Math.round((a[0] + b[0]) / 2), Math.round((a[1] + b[1]) / 2)];
}

/** Turns a press-and-drag into four corners, clockwise from the top left. */
function rectangleFrom(origin: Point, corner: Point, width: number, height: number): Point[] {
  return [
    clampPoint([Math.min(origin[0], corner[0]), Math.min(origin[1], corner[1])], width, height),
    clampPoint([Math.max(origin[0], corner[0]), Math.min(origin[1], corner[1])], width, height),
    clampPoint([Math.max(origin[0], corner[0]), Math.max(origin[1], corner[1])], width, height),
    clampPoint([Math.min(origin[0], corner[0]), Math.max(origin[1], corner[1])], width, height),
  ];
}

/** the pointer moved far enough that this is a drag, not a click */
function movedFar(start: Point, end: Point, slop = CLICK_SLOP): boolean {
  return Math.abs(end[0] - start[0]) > slop || Math.abs(end[1] - start[1]) > slop;
}

/** where the drag began, whatever kind of drag it is */
function dragAnchor(drag: NonNullable<DragState>): Point | null {
  if (drag.kind === "rectangle" || drag.kind === "circle") return drag.origin;
  if (drag.kind === "move") return drag.start;
  return null;
}

/** The zones a config carries. A backend that predates several zones only has
 *  `zone_points`, whose single polygon becomes a one-entry zone list. */
function configZonesOf(config: AppConfig): Point[][] {
  if (config.zones?.length) return config.zones;
  return config.zone_points?.length ? [config.zone_points] : [];
}

/** Which shape the pointer is currently doing something to. */
type DragState =
  | { kind: "rectangle"; origin: Point }
  | { kind: "circle"; origin: Point }
  | { kind: "freehand" }
  /** dragging the body of a zone. `origin` is the shape as it was when the
   *  drag began: every move is measured from it, never from the previous
   *  frame, so the deltas cannot pile up and race the pointer. */
  | { kind: "move"; zone: number; start: Point; origin: Point[] }
  /** dragging one corner of a zone */
  | { kind: "handle"; zone: number; index: number }
  | null;

/** A zone list is either the live one or a saved one; both are Point[][]. */
type ZoneList = Point[][];

function AppShell() {
  const navigate = useNavigate();
  const [state, setState] = useState<AppState | null>(null);

  // every zone being edited (not saved yet)
  const [zones, setZones] = useState<ZoneList>([]);
  // the zones the backend is actually monitoring
  const [savedZones, setSavedZones] = useState<ZoneList>([]);
  // Each zone's name and rules, index aligned with `zones`. A zone that is still
  // following the global defaults is flagged, so the card can say so instead of
  // showing numbers that look like a rule somebody typed in.
  const [zoneSpecs, setZoneSpecs] = useState<ZoneSpec[]>([]);
  // The frame size the zones currently on screen were drawn against. It belongs
  // to the source those zones came from, so a switch replaces it along with the
  // shapes instead of reusing the previous source's size.
  const [zoneReference, setZoneReference] = useState<[number, number] | undefined>(undefined);
  /** the zone whose name is being typed into, and the text so far */
  const [renaming, setRenaming] = useState<number | null>(null);
  const [renameDraft, setRenameDraft] = useState("");
  const [allowed, setAllowed] = useState(0);
  const [source, setSource] = useState("0");
  // uploaded test videos, straight from /videos. The facts shown (size,
  // resolution, duration) are the backend's, never guessed from the file.
  const [videos, setVideos] = useState<TestVideo[]>([]);
  const [videoLimit, setVideoLimit] = useState(10);
  const [videoMaxMb, setVideoMaxMb] = useState(500);
  /** percent while an upload is in flight, null when none is */
  const [uploadPct, setUploadPct] = useState<number | null>(null);
  /** saved zone counts per source string, for the source dropdowns. Saved
   *  counts only, never drafts: choosing a source is about what is stored.
   *  Missing keys are simply not fetched yet (or the fetch failed), and the
   *  option then shows no count rather than an invented one. */
  const [zoneCounts, setZoneCounts] = useState<Record<string, number>>({});
  const zoneCountsRef = useRef<Record<string, number>>({});
  /** a Live Monitor source pick waiting for the running-feed confirm */
  const [confirmLiveSource, setConfirmLiveSource] = useState<string | null>(null);
  /** which half of the Source card is showing while the camera is stopped */
  const [sourceMode, setSourceMode] = useState<"camera" | "video">("camera");
  /** a file is being dragged over the drop area, so it can be highlighted */
  const [dragOver, setDragOver] = useState(false);
  const [machineName, setMachineName] = useState("");
  const [mode, setMode] = useState<RuleMode>("TIMER");
  // demo defaults: a zone should alarm almost at once, and a person standing at
  // a machine is easiest to judge by the middle of the box
  const [dwellSeconds, setDwellSeconds] = useState(1);
  const [repeatAlertSeconds, setRepeatAlertSeconds] = useState(10);
  const [anchor, setAnchor] = useState<AnchorPoint>("center");
  /** detection confidence, edited on the Source card and saved with /config */
  const [confidence, setConfidence] = useState(0.35);
  /** side detection paths, edited on the Detection paths card and saved with
   *  /config like every other setting (fps = analyses per second per path) */
  const [fireEnabled, setFireEnabled] = useState(false);
  const [fireFps, setFireFps] = useState(2.0);
  /** gate persistence windows behind the Fire row's plain-words hint, read
   *  from /config (never edited here); older backends omit them, so the
   *  shipped 1.5 s / 2.5 s operating point is the fallback. */
  const [fireSeconds, setFireSeconds] = useState(1.5);
  const [smokeSeconds, setSmokeSeconds] = useState(2.5);
  /** smoke runs on the same model pass as fire, with its own switch. */
  const [smokeEnabled, setSmokeEnabled] = useState(false);
  const [crowdEnabled, setCrowdEnabled] = useState(false);
  const [crowdFps, setCrowdFps] = useState(2.0);
  const [ppeEnabled, setPpeEnabled] = useState(false);
  const [ppeFps, setPpeFps] = useState(1.0);
  /** hazard-zone system switch, same stored-settings source of truth as the
   *  side paths (toggled from the Live Monitor panel for now). */
  const [hazardEnabled, setHazardEnabled] = useState(true);
  /** replay an uploaded test clip when it ends. One stored setting, ticked
   *  from the Live Monitor source picker or the Settings Camera card; off
   *  plays the clip once and stops. The webcam never reads it. */
  const [loopVideo, setLoopVideo] = useState(false);

  // nothing is picked to begin with, so the hint on the video can say what to do
  const [tool, setTool] = useState<DrawTool | null>(null);
  const [editing, setEditing] = useState(true);
  const [drag, setDrag] = useState<DragState>(null);
  const [cursor, setCursor] = useState<Point | null>(null);
  /** index of the selected zone, and of the selected corner inside it */
  const [selected, setSelected] = useState<number | null>(null);
  const [selectedPoint, setSelectedPoint] = useState<number | null>(null);
  const [history, setHistory] = useState<ZoneList[]>([]);
  /** corners of the 4-point draft, before the 4th click makes the zone */
  const [draft, setDraft] = useState<Point[]>([]);
  const [displayWidth, setDisplayWidth] = useState(0);
  /** every message the user is told, in one place */
  const [toast, setToast] = useState<Toast>(null);
  const [busy, setBusy] = useState(false);
  /** the alert beep can be turned off for a demo without turning it off for good */
  const [muted, setMuted] = useState(false);
  /** second click of a two-step "remove every zone", so nothing is deleted from
   *  the camera by a single stray click */
  const [confirmingRemove, setConfirmingRemove] = useState(false);
  /** the recorded alerts, newest first. Empty until /events has answered. */
  const [alerts, setAlerts] = useState<AlertEvent[]>([]);
  /** which Alerts tab is open: the ones needing attention, or the cleared ones */
  const [alertTab, setAlertTab] = useState<"active" | "acknowledged">("active");
  /** the alert whose full picture is open in the evidence modal, if any */
  const [evidence, setEvidence] = useState<AlertEvent | null>(null);
  /** Cancel with unsaved geometry asks first; this is that question. */
  const [confirmingCancel, setConfirmingCancel] = useState(false);
  /** Switching source while the editor holds work the backend has not been
   *  told about asks first. This is that question, and the source waiting for
   *  the answer - the field itself goes back to the source in use until the
   *  operator has given it. */
  const [confirmingSwitch, setConfirmingSwitch] = useState(false);
  const [pendingSource, setPendingSource] = useState<string | null>(null);
  /** The moment the camera was seen to go from stopped to running in THIS page
   *  session. /state carries no start time, so it is written down the only way
   *  it can honestly be: when the transition is actually witnessed. A page that
   *  loads onto an already running camera has no such moment and must not invent
   *  one, so it shows the status alone. */
  const [since, setSince] = useState<number | null>(null);
  /** the previous poll's `running`. null until the first answer arrives, because
   *  "was it running before this page loaded?" has no answer to give. */
  const sawRunning = useRef<boolean | null>(null);

  // Two spellings of the one toast, so the existing call sites stay readable: a
  // notice is a passing message and clears itself, an error stays put.
  const setNotice = useCallback((text: string | null) => {
    setToast(text === null ? null : { text, tone: "notice" });
  }, []);
  const setError = useCallback((text: string | null) => {
    setToast(text === null ? null : { text, tone: "error" });
  }, []);

  useEffect(() => {
    if (toast?.tone !== "notice") return;
    const timer = window.setTimeout(() => setToast(null), NOTICE_MS);
    return () => window.clearTimeout(timer);
  }, [toast]);

  const imageRef = useRef<HTMLElement>(null);
  const streamVersion = useRef(0);
  /** a drag that really moved, so a click on the first corner is not mistaken
   *  for a drag of that corner */
  const movedRef = useRef(false);
  /** the sample points of the freehand stroke in progress */
  const strokeRef = useRef<Point[]>([]);
  /** the stroke mirrored into state while it is drawn, so the overlay can
   *  paint the live preview. The ref stays the source of truth for the
   *  commit; this is only what the picture shows mid-drag. */
  const [strokePreview, setStrokePreview] = useState<Point[] | null>(null);
  /** the source whose zones are currently loaded, so a change can be told from
   *  the mount and from an unrelated rerender */
  const loadedSource = useRef<string | null>(null);
  /** one fitting per source + reference + frame, so an edit never re-scales */
  const fitKey = useRef("");
  /** opened on the Start camera click, because a browser will not allow it later */
  const audioRef = useRef<AudioContext | null>(null);
  /** whether the last poll was already an alert, so the beep is one per alarm
   *  and not one per poll */
  const wasAlert = useRef(false);

  /** Called from the Start camera click: this is the one gesture a browser
   *  accepts as "the user wants sound", so the beep can play later. */
  const openAudio = useCallback(() => {
    const Ctor =
      window.AudioContext ??
      (window as unknown as { webkitAudioContext?: typeof AudioContext })
        .webkitAudioContext;
    if (!Ctor) return;
    if (!audioRef.current) audioRef.current = new Ctor();
    if (audioRef.current.state === "suspended") void audioRef.current.resume();
  }, []);

  /** The first click or key press anywhere opens the audio, because a browser
   *  will not allow sound after that and there is only one such gesture to
   *  spend.
   *
   *  The Start button was doing this job, so a page that loaded onto a camera
   *  that was ALREADY running never got one: there was no Start click to
   *  happen, and the alarm beeped silently into a suspended context. The first
   *  interaction with the page is just as good a gesture, and it is the
   *  interaction an operator reloads the page to make. Both listeners come off
   *  after it runs, so this cannot cost a second AudioContext. */
  const audioUnlocked = useRef(false);
  useEffect(() => {
    const unlock = (): void => {
      if (audioUnlocked.current) return;
      audioUnlocked.current = true;
      openAudio();
      window.removeEventListener("click", unlock);
      window.removeEventListener("keydown", unlock);
    };
    window.addEventListener("click", unlock);
    window.addEventListener("keydown", unlock);
    return () => {
      window.removeEventListener("click", unlock);
      window.removeEventListener("keydown", unlock);
    };
  }, [openAudio]);

  const copyZones = (list: ZoneList): ZoneList =>
    list.map((zone) => zone.map((point) => [point[0], point[1]] as Point));

  /** One spec per polygon, lined up by position. A zone the backend has no rule
   *  for is given the globals it is following, flagged as such, so the card can
   *  say "using the default" instead of implying somebody chose those numbers. */
  const specsFor = (
    zoneCount: number,
    saved: ZoneSpec[] | undefined,
    globals: { allowed: number; dwell: number }
  ): ZoneSpec[] =>
    Array.from({ length: zoneCount }, (_, index) => {
      const spec = saved?.[index];
      if (spec?.name) {
        return {
          name: spec.name,
          allowed_people: spec.allowed_people,
          dwell_seconds: spec.dwell_seconds,
          enabled: spec.enabled,
          using_defaults: spec.using_defaults,
        };
      }
      return {
        name: `Zone ${index + 1}`,
        allowed_people: globals.allowed,
        dwell_seconds: globals.dwell,
        enabled: true,
        using_defaults: true,
      };
    });

  // --- initial load -----------------------------------------------------
  useEffect(() => {
    getConfig()
      .then((cfg) => {
        // a backend that only knows one zone still works: its single polygon
        // becomes a one-entry zone list
        const cfgZones = configZonesOf(cfg);
        setZones(copyZones(cfgZones));
        setSavedZones(copyZones(cfgZones));
        setZoneReference(cfg.zone_reference);
        setZoneSpecs(
          specsFor(cfgZones.length, cfg.zone_specs, {
            allowed: cfg.allowed_people,
            dwell: cfg.rules.dwell_seconds,
          })
        );
        setAllowed(cfg.allowed_people);
        setSource(cfg.source);
        setConfidence(cfg.detector.confidence);
        // backends from before the Detection paths card send nothing: every
        // path stays off at its operating-point cadence
        setFireEnabled(
          cfg.fire_smoke?.fire_enabled ?? cfg.fire_smoke?.enabled ?? false
        );
        setSmokeEnabled(
          cfg.fire_smoke?.smoke_enabled ?? cfg.fire_smoke?.enabled ?? false
        );
        setFireFps(cfg.fire_smoke?.analysis_fps ?? 2.0);
        setFireSeconds(cfg.fire_smoke?.fire_seconds ?? 1.5);
        setSmokeSeconds(cfg.fire_smoke?.smoke_seconds ?? 2.5);
        setCrowdEnabled(cfg.crowd?.enabled ?? false);
        setCrowdFps(cfg.crowd?.analysis_fps ?? 2.0);
        setPpeEnabled(cfg.ppe?.enabled ?? false);
        setPpeFps(cfg.ppe?.analysis_fps ?? 1.0);
        setHazardEnabled(cfg.hazard?.enabled ?? true);
        // backends from before the Loop video box send nothing: off
        setLoopVideo(cfg.loop_video ?? false);
        setMachineName(cfg.machine.name);
        if (cfg.rules.mode === "INSTANT" || cfg.rules.mode === "TIMER" || cfg.rules.mode === "LIMIT") {
          setMode(cfg.rules.mode);
        }
        setDwellSeconds(cfg.rules.dwell_seconds);
        // a backend older than the anchor option simply has none, and feet is
        // the right answer for it anyway
        if (cfg.anchor === "bottom_center" || cfg.anchor === "center") {
          setAnchor(cfg.anchor);
        }
        // A page that arrives with no zones is still Live: there is nothing to
        // save and nothing to confirm, and walking the operator into the editor
        // unasked is the wizard thinking it knows better. The drawing tools are
        // one Edit zones press away.
        setEditing(false);
        setSettingsLoaded(true);
      })
      .catch((err) => {
        setError(`Cannot reach the backend: ${err.message}`);
        // the settings are live even when the first read failed, otherwise a
        // backend that comes up late could never be corrected from the UI
        setSettingsLoaded(true);
      });
  }, []);

  // --- recorded alerts --------------------------------------------------
  // One fetch feeds both the strip under the video and the Alerts page. It is
  // polled slowly: an alert list does not need four updates a second like the
  // dwell timers do. A failure is swallowed here because the recent strip is
  // only a convenience; the Alerts page is where a broken backend is reported.
  const [alertsError, setAlertsError] = useState<string | null>(null);
  /** True unacknowledged alerts on the backend, not just the rows loaded:
   *  the sidebar badge and the Active tab both read this. */
  const [unackedTotal, setUnackedTotal] = useState(0);
  /** True acknowledged alerts on the backend, not just the rows loaded:
   *  the Acknowledged tab reads this, the same way the Active tab reads
   *  unackedTotal. Without it the tab sticks at the /events page size. */
  const [ackedTotal, setAckedTotal] = useState(0);
  /** fire/smoke rows with their own tab totals. The sidebar badge sums the
   *  zone and fire unacknowledged totals, matching the two Active counts on
   *  the Alerts page. Evidence for one open fire row. */
  const [fireAlerts, setFireAlerts] = useState<FireAlert[]>([]);
  const [fireUnackedTotal, setFireUnackedTotal] = useState(0);
  const [fireAckedTotal, setFireAckedTotal] = useState(0);
  const [fireEvidence, setFireEvidence] = useState<FireAlert | null>(null);
  const alertsInflight = useRef(false);
  const loadAlerts = useCallback(async () => {
    if (alertsInflight.current) return;
    alertsInflight.current = true;
    try {
      // two filtered reads: the badge and the Active tab need the true
      // unacknowledged total even when the backlog is bigger than one page
      const [open, done] = await Promise.all([getAlerts(false), getAlerts(true)]);
      setAlerts([...(open.events ?? []), ...(done.events ?? [])]);
      setUnackedTotal(
        open.total ?? (open.events ?? []).filter((event) => !event.acknowledged).length
      );
      setAckedTotal(done.total ?? (done.events ?? []).filter((event) => event.acknowledged).length);
      setAlertsError(null);
    } catch (err) {
      if ((err as Error).message === "Request cancelled") return;
      setAlertsError((err as Error).message);
    } finally {
      alertsInflight.current = false;
    }
    // fire/smoke rows ride the same tick but keep their own totals, so the
    // zone badge and tab counts behave exactly as before. A failure is
    // swallowed: the rows simply stay as they were until the next tick.
    try {
      const [fireOpen, fireDone] = await Promise.all([
        getFireAlerts(false),
        getFireAlerts(true),
      ]);
      setFireAlerts([...(fireOpen.alerts ?? []), ...(fireDone.alerts ?? [])]);
      setFireUnackedTotal(fireOpen.unacked ?? (fireOpen.alerts ?? []).length);
      setFireAckedTotal(fireDone.acked ?? (fireDone.alerts ?? []).length);
    } catch {
      /* keep the previous rows; the next tick retries */
    }
  }, []);

  useEffect(() => {
    void loadAlerts();
    const id = window.setInterval(() => void loadAlerts(), ALERTS_POLL_MS);
    return () => window.clearInterval(id);
  }, [loadAlerts]);

  const recentAlerts = alerts
    .filter((event) => !event.acknowledged)
    .slice(0, RECENT_ALERT_COUNT);
  /** Unacknowledged fire/smoke rows for the Recent card, newest first. The
   *  card merges these with the zone rows above and caps the merged list, so
   *  acking here refreshes the card and the Alerts page together (same
   *  state, same fire route). */
  const recentFireAlerts = fireAlerts
    .filter((alert) => !alert.acknowledged)
    .slice(0, RECENT_ALERT_COUNT);

  /** Acknowledge one alert. The row is replaced in place from the answer so the
   *  list does not have to wait for the next poll. An open evidence modal
   *  follows the same row. */
  const acknowledge = useCallback(async (id: number) => {
    try {
      const { event } = await acknowledgeAlert(id);
      setAlerts((current) => current.map((row) => (row.id === event.id ? event : row)));
      setEvidence((current) => (current !== null && current.id === event.id ? event : current));
      if (event.acknowledged) {
        setUnackedTotal((total) => Math.max(0, total - 1));
        setAckedTotal((total) => total + 1);
      }
    } catch (err) {
      setAlertsError((err as Error).message);
    }
  }, []);

  const activeAlerts = alerts.filter((event) => !event.acknowledged);
  const clearedAlerts = alerts.filter((event) => event.acknowledged);

  /** Acknowledge one fire/smoke alert. The row is replaced in place from the
   *  answer and the fire tab totals move with it; zone totals are untouched. */
  const acknowledgeFire = useCallback(async (id: number) => {
    try {
      const { alert } = await acknowledgeFireAlert(id);
      setFireAlerts((current) => current.map((row) => (row.id === alert.id ? alert : row)));
      setFireEvidence((current) => (current !== null && current.id === alert.id ? alert : current));
      if (alert.acknowledged) {
        setFireUnackedTotal((total) => Math.max(0, total - 1));
        setFireAckedTotal((total) => total + 1);
      }
    } catch (err) {
      setAlertsError((err as Error).message);
    }
  }, []);

  const fireActiveAlerts = fireAlerts.filter((alert) => !alert.acknowledged);
  const fireClearedAlerts = fireAlerts.filter((alert) => alert.acknowledged);

  /** Alert Details panel rows for the open modals, built where the rules
   *  live (zone dwell limits) so the modal stays presentation-only. */
  const evidenceDetails = (() => {
    if (!evidence) return null;
    const limit =
      (evidence.zone_index !== null &&
      evidence.zone_index !== undefined &&
      zoneSpecs[evidence.zone_index]?.dwell_seconds) ??
      dwellSeconds;
    const dwell =
      evidence.dwell_seconds === null || evidence.dwell_seconds === undefined
        ? "—"
        : `${evidence.dwell_seconds.toFixed(1)}s / ${Number(limit).toFixed(1)}s`;
    return [
      { label: "Zone", value: evidence.zone_name || `Zone ${(evidence.zone_index ?? 0) + 1}` },
      { label: "Worker", value: evidence.worker_id !== null && evidence.worker_id >= 0 ? `Worker #${evidence.worker_id}` : "—" },
      { label: "Dwell", value: dwell },
      { label: "Captured at", value: clockTime(evidence.entered_at) || "—" },
      { label: "Confidence", value: "—" },
    ];
  })();

  const fireEvidenceDetails = (() => {
    if (!fireEvidence) return null;
    const video =
      fireEvidence.captured_at !== null && fireEvidence.captured_at !== undefined
        ? `video ${fireEvidence.captured_at.toFixed(1)}s`
        : "video time unknown";
    return [
      { label: "Type", value: fireEvidence.alert_type === "fire" ? "FIRE" : "SMOKE" },
      { label: "Severity", value: fireEvidence.severity },
      { label: "Confidence", value: fireEvidence.confidence.toFixed(2) },
      { label: "Captured at", value: `${video} · ${clockTime(fireEvidence.reported_at) || "—"}` },
    ];
  })();

  /** Export one fire/smoke alert as a hard negative (also acks it). The row
   *  and any open evidence come back from the answer; totals reload from
   *  the backend so zone counts are never touched by local math. */
  const [falseAlarming, setFalseAlarming] = useState<number | null>(null);
  const markFalseAlarm = useCallback(async (id: number) => {
    setFalseAlarming(id);
    try {
      const body = await markFireFalseAlarm(id);
      setFireAlerts((current) => current.map((row) => (row.id === id ? body.alert : row)));
      setFireEvidence((current) => (current !== null && current.id === id ? body.alert : current));
      await loadAlerts();
    } catch (err) {
      setAlertsError((err as Error).message);
    } finally {
      setFalseAlarming(null);
    }
  }, [loadAlerts]);

  /** Step the fire evidence modal through the fire rows of the open tab. */
  const stepFireEvidence = (direction: 1 | -1) => {
    if (!fireEvidence) return;
    const list = alertTab === "active" ? fireActiveAlerts : fireClearedAlerts;
    const index = list.findIndex((alert) => alert.id === fireEvidence.id);
    const next = index === -1 ? undefined : list[index + direction];
    if (next) setFireEvidence(next);
  };
  const fireEvidenceIndex =
    fireEvidence === null
      ? -1
      : (alertTab === "active" ? fireActiveAlerts : fireClearedAlerts).findIndex(
          (alert) => alert.id === fireEvidence.id
        );

  /** Step a fire evidence modal opened from the Recent card through the
   *  card's own rows (unacknowledged, newest first) — not the Alerts tab. */
  const stepLiveFireEvidence = (direction: 1 | -1) => {
    if (!fireEvidence) return;
    const index = fireActiveAlerts.findIndex((alert) => alert.id === fireEvidence.id);
    const next = index === -1 ? undefined : fireActiveAlerts[index + direction];
    if (next) setFireEvidence(next);
  };
  const liveFireEvidenceIndex =
    fireEvidence === null
      ? -1
      : fireActiveAlerts.findIndex((alert) => alert.id === fireEvidence.id);

  /** Acknowledge everything still active in one server call, then reload the
   *  list so the rows, the tabs and the badge all read the same answer. */
  const [ackingAll, setAckingAll] = useState(false);
  const acknowledgeAll = async () => {
    if (unackedTotal === 0 || ackingAll) return;
    setAckingAll(true);
    try {
      await acknowledgeAllAlerts();
      await loadAlerts();
    } catch (err) {
      setAlertsError((err as Error).message);
    } finally {
      setAckingAll(false);
    }
  };

  /** Step the evidence modal through the alerts of the open tab. */
  const stepEvidence = (direction: 1 | -1) => {
    if (!evidence) return;
    const list = alertTab === "active" ? activeAlerts : clearedAlerts;
    const index = list.findIndex((event) => event.id === evidence.id);
    const next = index === -1 ? undefined : list[index + direction];
    if (next) setEvidence(next);
  };
  const evidenceIndex =
    evidence === null
      ? -1
      : (alertTab === "active" ? activeAlerts : clearedAlerts).findIndex(
          (event) => event.id === evidence.id
        );

  // Escape closes the evidence picture, the way every other overlay closes.
  useEffect(() => {
    if (!evidence && !fireEvidence) return;
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setEvidence(null);
        setFireEvidence(null);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [evidence, fireEvidence]);

  // --- uploaded test videos --------------------------------------------
  // Read once on load. A failure is not fatal: the camera still runs, the list
  // is simply empty, and the message is left to whatever else is on screen.
  useEffect(() => {
    getVideos()
      .then((list) => {
        setVideos(list.videos);
        setVideoLimit(list.limit);
        setVideoMaxMb(list.max_mb);
      })
      .catch(() => {
        /* the monitor works without the list */
      });
  }, []);

  // --- poll the live state ---------------------------------------------
  // The newest frame sequence, so repeats of the same picture do not count
  // as new arrivals.
  const lastSeq = useRef<number | null>(null);
  /** true while a /state request is in flight: the interval skips its tick
   *  instead of piling a second request onto a slow backend. */
  const stateInflight = useRef(false);
  /** true after the backend went quiet, so the offline message is shown once
   *  instead of being re-created (and re-timed) on every 200 ms poll. */
  const stateOffline = useRef(false);
  /** retry counter for the MJPEG <img>: bumped when the stream breaks so the
   *  browser reconnects instead of sitting on a dead picture. */
  const [streamRetry, setStreamRetry] = useState(0);
  const streamRetryTimer = useRef<number | null>(null);
  useEffect(() => {
    let active = true;
    const controller = new AbortController();
    const showOffline = (message: string) => {
      if (stateOffline.current) return;
      stateOffline.current = true;
      setToast((current) =>
        current !== null ? current : { text: message, tone: "error" as const }
      );
    };
    const showBackendError = (message: string) => {
      setToast((current) => (current?.text === message ? current : { text: message, tone: "error" as const }));
    };
    const clearOffline = () => {
      if (!stateOffline.current) return;
      stateOffline.current = false;
      // any poll-failure toast goes: a single slow poll used to leave its
      // "Request timed out" toast up forever, over a video that had long
      // recovered. Backend-reported errors (next.error) are re-set below on
      // the same tick, so they survive this.
      setToast((current) =>
        current?.tone === "error" &&
        (current.text.startsWith("Cannot reach the backend") ||
          current.text.startsWith("Request timed out"))
          ? null
          : current
      );
    };
    const tick = async () => {
      if (stateInflight.current) return;
      stateInflight.current = true;
      try {
        const next = await getState({ signal: controller.signal });
        if (!active) return;
        clearOffline();
        setState(next);
        const seq = next.frame_seq ?? null;
        if (seq !== null && seq > 0 && seq !== lastSeq.current) {
          lastSeq.current = seq;
        }
        // The start time comes from the moment the off-to-running edge is seen
        // between two polls. The first poll only establishes what was already
        // true, so a page that loads with the camera running claims no start
        // time at all instead of inventing one.
        if (sawRunning.current !== null && next.running && !sawRunning.current) {
          setSince(Date.now());
        }
        sawRunning.current = next.running;
        if (next.error) showBackendError(next.error);
      } catch (err) {
        if (!active || (err as Error).message === "Request cancelled") return;
        showOffline((err as Error).message);
      } finally {
        stateInflight.current = false;
      }
    };
    tick();
    const id = setInterval(tick, POLL_MS);
    return () => {
      active = false;
      clearInterval(id);
      controller.abort();
      if (streamRetryTimer.current !== null) window.clearTimeout(streamRetryTimer.current);
    };
  }, []);

  // --- frame + coordinate helpers --------------------------------------
  // Points are stored in REAL video pixels. The real size is only known once
  // the pipeline has opened the source, so drawing is refused until then
  // instead of guessing 1280x720 and silently saving a zone that does not
  // match the video.
  const realFrame = {
    width: state?.frame_width ?? 0,
    height: state?.frame_height ?? 0,
  };
  const frameKnown = realFrame.width > 0 && realFrame.height > 0;
  const running = state?.running ?? false;
  const runningRef = useRef(running);
  runningRef.current = running;
  // The picture the editor draws on: the live frame while the camera runs,
  // else the saved reference size for this source - a dark canvas of real
  // dimensions, so zones can be drawn before the first start. Without a saved
  // size there is nothing truthful to draw on, so editing stays shut.
  const refKnown = (zoneReference?.[0] ?? 0) > 0 && (zoneReference?.[1] ?? 0) > 0;
  const surface =
    !running && editing
      ? { width: zoneReference?.[0] ?? 0, height: zoneReference?.[1] ?? 0 }
      : realFrame;
  const surfaceKnown = surface.width > 0 && surface.height > 0;
  const canEdit = running ? frameKnown : refKnown;
  const drawing = editing && surfaceKnown;

  // Handles keep their size on screen, so their radius has to follow how much
  // the picture is scaled down.
  useEffect(() => {
    const measure = () => {
      const width = imageRef.current?.getBoundingClientRect().width ?? 0;
      setDisplayWidth(width > 0 ? width : 0);
    };
    measure();
    window.addEventListener("resize", measure);
    let observer: ResizeObserver | undefined;
    if (typeof ResizeObserver !== "undefined" && imageRef.current) {
      observer = new ResizeObserver(measure);
      observer.observe(imageRef.current);
    }
    return () => {
      window.removeEventListener("resize", measure);
      observer?.disconnect();
    };
  }, [editing]);

  /** frame pixels per screen pixel */
  const unit = displayWidth > 0 && surface.width > 0 ? surface.width / displayWidth : 1;
  /** same value under the name the tool handlers use */
  const frameScale = unit;

  const toFrame = useCallback(
    (clientX: number, clientY: number): Point | null => {
      const rect = imageRef.current?.getBoundingClientRect();
      if (!rect || rect.width === 0 || !surfaceKnown) return null;
      const x = ((clientX - rect.left) / rect.width) * surface.width;
      const y = ((clientY - rect.top) / rect.height) * surface.height;
      return [x, y];
    },
    [surface.width, surface.height, surfaceKnown]
  );

  /** true when a point sits outside the drawing surface */
  const isOutOfFrame = useCallback(
    (point: Point): boolean =>
      !surfaceKnown ||
      point[0] < 0 ||
      point[1] < 0 ||
      point[0] > surface.width ||
      point[1] > surface.height,
    [surface.width, surface.height, surfaceKnown]
  );

  // --- zones that do not match the frame they are shown on ---------------
  // A zone drawn against another frame size is rescaled once, from the
  // reference size it was stored under to the frame it is being shown on.
  // Every zone is kept: a shape scaling cannot make fit is shown clamped onto
  // the frame instead of being thrown out of the state, and nothing here is
  // sent anywhere - the stored zones only change when the operator presses
  // Save. Runs once per source + reference + frame, so editing a corner never
  // triggers it.
  useEffect(() => {
    // stopped editing draws in the reference's own coordinates, so there is
    // nothing to fit: rescaling here would move the saved zones for no reason
    if (!running || !frameKnown || zones.length === 0) return;
    const key = `${source}|${zoneReference?.join("x") ?? ""}|${realFrame.width}x${realFrame.height}`;
    if (fitKey.current === key) return;
    fitKey.current = key;

    const results = zones.map((zone) =>
      fitZonePoints(zone, zoneReference, realFrame.width, realFrame.height)
    );
    // these zones now belong to this frame, whatever their old reference was
    const same =
      zoneReference && zoneReference[0] === realFrame.width && zoneReference[1] === realFrame.height;
    if (!same) setZoneReference([realFrame.width, realFrame.height]);
    if (!results.some((result) => result.adjusted)) return;

    const kept = results.map((result) => result.points);
    setNotice("Zone adjusted to fit the frame");
    setZones(kept);
    setSavedZones(kept);
  }, [zones, zoneReference, frameKnown, realFrame.width, realFrame.height, source, running]);

  // only meaningful once the drawing surface is known. An emptied list is
  // valid only when something was saved before: deleting the last zone must
  // stay saveable (saving it removes all zones, the same empty save the
  // Settings page files), while a fresh editor with nothing drawn yet keeps
  // Save greyed out.
  const countOk =
    zones.every((zone) => zone.length >= MIN_POINTS && zone.length <= MAX_POINTS) &&
    (zones.length > 0 || savedZones.length > 0);
  /** the first problem with any zone being edited, worded for the UI */
  const zoneIssue =
    zones
      .map((zone) => zoneProblem(zone, surface.width, surface.height))
      .find((issue) => issue) ?? null;
  const zoneValid = surfaceKnown && countOk && !zoneIssue;
  /** are the zones the backend monitors still usable on the camera we have now? */
  const savedZoneUsable =
    savedZones.length > 0 &&
    savedZones.every(
      (zone) =>
        !shapeProblem(zone) && (!surfaceKnown || !zoneProblem(zone, surface.width, surface.height))
    );
  /** what starting would send that the API is bound to refuse, worded for the
   *  button itself. /start sends the saved zones, so they are the ones that
   *  matter, and the ones on screen are included because deleting a zone must
   *  be able to unblock the camera. */
  const startZoneBlock =
    [...savedZones, ...zones]
      .map((zone) => shapeProblem(zone))
      .find((issue) => issue) ?? "";

  // --- drawing ---------------------------------------------------------
  /** remembers the zones as they were, so Ctrl+Z can step back through edits */
  const remember = useCallback(() => {
    setHistory((previous) => [...previous.slice(-19), copyZones(zones)]);
  }, [zones]);

  /** A newly drawn zone starts from the global defaults, so it behaves exactly
   *  like the old single zone until somebody gives it rules of its own. */
  const addDefaultSpec = useCallback(() => {
    setZoneSpecs((list) => [
      ...list,
      {
        name: `Zone ${list.length + 1}`,
        allowed_people: allowed,
        dwell_seconds: dwellSeconds,
        enabled: true,
        using_defaults: true,
      },
    ]);
  }, [allowed, dwellSeconds]);

  /** Renaming a zone, finished by Enter or by clicking away.
   *
   *  An empty name is not an error and is not silently thrown away: it falls
   *  back to the positional name, because a zone with no name at all would have
   *  nothing to show on the picture and nothing to say in an alert. Escape puts
   *  the old name back instead. */
  const commitRename = useCallback(() => {
    if (renaming === null) return;
    const typed = renameDraft.trim();
    setZoneSpecs((list) =>
      list.map((current, i) =>
        i === renaming ? { ...current, name: typed || `Zone ${renaming + 1}` } : current
      )
    );
    setRenaming(null);
    setRenameDraft("");
    setNotice(null);
  }, [renaming, renameDraft]);

  /** commits clicked-out corners as a new zone. The 4-point tool passes its
   *  four corners explicitly, the Polygon tool closes its open draft;
   *  validation failures keep the draft on screen so
   *  no click is lost: too few corners, a self-crossing bow-tie or a sliver
   *  too small to monitor are all reported, not discarded. */
  const closePolygon = useCallback(
    (explicit?: Point[]) => {
      const source = explicit ?? draft;
      if (source.length < MIN_POINTS) {
        setError(`A danger zone needs at least ${MIN_POINTS} points - keep clicking to add more`);
        return;
      }
      // A shaky hand lands its presses a few pixels apart; shedding the
      // near-duplicate here keeps a jittered 4th corner from becoming a
      // spike that trips the self-crossing check.
      const deduped = dropNearRepeats(source, DUP_RADIUS * frameScale);
      // Cleaning first matters: clicks that landed on the same spot collapse
      // into one corner, so a shape can be left with fewer than three even
      // though three or more clicks were made. Committing that would put a one
      // or two point "zone" on screen and the next save would refuse it.
      const points = clampPolygon(deduped, surface.width, surface.height);
      if (points.length < MIN_POINTS) {
        setError(
          `Those ${source.length} clicks made ${points.length} corner${
            points.length === 1 ? "" : "s"
          } - a danger zone needs ${MIN_POINTS} different corners`
        );
        return;
      }
      const shape = shapeProblem(points);
      if (shape) {
        setError(shape);
        return;
      }
      const sized = zoneProblem(points, surface.width, surface.height);
      if (sized) {
        setError(sized);
        return;
      }
      setDraft([]);
      setCursor(null);
      remember();
      setZones((list) => [...list, points]);
      addDefaultSpec();
      setSelected(zones.length);
      setNotice(null);
      setError(null);
    },
    [draft, zones.length, remember, surface.width, surface.height, addDefaultSpec, frameScale]
  );

  const undo = useCallback(() => {
    setHistory((previous) => {
      const last = previous[previous.length - 1];
      if (!last) return previous;
      setZones(copyZones(last));
      setSelected(null);
      setSelectedPoint(null);
      return previous.slice(0, -1);
    });
    setNotice(null);
  }, []);

  const removePoint = useCallback(
    (zoneIndex: number, index: number) => {
      if (zones[zoneIndex]?.length <= MIN_POINTS) {
        setNotice(`A danger zone needs at least ${MIN_POINTS} points`);
        return;
      }
      remember();
      setZones((current) =>
        current.map((zone, i) => (i === zoneIndex ? zone.filter((_, j) => j !== index) : zone))
      );
      // the zone stays selected with no corner picked, so a second Delete takes
      // the whole shape away
      setSelected(zoneIndex);
      setSelectedPoint(null);
      setNotice(null);
    },
    [zones, remember]
  );

  /** removes the whole selected shape, which is what Delete does in select mode */
  const removeZone = useCallback(
    (zoneIndex: number) => {
      remember();
      setZones((current) => current.filter((_, i) => i !== zoneIndex));
      // the rule list is index aligned with the shapes, so removing a shape has
      // to remove the rule that belonged to it. Leaving it would hand the next
      // zone somebody else's name and allowance.
      setZoneSpecs((current) => current.filter((_, i) => i !== zoneIndex));
      setSelected(null);
      setSelectedPoint(null);
      setNotice(null);
    },
    [remember]
  );

// pointer dragging: a new shape, a corner of one, or the whole body
  useEffect(() => {
    if (!drag) return;
    // kept outside the state so mouseup can read the last position without
    // waiting for another render
    let tip: Point | null = null;
    const move = (event: MouseEvent) => {
      const point = toFrame(event.clientX, event.clientY);
      if (!point) return;
      const anchor = dragAnchor(drag);
      if (anchor && movedFar(anchor, point)) movedRef.current = true;
      if (drag.kind === "handle") {
        const next = clampPoint(point, surface.width, surface.height);
        setZones((current) =>
          current.map((zone, i) =>
            i === drag.zone ? zone.map((existing, index) => (index === drag.index ? next : existing)) : zone
          )
        );
      } else if (drag.kind === "move") {
        // measured from the shape captured at mousedown, not from the shape
        // now on screen: using the on-screen shape would add the same delta
        // again on every event and the zone would run away from the pointer
        const dx = point[0] - drag.start[0];
        const dy = point[1] - drag.start[1];
        setZones((current) =>
          current.map((zone, i) =>
            i === drag.zone ? shiftZone(drag.origin, dx, dy, surface.width, surface.height) : zone
          )
        );
      } else if (drag.kind === "freehand") {
        // every sample is clamped and rounded, so a stroke that leaves the
        // picture cannot create a point outside the frame. The stroke is
        // mirrored into state so the overlay paints it live; without a state
        // update nothing re-renders mid-drag and the stroke stays invisible
        // until release.
        strokeRef.current.push(clampPoint(point, surface.width, surface.height));
        setStrokePreview([...strokeRef.current]);
      } else {
        tip = point;
        setCursor(point);
      }
    };
    const up = () => {
      if (drag.kind === "rectangle" || drag.kind === "circle") {
        const end = tip ?? drag.origin;
        const w = Math.abs(end[0] - drag.origin[0]);
        const h = Math.abs(end[1] - drag.origin[1]);
        // a rectangle needs both sides; a circle needs either one, because a
        // perfectly horizontal drag is how you draw a circle
        const tooSmall =
          drag.kind === "rectangle" ? w < 4 || h < 4 : Math.max(w, h) < 4;
        if (tooSmall) {
          // that was a plain click on the picture, not a shape
          setError(null);
        } else {
          const shape =
            drag.kind === "rectangle"
              ? rectangleFrom(drag.origin, end, surface.width, surface.height)
              : ellipseFromDrag(drag.origin, end, surface.width, surface.height);
          // the same guard as the click-out tool: nothing that cannot be a zone
          // is allowed to become one
          if (shape.length < MIN_POINTS) {
            setError(`A danger zone needs at least ${MIN_POINTS} points`);
          } else {
            remember();
            setZones((current) => [...current, shape]);
            addDefaultSpec();
            setSelected(zones.length);
            setSelectedPoint(null);
            setNotice(null);
            setError(null);
          }
        }
      } else if (drag.kind === "freehand") {
        const { polygon, problem } = freehandZone(strokeRef.current, surface.width, surface.height);
        strokeRef.current = [];
        setStrokePreview(null);
        if (polygon.length >= MIN_POINTS) {
          remember();
          setZones((current) => [...current, polygon]);
          addDefaultSpec();
          setSelected(zones.length);
          setSelectedPoint(null);
          setNotice(null);
          setError(null);
        } else if (problem) {
          // a real stroke that cannot be a zone: say why instead of dropping
          // it silently, or the operator redraws the same scribble forever
          setError(problem);
        } else {
          // that was a plain click on the picture, not a shape
          setError(null);
        }
      }
      setDrag(null);
      setCursor(null);
    };
    window.addEventListener("mousemove", move);
    window.addEventListener("mouseup", up);
    return () => {
      window.removeEventListener("mousemove", move);
      window.removeEventListener("mouseup", up);
    };
  }, [
    drag,
    toFrame,
    surface.width,
    surface.height,
    remember,
    zones.length,
  ]);

  /** press on the picture: pick a shape to edit, or start a new one */
  const onPictureDown = (event: React.MouseEvent) => {
    movedRef.current = false;
    if (!drawing) return;
    const point = toFrame(event.clientX, event.clientY);
    if (!point) return;

    if (tool === "select") {
      // nearest zone first, so overlapping zones still select the closest
      let best = -1;
      let bestDistance = SELECT_RADIUS / frameScale;
      zones.forEach((zone, index) => {
        if (zone.length < MIN_POINTS) return;
        // a click well inside a zone counts too, so a big shape is easy to grab
        const distance = pointInZone(point, zone) ? 0 : distanceToZone(point, zone);
        if (distance <= bestDistance) {
          best = index;
          bestDistance = distance;
        }
      });
      if (best === -1) {
        setSelected(null);
        setSelectedPoint(null);
        return;
      }
      remember();
      setSelected(best);
      setSelectedPoint(null);
      setDrag({ kind: "move", zone: best, start: point, origin: copyZones([zones[best]])[0] });
      return;
    }
    if (tool === "quad" || tool === "polygon") return;
    strokeRef.current = [];
    setStrokePreview(null);
    setDrag(
      tool === "freehand"
        ? { kind: "freehand" }
        : tool === "circle"
          ? { kind: "circle", origin: point }
          : { kind: "rectangle", origin: point }
    );
  };

  /** click with the 4-point tool: each click places one corner, and the 4th
   *  valid click creates the zone by itself - nothing to close, no second
   *  gesture to learn. A refused 4th corner (too close to the last one, or
   *  a self-crossing shape) leaves the first three on screen, so the next
   *  click is simply a different 4th corner.
   *
   *  The Polygon tool below is the DMS-style alternative: every click adds
   *  one corner (3-32) and the shape is closed on demand - click near the
   *  first corner, double-click, press Enter, or press Finish zone. */
  const onPictureClick = (event: React.MouseEvent) => {
    if (!drawing || (tool !== "quad" && tool !== "polygon")) return;
    const point = toFrame(event.clientX, event.clientY);
    if (!point || movedRef.current) return;
    const next = clampPoint(point, surface.width, surface.height);

    // A click within jitter of the corner just placed adds nothing. Without
    // this, a shaky hand piles up near-invisible corners that collapse into
    // a spike and trip the self-crossing check on close.
    const last = draft[draft.length - 1];
    if (
      last &&
      Math.hypot(last[0] - next[0], last[1] - next[1]) <= DUP_RADIUS * frameScale
    ) {
      setNotice("That is the same corner as the last click - click somewhere else");
      return;
    }
    if (tool === "polygon") {
      // DMS-style close gesture: clicking back near the first corner closes
      // the ring instead of adding a duplicate of it.
      const first = draft[0];
      if (
        first &&
        draft.length >= MIN_POINTS &&
        Math.hypot(first[0] - next[0], first[1] - next[1]) <= CLOSE_RADIUS * frameScale
      ) {
        closePolygon();
        return;
      }
      const corners = [...draft, next];
      if (corners.length >= MAX_POINTS) {
        // at the ceiling there is nowhere left to click, so the cap itself
        // finishes the shape rather than swallowing further corners
        closePolygon(corners);
        return;
      }
      setError(null);
      setNotice(
        corners.length >= MIN_POINTS
          ? `Corner ${corners.length} placed - keep clicking, then Finish zone or press Enter`
          : null
      );
      setDraft(corners);
      setSelected(null);
      setSelectedPoint(null);
      return;
    }
    const corners = [...draft, next];
    if (corners.length < QUAD_POINTS) {
      setError(null);
      setNotice(null);
      setDraft(corners);
      setSelected(null);
      setSelectedPoint(null);
      return;
    }
    closePolygon(corners);
  };

  /** double-click finishes an open Polygon draft, the same as Enter or Finish
   *  zone. The second click of the double-click lands on the same spot, so
   *  the jitter guard above refuses it and the draft is closed as it stands. */
  const onPictureDoubleClick = () => {
    if (!drawing || tool !== "polygon" || draft.length === 0) return;
    closePolygon();
  };

  /** drag of an existing corner */
  const onHandleDown = (zoneIndex: number, index: number) => (event: React.MouseEvent) => {
    event.stopPropagation();
    movedRef.current = false;
    if (!drawing) return;
    remember();
    setSelected(zoneIndex);
    setSelectedPoint(index);
    setDrag({ kind: "handle", zone: zoneIndex, index });
  };

  /** click on a "+" between two corners adds one there */
  const onEdgeClick = (zoneIndex: number, edge: number) => (event: React.MouseEvent) => {
    event.stopPropagation();
    if (!drawing) return;
    const zone = zones[zoneIndex];
    if (!zone || zone.length >= MAX_POINTS) {
      setError(`A zone can have at most ${MAX_POINTS} points - remove one or press Undo`);
      return;
    }
remember();
      setZones((current) =>
        current.map((existing, i) => {
          if (i !== zoneIndex) return existing;
          const after = existing[(edge + 1) % existing.length];
          return [...existing.slice(0, edge + 1), midpoint(existing[edge], after), ...existing.slice(edge + 1)];
        })
      );
  };

  // keyboard: Escape cancels, Ctrl+Z steps back, Delete removes. Enter
  // finishes an open Polygon draft (the 4-point tool finishes by itself on
  // the 4th corner, so there is no gesture left for Enter to trigger there).
  useEffect(() => {
    if (!drawing) return;
    const onKey = (event: KeyboardEvent) => {
      const key = event.key.toLowerCase();
      if ((event.ctrlKey || event.metaKey) && key === "z") {
        event.preventDefault();
        undo();
        return;
      }
      if (key === "escape") {
        // Escape gives up whatever is half done right now - an open polygon,
        // a drag in progress, a rubber band - but keeps the zones already
        // finished, so a mis-click never costs work that was already done
        setDraft([]);
        setSelected(null);
        setSelectedPoint(null);
        setDrag(null);
        setCursor(null);
        strokeRef.current = [];
        setStrokePreview(null);
        setNotice("Drawing cancelled");
        return;
      }
      if (key === "enter") {
        // Enter closes the Polygon draft the way double-click, Finish zone,
        // or clicking back near the first corner does. Too few corners is
        // reported by closePolygon, not swallowed.
        if (tool === "polygon" && draft.length > 0) {
          event.preventDefault();
          closePolygon();
        }
        return;
      }
      if (key !== "delete" && key !== "backspace") return;
      const tag = (event.target as HTMLElement | null)?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
      // Backspace with an open click-drawn draft drops its last corner first,
      // so a mis-click is undone without touching the finished zones.
      if (key === "backspace" && (tool === "quad" || tool === "polygon") && draft.length > 0) {
        event.preventDefault();
        setDraft((current) => current.slice(0, -1));
        return;
      }
      if (selected === null) return;
      event.preventDefault();
      // a selected corner goes first; with no corner the shape itself goes
      if (selectedPoint !== null) removePoint(selected, selectedPoint);
      else removeZone(selected);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [
    drawing,
    tool,
    draft.length,
    selected,
    selectedPoint,
    undo,
    removePoint,
    removeZone,
  ]);

  const clearZones = () => {
    remember();
    setZones([]);
    setZoneSpecs([]);
    setDraft([]);
    setSelected(null);
    setSelectedPoint(null);
    setNotice("All zones cleared - draw at least 3 points, then Save zone");
  };

  /** type-safe: a half-typed number in the coordinates table is simply skipped */
  const editCoordinate = (zoneIndex: number, index: number, axis: 0 | 1, raw: string) => {
    const value = Number(raw);
    if (!Number.isFinite(value)) return;
    // a typed coordinate is pulled onto the frame like a dragged one, so the
    // table can never put a corner off screen
    const limit = axis === 0 ? surface.width : surface.height;
    let next = Math.max(0, Math.round(value));
    if (limit > 0) next = Math.min(limit, next);
    setZones((current) =>
      current.map((zone, i) =>
        i === zoneIndex
          ? zone.map((point, j) =>
              j === index ? ([axis === 0 ? next : point[0], axis === 1 ? next : point[1]] as Point) : point
            )
          : zone
      )
    );
    setSelected(zoneIndex);
    setSelectedPoint(index);
    setNotice(null);
  };

  // --- payload + actions -------------------------------------------------
  const payload = useCallback(
    (list?: Point[][]): ZonePayload => {
      const shapes = list ?? zones;
      return {
        zones: shapes.map((zone) => clampPolygon(zone, surface.width, surface.height)),
        // index aligned with the shapes: a zone that is not there cannot have
        // somebody else's rules attached to it
        zone_specs: shapes.map((_, index) => {
          const spec = zoneSpecs[index];
          const name = spec?.name?.trim() || `Zone ${index + 1}`;
          // an empty name is never sent as empty: the zone would come back as
          // "Zone N" anyway, and sending "" would look like an emptied name
          return {
            name,
            allowed_people: spec?.allowed_people ?? allowed,
            dwell_seconds: spec?.dwell_seconds ?? dwellSeconds,
            enabled: spec?.enabled ?? true,
            using_defaults: spec?.using_defaults ?? true,
          };
        }),
        allowed_people: allowed,
        source,
        machine_name: machineName,
        mode,
        dwell_seconds: dwellSeconds,
        repeat_alert_seconds: repeatAlertSeconds,
        anchor,
        // so the same zone can be scaled onto a camera with another resolution
        frame_width: surface.width,
        frame_height: surface.height,
      };
    },
    [
      zones,
      zoneSpecs,
      allowed,
      source,
      machineName,
      mode,
      dwellSeconds,
      repeatAlertSeconds,
      anchor,
      surface.width,
      surface.height,
    ]
  );

  /** Every zone has to be usable, not just the first one: a shape that cannot be
   *  monitored is reported instead of being quietly dropped. */
  const firstProblem =
    zones
      .map((zone) => zoneProblem(zone, surface.width, surface.height))
      .find((issue) => issue) ?? null;

  /** `explicit` lets "remove all zones" save an empty list without going through
   *  a state update that has not rendered yet. Saving while stopped only saves:
   *  there is no camera to hand the zones to, so monitoring is left off. */
  const saveZone = async (explicit?: Point[][]) => {
    if (!surfaceKnown) {
      setError("Start the camera first");
      return;
    }
    const list = explicit ?? zones;
    if (explicit === undefined && firstProblem) {
      setError(firstProblem);
      return;
    }
    setBusy(true);
    setError(null);
    try {
      const body = payload(list);
      // an empty zone list is a real, wanted outcome: it takes the saved zones
      // off the camera and leaves monitoring running with no danger zone
      await saveConfig(body);
      if (running) await startMonitoring(body);
      setSavedZones(copyZones(list));
      // the dropdown counts are saved counts: this source's is now exact
      zoneCountsRef.current = { ...zoneCountsRef.current, [source]: list.length };
      setZoneCounts(zoneCountsRef.current);
      // Save is not a cancel: nothing is being thrown away, so it goes straight
      // back to Live and the editor state is dropped here rather than through
      // the cancel path that would ask about work just saved.
      setEditing(false);
      setConfirmingCancel(false);
      setConfirmingRemove(false);
      setDrag(null);
      setDraft([]);
      setSelected(null);
      setSelectedPoint(null);
      setHistory([]);
      setTool(null);
      setNotice(list.length === 0 ? ZONES_REMOVED : ZONE_SAVED);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  };

  /** The settings on their own, so a change to them never touches the zones.
   *  `zones` is left out on purpose: /config keeps the polygons it already has
   *  when the field is not sent, so a half-finished shape is never saved by a
   *  change to the alert rules. */
  const settingsBody = useCallback(
    (): ZonePayload =>
      ({
        allowed_people: allowed,
        source,
        machine_name: machineName,
        mode,
        dwell_seconds: dwellSeconds,
        repeat_alert_seconds: repeatAlertSeconds,
        anchor,
        confidence,
        zones: undefined,
        fire_smoke: { enabled: fireEnabled || smokeEnabled, analysis_fps: fireFps },
        fire: { enabled: fireEnabled },
        smoke: { enabled: smokeEnabled },
        crowd: { enabled: crowdEnabled, analysis_fps: crowdFps },
        ppe: { enabled: ppeEnabled, analysis_fps: ppeFps },
        hazard: { enabled: hazardEnabled },
        loop_video: loopVideo,
      }) as unknown as ZonePayload,
    [allowed, source, machineName, mode, dwellSeconds, repeatAlertSeconds, anchor, confidence,
      fireEnabled, smokeEnabled, fireFps, crowdEnabled, crowdFps, ppeEnabled, ppeFps, hazardEnabled,
      loopVideo]
  );

  /** Settings are saved without being pressed: half a second of quiet, then
   *  the backend has the new values and the bar says so. */
  const [settingsApplied, setSettingsApplied] = useState(false);
  /** the settings are not the user's until /config has answered: what arrives
   *  with the page is what is already saved */
  const [settingsLoaded, setSettingsLoaded] = useState(false);
  /** armed once the values that came from the backend have been seen, so the
   *  load itself is not mistaken for the user changing something */
  const settingsArmed = useRef(false);
  /** the last settings save that was refused, so a later good save can take
   *  that message away without touching anyone else's */
  const failed = useRef<string | null>(null);
  /** what the backend last accepted, as the settings body was sent. A change
   *  that changes nothing sends nothing, and a change to the source ALONE - the
   *  Camera source field is not a settings edit - is sent as a source switch of
   *  its own instead of riding along in a settings write. */
  const lastSent = useRef<string | null>(null);
  useEffect(() => {
    // opening the app must not write to the database or say "Applied"
    if (!settingsLoaded) return;
    if (!settingsArmed.current) {
      settingsArmed.current = true;
      lastSent.current = JSON.stringify(settingsBody());
      return;
    }
    const body = settingsBody();
    const serialized = JSON.stringify(body);
    // nothing actually changed - a switch that was answered with Cancel lands
    // here - so there is nothing to send and nothing to report
    if (lastSent.current === serialized) return;
    const { source: nextSource, ...current } = body;
    const { source: _sentSource, ...sent } = JSON.parse(lastSent.current ?? "{}") as ZonePayload;
    const sourceOnly =
      lastSent.current !== null && JSON.stringify(current) === JSON.stringify(sent);
    setSettingsApplied(false);
    const timer = window.setTimeout(() => {
      void (async () => {
        try {
          if (sourceOnly && typeof nextSource === "string") {
            await switchSource(nextSource);
          } else {
            await saveConfig(body);
          }
          lastSent.current = serialized;
          setSettingsApplied(true);
          // an older failed settings save is now stale, but any other message
          // (a saved zone, a running camera) must stay on screen
          if (failed.current && toast?.text === failed.current) setNotice(null);
          failed.current = null;
        } catch (err) {
          setSettingsApplied(false);
          failed.current = (err as Error).message;
          setError(failed.current);
        }
      })();
    }, SETTINGS_APPLY_MS);
    return () => window.clearTimeout(timer);
  }, [settingsBody, settingsLoaded]);

  // --- a different source shows its own zones, never the last one's --------
  // Zones are stored per source on the backend. Picking a new source swaps in
  // that source's saved zones; a source with none gets a clean editor and a
  // prompt, rather than the previous source's shapes squeezed onto a frame they
  // were never drawn for. The mount already loaded the first source, so it is
  // skipped here.
  useEffect(() => {
    if (!settingsLoaded) return;
    if (loadedSource.current === null) {
      loadedSource.current = source;
      return;
    }
    if (loadedSource.current === source) return;
    // a camera index or an uploaded clip resolves to a stored source; a
    // half-typed file path has no saved zones to fetch yet, and the source in
    // use stays the one on file until the field holds a real one
    const known = /^\d+$/.test(source) || videos.some((video) => samePath(video.source, source));
    if (!known) return;
    // The other source's zones are about to replace what is on screen, so work
    // the backend has not been told about is the operator's call to make: the
    // switch is held, the field goes back to the source in use, and Discard is
    // what lets it through.
    if (geometryChanged(zones, savedZones) || draft.length > 0) {
      setPendingSource(source);
      setConfirmingSwitch(true);
      setSource(loadedSource.current);
      return;
    }
    loadedSource.current = source;
    let live = true;
    getConfig(source)
      .then((cfg) => {
        if (!live) return;
        const next = configZonesOf(cfg);
        setZoneReference(cfg.zone_reference);
        setZones(copyZones(next));
        setSavedZones(copyZones(next));
        setZoneSpecs(specsFor(next.length, cfg.zone_specs, { allowed, dwell: dwellSeconds }));
        setSelected(null);
        setSelectedPoint(null);
        setDraft([]);
        setDrag(null);
        setNotice(next.length > 0 ? null : ZONES_FOR_SOURCE);
        // a deliberate pick while running moves the feed now that its zones
        // are on screen; anything else only filed the source
        if (restartAfterLoad.current !== null && restartAfterLoad.current === source) {
          restartAfterLoad.current = null;
          const specs = specsFor(next.length, cfg.zone_specs, {
            allowed,
            dwell: dwellSeconds,
          });
          const hit = videos.find((video) => samePath(video.source, source));
          const label = hit
            ? hit.name
            : /^\d+$/.test(source)
              ? `Webcam ${source}`
              : source;
          void startFeedOnZones(next, specs, source, `Playing ${label}`);
        }
      })
      .catch((err) => {
        if (live) setError((err as Error).message);
      });
    return () => {
      live = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [source, settingsLoaded, videos]);

  /** Discard: the editor goes back to what the backend already has, which is
   *  what makes the switch that was waiting for this answer safe to let go. A
   *  running feed then still has to move, which the zone-swap effect does once
   *  the new zones have loaded. */
  const discardUnsavedAndSwitch = () => {
    const next = pendingSource;
    setZones(copyZones(savedZones));
    setDraft([]);
    setConfirmingSwitch(false);
    setPendingSource(null);
    setError(null);
    if (next !== null) {
      if (running) restartAfterLoad.current = next;
      setSource(next);
    }
  };

  /** Cancel: nothing is thrown away and the source in use stays in use. */
  const cancelSourceSwitch = () => {
    restartAfterLoad.current = null;
    setConfirmingSwitch(false);
    setPendingSource(null);
    setError(null);
  };

  /** Step 2 of removing every zone. Works from any state: it empties the editor
   *  first, so the saved list is emptied whether or not the editor was open. */
  const removeAllZones = async () => {
    remember();
    setZones([]);
    setZoneSpecs([]);
    setSelected(null);
    setSelectedPoint(null);
    setDraft([]);
    setDrag(null);
    setTool("select");
    await saveZone([]);
  };

  /** Step 1. The camera does not need a zone to run: people are boxed straight
   *  away, so this starts the feed with the saved zones when they still fit, and
   *  with no zone at all when they do not. */
  const start = async () => {
    // the one user gesture a browser accepts as permission to make sound, so
    // the alarm can beep later without another click
    openAudio();
    const body: ZonePayload = {
      ...payload(),
      zones: savedZoneUsable
        ? savedZones.map((zone) => clampPolygon(zone, surface.width, surface.height))
        : [],
      // the on-screen Loop box wins over the stored switch without waiting
      // for the debounced save still in flight
      loop: loopVideo,
    };
    setBusy(true);
    setError(null);
    // Loading the detector takes seconds: say so at once instead of leaving
    // a dead button while the backend works.
    setNotice("Starting camera…");
    try {
      await startMonitoring(body);
      streamVersion.current += 1;
      setStreamRetry(0);
      setNotice("Monitoring started");
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const stop = async () => {
    setBusy(true);
    try {
      await stopMonitoring();
      setNotice("Monitoring stopped");
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  };

  /** True when the Camera source field points at one of the uploaded clips. The
   *  list is the backend's, so the badge and the choice cannot disagree with it. */
  const usingTestVideo =
    videos.some((video) => samePath(video.source, source)) ||
    // the backend flag is a fallback only when the list never loaded, so the
    // badge cannot hang around after the field is switched back to the webcam
    (videos.length === 0 && (state?.test_video ?? false));

  // a saved test video opens the Source card on the Test video half, so the
  // operator sees the clip that will play instead of an empty camera panel
  useEffect(() => {
    if (usingTestVideo) setSourceMode("video");
  }, [usingTestVideo]);

  /** True when the saved zones were drawn for a frame size this clip is not. */
  const frameMismatch = (video: TestVideo) => {
    const refW = zoneReference?.[0] ?? 0;
    const refH = zoneReference?.[1] ?? 0;
    return savedZones.length > 0 && refW > 0 && refH > 0 && (refW !== video.width || refH !== video.height);
  };

  /** Open the live-camera half and make sure the field holds a camera index. */
  /** A source switch answered with Discard still has to move a running feed:
   *  the field change alone only re-files settings. Set before setSource, read
   *  (and cleared) by the zone-swap effect once the new zones have loaded. */
  const restartAfterLoad = useRef<string | null>(null);

  /** Start the feed on explicit zones, used when the source itself changes
   *  under a running camera. Unlike start(), nothing is read from the editor
   *  state: the zones and rules come from the source just loaded. */
  const startFeedOnZones = async (
    list: Point[][],
    listSpecs: ZoneSpec[],
    src: string,
    notice: string
  ) => {
    setBusy(true);
    setError(null);
    try {
      const body: ZonePayload = {
        ...payload(),
        // The zones belong to the NEW source and are scaled by the backend
        // from their saved reference: clamping them to the OLD frame here
        // squeezes a 1920-wide drawing into a 640-wide box before it is sent.
        zones: list,
        zone_specs: list.map((_, index) => {
          const spec = listSpecs[index];
          return {
            name: spec?.name?.trim() || `Zone ${index + 1}`,
            allowed_people: spec?.allowed_people ?? allowed,
            dwell_seconds: spec?.dwell_seconds ?? dwellSeconds,
            enabled: spec?.enabled ?? true,
            using_defaults: spec?.using_defaults ?? true,
          };
        }),
        source: src,
        // same on-screen Loop value as Start: a source switch restarts the
        // feed, so it must honour the box too
        loop: loopVideo,
      };
      await startMonitoring(body);
      streamVersion.current += 1;
      setStreamRetry(0);
      setNotice(notice);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  };

  /** Move a running feed to a newly picked source right now, with that
   *  source's saved zones. While stopped there is no feed to move, so the
   *  pick only files the source and swaps the zones. */
  const restartFeed = async (nextSource: string, notice: string) => {
    setBusy(true);
    setError(null);
    try {
      const cfg = await getConfig(nextSource);
      const next = configZonesOf(cfg);
      const specs = specsFor(next.length, cfg.zone_specs, {
        allowed,
        dwell: dwellSeconds,
      });
      setZoneReference(cfg.zone_reference);
      setZones(copyZones(next));
      setSavedZones(copyZones(next));
      setZoneSpecs(specs);
      setSelected(null);
      setSelectedPoint(null);
      setDraft([]);
      setDrag(null);
      await startFeedOnZones(next, specs, nextSource, notice);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  };

  /** A deliberate source pick (tile or seg button, never a half-typed field):
   *  unsaved editor work gets the usual question, a stopped camera just files
   *  the pick, and a running feed moves to the new clip. */
  const playSource = (nextSource: string, idleNotice: string, playingNotice: string) => {
    if (editing && (unsavedGeometry || draft.length > 0)) {
      setPendingSource(nextSource);
      setConfirmingSwitch(true);
      // The confirm banner lives at the top of the page, far from the Use
      // button just pressed: without this the click looks dead and the next
      // Start opens the webcam again.
      setError("You have unsaved zone changes - click Discard at the top to switch to this video, or Cancel to keep editing.");
      return;
    }
    restartAfterLoad.current = null;
    if (nextSource !== source) setSource(nextSource);
    setError(null);
    if (!running || samePath(state?.source ?? "", nextSource)) {      if (!running) setNotice(idleNotice);
      return;
    }
    void restartFeed(nextSource, playingNotice);
  };

  const chooseLiveCamera = () => {
    setSourceMode("camera");
    const next = /^\d+$/.test(source) ? source : "0";
    playSource(next, "", `Playing webcam ${next}`);
  };

  const selectVideo = (video: TestVideo) => {
    setSourceMode("video");
    playSource(video.source, `Selected ${video.name}`, `Playing ${video.name}`);
  };

  /** Fetch the saved zone counts for sources we have none for yet: the
   *  current source plus every uploaded clip, through the same per-source
   *  /config the editor swaps zones with. One cheap GET each, failures left
   *  unknown rather than guessed. */
  const refreshZoneCounts = useCallback(async (sources: string[]) => {
    const missing = [...new Set(sources)].filter(
      (item) => item !== "" && !(item in zoneCountsRef.current)
    );
    if (missing.length === 0) return;
    const found: Record<string, number> = {};
    await Promise.all(
      missing.map(async (item) => {
        try {
          found[item] = configZonesOf(await getConfig(item)).length;
        } catch {
          /* unknown stays unknown */
        }
      })
    );
    if (Object.keys(found).length === 0) return;
    zoneCountsRef.current = { ...zoneCountsRef.current, ...found };
    setZoneCounts(zoneCountsRef.current);
  }, []);

  /** the saved count for a source, matched the way sources are matched
   *  everywhere (file spelling, not string equality), null while unknown */
  const zoneCountFor = (matchSource: string): number | null => {
    if (matchSource in zoneCounts) return zoneCounts[matchSource];
    const hit = Object.keys(zoneCounts).find((key) => samePath(key, matchSource));
    return hit === undefined ? null : zoneCounts[hit];
  };

  /** Live Monitor source pick: stopped just selects through the shared
   *  handler, but a running feed first asks, then performs the existing
   *  stop-then-start. Edit mode never reaches here (the dropdown is off). */
  const onLiveSourceChange = (value: string) => {
    const current = sourceMode === "camera" ? "camera" : source;
    if (value === current) return;
    if (running) {
      setConfirmLiveSource(value);
      return;
    }
    onCameraSourceChange(value);
  };

  const confirmLiveSourceSwitch = () => {
    const value = confirmLiveSource;
    setConfirmLiveSource(null);
    if (value === null) return;
    if (value === (sourceMode === "camera" ? "camera" : source)) return;
    onCameraSourceChange(value);
  };

  const cancelLiveSourceSwitch = () => {
    setConfirmLiveSource(null);
  };

  // Keep the dropdown counts fresh without ever fetching for a switch that
  // was held: only the clip list drives refetches (upload/delete, plus the
  // first load). A held switch parks its value in pendingSource and puts the
  // field back, so fetching on `source` would ask the backend about a source
  // that was never actually picked. Switching and zone saves keep the cache
  // exact through the code paths that already load those zones.
  const sourceRef = useRef(source);
  sourceRef.current = source;
  useEffect(() => {
    void refreshZoneCounts([sourceRef.current, ...videos.map((video) => video.source)]);
  }, [videos, refreshZoneCounts]);

  const uploadFile = async (file: File) => {
    setUploadPct(0);
    setError(null);
    try {
      const video = await uploadVideo(file, setUploadPct);
      setVideos((current) => [video, ...current]);
      // Selecting is a second step users miss: the upload alone leaves the
      // source on the webcam, so the next Start opens the webcam again.
      // Auto-select the clip just uploaded so Start plays it.
      setSourceMode("video");
      playSource(video.source, `Uploaded and selected ${video.name}`, `Playing ${video.name}`);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setUploadPct(null);
    }
  };

  const onPickFile = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    event.target.value = ""; // allow picking the same file again after a fix
    if (file) void uploadFile(file);
  };

  const onDropFiles = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setDragOver(false);
    const file = event.dataTransfer?.files?.[0];
    if (file) void uploadFile(file);
  };

  const onDeleteVideo = async (video: TestVideo) => {
    setBusy(true);
    try {
      await deleteVideo(video.id);
      setVideos((current) => current.filter((item) => item.id !== video.id));
      if (samePath(video.source, source)) setSource("0");
      setNotice(`Deleted ${video.name}`);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setConfirmingDelete(null);
      setBusy(false);
    }
  };

  /** "Edit zones": opens the editor over the running camera. Monitoring is left
   *  exactly as it is - the backend keeps using the saved zones until Save sends
   *  new ones, so the alarm never stops while somebody redraws a corner. */
  const enterEdit = () => {
    setEditing(true);
    // With saved zones there is something to Select straight away. With none,
    // Select would do nothing, so the tool stays unset and the hint over the
    // picture says what to pick instead of leaving a dead cursor on screen.
    setTool(zones.length > 0 ? "select" : null);
    setSelected(null);
    setSelectedPoint(null);
    setDraft([]);
  };

  // --- derived ----------------------------------------------------------
  /** whether the rules actually use the delay, which the backend decides.
   *  ZoneMonitor._enter() raises the alert the moment somebody walks in when the
   *  mode is INSTANT, and update_all() only ever compares dwell_seconds against
   *  the limit in TIMER mode, so the delay is dead in INSTANT and in LIMIT.
   *  The repeat interval is gated on the same mode. The allowed-people rule is
   *  NOT: the capacity check runs every frame whatever the mode says. */
  const usesDelay = mode === "TIMER";

  const alert = running && (state?.status ?? "SAFE") === "ALERT";
  const inside = state?.inside ?? 0;
  const perf = state?.perf;

  // how long the longest-alarming person has been standing in the zone: that is
  // the number the big red line ends with
  const alertSeconds = (state?.workers ?? [])
    .filter((worker) => worker.state === "ALERT")
    .reduce((longest, worker) => Math.max(longest, worker.dwell_seconds), 0);

  // one beep per alarm, not one per poll
  useEffect(() => {
    if (alert && !wasAlert.current && !muted) beep(audioRef.current);
    wasAlert.current = alert;
  }, [alert, muted]);

  // An empty saved list is a real state, not a broken one: nothing stored and
  // nothing drawn means there is nothing left to apply, so the app is in the
  // edit step and offers to draw again. Without this, "Remove all zones" left
  // the toolbar folded away and the one primary button disabled for good - the
  // only way out was to stop the camera and start it again.
  // ONE source of truth for what the video, the chips and the footer all show.
  // While the editor is open that is exactly the editor's zones: falling back
  // to the saved zones underneath is what made a cleared editor look like
  // nothing had been removed, and left the old zones active.
  const shownZones = editing ? zones : savedZones;

  /** Zone overlay shapes: the saved zones while drawing or while hazard
   *  evaluation runs, nothing while hazard is off (edit mode keeps the
   *  drawing canvas regardless). Display follows the applied backend state,
   *  falling back to the local switch before the first poll answers. */
  const appliedHazard = state?.hazard?.enabled ?? hazardEnabled;
  const appliedFire = state?.fire_smoke?.fire_enabled ?? fireEnabled;
  const appliedSmoke = state?.fire_smoke?.smoke_enabled ?? smokeEnabled;
  const appliedPpe = state?.ppe?.enabled ?? ppeEnabled;
  const hazardOff = running && appliedHazard === false;
  const overlayZones = editing || appliedHazard ? shownZones : [];

  /** people whose anchor point has fallen out of the picture.
   *
   *  The point is the one the zone test uses, so when it is off screen the
   *  person cannot be counted as inside and nothing else says so: their state
   *  simply reads OUTSIDE. The usual cause is Feet on somebody who walks close
   *  to the camera - their bottom edge is cropped off the bottom of the frame. */
  const anchorOffScreen = frameKnown
    ? (state?.workers ?? []).filter(
        (worker) => worker.point !== undefined && isOutOfFrame(worker.point)
      )
    : [];

  // a zone is only drawn over a picture that is actually there: no feed means
  // nothing to place it on, so it is left out entirely. The stopped editor
  // draws on the dark canvas instead, which has the reference size.
  // The overlay does not wait for the first zone. Gating the whole <svg> on
  // there already being a shape meant the very first drag or click-out had
  // nowhere to be drawn while it was being made - no rubber band while dragging,
  // no dashed line while clicking out corners - so the shape only appeared once
  // it had been committed. A picture and its size are all the overlay needs;
  // which shapes sit inside it is decided below.
  const overlayVisible = (running || editing) && surfaceKnown;
  // streamRetry busts the browser cache after a broken MJPEG connection so the
  // <img> reconnects on its own instead of sitting on a dead frame.
  const pictureUrl = `/stream?v=${streamVersion.current}&r=${streamRetry}`;

  /** Reconnect a broken MJPEG stream once, shortly after it fails. Guarded by
   *  `running` so a stopped camera never retries itself back to life, and by a
   *  single pending timer so a flapping connection cannot stack retries. */
  const handleStreamError = useCallback(() => {
    if (!runningRef.current) return;
    if (streamRetryTimer.current !== null) return;
    streamRetryTimer.current = window.setTimeout(() => {
      streamRetryTimer.current = null;
      if (runningRef.current) setStreamRetry((n) => n + 1);
    }, 2000);
  }, []);

  /** The three things this screen can be, as one value. `ruleMode` is the
   *  backend's TIMER / INSTANT / LIMIT setting, so the two are kept apart by name.
   *
   *  A monitoring product has no steps to walk through: the camera is either off,
   *  running, or being drawn on - live or on the stopped canvas. Everything on
   *  screen is decided from which of the three it is, so the picture, the tools
   *  and the status can never disagree about it. */
  const screenMode = !running && !editing ? "stopped" : editing ? "edit" : "live";

  /** true while the editor is holding zones the backend has not been told about.
   *
   *  Geometry only. Picking a tool or selecting a corner has changed nothing
   *  that would be lost, and nagging about those would train the operator to
   *  dismiss the prompt without reading it. The settings fields are not counted
   *  either: they save themselves the moment they change, so there is nothing
   *  pending to warn about. */
  const geometryChanged = useCallback(
    (list: ZoneList, saved: ZoneList): boolean => {
      if (list.length !== saved.length) return true;
      return list.some(
        (zone, index) =>
          zone.length !== saved[index].length ||
          zone.some(
            (point, corner) =>
              point[0] !== saved[index][corner][0] || point[1] !== saved[index][corner][1]
          )
      );
    },
    []
  );
  const unsavedGeometry = screenMode === "edit" && geometryChanged(zones, savedZones);

  /** "Monitoring since 14:32, uptime 1h 06m" - only when this page watched the
   *  camera start. Without a witnessed start the real time is unknown, so the
   *  words are left off rather than filled in with a guess. */
  const sinceLine = (() => {
    // No witnessed start means no honest clock, so nothing is shown rather
    // than a time measured from when this page happened to open.
    if (!running || since === null) return null;
    const now = Date.now();
    // 24 hour, and zero padded: an operator reading this off a screen and
    // writing it in a log must not have to guess whether it is morning.
    const clock = new Date(since).toLocaleTimeString("en-GB", {
      hour: "2-digit",
      minute: "2-digit",
      hour12: false,
    });
    const minutes = Math.max(0, Math.floor((now - since) / 60000));
    return { clock, hours: Math.floor(minutes / 60), minutes: minutes % 60 };
  })();

  const perfValue = (value: number | undefined, suffix = "") =>
    value === undefined ? "-" : `${value.toFixed(1)}${suffix}`;

  /** "5h 22m 18s" of monitoring this session, or null while there is nothing
   *  to time. Like sinceLine it is read at render, so it moves on with the
   *  state polls while the camera runs. */
  const uptimeText = (() => {
    if (!running || since === null) return null;
    const seconds = Math.max(0, Math.floor((Date.now() - since) / 1000));
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    return `${hours}h ${minutes}m ${seconds % 60}s`;
  })();

  /** System Status card: the four live numbers, each with its own dot. Green
   *  means the value is live, grey means idle or unknown — never invented. */
  const systemStatus = {
    fps: perfValue(perf?.fps),
    ms: perfValue(perf?.ms_per_frame),
    uptime: uptimeText,
    running,
  };

  /** The machine-wide rules, in one place: the settings page and nowhere else.
   *  No new settings are added here.
   *
   *  It used to be rendered twice - here and inside an expanded zone card - so
   *  a card and the page each carried a "people allowed" number that had to be
   *  kept in step by hand. A zone's own rules now live on the card
   *  (`zoneRuleFields`), and what is left here is about the machine: the alert
   *  mode, the repeat interval, the anchor, and the allowance every zone
   *  follows unless it has one of its own. */
  /** The machine-wide defaults for new zones: alert mode, delays, allowance
   *  and where on a person the detector looks. Saved automatically, with the
   *  Applied line saying when the backend has them. */
  /** Side detection paths card (Settings): one switch per system plus a
   *  cadence where the system has one of its own. Fire and Smoke are
   *  separate switches on one shared pass, so the fps field sits on the
   *  Fire row and says so; every switch writes the same stored setting the
   *  Live Monitor panel toggles. Saving rides the existing debounced
   *  settingsBody, so toggling a switch stores it and applies it live to a
   *  running camera without a restart. */
  const setPathEnabled = (key: DetectionPathRow["key"], enabled: boolean) => {
    if (key === "hazard") setHazardEnabled(enabled);
    else if (key === "fire") setFireEnabled(enabled);
    else if (key === "smoke") setSmokeEnabled(enabled);
    else if (key === "crowd") setCrowdEnabled(enabled);
    else setPpeEnabled(enabled);
  };
  const setPathFps = (key: DetectionPathRow["key"], fps: number) => {
    if (key === "fire") setFireFps(fps);
    else if (key === "crowd") setCrowdFps(fps);
    else if (key === "ppe") setPpeFps(fps);
  };
  /** Settings "Reset to defaults" button: put every detection system back on
   *  its shipped value (hazard on, the side models off at their operating
   *  cadences). The store is left alone - the change rides the same debounced
   *  settings save as a manual toggle, so it persists and applies live to a
   *  running camera without a restart. */
  const resetDetectionPaths = useCallback(() => {
    setHazardEnabled(DETECTION_DEFAULTS.hazard);
    setFireEnabled(DETECTION_DEFAULTS.fire);
    setSmokeEnabled(DETECTION_DEFAULTS.smoke);
    setFireFps(DETECTION_DEFAULTS.fireFps);
    setCrowdEnabled(DETECTION_DEFAULTS.crowd);
    setCrowdFps(DETECTION_DEFAULTS.crowdFps);
    setPpeEnabled(DETECTION_DEFAULTS.ppe);
    setPpeFps(DETECTION_DEFAULTS.ppeFps);
  }, []);
  const detectionPaths = (
    <DetectionPathsCard
      model={{
        rows: [
          {
            key: "hazard",
            title: "Hazard zone",
            blurb: "Watches the danger zones. Switching off stops every zone alert.",
            enabled: hazardEnabled,
          },
          {
            key: "fire",
            title: "Fire",
            blurb: "Alerts on flames after a short persistence check.",
            enabled: fireEnabled,
            fps: fireFps,
            fpsCaption: "shared with Smoke",
            gateSeconds: { fire: fireSeconds, smoke: smokeSeconds },
          },
          {
            key: "smoke",
            title: "Smoke",
            blurb: "Alerts on smoke after a short persistence check.",
            enabled: smokeEnabled,
          },
          {
            key: "crowd",
            title: "Crowd",
            blurb: "Counts people from the person tracker. Numbers only, never alerts.",
            enabled: crowdEnabled,
            fps: crowdFps,
          },
          {
            key: "ppe",
            title: "PPE",
            blurb: "Grades safety gear per frame. Grading only, alerts not enabled.",
            enabled: ppeEnabled,
            fps: ppeFps,
          },
        ],
        ready: settingsLoaded,
        onToggle: setPathEnabled,
        onFps: setPathFps,
        onReset: resetDetectionPaths,
      }}
    />
  );

  /** Live Monitor "Detection systems" panel: the same five stored switches
   *  the Settings card writes, so both cards share one source of truth.
   *  Saving rides the existing debounced settingsBody. */
  const setSystemEnabled = (key: DetectionSystemRow["key"], enabled: boolean) => {
    if (key === "hazard") setHazardEnabled(enabled);
    else if (key === "fire") setFireEnabled(enabled);
    else if (key === "smoke") setSmokeEnabled(enabled);
    else if (key === "crowd") setCrowdEnabled(enabled);
    else setPpeEnabled(enabled);
  };
  const detectionSystems = (
    <DetectionSystemsPanel
      model={{
        rows: [
          { key: "hazard", title: "Hazard zone", enabled: hazardEnabled },
          { key: "fire", title: "Fire", enabled: fireEnabled },
          { key: "smoke", title: "Smoke", enabled: smokeEnabled },
          { key: "crowd", title: "Crowd", enabled: crowdEnabled },
          { key: "ppe", title: "PPE", enabled: ppeEnabled },
        ],
        ready: settingsLoaded,
        onToggle: setSystemEnabled,
      }}
    />
  );

  /** Legend rows for the ticked systems only: unticked systems disappear
   *  from the key along with their boxes. */
  const legendItems = [
    ...(appliedHazard
      ? [
          { label: "Person safe", tone: "safe" as const },
          { label: "Person in zone", tone: "danger" as const },
          { label: "Danger zone", tone: "warn" as const },
        ]
      : []),
    ...(appliedFire ? [{ label: "Fire", tone: "fire" as const }] : []),
    ...(appliedSmoke ? [{ label: "Smoke", tone: "smoke" as const }] : []),
    ...(appliedPpe ? [{ label: "PPE violation", tone: "ppe" as const }] : []),
  ];

  const defaultsFields = () => (
    <>
      <p className="applied-line" role="status">
        {settingsApplied ? "Applied" : " "}
      </p>

      <label className="field">
        <span>Alert mode</span>
        <select
          value={mode}
          onChange={(e) => setMode(e.target.value as RuleMode)}
          aria-label="Alert mode"
        >
          <option value="TIMER">After a delay in the zone</option>
          <option value="INSTANT">As soon as they enter</option>
          <option value="LIMIT">Only when over the limit</option>
        </select>
        <span className="help">
          {mode === "INSTANT"
            ? "Every entry raises an alert, even within the allowed number. Use 'After a delay' or 'Only when over the limit' if people are allowed in this zone."
            : mode === "LIMIT"
              ? "Only a zone over its allowance raises an alert. Entries and dwell time are only recorded, so the timers below are not used."
              : "The alarm waits for the delay below before it fires"}
        </span>
      </label>

      <label className={`field ${usesDelay ? "" : "field-off"}`}>
        <span>Dwell limit (seconds)</span>
        <input
          type="number"
          min={0}
          step={0.5}
          aria-label="Dwell limit (seconds)"
          disabled={!usesDelay}
          value={dwellSeconds}
          onChange={(e) => setDwellSeconds(Math.max(0, Number(e.target.value) || 0))}
        />
        <span className="help">
          {mode === "LIMIT"
            ? "Not used while Only when over the limit is selected"
            : mode === "INSTANT"
              ? "Not used while As soon as they enter is selected"
              : "How long a person must stay in a zone that has no delay of its own before it alarms"}
        </span>
      </label>

      <label className={`field ${usesDelay ? "" : "field-off"}`}>
        <span>Repeat alert every (seconds)</span>
        <input
          type="number"
          min={0}
          step={1}
          aria-label="Repeat alert every (seconds)"
          disabled={!usesDelay}
          value={repeatAlertSeconds}
          onChange={(e) => setRepeatAlertSeconds(Math.max(0, Number(e.target.value) || 0))}
        />
        <span className="help">
          {mode === "LIMIT"
            ? "Not used: only over-capacity raises an alert"
            : mode === "INSTANT"
              ? "Not used: an immediate alert fires once per entry"
              : "0 = don’t repeat the alert"}
        </span>
      </label>

      <div className="field">
        <span>People allowed</span>
        <div className="slider-row">
          <input
            type="range"
            min={0}
            max={Math.max(20, allowed)}
            step={1}
            aria-label="Allowed people"
            value={allowed}
            onChange={(e) => setAllowed(Math.max(0, Number(e.target.value) || 0))}
          />
          <input
            type="number"
            min={0}
            aria-label="People allowed"
            value={allowed}
            onChange={(e) => setAllowed(Math.max(0, Number(e.target.value) || 0))}
          />
        </div>
        <span className="help">
          The default for every zone that has no limit of its own. Each zone is checked against its
          own number, so 0 here alarms as soon as anyone is inside any zone that has not been given
          a limit of its own
        </span>
      </div>

      <label className="confidence-field">
        <span>
          Detection confidence <strong>{confidence.toFixed(2)}</strong>
        </span>
        <input
          type="range"
          min={0.05}
          max={1}
          step={0.05}
          value={confidence}
          onChange={(e) => setConfidence(Number(e.target.value))}
          aria-label="Detection confidence"
        />
        <span className="help">
          Higher ignores weaker hits; lower catches more at the cost of false boxes.
        </span>
      </label>

      <fieldset className="field radio-field">
        <legend>Detect person at</legend>
        <label className="radio">
          <input
            type="radio"
            name="anchor"
            value="bottom_center"
            checked={anchor === "bottom_center"}
            onChange={() => setAnchor("bottom_center")}
          />
          <span>Feet</span>
        </label>
        <label className="radio">
          <input
            type="radio"
            name="anchor"
            value="center"
            checked={anchor === "center"}
            onChange={() => setAnchor("center")}
          />
          <span>Body center</span>
        </label>
        <span className="help">Use Body center for a desk demo</span>
      </fieldset>
    </>
  );

  /** The machine name Live Monitor shows next to "Live monitor". */
  const machineField = (
    <label className="field">
      <span>Machine name</span>
      <input
        value={machineName}
        onChange={(e) => setMachineName(e.target.value)}
        placeholder="Excavator 01"
        aria-label="Machine name"
      />
      <span className="help">Shown next to Live monitor at the top of the monitor page.</span>
    </label>
  );

  /** Removing every zone is a machine-wide action with a two-step confirm,
   *  living in its own Danger card. */
  const dangerCard = confirmingRemove ? (
    <div className="confirm-row" role="alertdialog" aria-label="Confirm removal">
      <span className="muted small">
        Remove all {savedZones.length} saved zone{savedZones.length === 1 ? "" : "s"}?
        Monitoring keeps running with no danger zone.
      </span>
      <div className="button-row">
        <button type="button" onClick={removeAllZones} disabled={busy}>
          Remove all zones
        </button>
        <button type="button" onClick={() => setConfirmingRemove(false)} disabled={busy}>
          Cancel
        </button>
      </div>
    </div>
  ) : (
    <button
      type="button"
      className="danger-action"
      onClick={() => setConfirmingRemove(true)}
      disabled={busy || (savedZones.length === 0 && zones.length === 0)}
    >
      Remove all zones
    </button>
  );

  /** leaves Edit mode. `force` skips the question, used once the user has
   *  answered it and by Save, which is not losing anything. */
  const leaveEdit = useCallback((force = false) => {
    if (!force && unsavedGeometry) {
      setConfirmingCancel(true);
      return;
    }
    setEditing(false);
    setConfirmingCancel(false);
    setConfirmingRemove(false);
    setDrag(null);
    setDraft([]);
    setSelected(null);
    setSelectedPoint(null);
    setHistory([]);
    setTool(null);
  }, [unsavedGeometry]);

  /** Cancel: ask only when there is something to lose. */
  const cancelEdit = () => leaveEdit();

  // A zone that cannot be monitored is said out loud, in the one toast area,
  // instead of being quietly dropped. Only for the zones being edited: a saved
  // zone that is still being fitted to this camera has its own notice, and
  // would otherwise talk over it.
  useEffect(() => {
    if (editing && zoneIssue) setError(zoneIssue);
  }, [editing, zoneIssue, setError]);

  // Every zone is labelled on its own, in its own top-left corner, with the
  // same name the chip under the video uses - two zones no longer share a
  // single tag floating over the bounding box of both, which made the tag look
  // like it belonged to the first one. Sizes are in frame pixels so a tag does
  // not shrink into nothing on a large video.
  const labelFont = 16 * unit;
  const labelPad = 8 * unit;
  const labelHeight = labelFont * 1.6;
  const labelWidth = (text: string) => text.length * labelFont * 0.56 + labelPad * 2;
  /** The one name a zone is known by: its chip, its tag and its aria-label.
   *  The name is read from the zone's own spec, so the label on the picture and
   *  the label in the list cannot disagree. A zone with no spec (a backend that
   *  predates named zones, or one drawn seconds ago) keeps the positional name
   *  it always had. */
  const zoneName = (index: number) => zoneSpecs[index]?.name?.trim() || `Zone ${index + 1}`;

  /** One zone's row of /state, or a neutral stand-in.
   *
   *  A backend with per-zone rows is read from them. One with a single zone uses
   *  the global numbers, because there is only one zone and they describe it.
   *  One with neither is reported as empty rather than borrowing the global
   *  inside count: with several zones that count belongs to no zone in
   *  particular, so showing it on a card would point the operator at the wrong
   *  one. */
  const reportFor = (index: number): ZoneReport => {
    const report = state?.zone_reports?.[index];
    if (report) return report;
    if (shownZones.length === 1) {
      const on = zoneSpecs[index]?.enabled ?? true;
      return {
        index,
        name: zoneName(index),
        // a switched-off zone holds nobody, mirroring what the backend
        // reports for unusable zones
        inside: on ? inside : 0,
        // the zone's own allowance, not the machine-wide one: with one zone
        // the tiles sit beside this number, so the wrong source shows a
        // limit the zone was never given
        allowed_people: zoneSpecs[index]?.allowed_people ?? allowed,
        dwell_seconds: on ? alertSeconds : 0,
        alerting: on && alert,
        enabled: on,
      };
    }
    return {
      index,
      name: zoneName(index),
      inside: 0,
      allowed_people: zoneSpecs[index]?.allowed_people ?? allowed,
      dwell_seconds: 0,
      alerting: false,
      enabled: zoneSpecs[index]?.enabled ?? true,
    };
  };

  /** The dot for one zone: green with nobody in it, orange when somebody is
   *  standing in it but still within the limit, red when that zone is
   *  alerting, neutral when the backend cannot say which zone it is. */
  const dotFor = (index: number): string => {
    const report = reportFor(index);
    if (!report.enabled) return "neutral";
    if (!state?.zone_reports && shownZones.length > 1) return "neutral";
    if (report.alerting) return "danger";
    return report.inside > 0 ? "warn" : "safe";
  };

  /** The zone the two tiles under the video are talking about.
   *
   *  The tiles used to show the global `inside` next to the global allowance,
   *  which stopped being a true comparison the moment zones had rules of their
   *  own: "3 / 4" beside two different per-zone limits says nothing about
   *  whether anybody is over anything. A total has no single limit to be judged
   *  against, so with per-zone rows the tiles follow the WORST zone and name it.
   *  The count and the limit then belong to the same zone, and the card that
   *  zone belongs to is the red one on the right.
   *
   *  "Worst" is the zone furthest over its own allowance; with every zone inside
   *  its limit it is the busiest one, because a tile reading "0 people" while
   *  the picture is full is its own kind of lie. Ties go to the lowest index.
   *  Without per-zone rows there is nothing to compare, so the tiles fall back
   *  to totals and say so - they are the only case where a total is shown. */
  const worstZone = (): ZoneReport | null => {
    if (!state?.zone_reports || shownZones.length === 0) return null;
    if (shownZones.length === 1) return reportFor(0);

    let best: ZoneReport | null = null;
    let bestScore = -Infinity;
    for (let index = 0; index < shownZones.length; index += 1) {
      const report = reportFor(index);
      if (!report.enabled) continue;
      const over = report.inside - report.allowed_people;
      const score = over > 0 ? 1000 + over : report.inside;
      if (score > bestScore) {
        bestScore = score;
        best = report;
      }
    }
    return best;
  };

  /** The dwell limit for one zone: the limit from its own row of /state, else
   *  the limit it is following as a default, else the machine-wide one. */
  const zoneDwellLimit = (report: ZoneReport | null): number => {
    if (!report) return state?.rules.dwell_seconds ?? 0;
    if (report.dwell_limit !== undefined) return report.dwell_limit;
    return zoneSpecs[report.index]?.dwell_seconds ?? state?.rules.dwell_seconds ?? 0;
  };

  /** The zone the two tiles talk about: the selected one while it is being
   *  edited, else the busiest one, else nothing when the backend sent no
   *  per-zone rows and there is more than one zone to choose from. */
  const focusIndex = (): number | null => {
    if (shownZones.length === 0) return null;
    if (editing && selected !== null && selected >= 0 && selected < shownZones.length) {
      return selected;
    }
    if (shownZones.length === 1) return 0;
    const worst = worstZone();
    return worst ? worst.index : null;
  };

  /** the two tiles, and what each one is honestly able to claim.
   *
   *  Both numbers always sit beside the SHOWN zone's own rules - its allowance
   *  and its alert-after - never the machine-wide defaults. A stopped camera
   *  sees nothing, so the counts read 0 beside those same real limits.
   *
   *  Each tile also carries its tone: red when over the limit, orange while
   *  somebody is inside but still within it, idle otherwise. */
  const tileFigures = () => {
    const focus = focusIndex();
    if (focus !== null) {
      const report = reportFor(focus);
      const named = shownZones.length > 1;
      const limit = zoneDwellLimit(report);
      const people = running ? report.inside : 0;
      const dwell = running ? report.dwell_seconds : 0;
      const peopleOver = running && report.inside > report.allowed_people;
      const dwellOver = running && report.dwell_seconds > limit;
      const active = running && report.inside > 0;
      return {
        peopleLabel: named ? `People in ${report.name}` : "People in zone",
        people,
        peopleLimit: report.allowed_people,
        peopleTotal: false,
        peopleTone: peopleOver ? ("over" as const) : active ? ("occupy" as const) : ("idle" as const),
        dwellLabel: named ? `Dwell time in ${report.name}` : "Dwell time",
        dwell,
        dwellLimit: limit,
        dwellTotal: false,
        dwellTone: dwellOver ? ("over" as const) : active ? ("occupy" as const) : ("idle" as const),
      };
    }
    // No per-zone rows. One zone is still described exactly by the global
    // numbers; several zones are only described by a total, and a total has no
    // single limit, so none is shown rather than the wrong one.
    const single = shownZones.length === 1;
    const peopleLimit = single ? (state?.allowed_people ?? 0) : null;
    const dwellLimit = single ? (state?.rules?.dwell_seconds ?? 0) : null;
    const people = running ? inside : 0;
    const dwell = running ? alertSeconds : 0;
    return {
      peopleLabel: single ? "People in zone" : "People in all zones (total)",
      people,
      peopleLimit,
      peopleTotal: !single,
      peopleTone:
        peopleLimit !== null && people > peopleLimit
          ? ("over" as const)
          : single && people > 0
            ? ("occupy" as const)
            : ("idle" as const),
      dwellLabel: single ? "Dwell time" : "Longest dwell in any zone (total)",
      dwell,
      dwellLimit,
      dwellTotal: !single,
      dwellTone:
        dwellLimit !== null && dwell > dwellLimit
          ? ("over" as const)
          : single && people > 0
            ? ("occupy" as const)
            : ("idle" as const),
    };
  };

  // What the two tiles under the video claim, worked out once here so both
  // tiles read from the same decision instead of each working it out again.
  const tiles = tileFigures();

  /** One zone's own rules, edited in place. These are the per-zone numbers; the
   *  Settings page keeps the global defaults, which is the only place they are
   *  edited, so two numbers cannot both claim to be "the" limit. */
  const zoneRuleFields = (index: number) => {
    const spec = zoneSpecs[index];
    const name = zoneName(index);
    const update = (patch: Partial<ZoneSpec>) =>
      setZoneSpecs((list) =>
        list.map((current, i) => (i === index ? { ...current, ...patch } : current))
      );
    return (
      <div className="zone-rules" data-testid="zone-rules">
        <label className="field">
          <span>Zone name</span>
          <input
            type="text"
            aria-label={`Name for ${name}`}
            value={spec?.name ?? name}
            onChange={(event) => update({ name: event.target.value })}
          />
          <span className="help">How this zone is named on the picture and in the alerts</span>
        </label>

        <label className="field">
          <span>People allowed</span>
          <input
            type="number"
            min={0}
            aria-label={`People allowed in ${name}`}
            value={spec?.allowed_people ?? 0}
            onChange={(event) =>
              update({
                allowed_people: Math.max(0, Number(event.target.value) || 0),
                using_defaults: false,
              })
            }
          />
          <span className="help">
            Only the people inside this zone count. 0 = alarm as soon as anyone is in
          </span>
        </label>

        <label className={`field ${usesDelay ? "" : "field-off"}`}>
          <span>Alert after (seconds)</span>
          <input
            type="number"
            min={0}
            step={0.5}
            aria-label={`Alert after (seconds) for ${name}`}
            disabled={!usesDelay}
            value={spec?.dwell_seconds ?? 0}
            onChange={(event) =>
              update({
                dwell_seconds: Math.max(0, Number(event.target.value) || 0),
                using_defaults: false,
              })
            }
          />
          <span className="help">
            {mode === "LIMIT"
              ? "Not used while Only when over the limit is selected"
              : mode === "INSTANT"
                ? "Not used while As soon as they enter is selected"
                : "How long somebody may stay in this zone before it alarms"}
          </span>
        </label>

        <label className="check">
          <input
            type="checkbox"
            aria-label={`Enabled for ${name}`}
            checked={spec?.enabled ?? true}
            onChange={(event) => update({ enabled: event.target.checked })}
          />
          <span>Enabled</span>
        </label>
        <p className="muted small">
          {spec?.using_defaults
            ? "Using the default rules from Settings. Change a number here and this zone keeps its own."
            : "This zone has its own rules. The Settings defaults no longer change them."}
        </p>
      </div>
    );
  };

  /** where one zone's tag goes: that zone's top-left corner, kept inside the
   *  frame, and centred on the zone when the zone is too narrow to hold it */
  const labelFor = (zone: Point[], index: number) => {
    const text = zoneName(index);
    const width = labelWidth(text);
    const left = Math.min(...zone.map((point) => point[0]));
    const right = Math.max(...zone.map((point) => point[0]));
    const top = Math.min(...zone.map((point) => point[1]));
    const x =
      right - left >= width
        ? left
        : left - (width - (right - left)) / 2;
    return {
      text,
      width,
      height: labelHeight,
      x: Math.max(0, Math.min(x, surface.width - width)),
      y: Math.max(0, Math.min(top, surface.height - labelHeight)),
    };
  };

  const rubberBand = (() => {
    if (!drawing || !cursor) return null;
    if (drag?.kind === "rectangle") {
      return {
        kind: "rectangle" as const,
        x: Math.min(drag.origin[0], cursor[0]),
        y: Math.min(drag.origin[1], cursor[1]),
        width: Math.abs(cursor[0] - drag.origin[0]),
        height: Math.abs(cursor[1] - drag.origin[1]),
      };
    }
    if (drag?.kind === "circle") {
      const cx = (drag.origin[0] + cursor[0]) / 2;
      const cy = (drag.origin[1] + cursor[1]) / 2;
      return {
        kind: "circle" as const,
        cx,
        cy,
        rx: Math.abs(cursor[0] - drag.origin[0]) / 2,
        ry: Math.abs(cursor[1] - drag.origin[1]) / 2,
      };
    }
    return null;
  })();

  /** the corners clicked out so far, closed back to the first point once
   *  three are down, with the cursor waiting to be the next corner */
  const polygonPreview =
    drawing && (tool === "quad" || tool === "polygon") && draft.length > 0
      ? [
          ...draft,
          ...(draft.length >= MIN_POINTS ? [draft[0]] : []),
          ...(cursor ? [cursor] : []),
        ]
      : null;

  /** the draft corners themselves, so the overlay can paint them */
  const draftPoints =
    drawing && (tool === "quad" || tool === "polygon") && draft.length > 0 ? [...draft] : null;

  /** the "+" marks on the selected zone only, so overlapping shapes do not turn
   *  into a thicket of click targets. Every edge gets one, including the
   *  closing edge from the last corner back to the first. */
  const selectedZone =
    selected !== null && selected >= 0 && selected < zones.length ? zones[selected] : [];
  const midpoints =
    drawing && selected !== null && tool === "select" && selectedZone.length >= 2 && selectedZone.length < MAX_POINTS
      ? selectedZone.map((point, index) => ({
          index,
          point: midpoint(point, selectedZone[(index + 1) % selectedZone.length]),
        }))
      : [];

  /** Row-level enable toggle (new in step 1). Inside Edit the change waits for
   *  Save like every other rule edit; on the Live page it persists at once, so
   *  the switch always does what it says. */
  const toggleZoneEnabled = (index: number, enabled: boolean) => {
    const nextSpecs = zoneSpecs.map((current, i) =>
      i === index ? { ...current, enabled } : current
    );
    while (nextSpecs.length < savedZones.length) {
      nextSpecs.push({
        name: `Zone ${nextSpecs.length + 1}`,
        allowed_people: allowed,
        dwell_seconds: dwellSeconds,
        enabled: true,
        using_defaults: true,
      });
    }
    setZoneSpecs(nextSpecs);
    if (screenMode === "edit") return;
    void (async () => {
      setBusy(true);
      try {
        const base = payload(savedZones);
        const body: ZonePayload = {
          ...base,
          zone_specs: savedZones.map((_, i) => {
            const spec = nextSpecs[i];
            return {
              name: spec?.name?.trim() || `Zone ${i + 1}`,
              allowed_people: spec?.allowed_people ?? allowed,
              dwell_seconds: spec?.dwell_seconds ?? dwellSeconds,
              enabled: spec?.enabled ?? true,
              using_defaults: spec?.using_defaults ?? true,
            };
          }),
        };
        await saveConfig(body);
        if (running) await startMonitoring(body);
        setNotice(null);
    } catch (err) {
      setError((err as Error).message);
      } finally {
        setBusy(false);
      }
    })();
  };

  /** "+ Add zone": opens the editor with the rectangle tool ready, live or on
   *  the stopped canvas. */
  const addZone = () => {
    if (busy || !canEdit) return;
    if (screenMode !== "edit") {
      setEditing(true);
      setSelected(null);
      setSelectedPoint(null);
      setDraft([]);
    }
    setTool("rectangle");
    setError(null);
    setNotice(null);
  };

  const pickTool = (next: DrawTool) => {
    setTool(next);
    setDraft([]);
    setCursor(null);
    setError(null);
    setNotice(null);
  };

  const toolbarUndo = () => {
    if (draft.length > 0) {
      setDraft((current) => current.slice(0, -1));
      return;
    }
    undo();
  };

  const toolbarDelete = () => {
    if (selected === null) return;
    if (selectedPoint !== null) removePoint(selected, selectedPoint);
    else removeZone(selected);
  };

  const addPoint = () => {
    if (selected === null) return;
    const zone = zones[selected];
    if (!zone || zone.length >= MAX_POINTS) {
      setError(`A zone can have at most ${MAX_POINTS} points - remove one or press Undo`);
      return;
    }
    remember();
    setNotice(null);
    setSelectedPoint(null);
    // a new corner starts at the zone's centre, not at the frame origin: [0, 0]
    // lands far off the shape and can fold the polygon over itself.
    const cx = Math.round(zone.reduce((sum, point) => sum + point[0], 0) / zone.length);
    const cy = Math.round(zone.reduce((sum, point) => sum + point[1], 0) / zone.length);
    const at = clampPoint([cx, cy], surface.width, surface.height);
    setZones((current) =>
      current.map((existing, i) => (i === selected ? [...existing, at] : existing))
    );
  };

  const openZoneCard = (index: number) => {
    if (screenMode !== "edit") return;
    setSelected(index);
    setSelectedPoint(null);
    setTool("select");
  };

  const startRename = (index: number) => {
    setRenaming(index);
    setRenameDraft(zoneSpecs[index]?.name ?? zoneName(index));
  };

  const dropOver = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setDragOver(true);
  };

  const onPictureHover = (event: React.MouseEvent) => {
    if (!drawing) return;
    const point = toFrame(event.clientX, event.clientY);
    if (point) setCursor(point);
  };

  /** The workers raising the banner right now, longest dwell first, so the
   *  banner can name who is where for how long. A stopped camera holds no
   *  verdict at all: last run's rows must not linger as a banner. */
  const bannerWorkers: BannerWorker[] = running
    ? (state?.workers ?? [])
        .filter((worker) => worker.state === "ALERT")
        .map((worker) => ({
          id: worker.worker_id,
          zone:
            worker.zone_name ||
            (worker.zone_index !== undefined && worker.zone_index !== null && worker.zone_index >= 0
              ? zoneName(worker.zone_index)
              : ""),
          dwell: worker.dwell_seconds,
        }))
        .sort((a, b) => b.dwell - a.dwell)
    : [];

  /** Somebody inside but still within the limit: the orange banner names them
   *  with "Xs of Ys" against their own zone's dwell limit. Only INSIDE
   *  workers qualify - anyone alerting takes the red banner instead. */
  const bannerInside: BannerInside[] = running
    ? (state?.workers ?? [])
        .filter((worker) => worker.state === "INSIDE")
        .map((worker) => {
          const index = worker.zone_index ?? -1;
          const report = state?.zone_reports?.[index];
          const limit =
            report?.dwell_limit ??
            (index >= 0 ? zoneSpecs[index]?.dwell_seconds : undefined) ??
            state?.rules.dwell_seconds ??
            0;
          return {
            id: worker.worker_id,
            zone: worker.zone_name || (index >= 0 ? zoneName(index) : ""),
            dwell: worker.dwell_seconds,
            limit,
          };
        })
        .sort((a, b) => b.dwell - a.dwell)
    : [];

  const overlayTags = overlayZones.map((zone, index) => {
    if (zone.length < MIN_POINTS) return null;
    const tag = labelFor(zone, index);
    return { text: tag.text, x: tag.x, y: tag.y, width: tag.width, height: tag.height };
  });

  const selectedHandles =
    drawing && tool === "select" && selected !== null && shownZones[selected]
      ? shownZones[selected].map((point, index) => ({
          point,
          invalid: surfaceKnown && isOutOfFrame(point),
          active: selectedPoint === index || (drag?.kind === "handle" && drag.index === index),
        }))
      : null;

  const videoWarnings: Record<string, boolean> = {};
  for (const video of videos) {
    videoWarnings[video.id] = frameMismatch(video);
  }

  /** Gate progress while fire/smoke is being confirmed, as one neutral line
   *  for the Live page — or null when nothing is confirming. Built from the
   *  per-type sample counts, never from alert state, so it cannot read as
   *  an alert. */
  const confirmingLine = (() => {
    if (!running) return null;
    const confirming = state?.fire_smoke?.confirming;
    if (!confirming) return null;
    for (const kind of ["fire", "smoke"] as const) {
      const progress = confirming[kind];
      if (progress) {
        return `Confirming possible ${kind} — ${progress.samples} of ${progress.required} samples`;
      }
    }
    return null;
  })();

  const liveModel: LiveMonitorModel = {
    machineName,
    screenMode,
    running,
    stoppedSourceLabel: (() => {
      const hit = videos.find((video) => samePath(video.source, source));
      if (hit) return hit.name;
      if (usingTestVideo) return source;
      return `Webcam ${/^\d+$/.test(source) ? source : "0"}`;
    })(),
    /** Live Monitor source picker: the same option values ("camera" or the
     *  clip path) and the same shared select handler as Settings, so both
     *  dropdowns always agree. Counts come from the saved-counts cache. */
    sourcePicker: (() => {
      const value = sourceMode === "camera" ? "camera" : source;
      const webcamSource = /^\d+$/.test(source) && sourceMode === "camera" ? source : "0";
      const options = [
        {
          value: "camera",
          label: `Webcam ${webcamSource}`,
          zones: zoneCountFor(webcamSource),
        },
        ...videos.map((video) => ({
          value: video.source,
          label: video.name,
          zones: zoneCountFor(video.source),
        })),
      ];
      const pending =
        confirmLiveSource === null
          ? null
          : {
              value: confirmLiveSource,
              label:
                options.find((option) => option.value === confirmLiveSource)?.label ??
                confirmLiveSource,
            };
      const currentZones = zoneCountFor(
        sourceMode === "camera" ? webcamSource : source
      );
      return {
        options,
        value,
        disabled: editing,
        onSelect: onLiveSourceChange,
        uploadDisabled: uploadPct !== null || videos.length >= videoLimit,
        onPickFile,
        uploadPct,
        pending,
        onConfirmSwitch: confirmLiveSourceSwitch,
        onCancelSwitch: cancelLiveSourceSwitch,
        noZoneHint: !editing && currentZones !== null && currentZones === 0,
        // the Loop box sits under the picker for a test clip only: the
        // webcam has no end to replay from. Same stored setting as the
        // Settings Camera card, so either box moves both.
        loopVideo:
          value === "camera"
            ? null
            : {
                checked: loopVideo,
                disabled: editing,
                onChange: (next: boolean) => setLoopVideo(next),
              },
      };
    })(),
    busy,
    startZoneBlock,
    drawing,
    /** the editor can open: live with a frame, or stopped with a saved size */
    canEdit,
    /** the video column shows a panel: the stream, or the dark edit canvas */
    stageVisible: running || editing,
    tool,
    frameKnown: surfaceKnown,
    frameWidth: surface.width,
    frameHeight: surface.height,
    unit,
    selectedShape: drawing && selected !== null ? selected : null,
    pictureUrl,
    imageRef,
    onStreamError: handleStreamError,
    // a saved file source that no longer exists is named in the stopped panel
    // so the operator knows which clip to re-upload instead of seeing a blank
    // camera; while running the stream (or the error toast) speaks for itself
    stoppedHint: !running ? (state?.source_error ?? null) : null,
    // a single-play clip that ended by itself, and only for the source that
    // ended: picking another clip while stopped must not inherit the message
    finishedClip:
      !running &&
      (state?.finished_clip ?? false) &&
      samePath(state?.source ?? "", source),
    overlayVisible,
    shownZones,
    overlayZones,
    legendItems,
    detectionSystems,
    rubberBand,
    polygonPreview,
    strokePreview,
    draftPoints,
    tags: overlayTags,
    tagFontSize: labelFont,
    tagPad: labelPad,
    handles: selectedHandles,
    handleZone: selectedHandles ? selected : null,
    midpoints,
    cursorHint:
      editing && tool === null && zones.length === 0 && surfaceKnown && canEdit
        ? "pick-tool"
        : !surfaceKnown && (running || editing)
          ? "waiting-frame"
          : null,
    alert,
    bannerWorkers,
    bannerInside,
    bannerMessage: running ? (state?.message ?? "") : "",
    muted,
    tiles,
    sidePaths: (
      <SidePathsReadout
        model={{ crowd: state?.crowd ?? null, ppe: state?.ppe ?? null, running }}
      />
    ),
    hazardOff,
    confirming: confirmingLine,
    anchorHint:
      anchorOffScreen.length > 0
        ? { count: anchorOffScreen.length, suggestBodyCenter: anchor === "bottom_center" }
        : null,
    legendVisible: (running || editing) && surfaceKnown,
    zoneCards: shownZones.map((zone, zoneIndex) => {
      const report = reportFor(zoneIndex);
      const name = zoneName(zoneIndex);
      const limit = zoneDwellLimit(report);
      // stopped is neutral grey zeroes: the figures describe the last run,
      // not anything the camera sees now. The on/off switch stays honest:
      // an enabled zone reads green, a switched-off one says Disabled.
      if (!running) {
        const stoppedEnabled = zoneSpecs[zoneIndex]?.enabled ?? true;
        return {
          name,
          dot: stoppedEnabled ? "safe" : "neutral",
          countText: "0 people",
          showFigures: false,
          peopleText: `0 / ${report.allowed_people}`,
          peopleOver: false,
          dwellText: `0.0s / ${limit.toFixed(1)}s`,
          dwellOver: false,
          open: false,
          corners: zone.length,
          usingDefaults: zoneSpecs[zoneIndex]?.using_defaults ?? true,
          enabled: stoppedEnabled,
          rules: null,
        };
      }
      return {
        name,
        dot: dotFor(zoneIndex),
        countText: `${report.inside} ${report.inside === 1 ? "person" : "people"}`,
        showFigures: !!state?.zone_reports,
        peopleText: `${report.inside} / ${report.allowed_people}`,
        peopleOver: report.inside > report.allowed_people,
        dwellText: `${report.dwell_seconds.toFixed(1)}s / ${limit.toFixed(1)}s`,
        dwellOver: report.dwell_seconds > limit,
        open: screenMode === "edit" && selected === zoneIndex,
        corners: zone.length,
        usingDefaults: zoneSpecs[zoneIndex]?.using_defaults ?? true,
        enabled: zoneSpecs[zoneIndex]?.enabled ?? true,
        rules: screenMode === "edit" && selected === zoneIndex ? zoneRuleFields(zoneIndex) : null,
      };
    }).map((card) =>
      hazardOff && screenMode !== "edit"
        ? { ...card, dot: "neutral", countText: "Off", showFigures: false }
        : card
    ),
    renaming,
    renameDraft,
    confirmingCancel,
    unsavedGeometry,
    zoneValid,
    coordRows:
      screenMode === "edit" && selected !== null && zones[selected]
        ? zones[selected].map((point) => ({
            x: point[0],
            y: point[1],
            invalid: surfaceKnown && isOutOfFrame(point),
          }))
        : null,
    coordTitle: selected !== null ? zoneName(selected) : "",
    coordZone: selected,
    canAddPoint: selected !== null && !!zones[selected] && zones[selected].length < MAX_POINTS,
    recentAlerts,
    recentFireAlerts,
    recentLimit: RECENT_ALERT_COUNT,
    onAcknowledgeFire: (id: number) => void acknowledgeFire(id),
    onOpenFireEvidence: (alert: FireAlert) => setFireEvidence(alert),
    fireEvidence,
    fireEvidenceDetails,
    onAcknowledgeFireEvidence: fireEvidence ? () => acknowledgeFire(fireEvidence.id) : null,
    onCloseFireEvidence: () => setFireEvidence(null),
    onFireEvidenceStep: stepLiveFireEvidence,
    fireEvidenceHasPrev: liveFireEvidenceIndex > 0,
    fireEvidenceHasNext:
      liveFireEvidenceIndex !== -1 && liveFireEvidenceIndex < fireActiveAlerts.length - 1,
    onFireFalseAlarm: (id: number) => void markFalseAlarm(id),
    fireFalseAlarming: falseAlarming,
    perfFps: perfValue(perf?.fps),
    historyEmpty: history.length === 0,
    draftEmpty: draft.length === 0,
    clearDisabled: zones.length === 0 && draft.length === 0,
    deleteDisabled: selected === null,
    onStart: start,
    onStop: stop,
    onEnterEdit: enterEdit,
    onSave: () => void saveZone(),
    onCancelEdit: cancelEdit,
    onConfirmDiscard: () => leaveEdit(true),
    onKeepEditing: () => setConfirmingCancel(false),
    onPickTool: pickTool,
    onToolbarUndo: toolbarUndo,
    onToolbarDelete: toolbarDelete,
    onClearZones: clearZones,
    onPictureDown,
    onPictureClick,
    onPictureDoubleClick,
    /** Finish button for the Polygon tool: closes the open draft, reporting
     *  too-few-corners instead of swallowing the press. */
    onFinishPolygon: () => {
      if (tool === "polygon" && draft.length > 0) closePolygon();
    },
    onPictureHover,
    onPictureLeave: () => setCursor(null),
    onHandleDown,
    onEdgeClick,
    onHandleDoubleClick: (zone, index) => removePoint(zone, index),
    onEditCoordinate: editCoordinate,
    onRemoveCoordinate: (zone, index) => removePoint(zone, index),
    onAddPoint: addPoint,
    onOpenZoneCard: openZoneCard,
    onRenameStart: startRename,
    onRenameDraft: setRenameDraft,
    onCommitRename: commitRename,
    onRenameCancel: () => setRenaming(null),
    onToggleZoneEnabled: toggleZoneEnabled,
    onDeleteZone: removeZone,
    onAddZone: addZone,
    onToggleMute: () => setMuted((current) => !current),
    onAcknowledge: (id) => void acknowledge(id),
    onViewAllAlerts: () => navigate("/alerts"),
  };

  /** Settings → Camera: source picking lives here now, not on the monitor.
   *  The monitor only starts whatever source is saved. */
  /** Which uploaded video is one tap away from deletion. Delete is a
   *  two-step confirm, the same as removing all zones. */
  const [confirmingDelete, setConfirmingDelete] = useState<string | null>(null);

  /** Camera card: one dropdown over the real sources (the webcam plus every
   *  uploaded clip), switching through the same handlers the old segmented
   *  buttons used. The resolution line only renders when a real frame size
   *  is known — a video's stored size, or the running feed's — never
   *  otherwise. The raw source path stays behind the Advanced fold. */
  const cameraSourceValue = sourceMode === "camera" ? "camera" : source;
  const cameraResolution = (() => {
    if (sourceMode === "video") {
      const hit = videos.find((video) => samePath(video.source, source));
      if (hit && hit.width > 0 && hit.height > 0) return `${hit.width}×${hit.height}`;
      return null;
    }
    if (surfaceKnown && surface.width > 0 && surface.height > 0) {
      return `${surface.width}x${surface.height}`;
    }
    return null;
  })();
  const onCameraSourceChange = (value: string) => {
    if (value === "camera") {
      chooseLiveCamera();
      return;
    }
    const hit = videos.find((video) => samePath(video.source, value));
    if (hit) {
      selectVideo(hit);
      return;
    }
    // a stale saved path: keep it selectable so it can be reviewed, without
    // inventing a source that is not there
    setSourceMode("video");
    setSource(value);
  };
  const cameraSection = (
    <>
      <label className="field">
        <span>Video source</span>
        <select
          aria-label="Video source"
          value={cameraSourceValue}
          onChange={(e) => onCameraSourceChange(e.target.value)}
        >
          <option value="camera">
            Live camera (webcam {/^\d+$/.test(source) && sourceMode === "camera" ? source : "0"})
          </option>
          {videos.map((video) => (
            <option key={video.id} value={video.source}>
              {video.name}
              {video.width > 0 && video.height > 0 ? ` — ${video.width}×${video.height}` : ""}
            </option>
          ))}
          {sourceMode === "video" &&
            !videos.some((video) => samePath(video.source, source)) && (
              <option value={source}>{source}</option>
            )}
        </select>
        <span className="help">Start the feed from the Live Monitor page.</span>
      </label>

      {usingTestVideo && (
        <>
          <label className="check">
            <input
              type="checkbox"
              aria-label="Loop video"
              data-testid="camera-loop-video"
              checked={loopVideo}
              onChange={(e) => setLoopVideo(e.target.checked)}
            />
            <span>Loop video</span>
          </label>
          <p className="muted small">Replay the clip when it ends. Off plays it once.</p>
        </>
      )}

      {cameraResolution !== null && (
        <p className="resolution-line" data-testid="camera-resolution">
          Resolution <strong>{cameraResolution}</strong>
        </p>
      )}

      <details className="advanced">
        <summary>Advanced</summary>
        <label className="field">
          <span>Camera source (0 = webcam, or a video file path)</span>
          <input value={source} onChange={(e) => setSource(e.target.value)} />
        </label>
      </details>
    </>
  );

  /** Videos card: the uploaded clips with their backend facts, the picker
   *  that makes one the camera source, the two-step delete, and the upload
   *  button with its progress bar. Drag-and-drop stays on the dropzone
   *  below. Behaviour is unchanged — only the columns and the layout moved
   *  to match the Settings restyle. */
  const videosCard = (
    <>
      {videos.length === 0 ? (
        <p className="muted small">No test videos uploaded yet.</p>
      ) : (
        <div className="video-table-wrap">
          <table className="video-table" data-testid="video-list">
            <thead>
              <tr>
                <th scope="col">Name</th>
                <th scope="col">Size</th>
                <th scope="col">Date</th>
                <th scope="col">Use</th>
                <th scope="col">Delete</th>
              </tr>
            </thead>
            <tbody>
              {videos.map((video) => {
                const selected = samePath(video.source, source);
                return (
                  <tr key={video.id} className={selected ? "selected-row" : ""}>
                    <td>
                      <span className="video-name" title={video.name}>
                        <span aria-hidden="true">🎬 </span>
                        {video.name}
                      </span>
                      {selected && videoWarnings[video.id] && (
                        <span className="tile-warn" data-testid="tile-frame-warning">
                          Saved zones were drawn for another frame size - redraw them.
                        </span>
                      )}
                    </td>
                    <td>{video.size_mb} MB</td>
                    <td>{formatVideoDate(video.created)}</td>
                    <td>
                      {selected ? (
                        <span className="in-use-tag">In use</span>
                      ) : (
                        <button
                          type="button"
                          className="use-btn"
                          aria-label={`Use ${video.name}`}
                          onClick={() => selectVideo(video)}
                        >
                          Use
                        </button>
                      )}
                    </td>
                    <td>
                      {confirmingDelete === video.id ? (
                        <span
                          className="confirm-inline"
                          role="alertdialog"
                          aria-label={`Confirm deletion of ${video.name}`}
                        >
                          <button
                            type="button"
                            className="delete-btn"
                            onClick={() => onDeleteVideo(video)}
                            disabled={busy}
                          >
                            Confirm
                          </button>
                          <button
                            type="button"
                            onClick={() => setConfirmingDelete(null)}
                            disabled={busy}
                          >
                            Cancel
                          </button>
                        </span>
                      ) : (
                        <button
                          type="button"
                          className="delete-btn delete-icon-btn"
                          aria-label={`Delete ${video.name}`}
                          onClick={() => setConfirmingDelete(video.id)}
                          disabled={busy || selected}
                          title={selected ? "The selected video cannot be deleted" : `Delete ${video.name}`}
                        >
                          <span aria-hidden="true">🗑️</span>
                        </button>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
      {videos.length >= videoLimit && (
        <p className="muted small">
          The {videoLimit} video limit is reached. Delete one to upload another.
        </p>
      )}
      <div className="upload-row">
        <label
          className={`upload-button ${
            uploadPct !== null || videos.length >= videoLimit ? "disabled" : ""
          }`}
        >
          Upload Video
          <input
            type="file"
            accept="video/mp4,video/x-msvideo,video/quicktime,video/webm,.mp4,.avi,.mov,.mkv,.webm"
            onChange={onPickFile}
            disabled={uploadPct !== null || videos.length >= videoLimit}
            data-testid="video-file"
          />
        </label>
        {uploadPct !== null && (
          <div className="upload-bar" data-testid="upload-field">
            <progress max={100} value={uploadPct} data-testid="upload-progress" />
            <span className="muted small">Uploading… {uploadPct}%</span>
          </div>
        )}
      </div>
      <div
        className={`dropzone ${dragOver ? "dragover" : ""}`}
        data-testid="dropzone"
        onDragOver={dropOver}
        onDragLeave={() => setDragOver(false)}
        onDrop={onDropFiles}
      >
        <span className="dropzone-icon" aria-hidden="true">
          ⬆
        </span>
        <p className="dropzone-title">Drag &amp; drop your video here</p>
        <span className="help">MP4, AVI, MOV, MKV or WebM · up to {videoMaxMb} MB</span>
      </div>
    </>
  );

  const alertsModel = {
    alertTab,
    activeAlerts,
    activeTotal: unackedTotal,
    clearedAlerts,
    clearedTotal: ackedTotal,
    fireAlerts: alertTab === "active" ? fireActiveAlerts : fireClearedAlerts,
    fireActiveTotal: fireUnackedTotal,
    fireClearedTotal: fireAckedTotal,
    alertsError,
    evidence,
    ackingAll,
    evidenceHasPrev: evidenceIndex > 0,
    evidenceHasNext:
      evidenceIndex !== -1 &&
      evidenceIndex < (alertTab === "active" ? activeAlerts : clearedAlerts).length - 1,
    onTab: setAlertTab,
    onAcknowledge: (id: number) => void acknowledge(id),
    onAcknowledgeAll: () => void acknowledgeAll(),
    onAcknowledgeFire: (id: number) => void acknowledgeFire(id),
    onEvidence: setEvidence,
    onEvidenceStep: stepEvidence,
    fireEvidence,
    fireEvidenceHasPrev: fireEvidenceIndex > 0,
    fireEvidenceHasNext:
      fireEvidenceIndex !== -1 &&
      fireEvidenceIndex < (alertTab === "active" ? fireActiveAlerts : fireClearedAlerts).length - 1,
    onFireEvidence: setFireEvidence,
    onFireEvidenceStep: stepFireEvidence,
    machineName: machineName || undefined,
    onFalseAlarm: (id: number) => void markFalseAlarm(id),
    falseAlarming,
    evidenceDetails,
    onAcknowledgeEvidence: evidence ? () => acknowledge(evidence.id) : null,
    fireEvidenceDetails,
    onAcknowledgeFireEvidence: fireEvidence ? () => acknowledgeFire(fireEvidence.id) : null,
  };

  return (
    <Layout
      usingTestVideo={usingTestVideo}
      statusMode={screenMode === "edit" ? "editing" : running ? "monitoring" : "idle"}
      statusText={screenMode === "edit" ? "Editing" : running ? "RUNNING" : "STOPPED"}
      sinceLine={sinceLine}
      recentAlertCount={unackedTotal + fireUnackedTotal}
      toast={toast}
      confirmingSwitch={confirmingSwitch}
      busy={busy}
      onDiscardSwitch={discardUnsavedAndSwitch}
      onCancelSwitch={cancelSourceSwitch}
    >
      <Routes>
        <Route path="/settings" element={<SettingsPage model={{ defaultsFields: defaultsFields(), machineField, cameraCard: cameraSection, videosCard, dangerCard, pathsCard: detectionPaths, system: systemStatus }} />} />
        <Route path="/alerts" element={<AlertsPage model={alertsModel} />} />
        <Route path="/history" element={<HistoryPage />} />
        <Route path="/" element={<LiveMonitorPage model={liveModel} />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </Layout>
  );
}

/** HashRouter (not BrowserRouter): the built UI is served by FastAPI at `/`
 *  only, so deep links like `/alerts` would 404 on reload. The hash keeps
 *  every route on the same served document, in dev and in the build. The v7
 *  flags silence the React Router future-flag console warnings; the behaviour
 *  they switch on (startTransition wrapping, relative splat paths) is what
 *  the app already assumes. */
export default function App() {
  return (
    <HashRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
      <AppShell />
    </HashRouter>
  );
}
