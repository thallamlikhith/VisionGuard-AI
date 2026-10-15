import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import App from "./App";
import AlertsPage from "./pages/AlertsPage";
import LiveMonitorPage from "./pages/LiveMonitorPage";
import type { LiveMonitorModel } from "./pages/LiveMonitorPage";
import { DailyChart, pageWindow } from "./pages/HistoryPage";
import { istDateTime } from "./utils/format";
import type { AlertEvent, AppConfig, AppState, FireAlert, Point, TestVideo } from "./types";

const CONFIG: AppConfig = {
  machine: { id: "excavator_01", name: "Excavator 01" },
  rules: { mode: "TIMER", dwell_seconds: 1 },
  detector: { model: "yolo11n.pt", confidence: 0.35, person_class_id: 0 },
  zones: [],
  zone_points: [],
  allowed_people: 0,
  anchor: "center",
  source: "0",
};

function makeState(overrides: Partial<AppState> = {}): AppState {
  return {
    running: false,
    error: null,
    source: "0",
    frame_width: 1280,
    frame_height: 720,
    frame_seq: 0,
    machine: { id: "excavator_01", name: "Excavator 01" },
    rules: { mode: "TIMER", dwell_seconds: 1 },
    zone_polygons: [],
    zone_points: [],
    allowed_people: 0,
    inside: 0,
    status: "SAFE",
    capacity_alert: false,
    worker_alert: false,
    message: "",
    workers: [],
    ...overrides,
  };
}

let config = CONFIG;
/** per-source config answers, so a test can give a clip its own zones */
let configBySource: Record<string, AppConfig> = {};
let state = makeState();
/** the uploaded videos /videos will answer with */
let videos: TestVideo[] = [];
/** the alert rows /events will answer with; ack POSTs flip them in place */
let alertRows: AlertEvent[] = [];
/** the fire/smoke rows /fire-alerts will answer with; ack POSTs flip them */
let fireRows: FireAlert[] = [];
/** when set, /events reports this total instead of the loaded rows, so the
 *  badge can be tested against a backlog bigger than one page */
let eventsTotalOverride: number | null = null;
/** extra history rows /events serves; the mock honours event_type, limit and offset */
let historyRows: AlertEvent[] = [];
/** fire/smoke rows /history merges in, in backend wire shape (kind + fire
 *  fields); the page normalises them, the mock does not have to */
let historyFireRows: Record<string, unknown>[] = [];
/** the answer for /events/filters */
let historyFilters = { workers: [] as number[], zones: [] as { index: number; name: string }[] };

/** uploads go through XMLHttpRequest, so the fake below stands in for it and
 *  lets a test drive the progress event and the response by hand */
let xhrInstances: FakeXHR[] = [];

class FakeXHR {
  upload: {
    onprogress?: (event: { lengthComputable: boolean; loaded: number; total: number }) => void;
  } = {};
  onload: (() => void) | null = null;
  onerror: (() => void) | null = null;
  status = 0;
  responseText = "";

  open() {
    /* the app only opens POST /videos */
  }

  send() {
    xhrInstances.push(this);
    // half way, so a progress bar has something to show before the reply
    this.upload.onprogress?.({ lengthComputable: true, loaded: 50, total: 100 });
  }

  respond(status: number, body: unknown) {
    this.status = status;
    this.responseText = JSON.stringify(body);
    this.onload?.();
  }
}

function makeVideo(overrides: Partial<TestVideo> = {}): TestVideo {
  return {
    id: "v1",
    file: "v1.mp4",
    name: "kiln.mp4",
    size_mb: 5,
    width: 640,
    height: 480,
    fps: 25,
    duration: 2,
    created: "2026-01-01T00:00:00Z",
    source: "/data/videos/v1.mp4",
    ...overrides,
  };
}

function jsonResponse(payload: unknown, status = 200) {
  return Promise.resolve({
    ok: status >= 200 && status < 300,
    status,
    json: () => Promise.resolve(payload),
  } as Response);
}

/** every JSON body POSTed to a url, oldest first */
function postedBodies(url: string): Record<string, unknown>[] {
  return fetchMock.mock.calls
    .map((call) => [String(call[0]), call[1] as RequestInit | undefined] as const)
    .filter(([called, init]) => called === url && init?.method === "POST")
    .map(([, init]) => JSON.parse(init?.body as string) as Record<string, unknown>);
}

const fetchMock = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
  const url = String(input);
  if (url.startsWith("/config")) {
    if (init?.method === "POST") return jsonResponse({ saved: true });
    const source = url.includes("?")
      ? new URLSearchParams(url.slice(url.indexOf("?") + 1)).get("source")
      : null;
    if (source && configBySource[source]) return jsonResponse(configBySource[source]);
    return jsonResponse(config);
  }
  if (url.startsWith("/start")) return jsonResponse({ running: true });
  if (url.startsWith("/stop")) return jsonResponse({ running: false });
  if (url.startsWith("/state")) return jsonResponse(state);
  if (url.startsWith("/events/filters")) return jsonResponse(historyFilters);
  if (url === "/events/ack-all" && init?.method === "POST") {
    let cleared = 0;
    alertRows = alertRows.map((row) => {
      if (!row.acknowledged) {
        cleared += 1;
        return { ...row, acknowledged: true, acknowledged_at: new Date().toISOString() };
      }
      return row;
    });
    if (eventsTotalOverride !== null) eventsTotalOverride = Math.max(0, eventsTotalOverride - cleared);
    return jsonResponse({ acknowledged: cleared });
  }
  if (url.startsWith("/history/export.csv")) {
    return jsonResponse("Time,Worker,Zone,Source,Event,Dwell (s),People inside,Acknowledged,Acknowledged at");
  }
  if (url.startsWith("/history")) {
    const query = new URLSearchParams(url.includes("?") ? url.slice(url.indexOf("?") + 1) : "");
    const types = query.getAll("event_type");
    let rows: Record<string, unknown>[] = [
      ...historyRows.map((row) => ({ ...row, kind: "zone" })),
      ...historyFireRows.map((row) => ({ ...row, kind: "fire" })),
    ];
    if (types.length > 0) {
      rows = rows.filter((row) =>
        row.kind === "fire"
          ? types.includes(String(row.alert_type ?? "").toUpperCase())
          : types.includes(String(row.event_type))
      );
    }
    const acked = query.get("acknowledged");
    if (acked !== null) rows = rows.filter((row) => row.acknowledged === (acked === "true"));
    const flagged = query.get("false_alarm");
    if (flagged !== null) {
      rows = rows.filter((row) =>
        row.kind === "fire" ? row.false_alarm === (flagged === "true") : flagged !== "true"
      );
    }
    const worker = query.get("worker_id");
    if (worker !== null) {
      rows = rows.filter((row) =>
        row.kind === "fire" ? false : row.worker_id === Number(worker)
      );
    }
    const zone = query.get("zone");
    if (zone !== null) {
      rows = rows.filter((row) =>
        row.kind === "fire"
          ? row.source_label === zone
          : /^-?\d+$/.test(zone)
            ? row.zone_index === Number(zone)
            : row.zone_name === zone
      );
    }
    const total = rows.length;
    const limit = Number(query.get("limit") ?? 100);
    const offset = Number(query.get("offset") ?? 0);
    rows = rows.slice(offset, offset + limit);
    return jsonResponse({ count: rows.length, total, rows });
  }
  if (url.startsWith("/events")) {
    const ack = url.match(/^\/events\/(\d+)\/ack$/);
    if (ack && init?.method === "POST") {
      const id = Number(ack[1]);
      alertRows = alertRows.map((row) =>
        row.id === id
          ? { ...row, acknowledged: true, acknowledged_at: new Date().toISOString() }
          : row
      );
      if (eventsTotalOverride !== null) eventsTotalOverride = Math.max(0, eventsTotalOverride - 1);
      return jsonResponse({ event: alertRows.find((row) => row.id === id) ?? null });
    }
    const query = new URLSearchParams(url.includes("?") ? url.slice(url.indexOf("?") + 1) : "");
    const types = query.getAll("event_type");
    let rows = [...alertRows, ...historyRows];
    if (types.length > 0) rows = rows.filter((row) => types.includes(row.event_type));
    const acked = query.get("acknowledged");
    if (acked !== null) rows = rows.filter((row) => row.acknowledged === (acked === "true"));
    const worker = query.get("worker_id");
    if (worker !== null) rows = rows.filter((row) => row.worker_id === Number(worker));
    const zone = query.get("zone");
    if (zone !== null) {
      rows = rows.filter((row) =>
        /^-?\d+$/.test(zone)
          ? row.zone_index === Number(zone)
          : row.zone_name === zone
      );
    }
    const total =
      eventsTotalOverride !== null && acked !== "true" ? eventsTotalOverride : rows.length;
    const limit = Number(query.get("limit") ?? 100);
    const offset = Number(query.get("offset") ?? 0);
    rows = rows.slice(offset, offset + limit);
    return jsonResponse({ count: rows.length, total, events: rows });
  }
  if (url.startsWith("/fire-alerts")) {
    const fa = url.match(/^\/fire-alerts\/(\d+)\/false-alarm$/);
    if (fa && init?.method === "POST") {
      const id = Number(fa[1]);
      const now = new Date().toISOString();
      fireRows = fireRows.map((row) =>
        row.id === id
          ? { ...row, acknowledged: true, acknowledged_at: now, false_alarm: true, false_alarm_at: now }
          : row
      );
      return jsonResponse({
        alert: fireRows.find((row) => row.id === id) ?? null,
        exported: { image: "fire_x.jpg", metadata: "fire_x.json" },
        duplicate: false,
      });
    }
    const ack = url.match(/^\/fire-alerts\/(\d+)\/ack$/);
    if (ack && init?.method === "POST") {
      const id = Number(ack[1]);
      fireRows = fireRows.map((row) =>
        row.id === id
          ? { ...row, acknowledged: true, acknowledged_at: new Date().toISOString() }
          : row
      );
      return jsonResponse({ alert: fireRows.find((row) => row.id === id) ?? null });
    }
    const query = new URLSearchParams(url.includes("?") ? url.slice(url.indexOf("?") + 1) : "");
    const acked = query.get("acknowledged");
    let rows = [...fireRows];
    if (acked !== null) rows = rows.filter((row) => row.acknowledged === (acked === "true"));
    return jsonResponse({
      alerts: rows,
      total: rows.length,
      unacked: fireRows.filter((row) => !row.acknowledged).length,
      acked: fireRows.filter((row) => row.acknowledged).length,
    });
  }
  if (url.startsWith("/videos/") && init?.method === "DELETE") {
    const id = url.slice("/videos/".length);
    videos = videos.filter((video) => video.id !== id);
    return jsonResponse({ deleted: id });
  }
  if (url === "/videos") return jsonResponse({ videos, limit: 10, max_mb: 500 });
  return jsonResponse({});
});

/** the base fetch routing, captured at load: afterEach's restoreAllMocks
 *  clears the explicit implementation slot (getMockImplementation goes
 *  undefined) while calls keep routing to it, so any test that wraps the
 *  mock must fall back to this reference, never to getMockImplementation. */
const baseFetchImpl = fetchMock.getMockImplementation()!;

/** a 1280x720 <img> on screen */
function mockImageRect(width = 1280, height = 720) {
  vi.spyOn(HTMLImageElement.prototype, "getBoundingClientRect").mockReturnValue({
    left: 0,
    top: 0,
    width,
    height,
    right: width,
    bottom: height,
    x: 0,
    y: 0,
    toJSON: () => ({}),
  } as DOMRect);
}

beforeEach(() => {
  // the app routes by hash, which survives across tests in one jsdom window:
  // without this every test after a navigation starts on the previous page
  window.location.hash = "#/";
  config = CONFIG;
  configBySource = {};
  state = makeState();
  videos = [];
  alertRows = [];
  fireRows = [];
  eventsTotalOverride = null;
  historyRows = [];
  historyFireRows = [];
  historyFilters = { workers: [], zones: [] };
  xhrInstances = [];
  fetchMock.mockClear();
  vi.stubGlobal("fetch", fetchMock);
  vi.stubGlobal("XMLHttpRequest", FakeXHR);
  // jsdom reports zero-sized rects, which would clamp every drawn point to 0,0
  mockImageRect();
});

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

function button(name: RegExp): HTMLButtonElement {
  return screen.getByRole("button", { name }) as HTMLButtonElement;
}

async function waitForUi() {
  await waitFor(() => expect(screen.getByTestId("live-page")).toBeTruthy());
}

/** the last JSON body POSTed to a url */
function lastBody(url: string): Record<string, unknown> {
  const calls = fetchMock.mock.calls
    .map((call) => [String(call[0]), call[1] as RequestInit | undefined] as const)
    .filter(([called, init]) => called === url && init?.method === "POST");
  expect(calls.length).toBeGreaterThan(0);
  return JSON.parse((calls[calls.length - 1][1] as RequestInit).body as string);
}

/** how many POSTs the app has sent to a URL (the initial GET does not count) */
function postCount(url: string): number {
  return fetchMock.mock.calls.filter(
    (call) => String(call[0]) === url && (call[1] as RequestInit | undefined)?.method === "POST"
  ).length;
}

/** how many times the app asked the backend to stop the camera */
function stops(): number {
  return fetchMock.mock.calls.filter(
    (call) => String(call[0]) === "/stop" && (call[1] as RequestInit | undefined)?.method === "POST"
  ).length;
}

/** open the camera, so the frame size and the feed are there */
async function startCamera(user: ReturnType<typeof userEvent.setup>) {
  const start = screen.queryByRole("button", { name: /^start camera$/i }) as HTMLButtonElement | null;
  if (!start || start.disabled) return;
  // the poll picks this up on its next tick
  state = { ...state, running: true };
  await user.click(start);
  await waitFor(() =>
    expect(screen.queryByRole("button", { name: /^start camera$/i })).toBeNull()
  );
}

/** the drawing toolbar is only there in Edit mode, so reach Edit first.
 *
 *  The control-room layout has no steps: Live shows the picture, and Edit is
 *  entered with one button. */
async function openToolbar(user: ReturnType<typeof userEvent.setup>) {
  await startCamera(user);
  const edit = screen.queryByRole("button", { name: /^edit zones$/i }) as HTMLButtonElement | null;
  if (edit && !edit.disabled) await user.click(edit);
  await waitFor(() => expect(screen.getByTestId("toolbar")).toBeTruthy());
}

/** the alert rules live on their own page now, behind the sidebar */
async function openSettings(user: ReturnType<typeof userEvent.setup>) {
  await user.click(screen.getByRole("button", { name: /settings/i }));
  await waitFor(() => expect(screen.getByTestId("settings-page")).toBeTruthy());
}

/** render and open the settings page, which is where the rules now live */
async function renderSettings() {
  const user = userEvent.setup();
  render(<App />);
  await waitForUi();
  await openSettings(user);
  return user;
}

/** render and open the editor on the saved zones */
async function renderEditing() {
  const user = userEvent.setup();
  state = { ...state, running: true };
  render(<App />);
  await waitForUi();
  await user.click(button(/^edit zones$/i));
  await waitFor(() => expect(screen.getByTestId("toolbar")).toBeTruthy());
  return user;
}

/** switch to a tool from the toolbar */
async function pickTool(user: ReturnType<typeof userEvent.setup>, name: RegExp) {
  await user.click(toolButton(name));
}

/** click the video at a frame coordinate */
function clickVideo(x: number, y: number) {
  fireEvent.click(screen.getByAltText("camera stream"), { clientX: x, clientY: y });
}

/** press on the picture, drag to (x, y), release */
function dragVideo(fromX: number, fromY: number, toX: number, toY: number) {
  const image = screen.getByAltText("camera stream");
  fireEvent.mouseDown(image, { clientX: fromX, clientY: fromY });
  fireEvent.mouseMove(window, { clientX: toX, clientY: toY });
  fireEvent.mouseUp(window, { clientX: toX, clientY: toY });
}

/** the zone names, one per card in the right column, e.g. ["Zone 1", "Zone 2"] */
function chipLabels(): string[] {
  return [...document.querySelectorAll(".zone-card-name")].map(
    (node) => (node.textContent ?? "").replace(/\s+/g, " ").trim()
  );
}

/** how many polygons the overlay is currently painting */
function shapeCount(): number {
  return document.querySelectorAll("[data-testid^='zone-shape-']").length;
}

/** the shape being dragged out right now, if any */
function rubberBand(): Element | null {
  return document.querySelector(".rubber-band");
}

/** the dashed outline of the polygon being clicked out */
function polygonPreview(): Element | null {
  return document.querySelector(".zone-preview");
}

/** the name on every zone tag, in order */
function tagLabels(): string[] {
  return [...document.querySelectorAll(".zone-tag-text")].map((node) => node.textContent ?? "");
}

/** where a zone's tag sits, as "x y" in frame pixels */
function tagPosition(index = 0): { x: number; y: number } {
  const tag = document.querySelector(`[data-testid='zone-tag-${index}']`);
  expect(tag).toBeTruthy();
  const match = /translate\((-?[\d.]+) (-?[\d.]+)\)/.exec(tag?.getAttribute("transform") ?? "");
  expect(match).toBeTruthy();
  return { x: Number(match?.[1]), y: Number(match?.[2]) };
}

/** the width a "Zone N" tag takes on a 1280-wide frame with the mocked image */
function tagWidth(text = "Zone 1"): number {
  return text.length * 16 * 0.56 + 16;
}

/** the corner count of the nth drawn zone, read off its SVG polygon */
function shapePoints(index = 0): string[] {
  const shape = document.querySelector(`[data-testid='zone-shape-${index}']`);
  expect(shape).toBeTruthy();
  return ((shape as SVGPolygonElement).getAttribute("points") ?? "").trim().split(/\s+/);
}

/** every toolbar tool button, in order */
function toolButtons(): HTMLElement[] {
  return within(screen.getByTestId("toolbar")).getAllByRole("button");
}

/** a toolbar button, so it cannot be confused with the same name in the side card */
function toolButton(name: RegExp): HTMLButtonElement {
  return within(screen.getByTestId("toolbar")).getByRole("button", { name }) as HTMLButtonElement;
}

/** the class attribute of an SVG node, which className does not give as a string */
function svgClass(node: Element | null): string {
  return node?.getAttribute("class") ?? "";
}

/** drags a rectangle out of the rectangle tool, in frame coordinates */
async function drawRectangle(
  user: ReturnType<typeof userEvent.setup>,
  from: [number, number],
  to: [number, number]
) {
  await pickTool(user, /^rectangle$/i);
  dragVideo(from[0], from[1], to[0], to[1]);
}

describe("Paint-style toolbar", () => {
  it("shows every tool with an icon and a label, in Paint order", async () => {
    render(<App />);
    await waitForUi();
    await openToolbar(userEvent.setup());

    expect(toolButtons().map((node) => node.getAttribute("aria-label"))).toEqual([
      "Select",
      "Rectangle",
      "4-point",
      "Polygon",
      "Circle",
      "Freehand",
      "Undo",
      "Delete",
      "Clear",
    ]);
    toolButtons().forEach((node) => {
      expect(node.querySelector(".tool-icon")).toBeTruthy();
      expect(node.querySelector(".tool-label")?.textContent).toBeTruthy();
    });
  });

  it("keeps exactly one tool active and marks it pressed", async () => {
    const user = userEvent.setup();
    render(<App />);
    await waitForUi();
    await openToolbar(user);

    await pickTool(user, /^4-point$/i);
    const pressed = toolButtons().filter((node) => node.getAttribute("aria-pressed") === "true");
    expect(pressed.length).toBe(1);
    expect(pressed[0].getAttribute("aria-label")).toBe("4-point");
    expect(pressed[0].className).toContain("active");
  });

  it("waits to be asked for, and never turns up on its own", async () => {
    const user = userEvent.setup();
    state = makeState({ running: true });
    render(<App />);
    await waitForUi();

    // a running camera with no zone is Live: nothing to save, so nothing is put
    // in front of the operator without being asked
    expect(screen.queryByTestId("toolbar")).toBeNull();
    expect(document.querySelector(".status-pill")?.textContent).toBe("RUNNING");

    await user.click(button(/^edit zones$/i));
    await waitFor(() => expect(screen.getByTestId("toolbar")).toBeTruthy());
  });

  it("stays away while the feed is stopped", async () => {
    render(<App />);
    await waitForUi();
    expect(screen.queryByTestId("toolbar")).toBeNull();
  });

  it("shows a crosshair while a shape tool is active and a plain cursor for Select", async () => {
    const user = userEvent.setup();
    render(<App />);
    await waitForUi();
    await openToolbar(user);

    await pickTool(user, /^rectangle$/i);
    expect(document.querySelector(".stage")?.className).toContain("tool-rectangle");
    await pickTool(user, /^select$/i);
    expect(document.querySelector(".stage")?.className).toContain("tool-select");
  });

  it("has no Freeze frame control, and always draws on the live feed", async () => {
    const user = userEvent.setup();
    render(<App />);
    await waitForUi();
    await openToolbar(user);

    expect(screen.queryByLabelText(/freeze frame/i)).toBeNull();
    expect(screen.queryByText(/still frame/i)).toBeNull();
    // drawing works over the live feed, which is the only picture there is
    expect((screen.getByAltText("camera stream") as HTMLImageElement).getAttribute("src")).toContain(
      "/stream"
    );
    expect(screen.queryByAltText("frozen snapshot")).toBeNull();
  });
});

describe("rectangle tool", () => {
  it("turns one press, drag and release into a 4 point zone", async () => {
    const user = userEvent.setup();
    render(<App />);
    await waitForUi();
    await openToolbar(user);

    await drawRectangle(user, [200, 150], [700, 500]);
    await waitFor(() => expect(shapeCount()).toBe(1));
    expect(shapePoints(0).length).toBe(4);
    // the chip under the video is the one place the count is shown
    expect(chipLabels()).toEqual(["Zone 1"]);
  });

  it("makes no zone out of a click that never moved", async () => {
    const user = userEvent.setup();
    render(<App />);
    await waitForUi();
    await openToolbar(user);

    await pickTool(user, /^rectangle$/i);
    fireEvent.mouseDown(screen.getByAltText("camera stream"), { clientX: 300, clientY: 300 });
    fireEvent.mouseUp(window, { clientX: 300, clientY: 300 });
    expect(shapeCount()).toBe(0);
  });
});

describe("circle tool", () => {
  it("turns a centre-to-edge drag into exactly 24 points", async () => {
    const user = userEvent.setup();
    render(<App />);
    await waitForUi();
    await openToolbar(user);

    await pickTool(user, /^circle$/i);
    dragVideo(640, 360, 840, 360);
    await waitFor(() => expect(shapeCount()).toBe(1));
    expect(shapePoints(0).length).toBe(24);
  });

  it("reads an uneven drag as an ellipse, so it is wider than it is tall", async () => {
    const user = userEvent.setup();
    render(<App />);
    await waitForUi();
    await openToolbar(user);

    await pickTool(user, /^circle$/i);
    dragVideo(640, 360, 940, 460);
    await waitFor(() => expect(shapeCount()).toBe(1));

    const points = shapePoints(0).map((pair) => pair.split(",").map(Number));
    const xs = points.map((p) => p[0]);
    const ys = points.map((p) => p[1]);
    const width = Math.max(...xs) - Math.min(...xs);
    const height = Math.max(...ys) - Math.min(...ys);
    expect(width).toBeGreaterThan(height * 1.5);
  });

  it("keeps a circle dragged past the edge whole and inside the frame", async () => {
    const user = userEvent.setup();
    render(<App />);
    await waitForUi();
    await openToolbar(user);

    await pickTool(user, /^circle$/i);
    // centre at 1000,300 with a 300px radius, dragged out to 1300,600 on a
    // 1280x720 feed, so the right of the circle would hang off the picture
    dragVideo(1000, 300, 1300, 600);
    await waitFor(() => expect(shapeCount()).toBe(1));

    const points = shapePoints(0).map((pair) => pair.split(",").map(Number));
    const xs = points.map((p) => p[0]);
    const ys = points.map((p) => p[1]);

    // still a circle: flattening it against the edge would leave a sliver
    const width = Math.max(...xs) - Math.min(...xs);
    const height = Math.max(...ys) - Math.min(...ys);
    expect(Math.abs(width - height)).toBeLessThanOrEqual(2);
    // and it is a circle rather than a handful of corners piled up on one pixel
    expect(points.length).toBeGreaterThanOrEqual(20);

    for (const [x, y] of points) {
      expect(x).toBeGreaterThanOrEqual(0);
      expect(x).toBeLessThanOrEqual(1280);
      expect(y).toBeGreaterThanOrEqual(0);
      expect(y).toBeLessThanOrEqual(720);
    }
    // no two neighbours sit on the same pixel, which is what the API rejects
    for (let i = 1; i < points.length; i += 1) {
      expect(points[i]).not.toEqual(points[i - 1]);
    }
  });
});

describe("freehand tool", () => {
  it("turns a scribble into a simplified zone under the 32 point ceiling", async () => {
    const user = userEvent.setup();
    render(<App />);
    await waitForUi();
    await openToolbar(user);

    await pickTool(user, /^freehand$/i);
    const image = screen.getByAltText("camera stream");
    // a wobbly closed loop: 60 samples around a circle
    fireEvent.mouseDown(image, { clientX: 640, clientY: 200 });
    for (let i = 1; i <= 60; i += 1) {
      const angle = (2 * Math.PI * i) / 60;
      fireEvent.mouseMove(window, {
        clientX: 640 + 300 * Math.cos(angle),
        clientY: 360 + 200 * Math.sin(angle),
      });
    }
    fireEvent.mouseUp(window, { clientX: 640, clientY: 560 });

    await waitFor(() => expect(shapeCount()).toBe(1));
    const count = shapePoints(0).length;
    expect(count).toBeGreaterThanOrEqual(3);
    expect(count).toBeLessThanOrEqual(24);
  });

  it("drops a stroke too small to be a zone", async () => {
    const user = userEvent.setup();
    render(<App />);
    await waitForUi();
    await openToolbar(user);

    await pickTool(user, /^freehand$/i);
    const image = screen.getByAltText("camera stream");
    fireEvent.mouseDown(image, { clientX: 300, clientY: 300 });
    fireEvent.mouseMove(window, { clientX: 302, clientY: 301 });
    fireEvent.mouseUp(window, { clientX: 302, clientY: 301 });
    expect(shapeCount()).toBe(0);
  });

  it("paints the stroke live while dragging and clears it on release", async () => {
    const user = userEvent.setup();
    render(<App />);
    await waitForUi();
    await openToolbar(user);

    await pickTool(user, /^freehand$/i);
    const image = screen.getByAltText("camera stream");
    fireEvent.mouseDown(image, { clientX: 400, clientY: 300 });
    // nothing painted yet: the press point itself is not sampled
    expect(screen.queryByTestId("stroke-preview")).toBeNull();
    fireEvent.mouseMove(window, { clientX: 500, clientY: 300 });
    fireEvent.mouseMove(window, { clientX: 500, clientY: 400 });
    // the stroke draws live in the preview colour as the pointer moves
    const preview = screen.getByTestId("stroke-preview");
    expect(preview.tagName).toBe("polyline");
    expect(preview.getAttribute("points")?.split(" ").length).toBeGreaterThan(1);
    // too small to be a zone, but the preview still clears on release
    fireEvent.mouseUp(window, { clientX: 500, clientY: 400 });
    expect(screen.queryByTestId("stroke-preview")).toBeNull();
    expect(shapeCount()).toBe(0);
  });

  it("says why a stroke that crosses itself cannot be a zone", async () => {
    const user = userEvent.setup();
    render(<App />);
    await waitForUi();
    await openToolbar(user);

    await pickTool(user, /^freehand$/i);
    const image = screen.getByAltText("camera stream");
    // a bow tie: the two diagonals cross in the middle. The press point
    // itself is not sampled, so the stroke returns to the start to close it.
    fireEvent.mouseDown(image, { clientX: 500, clientY: 200 });
    fireEvent.mouseMove(window, { clientX: 700, clientY: 600 });
    fireEvent.mouseMove(window, { clientX: 500, clientY: 600 });
    fireEvent.mouseMove(window, { clientX: 700, clientY: 200 });
    fireEvent.mouseMove(window, { clientX: 500, clientY: 200 });
    fireEvent.mouseUp(window, { clientX: 500, clientY: 200 });

    expect(shapeCount()).toBe(0);
    // and the app says why, instead of the stroke vanishing silently
    expect(screen.getByText(/crosses over itself/i)).toBeTruthy();
  });
});

describe("4-point tool", () => {
  /** corners of the open shape, as painted dots */
  function draftPointCount(): number {
    return document.querySelectorAll(
      '[data-testid="draft-first-point"],[data-testid^="draft-point-"]'
    ).length;
  }

  /** four corners of a clean quad, in order */
  function clickQuad() {
    clickVideo(300, 300);
    clickVideo(600, 300);
    clickVideo(600, 600);
    clickVideo(300, 600);
  }

  it("shows the hint while drawing", async () => {
    const user = userEvent.setup();
    render(<App />);
    await waitForUi();
    await openToolbar(user);

    await pickTool(user, /^4-point$/i);
    expect(screen.getByTestId("quad-hint").textContent).toMatch(/click 4 corners/i);
  });

  it("adds exactly one point per click", async () => {
    const user = userEvent.setup();
    render(<App />);
    await waitForUi();
    await openToolbar(user);

    await pickTool(user, /^4-point$/i);
    clickVideo(300, 300);
    expect(draftPointCount()).toBe(1);
    expect(shapeCount()).toBe(0);

    clickVideo(600, 300);
    expect(draftPointCount()).toBe(2);
  });

  it("creates the zone on the 4th click", async () => {
    const user = userEvent.setup();
    render(<App />);
    await waitForUi();
    await openToolbar(user);

    await pickTool(user, /^4-point$/i);
    clickQuad();
    await waitFor(() => expect(shapeCount()).toBe(1));
    expect(shapePoints(0).length).toBe(4);
    expect(tagLabels()).toEqual(["Zone 1"]);
  });

  it("commits nothing on three clicks, and Enter does not finish", async () => {
    const user = userEvent.setup();
    render(<App />);
    await waitForUi();
    await openToolbar(user);

    await pickTool(user, /^4-point$/i);
    clickVideo(300, 300);
    clickVideo(600, 300);
    clickVideo(600, 600);
    expect(shapeCount()).toBe(0);
    expect(draftPointCount()).toBe(3);

    // no second gesture finishes a quad: Enter leaves the draft alone
    fireEvent.keyDown(window, { key: "Enter" });
    expect(shapeCount()).toBe(0);
    expect(draftPointCount()).toBe(3);
  });

  it("ignores a click within jitter of the last corner", async () => {
    const user = userEvent.setup();
    render(<App />);
    await waitForUi();
    await openToolbar(user);

    await pickTool(user, /^4-point$/i);
    clickVideo(400, 300);
    // a shaky second press a few pixels off: not a new corner
    clickVideo(402, 301);
    expect(draftPointCount()).toBe(1);
    expect(shapeCount()).toBe(0);
    expect(screen.getByText(/same corner as the last click/i)).toBeTruthy();

    // and the shape still finishes cleanly afterwards
    clickVideo(600, 300);
    clickVideo(600, 600);
    clickVideo(400, 600);
    await waitFor(() => expect(shapeCount()).toBe(1));
    expect(shapePoints(0).length).toBe(4);
  });

  it("refuses a crossing 4th corner and keeps the first three", async () => {
    const user = userEvent.setup();
    render(<App />);
    await waitForUi();
    await openToolbar(user);

    await pickTool(user, /^4-point$/i);
    clickVideo(200, 200);
    clickVideo(600, 200);
    clickVideo(200, 600);
    // the diagonal back across makes a bow-tie: refused, nothing committed
    clickVideo(600, 600);
    await waitFor(() => expect(screen.getByText(/crosses over itself/i)).toBeTruthy());
    expect(shapeCount()).toBe(0);
    // the first three are still on screen, so the next click is simply a
    // different 4th corner
    expect(draftPointCount()).toBe(3);

    clickVideo(200, 400);
    await waitFor(() => expect(shapeCount()).toBe(1));
    expect(shapePoints(0).length).toBe(4);
  });

  it("draws on the stopped canvas: four clicks make the zone", async () => {
    config = { ...CONFIG, zones: [], zone_reference: [1280, 720] };
    const user = userEvent.setup();
    render(<App />);
    await waitForUi();
    // jsdom gives divs a zero rect; the canvas needs a real one to map clicks
    vi.spyOn(HTMLDivElement.prototype, "getBoundingClientRect").mockReturnValue({
      left: 0,
      top: 0,
      width: 1280,
      height: 720,
      right: 1280,
      bottom: 720,
      x: 0,
      y: 0,
      toJSON: () => ({}),
    } as DOMRect);
    await user.click(button(/^edit zones$/i));
    await waitFor(() => expect(screen.getByTestId("blank-canvas")).toBeTruthy());
    await pickTool(user, /^4-point$/i);

    const canvas = screen.getByTestId("blank-canvas");
    fireEvent.click(canvas, { clientX: 300, clientY: 300 });
    expect(draftPointCount()).toBe(1);
    fireEvent.click(canvas, { clientX: 600, clientY: 300 });
    fireEvent.click(canvas, { clientX: 600, clientY: 600 });
    expect(draftPointCount()).toBe(3);
    // the 4th click makes the zone by itself: no closing click needed
    fireEvent.click(canvas, { clientX: 300, clientY: 600 });
    await waitFor(() => expect(shapeCount()).toBe(1));
    expect(shapePoints(0).length).toBe(4);
  });

  it("throws the open shape away with Escape but keeps the finished ones", async () => {
    const user = userEvent.setup();
    render(<App />);
    await waitForUi();
    await openToolbar(user);

    // one finished zone first, so Escape has something it must not destroy
    await drawRectangle(user, [100, 100], [400, 300]);
    await waitFor(() => expect(shapeCount()).toBe(1));

    await pickTool(user, /^4-point$/i);
    clickVideo(600, 400);
    clickVideo(900, 400);
    clickVideo(900, 650);
    fireEvent.keyDown(window, { key: "Escape" });
    await waitFor(() => expect(screen.getByText(/drawing cancelled/i)).toBeTruthy());
    // the half drawn shape is gone, the saved-looking rectangle is not
    expect(shapeCount()).toBe(1);
  });

  it("removes the last placed corner with Backspace", async () => {
    const user = userEvent.setup();
    render(<App />);
    await waitForUi();
    await openToolbar(user);

    await pickTool(user, /^4-point$/i);
    clickVideo(300, 300);
    clickVideo(600, 300);
    clickVideo(600, 600);
    fireEvent.keyDown(window, { key: "Backspace" });
    expect(draftPointCount()).toBe(2);

    // re-click a different third corner and finish: still four in the zone
    clickVideo(650, 550);
    clickVideo(300, 600);
    await waitFor(() => expect(shapeCount()).toBe(1));
    expect(shapePoints(0).length).toBe(4);
  });
});

describe("polygon tool (N-point, DMS-style)", () => {
  /** corners of the open shape, as painted dots */
  function draftPointCount(): number {
    return document.querySelectorAll(
      '[data-testid="draft-first-point"],[data-testid^="draft-point-"]'
    ).length;
  }

  it("shows the hint and a Finish button while drawing", async () => {
    const user = userEvent.setup();
    render(<App />);
    await waitForUi();
    await openToolbar(user);

    await pickTool(user, /^polygon$/i);
    expect(screen.getByTestId("polygon-hint").textContent).toMatch(/click to add corners/i);
    expect(screen.getByRole("button", { name: /^finish zone$/i })).toBeTruthy();
  });

  it("keeps collecting corners past four without committing", async () => {
    const user = userEvent.setup();
    render(<App />);
    await waitForUi();
    await openToolbar(user);

    await pickTool(user, /^polygon$/i);
    clickVideo(200, 150);
    clickVideo(500, 150);
    clickVideo(650, 350);
    clickVideo(500, 550);
    clickVideo(200, 550);
    expect(draftPointCount()).toBe(5);
    expect(shapeCount()).toBe(0);

    // a sixth corner still does not finish by itself, unlike the 4-point tool
    clickVideo(120, 350);
    expect(draftPointCount()).toBe(6);
    expect(shapeCount()).toBe(0);
  });

  it("finishes a six-corner zone with Enter", async () => {
    const user = userEvent.setup();
    render(<App />);
    await waitForUi();
    await openToolbar(user);

    await pickTool(user, /^polygon$/i);
    clickVideo(200, 150);
    clickVideo(500, 150);
    clickVideo(650, 350);
    clickVideo(500, 550);
    clickVideo(200, 550);
    clickVideo(120, 350);
    fireEvent.keyDown(window, { key: "Enter" });
    await waitFor(() => expect(shapeCount()).toBe(1));
    expect(shapePoints(0).length).toBe(6);
    expect(tagLabels()).toEqual(["Zone 1"]);
  });

  it("finishes with the Finish zone button", async () => {
    const user = userEvent.setup();
    render(<App />);
    await waitForUi();
    await openToolbar(user);

    await pickTool(user, /^polygon$/i);
    clickVideo(200, 150);
    clickVideo(500, 150);
    clickVideo(650, 350);
    clickVideo(500, 550);
    clickVideo(200, 550);
    await user.click(screen.getByRole("button", { name: /^finish zone$/i }));
    await waitFor(() => expect(shapeCount()).toBe(1));
    expect(shapePoints(0).length).toBe(5);
  });

  it("closes by clicking back near the first corner", async () => {
    const user = userEvent.setup();
    render(<App />);
    await waitForUi();
    await openToolbar(user);

    await pickTool(user, /^polygon$/i);
    clickVideo(200, 150);
    clickVideo(500, 150);
    clickVideo(650, 350);
    clickVideo(500, 550);
    clickVideo(200, 550);
    // back on the first corner: closes instead of adding a duplicate
    clickVideo(200, 150);
    await waitFor(() => expect(shapeCount()).toBe(1));
    expect(shapePoints(0).length).toBe(5);
  });

  it("finishes with a double-click", async () => {
    const user = userEvent.setup();
    render(<App />);
    await waitForUi();
    await openToolbar(user);

    await pickTool(user, /^polygon$/i);
    clickVideo(200, 150);
    clickVideo(500, 150);
    clickVideo(650, 350);
    clickVideo(500, 550);
    clickVideo(200, 550);
    fireEvent.dblClick(screen.getByAltText("camera stream"));
    await waitFor(() => expect(shapeCount()).toBe(1));
    expect(shapePoints(0).length).toBe(5);
  });

  it("says more corners are needed when finished too early", async () => {
    const user = userEvent.setup();
    render(<App />);
    await waitForUi();
    await openToolbar(user);

    await pickTool(user, /^polygon$/i);
    clickVideo(300, 300);
    clickVideo(600, 300);
    fireEvent.keyDown(window, { key: "Enter" });
    await waitFor(() =>
      expect(screen.getByText(/needs at least 3 points - keep clicking/i)).toBeTruthy()
    );
    expect(shapeCount()).toBe(0);
    // the two corners stay on screen, so work is not lost
    expect(draftPointCount()).toBe(2);
  });

  it("removes the last placed corner with Backspace", async () => {
    const user = userEvent.setup();
    render(<App />);
    await waitForUi();
    await openToolbar(user);

    await pickTool(user, /^polygon$/i);
    clickVideo(200, 150);
    clickVideo(500, 150);
    clickVideo(650, 350);
    fireEvent.keyDown(window, { key: "Backspace" });
    expect(draftPointCount()).toBe(2);
  });
});

describe("saved polygons with more than 4 points", () => {
  /** six corners around the centre of a 1280x720 frame, in order: the shape
   *  the old Polygon tool used to draw, still valid everywhere */
  const HEXAGON: [number, number][] = [0, 1, 2, 3, 4, 5].map((i) => {
    const angle = (2 * Math.PI * i) / 6;
    return [Math.round(640 + 250 * Math.cos(angle)), Math.round(360 + 220 * Math.sin(angle))];
  });

  it("loads a saved 6-point zone, shows it and moves it with Select", async () => {
    config = { ...CONFIG, zones: [HEXAGON.map(([x, y]) => [x, y] as Point)] };
    state = { ...state, running: true };
    const user = userEvent.setup();
    render(<App />);
    await waitForUi();
    await openToolbar(user);

    // all six corners on screen, untouched by the 4-point tool existing
    await waitFor(() => expect(shapeCount()).toBe(1));
    expect(shapePoints(0).length).toBe(6);

    // Select grabs the whole shape and drags it elsewhere. The press goes
    // to the picture itself: in a real browser clicks pass through the
    // overlay (pointer-events), while jsdom has no hit-testing, so tests
    // must target the image, not the shape element.
    await pickTool(user, /^select$/i);
    const image = screen.getByAltText("camera stream");
    const before = shapePoints(0).join(" ");
    fireEvent.mouseDown(image, { clientX: 640, clientY: 360 });
    fireEvent.mouseMove(window, { clientX: 740, clientY: 410 });
    fireEvent.mouseUp(window, { clientX: 740, clientY: 410 });
    await waitFor(() => expect(shapePoints(0).join(" ")).not.toBe(before));
    expect(shapePoints(0).length).toBe(6);
  });
  it("edits a saved 6-point zone by its corner handles", async () => {
    config = { ...CONFIG, zones: [HEXAGON.map(([x, y]) => [x, y] as Point)] };
    state = { ...state, running: true };
    const user = userEvent.setup();
    render(<App />);
    await waitForUi();
    await openToolbar(user);
    await waitFor(() => expect(shapeCount()).toBe(1));

    // Select the zone, then drag its first corner elsewhere. Handles sit
    // on the overlay itself, so they are a real event target; the press
    // that selects first goes to the picture (see above).
    await pickTool(user, /^select$/i);
    const image = screen.getByAltText("camera stream");
    fireEvent.mouseDown(image, { clientX: 640, clientY: 360 });
    fireEvent.mouseUp(image, { clientX: 640, clientY: 360 });
    await waitFor(() =>
      expect(document.querySelectorAll("circle.handle-hit").length).toBe(6)
    );
    const before = shapePoints(0).join(" ");
    fireEvent.mouseDown(document.querySelectorAll("circle.handle-hit")[0] as Element, {
      clientX: HEXAGON[0][0],
      clientY: HEXAGON[0][1],
    });
    fireEvent.mouseMove(window, { clientX: HEXAGON[0][0] + 60, clientY: HEXAGON[0][1] + 40 });
    fireEvent.mouseUp(window, { clientX: HEXAGON[0][0] + 60, clientY: HEXAGON[0][1] + 40 });
    await waitFor(() => expect(shapePoints(0).join(" ")).not.toBe(before));
    expect(shapePoints(0).length).toBe(6);
  });

  it("adds a corner to a saved 6-point zone through an edge midpoint", async () => {
    config = { ...CONFIG, zones: [HEXAGON.map(([x, y]) => [x, y] as Point)] };
    state = { ...state, running: true };
    const user = userEvent.setup();
    render(<App />);
    await waitForUi();
    await openToolbar(user);
    await waitFor(() => expect(shapeCount()).toBe(1));

    await pickTool(user, /^select$/i);
    // clicking the zone selects it, which paints one "+" per edge
    const image = screen.getByAltText("camera stream");
    fireEvent.mouseDown(image, { clientX: 640, clientY: 360 });
    fireEvent.mouseUp(image, { clientX: 640, clientY: 360 });
    await waitFor(() =>
      expect(document.querySelectorAll(".midpoint").length).toBe(6)
    );
    await user.click(document.querySelectorAll(".midpoint")[0] as Element);
    await waitFor(() => expect(shapePoints(0).length).toBe(7));
  });

  it("saves a 6-point zone and still draws it after stop, start and reload", async () => {
    config = { ...CONFIG, zones: [HEXAGON.map(([x, y]) => [x, y] as Point)] };
    state = { ...state, running: true };
    const user = userEvent.setup();
    render(<App />);
    await waitForUi();
    await waitFor(() => expect(shapeCount()).toBe(1));
    expect(shapePoints(0).length).toBe(6);

    await openToolbar(user);
    await user.click(button(/^save$/i));
    await waitFor(() => expect(postCount("/config")).toBeGreaterThan(0));
    const body = lastBody("/config");
    expect((body.zones as number[][][])[0].length).toBe(6);
    // still drawn right after Save
    await waitFor(() => expect(shapeCount()).toBe(1));

    // stopping the camera parks the live picture; restarting it draws the
    // saved hexagon again
    state = { ...state, running: false };
    await waitFor(() => expect(screen.getByRole("button", { name: /^start camera$/i })).toBeTruthy());
    state = { ...state, running: true };
    await waitFor(() => expect(shapeCount()).toBe(1));
    expect(shapePoints(0).length).toBe(6);

    // a page reload re-reads the saved zones from /config
    cleanup();
    render(<App />);
    await waitForUi();
    await waitFor(() => expect(shapeCount()).toBe(1));
    expect(shapePoints(0).length).toBe(6);
  });
});

describe("several zones", () => {
  it("keeps every drawn zone instead of replacing the last one", async () => {
    const user = userEvent.setup();
    render(<App />);
    await waitForUi();
    await openToolbar(user);

    await drawRectangle(user, [100, 100], [400, 300]);
    await waitFor(() => expect(shapeCount()).toBe(1));
    await drawRectangle(user, [600, 400], [1000, 650]);
    await waitFor(() => expect(shapeCount()).toBe(2));
    expect(chipLabels()).toEqual(["Zone 1", "Zone 2"]);
  });

  it("lists the zones and lets one be picked for editing", async () => {
    const user = userEvent.setup();
    render(<App />);
    await waitForUi();
    await openToolbar(user);

    await drawRectangle(user, [100, 100], [400, 300]);
    await drawRectangle(user, [600, 400], [1000, 650]);
    const list = await screen.findByTestId("zone-cards");
    expect(within(list).getAllByRole("button", { name: /^edit zone/i })).toHaveLength(2);

    await user.click(within(list).getByRole("button", { name: /edit zone 1/i }));
    await waitFor(() =>
      expect(svgClass(document.querySelector("[data-testid='zone-shape-0']"))).toContain("selected")
    );
  });

  it("saves every zone to the backend, so anyone in any of them counts", async () => {
    const user = userEvent.setup();
    render(<App />);
    await waitForUi();
    await openToolbar(user);

    await drawRectangle(user, [100, 100], [400, 300]);
    await drawRectangle(user, [600, 400], [1000, 650]);
    await user.click(button(/^save$/i));

    await waitFor(() => expect(fetchMock.mock.calls.some((c) => String(c[0]) === "/config")).toBe(true));
    const body = lastBody("/config");
    const zones = body.zones as number[][][];
    expect(zones.length).toBe(2);
    expect(zones[0].length).toBe(4);
    expect(zones[1].length).toBe(4);
    // the legacy single-zone field is gone: zones is canonical now
    expect(body.points).toBeUndefined();
  });
});

describe("the first zone, before anything is saved", () => {
  it("paints the box being dragged, not only after the release", async () => {
    const user = userEvent.setup();
    state = makeState({ running: true });
    render(<App />);
    await waitForUi();
    await openToolbar(user);

    await pickTool(user, /^rectangle$/i);
    expect(shapeCount()).toBe(0);
    const image = screen.getByAltText("camera stream");
    fireEvent.mouseDown(image, { clientX: 200, clientY: 150 });
    fireEvent.mouseMove(window, { clientX: 700, clientY: 500 });

    // the overlay is already on the picture, with nothing saved
    expect(screen.getByTestId("zone-stage")).toBeTruthy();
    expect(rubberBand()?.tagName).toBe("rect");
    expect((rubberBand() as SVGRectElement).getAttribute("width")).toBe("500");
    expect(shapeCount()).toBe(0);
    expect(tagLabels()).toEqual([]);
    fireEvent.mouseUp(window, { clientX: 700, clientY: 500 });

    await waitFor(() => expect(shapeCount()).toBe(1));
    expect(rubberBand()).toBeNull();
    expect(tagLabels()).toEqual(["Zone 1"]);
  });

  it("paints the circle being dragged", async () => {
    const user = userEvent.setup();
    state = makeState({ running: true });
    render(<App />);
    await waitForUi();
    await openToolbar(user);

    await pickTool(user, /^circle$/i);
    const image = screen.getByAltText("camera stream");
    fireEvent.mouseDown(image, { clientX: 300, clientY: 200 });
    fireEvent.mouseMove(window, { clientX: 500, clientY: 400 });

    expect(rubberBand()?.tagName).toBe("ellipse");
    expect(shapeCount()).toBe(0);
    fireEvent.mouseUp(window, { clientX: 500, clientY: 400 });

    await waitFor(() => expect(shapeCount()).toBe(1));
    expect(tagLabels()).toEqual(["Zone 1"]);
  });

  it("paints the quad while its corners are clicked out", async () => {
    const user = userEvent.setup();
    state = makeState({ running: true });
    render(<App />);
    await waitForUi();
    await openToolbar(user);

    await pickTool(user, /^4-point$/i);
    clickVideo(200, 150);
    expect(polygonPreview()).toBeTruthy();

    clickVideo(700, 150);
    clickVideo(700, 500);
    expect((polygonPreview() as SVGPolylineElement).getAttribute("points")).toBeTruthy();
    expect(shapeCount()).toBe(0);

    // the 4th click makes the zone by itself
    clickVideo(200, 500);
    await waitFor(() => expect(shapeCount()).toBe(1));
    expect(polygonPreview()).toBeNull();
    expect(tagLabels()).toEqual(["Zone 1"]);
  });

  it("keeps the painted zone after it is saved", async () => {
    const user = userEvent.setup();
    state = makeState({ running: true });
    render(<App />);
    await waitForUi();
    await openToolbar(user);

    await drawRectangle(user, [200, 150], [700, 500]);
    await user.click(button(/^save$/i));
    await waitFor(() => expect(fetchMock.mock.calls.some((c) => String(c[0]) === "/config")).toBe(true));

    expect(shapeCount()).toBe(1);
    expect(tagLabels()).toEqual(["Zone 1"]);
  });
});

describe("zone labels", () => {
  it("names every zone on the overlay, and the chips agree", async () => {
    const user = userEvent.setup();
    state = makeState({ running: true });
    render(<App />);
    await waitForUi();
    await openToolbar(user);

    await drawRectangle(user, [100, 100], [400, 300]);
    await drawRectangle(user, [600, 400], [1000, 650]);

    expect(tagLabels()).toEqual(["Zone 1", "Zone 2"]);
    expect(chipLabels()).toEqual(["Zone 1", "Zone 2"]);
  });

  it("puts each tag in its own zone's top-left corner", async () => {
    const user = userEvent.setup();
    state = makeState({ running: true });
    render(<App />);
    await waitForUi();
    await openToolbar(user);

    await drawRectangle(user, [100, 100], [400, 300]);
    await drawRectangle(user, [600, 400], [1000, 650]);

    expect(tagPosition(0)).toEqual({ x: 100, y: 100 });
    expect(tagPosition(1)).toEqual({ x: 600, y: 400 });
  });

  it("centres a tag on a zone too narrow to hold it, and keeps it in the frame", async () => {
    const user = userEvent.setup();
    state = makeState({ running: true });
    render(<App />);
    await waitForUi();
    await openToolbar(user);

    // 30px wide, against a tag about 70px wide
    await drawRectangle(user, [500, 300], [530, 500]);

    const { x, y } = tagPosition(0);
    expect(y).toBe(300);
    expect(x + tagWidth() / 2).toBeCloseTo(515, 1);
    expect(x).toBeGreaterThanOrEqual(0);
    expect(x + tagWidth()).toBeLessThanOrEqual(1280);
  });

  it("names the machine once, in the video column header, and not on the zones", async () => {
    const user = userEvent.setup();
    state = makeState({ running: true });
    render(<App />);
    await waitForUi();
    await openToolbar(user);
    await drawRectangle(user, [200, 150], [700, 500]);

    const names = document.querySelectorAll('[data-testid="machine-name"]');
    expect(names).toHaveLength(1);
    expect(names[0].textContent).toBe("Excavator 01");
    expect(names[0].closest(".video-col")).toBeTruthy();
    expect(tagLabels()).toEqual(["Zone 1"]);
    expect(document.querySelector(".overlay")?.textContent).not.toMatch(/excavator/i);
  });
});

describe("select tool", () => {
  it("moves the whole zone when its body is dragged", async () => {
    const user = userEvent.setup();
    render(<App />);
    await waitForUi();
    await openToolbar(user);

    await drawRectangle(user, [300, 300], [600, 500]);
    await waitFor(() => expect(shapeCount()).toBe(1));

    await pickTool(user, /^select$/i);
    dragVideo(450, 400, 550, 450);
    // the whole shape slides by the same 100,50
    await waitFor(() =>
      expect(shapePoints(0)).toEqual(["400,350", "700,350", "700,550", "400,550"])
    );
  });

  it("follows the pointer exactly across several mousemoves, without accelerating", async () => {
    const user = userEvent.setup();
    render(<App />);
    await waitForUi();
    await openToolbar(user);

    await drawRectangle(user, [300, 300], [600, 500]);
    await waitFor(() => expect(shapeCount()).toBe(1));
    await pickTool(user, /^select$/i);

    const image = screen.getByAltText("camera stream");
    // one press, then the pointer walks to the final spot over three moves: the
    // shape must end up shifted by the total 100,50, not by a running sum
    fireEvent.mouseDown(image, { clientX: 450, clientY: 400 });
    fireEvent.mouseMove(window, { clientX: 480, clientY: 410 });
    fireEvent.mouseMove(window, { clientX: 520, clientY: 430 });
    fireEvent.mouseMove(window, { clientX: 550, clientY: 450 });
    fireEvent.mouseUp(window, { clientX: 550, clientY: 450 });

    await waitFor(() =>
      expect(shapePoints(0)).toEqual(["400,350", "700,350", "700,550", "400,550"])
    );
  });

  it("drags one corner to reshape the zone", async () => {
    const user = userEvent.setup();
    render(<App />);
    await waitForUi();
    await openToolbar(user);

    await drawRectangle(user, [300, 300], [600, 500]);
    await waitFor(() => expect(shapeCount()).toBe(1));
    await pickTool(user, /^select$/i);

    const handles = document.querySelectorAll("circle.handle-hit");
    expect(handles.length).toBe(4);
    fireEvent.mouseDown(handles[0], { clientX: 300, clientY: 300 });
    fireEvent.mouseMove(window, { clientX: 250, clientY: 250 });
    fireEvent.mouseUp(window, { clientX: 250, clientY: 250 });

    await waitFor(() => expect(shapePoints(0)[0]).toBe("250,250"));
  });

  it("adds a corner when an edge midpoint plus is clicked", async () => {
    const user = userEvent.setup();
    render(<App />);
    await waitForUi();
    await openToolbar(user);

    await drawRectangle(user, [300, 300], [600, 500]);
    await waitFor(() => expect(shapeCount()).toBe(1));
    await pickTool(user, /^select$/i);
    fireEvent.click(document.querySelector("[data-testid='zone-shape-0']") as Element, {
      clientX: 450,
      clientY: 400,
    });
    await waitFor(() => expect(document.querySelectorAll(".midpoint").length).toBe(4));

    await user.click(document.querySelectorAll(".midpoint")[0] as Element);
    await waitFor(() => expect(shapePoints(0).length).toBe(5));
  });

  it("removes a corner with Delete, then the whole shape when no corner is picked", async () => {
    const user = userEvent.setup();
    render(<App />);
    await waitForUi();
    await openToolbar(user);

    await drawRectangle(user, [300, 300], [600, 500]);
    await waitFor(() => expect(shapeCount()).toBe(1));
    await pickTool(user, /^select$/i);
    fireEvent.click(document.querySelector("[data-testid='zone-shape-0']") as Element, {
      clientX: 450,
      clientY: 400,
    });
    await waitFor(() => expect(document.querySelectorAll("circle.handle-hit").length).toBe(4));

    fireEvent.mouseDown(document.querySelectorAll(".handle-hit")[0] as Element, {
      clientX: 300,
      clientY: 300,
    });
    fireEvent.mouseUp(window, { clientX: 300, clientY: 300 });
    fireEvent.keyDown(window, { key: "Delete" });
    await waitFor(() => expect(shapePoints(0).length).toBe(3));

    fireEvent.keyDown(window, { key: "Delete" });
    await waitFor(() => expect(shapeCount()).toBe(0));
  });

  it("never lets Delete take a zone below three corners", async () => {
    const user = userEvent.setup();
    render(<App />);
    await waitForUi();
    await openToolbar(user);

    await pickTool(user, /^4-point$/i);
    clickVideo(300, 300);
    clickVideo(700, 300);
    clickVideo(700, 600);
    clickVideo(300, 300);
    await waitFor(() => expect(shapeCount()).toBe(1));

    // with only three corners the row remove button is disabled, so the table
    // cannot shrink a zone into something the backend would reject
    const details = await openZoneCoordinates(user);
    const removes = within(details).getAllByRole("button", { name: /^remove point/i });
    expect(removes).toHaveLength(3);
    removes.forEach((node) => expect((node as HTMLButtonElement).disabled).toBe(true));

    // Delete with no corner picked takes the whole shape instead of a corner
    await pickTool(user, /^select$/i);
    await user.click(button(/^edit zone 1$/i));
    fireEvent.keyDown(window, { key: "Delete" });
    await waitFor(() => expect(shapeCount()).toBe(0));
  });
});

describe("undo, clear and clamp", () => {
  it("steps back a drawn zone with Ctrl+Z", async () => {
    const user = userEvent.setup();
    render(<App />);
    await waitForUi();
    await openToolbar(user);

    await drawRectangle(user, [100, 100], [400, 300]);
    await waitFor(() => expect(shapeCount()).toBe(1));
    await drawRectangle(user, [600, 400], [1000, 650]);
    await waitFor(() => expect(shapeCount()).toBe(2));

    fireEvent.keyDown(window, { key: "z", ctrlKey: true });
    await waitFor(() => expect(shapeCount()).toBe(1));
  });

  it("undoes the last corner of an open shape one click at a time", async () => {
    const user = userEvent.setup();
    render(<App />);
    await waitForUi();
    await openToolbar(user);

    await pickTool(user, /^4-point$/i);
    clickVideo(300, 300);
    clickVideo(600, 300);
    clickVideo(600, 600);
    // two clicks back off the open shape: three corners down to one
    await user.click(toolButton(/^undo$/i));
    await user.click(toolButton(/^undo$/i));

    // one corner left: three more clicks finish the quad
    clickVideo(600, 300);
    clickVideo(600, 600);
    clickVideo(300, 600);
    await waitFor(() => expect(shapeCount()).toBe(1));
    expect(shapePoints(0).length).toBe(4);
  });

  it("clears every zone at once", async () => {
    const user = userEvent.setup();
    render(<App />);
    await waitForUi();
    await openToolbar(user);

    await drawRectangle(user, [100, 100], [400, 300]);
    await drawRectangle(user, [600, 400], [1000, 650]);
    await waitFor(() => expect(shapeCount()).toBe(2));

    await user.click(toolButton(/^clear$/i));
    await waitFor(() => expect(shapeCount()).toBe(0));
    expect(chipLabels()).toHaveLength(0);
  });

  it("clamps a drag that leaves the frame instead of saving an unusable point", async () => {
    const user = userEvent.setup();
    render(<App />);
    await waitForUi();
    await openToolbar(user);

    await drawRectangle(user, [200, 200], [1400, 900]);
    await waitFor(() => expect(shapeCount()).toBe(1));
    shapePoints(0).forEach((pair) => {
      const [x, y] = pair.split(",").map(Number);
      expect(x).toBeGreaterThanOrEqual(0);
      expect(x).toBeLessThanOrEqual(1280);
      expect(y).toBeGreaterThanOrEqual(0);
      expect(y).toBeLessThanOrEqual(720);
    });
  });

  it("clamps a corner dragged past the edge of the frame", async () => {
    const user = userEvent.setup();
    render(<App />);
    await waitForUi();
    await openToolbar(user);

    await drawRectangle(user, [300, 300], [600, 500]);
    await waitFor(() => expect(shapeCount()).toBe(1));
    await pickTool(user, /^select$/i);
    fireEvent.mouseDown(document.querySelectorAll("circle.handle-hit")[0] as Element, {
      clientX: 300,
      clientY: 300,
    });
    fireEvent.mouseMove(window, { clientX: -400, clientY: -400 });
    fireEvent.mouseUp(window, { clientX: -400, clientY: -400 });

    await waitFor(() => expect(shapePoints(0)[0]).toBe("0,0"));
  });

  it("rounds every pointer coordinate to a whole pixel", async () => {
    const user = userEvent.setup();
    render(<App />);
    await waitForUi();
    await openToolbar(user);

    await drawRectangle(user, [200.4, 150.6], [700.2, 500.9]);
    await waitFor(() => expect(shapeCount()).toBe(1));
    shapePoints(0).forEach((pair) => {
      pair.split(",").forEach((value) => expect(Number.isInteger(Number(value))).toBe(true));
    });
  });
});

describe("zone validation", () => {
  it("keeps Save greyed out until the zone can be saved", async () => {
    const user = await renderEditing();

    await pickTool(user, /^4-point$/i);
    clickVideo(300, 300);
    clickVideo(600, 300);
    // an open shape is not a zone yet, so Save stays blocked and says what is
    // missing instead of vanishing
    expect(primaryButton().disabled).toBe(true);
    expect(screen.getByText(/draw a valid zone first/i)).toBeTruthy();

    clickVideo(600, 600);
    clickVideo(300, 300);
    await waitFor(() => expect(primaryButton().disabled).toBe(false));
  });

  it("says nothing is wrong once a valid zone is drawn", async () => {
    const user = userEvent.setup();
    render(<App />);
    await waitForUi();
    await openToolbar(user);

    await drawRectangle(user, [200, 150], [700, 500]);

    // one button, one verdict: unblocked and no complaint about the zone
    await waitFor(() => expect(primaryButton().disabled).toBe(false));
    expect(screen.queryByText(/cover at least/i)).toBeNull();
  });

  it("rejects a zone that covers almost none of the frame", async () => {
    const user = userEvent.setup();
    render(<App />);
    await waitForUi();
    await openToolbar(user);

    await drawRectangle(user, [600, 300], [604, 304]);
    await waitFor(() => expect(primaryButton().disabled).toBe(true));
    expect(screen.getAllByText(/cover at least/i).length).toBeGreaterThan(0);
  });

  it("does not silently assume a frame size while it is still unknown", async () => {
    state = makeState({ running: true, frame_width: 0, frame_height: 0 });
    render(<App />);
    await waitForUi();

    expect(screen.queryByTestId("toolbar")).toBeNull();
    expect(screen.getAllByText(/waiting for the first video frame/i).length).toBeGreaterThan(0);
  });

  it("will not send the API a zone whose corners collapse, even with no frame yet", async () => {
    // three corners on the same spot is three points to the UI and one point to
    // the API, which answers a start with a bare 400 the user cannot act on
    config = { ...CONFIG, zones: [[[400, 300], [400, 300], [400, 300]]] };
    state = makeState({ running: false, frame_width: 0, frame_height: 0 });
    render(<App />);
    await waitForUi();

    const start = button(/^start camera$/i);
    await waitFor(() => expect(start.disabled).toBe(true));
    expect(screen.getByText(/must all be different|3 different corners|needs at least 3 points/i)).toBeTruthy();
    expect(postCount("/start")).toBe(0);
  });

  it("still starts with a good zone before the first frame arrives", async () => {
    // the checks that need a frame are skipped, not the whole validation, so a
    // camera that has never run is not blocked from starting
    config = { ...CONFIG, zones: [[[100, 100], [400, 100], [400, 400]]] };
    state = makeState({ running: false, frame_width: 0, frame_height: 0 });
    render(<App />);
    await waitForUi();

    const start = button(/^start camera$/i);
    await waitFor(() => expect(start.disabled).toBe(false));
    state = { ...state, running: true };
    await userEvent.setup().click(start);
    await waitFor(() => expect(postCount("/start")).toBe(1));
    expect(lastBody("/start").zones).toEqual([[[100, 100], [400, 100], [400, 400]]]);
  });

  it("accepts a closed ring where the last point repeats the first (DMS/GeoJSON style)", async () => {
    // DMS stores polygons as closed LinearRings; the backend strips the closing
    // duplicate via drop_repeats, so the UI must too instead of reporting
    // "Point 4 and point 1 are at the same position".
    config = { ...CONFIG, zones: [[[100, 100], [400, 100], [400, 400], [100, 100]]] };
    state = makeState({ running: false, frame_width: 0, frame_height: 0 });
    render(<App />);
    await waitForUi();

    const start = button(/^start camera$/i);
    await waitFor(() => expect(start.disabled).toBe(false));
    state = { ...state, running: true };
    await userEvent.setup().click(start);
    await waitFor(() => expect(postCount("/start")).toBe(1));
  });
});

// This block replaced the three step wizard tests (Start / Draw / Save). A
// control room has no steps: the screen is stopped, live, or live with somebody
// editing, and each of those is asserted here.
describe("the Start / Edit / Save flow", () => {
  it("shows only Start while the camera is off", async () => {
    render(<App />);
    await waitForUi();

    expect(document.querySelector(".status-pill")?.textContent).toBe("STOPPED");
    expect(screen.getByTestId("start-panel")).toBeTruthy();
    expect(button(/^start camera$/i)).toBeTruthy();
    // nothing else: no tools, no Save, nothing to confirm
    expect(screen.queryByTestId("toolbar")).toBeNull();
    expect(screen.queryByTestId("edit-actions")).toBeNull();
    expect(screen.queryByTestId("state-banner")).toBeNull();
  });

  it("shows only Edit zones and Stop camera while live", async () => {
    const user = await renderEditing();
    await user.click(button(/^cancel$/i));

    await waitFor(() =>
      expect(document.querySelector(".status-pill")?.textContent).toBe("RUNNING")
    );
    expect(button(/^edit zones$/i)).toBeTruthy();
    expect(button(/^stop camera$/i)).toBeTruthy();
    expect(screen.queryByTestId("toolbar")).toBeNull();
    expect(screen.queryByTestId("edit-actions")).toBeNull();
  });

  it("keeps the tools hidden in Live and brings them out in Edit", async () => {
    config = { ...CONFIG, zones: [[[100, 100], [400, 100], [400, 400], [100, 400]]] };
    const user = userEvent.setup();
    state = makeState({ running: true });
    render(<App />);
    await waitForUi();

    // Live: watching, with nothing that could change a zone
    expect(screen.queryByTestId("toolbar")).toBeNull();

    await user.click(button(/^edit zones$/i));
    await waitFor(() => expect(screen.getByTestId("toolbar")).toBeTruthy());
    // and the camera was never stopped to get there
    expect(stops()).toBe(0);
  });

  it("asks before throwing changed geometry away, and not otherwise", async () => {
    const user = await renderEditing();

    // nothing drawn yet: Cancel just leaves
    await user.click(button(/^cancel$/i));
    await waitFor(() =>
      expect(document.querySelector(".status-pill")?.textContent).toBe("RUNNING")
    );

    // now draw something and come back
    await user.click(button(/^edit zones$/i));
    await waitFor(() => expect(screen.getByTestId("toolbar")).toBeTruthy());
    await drawRectangle(user, [200, 150], [700, 500]);

    await user.click(button(/^cancel$/i));
    expect(screen.getByRole("alertdialog", { name: /confirm cancel/i })).toBeTruthy();
    expect(document.querySelector(".status-pill")?.textContent).toBe("Editing");

    // a way back in without losing anything
    await user.click(button(/^keep editing$/i));
    expect(screen.queryByRole("alertdialog", { name: /confirm cancel/i })).toBeNull();

    await user.click(button(/^cancel$/i));
    await user.click(button(/^discard changes$/i));
    await waitFor(() =>
      expect(document.querySelector(".status-pill")?.textContent).toBe("RUNNING")
    );
    expect(drawnZones()).toHaveLength(0);
  });

  it("does not count a picked tool or a tapped card as unsaved work", async () => {
    config = { ...CONFIG, zones: [[[100, 100], [400, 100], [400, 400], [100, 400]]] };
    const user = await renderEditing();

    await pickTool(user, /^select$/i);
    await user.click(button(/^edit zone 1$/i));
    expect(screen.getByTestId("coord-details")).toBeTruthy();

    // none of that changes a zone, so Cancel leaves without asking
    await user.click(button(/^cancel$/i));
    await waitFor(() =>
      expect(document.querySelector(".status-pill")?.textContent).toBe("RUNNING")
    );
    expect(drawnZones()).toHaveLength(1);
  });

  it("reports how long it has been watching, once it watched the camera start", async () => {
    const user = userEvent.setup();
    render(<App />);
    await waitForUi();

    // nothing has been watched yet, so nothing is claimed
    expect(screen.queryByTestId("uptime")).toBeNull();

    await startCamera(user);
    await waitFor(() => expect(screen.getByTestId("uptime")).toBeTruthy());
    expect(screen.getByTestId("uptime").textContent).toMatch(
      /^Monitoring since \d{1,2}:\d{2}, uptime \d+h \d{1,2}m$/
    );
  });

  it("claims no start time when the page loads onto a running camera", async () => {
    state = makeState({ running: true });
    render(<App />);
    await waitForUi();

    // the first poll only says the camera is already running: there was no
    // moment on this page to witness, so no time is invented
    expect(screen.queryByTestId("uptime")).toBeNull();
  });

  it("puts the verdict chip on the video and the counts under it, no banner when calm", async () => {
    state = makeState({ running: true, inside: 1 });
    render(<App />);
    await waitForUi();

    const order = [...document.querySelectorAll(".stage, .state-banner, .tiles")].map((node) =>
      node.className.split(" ")[0]
    );
    expect(order).toEqual(["stage", "tiles"]);
    expect(screen.getByTestId("status-badge").textContent).toBe("SAFE");
  });

  it("resets the banner, tiles and zone cards when the camera stops", async () => {
    const cfg = namedZones();
    cfg.zone_specs = cfg.zone_specs?.map((spec, index) =>
      index === 1 ? { ...spec, enabled: false } : spec
    );
    config = cfg;
    // the camera is off but the last /state still says ALERT: none of it may
    // linger on screen
    state = makeState({
      running: false,
      status: "ALERT",
      inside: 1,
      message: "DANGER - Person in danger zone",
      workers: [
        {
          worker_id: 3,
          state: "ALERT",
          dwell_seconds: 5,
          zone_index: 0,
          zone_name: "Kiln Platform",
        },
      ],
      zone_reports: [
        {
          index: 0,
          name: "Kiln Platform",
          inside: 1,
          allowed_people: 1,
          dwell_seconds: 12.3,
          alerting: true,
          enabled: true,
        },
        {
          index: 1,
          name: "Sand Store",
          inside: 0,
          allowed_people: 6,
          dwell_seconds: 0,
          alerting: false,
          enabled: true,
        },
      ],
    });
    render(<App />);
    await waitForUi();

    // stopped shows no verdict and no counts at all: only the dark card and
    // the Start / Edit buttons
    expect(screen.queryByTestId("stopped-banner")).toBeNull();
    expect(screen.queryByTestId("state-banner")).toBeNull();
    expect(screen.queryByTestId("status-tiles")).toBeNull();
    expect(screen.queryByTestId("stopped-source")).toBeNull();
    expect(screen.getByTestId("start-panel")).toBeTruthy();

    await waitFor(() => expect(screen.getByTestId("zone-cards")).toBeTruthy());
    const counts = [...document.querySelectorAll(".zone-card-count")].map(
      (node) => (node.textContent ?? "").replace(/\s+/g, " ").trim()
    );
    expect(counts).toEqual(["0 people", "0 people"]);
    // the on/off switch stays honest while stopped: enabled reads green, the
    // switched-off zone says Disabled instead of looking broken
    const dots = [...document.querySelectorAll(".zone-dot")].map((node) => node.className);
    expect(dots[0]).toContain("safe");
    expect(dots[1]).toContain("neutral");
    expect(screen.getAllByText("Disabled")).toHaveLength(1);
  });

  it("shows each stopped zone's real on/off state, not a greyed-out look", async () => {
    const cfg = namedZones();
    cfg.zone_specs = cfg.zone_specs?.map((spec, index) =>
      index === 0 ? { ...spec, enabled: false } : spec
    );
    config = cfg;
    state = makeState({ running: false });
    render(<App />);
    await waitForUi();

    await waitFor(() => expect(screen.getByTestId("zone-cards")).toBeTruthy());
    const dots = [...document.querySelectorAll(".zone-dot")].map((node) => node.className);
    expect(dots[0]).toContain("neutral");
    expect(dots[1]).toContain("safe");
    const labels = [...document.querySelectorAll(".zone-disabled")].map((node) =>
      (node.textContent ?? "").trim()
    );
    expect(labels).toEqual(["Disabled"]);
  });

  it("shows one card per zone and no counts it cannot know", async () => {
    config = {
      ...CONFIG,
      zones: [
        [[100, 100], [400, 100], [400, 400], [100, 400]],
        [[500, 200], [900, 200], [900, 500], [500, 500]],
      ],
    };
    state = makeState({ running: true, inside: 0 });
    render(<App />);
    await waitForUi();

    await waitFor(() => expect(screen.getByTestId("zone-cards")).toBeTruthy());
    expect(chipLabels()).toEqual(["Zone 1", "Zone 2"]);
    // the two global counts live in the tiles, once, from /state
    expect(document.querySelectorAll(".tile")).toHaveLength(2);
    expect(screen.getByTestId("tile-inside").textContent).toMatch(/^0/);
  });
});

// The wizard's drawing-tool tests moved under the new name above.
describe("the zone editing tools", () => {
  it("has no Draw button: the toolbar is how a zone is made", async () => {
    await renderEditing();

    expect(screen.queryByRole("button", { name: /^draw zone$/i })).toBeNull();
    // the one primary button stays put and simply says what cannot be done yet
    expect(primaryButton().textContent).toBe("Save");
    expect(primaryButton().disabled).toBe(true);
    expect(screen.getByText(/draw a valid zone first/i)).toBeTruthy();
  });

  it("enables Save once a valid zone is drawn", async () => {
    const user = await renderEditing();

    await drawRectangle(user, [200, 150], [700, 500]);
    await waitFor(() => expect(button(/^save$/i)).toBeTruthy());
    expect((button(/^save$/i) as HTMLButtonElement).disabled).toBe(false);
  });

  it("goes back to Live after saving, without stopping the camera", async () => {
    const user = await renderEditing();

    await drawRectangle(user, [200, 150], [700, 500]);
    await user.click(button(/^save$/i));
    await waitFor(() => expect(screen.getByText(/zone saved - monitoring is live/i)).toBeTruthy());
    // the toolbar folds away until Edit zones is pressed
    expect(screen.queryByTestId("toolbar")).toBeNull();
    expect(button(/^edit zones$/i)).toBeTruthy();
  });

  it("brings the toolbar back on Edit zones without stopping the camera", async () => {
    const user = await renderEditing();

    await drawRectangle(user, [200, 150], [700, 500]);
    await user.click(button(/^save$/i));
    await waitFor(() => expect(button(/^edit zones$/i)).toBeTruthy());

    await user.click(button(/^edit zones$/i));
    await waitFor(() => expect(screen.getByTestId("toolbar")).toBeTruthy());
    expect(stops()).toBe(0);
  });
});

describe("zone coordinates", () => {
  /** the folded point list of the picked zone */
  function coordPanel(): HTMLDetailsElement {
    return screen.getByTestId("coord-details") as HTMLDetailsElement;
  }

  it("keeps the coordinates out of the settings", async () => {
    const user = userEvent.setup();
    render(<App />);
    await waitForUi();
    await openToolbar(user);

    await drawRectangle(user, [200, 150], [700, 500]);
    // the point list lives under the video, never inside the settings
    expect(coordPanel()).toBeTruthy();
    expect(coordPanel().closest("details")).toBe(coordPanel());
    expect(coordPanel().closest(".side")).toBeNull();
  });

  it("shows no point list at all when nothing is selected", async () => {
    const user = userEvent.setup();
    config = { ...CONFIG, zones: [] };
    render(<App />);
    await waitForUi();
    await openToolbar(user);

    // clearing leaves shapes, but none of them picked
    await pickTool(user, /^clear$/i);
    await drawRectangle(user, [200, 150], [700, 500]);
    fireEvent.keyDown(window, { key: "Escape" });
    await user.click(button(/^clear$/i));
    await waitFor(() => expect(screen.queryAllByRole("button", { name: /edit zone/i })).toHaveLength(0));
    expect(screen.queryByTestId("coord-details")).toBeNull();
  });

  it("folds the point list closed when a zone is picked", async () => {
    const user = userEvent.setup();
    render(<App />);
    await waitForUi();
    await openToolbar(user);

    await drawRectangle(user, [200, 150], [700, 500]);
    await user.click(button(/^edit zone 1$/i));
    const panel = coordPanel();
    expect(panel.open).toBe(false);
    expect(within(panel).getByText(/zone 1 coordinates/i)).toBeTruthy();

    await user.click(within(panel).getByText(/zone 1 coordinates/i));
    expect(panel.open).toBe(true);
    expect(within(panel).getAllByRole("row").length).toBe(5); // header + 4 points
  });

  it("shows only the picked zone, not every zone", async () => {
    const user = userEvent.setup();
    render(<App />);
    await waitForUi();
    await openToolbar(user);

    await drawRectangle(user, [200, 150], [700, 500]);
    await drawRectangle(user, [800, 200], [1100, 500]);
    await user.click(button(/^edit zone 2$/i));
    await user.click(within(coordPanel()).getByText(/zone 2 coordinates/i));

    const panel = coordPanel();
    expect(within(panel).getAllByRole("row").length).toBe(5);
    expect(panel.textContent).not.toMatch(/zone 1/i);
  });

  it("lets a typed coordinate move a corner", async () => {
    const user = userEvent.setup();
    render(<App />);
    await waitForUi();
    await openToolbar(user);

    await drawRectangle(user, [200, 150], [700, 500]);
    await user.click(button(/^edit zone 1$/i));
    await user.click(within(coordPanel()).getByText(/zone 1 coordinates/i));

    const input = screen.getByLabelText(/point 1 x/i);
    await user.clear(input);
    await user.type(input, "222");
    await waitFor(() => expect(shapePoints(0)[0].startsWith("222")).toBe(true));
  });

  it("adds a point to the picked zone", async () => {
    const user = userEvent.setup();
    render(<App />);
    await waitForUi();
    await openToolbar(user);

    await drawRectangle(user, [200, 150], [700, 500]);
    await user.click(button(/^edit zone 1$/i));
    await user.click(within(coordPanel()).getByText(/zone 1 coordinates/i));

    await user.click(within(coordPanel()).getByRole("button", { name: /add point/i }));
    await waitFor(() => expect(within(coordPanel()).getAllByRole("row").length).toBe(6));
  });
});

describe("coordinates and saved zones", () => {
  it("scales a saved zone up from the resolution it was drawn on", async () => {
    // drawn on a 640x360 feed, now opened on a 1280x720 one. It is under the
    // 0.5% floor as it stands, so it has to be scaled to be monitorable at all.
    config = {
      ...CONFIG,
      zones: [
        [
          [100, 50],
          [200, 50],
          [200, 90],
        ],
      ],
      zone_reference: [640, 360],
    };
    state = makeState({ running: true });
    render(<App />);
    await waitForUi();

    await waitFor(() => expect(screen.getByText(/zone adjusted to fit the frame/i)).toBeTruthy());
    await waitFor(() => expect(shapePoints(0)).toEqual(["200,100", "400,100", "400,180"]));
  });

  it("keeps a saved zone that still fits after scaling to another resolution", async () => {
    // drawn on 640x360, now opened on a 320x180 one
    config = {
      ...CONFIG,
      zones: [
        [
          [100, 50],
          [500, 50],
          [500, 300],
        ],
      ],
      zone_reference: [640, 360],
    };
    state = makeState({ running: true, frame_width: 320, frame_height: 180 });
    render(<App />);
    await waitForUi();

    await waitFor(() => expect(screen.getByText(/zone adjusted to fit the frame/i)).toBeTruthy());
    await waitFor(() => expect(shapePoints(0)).toEqual(["50,25", "250,25", "250,150"]));
  });

  it("clamps a saved zone that hangs off the frame onto the edge", async () => {
    // no reference size, so there is nothing to scale from: the saved points
    // sit well outside the 640x480 video this machine opened and are pulled
    // back onto the edge instead of being thrown away
    config = {
      ...CONFIG,
      zones: [
        [
          [100, 100],
          [1100, 100],
          [1100, 400],
        ],
      ],
    };
    state = makeState({ running: true, frame_width: 640, frame_height: 480 });
    render(<App />);
    await waitForUi();

    await waitFor(() => expect(screen.getByText(/zone adjusted to fit the frame/i)).toBeTruthy());
    expect(shapePoints(0)).toEqual(["100,100", "640,100", "640,400"]);
  });

  it("keeps a zone that cannot fit the new frame instead of dropping it", async () => {
    // a 30x30 corner drawn on a 640x480 feed: on the 1280x720 one it scales
    // to 60x60, which is under the 0.5% floor. It used to be thrown out of the
    // state - saved shape and all - instead of being shown and left alone
    config = {
      ...CONFIG,
      zones: [
        [
          [100, 100],
          [130, 100],
          [130, 130],
          [100, 130],
        ],
      ],
      zone_reference: [640, 480],
    };
    state = makeState({ running: true });
    render(<App />);
    await waitForUi();

    await waitFor(() => expect(screen.getByText(/zone adjusted to fit the frame/i)).toBeTruthy());
    // still on the picture and still on the cards: only Save changes what is
    // stored, and a zone the floor rejects is not the app's to delete
    expect(shapeCount()).toBe(1);
    expect(chipLabels()).toHaveLength(1);
  });

  it("scales every saved zone, not just the first", async () => {
    // both zones are under the 0.5% floor on the new feed until they are scaled
    config = {
      ...CONFIG,
      zones: [
        [
          [100, 50],
          [200, 50],
          [200, 90],
        ],
        [
          [350, 250],
          [450, 250],
          [450, 290],
        ],
      ],
      zone_reference: [640, 360],
    };
    state = makeState({ running: true });
    render(<App />);
    await waitForUi();

    await waitFor(() => expect(screen.getByText(/zone adjusted to fit the frame/i)).toBeTruthy());
    await waitFor(() => expect(shapeCount()).toBe(2));
    expect(shapePoints(0)).toEqual(["200,100", "400,100", "400,180"]);
    expect(shapePoints(1)).toEqual(["700,500", "900,500", "900,580"]);
  });

  it("reads a backend that only publishes one zone", async () => {
    config = { ...CONFIG, zone_points: [[100, 100], [400, 100], [400, 400], [100, 400]] };
    state = makeState({ running: true });
    render(<App />);
    await waitForUi();

    await waitFor(() => expect(screen.getByTestId("zone-stage")).toBeTruthy());
    expect(shapeCount()).toBe(1);
  });

  it("draws no zone over the video while the feed is stopped", async () => {
    render(<App />);
    await waitForUi();
    expect(screen.queryByTestId("zone-stage")).toBeNull();
  });
});

describe("alerts while editing", () => {
  /** counts the beeps the app asks for, so the sound is checked and not assumed */
  function stubAudio() {
    const beeps = { count: 0 };
    class FakeParam {
      setValueAtTime() {
        return this;
      }
      exponentialRampToValueAtTime() {
        return this;
      }
    }
    class FakeNode {
      connect() {
        return this;
      }
      frequency = new FakeParam();
      gain = new FakeParam();
    }
    class FakeOscillator extends FakeNode {
      type = "";
      start = vi.fn(() => {
        beeps.count += 1;
      });
      stop = vi.fn();
    }
    class FakeContext {
      currentTime = 0;
      state = "running";
      destination = new FakeNode();
      createGain() {
        return new FakeNode();
      }
      createOscillator() {
        return new FakeOscillator();
      }
      resume() {
        this.state = "running";
        return Promise.resolve();
      }
    }
    vi.stubGlobal("AudioContext", FakeContext);
    return beeps;
  }

  it("keeps the ALERT banner above the EDITING banner, with the real message", async () => {
    config = { ...CONFIG, zones: [[[100, 100], [400, 100], [400, 400], [100, 400]]] };
    state = makeState({
      running: true,
      inside: 1,
      status: "ALERT",
      worker_alert: true,
      message: "person in danger zone",
    });
    const user = await renderEditing();

    // monitoring never stopped for the editor, so the alarm is still the news
    const banners = [...document.querySelectorAll(".state-banner")].map((node) => node.textContent);
    expect(banners).toHaveLength(2);
    expect(banners[0]).toContain("ALERT");
    expect(banners[0]).toContain("person in danger zone");
    expect(banners[1]).toContain("EDITING");

    // and the order is alarm first, then what you are doing
    const order = [...document.querySelectorAll(".state-banner")].map((node) =>
      node.classList.contains("alert") ? "alert" : "editing"
    );
    expect(order).toEqual(["alert", "editing"]);

    // the mute button comes with the banner, so it is reachable mid-edit
    const mute = screen.getByRole("button", { name: /mute alert sound/i });
    expect(mute).toBeTruthy();
    await user.click(mute);
    await waitFor(() =>
      expect(screen.getByRole("button", { name: /unmute alert sound/i })).toBeTruthy()
    );
  });

  it("shows nothing extra in Edit while the feed is safe", async () => {
    state = makeState({ running: true, inside: 0, status: "SAFE" });
    await renderEditing();

    // no second banner, and the SAFE one is not there either: Edit says what it
    // is doing, and nothing pretends to be a verdict
    expect(document.querySelectorAll(".state-banner")).toHaveLength(1);
    expect(screen.getByTestId("edit-banner").textContent).toContain("EDITING");
    expect(screen.queryByTestId("state-banner")).toBeNull();
  });

  it("still goes from SAFE to ALERT in Edit, and the beep follows the alarm", async () => {
    const beeps = stubAudio();
    const user = userEvent.setup();
    render(<App />);
    await waitForUi();
    // the browser only lets sound out after a real press, and Start is that
    // press, so the alarm has to be set off after it
    await startCamera(user);
    await user.click(button(/^edit zones$/i));
    await waitFor(() => expect(screen.getByTestId("edit-banner")).toBeTruthy());

    state = makeState({
      running: true,
      inside: 1,
      status: "ALERT",
      worker_alert: true,
      message: "in",
    });
    await waitFor(() => expect(screen.getByTestId("state-banner")).toBeTruthy());
    expect(beeps.count).toBe(1);

    // one beep per alarm, not one per poll
    await waitFor(() => expect(document.querySelectorAll(".state-banner")).toHaveLength(2));
    expect(beeps.count).toBe(1);
  });

  it("does not beep again for the same alarm, and does not beep while muted", async () => {
    const beeps = stubAudio();
    const user = userEvent.setup();
    render(<App />);
    await waitForUi();
    await startCamera(user);
    await user.click(button(/^edit zones$/i));
    await waitFor(() => expect(screen.getByTestId("edit-banner")).toBeTruthy());

    state = makeState({ running: true, inside: 1, status: "ALERT", worker_alert: true });
    await waitFor(() => expect(beeps.count).toBe(1));

    // the operator silences this one, then it clears
    await user.click(screen.getByRole("button", { name: /mute alert sound/i }));
    expect(beeps.count).toBe(1);
    state = makeState({ running: true, inside: 0, status: "SAFE" });
    await waitFor(() => expect(screen.queryByTestId("state-banner")).toBeNull());

    // the next alarm is an alarm, not a beep
    state = makeState({ running: true, inside: 2, status: "ALERT", worker_alert: true });
    await waitFor(() =>
      expect(screen.getByRole("button", { name: /unmute alert sound/i })).toBeTruthy()
    );
    expect(beeps.count).toBe(1);
  });

  it("keeps the banner tied to /state, not to the editor", async () => {
    config = { ...CONFIG, zones: [[[100, 100], [400, 100], [400, 400], [100, 400]]] };
    state = makeState({ running: true, inside: 1, status: "ALERT", message: "one" });
    const user = await renderEditing();
    expect(screen.getByTestId("state-banner").textContent).toContain("one");

    // the backend's message moves on, and so does the banner
    state = makeState({ running: true, inside: 2, status: "ALERT", message: "two" });
    await waitFor(() => expect(screen.getByTestId("state-banner").textContent).toContain("two"));

    // and leaving the editor leaves the alarm exactly where it is
    await user.click(button(/^cancel$/i));
    expect(screen.getByTestId("state-banner").textContent).toContain("two");
  });
});

describe("the settings page", () => {
  it("shows every card at once, with nothing folded away", async () => {
    await renderSettings();

    expect(screen.getByTestId("settings-page")).toBeTruthy();
    // the fields are all there without asking
    expect(screen.getByLabelText(/^alert mode$/i)).toBeTruthy();
    expect(screen.getByLabelText(/^people allowed$/i)).toBeTruthy();
    expect(
      document.querySelector("fieldset.radio-field legend")?.textContent
    ).toMatch(/detect person at/i);
    // one card per section, no fold toggles anywhere
    for (const card of ["defaults", "system", "danger", "camera", "paths", "videos", "machine"]) {
      expect(screen.getByTestId(`${card}-card`)).toBeTruthy();
    }
    expect(screen.queryByText(/^hide settings$/i)).toBeNull();
    expect(screen.queryByText(/^show settings$/i)).toBeNull();
  });

  it("keeps showing the cards while the camera runs, and after it stops", async () => {
    state = makeState({ running: true });
    await renderSettings();
    expect(screen.getByTestId("settings-page")).toBeTruthy();

    await act(async () => {
      state = makeState({ running: false });
    });
    await waitFor(() => expect(screen.getByTestId("settings-page")).toBeTruthy());
  });

  it("lays the cards out in two columns: rules, status and danger left, camera and paths right", async () => {
    await renderSettings();

    const columns = [...document.querySelectorAll(".settings-grid > .settings-col")];
    expect(columns).toHaveLength(2);
    const left = columns[0].querySelectorAll(":scope > .card");
    const right = columns[1].querySelectorAll(":scope > .card");
    expect([...left].map((node) => node.getAttribute("data-testid"))).toEqual([
      "defaults-card",
      "system-card",
      "danger-card",
    ]);
    expect([...right].map((node) => node.getAttribute("data-testid"))).toEqual([
      "camera-card",
      "paths-card",
      "videos-card",
      "machine-card",
    ]);
    expect(screen.getByText("Detection Defaults")).toBeTruthy();
    expect(screen.getByText("Videos (test inputs)")).toBeTruthy();
  });

  it("shows the detection paths card with one row per system", async () => {
    config = {
      ...CONFIG,
      fire_smoke: { enabled: true, analysis_fps: 2 },
      crowd: { enabled: false, analysis_fps: 2 },
      ppe: { enabled: false, analysis_fps: 1 },
    };
    await renderSettings();
    expect(screen.getByTestId("paths-card")).toBeTruthy();
    expect(screen.getByText("Detection Paths")).toBeTruthy();
    // five systems, fire and smoke as separate switches on one shared pass
    for (const row of ["hazard", "fire", "smoke", "crowd", "ppe"]) {
      expect(screen.getByTestId(`path-row-${row}`)).toBeTruthy();
    }
    expect(
      (screen.getByLabelText("Fire enabled") as HTMLInputElement).checked
    ).toBe(true);
    expect(
      (screen.getByLabelText("Smoke enabled") as HTMLInputElement).checked
    ).toBe(true);
    expect(
      (screen.getByLabelText("Crowd enabled") as HTMLInputElement).checked
    ).toBe(false);
    // the shared cadence sits on the Fire row and says so
    expect(screen.getByLabelText("Fire target FPS")).toBeTruthy();
    expect(screen.getByText(/shared with Smoke/)).toBeTruthy();
    expect(screen.queryByLabelText("Smoke target FPS")).toBeNull();
    expect(screen.queryByLabelText("Hazard zone target FPS")).toBeNull();
  });

  it("toggling a detection path saves it to /config", async () => {
    const user = await renderSettings();
    await user.click(screen.getByLabelText("Crowd enabled"));
    await waitFor(
      () => {
        const bodies = postedBodies("/config");
        const last = bodies[bodies.length - 1] as {
          crowd?: { enabled?: boolean };
        };
        expect(last.crowd?.enabled).toBe(true);
      },
      { timeout: 4000 }
    );
  });

  it("resets detection paths to defaults and saves them", async () => {
    const user = await renderSettings();
    // first enable a few things so the reset actually changes them
    await user.click(screen.getByLabelText("Crowd enabled"));
    await user.click(screen.getByLabelText("PPE enabled"));
    await user.clear(screen.getByLabelText("Fire target FPS"));
    await user.type(screen.getByLabelText("Fire target FPS"), "3");
    await waitFor(() => expect(screen.getByText("Applied")).toBeTruthy());

    // now click reset
    await user.click(screen.getByTestId("paths-reset"));
    await waitFor(
      () => {
        const bodies = postedBodies("/config");
        const last = bodies[bodies.length - 1] as {
          hazard?: { enabled?: boolean };
          fire_smoke?: { enabled?: boolean; analysis_fps?: number };
          crowd?: { enabled?: boolean; analysis_fps?: number };
          ppe?: { enabled?: boolean; analysis_fps?: number };
        };
        expect(last.hazard?.enabled).toBe(true);
        expect(last.fire_smoke?.enabled).toBe(false);
        expect(last.fire_smoke?.analysis_fps).toBe(2);
        expect(last.crowd?.enabled).toBe(false);
        expect(last.crowd?.analysis_fps).toBe(2);
        expect(last.ppe?.enabled).toBe(false);
        expect(last.ppe?.analysis_fps).toBe(1);
      },
      { timeout: 4000 }
    );
  });

  it("shows one shared Loop video box on the Camera card and the live picker", async () => {
    const clip = makeVideo();
    videos = [clip];
    // a backend from before the box sends nothing: off
    config = { ...CONFIG, source: clip.source };
    const user = await renderSettings();
    const cameraBox = screen.getByTestId("camera-loop-video") as HTMLInputElement;
    expect(cameraBox.checked).toBe(false);
    await user.click(cameraBox);
    await waitFor(
      () => {
        const bodies = postedBodies("/config");
        const last = bodies[bodies.length - 1] as { loop_video?: boolean };
        expect(last.loop_video).toBe(true);
      },
      { timeout: 4000 }
    );

    // the live picker shows the same box, already ticked: one setting
    await user.click(screen.getByRole("button", { name: /^live monitor$/i }));
    await waitForUi();
    const liveBox = screen.getByTestId("live-loop-video") as HTMLInputElement;
    expect(liveBox.checked).toBe(true);
    // and ticking it there unticks the Camera card too
    await user.click(liveBox);
    await user.click(screen.getByRole("button", { name: /^settings$/i }));
    await waitFor(() => expect(screen.getByTestId("settings-page")).toBeTruthy());
    expect(
      (screen.getByTestId("camera-loop-video") as HTMLInputElement).checked
    ).toBe(false);
  });

  it("hides the Loop box for the webcam and sends the on-screen value on Start", async () => {
    const clip = makeVideo();
    videos = [clip];
    config = { ...CONFIG, source: clip.source, loop_video: true };
    const user = userEvent.setup();
    render(<App />);
    await waitForUi();
    // the saved clip opens the video half (the effect lands after load)
    await waitFor(() =>
      expect(
        (screen.getByTestId("live-source-select") as HTMLSelectElement).value
      ).toBe(clip.source)
    );
    // a clip source shows the picker box, ticked from the saved setting
    expect(
      (screen.getByTestId("live-loop-video") as HTMLInputElement).checked
    ).toBe(true);

    // back on the webcam there is nothing to replay, so no box
    await user.selectOptions(
      screen.getByTestId("live-source-select"),
      "camera"
    );
    await waitFor(() => expect(screen.queryByTestId("live-loop-video")).toBeNull());

    // Start carries the on-screen value without waiting for the save
    await userEvent.setup().click(button(/^start camera$/i));
    await waitFor(
      () => {
        const bodies = postedBodies("/start");
        const last = bodies[bodies.length - 1] as { loop?: boolean };
        expect(last.loop).toBe(true);
      },
      { timeout: 4000 }
    );
  });

  it("says Clip finished after a single play, Camera stopped otherwise", async () => {
    const clip = makeVideo();
    videos = [clip];
    config = { ...CONFIG, source: clip.source };
    state = makeState({ running: false, finished_clip: true, source: clip.source });
    render(<App />);
    await waitForUi();
    expect(screen.getByTestId("start-panel")).toBeTruthy();
    expect(screen.getByText("Clip finished")).toBeTruthy();
    // Start is offered again, not blocked
    expect(button(/^start camera$/i).disabled).toBe(false);

    // a finish belongs to the source that ended: another clip shows stop
    cleanup();
    state = makeState({ running: false, finished_clip: true, source: clip.source });
    config = { ...CONFIG, source: "0" };
    render(<App />);
    await waitForUi();
    expect(screen.queryByText("Clip finished")).toBeNull();
    expect(screen.getByText("Camera stopped")).toBeTruthy();

    // a plain stop says Camera stopped too
    cleanup();
    state = makeState({ running: false, source: clip.source });
    config = { ...CONFIG, source: clip.source };
    render(<App />);
    await waitForUi();
    expect(screen.queryByText("Clip finished")).toBeNull();
    expect(screen.getByText("Camera stopped")).toBeTruthy();
  });

  it("keeps the allowed-people slider and number in step", async () => {
    const user = await renderSettings();
    const slider = screen.getByLabelText("Allowed people") as HTMLInputElement;
    const number = screen.getByLabelText(/^people allowed$/i) as HTMLInputElement;
    expect(slider.value).toBe(number.value);

    fireEvent.change(slider, { target: { value: "7" } });
    expect(number.value).toBe("7");

    await user.clear(number);
    await user.type(number, "3");
    expect(slider.value).toBe("3");
    await waitFor(() => expect(screen.getByText("Applied")).toBeTruthy());
  });

  it("lists the real sources in the camera dropdown and picks one", async () => {
    videos = [makeVideo({ id: "one", name: "line.mp4", source: "/data/videos/one.mp4" })];
    const user = await renderSettings();

    const select = screen.getByLabelText("Video source") as HTMLSelectElement;
    const options = [...select.options].map((o) => o.textContent ?? "");
    expect(options[0]).toMatch(/live camera/i);
    expect(options.some((text) => text.includes("line.mp4"))).toBe(true);
    expect(options.some((text) => text.includes("640×480"))).toBe(true);

    await user.selectOptions(select, "/data/videos/one.mp4");
    // the one camera source field is what changed, so /config carries it
    await waitFor(() =>
      expect(postedBodies("/config").some((body) => body.source === "/data/videos/one.mp4")).toBe(
        true
      )
    );
    // the clip's real size is shown, since it is known
    expect(screen.getByTestId("camera-resolution").textContent).toContain("640×480");
  });

  it("shows the resolution when the frame size is known, hides it otherwise", async () => {
    await renderSettings();
    // stopped, but the backend reported a size: the last real one
    expect(screen.getByTestId("camera-resolution").textContent).toContain("1280x720");

    cleanup();
    window.location.hash = "#/";
    state = makeState({ frame_width: 0, frame_height: 0 });
    await renderSettings();
    // never seen a frame: nothing to show, and nothing invented either
    expect(screen.queryByTestId("camera-resolution")).toBeNull();
  });

  it("keeps the Settings paths and the Live Monitor panel on the same switches", async () => {
    config = {
      ...CONFIG,
      fire_smoke: { enabled: true, analysis_fps: 2 },
    };
    const user = await renderSettings();
    // off in Settings...
    await user.click(screen.getByLabelText("Fire enabled"));
    await waitFor(() =>
      expect(
        (screen.getByLabelText("Fire enabled") as HTMLInputElement).checked
      ).toBe(false)
    );

    // ...off on the Live Monitor panel too: one source of truth
    await user.click(button(/^live monitor$/i));
    await waitFor(() =>
      expect(screen.getByTestId("systems-card")).toBeTruthy()
    );
    expect(
      (within(screen.getByTestId("systems-card")).getByLabelText(
        "Fire enabled"
      ) as HTMLInputElement).checked
    ).toBe(false);

    // and back on from the panel...
    await user.click(
      within(screen.getByTestId("systems-card")).getByLabelText("Fire enabled")
    );

    // ...on in Settings too
    await user.click(button(/^settings$/i));
    await waitFor(() =>
      expect(
        (screen.getByLabelText("Fire enabled") as HTMLInputElement).checked
      ).toBe(true)
    );
  });

  it("shows live crowd numbers and the PPE grading caption on the monitor", async () => {
    state = makeState({
      running: true,
      crowd: {
        enabled: true,
        metrics: { person_count: 3, coverage_fraction: 0.125, density_per_megapixel: 2.5 },
      },
      ppe: { enabled: true, grade: "high", deferred: true, stats: {} },
    });
    render(<App />);
    await waitForUi();
    expect(screen.getByTestId("side-paths").textContent).toContain("3");
    expect(screen.getByTestId("ppe-deferred-note").textContent).toMatch(
      /Grading only, alerts not enabled/
    );
  });

  it("shows off cards without live tiles when every path is off", async () => {
    state = makeState({ running: true });
    render(<App />);
    await waitForUi();
    expect(screen.getByTestId("crowd-card").textContent).toContain("Off");
    expect(screen.getByTestId("ppe-card").textContent).toContain("Off");
    expect(screen.queryByTestId("crowd-tile")).toBeNull();
    expect(screen.queryByTestId("ppe-tile")).toBeNull();
  });

  function makeFire(overrides = {}): FireAlert {
    return {
      id: 1,
      alert_type: "fire",
      severity: "ORANGE",
      confidence: 0.76,
      bbox_xyxy: [1, 2, 30, 40],
      captured_at: 1.0,
      reported_at: "2026-10-08T10:24:00Z",
      has_snapshot: true,
      acknowledged: false,
      acknowledged_at: null,
      false_alarm: false,
      false_alarm_at: null,
      ...overrides,
    };
  }

  async function openAlerts(user: ReturnType<typeof userEvent.setup>) {
    await user.click(screen.getByRole("button", { name: /^alerts( \d+)?$/i }));
    await waitFor(() => expect(screen.getByTestId("fire-subhead")).toBeTruthy());
  }

  it("shows fire rows with their own counts under a subhead", async () => {
    fireRows = [makeFire(), makeFire({ id: 2, acknowledged: true })];
    const user = userEvent.setup();
    render(<App />);
    await waitForUi();
    await openAlerts(user);
    expect(screen.getByTestId("fire-subhead").textContent).toContain("Fire & smoke (1)");
    expect(screen.getByTestId("fire-alert-list").textContent).toContain("ORANGE");
    await user.click(screen.getByRole("tab", { name: /acknowledged/i }));
    await waitFor(() =>
      expect(screen.getByTestId("fire-subhead").textContent).toContain("Fire & smoke (1)")
    );
  });

  it("acknowledging a fire row posts to its own route and moves it", async () => {
    fireRows = [makeFire()];
    const user = userEvent.setup();
    render(<App />);
    await waitForUi();
    await openAlerts(user);
    await user.click(screen.getByRole("button", { name: /^acknowledge$/i }));
    await waitFor(() =>
      expect(
        fetchMock.mock.calls.some(
          ([url, init]) =>
            url === "/fire-alerts/1/ack" && (init as RequestInit)?.method === "POST"
        )
      ).toBe(true)
    );
    await user.click(screen.getByRole("tab", { name: /acknowledged/i }));
    await waitFor(() =>
      expect(screen.getByTestId("fire-alert-list").textContent).toContain("ORANGE")
    );
  });

  it("counts zone plus fire unacked on the sidebar badge", async () => {
    alertRows = [
      {
        id: 7,
        worker_id: 1,
        machine_id: "excavator_01",
        event_type: "DWELL_ALERT",
        entered_at: "2026-10-08T10:24:00Z",
        alert_at: "2026-10-08T10:24:00Z",
        dwell_seconds: 5,
        zone_index: 0,
        zone_name: "Zone 1",
        has_snapshot: false,
        acknowledged: false,
        acknowledged_at: null,
      },
    ];
    fireRows = [makeFire(), makeFire({ id: 3 })];
    const user = userEvent.setup();
    render(<App />);
    await waitForUi();
    // one zone plus two fire unacked
    expect(screen.getByTestId("alerts-badge").textContent).toBe("3");

    // the badge matches the sum of the Active counts on the Alerts page
    await user.click(screen.getByRole("button", { name: /^alerts( \d+)?$/i }));
    await waitFor(() => expect(screen.getByTestId("alert-list")).toBeTruthy());
    expect(screen.getByRole("tab", { name: "Active (1)" })).toBeTruthy();
    expect(screen.getByTestId("fire-subhead").textContent).toContain("Fire & smoke (2)");
  });

  it("shows the confirming indicator while the gate is confirming", async () => {
    state = makeState({
      running: true,
      fire_smoke: {
        enabled: true,
        confirming: { fire: null, smoke: { samples: 2, required: 5 } },
      },
    });
    render(<App />);
    await waitForUi();
    expect(screen.getByTestId("confirming").textContent).toContain(
      "Confirming possible smoke — 2 of 5 samples"
    );
  });

  it("opens fire evidence with its own snapshot", async () => {
    fireRows = [makeFire()];
    const user = userEvent.setup();
    render(<App />);
    await waitForUi();
    await openAlerts(user);
    await user.click(screen.getByRole("button", { name: /view evidence for fire alert 1/i }));
    await waitFor(() => expect(screen.getByRole("dialog")).toBeTruthy());
    expect(screen.getByText("FIRE · ORANGE")).toBeTruthy();
    const img = screen.getByAltText("Alert evidence") as HTMLImageElement;
    expect(img.getAttribute("src")).toBe("/fire-alerts/1/snapshot");
  });

  it("marks a fire row as false alarm after confirming, lowering the badge", async () => {
    fireRows = [makeFire()];
    const user = userEvent.setup();
    render(<App />);
    await waitForUi();
    // a lone fire row still raises the badge
    expect(screen.getByTestId("alerts-badge").textContent).toBe("1");
    await openAlerts(user);
    await user.click(screen.getByRole("button", { name: /^false alarm$/i }));
    await user.click(screen.getByRole("button", { name: /^confirm$/i }));
    await waitFor(() =>
      expect(
        fetchMock.mock.calls.some(
          ([url, init]) =>
            url === "/fire-alerts/1/false-alarm" && (init as RequestInit)?.method === "POST"
        )
      ).toBe(true)
    );
    // marking also acknowledges, so the row moves to the Acknowledged tab
    await user.click(screen.getByRole("tab", { name: /acknowledged/i }));
    await waitFor(() =>
      expect(screen.getByTestId("fire-falsealarm-1").textContent).toMatch(/Marked as false alarm/)
    );
    // nothing left unacked anywhere: no badge
    expect(screen.queryByTestId("alerts-badge")).toBeNull();
  });

  it("marks from the evidence modal and keeps it open showing marked", async () => {
    fireRows = [makeFire()];
    const user = userEvent.setup();
    render(<App />);
    await waitForUi();
    await openAlerts(user);
    await user.click(screen.getByRole("button", { name: /view evidence for fire alert 1/i }));
    await waitFor(() => expect(screen.getByRole("dialog")).toBeTruthy());
    const dialog = within(screen.getByRole("dialog"));
    await user.click(dialog.getByRole("button", { name: /^false alarm$/i }));
    await user.click(dialog.getByRole("button", { name: /^confirm$/i }));
    await waitFor(() =>
      expect(screen.getByTestId("modal-false-alarm-marked").textContent).toMatch(
        /Marked as false alarm/
      )
    );
    expect(screen.getByRole("dialog")).toBeTruthy();
  });

  it("shows a neutral off banner instead of SAFE when hazard is off", async () => {
    state = makeState({ running: true, status: "off", hazard: { enabled: false } });
    render(<App />);
    await waitForUi();
    const banner = screen.getByTestId("state-banner");
    expect(banner.textContent).toContain("OFF");
    expect(banner.textContent).toContain("Hazard monitoring off");
    expect(banner.className).toContain("idle");
  });

  it("reports the pipeline numbers and the uptime from /state", async () => {
    state = makeState({
      running: true,
      frame_seq: 42,
      perf: { fps: 29.5, ms_per_frame: 33.9 },
    });
    await renderSettings();

    const status = screen.getByTestId("system-status");
    expect(within(status).getByTestId("system-fps").textContent).toContain("29.5");
    expect(within(status).getByTestId("system-ms").textContent).toContain("33.9");
    expect(within(status).getByTestId("system-state").textContent).toContain("Running");
    // green dots while live, and the muted numbers still read as numbers
    expect(status.querySelectorAll(".status-dot.on")).not.toHaveLength(0);

    await act(async () => {
      state = makeState({ running: false, frame_seq: 0 });
    });
    await waitFor(() =>
      expect(within(status).getByTestId("system-state").textContent).toContain("Stopped")
    );
    expect(within(status).getByTestId("system-fps").textContent).toContain("-");
    expect(within(status).getByTestId("system-ms").textContent).toContain("-");
    expect(within(status).getByTestId("system-uptime").textContent).toContain("—");
  });

  it("saves the machine name from its own card and shows it on the monitor", async () => {
    const user = await renderSettings();

    const field = within(screen.getByTestId("machine-card")).getByLabelText(
      /machine name/i
    ) as HTMLInputElement;
    await user.clear(field);
    await user.type(field, "Kiln 3");
    await waitFor(() => expect(screen.getByText("Applied")).toBeTruthy());

    await user.click(button(/live monitor/i));
    await waitForUi();
    expect(screen.getByTestId("machine-name").textContent).toContain("Kiln 3");
  });

  it("lists the uploaded videos with use and delete actions, or the empty state", async () => {
    videos = [makeVideo({ id: "one", name: "line.mp4", source: "/data/videos/one.mp4" })];
    const user = await renderSettings();
    await openMoreSettings(user);

    const table = screen.getByTestId("video-list");
    const headers = [...table.querySelectorAll("thead th")].map((node) =>
      (node.textContent ?? "").trim()
    );
    expect(headers).toEqual(["Name", "Size", "Date", "Use", "Delete"]);

    cleanup();
    videos = [];
    window.location.hash = "#/";
    await renderSettings();
    expect(screen.getByText("No test videos uploaded yet.")).toBeTruthy();
    expect(screen.queryByTestId("video-list")).toBeNull();
  });
});

describe("the rules inside a zone card", () => {
  it("carries that zone's own allowance, delay and on/off switch", async () => {
    // This used to repeat the global settings inside the card, which made two
    // numbers claim to be the limit and a zone's own rule impossible to set.
    config = {
      ...CONFIG,
      zones: [[[100, 100], [400, 100], [400, 400], [100, 400]]],
      zone_specs: [
        {
          name: "Kiln Platform",
          allowed_people: 2,
          dwell_seconds: 3.5,
          enabled: true,
          using_defaults: false,
        },
      ],
    };
    const user = await renderEditing();

    await user.click(button(/^edit kiln platform$/i));
    const card = screen.getByTestId("zone-rules");

    expect(within(card).getByLabelText(/name for kiln platform/i)).toBeTruthy();
    expect((within(card).getByLabelText(/people allowed in kiln platform/i) as HTMLInputElement).value).toBe(
      "2"
    );
    expect(
      (within(card).getByLabelText(/alert after \(seconds\) for kiln platform/i) as HTMLInputElement).value
    ).toBe("3.5");
    expect((within(card).getByLabelText(/enabled for kiln platform/i) as HTMLInputElement).checked).toBe(
      true
    );
  });

  it("says which rules are the defaults and which are this zone's own", async () => {
    config = {
      ...CONFIG,
      zones: [[[100, 100], [400, 100], [400, 400], [100, 400]]],
      zone_specs: [
        {
          name: "Borrowed",
          allowed_people: CONFIG.allowed_people,
          dwell_seconds: CONFIG.rules.dwell_seconds,
          enabled: true,
          using_defaults: true,
        },
      ],
    };
    const user = await renderEditing();

    await user.click(button(/^edit borrowed$/i));
    const card = screen.getByTestId("zone-rules");

    expect(within(card).getByText(/using the default rules/i)).toBeTruthy();
  });

  it("keeps the machine-wide settings on the settings page only", async () => {
    config = { ...CONFIG, zones: [[[100, 100], [400, 100], [400, 400], [100, 400]]] };
    const user = await renderEditing();
    await user.click(button(/^edit zone 1$/i));

    // the global fields are not repeated in the card: the mode, the repeat
    // interval and the anchor are about the machine, not about one area
    const card = screen.getByTestId("zone-rules");
    expect(within(card).queryByLabelText(/^alert mode$/i)).toBeNull();
    expect(within(card).queryByLabelText(/repeat alert every/i)).toBeNull();
    expect(within(card).queryByLabelText(/detect person at/i)).toBeNull();
    // remove-all is a machine-wide action and stays on the settings page
    expect(within(card).queryByRole("button", { name: /remove all zones/i })).toBeNull();

    await user.click(button(/settings/i));
    await waitFor(() => expect(screen.getByTestId("settings-page")).toBeTruthy());
    expect(screen.getByLabelText(/^alert mode$/i)).toBeTruthy();
    expect(screen.getByLabelText(/repeat alert every \(seconds\)/i)).toBeTruthy();
  });
});

const ZONE_A: Point[] = [
  [100, 100],
  [400, 100],
  [400, 400],
  [100, 400],
];
const ZONE_B: Point[] = [
  [500, 200],
  [900, 200],
  [900, 500],
  [500, 500],
];

/** two saved, named zones with their own rules */
function namedZones(): AppConfig {
  return {
    ...CONFIG,
    allowed_people: 4,
    zones: [ZONE_A, ZONE_B],
    zone_specs: [
      {
        name: "Kiln Platform",
        allowed_people: 1,
        dwell_seconds: 3.5,
        enabled: true,
        using_defaults: false,
      },
      {
        name: "Sand Store",
        allowed_people: 6,
        dwell_seconds: 2,
        enabled: true,
        using_defaults: false,
      },
    ],
  };
}

describe("named zones", () => {
  it("loads the saved names onto the cards and the picture tags", async () => {
    config = namedZones();
    state = makeState({
      running: true,
      zone_polygons: [ZONE_A, ZONE_B],
      zone_specs: namedZones().zone_specs,
    });
    render(<App />);
    await waitForUi();

    await waitFor(() => expect(screen.getByTestId("zone-cards")).toBeTruthy());
    expect(chipLabels()).toEqual(["Kiln Platform", "Sand Store"]);
    expect(tagLabels()).toEqual(["Kiln Platform", "Sand Store"]);
  });

  it("renames a zone on the card and the picture without entering the editor", async () => {
    config = namedZones();
    state = makeState({ running: true, zone_polygons: [ZONE_A, ZONE_B] });
    const user = userEvent.setup();
    render(<App />);
    await waitForUi();

    await waitFor(() => expect(screen.getByTestId("zone-cards")).toBeTruthy());
    await user.click(screen.getByRole("button", { name: /rename kiln platform/i }));

    const box = screen.getByLabelText(/new name for kiln platform/i) as HTMLInputElement;
    await user.clear(box);
    await user.type(box, "Burner Deck{enter}");

    await waitFor(() => expect(chipLabels()).toEqual(["Burner Deck", "Sand Store"]));
    expect(tagLabels()).toEqual(["Burner Deck", "Sand Store"]);
    // renaming a name is not a geometry edit: no "unsaved geometry" question
    expect(screen.queryByText(/unsaved/i)).toBeNull();
  });

  it("puts a half typed name back when the rename box is escaped", async () => {
    config = namedZones();
    state = makeState({ running: true, zone_polygons: [ZONE_A, ZONE_B] });
    const user = userEvent.setup();
    render(<App />);
    await waitForUi();

    await waitFor(() => expect(screen.getByTestId("zone-cards")).toBeTruthy());
    await user.click(screen.getByRole("button", { name: /rename kiln platform/i }));
    const box = screen.getByLabelText(/new name for kiln platform/i) as HTMLInputElement;
    await user.clear(box);
    await user.type(box, "Half Typed");
    await user.keyboard("{Escape}");

    await waitFor(() => expect(chipLabels()).toEqual(["Kiln Platform", "Sand Store"]));
    expect(screen.queryByDisplayValue("Half Typed")).toBeNull();
  });

  it("saves the new name with the rest of the zone, index aligned", async () => {
    config = namedZones();
    state = makeState({ running: true, zone_polygons: [ZONE_A, ZONE_B] });
    const user = userEvent.setup();
    render(<App />);
    await waitForUi();

    await waitFor(() => expect(screen.getByTestId("zone-cards")).toBeTruthy());
    await user.click(screen.getByRole("button", { name: /rename kiln platform/i }));
    const box = screen.getByLabelText(/new name for kiln platform/i) as HTMLInputElement;
    await user.clear(box);
    await user.type(box, "Burner Deck{enter}");

    // the rename is saved with the rest of the zone when the operator saves
    await user.click(button(/^edit zones$/i));
    await waitFor(() => expect(screen.getByTestId("toolbar")).toBeTruthy());
    await user.click(button(/^save$/i));

    await waitFor(() => expect(postCount("/config")).toBeGreaterThan(0));
    const body = lastBody("/config");
    const specs = body.zone_specs as { name: string; allowed_people: number }[];
    expect(specs).toHaveLength(2);
    expect(specs[0].name).toBe("Burner Deck");
    // the rename must not disturb the rules already attached to that zone
    expect(specs[0].allowed_people).toBe(1);
    expect(specs[1].name).toBe("Sand Store");
    expect(specs[1].allowed_people).toBe(6);
  });

  it("gives a zone drawn just now a spec that follows the defaults", async () => {
    config = { ...CONFIG, allowed_people: 3 };
    const user = await renderEditing();

    await drawRectangle(user, [100, 100], [400, 300]);
    await user.click(button(/^save$/i));

    await waitFor(() => expect(postCount("/config")).toBeGreaterThan(0));
    expect(lastBody("/config").zone_specs).toEqual([
      {
        name: "Zone 1",
        allowed_people: 3,
        dwell_seconds: CONFIG.rules.dwell_seconds,
        enabled: true,
        using_defaults: true,
      },
    ]);
  });

  it("takes one zone off the defaults the moment its own number is typed", async () => {
    config = namedZones();
    const user = await renderEditing();

    await user.click(button(/^edit sand store$/i));
    const allowance = screen.getByLabelText(/people allowed in sand store/i) as HTMLInputElement;
    expect(within(screen.getByTestId("zone-rules")).getByText(/has its own rules/i)).toBeTruthy();

    await user.clear(allowance);
    await user.type(allowance, "2");
    // the other zone is untouched: defaults are per zone, not per machine
    await user.click(button(/^edit kiln platform$/i));
    const other = screen.getByLabelText(/people allowed in kiln platform/i) as HTMLInputElement;
    expect((other as HTMLInputElement).value).toBe("1");
  });

  it("keeps a zone's own delay, and greys it only when the global mode ignores it", async () => {
    config = namedZones();
    const user = await renderEditing();

    await user.click(button(/^edit kiln platform$/i));
    const dwell = screen.getByLabelText(
      /alert after \(seconds\) for kiln platform/i
    ) as HTMLInputElement;
    expect(dwell.value).toBe("3.5");
    expect(dwell.disabled).toBe(false);
  });

  it("greys the zone's own delay while over-the-limit mode ignores it", async () => {
    config = {
      ...namedZones(),
      rules: { mode: "LIMIT", dwell_seconds: 1 },
    };
    const user = await renderEditing();

    await user.click(button(/^edit kiln platform$/i));
    const dwell = screen.getByLabelText(
      /alert after \(seconds\) for kiln platform/i
    ) as HTMLInputElement;
    expect(dwell.disabled).toBe(true);
    expect(
      screen.getByText(/not used while only when over the limit is selected/i)
    ).toBeTruthy();
  });

  it("turns a zone off without touching the others", async () => {
    config = namedZones();
    const user = await renderEditing();

    await user.click(button(/^edit kiln platform$/i));
    await user.click(screen.getByLabelText(/enabled for kiln platform/i));

    await user.click(button(/^edit sand store$/i));
    expect((screen.getByLabelText(/enabled for sand store/i) as HTMLInputElement).checked).toBe(true);
  });

  it("forgets a deleted zone's rules with the zone", async () => {
    config = namedZones();
    const user = await renderEditing();

    await user.click(button(/^edit kiln platform$/i));
    await user.click(button(/delete zone/i));
    await user.click(button(/^save$/i));

    await waitFor(() => expect(postCount("/config")).toBeGreaterThan(0));
    const body = lastBody("/config");
    expect((body.zones as Point[][]).length).toBe(1);
    expect(body.zone_specs).toEqual([
      {
        name: "Sand Store",
        allowed_people: 6,
        dwell_seconds: 2,
        enabled: true,
        using_defaults: false,
      },
    ]);
  });

  it("lets the last zone be deleted: Save stays enabled and files empty", async () => {
    config = {
      ...CONFIG,
      allowed_people: 1,
      zones: [ZONE_A],
      zone_specs: [
        {
          name: "Kiln Platform",
          allowed_people: 1,
          dwell_seconds: 3.5,
          enabled: true,
          using_defaults: false,
        },
      ],
    };
    const user = await renderEditing();

    await user.click(button(/^edit kiln platform$/i));
    await user.click(button(/delete zone/i));
    // zero zones is a real, saveable outcome, not an invalid drawing
    expect((button(/^save$/i) as HTMLButtonElement).disabled).toBe(false);
    await user.click(button(/^save$/i));

    await waitFor(() => expect(postCount("/config")).toBeGreaterThan(0));
    expect(lastBody("/config").zones).toEqual([]);
  });
});

describe("what each zone card says about its own state", () => {
  it("reads each card's count and dot from that zone's own row of /state", async () => {
    config = namedZones();
    state = makeState({
      running: true,
      zone_polygons: [ZONE_A, ZONE_B],
      zone_specs: namedZones().zone_specs,
      inside: 1,
      status: "SAFE",
      zone_reports: [
        {
          index: 0,
          name: "Kiln Platform",
          inside: 1,
          allowed_people: 1,
          dwell_seconds: 3.5,
          alerting: false,
          enabled: true,
        },
        {
          index: 1,
          name: "Sand Store",
          inside: 0,
          allowed_people: 6,
          dwell_seconds: 2,
          alerting: false,
          enabled: true,
        },
      ],
    });
    render(<App />);
    await waitForUi();

    await waitFor(() => expect(screen.getByTestId("zone-cards")).toBeTruthy());
    const counts = [...document.querySelectorAll(".zone-card-count")].map(
      (node) => (node.textContent ?? "").replace(/\s+/g, " ").trim()
    );
    expect(counts).toEqual(["1 person", "0 people"]);

    // one person is standing in the first zone only: that zone's dot is amber,
    // the empty one stays green, and the global verdict is not consulted
    const dots = [...document.querySelectorAll(".zone-dot")].map((node) => node.className);
    expect(dots[0]).toContain("warn");
    expect(dots[1]).toContain("safe");
  });

  it("turns only the alerting zone's dot red", async () => {
    config = namedZones();
    state = makeState({
      running: true,
      zone_polygons: [ZONE_A, ZONE_B],
      zone_specs: namedZones().zone_specs,
      inside: 2,
      status: "ALERT",
      worker_alert: true,
      zone_reports: [
        {
          index: 0,
          name: "Kiln Platform",
          inside: 0,
          allowed_people: 1,
          dwell_seconds: 3.5,
          alerting: false,
          enabled: true,
        },
        {
          index: 1,
          name: "Sand Store",
          inside: 2,
          allowed_people: 6,
          dwell_seconds: 2,
          alerting: true,
          enabled: true,
        },
      ],
    });
    render(<App />);
    await waitForUi();

    await waitFor(() => expect(screen.getByTestId("zone-cards")).toBeTruthy());
    const dots = [...document.querySelectorAll(".zone-dot")].map((node) => node.className);
    expect(dots[0]).toContain("safe");
    expect(dots[1]).toContain("danger");
  });

  it("says a disabled zone is not being watched, rather than showing it as empty", async () => {
    config = namedZones();
    state = makeState({
      running: true,
      zone_polygons: [ZONE_A, ZONE_B],
      zone_specs: namedZones().zone_specs,
      zone_reports: [
        {
          index: 0,
          name: "Kiln Platform",
          inside: 0,
          allowed_people: 1,
          dwell_seconds: 3.5,
          alerting: false,
          enabled: false,
        },
        {
          index: 1,
          name: "Sand Store",
          inside: 0,
          allowed_people: 6,
          dwell_seconds: 2,
          alerting: false,
          enabled: true,
        },
      ],
    });
    render(<App />);
    await waitForUi();

    await waitFor(() => expect(screen.getByTestId("zone-cards")).toBeTruthy());
    const dots = [...document.querySelectorAll(".zone-dot")].map((node) => node.className);
    // unknown, not "safe": nothing is watching that zone, so nothing can be said
    expect(dots[0]).toContain("neutral");
    expect(dots[1]).toContain("safe");
  });

  it("keeps every dot neutral when the backend reports no per-zone rows", async () => {
    config = namedZones();
    state = makeState({ running: true, zone_polygons: [ZONE_A, ZONE_B], inside: 1 });
    render(<App />);
    await waitForUi();

    await waitFor(() => expect(screen.getByTestId("zone-cards")).toBeTruthy());
    const dots = [...document.querySelectorAll(".zone-dot")].map((node) => node.className);
    // borrowing the global count onto one arbitrary card would point the
    // operator at the wrong zone, so every card says "I do not know"
    expect(dots.every((name) => name.includes("neutral"))).toBe(true);
    const counts = [...document.querySelectorAll(".zone-card-count")].map(
      (node) => (node.textContent ?? "").replace(/\s+/g, " ").trim()
    );
    expect(counts).toEqual(["0 people", "0 people"]);
  });

  it("uses the global numbers for the one zone of an older backend", async () => {
    config = { ...CONFIG, zones: [ZONE_A], allowed_people: 2 };
    state = makeState({ running: true, zone_polygons: [ZONE_A], inside: 1 });
    render(<App />);
    await waitForUi();

    await waitFor(() => expect(screen.getByTestId("zone-cards")).toBeTruthy());
    const count = document.querySelector(".zone-card-count");
    expect((count?.textContent ?? "").replace(/\s+/g, " ").trim()).toBe("1 person");
    expect(document.querySelector(".zone-dot")?.className).toContain("warn");
  });

  it("says which zone the alarm came from, without inventing it", async () => {
    config = namedZones();
    state = makeState({
      running: true,
      zone_polygons: [ZONE_A, ZONE_B],
      zone_specs: namedZones().zone_specs,
      inside: 1,
      status: "ALERT",
      worker_alert: true,
      message: "person in Kiln Platform for 2.5s",
      zone_reports: [
        {
          index: 0,
          name: "Kiln Platform",
          inside: 1,
          allowed_people: 1,
          dwell_seconds: 3.5,
          alerting: true,
          enabled: true,
        },
        {
          index: 1,
          name: "Sand Store",
          inside: 0,
          allowed_people: 6,
          dwell_seconds: 2,
          alerting: false,
          enabled: true,
        },
      ],
      workers: [
        {
          worker_id: 1,
          state: "ALERT",
          dwell_seconds: 2.5,
          zones: [0],
          zone_index: 0,
          zone_name: "Kiln Platform",
        },
      ],
    });
    render(<App />);
    await waitForUi();

    // the banner repeats the backend's sentence, which names the zone: the
    // frontend never picks a zone of its own to word an alarm
    await waitFor(() =>
      expect(document.querySelector(".state-banner.alert")?.textContent).toContain("Kiln Platform")
    );
    // and the card that belongs to it is the red one
    const dots = [...document.querySelectorAll(".zone-dot")].map((node) => node.className);
    expect(dots[0]).toContain("danger");
    expect(dots[1]).toContain("safe");
  });

  it("puts each zone's people and longest dwell beside that zone's own limits", async () => {
    config = namedZones();
    state = makeState({
      running: true,
      zone_polygons: [ZONE_A, ZONE_B],
      zone_specs: namedZones().zone_specs,
      zone_reports: [
        {
          index: 0,
          name: "Kiln Platform",
          inside: 3,
          allowed_people: 1,
          dwell_seconds: 3.5,
          dwell_limit: 3.5,
          alerting: true,
          enabled: true,
        },
        {
          index: 1,
          name: "Sand Store",
          inside: 0,
          allowed_people: 6,
          dwell_seconds: 0.4,
          dwell_limit: 2,
          alerting: false,
          enabled: true,
        },
      ],
    });
    render(<App />);
    await waitForUi();

    await waitFor(() => expect(screen.getByTestId("zone-cards")).toBeTruthy());
    // 3 people against Kiln Platform's allowance of 1, not the machine-wide 4,
    // and 3.5s against its own 3.5s limit, not the global one
    expect(screen.getByTestId("zone-people-0").textContent).toMatch(/^3 \/ 1$/);
    expect(screen.getByTestId("zone-dwell-0").textContent).toMatch(/^3\.5s \/ 3\.5s$/);
    expect(screen.getByTestId("zone-people-1").textContent).toMatch(/^0 \/ 6$/);
    expect(screen.getByTestId("zone-dwell-1").textContent).toMatch(/^0\.4s \/ 2\.0s$/);

    // over its own limit is the one thing on the card worth colouring
    expect(screen.getByTestId("zone-people-0").className).toContain("over");
    expect(screen.getByTestId("zone-people-1").className).toBe("");
  });

  it("shows the figures while the card is closed, not only once it is opened", async () => {
    config = namedZones();
    state = makeState({
      running: true,
      zone_polygons: [ZONE_A, ZONE_B],
      zone_specs: namedZones().zone_specs,
      zone_reports: [
        {
          index: 0,
          name: "Kiln Platform",
          inside: 3,
          allowed_people: 1,
          dwell_seconds: 3.5,
          dwell_limit: 3.5,
          alerting: true,
          enabled: true,
        },
        {
          index: 1,
          name: "Sand Store",
          inside: 0,
          allowed_people: 6,
          dwell_seconds: 0.4,
          dwell_limit: 2,
          alerting: false,
          enabled: true,
        },
      ],
    });
    render(<App />);
    await waitForUi();

    await waitFor(() => expect(screen.getByTestId("zone-cards")).toBeTruthy());
    expect(document.querySelector(".zone-card-body")).toBeNull();
    expect(screen.getByTestId("zone-people-0")).toBeTruthy();
  });
});

describe("the two tiles under the video", () => {
  it("names the worst zone instead of quoting a total against one zone's limit", async () => {
    config = namedZones();
    state = makeState({
      running: true,
      zone_polygons: [ZONE_A, ZONE_B],
      zone_specs: namedZones().zone_specs,
      // 4 people in the machine, but the machine-wide allowance is 0: the old
      // "4 / 0" read as a breach while neither zone was over anything
      inside: 4,
      allowed_people: 0,
      zone_reports: [
        {
          index: 0,
          name: "Kiln Platform",
          inside: 4,
          allowed_people: 6,
          dwell_seconds: 3.5,
          dwell_limit: 3.5,
          alerting: false,
          enabled: true,
        },
        {
          index: 1,
          name: "Sand Store",
          inside: 0,
          allowed_people: 2,
          dwell_seconds: 0,
          dwell_limit: 2,
          alerting: false,
          enabled: true,
        },
      ],
    });
    render(<App />);
    await waitForUi();

    // the busiest zone is named, and its count sits beside ITS allowance
    expect(screen.getByTestId("tile-inside-label").textContent).toBe("People in Kiln Platform");
    expect(screen.getByTestId("tile-inside").textContent).toMatch(/^4 \/ 6$/);
    expect(screen.getByTestId("tile-dwell-label").textContent).toBe("Dwell time in Kiln Platform");
    expect(screen.getByTestId("tile-dwell").textContent).toMatch(/^3\.5s \/ 3\.5s$/);
    expect(screen.getByTestId("tile-inside").className).not.toContain("warn");
  });

  it("follows the zone that is actually over its allowance", async () => {
    config = namedZones();
    state = makeState({
      running: true,
      zone_polygons: [ZONE_A, ZONE_B],
      zone_specs: namedZones().zone_specs,
      inside: 3,
      zone_reports: [
        {
          index: 0,
          name: "Kiln Platform",
          inside: 1,
          allowed_people: 1,
          dwell_seconds: 1,
          dwell_limit: 3.5,
          alerting: false,
          enabled: true,
        },
        {
          index: 1,
          name: "Sand Store",
          inside: 2,
          allowed_people: 2,
          dwell_seconds: 0.5,
          dwell_limit: 2,
          alerting: true,
          enabled: true,
        },
      ],
    });
    render(<App />);
    await waitForUi();

    expect(screen.getByTestId("tile-inside-label").textContent).toBe("People in Sand Store");
    expect(screen.getByTestId("tile-inside").textContent).toMatch(/^2 \/ 2$/);
    expect(screen.getByTestId("tile-dwell").textContent).toMatch(/^0\.5s \/ 2\.0s$/);
  });

  it("leaves a disabled zone out of being named as the worst one", async () => {
    config = namedZones();
    state = makeState({
      running: true,
      zone_polygons: [ZONE_A, ZONE_B],
      zone_specs: namedZones().zone_specs,
      zone_reports: [
        {
          index: 0,
          name: "Kiln Platform",
          inside: 9,
          allowed_people: 1,
          dwell_seconds: 9,
          dwell_limit: 3.5,
          alerting: false,
          enabled: false,
        },
        {
          index: 1,
          name: "Sand Store",
          inside: 1,
          allowed_people: 2,
          dwell_seconds: 0.5,
          dwell_limit: 2,
          alerting: false,
          enabled: true,
        },
      ],
    });
    render(<App />);
    await waitForUi();

    // the zone that is switched off is not being watched, so it is not the one
    // the tiles point at, however full it looks in the picture
    expect(screen.getByTestId("tile-inside-label").textContent).toBe("People in Sand Store");
    expect(screen.getByTestId("tile-inside").textContent).toMatch(/^1 \/ 2$/);
  });

  it("labels a total as a total, and shows no limit, when the backend sends no zone rows", async () => {
    config = { ...CONFIG, zones: [ZONE_A, ZONE_B] };
    state = makeState({ running: true, inside: 3, allowed_people: 1 });
    render(<App />);
    await waitForUi();

    // no per-zone rows: a total is all there is, so it says so and puts no
    // single zone's allowance beside it
    expect(screen.getByTestId("tile-inside-label").textContent).toBe(
      "People in all zones (total)"
    );
    expect(screen.getByTestId("tile-inside").textContent).toMatch(/^3\b/);
    expect(screen.getByTestId("tile-inside").textContent).not.toMatch(/\//);
    expect(screen.getByTestId("tile-dwell-label").textContent).toBe(
      "Longest dwell in any zone (total)"
    );
    expect(screen.getByTestId("tile-dwell").textContent).not.toMatch(/\//);
  });

  it("leaves a single zone worded as before", async () => {
    config = { ...CONFIG, zones: [ZONE_A], allowed_people: 2 };
    state = makeState({
      running: true,
      zone_polygons: [ZONE_A],
      inside: 1,
      allowed_people: 2,
      zone_reports: [
        {
          index: 0,
          name: "Zone 1",
          inside: 1,
          allowed_people: 2,
          dwell_seconds: 0.5,
          dwell_limit: 4,
          alerting: false,
          enabled: true,
        },
      ],
    });
    render(<App />);
    await waitForUi();

    expect(screen.getByTestId("tile-inside-label").textContent).toBe("People in zone");
    expect(screen.getByTestId("tile-inside").textContent).toMatch(/^1 \/ 2$/);
    expect(screen.getByTestId("tile-dwell-label").textContent).toBe("Dwell time");
    expect(screen.getByTestId("tile-dwell").textContent).toMatch(/^0\.5s \/ 4\.0s$/);
  });

  it("moves the tiles when a zone's own allowance and alert-after change", async () => {
    config = namedZones();
    state = makeState({
      running: true,
      zone_polygons: [ZONE_A, ZONE_B],
      zone_specs: namedZones().zone_specs,
      inside: 0,
      zone_reports: [
        {
          index: 0,
          name: "Kiln Platform",
          inside: 0,
          allowed_people: 1,
          dwell_seconds: 0,
          dwell_limit: 3.5,
          alerting: false,
          enabled: true,
        },
        {
          index: 1,
          name: "Sand Store",
          inside: 0,
          allowed_people: 6,
          dwell_seconds: 0,
          dwell_limit: 2,
          alerting: false,
          enabled: true,
        },
      ],
    });
    const user = await renderEditing();

    // the tiles sit beside Kiln's own rules, not the machine defaults
    expect(screen.getByTestId("tile-inside").textContent).toMatch(/^0 \/ 1$/);
    expect(screen.getByTestId("tile-dwell").textContent).toMatch(/^0\.0s \/ 3\.5s$/);

    await user.click(button(/^edit kiln platform$/i));
    const allowance = screen.getByLabelText(/people allowed in kiln platform/i);
    await user.clear(allowance);
    await user.type(allowance, "6");
    const dwell = screen.getByLabelText(/alert after \(seconds\) for kiln platform/i);
    await user.clear(dwell);
    await user.type(dwell, "9");
    await user.click(button(/^save$/i));

    // the save carries the zone's new rules to the backend
    await waitFor(() => expect(postCount("/config")).toBeGreaterThan(0));
    const specs = lastBody("/config").zone_specs as Array<Record<string, unknown>>;
    expect(specs[0]).toMatchObject({ allowed_people: 6, dwell_seconds: 9 });

    // the backend applies them, and the next poll moves the tiles with them
    state = makeState({
      running: true,
      zone_polygons: [ZONE_A, ZONE_B],
      zone_specs: namedZones().zone_specs,
      inside: 1,
      zone_reports: [
        {
          index: 0,
          name: "Kiln Platform",
          inside: 1,
          allowed_people: 6,
          dwell_seconds: 0,
          dwell_limit: 9,
          alerting: false,
          enabled: true,
        },
        {
          index: 1,
          name: "Sand Store",
          inside: 0,
          allowed_people: 6,
          dwell_seconds: 0,
          dwell_limit: 2,
          alerting: false,
          enabled: true,
        },
      ],
    });
    await waitFor(() =>
      expect(screen.getByTestId("tile-inside").textContent).toMatch(/^1 \/ 6$/)
    );
    expect(screen.getByTestId("tile-dwell").textContent).toMatch(/^0\.0s \/ 9\.0s$/);
  });

  it("names the selected zone while it is being edited, not the busiest one", async () => {
    config = namedZones();
    state = makeState({
      running: true,
      zone_polygons: [ZONE_A, ZONE_B],
      zone_specs: namedZones().zone_specs,
      inside: 2,
      zone_reports: [
        {
          index: 0,
          name: "Kiln Platform",
          inside: 0,
          allowed_people: 1,
          dwell_seconds: 0,
          dwell_limit: 3.5,
          alerting: false,
          enabled: true,
        },
        {
          index: 1,
          name: "Sand Store",
          inside: 2,
          allowed_people: 2,
          dwell_seconds: 0.5,
          dwell_limit: 2,
          alerting: false,
          enabled: true,
        },
      ],
    });
    const user = await renderEditing();

    // Sand Store is busier, so the tiles point at it
    expect(screen.getByTestId("tile-inside-label").textContent).toBe("People in Sand Store");

    // picking Kiln Platform to edit it moves the tiles to Kiln's own rules
    await user.click(button(/^edit kiln platform$/i));
    expect(screen.getByTestId("tile-inside-label").textContent).toBe("People in Kiln Platform");
    expect(screen.getByTestId("tile-inside").textContent).toMatch(/^0 \/ 1$/);
    expect(screen.getByTestId("tile-dwell").textContent).toMatch(/^0\.0s \/ 3\.5s$/);
  });
});

describe("unlocking the sound", () => {
  /** A browser only hands out sound during a real interaction, and this app's
   *  alarm is silent until it has. Stubs the constructor so the test can see
   *  WHEN the context is made, and whether it is merely suspended. */
  function stubSuspendedAudio() {
    const audio = { created: 0, resumed: 0, state: "suspended" };
    class FakeContext {
      state = "suspended";
      currentTime = 0;
      destination = {} as AudioDestinationNode;
      createGain() {
        return {} as GainNode;
      }
      createOscillator() {
        return {} as OscillatorNode;
      }
      resume() {
        audio.resumed += 1;
        this.state = "running";
        return Promise.resolve();
      }
      constructor() {
        audio.created += 1;
      }
    }
    vi.stubGlobal("AudioContext", FakeContext);
    return audio;
  }

  it("opens the sound on the first click, with no Start button involved", async () => {
    const audio = stubSuspendedAudio();
    // the page has loaded onto a camera that was already running: there is no
    // Start click coming, which is the whole reason the alarm used to be silent
    state = makeState({ running: true, inside: 1, status: "ALERT" });
    render(<App />);
    await waitForUi();

    expect(audio.created).toBe(0);

    fireEvent.click(document.body);
    await waitFor(() => expect(audio.created).toBe(1));
    expect(audio.resumed).toBe(1);
  });

  it("opens the sound on the first key press instead", async () => {
    const audio = stubSuspendedAudio();
    state = makeState({ running: true });
    render(<App />);
    await waitForUi();

    fireEvent.keyDown(window, { key: "Escape" });
    await waitFor(() => expect(audio.created).toBe(1));
  });

  it("does it once, however much clicking follows", async () => {
    const audio = stubSuspendedAudio();
    state = makeState({ running: true });
    render(<App />);
    await waitForUi();

    fireEvent.click(document.body);
    await waitFor(() => expect(audio.created).toBe(1));
    fireEvent.click(document.body);
    fireEvent.keyDown(window, { key: "a" });
    fireEvent.click(document.body);

    // the handlers are synchronous, so anything meant to have run has run
    expect(audio.created).toBe(1);
    expect(audio.resumed).toBe(1);
  });

  it("leaves the first click alone when nothing has happened yet", async () => {
    const audio = stubSuspendedAudio();
    state = makeState({ running: false });
    render(<App />);
    await waitForUi();

    // stopped or not, the interaction counts: the operator may press Start next
    fireEvent.click(document.body);
    await waitFor(() => expect(audio.created).toBe(1));
    expect(audio.resumed).toBe(1);
  });
});

describe("status and footer", () => {
  it("shows STOPPED while nothing is running and RUNNING once the feed is live", async () => {
    const user = userEvent.setup();
    render(<App />);
    await waitForUi();
    expect(screen.getByText("STOPPED")).toBeTruthy();

    await startCamera(user);
    await waitFor(() => expect(document.querySelector(".status-pill")?.textContent).toBe("RUNNING"));
  });

  it("shows the ALERT banner as soon as a person is inside", async () => {
    state = makeState({
      running: true,
      inside: 1,
      status: "ALERT",
      worker_alert: true,
      message: "worker in danger zone",
    });
    render(<App />);
    await waitForUi();

    // the banner reads the real message from /state, not a fixed string
    await waitFor(() => expect(screen.getByTestId("state-banner")).toBeTruthy());
    const banner = screen.getByTestId("state-banner");
    expect(banner.textContent).toContain("ALERT");
    expect(banner.textContent).toContain("worker in danger zone");
  });

  it("offers a mute toggle on the alert banner", async () => {
    const user = userEvent.setup();
    state = makeState({ running: true, inside: 1, status: "ALERT", worker_alert: true });
    render(<App />);
    await waitForUi();

    const mute = await screen.findByRole("button", { name: /mute alert sound/i });
    expect(mute.getAttribute("aria-pressed")).toBe("false");
    await user.click(mute);
    await waitFor(() =>
      expect(screen.getByRole("button", { name: /unmute alert sound/i })).toBeTruthy()
    );
  });

  it("shows the SAFE chip on the video with no banner, and gives way to Start when stopped", async () => {
    state = makeState({ running: true, inside: 0 });
    render(<App />);
    await waitForUi();
    expect(screen.queryByTestId("state-banner")).toBeNull();
    expect(screen.getByTestId("status-badge").textContent).toBe("SAFE");

    // stopped is not a status banner at all: no picture, no verdict, no tiles
    cleanup();
    state = makeState({ running: false });
    render(<App />);
    await waitForUi();
    expect(screen.queryByTestId("state-banner")).toBeNull();
    expect(screen.queryByTestId("status-tiles")).toBeNull();
    expect(screen.getByTestId("start-panel")).toBeTruthy();
    expect(screen.queryByTestId("stopped-banner")).toBeNull();
  });

  it("renders exactly one video panel: the stopped card or the stream, never both", async () => {
    render(<App />);
    await waitForUi();

    // stopped: one dark card, no stream picture at all
    expect(screen.getByTestId("start-panel")).toBeTruthy();
    expect(document.querySelectorAll('[alt="camera stream"]')).toHaveLength(0);

    cleanup();
    state = makeState({ running: true });
    render(<App />);
    await waitForUi();
    expect(document.querySelectorAll('[alt="camera stream"]')).toHaveLength(1);
    expect(screen.queryByTestId("start-panel")).toBeNull();
  });

  it("names a missing test video in the stopped panel instead of a blank camera", async () => {
    state = makeState({
      running: false,
      source: "C:\\videos\\gone.mp4",
      source_error: "Test video not found at C:\\videos\\gone.mp4",
    });
    render(<App />);
    await waitForUi();
    expect(screen.getByTestId("start-panel")).toBeTruthy();
    expect(screen.getByTestId("stopped-hint").textContent).toContain(
      "Test video not found at C:\\videos\\gone.mp4"
    );

    // a usable source shows no hint at all
    cleanup();
    state = makeState({ running: false, source: "0" });
    render(<App />);
    await waitForUi();
    expect(screen.getByTestId("start-panel")).toBeTruthy();
    expect(screen.queryByTestId("stopped-hint")).toBeNull();
  });

  it("clears a timeout toast once the polls succeed again", async () => {
    let failState = true;
    fetchMock.mockImplementation((input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (failState && url.startsWith("/state") && init?.method !== "POST") {
        // through the real request() wrapper this reads as a timer abort,
        // exactly like a poll past its 8 s budget
        const abort = new Error("aborted");
        abort.name = "AbortError";
        return Promise.reject(abort);
      }
      return baseFetchImpl(input, init);
    });
    try {
      render(<App />);
      await waitForUi();
      // every poll failing raises the error toast once...
      await waitFor(() =>
        expect(
          screen.getByText("Request timed out: the backend took too long to answer")
        ).toBeTruthy()
      );
      // ...and the first good poll takes it away instead of leaving it stuck
      failState = false;
      await waitFor(() =>
        expect(
          screen.queryByText("Request timed out: the backend took too long to answer")
        ).toBeNull()
      );
    } finally {
      fetchMock.mockImplementation(baseFetchImpl);
    }
  });

  it("opens the editor on a dark canvas while stopped, sized from the saved reference", async () => {
    config = { ...CONFIG, zones: [], zone_reference: [640, 480] };
    state = makeState({ running: false, frame_width: 0, frame_height: 0 });
    const user = userEvent.setup();
    render(<App />);
    await waitForUi();

    await user.click(button(/^edit zones$/i));

    // the stopped card gives way to the drawing board: no stream, the saved
    // size written on it, tools out, Save waiting for a first shape
    await waitFor(() => expect(screen.getByTestId("blank-canvas")).toBeTruthy());
    expect(screen.getByTestId("blank-canvas").textContent).toContain("640 × 480");
    expect(document.querySelectorAll('[alt="camera stream"]')).toHaveLength(0);
    expect(screen.getByTestId("toolbar")).toBeTruthy();
    expect(screen.queryByTestId("start-panel")).toBeNull();
    expect((button(/^save$/i) as HTMLButtonElement).disabled).toBe(true);
  });

  it("keeps edit shut while stopped with no saved picture size", async () => {
    config = { ...CONFIG, zones: [] };
    state = makeState({ running: false, frame_width: 0, frame_height: 0 });
    render(<App />);
    await waitForUi();

    // nothing truthful to draw on: the button says why it is grey
    const edit = button(/^edit zones$/i) as HTMLButtonElement;
    expect(edit.disabled).toBe(true);
    expect(edit.title).toMatch(/picture size/);
  });

  it("shows a placeholder until the pipeline reports timings", async () => {
    // a stopped pipeline reports no frame at all, like the real backend
    state = makeState({ frame_width: 0, frame_height: 0 });
    await renderSettings();

    expect(screen.getByTestId("system-fps").textContent).toContain("-");
    expect(screen.getByTestId("system-ms").textContent).toContain("-");
    expect(screen.getByTestId("system-uptime").textContent).toContain("—");
    expect(screen.getByTestId("system-state").textContent).toContain("Stopped");
  });

  it("shows the FPS and ms/frame on the settings page, FPS also as a video badge", async () => {
    state = makeState({ running: true, perf: { fps: 29.5, ms_per_frame: 33.9 } });

    // the system block keeps the exact readings; timings left the monitor
    await renderSettings();
    expect(screen.getByTestId("system-fps").textContent).toContain("29.5");
    expect(screen.getByTestId("system-ms").textContent).toContain("33.9");
    expect(screen.getByTestId("system-state").textContent).toContain("Running");
    // the running feed's real frame size sits on the Camera card instead
    expect(screen.getByTestId("camera-resolution").textContent).toContain("1280x720");
    expect(document.querySelector(".perf-footer")).toBeNull();

    // the LIVE overlay badge is the monitor's only FPS indicator
    cleanup();
    window.location.hash = "#/";
    render(<App />);
    await waitForUi();
    expect(screen.getByTestId("live-badge")).toBeTruthy();
    expect(screen.getByTestId("fps-badge").textContent).toContain("29.5");
  });

  it("counts the zones in the chips under the video", async () => {
    config = {
      ...CONFIG,
      zones: [
        [[100, 100], [400, 100], [400, 400], [100, 400]],
        [[500, 200], [900, 200], [900, 500], [500, 500]],
      ],
    };
    state = makeState({ running: true });
    render(<App />);
    await waitForUi();

    await waitFor(() => expect(chipLabels()).toEqual(["Zone 1", "Zone 2"]));
    // the old footer is gone entirely: nothing on the monitor repeats the
    // picture timings except the LIVE badge
    expect(document.querySelector(".perf-footer")).toBeNull();
  });
});

describe("rules", () => {
  it("labels the source field with the full camera / path wording", async () => {
    const user = await renderSettings();
    await openMoreSettings(user);
    expect(screen.getByLabelText(/camera source.*video file path/i)).toBeTruthy();
  });

  it("shows the alert delay with a plain explanation and no fold", async () => {
    await renderSettings();

    // the delay here is the DEFAULT: each zone that has a delay of its own
    // overrides it, which is why the field says so
    const delay = screen.getByLabelText(/dwell limit \(seconds\)/i) as HTMLInputElement;
    expect(delay.value).toBe("1");
    expect(
      screen.getByText(/how long a person must stay in a zone that has no delay of its own/i)
    ).toBeTruthy();
    // the rules sit on the defaults card itself, not behind a fold
    expect(screen.getByLabelText(/repeat alert every \(seconds\)/i)).toBeTruthy();
    expect(screen.getByLabelText(/^people allowed$/i)).toBeTruthy();
    expect(screen.getByRole("radio", { name: /^feet$/i })).toBeTruthy();
  });

  it("keeps the camera source behind a closed Advanced fold, machine name on its own card", async () => {
    const user = await renderSettings();

    const details = document.querySelector("details.advanced") as HTMLDetailsElement;
    expect(details).toBeTruthy();
    // closed, so the raw source field is in the page but not on screen yet
    expect(details.open).toBe(false);
    expect(within(details).getByText("Advanced")).toBeTruthy();

    // the machine name is a card of its own now, not an advanced detail
    const machineCard = screen.getByTestId("machine-card");
    expect(within(machineCard).getByLabelText(/machine name/i)).toBeTruthy();
    expect(details.querySelector("input[placeholder='Excavator 01']")).toBeNull();

    await openMoreSettings(user);
    expect(screen.getByLabelText(/camera source.*video file path/i)).toBeTruthy();
  });

  it("names the alert mode in plain words", async () => {
    await renderSettings();

    const select = screen.getByLabelText(/^alert mode$/i) as HTMLSelectElement;
    expect(select.value).toBe("TIMER");
    // the wording follows what the backend does: a delay is compared against
    // the limit, the immediate mode alerts on entry and never reads it
    expect([...select.options].map((o) => o.textContent)).toEqual([
      "After a delay in the zone",
      "As soon as they enter",
      "Only when over the limit",
    ]);
  });

  it("greys the timers in over-the-limit mode and explains the entry mode", async () => {
    const user = await renderSettings();

    await user.selectOptions(screen.getByLabelText(/^alert mode$/i), "LIMIT");
    const dwell = screen.getByLabelText(/dwell limit \(seconds\)/i) as HTMLInputElement;
    const repeat = screen.getByLabelText(/repeat alert every \(seconds\)/i) as HTMLInputElement;
    expect(dwell.disabled).toBe(true);
    expect(repeat.disabled).toBe(true);
    expect(
      screen.getByText(/not used while only when over the limit is selected/i)
    ).toBeTruthy();
    expect(screen.getByText(/not used: only over-capacity raises an alert/i)).toBeTruthy();

    await user.selectOptions(screen.getByLabelText(/^alert mode$/i), "INSTANT");
    expect(
      screen.getByText(
        /every entry raises an alert, even within the allowed number/i
      )
    ).toBeTruthy();
  });

  it("names the two anchors in plain words, with a desk hint", async () => {
    await renderSettings();

    const feet = screen.getByRole("radio", { name: /^feet$/i }) as HTMLInputElement;
    const body = screen.getByRole("radio", { name: /^body center$/i }) as HTMLInputElement;
    expect(feet.checked).toBe(false);
    expect(body.checked).toBe(true);
    expect(feet.getAttribute("value")).toBe("bottom_center");
    expect(body.getAttribute("value")).toBe("center");
    expect(screen.getByText(/use body center for a desk demo/i)).toBeTruthy();
  });

  it("explains that zero repeats nothing and zero people alarms on anyone", async () => {
    await renderSettings();

    expect(screen.getByText(/don’t repeat the alert/i)).toBeTruthy();
    expect(screen.getByText(/alarms as soon as anyone is inside any zone/i)).toBeTruthy();
    // both zero values are actually allowed in the inputs
    expect((screen.getByLabelText(/repeat alert every \(seconds\)/i) as HTMLInputElement).min).toBe(
      "0"
    );
    expect((screen.getByLabelText(/^people allowed$/i) as HTMLInputElement).min).toBe(
      "0"
    );
  });

  it("saves a changed setting by itself and says Applied", async () => {
    const user = await renderSettings();
    const before = postCount("/config");

    const delay = screen.getByLabelText(/dwell limit \(seconds\)/i);
    await user.clear(delay);
    await user.type(delay, "3");

    // nothing is posted while the typing is still going
    await waitFor(() => expect(screen.getByText("Applied")).toBeTruthy());
    const bodies = postedBodies("/config");
    expect(bodies.length).toBeGreaterThan(before);
    expect(bodies[bodies.length - 1].dwell_seconds).toBe(3);
    // the zones are not part of a settings change
    expect(bodies[bodies.length - 1].zones).toBeUndefined();
  });

  it("changes the camera source on its own, with no settings ride-along", async () => {
    const user = await renderSettings();
    await openMoreSettings(user);
    const before = postCount("/config");

    // one edit, so the debounce has exactly one thing to say
    fireEvent.change(screen.getByLabelText(/camera source.*video file path/i), {
      target: { value: "2" },
    });

    await waitFor(() => expect(postCount("/config")).toBeGreaterThan(before));
    // only the source travels: a field that is not a settings edit must not
    // rewrite the rules sitting around it
    expect(postedBodies("/config").slice(before)).toEqual([{ source: "2" }]);
  });

  it("does not write the settings that arrived with the page", async () => {
    render(<App />);
    await waitForUi();
    await new Promise((resolve) => setTimeout(resolve, 700));
    expect(postCount("/config")).toBe(0);
  });

  it("sends every rule in one settings update", async () => {
    const user = await renderSettings();

    // the timers first, while the delay mode is still the one that uses them:
    // switching to INSTANT disables both, and a disabled input cannot be typed
    // into - which is the point of it
    const repeat = screen.getByLabelText(/repeat alert every \(seconds\)/i) as HTMLInputElement;
    await user.clear(repeat);
    await user.type(repeat, "0");
    await waitFor(() => expect(screen.getByText("Applied")).toBeTruthy());

    await user.selectOptions(screen.getByLabelText(/^alert mode$/i), "INSTANT");
    await user.click(screen.getByRole("radio", { name: /^feet$/i }));
    await waitFor(() => expect(screen.getByText("Applied")).toBeTruthy());

    const bodies = postedBodies("/config");
    const body = bodies[bodies.length - 1];
    expect(body.mode).toBe("INSTANT");
    expect(body.anchor).toBe("bottom_center");
    expect(body.repeat_alert_seconds).toBe(0);
    // a rule the mode ignores is still sent, so switching back keeps its value
    expect(body.dwell_seconds).toBe(1);
  });

  it("shows the failed reason when a settings save is refused", async () => {
    const user = await renderSettings();

    fetchMock.mockImplementation((input, init) => {
      if (String(input) === "/config" && init?.method === "POST") {
        return jsonResponse({ detail: "settings rejected" }, 422);
      }
      return jsonResponse({ saved: true });
    });

    const delay = screen.getByLabelText(/dwell limit \(seconds\)/i);
    await user.clear(delay);
    await user.type(delay, "4");
    await waitFor(() => expect(screen.getByText(/settings rejected/i)).toBeTruthy());
    expect(screen.queryByText("Applied")).toBeNull();
  });

  it("sends mode, dwell and repeat seconds with the zones", async () => {
    const user = await renderEditing();

    await drawRectangle(user, [200, 150], [700, 500]);
    await user.click(button(/^save$/i));
    await waitFor(() => expect(fetchMock.mock.calls.some((c) => String(c[0]) === "/config")).toBe(true));

    const body = lastBody("/config");
    expect(body.mode).toBe("TIMER");
    expect(body.dwell_seconds).toBe(1);
    expect(body.repeat_alert_seconds).toBe(10);
    expect(body.frame_width).toBe(1280);
    expect(body.frame_height).toBe(720);
  });

  it("starts the camera with no zone at all when nothing is saved", async () => {
    const user = userEvent.setup();
    render(<App />);
    await waitForUi();
    await startCamera(user);

    const body = lastBody("/start");
    expect(body.zones).toEqual([]);
  });

  it("sends the saved zones on start", async () => {
    const user = userEvent.setup();
    config = { ...CONFIG, zones: [[[100, 100], [400, 100], [400, 400], [100, 400]]] };
    render(<App />);
    await waitForUi();
    await startCamera(user);

    const body = lastBody("/start");
    expect((body.zones as number[][][]).length).toBe(1);
  });
});

/** how many zone polygons are actually drawn on the video */
function drawnZones(): Element[] {
  return [...document.querySelectorAll('svg [data-testid^="zone-shape-"]')];
}

/** the one primary action under the video */
function primaryButton(): HTMLButtonElement {
  return document.querySelector(".card-primary button.primary") as HTMLButtonElement;
}

/** Advanced holds the camera and machine details */
async function openMoreSettings(user: ReturnType<typeof userEvent.setup>) {
  const details = document.querySelector("details.advanced") as HTMLDetailsElement;
  await user.click(within(details).getByText(/advanced/i));
  return details;
}

/** picks a zone and opens its folded point list */
async function openZoneCoordinates(
  user: ReturnType<typeof userEvent.setup>,
  zone = 1
): Promise<HTMLDetailsElement> {
  await user.click(button(new RegExp(`^edit zone ${zone}$`, "i")));
  const panel = screen.getByTestId("coord-details") as HTMLDetailsElement;
  await user.click(within(panel).getByText(new RegExp(`zone ${zone} coordinates`, "i")));
  return panel;
}

describe("removing saved zones", () => {
  const SAVED: Point[][] = [
    [
      [100, 100],
      [400, 100],
      [400, 400],
      [100, 400],
    ],
    [
      [500, 200],
      [900, 200],
      [900, 500],
      [500, 500],
    ],
  ];

  /** a camera already monitoring SAVED, with the editor open on them */
  async function renderSavedZones() {
    config = { ...CONFIG, zones: SAVED };
    const user = await renderEditing();
    return user;
  }

  /** Remove all zones lives on the settings page, under Advanced */
  async function openRemoveAll() {
    const user = await renderSettings();
    await openMoreSettings(user);
    return user;
  }

  it("saves an empty zone list, so the old zones really go", async () => {
    config = { ...CONFIG, zones: SAVED };
    const user = await openRemoveAll();

    // Remove all zones -> confirm -> the empty save
    await user.click(button(/^remove all zones$/i));
    await user.click(button(/^remove all zones$/i));

    await waitFor(() => expect(postCount("/config")).toBeGreaterThan(0));
    expect(lastBody("/config").zones).toEqual([]);
    // the camera was off, so saving stays a save: nothing restarts the feed
    expect(postCount("/start")).toBe(0);
  });

  it("offers to draw again once every zone is gone, instead of a dead button", async () => {
    config = { ...CONFIG, zones: SAVED };
    state = makeState({ running: true });
    const user = await openRemoveAll();

    await user.click(button(/^remove all zones$/i));
    await user.click(button(/^remove all zones$/i));
    await user.click(button(/live monitor/i));
    await waitFor(() => expect(button(/^edit zones$/i)).toBeTruthy());

    // the toolbar is folded away again, but Edit zones brings it back
    expect(screen.queryByTestId("toolbar")).toBeNull();
    await user.click(button(/^edit zones$/i));
    expect(screen.getByTestId("toolbar")).toBeTruthy();
    // and Save stays greyed out until something is drawn
    expect((button(/^save$/i) as HTMLButtonElement).disabled).toBe(true);
  });

  it("asks before sending, and Cancel sends nothing", async () => {
    config = { ...CONFIG, zones: SAVED };
    const user = await openRemoveAll();

    await user.click(button(/^remove all zones$/i));
    // the confirm row replaces the plain button with one that means it
    expect(screen.getByRole("alertdialog", { name: /confirm removal/i })).toBeTruthy();
    expect(button(/^cancel$/i)).toBeTruthy();

    // and nothing is sent until that second click
    expect(postCount("/config")).toBe(0);
    await user.click(button(/^cancel$/i));
    expect(postCount("/config")).toBe(0);
  });

  it("offers Remove all zones on the settings page, whatever the camera is doing", async () => {
    config = { ...CONFIG, zones: SAVED };
    state = makeState({ running: false });
    const user = await renderSettings();
    await openMoreSettings(user);

    // reachable without opening the editor at all, and without a live feed
    expect(screen.queryByTestId("toolbar")).toBeNull();
    await user.click(button(/^remove all zones$/i));
    expect(button(/^cancel$/i)).toBeTruthy();

    await user.click(button(/^remove all zones$/i));
    await waitFor(() => expect(lastBody("/config").zones).toEqual([]));
    // stopped stays stopped: removing zones never starts the camera
    expect(postCount("/start")).toBe(0);
  });

  it("draws no saved zone underneath the editor once they are cleared", async () => {
    const user = await renderSavedZones();

    // the saved zones are on screen to begin with
    expect(drawnZones()).toHaveLength(2);
    expect(chipLabels()).toHaveLength(2);

    await pickTool(user, /^clear$/i);
    expect(drawnZones()).toHaveLength(0);
    expect(chipLabels()).toHaveLength(0);
  });

  it("counts the editor zones, not the saved ones", async () => {
    const user = await renderSavedZones();

    await pickTool(user, /^clear$/i);

    // one source of truth: the chips and the video itself both say there is
    // nothing left, so the two views can never disagree
    expect(chipLabels()).toHaveLength(0);
    expect(drawnZones()).toHaveLength(0);
  });

  it("counts every zone, not just the one being edited", async () => {
    const user = await renderSavedZones();

    // the two saved zones plus one more drawn in the editor
    await drawRectangle(user, [700, 500], [1000, 650]);

    // the chips and the video all count all three
    expect(chipLabels()).toEqual(["Zone 1", "Zone 2", "Zone 3"]);
    expect(drawnZones()).toHaveLength(3);
  });
});

describe("anchor point", () => {
  it("offers Feet and Body center, with Body center to begin with", async () => {
    await renderSettings();

    expect(
      (screen.getByRole("radio", { name: /^feet$/i }) as HTMLInputElement).checked
    ).toBe(false);
    expect(
      (screen.getByRole("radio", { name: /^body center$/i }) as HTMLInputElement).checked
    ).toBe(true);
  });

  it("reads the anchor the backend saved", async () => {
    config = { ...CONFIG, anchor: "bottom_center" };
    await renderSettings();

    await waitFor(() =>
      expect(
        (screen.getByRole("radio", { name: /^feet$/i }) as HTMLInputElement).checked
      ).toBe(true)
    );
  });

  it("sends the chosen anchor with the zones", async () => {
    const user = userEvent.setup();
    await renderSettings();
    await user.click(screen.getByRole("radio", { name: /^feet$/i }));
    await user.click(button(/live monitor/i));
    await openToolbar(user);
    await drawRectangle(user, [200, 150], [700, 500]);
    await user.click(button(/^save$/i));

    await waitFor(() => expect(postCount("/config")).toBeGreaterThan(0));
    expect(lastBody("/config").anchor).toBe("bottom_center");
    expect(lastBody("/start").anchor).toBe("bottom_center");
  });

  it("sends the body center unless it is changed", async () => {
    const user = await renderEditing();

    await drawRectangle(user, [200, 150], [700, 500]);
    await user.click(button(/^save$/i));

    await waitFor(() => expect(postCount("/config")).toBeGreaterThan(0));
    expect(lastBody("/config").anchor).toBe("center");
  });
});

describe("the drawing hint on the video", () => {
  /** the hint over the picture, which is the one that must not sit on a label */
  function videoHint(): Element | null {
    return document.querySelector(".stage .hint");
  }

  it("shows while editing with no tool picked and nothing drawn", async () => {
    const user = await renderEditing();
    expect(videoHint()?.textContent).toMatch(/pick a tool above/i);

    // Live shows no tools, so it never points the operator at one
    await user.click(button(/^cancel$/i));
    await waitFor(() => expect(videoHint()).toBeNull());
  });

  it("goes away the moment a tool is picked", async () => {
    const user = await renderEditing();
    expect(videoHint()).toBeTruthy();

    await pickTool(user, /^rectangle$/i);

    expect(videoHint()).toBeNull();
  });

  it("does not come back once a zone exists", async () => {
    const user = userEvent.setup();
    state = makeState({ running: true });
    render(<App />);
    await waitForUi();
    await openToolbar(user);

    await drawRectangle(user, [200, 150], [700, 500]);
    expect(videoHint()).toBeNull();
  });

  it("never sits on the video while a zone and its label are drawn", async () => {
    const user = userEvent.setup();
    state = makeState({ running: true });
    render(<App />);
    await waitForUi();
    await openToolbar(user);

    await drawRectangle(user, [200, 150], [700, 500]);

    expect(document.querySelector(".zone-tag-text")?.textContent).toBe("Zone 1");
    expect(videoHint()).toBeNull();
  });
});

describe("the right hand cards", () => {
  /** the headings of the cards beside the video, which must stay Status + Settings */
  function sideHeadings(): string[] {
    return [...document.querySelectorAll("aside.side section.card h2")].map(
      (node) => node.textContent ?? ""
    );
  }

  it("holds the danger zones and the recent alerts", async () => {
    render(<App />);
    await waitForUi();

    // the right column is systems, zones, counts, grade, recent alerts
    expect(sideHeadings()).toEqual([
      "Detection systems",
      "Danger zones",
      "People counted",
      "PPE grade",
      "Recent alerts",
    ]);
    expect(screen.queryByTestId("status-card")).toBeNull();
  });

  it("shows one status pill in the top bar and none in the side column", async () => {
    state = makeState({ running: true });
    render(<App />);
    await waitForUi();

    expect(document.querySelectorAll(".status-pill")).toHaveLength(1);
    expect(document.querySelector(".topbar .status-pill")?.textContent).toBe("RUNNING");
  });

  it("keeps the removal button out of the way and away from the drawing tools", async () => {
    config = { ...CONFIG, zones: [[[100, 100], [400, 100], [400, 400], [100, 400]]] };
    const user = await renderEditing();

    // Undo and Clear live in the toolbar, and nowhere else: removing every
    // zone is not next to them where it could be hit by accident
    const tools = toolButtons().map((node) => node.getAttribute("aria-label"));
    expect(tools).toContain("Undo");
    expect(tools).toContain("Clear");
    expect(screen.queryByRole("button", { name: /^remove all zones$/i })).toBeNull();

    await openSettings(user);
    await openMoreSettings(user);

    // it is on the settings page, under Advanced, and the zone cards cannot
    // remove anything either
    expect(button(/^remove all zones$/i)).toBeTruthy();
    const asideLabels = [...document.querySelectorAll("aside.side button")].map((node) =>
      (node.getAttribute("aria-label") ?? node.textContent ?? "").trim()
    );
    expect(asideLabels.some((label) => /^(undo|clear|reset zone)$/i.test(label))).toBe(false);
    expect(screen.queryByRole("button", { name: /^reset zone$/i })).toBeNull();
    // the settings page carries no drawing tools at all
    expect(screen.queryByTestId("toolbar")).toBeNull();
  });
});

describe("the video stage", () => {
  it("takes the frame's aspect ratio, so no strip is left beside the picture", async () => {
    state = makeState({ running: true, frame_width: 1024, frame_height: 768 });
    render(<App />);
    await waitForUi();

    await waitFor(() =>
      expect(document.querySelector(".stage")?.getAttribute("style")).toContain("1024 / 768")
    );
  });
});

describe("the off-screen anchor hint", () => {
  function withPoint(point: [number, number]) {
    return makeState({
      running: true,
      frame_width: 640,
      frame_height: 480,
      workers: [{ worker_id: 4, state: "OUTSIDE", dwell_seconds: 0, point }],
    });
  }

  it("warns when a person's anchor point has fallen out of the frame", async () => {
    // feet at y=520 on a 480 tall frame: the usual case is somebody who walked
    // close to the camera and had their bottom edge cropped off
    state = withPoint([320, 520]);
    render(<App />);
    await waitForUi();

    await waitFor(() => expect(screen.getByText(/anchor point is outside the frame/i)).toBeTruthy());
  });

  it("says why it matters, because the state alone looks unremarkable", async () => {
    state = withPoint([320, 520]);
    render(<App />);
    await waitForUi();

    // without this the operator just sees OUTSIDE and assumes the zone is wrong
    expect(screen.getByText(/cannot be counted as inside a zone/i)).toBeTruthy();
  });

  it("suggests Body center when the anchor is Feet, which does fix cropped feet", async () => {
    config = { ...CONFIG, anchor: "bottom_center" };
    state = withPoint([320, 520]);
    render(<App />);
    await waitForUi();

    expect(screen.getByText(/try body center/i)).toBeTruthy();
  });

  it("does not suggest Body center when that is already the anchor", async () => {
    // suggesting the anchor they are already using would be nonsense advice
    config = { ...CONFIG, anchor: "center" };
    state = withPoint([320, 520]);
    render(<App />);
    await waitForUi();

    expect(screen.getByTestId("anchor-hint")).toBeTruthy();
    expect(screen.queryByText(/try body center/i)).toBeNull();
  });

  it("says nothing when the anchor point is inside the frame", async () => {
    state = withPoint([320, 400]);
    render(<App />);
    await waitForUi();

    expect(screen.queryByTestId("anchor-hint")).toBeNull();
  });

  it("stays quiet when a backend sends no point at all", async () => {
    // an older backend has no "point" field; the UI must still run
    state = makeState({
      running: true,
      frame_width: 640,
      frame_height: 480,
      workers: [{ worker_id: 4, state: "OUTSIDE", dwell_seconds: 0 }],
    });
    render(<App />);
    await waitForUi();

    expect(screen.queryByTestId("anchor-hint")).toBeNull();
  });

  it("counts the people it is talking about", async () => {
    state = makeState({
      running: true,
      frame_width: 640,
      frame_height: 480,
      workers: [
        { worker_id: 1, state: "OUTSIDE", dwell_seconds: 0, point: [100, 520] },
        { worker_id: 2, state: "INSIDE", dwell_seconds: 0, point: [300, 200] },
        { worker_id: 3, state: "OUTSIDE", dwell_seconds: 0, point: [500, 900] },
      ],
    });
    render(<App />);
    await waitForUi();

    // one is in frame, so this is not "one person" but two
    expect(screen.getByText(/2 people's anchor points are outside the frame/i)).toBeTruthy();
  });
});

describe("the blue status bar", () => {
  beforeEach(() => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("clears a confirmation after four seconds", async () => {
    const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
    state = makeState({ running: true });
    render(<App />);
    await waitForUi();
    await openToolbar(user);

    await drawRectangle(user, [200, 150], [700, 500]);
    await user.click(button(/^save$/i));
    await waitFor(() => expect(screen.getByText(/zone saved/i)).toBeTruthy());

    // still there just before the four seconds are up
    await act(async () => {
      vi.advanceTimersByTime(3900);
    });
    expect(screen.getByText(/zone saved/i)).toBeTruthy();

    await act(async () => {
      vi.advanceTimersByTime(200);
    });
    expect(screen.queryByText(/zone saved/i)).toBeNull();
  });

  it("keeps an error on screen, because it explains why nothing happened", async () => {
    state = makeState({ running: false, error: "camera busy" });
    render(<App />);
    await waitForUi();

    await waitFor(() => expect(screen.getByText(/camera busy/i)).toBeTruthy());
    await act(async () => {
      vi.advanceTimersByTime(10000);
    });
    expect(screen.getByText(/camera busy/i)).toBeTruthy();
  });
});

describe("the polish pass", () => {
  it("has no stepper: the three states say what is happening instead", async () => {
    const user = userEvent.setup();
    render(<App />);
    await waitForUi();

    // the old Start / Draw / Save guide is gone in every state
    expect(document.querySelectorAll("ol.stepper")).toHaveLength(0);
    expect(screen.queryByTestId("setup-done")).toBeNull();
    expect(document.querySelector(".status-pill")?.textContent).toBe("STOPPED");

    await startCamera(user);
    await waitFor(() =>
      expect(document.querySelector(".status-pill")?.textContent).toBe("RUNNING")
    );
    expect(document.querySelectorAll("ol.stepper")).toHaveLength(0);

    await user.click(button(/^edit zones$/i));
    await waitFor(() => expect(screen.getByTestId("toolbar")).toBeTruthy());
    expect(document.querySelector(".status-pill")?.textContent).toBe("Editing");
  });

  it("keeps the settings off the monitoring screen and on their own page", async () => {
    state = makeState({ running: true });
    render(<App />);
    await waitForUi();

    // Live is for watching: the settings page never competes with the picture
    expect(screen.queryByTestId("settings-page")).toBeNull();
    expect(document.querySelector(".side")?.textContent).toMatch(/danger zones/i);
  });

  it("greys the delay and the repeat while the mode ignores them", async () => {
    const user = await renderSettings();

    const dwell = screen.getByLabelText(/dwell limit \(seconds\)/i) as HTMLInputElement;
    const repeat = screen.getByLabelText(/repeat alert every \(seconds\)/i) as HTMLInputElement;
    // TIMER is what the settings load as, and it does use both
    expect(dwell.disabled).toBe(false);
    expect(repeat.disabled).toBe(false);

    await user.selectOptions(screen.getByLabelText(/^alert mode$/i), "INSTANT");
    expect(dwell.disabled).toBe(true);
    expect(repeat.disabled).toBe(true);
    expect(screen.getByText(/not used while as soon as they enter is selected/i)).toBeTruthy();
    expect(screen.getByText(/not used: an immediate alert fires once per entry/i)).toBeTruthy();

    // and back again
    await user.selectOptions(screen.getByLabelText(/^alert mode$/i), "TIMER");
    expect(dwell.disabled).toBe(false);
    expect(repeat.disabled).toBe(false);
    expect(
      screen.getByText(/how long a person must stay in a zone that has no delay of its own/i)
    ).toBeTruthy();
  });

  it("says the allowed-people rule is checked whatever the alert mode is", async () => {
    await renderSettings();
    // the capacity check runs every frame, so the field is never dimmed
    expect(screen.getByLabelText(/^people allowed$/i).closest(".field-off")).toBeNull();
    expect(
      screen.getByText(/each zone is checked against its own number/i)
    ).toBeTruthy();
  });

  it("keys the box colours under the video", async () => {
    state = makeState({
      running: true,
      hazard: { enabled: true },
      fire_smoke: { enabled: true, fire_enabled: true, smoke_enabled: true },
      ppe: { enabled: true },
    });
    render(<App />);
    await waitForUi();

    const legend = await screen.findByTestId("legend");
    expect([...legend.querySelectorAll(".legend-item")].map((n) => n.textContent)).toEqual([
      "Person safe",
      "Person in zone",
      "Danger zone",
      "Fire",
      "Smoke",
      "PPE violation",
    ]);
    expect(legend.querySelector(".legend-item.safe")).toBeTruthy();
    expect(legend.querySelector(".legend-item.danger")).toBeTruthy();
    expect(legend.querySelector(".legend-item.warn")).toBeTruthy();
    expect(legend.querySelector(".legend-item.fire")).toBeTruthy();
    expect(legend.querySelector(".legend-item.smoke")).toBeTruthy();
    expect(legend.querySelector(".legend-item.ppe")).toBeTruthy();
  });

  it("hides unticked systems from the legend", async () => {
    state = makeState({
      running: true,
      hazard: { enabled: false },
      fire_smoke: { enabled: true, fire_enabled: false, smoke_enabled: true },
      ppe: { enabled: false },
    });
    render(<App />);
    await waitForUi();

    const legend = await screen.findByTestId("legend");
    const labels = [...legend.querySelectorAll(".legend-item")].map((n) => n.textContent);
    expect(labels).toEqual(["Smoke"]);
  });

  it("shows the detection systems panel bound to the saved switches", async () => {
    config = {
      ...CONFIG,
      hazard: { enabled: false, analysis_fps: 2 },
      fire_smoke: { enabled: true, fire_enabled: true, smoke_enabled: false, analysis_fps: 2 },
    };
    state = makeState({ running: true, hazard: { enabled: false } });
    render(<App />);
    await waitForUi();

    const panel = screen.getByTestId("systems-card");
    const boxes = [...panel.querySelectorAll('input[type="checkbox"]')] as HTMLInputElement[];
    expect(boxes.map((box) => box.checked)).toEqual([false, true, false, false, false]);
  });

  it("hides the zone overlay when hazard is unticked but keeps the cards", async () => {
    const ZONE: [number, number][] = [[10, 10], [200, 10], [200, 200]];
    config = { ...CONFIG, zones: [ZONE] };
    state = makeState({ running: true, hazard: { enabled: true } });
    const user = userEvent.setup();
    render(<App />);
    await waitForUi();
    await waitFor(() =>
      expect(document.querySelectorAll('[data-testid^="zone-shape-"]')).toHaveLength(1)
    );
    // untick in the panel, then the backend applies it (mirrored the way the
    // start helper flips the running flag on the mock)
    await user.click(screen.getByLabelText("Hazard zone enabled"));
    state = makeState({ running: true, hazard: { enabled: false } });
    await waitFor(() =>
      expect(document.querySelectorAll('[data-testid^="zone-shape-"]')).toHaveLength(0)
    );
    // the card stays with its config, neutralised instead of green
    expect(screen.getByTestId("zone-cards")).toBeTruthy();
    expect(screen.getByTestId("state-banner").textContent).toContain("Hazard monitoring off");
  });

  it("clamps coordinates typed into the point table onto the frame", async () => {
    const user = userEvent.setup();
    state = makeState({ running: true });
    render(<App />);
    await waitForUi();
    await openToolbar(user);

    await drawRectangle(user, [200, 150], [700, 500]);

    // the point table is the one way a corner could leave the frame: drawing
    // clamps every point, so nothing gets there by dragging
    await openZoneCoordinates(user, 1);
    const x = screen.getByLabelText(/^point 1 x$/i) as HTMLInputElement;
    await user.clear(x);
    await user.type(x, "1400");
    await waitFor(() => expect(x.value).toBe("1280"));

    const y = screen.getByLabelText(/^point 1 y$/i) as HTMLInputElement;
    await user.clear(y);
    await user.type(y, "9999");
    await waitFor(() => expect(y.value).toBe("720"));
  });

it("still asks twice before removing every zone", async () => {
    config = {
      ...CONFIG,
      zones: [[[100, 100], [400, 100], [400, 400], [100, 400]]],
    };
    const user = await renderSettings();
    await openMoreSettings(user);

    await user.click(button(/^remove all zones$/i));
    // the first click only asks
    expect(screen.getByRole("alertdialog", { name: /confirm removal/i })).toBeTruthy();
    expect(postCount("/config")).toBe(0);
    expect(postCount("/start")).toBe(0);

    // and Cancel is a way out
    await user.click(button(/^cancel$/i));
    expect(screen.queryByRole("alertdialog", { name: /confirm removal/i })).toBeNull();
    expect(postCount("/config")).toBe(0);

await user.click(button(/^remove all zones$/i));
    await user.click(button(/^remove all zones$/i));
    await waitFor(() => expect(lastBody("/config").zones).toEqual([]));
  });
});

describe("switching source with unsaved zone work", () => {
  const CLIP_ZONE: Point[] = [
    [600, 300],
    [900, 300],
    [900, 600],
    [600, 600],
  ];

  it("holds the switch, asks first, and only Discard lets it through", async () => {
    config = namedZones();
    configBySource["1"] = { ...CONFIG, source: "1", zones: [CLIP_ZONE], zone_specs: [] };
    // the camera is live with its two saved zones, and the editor is open
    const user = await renderEditing();

    // a third rectangle: the editor now holds work the backend has not seen
    await drawRectangle(user, [700, 400], [1100, 650]);
    await waitFor(() => expect(shapeCount()).toBe(3));

    // the Camera source field lives on the settings page, under Advanced
    await openSettings(user);
    await openMoreSettings(user);
    const field = () => screen.getByLabelText(/camera source.*video file path/i) as HTMLInputElement;

    fireEvent.change(field(), { target: { value: "1" } });

    // the question is asked and the field goes back to the source in use
    await waitFor(() =>
      expect(screen.getByRole("alertdialog", { name: /confirm source switch/i })).toBeTruthy()
    );
    expect(screen.getByText(/you have unsaved zone changes\. discard them\?/i)).toBeTruthy();
    expect(field().value).toBe("0");
    // the other source's zones were not fetched either, and a switch that was
    // held never got as far as the backend
    expect(fetchMock.mock.calls.some((call) => String(call[0]).includes("source=1"))).toBe(false);

    await user.click(button(/^cancel$/i));
    await waitFor(() =>
      expect(screen.queryByRole("alertdialog", { name: /confirm source switch/i })).toBeNull()
    );
    expect(field().value).toBe("0");
    expect(postCount("/config")).toBe(0);

    // the work survived: back on the monitor all three zones are still there
    await user.click(screen.getByRole("button", { name: /live monitor/i }));
    await waitFor(() => expect(chipLabels()).toHaveLength(3));

    // ask again, and this time Discard is what lets the switch through
    await openSettings(user);
    await openMoreSettings(user);
    fireEvent.change(field(), { target: { value: "1" } });
    await waitFor(() =>
      expect(screen.getByRole("alertdialog", { name: /confirm source switch/i })).toBeTruthy()
    );
    await user.click(button(/^discard$/i));

    await waitFor(() => expect(field().value).toBe("1"));
    await waitFor(() =>
      expect(
        fetchMock.mock.calls.some(
          (call) => String(call[0]) === "/config?source=" + encodeURIComponent("1")
        )
      ).toBe(true)
    );
    // and the editor now holds the new source's own zones, not the old ones
    await user.click(screen.getByRole("button", { name: /live monitor/i }));
    await waitFor(() => expect(chipLabels()).toEqual(["Zone 1"]));
  });
});

describe("uploaded test videos", () => {
  /** render the live monitor while the camera is stopped. Uploading and
   *  picking a source live on the Settings page now, under Camera. */
  async function renderStopped() {
    const user = userEvent.setup();
    render(<App />);
    await waitForUi();
    return user;
  }

  /** open Settings, where the upload button and the video list live */
  async function openTestVideo(user: ReturnType<typeof userEvent.setup>) {
    await user.click(button(/^settings$/i));
    await waitFor(() => expect(screen.getByTestId("video-file")).toBeTruthy());
  }

  /** back to the monitor, where the feed is started */
  async function openMonitor(user: ReturnType<typeof userEvent.setup>) {
    await user.click(button(/^live monitor$/i));
    await waitFor(() => expect(screen.getByTestId("live-page")).toBeTruthy());
  }

  /** pick a file and hand back the fake request waiting for a reply */
  async function pickFile(name = "clip.mp4") {
    const input = screen.getByTestId("video-file") as HTMLInputElement;
    const file = new File(["video-bytes"], name, { type: "video/mp4" });
    fireEvent.change(input, { target: { files: [file] } });
    await waitFor(() => expect(xhrInstances.length).toBeGreaterThan(0));
    return xhrInstances[xhrInstances.length - 1];
  }

  /** drop a file onto the dropzone and hand back the fake request */
  async function dropFile(name = "clip.mp4") {
    const file = new File(["video-bytes"], name, { type: "video/mp4" });
    fireEvent.drop(screen.getByTestId("dropzone"), { dataTransfer: { files: [file] } });
    await waitFor(() => expect(xhrInstances.length).toBeGreaterThan(0));
    return xhrInstances[xhrInstances.length - 1];
  }

  const uploaded = {
    id: "abc",
    file: "abc.mp4",
    name: "kiln run.mp4",
    size_mb: 12.5,
    width: 640,
    height: 480,
    fps: 25,
    duration: 3.2,
    created: "2026-01-01T00:00:00Z",
    source: "/data/videos/abc.mp4",
  };

  it("shows a progress bar while uploading, then lists the video with the backend's facts", async () => {
    const user = await renderStopped();
    await openTestVideo(user);
    const xhr = await pickFile("kiln run.mp4");

    const bar = (await screen.findByTestId("upload-progress")) as HTMLProgressElement;
    expect(bar.value).toBe(50);

    await act(async () => {
      xhr.respond(200, uploaded);
    });

    await waitFor(() => expect(screen.getByTestId("video-list")).toBeTruthy());
    expect(screen.getByText("kiln run.mp4")).toBeTruthy();
    // size and date come from the reply, not the file object
    expect(screen.getByText("12.5 MB")).toBeTruthy();
    expect(screen.getByText("2026-01-01")).toBeTruthy();
  });

  it("uploads a video dropped onto the card", async () => {
    const user = await renderStopped();
    await openTestVideo(user);
    const xhr = await dropFile("dropped.mp4");

    await act(async () => {
      xhr.respond(200, { ...uploaded, id: "d1", name: "dropped.mp4", source: "/data/videos/d1.mp4" });
    });

    await waitFor(() => expect(screen.getByText("dropped.mp4")).toBeTruthy());
  });

  it.each([
    [413, "that video is larger than the 500 MB limit"],
    [415, "'txt' is not a video. Allowed types: .mp4, .avi, .mov, .mkv, .webm"],
    [422, "that file could not be read as a video - no frame could be decoded"],
  ])("shows the backend's message when an upload fails with %i", async (status, detail) => {
    const user = await renderStopped();
    await openTestVideo(user);
    const xhr = await pickFile();

    await act(async () => {
      xhr.respond(status, { detail });
    });

    await waitFor(() => expect(screen.getByText(detail)).toBeTruthy());
  });

  it("selects a video, sets the camera source and shows the TEST VIDEO badge", async () => {
    videos = [makeVideo({ id: "one", name: "line.mp4", source: "/data/videos/one.mp4" })];
    const user = await renderStopped();
    await openTestVideo(user);

    await user.click(screen.getByRole("button", { name: /^use line\.mp4/i }));

    await waitFor(() => expect(screen.getByTestId("test-video-tag")).toBeTruthy());
    // the one camera source field is what changed, so /config carries it
    await waitFor(() =>
      expect(postedBodies("/config").some((body) => body.source === "/data/videos/one.mp4")).toBe(
        true
      )
    );
  });

  it("swaps in the selected clip's own zones, never the ones on the camera", async () => {
    // the camera has two named zones; the clip has one of its own
    config = { ...namedZones(), zones: [ZONE_A, ZONE_B] };
    const clipZone: Point[] = [
      [40, 40],
      [240, 40],
      [240, 240],
      [40, 240],
    ];
    configBySource["/data/videos/one.mp4"] = {
      ...CONFIG,
      source: "/data/videos/one.mp4",
      zones: [clipZone],
      zone_specs: [],
    };
    videos = [makeVideo({ id: "one", name: "line.mp4", source: "/data/videos/one.mp4" })];

    const user = await renderStopped();
    await openTestVideo(user);
    await user.click(screen.getByRole("button", { name: /^use line\.mp4/i }));

    // the clip's zones are asked for by that source specifically
    await waitFor(() =>
      expect(
        fetchMock.mock.calls.some(
          (call) =>
            String(call[0]) ===
            "/config?source=" + encodeURIComponent("/data/videos/one.mp4")
        )
      ).toBe(true)
    );

    // starting the feed sends the clip's zones, not the camera's two
    await openMonitor(user);
    await user.click(button(/^start camera$/i));
    await waitFor(() => expect(lastBody("/start").zones).toEqual([clipZone]));
  });

  it("moves a running feed to the picked clip with that clip's zones", async () => {
    // the camera has two named zones; the clip has one of its own
    config = { ...namedZones(), zones: [ZONE_A, ZONE_B] };
    const clipZone: Point[] = [
      [40, 40],
      [240, 40],
      [240, 240],
      [40, 240],
    ];
    configBySource["/data/videos/one.mp4"] = {
      ...CONFIG,
      source: "/data/videos/one.mp4",
      zones: [clipZone],
      zone_specs: [],
    };
    videos = [makeVideo({ id: "one", name: "line.mp4", source: "/data/videos/one.mp4" })];
    state = makeState({ running: true });

    const user = await renderStopped();
    await openTestVideo(user);
    await user.click(screen.getByRole("button", { name: /^use line\.mp4$/i }));

    // the feed restarts on the new clip instead of staying on the old picture
    await waitFor(() => {
      const body = lastBody("/start");
      expect(body.source).toBe("/data/videos/one.mp4");
      expect(body.zones).toEqual([clipZone]);
    });
  });

  it("marks the selected video in use and offers it as the monitor source", async () => {
    videos = [makeVideo({ id: "one", name: "line.mp4", source: "/data/videos/one.mp4" })];
    const user = await renderStopped();
    await openTestVideo(user);

    await user.click(screen.getByRole("button", { name: /^use line\.mp4/i }));
    await waitFor(() => expect(screen.getByTestId("test-video-tag")).toBeTruthy());
    // the picked row is marked and cannot be picked twice
    expect(screen.getByText("In use")).toBeTruthy();
    expect(screen.queryByRole("button", { name: /^use line\.mp4/i })).toBeNull();

    // and starting the feed opens that clip
    await openMonitor(user);
    await user.click(button(/^start camera$/i));
    await waitFor(() => expect(lastBody("/start").source).toBe("/data/videos/one.mp4"));
  });

  it("deletes a video from the list, but only after confirming", async () => {
    videos = [makeVideo({ id: "one", name: "line.mp4", source: "/data/videos/one.mp4" })];
    const user = await renderStopped();
    await openTestVideo(user);

    expect(screen.getByText("line.mp4")).toBeTruthy();
    await user.click(screen.getByRole("button", { name: /^delete line\.mp4$/i }));

    // first click only asks: the row is still there with a way out
    expect(screen.getByText("line.mp4")).toBeTruthy();
    expect(
      screen.getByRole("alertdialog", { name: /confirm deletion of line\.mp4/i })
    ).toBeTruthy();
    await user.click(button(/^cancel$/i));
    expect(screen.getByText("line.mp4")).toBeTruthy();

    // confirming really deletes it
    await user.click(screen.getByRole("button", { name: /^delete line\.mp4$/i }));
    await user.click(button(/^confirm$/i));
    await waitFor(() => expect(screen.queryByText("line.mp4")).toBeNull());
  });

  it("refuses to delete the video that is in use", async () => {
    videos = [makeVideo({ id: "one", name: "line.mp4", source: "/data/videos/one.mp4" })];
    config = { ...CONFIG, source: "/data/videos/one.mp4" };
    const user = await renderStopped();
    await openTestVideo(user);

    // the source is a test video, so Settings opens straight on that half
    await waitFor(() => expect(screen.getByText("line.mp4")).toBeTruthy());
    expect(button(/^delete line\.mp4$/i).disabled).toBe(true);
  });

  it("warns that the zones do not fit a video of another size, without deleting them", async () => {
    config = { ...namedZones(), zone_reference: [1280, 720] };
    videos = [
      makeVideo({ id: "small", name: "small.mp4", width: 640, height: 480, source: "/data/videos/small.mp4" }),
    ];
    const user = await renderStopped();
    await openTestVideo(user);

    await user.click(screen.getByRole("button", { name: /^use small\.mp4/i }));

    await waitFor(() =>
      expect(screen.getByText(/drawn for another frame size/i)).toBeTruthy()
    );
    // the zones are still there: the warning asks, it does not clear anything
    await user.click(screen.getByRole("button", { name: /live monitor/i }));
    await waitFor(() => expect(chipLabels()).toEqual(["Kiln Platform", "Sand Store"]));
  });

  it("says the limit is reached and stops offering uploads", async () => {
    videos = Array.from({ length: 10 }, (_, index) =>
      makeVideo({ id: `v${index}`, name: `clip${index}.mp4`, source: `/data/videos/v${index}.mp4` })
    );
    const user = await renderStopped();
    await openTestVideo(user);

    expect(screen.getByText(/10 video limit is reached/i)).toBeTruthy();
    expect((screen.getByTestId("video-file") as HTMLInputElement).disabled).toBe(true);
  });

  it("truncates long video names with the full name in a tooltip", async () => {
    videos = [
      makeVideo({
        id: "long",
        name: "a-very-long-recording-name-from-the-night-shift-camera.mp4",
        source: "/data/videos/long.mp4",
      }),
    ];
    const user = await renderStopped();
    await openTestVideo(user);

    const name = document.querySelector(".video-name") as HTMLElement;
    expect(name.textContent).toContain("a-very-long-recording-name");
    expect(name.getAttribute("title")).toBe(
      "a-very-long-recording-name-from-the-night-shift-camera.mp4"
    );
  });

  it("keeps Size, Date, Use and Delete on one line each", async () => {
    videos = [makeVideo({ id: "one", name: "line.mp4", source: "/data/videos/one.mp4" })];
    config = { ...CONFIG, source: "/data/videos/one.mp4" };
    const user = await renderStopped();
    await openTestVideo(user);

    // fixed layout: sized columns never push past the card, so no sideways
    // scroll can appear; the name takes the rest and truncates
    const table = screen.getByTestId("video-list");
    expect(table.querySelectorAll("thead th")).toHaveLength(5);
    expect(screen.getByText("In use").closest("td")?.cellIndex).toBe(3);
  });

  it("locks the no-scroll table rules into the stylesheet", async () => {
    const fs = await import("node:fs");
    const css = fs.readFileSync(`${process.cwd()}/src/styles.css`, "utf-8");

    expect(css).toContain("table-layout: fixed");
    expect(css).not.toMatch(/\.video-table-wrap\s*\{[^}]*overflow-x:\s*auto/);
    expect(css).toContain("text-overflow: ellipsis");
    expect(css).toContain("minmax(0, 5fr) minmax(0, 7fr)");
    // narrow screens stack each video instead of scrolling sideways
    expect(css).toContain("@media (max-width: 900px)");
    expect(css).toMatch(/\.video-table thead\s*\{\s*display:\s*none/);
  });
});

describe("live source picker", () => {
  const ZONE_A: Point[] = [
    [100, 100],
    [400, 100],
    [400, 400],
    [100, 400],
  ];
  const ZONE_B: Point[] = [
    [500, 200],
    [900, 200],
    [900, 500],
    [500, 500],
  ];

  function seedClips() {
    videos = [
      makeVideo({ id: "one", name: "line.mp4", source: "/data/videos/one.mp4" }),
      makeVideo({ id: "two", name: "yard.mp4", source: "/data/videos/yard.mp4" }),
    ];
    configBySource["/data/videos/one.mp4"] = { ...CONFIG, zones: [ZONE_A, ZONE_B] };
    configBySource["/data/videos/yard.mp4"] = { ...CONFIG, zones: [] };
  }

  function liveSelect(): HTMLSelectElement {
    return screen.getByTestId("live-source-select") as HTMLSelectElement;
  }

  function liveOptions(): string[] {
    return [...liveSelect().querySelectorAll("option")].map((node) => node.textContent ?? "");
  }

  async function openLive() {
    const user = userEvent.setup();
    render(<App />);
    await waitForUi();
    return user;
  }

  it("lists the webcam and every clip with its saved zone count", async () => {
    seedClips();
    await openLive();

    await waitFor(() => expect(liveOptions()).toContain("line.mp4 · 2 zones"));
    expect(liveOptions()).toContain("Webcam 0 · 0 zones");
    expect(liveOptions()).toContain("yard.mp4 · 0 zones");
  });

  it("selecting on Live updates Settings, and back again", async () => {
    seedClips();
    const user = await openLive();
    await waitFor(() => expect(liveOptions()).toContain("line.mp4 · 2 zones"));

    await user.selectOptions(liveSelect(), "/data/videos/one.mp4");
    await waitFor(() => expect(liveSelect().value).toBe("/data/videos/one.mp4"));

    // Settings shows the same source: one stored setting, two faces
    await user.click(button(/^settings$/i));
    const cameraSelect = screen.getByLabelText("Video source") as HTMLSelectElement;
    expect(cameraSelect.value).toBe("/data/videos/one.mp4");

    await user.selectOptions(cameraSelect, "camera");
    await user.click(button(/^live monitor$/i));
    await waitFor(() => expect(liveSelect().value).toBe("camera"));
  });

  it("asks before switching a running feed, Switch restarts and Keep keeps", async () => {
    seedClips();
    state = makeState({ running: true, source: "0" });
    const user = await openLive();
    await waitFor(() => expect(liveOptions()).toContain("line.mp4 · 2 zones"));
    const starts = postCount("/start");

    await user.selectOptions(liveSelect(), "/data/videos/one.mp4");
    // the question first, the switch only after it
    await waitFor(() => expect(screen.getByTestId("live-source-confirm")).toBeTruthy());
    expect(screen.getByTestId("live-source-confirm").textContent).toMatch(
      /stops the current feed/i
    );
    expect(liveSelect().value).toBe("camera");
    expect(postCount("/start")).toBe(starts);

    await user.click(screen.getByRole("button", { name: /^switch$/i }));
    await waitFor(() => expect(liveSelect().value).toBe("/data/videos/one.mp4"));
    expect(screen.queryByTestId("live-source-confirm")).toBeNull();
    await waitFor(() => expect(postCount("/start")).toBeGreaterThan(starts));
  });

  it("cancel keeps the running feed on its source", async () => {
    seedClips();
    state = makeState({ running: true, source: "0" });
    const user = await openLive();
    await waitFor(() => expect(liveOptions()).toBeTruthy());
    const starts = postCount("/start");

    await user.selectOptions(liveSelect(), "/data/videos/one.mp4");
    await waitFor(() => expect(screen.getByTestId("live-source-confirm")).toBeTruthy());
    await user.click(screen.getByRole("button", { name: /^keep current$/i }));

    expect(screen.queryByTestId("live-source-confirm")).toBeNull();
    expect(liveSelect().value).toBe("camera");
    expect(postCount("/start")).toBe(starts);
  });

  it("selects straight through while stopped, with no question", async () => {
    seedClips();
    const user = await openLive();
    await waitFor(() => expect(liveOptions()).toBeTruthy());

    await user.selectOptions(liveSelect(), "/data/videos/yard.mp4");
    await waitFor(() => expect(liveSelect().value).toBe("/data/videos/yard.mp4"));
    expect(screen.queryByTestId("live-source-confirm")).toBeNull();
  });

  it("hints when the selected clip has no saved zone", async () => {
    seedClips();
    config = { ...CONFIG, source: "/data/videos/yard.mp4" };
    await openLive();
    await waitFor(() => expect(liveOptions()).toBeTruthy());

    expect(screen.getByTestId("live-no-zone-hint").textContent).toMatch(
      /no zone saved for this clip/i
    );
  });

  it("hides the hint once the clip has zones", async () => {
    seedClips();
    config = { ...CONFIG, source: "/data/videos/one.mp4" };
    await openLive();
    await waitFor(() => expect(liveOptions()).toContain("line.mp4 · 2 zones"));

    expect(screen.queryByTestId("live-no-zone-hint")).toBeNull();
  });

  it("uploads from Live Monitor and the clip appears in both places", async () => {
    videos = [];
    const user = await openLive();

    const file = new File(["video-bytes"], "fresh.mp4", { type: "video/mp4" });
    fireEvent.change(screen.getByTestId("live-video-file"), { target: { files: [file] } });
    const bar = (await screen.findByTestId("live-upload-progress")) as HTMLProgressElement;
    expect(bar).toBeTruthy();

    await act(async () => {
      xhrInstances[xhrInstances.length - 1].respond(200, {
        id: "fresh",
        file: "fresh.mp4",
        name: "fresh.mp4",
        size_mb: 3,
        width: 640,
        height: 480,
        fps: 25,
        duration: 2,
        created: "2026-01-01T00:00:00Z",
        source: "/data/videos/fresh.mp4",
      });
    });

    await waitFor(() => expect(liveOptions()).toContain("fresh.mp4 · 0 zones"));
    // and Settings lists it too: the same uploaded clip, not a copy
    await user.click(button(/^settings$/i));
    await waitFor(() => expect(screen.getByText("fresh.mp4")).toBeTruthy());
  });

  it("disables the dropdown in edit mode so a switch cannot drop edits", async () => {
    seedClips();
    state = makeState({ running: true });
    const user = userEvent.setup();
    render(<App />);
    await waitForUi();
    expect(liveSelect().disabled).toBe(false);

    await user.click(button(/^edit zones$/i));
    await waitFor(() => expect(screen.getByTestId("toolbar")).toBeTruthy());
    expect(liveSelect().disabled).toBe(true);
  });
});

describe("scroll containment", () => {
  it("locks the document shell and scrolls only the main column", async () => {
    const fs = await import("node:fs");
    const css = fs.readFileSync(`${process.cwd()}/src/styles.css`, "utf-8");

    // the document itself never scrolls or bounces on a touchpad flick;
    // overflow: hidden is the fallback, overflow: clip is the real lock
    // (hidden still allows programmatic scrolling - scrollIntoView, focus,
    // restored offsets - while clip forbids scrolling entirely)
    for (const selector of ["html", "body", "#root"]) {
      expect(css).toMatch(new RegExp(`${selector}\\s*\\{[^}]*overflow:\\s*hidden`));
      expect(css).toMatch(new RegExp(`${selector}\\s*\\{[^}]*overflow:\\s*clip`));
    }
    expect(css).toMatch(/html\s*\{[^}]*overscroll-behavior:\s*none/);
    // the shell is pinned to the viewport itself, so a restored or
    // programmatic document offset cannot shift the sidebar upward
    expect(css).toMatch(/\.app-shell\s*\{[^}]*position:\s*fixed[^}]*inset:\s*0/);
    expect(css).toMatch(/\.app-shell\s*\{[^}]*overflow:\s*clip/);
    // the shell is exactly the viewport: dynamic viewport units where
    // supported so a mobile URL bar leaves no gap, plain vh elsewhere
    expect(css).toMatch(/\.app-shell\s*\{[^}]*height:\s*100vh[^}]*height:\s*100dvh/);
    // ...and the main column is the only vertical scroll container, with
    // chained overscroll contained (the modal card's own overflow: auto is
    // a separate box, not a second page scrollbar)
    expect(css).toMatch(/\.page\s*\{[^}]*overflow-y:\s*auto[^}]*overscroll-behavior:\s*contain/);
    const verticalScrollers = css.match(/overflow-y:\s*auto/g) ?? [];
    expect(verticalScrollers.length).toBe(1);
  });

  it("resets any document scroll offset and opts out of restored scroll", async () => {
    const { installScrollLock } = await import("./utils/scrollLock");
    const uninstall = installScrollLock();
    try {
      // no restored offset survives a reload
      expect(window.history.scrollRestoration).toBe("manual");
      // a programmatic offset on the document is reset by the guard
      document.documentElement.scrollTop = 57;
      document.documentElement.scrollLeft = 13;
      window.dispatchEvent(new Event("scroll"));
      expect(document.documentElement.scrollTop).toBe(0);
      expect(document.documentElement.scrollLeft).toBe(0);
      const scrolling = document.scrollingElement as HTMLElement | null;
      if (scrolling) {
        expect(scrolling.scrollTop).toBe(0);
        expect(scrolling.scrollLeft).toBe(0);
      }
    } finally {
      uninstall();
    }
  });

  it("renders the top bar and the content inside the single scroll container", async () => {
    render(<App />);
    await waitForUi();
    // sidebar beside the column, never inside it
    expect(document.querySelector(".app-shell > nav.sidebar")).toBeTruthy();
    const page = document.querySelector(".app-shell > .page");
    expect(page).toBeTruthy();
    expect(page?.querySelector(".topbar")).toBeTruthy();
    expect(page?.querySelector("main.main")).toBeTruthy();
  });
});

describe("the three occupancy states", () => {
  it("paints an occupied zone orange and reserves red for the alerting one", async () => {
    config = namedZones();
    state = makeState({
      running: true,
      zone_polygons: [ZONE_A, ZONE_B],
      zone_specs: namedZones().zone_specs,
      inside: 3,
      status: "ALERT",
      worker_alert: true,
      zone_reports: [
        {
          index: 0,
          name: "Kiln Platform",
          inside: 1,
          allowed_people: 1,
          dwell_seconds: 1.0,
          dwell_limit: 3.5,
          alerting: false,
          enabled: true,
        },
        {
          index: 1,
          name: "Sand Store",
          inside: 2,
          allowed_people: 1,
          dwell_seconds: 5.0,
          dwell_limit: 2,
          alerting: true,
          enabled: true,
        },
      ],
    });
    render(<App />);
    await waitForUi();

    await waitFor(() => expect(screen.getByTestId("zone-cards")).toBeTruthy());
    const shape = (index: number) =>
      document.querySelector(`[data-testid='zone-shape-${index}']`);
    expect(shape(0)?.getAttribute("class")).toContain("occupy");
    expect(shape(0)?.getAttribute("class")).not.toContain("alert");
    expect(shape(1)?.getAttribute("class")).toContain("alert");
    // the tags follow their zone
    expect(
      document.querySelector("[data-testid='zone-tag-0']")?.getAttribute("class")
    ).toContain("occupy");
    expect(
      document.querySelector("[data-testid='zone-tag-1']")?.getAttribute("class")
    ).toContain("alert");
  });

  it("shows the orange banner with Xs of Ys while inside within the limit", async () => {
    state = makeState({
      running: true,
      inside: 1,
      status: "SAFE",
      rules: { mode: "TIMER", dwell_seconds: 5 },
      workers: [
        {
          worker_id: 3,
          state: "INSIDE",
          dwell_seconds: 4.1,
          zone_index: 0,
          zone_name: "Kiln Platform",
        },
      ],
    });
    render(<App />);
    await waitForUi();

    const banner = await screen.findByTestId("state-banner");
    expect(banner.className).toContain("occupy");
    expect(banner.textContent).toContain("Worker #3 in Kiln Platform (4.1s of 5.0s)");
    // within the limit there is no alarm to silence
    expect(screen.queryByRole("button", { name: /mute alert sound/i })).toBeNull();
  });

  it("paints the people tile red once a zone is over its allowance", async () => {    config = namedZones();
    state = makeState({
      running: true,
      zone_polygons: [ZONE_A, ZONE_B],
      zone_specs: namedZones().zone_specs,
      inside: 3,
      status: "ALERT",
      zone_reports: [
        {
          index: 0,
          name: "Kiln Platform",
          inside: 3,
          allowed_people: 1,
          dwell_seconds: 1.0,
          dwell_limit: 3.5,
          alerting: true,
          enabled: true,
        },
        {
          index: 1,
          name: "Sand Store",
          inside: 0,
          allowed_people: 6,
          dwell_seconds: 0,
          dwell_limit: 2,
          alerting: false,
          enabled: true,
        },
      ],
    });
    render(<App />);
    await waitForUi();

    await waitFor(() => expect(screen.getByTestId("tile-inside").textContent).toMatch(/^3 \/ 1$/));
    expect(screen.getByTestId("tile-inside").className).toContain("over");
  });

  it("stays orange with one person allowed in, red with two, in over-the-limit mode", async () => {
    config = { ...namedZones(), rules: { mode: "LIMIT", dwell_seconds: 1 } };
    state = makeState({
      running: true,
      zone_polygons: [ZONE_A, ZONE_B],
      zone_specs: namedZones().zone_specs,
      inside: 1,
      status: "SAFE",
      workers: [
        {
          worker_id: 3,
          state: "INSIDE",
          dwell_seconds: 4.1,
          zone_index: 0,
          zone_name: "Kiln Platform",
        },
      ],
      zone_reports: [
        {
          index: 0,
          name: "Kiln Platform",
          inside: 1,
          allowed_people: 1,
          dwell_seconds: 4.1,
          dwell_limit: 3.5,
          alerting: false,
          enabled: true,
        },
        {
          index: 1,
          name: "Sand Store",
          inside: 0,
          allowed_people: 6,
          dwell_seconds: 0,
          dwell_limit: 2,
          alerting: false,
          enabled: true,
        },
      ],
    });
    render(<App />);
    await waitForUi();

    // one person where one is allowed: orange all over, no alarm to silence
    const calm = await screen.findByTestId("state-banner");
    expect(calm.className).toContain("occupy");
    expect(calm.textContent).toContain("Worker #3 in Kiln Platform (4.1s of 3.5s)");
    expect(screen.queryByRole("button", { name: /mute alert sound/i })).toBeNull();
    expect(document.querySelector("[data-testid='zone-shape-0']")?.getAttribute("class")).toContain(
      "occupy"
    );

    // a second person steps in: red banner, red zone, red tile
    cleanup();
    state = makeState({
      running: true,
      zone_polygons: [ZONE_A, ZONE_B],
      zone_specs: namedZones().zone_specs,
      inside: 2,
      status: "ALERT",
      capacity_alert: true,
      message: "2 people inside Kiln Platform, 1 allowed",
      workers: [
        {
          worker_id: 3,
          state: "INSIDE",
          dwell_seconds: 5.0,
          zone_index: 0,
          zone_name: "Kiln Platform",
        },
        {
          worker_id: 4,
          state: "INSIDE",
          dwell_seconds: 0.5,
          zone_index: 0,
          zone_name: "Kiln Platform",
        },
      ],
      zone_reports: [
        {
          index: 0,
          name: "Kiln Platform",
          inside: 2,
          allowed_people: 1,
          dwell_seconds: 5.0,
          dwell_limit: 3.5,
          alerting: true,
          enabled: true,
        },
        {
          index: 1,
          name: "Sand Store",
          inside: 0,
          allowed_people: 6,
          dwell_seconds: 0,
          dwell_limit: 2,
          alerting: false,
          enabled: true,
        },
      ],
    });
    render(<App />);
    await waitForUi();

    const alarm = await screen.findByTestId("state-banner");
    expect(alarm.className).toContain("alert");
    expect(alarm.textContent).toContain("2 people inside Kiln Platform, 1 allowed");
    expect(document.querySelector("[data-testid='zone-shape-0']")?.getAttribute("class")).toContain(
      "alert"
    );
    await waitFor(() => expect(screen.getByTestId("tile-inside").textContent).toMatch(/^2 \/ 1$/));
    expect(screen.getByTestId("tile-inside").className).toContain("over");
  });
});

describe("the history chart", () => {
  it("reads the chart window from config, falling back to 14 days", async () => {
    config = { ...CONFIG, limits: { history_days: 7 } };
    const user = userEvent.setup();
    render(<App />);
    await waitForUi();
    await user.click(button(/^history$/i));
    await waitFor(() => expect(screen.getByTestId("history-chart")).toBeTruthy());
    expect(screen.getByText("Alerts per day (last 7 days)")).toBeTruthy();
    await waitFor(() =>
      expect(
        fetchMock.mock.calls.some(
          ([url]) => String(url).startsWith("/events/daily") && String(url).includes("days=7")
        )
      ).toBe(true)
    );
  });

  it("says there is nothing to plot when the backend has no alerts", async () => {
    const user = userEvent.setup();
    render(<App />);
    await waitForUi();

    // the mocked /events answers with no days array, which is empty, not broken
    await user.click(button(/^history$/i));
    await waitFor(() => expect(screen.getByTestId("history-chart")).toBeTruthy());
    expect(screen.getByText(/alerts per day/i)).toBeTruthy();
    expect(screen.getByTestId("history-empty")).toBeTruthy();
  });

  it("stacks dwell under entry under capacity with a tooltip per day", () => {    render(
      <DailyChart
        days={[
          { date: "2026-10-01", entry: 1, dwell: 2, capacity: 1 },
          { date: "2026-10-02", entry: 0, dwell: 0, capacity: 0 },
        ]}
      />
    );

    const dwell = screen.getByTestId("dwell-bar-2026-10-01");
    const entry = screen.getByTestId("entry-bar-2026-10-01");
    const capacity = screen.getByTestId("capacity-bar-2026-10-01");
    // entry sits on top of dwell, capacity on top of entry
    expect(Number(entry.getAttribute("y"))).toBeLessThan(Number(dwell.getAttribute("y")));
    expect(Number(capacity.getAttribute("y"))).toBeLessThan(Number(entry.getAttribute("y")));
    expect(Number(entry.getAttribute("height"))).toBeGreaterThan(0);
    expect(
      document.querySelector("[data-testid='daily-chart'] title")?.textContent
    ).toContain("2026-10-01: 1 entry, 2 dwell, 1 capacity");
    // date labels on the X axis, zero days included
    expect(screen.getByText("10-01")).toBeTruthy();
    expect(screen.getByText("10-02")).toBeTruthy();
  });

  it("stacks fire over capacity and smoke on top, with a key", () => {
    render(
      <DailyChart
        days={[
          { date: "2026-10-01", entry: 1, dwell: 2, capacity: 1, fire: 2, smoke: 1 },
          { date: "2026-10-02", entry: 0, dwell: 0, capacity: 0, fire: 0, smoke: 0 },
        ]}
      />
    );

    const capacity = screen.getByTestId("capacity-bar-2026-10-01");
    const fire = screen.getByTestId("fire-bar-2026-10-01");
    const smoke = screen.getByTestId("smoke-bar-2026-10-01");
    expect(Number(fire.getAttribute("y"))).toBeLessThan(Number(capacity.getAttribute("y")));
    expect(Number(smoke.getAttribute("y"))).toBeLessThan(Number(fire.getAttribute("y")));
    expect(Number(fire.getAttribute("height"))).toBeGreaterThan(0);
    expect(
      document.querySelector("[data-testid='daily-chart'] title")?.textContent
    ).toContain("2026-10-01: 1 entry, 2 dwell, 1 capacity, 2 fire, 1 smoke");
    expect(screen.getByText("Fire")).toBeTruthy();
    expect(screen.getByText("Smoke")).toBeTruthy();
  });
});

describe("entry alerts", () => {
  const entryEvent: AlertEvent = {
    id: 11,
    worker_id: 3,
    machine_id: "excavator_01",
    event_type: "ENTRY_ALERT",
    entered_at: "2026-10-08T10:24:00Z",
    alert_at: "2026-10-08T10:24:00Z",
    dwell_seconds: 0,
    zone_index: 0,
    zone_name: "Kiln Platform",
    has_snapshot: false,
    acknowledged: false,
    acknowledged_at: null,
  };

  function alertsModel(overrides = {}) {
    return {
      alertTab: "active" as const,
      activeAlerts: [entryEvent],
      activeTotal: 1,
      clearedAlerts: [],
      clearedTotal: 0,
      fireAlerts: [],
      fireActiveTotal: 0,
      fireClearedTotal: 0,
      alertsError: null,
      evidence: null,
      ackingAll: false,
      evidenceHasPrev: false,
      evidenceHasNext: false,
      fireEvidence: null,
      fireEvidenceHasPrev: false,
      fireEvidenceHasNext: false,
      onTab: () => {},
      onAcknowledge: () => {},
      onAcknowledgeAll: () => {},
      onAcknowledgeFire: () => {},
      onFireEvidence: () => {},
      onFireEvidenceStep: () => {},
      onEvidence: () => {},
      onEvidenceStep: () => {},
      ...overrides,
    };
  }

  it("words an entry alert as entered danger zone with a red entry badge", () => {
    render(<AlertsPage model={alertsModel()} />);

    expect(screen.getByText("Entered danger zone")).toBeTruthy();
    const badge = screen.getByText("ENTRY ALERT");
    expect(badge.className).toContain("badge-entry");
    expect(badge.className).not.toContain("badge-capacity");
  });

  it("words a capacity alert as zone plus crowd, never Worker #-1", () => {
    const capacityEvent: AlertEvent = {
      id: 12,
      worker_id: -1,
      machine_id: "excavator_01",
      event_type: "CAPACITY_ALERT",
      entered_at: "2026-10-08T10:24:00Z",
      alert_at: "2026-10-08T10:24:00Z",
      dwell_seconds: 0,
      zone_index: 0,
      zone_name: "Zone 1",
      inside: 2,
      has_snapshot: false,
      acknowledged: false,
      acknowledged_at: null,
    };
    render(
      <AlertsPage
        model={alertsModel({ activeAlerts: [capacityEvent], evidence: capacityEvent })}
      />
    );

    expect(screen.getAllByText(/Zone 1 · 2 people/)).toHaveLength(2);
    expect(screen.queryByText(/Worker #-1/)).toBeNull();
    // the evidence title names the crowd too
    expect(screen.getByText(/Too many people - Zone 1 · 2 people/)).toBeTruthy();
  });

  it("names the zone alone when an old capacity row carries no count", () => {
    const legacy: AlertEvent = {
      id: 13,
      worker_id: -1,
      machine_id: "excavator_01",
      event_type: "CAPACITY_ALERT",
      entered_at: "2026-10-08T10:24:00Z",
      alert_at: "2026-10-08T10:24:00Z",
      dwell_seconds: 0,
      zone_index: 1,
      zone_name: "",
      has_snapshot: false,
      acknowledged: false,
      acknowledged_at: null,
    };
    render(<AlertsPage model={alertsModel({ activeAlerts: [legacy] })} />);

    // zone cell and worker cell both name it (capacity rows are about crowds)
    expect(screen.getAllByText(/Zone 2/)).toHaveLength(2);
    expect(screen.queryByText(/Worker #-1/)).toBeNull();
  });

  it("asks the backend for all three alert kinds", async () => {
    render(<App />);
    await waitForUi();

    await waitFor(() =>
      expect(
        fetchMock.mock.calls.some(
          (call) =>
            String(call[0]).includes("ENTRY_ALERT") &&
            String(call[0]).includes("DWELL_ALERT") &&
            String(call[0]).includes("CAPACITY_ALERT")
        )
      ).toBe(true)
    );
  });

  it("sends recent alerts to the alerts page through view all", () => {
    let seen: string | null = null;
    const model = {
      machineName: "Kiln 01",
      screenMode: "live",
      running: true,
      stoppedSourceLabel: "Webcam 0",
      busy: false,
      startZoneBlock: "",
      drawing: false,
      tool: null,
      frameKnown: false,
      frameWidth: 0,
      frameHeight: 0,
      unit: 1,
      selectedShape: null,
      pictureUrl: "/stream",
      imageRef: { current: null },
      overlayVisible: false,
      shownZones: [],
      rubberBand: null,
      polygonPreview: null,
      tags: [],
      tagFontSize: 16,
      tagPad: 8,
      handles: null,
      handleZone: null,
      midpoints: [],
      cursorHint: null,
      alert: false,
      bannerWorkers: [],
      bannerInside: [],
      bannerMessage: "",
      muted: false,
      tiles: {
        peopleLabel: "People in zone",
        people: 0,
        peopleLimit: 0,
        peopleTotal: false,
        peopleTone: "idle",
        dwellLabel: "Dwell time",
        dwell: 0,
        dwellLimit: 0,
        dwellTotal: false,
        dwellTone: "idle",
      },
      anchorHint: null,
      legendVisible: false,
      zoneCards: [],
      renaming: null,
      renameDraft: "",
      confirmingCancel: false,
      unsavedGeometry: false,
      zoneValid: false,
      coordRows: null,
      coordTitle: "",
      coordZone: null,
      canAddPoint: false,
      recentAlerts: [entryEvent],
      perfFps: "-",
      historyEmpty: true,
      draftEmpty: true,
      clearDisabled: true,
      deleteDisabled: true,
      onViewAllAlerts: () => {
        seen = "/alerts";
      },
    } as unknown as LiveMonitorModel;
    render(<LiveMonitorPage model={model} />);

    expect(screen.getByText("ENTRY ALERT")).toBeTruthy();
    fireEvent.click(screen.getByText("View all"));
    expect(seen).toBe("/alerts");
  });

  it("words a capacity alert in recent alerts as zone plus crowd", () => {
    const capacityEvent: AlertEvent = {
      id: 12,
      worker_id: -1,
      machine_id: "excavator_01",
      event_type: "CAPACITY_ALERT",
      entered_at: new Date(Date.now() - 5000).toISOString(),
      alert_at: new Date(Date.now() - 5000).toISOString(),
      dwell_seconds: 0,
      zone_index: 0,
      zone_name: "Zone 1",
      inside: 2,
      has_snapshot: false,
      acknowledged: false,
      acknowledged_at: null,
    };
    const model = {
      machineName: "Kiln 01",
      screenMode: "live",
      running: true,
      stoppedSourceLabel: "Webcam 0",
      busy: false,
      startZoneBlock: "",
      drawing: false,
      tool: null,
      frameKnown: false,
      frameWidth: 0,
      frameHeight: 0,
      unit: 1,
      selectedShape: null,
      pictureUrl: "/stream",
      imageRef: { current: null },
      overlayVisible: false,
      shownZones: [],
      rubberBand: null,
      polygonPreview: null,
      tags: [],
      tagFontSize: 16,
      tagPad: 8,
      handles: null,
      handleZone: null,
      midpoints: [],
      cursorHint: null,
      alert: false,
      bannerWorkers: [],
      bannerInside: [],
      bannerMessage: "",
      muted: false,
      tiles: {
        peopleLabel: "People in zone",
        people: 0,
        peopleLimit: 0,
        peopleTotal: false,
        peopleTone: "idle",
        dwellLabel: "Dwell time",
        dwell: 0,
        dwellLimit: 0,
        dwellTotal: false,
        dwellTone: "idle",
      },
      anchorHint: null,
      legendVisible: false,
      zoneCards: [],
      renaming: null,
      renameDraft: "",
      confirmingCancel: false,
      unsavedGeometry: false,
      zoneValid: false,
      coordRows: null,
      coordTitle: "",
      coordZone: null,
      canAddPoint: false,
      recentAlerts: [capacityEvent],
      perfFps: "-",
      historyEmpty: true,
      draftEmpty: true,
      clearDisabled: true,
      deleteDisabled: true,
      canEdit: true,
      stageVisible: true,
      onViewAllAlerts: () => {},
    } as unknown as LiveMonitorModel;
    render(<LiveMonitorPage model={model} />);

    expect(screen.getByText(/Zone 1 · 2 people/)).toBeTruthy();
    expect(screen.queryByText(/Worker #-1/)).toBeNull();
  });
});

describe("recent alerts card with fire", () => {
  const zoneEvent: AlertEvent = {
    id: 21,
    worker_id: 3,
    machine_id: "excavator_01",
    event_type: "DWELL_ALERT",
    entered_at: "2026-10-08T10:24:00Z",
    alert_at: "2026-10-08T10:24:05Z",
    dwell_seconds: 12.3,
    zone_index: 0,
    zone_name: "Kiln Platform",
    has_snapshot: false,
    acknowledged: false,
    acknowledged_at: null,
  };
  const cardFire: FireAlert = {
    id: 7,
    alert_type: "fire",
    severity: "ORANGE",
    confidence: 0.85,
    bbox_xyxy: [1, 2, 30, 40],
    captured_at: 3.0,
    reported_at: "2026-10-08T10:25:00Z",
    has_snapshot: true,
    acknowledged: false,
    acknowledged_at: null,
    false_alarm: false,
    false_alarm_at: null,
    source_label: "Excavator 01",
  };

  function cardRowIds(): (string | null)[] {
    return within(screen.getByTestId("recent-alerts"))
      .getAllByTestId(/recent-alert-(zone|fire)-/)
      .map((row) => row.getAttribute("data-testid"));
  }

  it("merges zone and fire newest-first with the fire chip, time and thumb", async () => {
    alertRows = [zoneEvent];
    fireRows = [cardFire];
    render(<App />);
    await waitForUi();

    // the fire row is newer, so it leads
    expect(cardRowIds()).toEqual(["recent-alert-fire-7", "recent-alert-zone-21"]);
    expect(screen.queryByTestId("recent-alerts-empty")).toBeNull();
    // chip with confidence, severity, source and clock
    expect(screen.getByText("FIRE 0.85")).toBeTruthy();
    expect(screen.getByText("ORANGE")).toBeTruthy();
    const thumb = screen.getByAltText("fire alert 7") as HTMLImageElement;
    expect(thumb.getAttribute("src")).toBe("/fire-alerts/7/snapshot");
    // the zone row keeps its old shape
    expect(screen.getByText("Stayed too long")).toBeTruthy();
  });

  it("caps the merged list at five", async () => {
    // newest-first, the way /events serves them
    alertRows = [0, 1, 2, 3, 4, 5]
      .map((index) => ({
        ...zoneEvent,
        id: 30 + index,
        entered_at: `2026-10-08T10:2${index}:00Z`,
        alert_at: `2026-10-08T10:2${index}:05Z`,
      }))
      .reverse();
    fireRows = [{ ...cardFire, id: 40, reported_at: "2026-10-08T10:30:00Z" }];
    render(<App />);
    await waitForUi();

    expect(cardRowIds()).toEqual([
      "recent-alert-fire-40",
      "recent-alert-zone-35",
      "recent-alert-zone-34",
      "recent-alert-zone-33",
      "recent-alert-zone-32",
    ]);
  });

  it("acks a fire row from the card and the Alerts page follows", async () => {
    alertRows = [];
    fireRows = [cardFire];
    const user = userEvent.setup();
    render(<App />);
    await waitForUi();
    expect(screen.getByTestId("recent-alert-fire-7")).toBeTruthy();

    await user.click(
      within(screen.getByTestId("recent-alert-fire-7")).getByText("Acknowledge")
    );
    await waitFor(() => expect(screen.queryByTestId("recent-alert-fire-7")).toBeNull());
    // the card is empty only because nothing is left anywhere
    expect(screen.getByTestId("recent-alerts-empty")).toBeTruthy();

    // the Alerts page reads the same answer: acknowledged tab holds it
    await user.click(screen.getByRole("button", { name: /^alerts( \d+)?$/i }));
    await user.click(screen.getByRole("tab", { name: /acknowledged/i }));
    await waitFor(() =>
      expect(screen.getByTestId("fire-subhead").textContent).toContain("Fire & smoke (1)")
    );
  });

  it("opens the fire Evidence modal from the card row", async () => {
    alertRows = [];
    fireRows = [cardFire];
    const user = userEvent.setup();
    render(<App />);
    await waitForUi();

    await user.click(screen.getByTestId("recent-alert-fire-7"));
    const modal = screen.getByRole("dialog", { name: "Alert evidence" });
    expect(within(modal).getByText("FIRE · ORANGE")).toBeTruthy();
    expect(within(modal).getByTestId("modal-details").textContent).toContain("0.85");
  });

  it("shows the empty state only when both lists are empty", async () => {    alertRows = [zoneEvent];
    fireRows = [];
    render(<App />);
    await waitForUi();
    expect(screen.queryByTestId("recent-alerts-empty")).toBeNull();

    cleanup();
    alertRows = [];
    fireRows = [cardFire];
    render(<App />);
    await waitForUi();
    expect(screen.queryByTestId("recent-alerts-empty")).toBeNull();

    cleanup();
    alertRows = [];
    fireRows = [];
    render(<App />);
    await waitForUi();
    expect(screen.getByTestId("recent-alerts-empty")).toBeTruthy();
  });

  it("lowers the badge when a fire alert is acknowledged", async () => {
    alertRows = [zoneEvent];
    fireRows = [cardFire, { ...cardFire, id: 8 }];
    const user = userEvent.setup();
    render(<App />);
    await waitForUi();
    expect(screen.getByTestId("alerts-badge").textContent).toBe("3");

    await user.click(screen.getByRole("button", { name: /^alerts( \d+)?$/i }));
    await waitFor(() => expect(screen.getByTestId("fire-alert-list")).toBeTruthy());
    const ackButtons = within(screen.getByTestId("fire-alert-list")).getAllByText("Acknowledge");
    fireEvent.click(ackButtons[0]);
    await waitFor(() => expect(screen.getByTestId("alerts-badge").textContent).toBe("2"));
    // the zone Active count is untouched by the fire ack
    expect(screen.getByRole("tab", { name: "Active (1)" })).toBeTruthy();
  });

  it("shows the badge for fire-only unacked and hides it at zero", async () => {
    alertRows = [];
    fireRows = [cardFire];
    const user = userEvent.setup();
    render(<App />);
    await waitForUi();
    expect(screen.getByTestId("alerts-badge").textContent).toBe("1");

    await user.click(
      within(screen.getByTestId("recent-alert-fire-7")).getByText("Acknowledge")
    );
    await waitFor(() => expect(screen.queryByTestId("alerts-badge")).toBeNull());
  });
});

describe("alerts page rebuild", () => {
  const dwellEvent: AlertEvent = {
    id: 11,
    worker_id: 3,
    machine_id: "excavator_01",
    event_type: "DWELL_ALERT",
    entered_at: "2026-10-08T10:24:00Z",
    alert_at: "2026-10-08T10:24:05Z",
    dwell_seconds: 12.3,
    zone_index: 0,
    zone_name: "Kiln Platform",
    has_snapshot: true,
    acknowledged: false,
    acknowledged_at: null,
  };
  const capacityEvent: AlertEvent = {
    id: 12,
    worker_id: -1,
    machine_id: "excavator_01",
    event_type: "CAPACITY_ALERT",
    entered_at: "2026-10-08T10:24:00Z",
    alert_at: "2026-10-08T10:24:00Z",
    dwell_seconds: 0,
    zone_index: 0,
    zone_name: "Zone 1",
    inside: 2,
    has_snapshot: false,
    acknowledged: false,
    acknowledged_at: null,
  };
  const ackedEvent: AlertEvent = {
    ...dwellEvent,
    id: 13,
    acknowledged: true,
    acknowledged_at: "2026-10-08T10:30:05Z",
  };

  function rebuildModel(overrides = {}) {
    return {
      alertTab: "active" as const,
      activeAlerts: [dwellEvent, capacityEvent],
      activeTotal: 2,
      clearedAlerts: [ackedEvent],
      clearedTotal: 1,
      alertsError: null,
      evidence: null,
      ackingAll: false,
      evidenceHasPrev: false,
      evidenceHasNext: false,
      onTab: () => {},
      onAcknowledge: () => {},
      onAcknowledgeAll: () => {},
      onEvidence: () => {},
      onEvidenceStep: () => {},
      ...overrides,
    };
  }

  it("shows both tabs with live counts, Active first", () => {
    render(<AlertsPage model={rebuildModel()} />);

    const tabs = screen.getByRole("tablist", { name: /alert filter/i });
    expect(within(tabs).getByRole("tab", { name: "Active (2)" })).toBeTruthy();
    expect(within(tabs).getByRole("tab", { name: "Acknowledged (1)" })).toBeTruthy();
    expect(
      within(tabs).getByRole("tab", { name: "Active (2)" }).getAttribute("aria-selected")
    ).toBe("true");
  });

  it("lays out the active card with badge, worker, entered time and red dwell", () => {
    const seen: number[] = [];
    const shown: (AlertEvent | null)[] = [];
    render(
      <AlertsPage
        model={rebuildModel({
          onAcknowledge: (id: number) => seen.push(id),
          onEvidence: (event: AlertEvent | null) => shown.push(event),
        })}
      />
    );

    const badge = screen.getByText("DWELL ALERT");
    expect(badge.className).toContain("badge-dwell");
    expect(screen.getByText("Stayed too long")).toBeTruthy();
    expect(screen.getByText("Worker #3")).toBeTruthy();
    // one entered time per row in the Time column
    expect(screen.getAllByText(/Entered \d\d:\d\d:\d\d/)).toHaveLength(2);

    const dwell = screen.getByTestId("alert-duration-11");
    expect(dwell.textContent).toBe("12.3s");
    // a live row is not greyed out: the big number reads as an alarm
    expect(dwell.closest("tr")?.className).not.toContain("acked");

    fireEvent.click(screen.getAllByText("Acknowledge")[0]);
    expect(seen).toEqual([11]);
    fireEvent.click(screen.getAllByText("View Evidence")[0]);
    expect(shown).toEqual([dwellEvent]);
  });

  it("omits the dwell number on capacity cards but keeps both buttons", () => {
    render(<AlertsPage model={rebuildModel()} />);

    expect(screen.queryByTestId("alert-duration-12")).toBeNull();
    expect(screen.getByText(/Zone 1 · 2 people/)).toBeTruthy();
    // no snapshot, yet View Evidence is still offered
    expect(screen.getAllByText("View Evidence")).toHaveLength(2);
  });

  it("greys the acknowledged card with the ack time and a disabled check", () => {
    const shown: (AlertEvent | null)[] = [];
    render(
      <AlertsPage
        model={rebuildModel({
          alertTab: "acknowledged" as const,
          onEvidence: (event: AlertEvent | null) => shown.push(event),
        })}
      />
    );

    const row = screen.getByText("Stayed too long").closest("tr") as HTMLElement;
    expect(row.className).toContain("acked");
    expect(screen.getByTestId("acked-line-13").textContent).toMatch(/Acknowledged \d\d:\d\d:\d\d/);
    const done = screen.getByText("✓ Acknowledged") as HTMLButtonElement;
    expect(done.disabled).toBe(true);
    fireEvent.click(screen.getByText("View Evidence"));
    expect(shown).toEqual([ackedEvent]);
  });

  it("shows the empty states with a green check on the quiet board", () => {
    render(<AlertsPage model={rebuildModel({ activeAlerts: [] })} />);
    const empty = screen.getByTestId("alerts-empty");
    expect(empty.textContent).toContain("No active alerts");
    expect(empty.querySelector(".alerts-empty-mark")?.textContent).toContain("✓");

    cleanup();
    render(<AlertsPage model={rebuildModel({ alertTab: "acknowledged" as const, clearedAlerts: [], clearedTotal: 0 })} />);
    expect(screen.getByTestId("alerts-empty").textContent).toContain("Nothing acknowledged yet");
  });

  it("asks before acknowledging everything, and cancels cleanly", () => {
    let calls = 0;
    render(<AlertsPage model={rebuildModel({ onAcknowledgeAll: () => (calls += 1) })} />);

    fireEvent.click(screen.getByText("Acknowledge all"));
    expect(
      screen.getByRole("alertdialog", { name: /confirm acknowledge all/i })
    ).toBeTruthy();
    fireEvent.click(screen.getByText("Cancel"));
    expect(calls).toBe(0);
    expect(screen.queryByRole("alertdialog", { name: /confirm acknowledge all/i })).toBeNull();

    fireEvent.click(screen.getByText("Acknowledge all"));
    fireEvent.click(screen.getByText("Confirm"));
    expect(calls).toBe(1);
  });

  it("hides acknowledge-all when there is nothing active", () => {
    render(<AlertsPage model={rebuildModel({ activeAlerts: [], activeTotal: 0 })} />);
    expect(screen.queryByText("Acknowledge all")).toBeNull();
  });

  it("confirms acknowledge-all with the formatted real total", () => {
    render(<AlertsPage model={rebuildModel({ activeTotal: 1051 })} />);

    fireEvent.click(screen.getByText("Acknowledge all"));
    expect(screen.getByText("Acknowledge 1,051 alerts?")).toBeTruthy();
  });

  it("titles the evidence with type, worker, zone and clock, plus the dwell", () => {
    render(
      <AlertsPage
        model={rebuildModel({ evidence: dwellEvent, evidenceHasNext: true })}
      />
    );

    expect(
      screen.getByText(/Stayed too long - Worker #3 - Kiln Platform - \d\d:\d\d:\d\d/)
    ).toBeTruthy();
    expect(screen.getByText(/Dwell 12\.3s/)).toBeTruthy();
    const next = screen.getByRole("button", { name: /next alert/i }) as HTMLButtonElement;
    const prev = screen.getByRole("button", { name: /previous alert/i }) as HTMLButtonElement;
    expect(next.disabled).toBe(false);
    expect(prev.disabled).toBe(true);
  });

  it("steps through the open tab from the modal and closes it", () => {
    const steps: number[] = [];
    const closed: (AlertEvent | null)[] = [];
    render(
      <AlertsPage
        model={rebuildModel({
          evidence: dwellEvent,
          evidenceHasPrev: true,
          evidenceHasNext: true,
          onEvidenceStep: (direction: 1 | -1) => steps.push(direction),
          onEvidence: (event: AlertEvent | null) => closed.push(event),
        })}
      />
    );

    fireEvent.click(screen.getByRole("button", { name: /next alert/i }));
    fireEvent.click(screen.getByRole("button", { name: /previous alert/i }));
    expect(steps).toEqual([1, -1]);
    fireEvent.click(screen.getByRole("button", { name: /close evidence/i }));
    expect(closed).toEqual([null]);
  });

  it("shows a grey block while the thumbnail loads and an icon when it fails", () => {
    render(<AlertsPage model={rebuildModel()} />);

    // jsdom never finishes the download, so the placeholder stays
    expect(document.querySelector(".alert-thumb.loading")).toBeTruthy();
    const img = screen.getByAltText("Alert 11");
    fireEvent.error(img);
    expect(screen.getByRole("img", { name: /no snapshot/i })).toBeTruthy();
    // rows without a picture never had a thumbnail at all
    expect(document.querySelectorAll(".alert-thumb.empty")).toHaveLength(1);
  });

  it("moves an acknowledged card over, updates the counts and the sidebar badge", async () => {
    const user = userEvent.setup();
    alertRows = [
      { ...dwellEvent, acknowledged: false, acknowledged_at: null },
      { ...capacityEvent, acknowledged: false, acknowledged_at: null },
    ];
    render(<App />);
    await waitForUi();
    await user.click(screen.getByRole("button", { name: /^alerts( \d+)?$/i }));
    await waitFor(() => expect(screen.getByTestId("alert-list")).toBeTruthy());

    expect(screen.getByRole("tab", { name: "Active (2)" })).toBeTruthy();
    expect(screen.getByTestId("alerts-badge").textContent).toBe("2");

    fireEvent.click(screen.getAllByText("Acknowledge")[0]);
    await waitFor(() =>
      expect(screen.getByRole("tab", { name: "Active (1)" })).toBeTruthy()
    );
    expect(screen.getByRole("tab", { name: "Acknowledged (1)" })).toBeTruthy();
    expect(screen.getByTestId("alerts-badge").textContent).toBe("1");

    await user.click(screen.getByRole("tab", { name: "Acknowledged (1)" }));
    expect(screen.getByTestId("acked-line-11").textContent).toMatch(/Acknowledged/);
    // and the acknowledged tab no longer offers per-row dwell for capacity
    expect(screen.queryByTestId("alert-duration-12")).toBeNull();
  });

  it("acknowledges everything after confirming, clearing the badge", async () => {
    const user = userEvent.setup();
    alertRows = [
      { ...dwellEvent, acknowledged: false, acknowledged_at: null },
      { ...capacityEvent, acknowledged: false, acknowledged_at: null },
    ];
    render(<App />);
    await waitForUi();
    await user.click(screen.getByRole("button", { name: /^alerts( \d+)?$/i }));
    await waitFor(() => expect(screen.getByTestId("alert-list")).toBeTruthy());

    await user.click(screen.getByText("Acknowledge all"));
    await user.click(screen.getByText("Confirm"));
    await waitFor(() =>
      expect(screen.getByTestId("alerts-empty").textContent).toContain("No active alerts")
    );
    expect(screen.getByRole("tab", { name: "Acknowledged (2)" })).toBeTruthy();
    expect(screen.queryByTestId("alerts-badge")).toBeNull();
    // one bulk call, not one POST per row
    const posts = fetchMock.mock.calls
      .map((call) => [String(call[0]), (call[1] as RequestInit | undefined)?.method] as const)
      .filter(([, method]) => method === "POST")
      .map(([url]) => url);
    expect(posts.filter((url) => url === "/events/ack-all")).toHaveLength(1);
    expect(posts.filter((url) => /\/events\/\d+\/ack$/.test(url))).toHaveLength(0);
  });

  it("badges the true unacknowledged total, capped at 99+, and matches the tab", async () => {
    const user = userEvent.setup();
    alertRows = [
      { ...dwellEvent, acknowledged: false, acknowledged_at: null },
      { ...capacityEvent, acknowledged: false, acknowledged_at: null },
    ];
    eventsTotalOverride = 150;
    render(<App />);
    await waitForUi();

    // two rows loaded, but the badge and the tab read the backend total
    await waitFor(() => expect(screen.getByTestId("alerts-badge").textContent).toBe("99+"));
    await user.click(screen.getByRole("button", { name: /^alerts/i }));
    await waitFor(() => expect(screen.getByTestId("alert-list")).toBeTruthy());
    expect(screen.getByRole("tab", { name: "Active (150)" })).toBeTruthy();

    // acknowledging one ticks the real total down without a reload
    fireEvent.click(screen.getAllByText("Acknowledge")[0]);
    await waitFor(() =>
      expect(screen.getByRole("tab", { name: "Active (149)" })).toBeTruthy()
    );
    expect(screen.getByTestId("alerts-badge").textContent).toBe("99+");
  });

  it("steps the evidence modal through the open tab", async () => {
    const user = userEvent.setup();
    alertRows = [
      { ...dwellEvent, acknowledged: false, acknowledged_at: null },
      { ...capacityEvent, acknowledged: false, acknowledged_at: null },
    ];
    render(<App />);
    await waitForUi();
    await user.click(screen.getByRole("button", { name: /^alerts( \d+)?$/i }));
    await waitFor(() => expect(screen.getByTestId("alert-list")).toBeTruthy());

    fireEvent.click(screen.getAllByText("View Evidence")[0]);
    expect(
      screen.getByText(/Stayed too long - Worker #3 - Kiln Platform/)
    ).toBeTruthy();

    await user.click(screen.getByRole("button", { name: /next alert/i }));
    expect(screen.getByText(/Too many people - Zone 1 · 2 people/)).toBeTruthy();
    await user.click(screen.getByRole("button", { name: /previous alert/i }));
    expect(
      screen.getByText(/Stayed too long - Worker #3 - Kiln Platform/)
    ).toBeTruthy();

    await user.keyboard("{Escape}");
    await waitFor(() =>
      expect(screen.queryByRole("dialog", { name: /alert evidence/i })).toBeNull()
    );
  });
});

describe("history page", () => {
  function historyRow(overrides: Partial<AlertEvent> & { id: number }): AlertEvent {
    return {
      worker_id: 1,
      machine_id: "excavator_01",
      event_type: "ZONE_ENTRY",
      entered_at: "2026-10-08T00:00:00Z",
      alert_at: null,
      dwell_seconds: 0,
      zone_index: 0,
      zone_name: "Zone 1",
      has_snapshot: false,
      acknowledged: false,
      acknowledged_at: null,
      ...overrides,
    };
  }

  function seedHistory() {
    historyFilters = {
      workers: [1, 2],
      zones: [
        { index: 0, name: "Zone 1" },
        { index: 1, name: "Kiln Platform" },
      ],
    };
    historyRows = [
      historyRow({
        id: 101,
        event_type: "DWELL_ALERT",
        alert_at: "2026-10-08T00:00:06Z",
        dwell_seconds: 6,
        has_snapshot: true,
      }),
      historyRow({ id: 102, event_type: "CAPACITY_ALERT", worker_id: -1, inside: 2 }),
      historyRow({
        id: 103,
        worker_id: 2,
        zone_index: 1,
        zone_name: "Kiln Platform",
        entered_at: "2026-10-07T12:00:00Z",
      }),
      historyRow({
        id: 104,
        event_type: "ZONE_EXIT",
        worker_id: 2,
        zone_index: 1,
        zone_name: "Kiln Platform",
        entered_at: "2026-10-07T12:01:00Z",
      }),
      historyRow({ id: 105, event_type: "ENTRY_ALERT", alert_at: "2026-10-08T00:02:00Z" }),
      historyRow({
        id: 106,
        event_type: "DWELL_ALERT",
        entered_at: "2026-10-06T12:00:00Z",
        alert_at: "2026-10-06T12:00:09Z",
        dwell_seconds: 9,
      }),
      historyRow({ id: 107, entered_at: "2026-10-06T12:05:00Z" }),
    ];
    historyFireRows = [
      {
        id: 201,
        alert_type: "fire",
        severity: "ORANGE",
        confidence: 0.76,
        bbox_xyxy: [1, 2, 30, 40],
        captured_at: 1.0,
        reported_at: "2026-10-08T00:01:00Z",
        has_snapshot: true,
        acknowledged: false,
        acknowledged_at: null,
        false_alarm: false,
        false_alarm_at: null,
        source_label: "Kiln Cam",
      },
    ];
  }

  async function openHistory() {
    const user = userEvent.setup();
    render(<App />);
    await waitForUi();
    await user.click(screen.getByRole("button", { name: /^history$/i }));
    await waitFor(() => expect(screen.getByTestId("history-page")).toBeTruthy());
    return user;
  }

  function eventsCalls(): string[] {
    return fetchMock.mock.calls
      .map((call) => String(call[0]))
      .filter((url) => url.startsWith("/history?") && !url.startsWith("/history/export"));
  }

  it("shows the filters with real options, the table and the chart note", async () => {
    seedHistory();
    await openHistory();

    expect(screen.getByRole("option", { name: "Worker #1" })).toBeTruthy();
    expect(screen.getByRole("option", { name: "Kiln Platform" })).toBeTruthy();
    expect(
      screen.getByText(/the chart always shows the last 14 days/i)
    ).toBeTruthy();
    const headers = [...screen.getByTestId("history-table").querySelectorAll("thead th")].map(
      (node) => (node.textContent ?? "").trim()
    );
    expect(headers).toEqual(["Time", "Worker", "Zone", "Source", "Event", "Dwell", "Status", "Evidence"]);
    // alerts only by default: the two ZONE_ENTRY rows and the ZONE_EXIT row
    // stay out until asked for, so the history is not flooded by visits.
    // Fire and smoke are alerts too, so the seeded fire row is in.
    expect(screen.getByTestId("history-count").textContent).toBe("Showing 1 to 5 of 5 rows");
    expect(screen.getByTestId("export-csv").textContent).toBe("Export CSV (5 rows)");
  });

  it("shows the fire row with its source, badge and real status", async () => {
    seedHistory();
    await openHistory();

    expect(screen.getByText("Kiln Cam")).toBeTruthy();
    const badge = screen.getByText("FIRE");
    expect(badge.className).toContain("badge-fire");
    expect(screen.getByText("ORANGE · 0.76")).toBeTruthy();
    expect(screen.getByTestId("history-status-fire-201").textContent).toBe("Active");
    expect(screen.getByTestId("history-status-zone-101").textContent).toBe("Active");
    expect(screen.getByTestId("history-time-fire-201").textContent).toBe("2026-10-08 05:31:00");
  });

  it("shows Unknown source for fire rows stored before labels existed", async () => {
    seedHistory();
    historyFireRows = [
      {
        id: 202,
        alert_type: "smoke",
        reported_at: "2026-10-08T00:03:00Z",
        has_snapshot: false,
        acknowledged: false,
      },
    ];
    await openHistory();
    expect(screen.getByText("Unknown source")).toBeTruthy();
  });

  it("filters by status through the ack and false-alarm flags", async () => {
    seedHistory();
    const user = await openHistory();

    await user.selectOptions(screen.getByLabelText(/^status$/i), "acked");
    await user.click(screen.getByText("Apply"));
    await waitFor(() =>
      expect(screen.getByTestId("history-table-empty")).toBeTruthy()
    );
    let sent = eventsCalls()[eventsCalls().length - 1];
    expect(sent).toContain("acknowledged=true");

    await user.selectOptions(screen.getByLabelText(/^status$/i), "false");
    await user.click(screen.getByText("Apply"));
    await waitFor(() =>
      expect(screen.getByTestId("history-table-empty")).toBeTruthy()
    );
    sent = eventsCalls()[eventsCalls().length - 1];
    expect(sent).toContain("false_alarm=true");

    await user.click(screen.getByText("Clear"));
    await waitFor(() =>
      expect(screen.getByTestId("history-count").textContent).toBe("Showing 1 to 5 of 5 rows")
    );
  });

  it("opens fire evidence with the fire snapshot route", async () => {
    seedHistory();
    await openHistory();
    await waitFor(() => expect(screen.getByTestId("history-table")).toBeTruthy());

    fireEvent.click(screen.getByRole("button", { name: /view evidence for fire alert 201/i }));
    expect(screen.getByText("FIRE · ORANGE")).toBeTruthy();
    const image = document.querySelector("[role='dialog'] img") as HTMLImageElement | null;
    expect(image?.getAttribute("src")).toBe("/fire-alerts/201/snapshot");
    fireEvent.click(screen.getByRole("button", { name: /close evidence/i }));
    await waitFor(() =>
      expect(screen.queryByRole("dialog", { name: /alert evidence/i })).toBeNull()
    );
  });

  it("opts zone entry and exit rows back in through the type filter", async () => {
    seedHistory();
    const user = await openHistory();

    await user.selectOptions(screen.getByLabelText(/event type/i), "ZONE_ENTRY");
    await user.click(screen.getByText("Apply"));

    await waitFor(() =>
      expect(screen.getByTestId("history-count").textContent).toBe("Showing 1 to 2 of 2 rows")
    );
    const sent = eventsCalls()[eventsCalls().length - 1];
    expect(sent).toContain("event_type=ZONE_ENTRY");

    await user.click(screen.getByText("Clear"));
    await waitFor(() =>
      expect(screen.getByTestId("history-count").textContent).toBe("Showing 1 to 5 of 5 rows")
    );
  });

  it("applies the filters to the table and clears them again", async () => {
    seedHistory();
    const user = await openHistory();

    await user.selectOptions(screen.getByLabelText(/event type/i), "DWELL_ALERT");
    await user.selectOptions(screen.getByLabelText(/^worker$/i), "1");
    fireEvent.change(screen.getByLabelText(/date from/i), { target: { value: "2026-10-08" } });
    await user.click(screen.getByText("Apply"));

    await waitFor(() =>
      expect(screen.getByTestId("history-count").textContent).toBe("Showing 1 to 2 of 2 rows")
    );
    const sent = eventsCalls()[eventsCalls().length - 1];
    expect(sent).toContain("event_type=DWELL_ALERT");
    expect(sent).toContain("worker_id=1");
    expect(sent).toContain("date_from=2026-10-08");

    await user.click(screen.getByText("Clear"));
    await waitFor(() =>
      expect(screen.getByTestId("history-count").textContent).toBe("Showing 1 to 5 of 5 rows")
    );
  });

  it("paginates five rows at a time with numbers", async () => {
    seedHistory();
    const user = await openHistory();
    await waitFor(() => expect(screen.getByTestId("history-table")).toBeTruthy());

    // page across the whole log, entry/exit rows included
    await user.selectOptions(screen.getByLabelText(/event type/i), "");
    await user.click(screen.getByText("Apply"));
    await waitFor(() =>
      expect(screen.getByTestId("history-count").textContent).toBe("Showing 1 to 5 of 8 rows")
    );

    expect(screen.getByRole("button", { name: /previous page/i })).toBeTruthy();
    expect(
      (screen.getByRole("button", { name: /previous page/i }) as HTMLButtonElement).disabled
    ).toBe(true);
    await user.click(screen.getByRole("button", { name: /next page/i }));
    expect(screen.getByTestId("history-count").textContent).toBe("Showing 6 to 8 of 8 rows");
    expect(
      (screen.getByRole("button", { name: /next page/i }) as HTMLButtonElement).disabled
    ).toBe(true);
    await user.click(screen.getByRole("button", { name: "Page 1" }));
    expect(screen.getByTestId("history-count").textContent).toBe("Showing 1 to 5 of 8 rows");
  });

  it("points the export button at the filtered CSV without paging", async () => {
    seedHistory();
    const user = await openHistory();

    await user.selectOptions(screen.getByLabelText(/event type/i), "CAPACITY_ALERT");
    await user.click(screen.getByText("Apply"));
    await waitFor(() =>
      expect(screen.getByTestId("history-count").textContent).toBe("Showing 1 to 1 of 1 rows")
    );

    const link = screen.getByTestId("export-csv") as HTMLAnchorElement;
    expect(link.textContent).toBe("Export CSV (1 rows)");
    expect(link.getAttribute("href")).toBe("/history/export.csv?event_type=CAPACITY_ALERT");
  });

  it("shows IST times, capacity wording and the entry/exit badges", async () => {
    seedHistory();
    const user = await openHistory();
    await waitFor(() => expect(screen.getByTestId("history-table")).toBeTruthy());

    // entry/exit rows are opt-in: ask for everything first
    await user.selectOptions(screen.getByLabelText(/event type/i), "");
    await user.click(screen.getByText("Apply"));
    await waitFor(() => expect(screen.getByText("ZONE ENTRY")).toBeTruthy());

    // 00:00:06 UTC is 05:30:06 in IST, whatever the test machine runs
    expect(screen.getByTestId("history-time-zone-101").textContent).toBe("2026-10-08 05:30:06");
    expect(screen.getByText("Zone 1 · 2 people")).toBeTruthy();
    expect(screen.getByText("Zone 1 · 2 people")).toBeTruthy();
    const entryBadge = screen.getByText("ZONE ENTRY");
    expect(entryBadge.className).toContain("badge-zone-entry");
    const exitBadge = screen.getByText("ZONE EXIT");
    expect(exitBadge.className).toContain("badge-zone-exit");
  });

  it("opens evidence from the table, disabled without a snapshot", async () => {
    seedHistory();
    const user = userEvent.setup();
    render(<App />);
    await waitForUi();
    fireEvent.click(screen.getByRole("button", { name: /^history$/i }));
    await waitFor(() => expect(screen.getByTestId("history-page")).toBeTruthy());
    // event 103 is a zone entry row: opt into all types first
    await user.selectOptions(screen.getByLabelText(/event type/i), "");
    await user.click(screen.getByText("Apply"));
    await waitFor(() =>
      expect(screen.getByRole("button", { name: /view evidence for event 103/i })).toBeTruthy()
    );

    expect(
      (screen.getByRole("button", { name: /view evidence for event 103/i }) as HTMLButtonElement)
        .disabled
    ).toBe(true);
    fireEvent.click(screen.getByRole("button", { name: /view evidence for event 101/i }));
    expect(
      screen.getByText(/Stayed too long - Worker #1 - Zone 1 - \d\d:\d\d:\d\d/)
    ).toBeTruthy();

    // step to the next row of the page, then close
    fireEvent.click(screen.getByRole("button", { name: /next alert/i }));
    expect(screen.getByText(/Too many people - Zone 1 · 2 people/)).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: /close evidence/i }));
    await waitFor(() =>
      expect(screen.queryByRole("dialog", { name: /alert evidence/i })).toBeNull()
    );
  });

  it("shows loading first and an empty state when nothing matches", async () => {
    seedHistory();
    const user = userEvent.setup();
    render(<App />);
    await waitForUi();
    fireEvent.click(screen.getByRole("button", { name: /^history$/i }));
    expect(screen.getByText("Loading events…")).toBeTruthy();

    await waitFor(() => expect(screen.getByTestId("history-table")).toBeTruthy());
    await user.selectOptions(screen.getByLabelText(/event type/i), "ENTRY_ALERT");
    await user.selectOptions(screen.getByLabelText(/^worker$/i), "2");
    await user.click(screen.getByText("Apply"));
    await waitFor(() =>
      expect(screen.getByTestId("history-table-empty").textContent).toContain(
        "No events match these filters."
      )
    );
  });

  it("formats stamps in IST and windows long page lists", () => {
    expect(istDateTime("2026-10-08T00:00:00Z")).toBe("2026-10-08 05:30:00");
    expect(istDateTime(null)).toBe("");
    expect(istDateTime("not-a-date")).toBe("");
    expect(pageWindow(1, 3)).toEqual([1, 2, 3]);
    expect(pageWindow(5, 12)).toEqual([1, 2, "…", 4, 5, 6, "…", 11, 12]);
  });
});

describe("full screen stage", () => {
  function mockFullscreenElement(value: Element | null) {
    Object.defineProperty(document, "fullscreenElement", {
      value,
      configurable: true,
      writable: true,
    });
  }

  it("toggles full screen on the stage and follows Esc", async () => {
    mockFullscreenElement(null);
    const requested: Element[] = [];
    HTMLDivElement.prototype.requestFullscreen = vi.fn(function (this: Element) {
      requested.push(this);
      return Promise.resolve();
    }) as unknown as typeof HTMLDivElement.prototype.requestFullscreen;
    state = makeState({ running: true });
    render(<App />);
    await waitForUi();

    const button = screen.getByTestId("fullscreen-btn");
    expect(button.getAttribute("aria-label")).toBe("Full screen");
    fireEvent.click(button);
    expect(requested).toHaveLength(1);
    expect((requested[0] as HTMLElement).className).toContain("stage");

    // Esc exits natively; the fullscreenchange listener resyncs the button
    mockFullscreenElement(requested[0]);
    fireEvent.click(button); // already full screen: asks to exit
    mockFullscreenElement(null);
    document.dispatchEvent(new Event("fullscreenchange"));
    await waitFor(() =>
      expect(screen.getByTestId("fullscreen-btn").getAttribute("aria-label")).toBe("Full screen")
    );
    delete (HTMLDivElement.prototype as Partial<HTMLDivElement>).requestFullscreen;
  });

  it("does nothing where the Fullscreen API is missing", async () => {
    mockFullscreenElement(null);
    state = makeState({ running: true });
    render(<App />);
    await waitForUi();

    // no requestFullscreen on the element: clicking must not throw
    fireEvent.click(screen.getByTestId("fullscreen-btn"));
    expect(screen.getByTestId("fullscreen-btn")).toBeTruthy();
  });

  it("maps pointer positions through the live rect, whatever its size", async () => {
    // a stage twice as wide as the frame (as in full screen): the rubber
    // band still reads frame pixels, so zones never shift
    vi.spyOn(HTMLImageElement.prototype, "getBoundingClientRect").mockReturnValue({
      left: 0,
      top: 0,
      width: 2560,
      height: 1440,
      right: 2560,
      bottom: 1440,
      x: 0,
      y: 0,
      toJSON: () => ({}),
    } as DOMRect);
    state = makeState({ running: true });
    const user = userEvent.setup();
    render(<App />);
    await waitForUi();
    await openToolbar(user);
    await pickTool(user, /^rectangle$/i);
    const image = screen.getByAltText("camera stream");
    fireEvent.mouseDown(image, { clientX: 0, clientY: 0 });
    fireEvent.mouseMove(window, { clientX: 1280, clientY: 720 });
    // half the display width is half the 1280-wide frame
    expect(rubberBand()?.tagName).toBe("rect");
    expect((rubberBand() as SVGRectElement).getAttribute("width")).toBe("640");
    fireEvent.mouseUp(window, { clientX: 1280, clientY: 720 });
    await waitFor(() => expect(shapeCount()).toBe(1));
  });
});
