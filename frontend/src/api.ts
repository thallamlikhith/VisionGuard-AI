import type {
  AlertEvent,
  AppConfig,
  AppState,
  DailyCounts,
  EventList,
  FilterOptions,
  FireAlert,
  FireAlertList,
  HistoryList,
  HistoryRow,
  TestVideo,
  VideoList,
  ZonePayload,
} from "./types";

async function request<T>(
  url: string,
  options?: RequestInit & { timeoutMs?: number },
): Promise<T> {
  const { timeoutMs = 15000, ...init } = options ?? {};
  const external = init.signal ?? null;
  const controller = new AbortController();
  const onAbort = () => controller.abort();
  if (external) {
    if (external.aborted) controller.abort();
    else external.addEventListener("abort", onAbort, { once: true });
  }
  const timer = window.setTimeout(() => controller.abort(), timeoutMs);
  let response: Response;
  try {
    response = await fetch(url, {
      headers: { "Content-Type": "application/json" },
      ...init,
      signal: controller.signal,
    });
  } catch (err) {
    if ((err as Error).name === "AbortError") {
      if (external?.aborted) throw new Error("Request cancelled");
      throw new Error("Request timed out: the backend took too long to answer");
    }
    throw new Error("Cannot reach the backend: is it running on port 8090?");
  } finally {
    window.clearTimeout(timer);
    external?.removeEventListener("abort", onAbort);
  }
  if (!response.ok) {
    let detail = `${response.status}`;
    try {
      const body = await response.json();
      detail = body.detail ?? detail;
    } catch {
      /* keep the status code */
    }
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return response.json() as Promise<T>;
}

/** fetch options every GET accepts: an abort signal plus a timeout override. */
export interface GetOptions {
  signal?: AbortSignal;
  timeoutMs?: number;
}

/** The saved settings. Passing a source reads that source's own zones, which is
 *  how the editor swaps in the zones of the clip just selected. */
export const getConfig = (source?: string, opts?: GetOptions) => {
  const query = source === undefined ? "" : `?source=${encodeURIComponent(source)}`;
  return request<AppConfig>(`/config${query}`, { signal: opts?.signal, timeoutMs: opts?.timeoutMs });
};

export const getState = (opts?: GetOptions) =>
  request<AppState>("/state", { signal: opts?.signal, timeoutMs: opts?.timeoutMs ?? 8000 });

export const saveConfig = (payload: ZonePayload) =>
  request<{ saved: boolean }>("/config", {
    method: "POST",
    body: JSON.stringify(payload),
  });

/** Switching the Camera source on its own. Nothing else is sent, so the
 *  backend only changes the source: the zones stored for every source are
 *  re-filed exactly as they are and are never validated against the frame of
 *  whichever source happens to be playing. */
export const switchSource = (source: string) =>
  request<{ saved: boolean }>("/config", {
    method: "POST",
    body: JSON.stringify({ source }),
  });

export const startMonitoring = (payload: ZonePayload) =>
  request<{ running: boolean }>("/start", {
    method: "POST",
    body: JSON.stringify(payload),
    // loading the detector takes seconds on first start
    timeoutMs: 90000,
  });

export const stopMonitoring = () =>
  request<{ running: boolean }>("/stop", { method: "POST" });

/** the three alert kinds, asked for as repeated query parameters */
export const getAlerts = (acknowledged?: boolean, opts?: GetOptions) => {
  const params = new URLSearchParams();
  params.append("event_type", "ENTRY_ALERT");
  params.append("event_type", "DWELL_ALERT");
  params.append("event_type", "CAPACITY_ALERT");
  if (acknowledged !== undefined) params.set("acknowledged", String(acknowledged));
  params.set("limit", "100");
  return request<EventList>(`/events?${params.toString()}`, {
    signal: opts?.signal,
    timeoutMs: opts?.timeoutMs ?? 8000,
  });
};

/** History table filters. Empty strings and empty lists mean "no filter". */
export interface HistoryQuery {
  eventTypes: string[];
  /** "" = every state; active/acked map to the ack flag, false to flagged
   *  fire rows only. */
  status?: "" | "active" | "acked" | "false";
  workerId?: number;
  zone?: string;
  dateFrom?: string;
  dateTo?: string;
  limit: number;
  offset: number;
}

export function historyParams(query: HistoryQuery): URLSearchParams {
  const params = new URLSearchParams();
  for (const type of query.eventTypes) params.append("event_type", type);
  if (query.status === "active") params.set("acknowledged", "false");
  else if (query.status === "acked") params.set("acknowledged", "true");
  else if (query.status === "false") params.set("false_alarm", "true");
  if (query.workerId !== undefined) params.set("worker_id", String(query.workerId));
  if (query.zone) params.set("zone", query.zone);
  if (query.dateFrom) params.set("date_from", query.dateFrom);
  if (query.dateTo) params.set("date_to", query.dateTo);
  params.set("limit", String(query.limit));
  params.set("offset", String(query.offset));
  return params;
}

/** one merged /history row as the backend sends it: zone rows in zone
 *  shape, fire rows in fire shape, told apart by kind. */
interface HistoryWireRow extends Partial<AlertEvent>, Omit<Partial<FireAlert>, "alert_type"> {
  kind: "zone" | "fire";
  alert_type?: string;
}

function toHistoryRow(raw: HistoryWireRow): HistoryRow {
  if (raw.kind === "fire") {
    const fireType = String(raw.alert_type ?? "").toLowerCase() === "smoke" ? "SMOKE" : "FIRE";
    return {
      id: Number(raw.id),
      worker_id: null,
      machine_id: null,
      event_type: fireType,
      reason: null,
      inside: null,
      entered_at: null,
      alert_at: (raw.reported_at as string | null) ?? null,
      dwell_seconds: null,
      zone_index: null,
      zone_name: null,
      has_snapshot: bool(raw.has_snapshot),
      acknowledged: bool(raw.acknowledged),
      acknowledged_at: (raw.acknowledged_at as string | null) ?? null,
      kind: "fire",
      alert_type: raw.alert_type,
      severity: raw.severity,
      confidence: raw.confidence ?? null,
      reported_at: (raw.reported_at as string | null) ?? null,
      source_label: raw.source_label ?? null,
      false_alarm: bool(raw.false_alarm),
      false_alarm_at: (raw.false_alarm_at as string | null) ?? null,
    };
  }
  return { ...(raw as AlertEvent), kind: "zone" };
}

function bool(value: unknown): boolean {
  return value === true;
}

/** One page of merged history rows plus the total the filters match. */
export const getHistory = async (query: HistoryQuery): Promise<HistoryList> => {
  const body = await request<{ count: number; total: number; rows: HistoryWireRow[] }>(
    `/history?${historyParams(query).toString()}`
  );
  return {
    count: body.count,
    total: body.total,
    rows: (body.rows ?? []).map(toHistoryRow),
  };
};

/** Distinct workers and zones for the history dropdowns. */
export const getFilterOptions = () => request<FilterOptions>("/events/filters");

/** The backend CSV download for the current filters (paging ignored). */
export function exportCsvUrl(query: HistoryQuery): string {
  const params = historyParams(query);
  params.delete("limit");
  params.delete("offset");
  const query_string = params.toString();
  return query_string ? `/history/export.csv?${query_string}` : "/history/export.csv";
}

export const acknowledgeAlert = (id: number) =>
  request<{ event: AlertEvent }>(`/events/${id}/ack`, { method: "POST" });

/** Fire/smoke alerts, newest first. The backend keeps true totals for both
 *  tabs, the same way /events does for zone alerts. */
export const getFireAlerts = (acknowledged?: boolean, opts?: GetOptions) => {
  const params = new URLSearchParams();
  if (acknowledged !== undefined) params.set("acknowledged", String(acknowledged));
  params.set("limit", "100");
  const query = params.toString();
  return request<FireAlertList>(`/fire-alerts${query ? `?${query}` : ""}`, {
    signal: opts?.signal,
    timeoutMs: opts?.timeoutMs ?? 8000,
  });
};

export const acknowledgeFireAlert = (id: number) =>
  request<{ alert: FireAlert }>(`/fire-alerts/${id}/ack`, { method: "POST" });

/** Export one fire/smoke alert as a retraining hard negative (also acks it).
 *  Idempotent: a repeat mark answers duplicate without new files. */
export const markFireFalseAlarm = (id: number) =>
  request<{ alert: FireAlert; exported: { image: string; metadata: string } | null; duplicate: boolean }>(
    `/fire-alerts/${id}/false-alarm`,
    { method: "POST" }
  );

/** Acknowledge every unacknowledged alert in one call. */
export const acknowledgeAllAlerts = () =>
  request<{ acknowledged: number }>("/events/ack-all", { method: "POST" });

/** incidents per day for the chart: one row per day, zeros included. */
export const getDaily = (days = 14, zone?: string, workerId?: number) => {
  const params = new URLSearchParams();
  params.set("days", String(days));
  if (zone) params.set("zone", zone);
  if (workerId !== undefined) params.set("worker_id", String(workerId));
  return request<DailyCounts>(`/events/daily?${params.toString()}`);
};

/** the URL an <img> can point at; the browser fetches it with no body needed */
export const alertSnapshotUrl = (id: number) => `/events/${id}/snapshot`;

/** the annotated picture saved with a fire/smoke alert */
export const fireAlertSnapshotUrl = (id: number) => `/fire-alerts/${id}/snapshot`;

export const getVideos = (opts?: GetOptions) =>
  request<VideoList>("/videos", { signal: opts?.signal, timeoutMs: opts?.timeoutMs ?? 8000 });

export const deleteVideo = (id: string) =>
  request<{ deleted: string }>(`/videos/${id}`, { method: "DELETE" });

/** Upload with a progress callback. XMLHttpRequest is used instead of fetch
 *  because fetch still cannot report upload progress, and a 500 MB clip needs
 *  the bar. The browser sets the multipart content type itself. */
export const uploadVideo = (file: File, onProgress?: (percent: number) => void) =>
  new Promise<TestVideo>((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/videos");
    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable && onProgress) {
        onProgress(Math.round((event.loaded / event.total) * 100));
      }
    };
    xhr.onload = () => {
      let body: { detail?: unknown } & Partial<TestVideo> = {};
      try {
        body = JSON.parse(xhr.responseText);
      } catch {
        /* a non-json body still carries a status code */
      }
      if (xhr.status >= 200 && xhr.status < 300 && body && body.id) {
        resolve(body as TestVideo);
        return;
      }
      reject(new Error(body?.detail ? String(body.detail) : `Upload failed (${xhr.status})`));
    };
    xhr.onerror = () => reject(new Error("Upload failed: cannot reach the backend"));
    const form = new FormData();
    form.append("file", file);
    xhr.send(form);
  });
