export type Point = [number, number];

export type WorkerState = "OUTSIDE" | "INSIDE" | "ALERT";

/** the Paint-style tools: pick/move, draw one of four shapes, or click out an
 *  N-point polygon (DMS-style: unlimited corners, closed on demand) */
export type DrawTool = "select" | "rectangle" | "quad" | "polygon" | "circle" | "freehand";

export type RuleMode = "TIMER" | "INSTANT" | "LIMIT";

/** which point of a person has to be inside the zone: feet for a machine on
 *  the ground, body centre for somebody sitting at a desk */
export type AnchorPoint = "bottom_center" | "center";

export interface Worker {
  worker_id: number;
  state: WorkerState;
  dwell_seconds: number;
  /** the point the inside/outside test used, in frame pixels: the bottom-centre
   *  of the box for Feet, the middle of it for Body center. Optional, because a
   *  backend older than this field sends nothing and the UI must still run. */
  point?: [number, number];
  /** the zone this row is reported against, -1 when they are in none. With one
   *  zone this is always 0, so the old rendering is unchanged. */
  zone_index?: number;
  /** the name of that zone, empty when they are in none */
  zone_name?: string;
  /** every zone they are actually standing in, ascending. This is not the same
   *  as zone_index: somebody in two zones is listed under both but reported
   *  against one. */
  zones?: number[];
}

/** one zone's name and the rules that belong to it alone. */
export interface ZoneSpec {
  name: string;
  allowed_people: number;
  dwell_seconds: number;
  enabled: boolean;
  /** still following the global defaults; a zone with its own rule typed in
   *  keeps it when the globals change, so the UI has to say which it is */
  using_defaults: boolean;
  count_by?: "foot_point" | "overlap";
  overlap_threshold?: number;
}

/** what one zone is doing right now, read from /state. */
export interface ZoneReport {
  index: number;
  name: string;
  /** how many people are inside this zone */
  inside: number;
  allowed_people: number;
  /** the longest anyone has been in THIS zone */
  dwell_seconds: number;
  /** THIS zone's own dwell limit. Optional because a backend from before
   *  per-zone reports may not send it; the UI then falls back to the global. */
  dwell_limit?: number;
  /** over its allowance, or somebody in it has passed its dwell limit */
  alerting: boolean;
  enabled: boolean;
}

/** rolling timings published by the pipeline; every field is optional so the
 *  UI still renders when an older backend does not send it at all. */
export interface PipelinePerf {
  fps?: number;
  ms_per_frame?: number;
  inference_ms?: number;
  draw_ms?: number;
  encode_ms?: number;
}

/** gate progress while one presence episode is being confirmed: consecutive
 *  samples seen out of the required window. Null while there is nothing to
 *  confirm (no samples yet, or already crossed). */
export interface GateProgress {
  samples: number;
  required: number;
}

export interface FireSmokeState {
  enabled: boolean;
  /** per-type switches (filter, not extra inference). Optional, because
   *  backends from before the split send neither. */
  fire_enabled?: boolean;
  smoke_enabled?: boolean;
  analyzed?: number;
  emitted?: number;
  last_types?: string[];
  last_crossed?: string[];
  last_frame_index?: number | null;
  last_capture_ts?: number | null;
  analysis_fps?: number;
  confirming?: { fire: GateProgress | null; smoke: GateProgress | null } | null;
}

export interface CrowdMetrics {
  person_count: number;
  coverage_fraction: number;
  density_per_megapixel: number;
}

export interface CrowdState {
  enabled: boolean;
  source?: string;
  metrics?: CrowdMetrics;
  last_frame_index?: number | null;
  last_capture_ts?: number | null;
  analysis_fps?: number;
}

export interface PpeStats {
  violation_ratio?: number | null;
  violation_count?: number | null;
  person_count?: number | null;
  fall_count?: number | null;
  other_count?: number | null;
  /** aggregate [name, count] pairs per violation type, no per-worker data. */
  violations_by_class?: [string, number][];
}

export interface PpeState {
  enabled: boolean;
  grade?: string;
  analyzed?: number;
  stats?: PpeStats;
  last_frame_index?: number | null;
  last_capture_ts?: number | null;
  /** true while PPE alert routing is deferred (grade only). Read from the
   *  backend's PPE_EMISSION_DEFERRED constant by way of /state, never set
   *  here, so the caption below cannot drift from the backend. */
  deferred?: boolean;
}

/** one side path's saved settings: switch + cadence, both optional so a
 *  partial save keeps the rest. */
export interface SidePathSettings {
  enabled?: boolean;
  analysis_fps?: number;
}

/** saved side-path settings block, as /config reports it. Fields are
 *  optional because backends from before the Detection paths card send
 *  nothing and the UI must still run. */
export interface SidePathConfig {
  enabled: boolean;
  analysis_fps: number;
  /** per-type fire/smoke switches (fire_smoke block only). */
  fire_enabled?: boolean;
  smoke_enabled?: boolean;
  /** gate persistence windows in seconds (fire_smoke block only). Optional
   *  because older backends do not report them; the UI falls back to the
   *  shipped 1.5 s / 2.5 s operating point for its gate-time hint. */
  fire_seconds?: number;
  smoke_seconds?: number;
}

export interface AppConfig {
  machine: { id: string; name: string };
  rules: { mode: string; dwell_seconds: number };
  detector: { model: string; confidence: number; person_class_id: number };
  /** every danger area, one polygon each. A machine can have several. */
  zones: Point[][];
  /** each zone's name and rules, index aligned with `zones`. Optional, because
   *  a database saved before zones had names has none: the UI falls back to the
   *  global defaults and labels them "Zone N" itself. */
  zone_specs?: ZoneSpec[];
  /** the first zone only, for a backend that predates several zones */
  zone_points?: Point[];
  /** the frame size the saved points were drawn against; when it differs from
   *  the live frame the UI scales the points before drawing them */
  zone_reference?: [number, number];
  allowed_people: number;
  anchor: AnchorPoint;
  source: string;
  /** saved side detection paths (Settings "Detection paths" card).
   *  Optional, because backends from before the card send nothing. */
  fire_smoke?: SidePathConfig;
  crowd?: SidePathConfig;
  ppe?: SidePathConfig;
  /** hazard-zone system switch. Optional, like the blocks above. */
  hazard?: SidePathConfig;
  /** stored test-clip replay switch. Optional, because backends from before
   *  the Loop video box send nothing: off (play once) is the fallback. */
  loop_video?: boolean;
  /** upload + history caps. Optional, because backends from before the
   *  limits block send nothing: each falls back to its current value. */
  limits?: {
    max_upload_size_mb?: number;
    max_videos?: number;
    history_days?: number;
  };
}

export interface AppState {
  running: boolean;
  error: string | null;
  /** why the saved source cannot be opened while stopped (a test clip that was
   *  never cloned or has been deleted), or null when the source looks usable.
   *  Only set while stopped; starting reports failures through `error`. */
  source_error?: string | null;
  source: string;
  frame_width: number;
  frame_height: number;
  frame_seq: number;
  machine: { id: string; name: string };
  rules: { mode: string; dwell_seconds: number };
  /** every danger area being monitored, one polygon each */
  zone_polygons?: Point[][];
  /** the rules the running camera is using, one per zone */
  zone_specs?: ZoneSpec[];
  /** one row per zone: its own inside count, allowance and alert. This is what
   *  each card renders, so two zones never share one number. */
  zone_reports?: ZoneReport[];
  /** the first zone only, for a backend that predates several zones */
  zone_points?: Point[];
  allowed_people: number;
  inside: number;
  status: "SAFE" | "ALERT" | "off";
  /** hazard-zone system switch. Optional, because backends from before the
   *  switch report neither it nor the "off" status. */
  hazard?: { enabled: boolean };
  /** the allowed-people rule is over capacity */
  capacity_alert: boolean;
  /** at least one worker is in the dwell ALERT state */
  worker_alert: boolean;
  message: string;
  workers: Worker[];
  /** frame rate and per-stage timings, null while nothing has been measured */
  perf?: PipelinePerf | null;
  /** side detection paths as the running pipeline (or the saved settings
   *  while stopped) reports them. Optional throughout, because backends
   *  from before Phase 1 send nothing and the UI must still run. */
  fire_smoke?: FireSmokeState;
  crowd?: CrowdState;
  ppe?: PpeState;
  /** the source being monitored is one of the uploaded test videos. Decided by
   *  the backend, so the badge does not have to guess from the path. */
  test_video?: boolean;
  /** a single-play file ended by itself. Optional, because older backends
   *  never report it: the stopped panel falls back to "Camera stopped". */
  finished_clip?: boolean;
}

/** one uploaded test video, with everything read off the real file. */
export interface TestVideo {
  id: string;
  /** the stored file name inside data/videos/, never used as a path by the UI */
  file: string;
  /** the browser's file name, shown only */
  name: string;
  size_mb: number;
  width: number;
  height: number;
  fps: number;
  duration: number;
  created: string;
  /** the backend path to hand to the Camera source field */
  source: string;
}

export interface VideoList {
  videos: TestVideo[];
  /** how many videos may be kept at once */
  limit: number;
  /** the upload size ceiling in MB, as the backend enforces it */
  max_mb: number;
}

/** one row of /fire-alerts: a recorded fire/smoke moment, optional picture.
 *  Ids come from their own table, so they never collide with zone event ids. */
export interface FireAlert {
  id: number;
  alert_type: "fire" | "smoke";
  severity: string;
  confidence: number;
  bbox_xyxy: [number, number, number, number] | null;
  /** video seconds of the analysed frame, null when unknown */
  captured_at: number | null;
  reported_at: string | null;
  /** true when the backend saved an annotated picture for this alert */
  has_snapshot: boolean;
  acknowledged: boolean;
  acknowledged_at: string | null;
  /** true once exported as a retraining hard negative (implies acknowledged) */
  false_alarm: boolean;
  false_alarm_at: string | null;
  /** the label stored at alert time: which source the row came from. Absent
   *  on rows written before labels existed (they read back null). */
  source_label?: string | null;
}

export interface FireAlertList {
  alerts: FireAlert[];
  total: number;
  unacked: number;
  acked: number;
}

/** one row of /events: a recorded moment, with an optional picture. */
export interface AlertEvent {
  id: number;
  worker_id: number | null;
  machine_id: number | string | null;
  event_type: string;
  /** why the row was written: ENTRY, DWELL or CAPACITY. Rows written before
   *  the reason existed have none and fall back to event_type wording. */
  reason?: string | null;
  /** how many people were inside the zone when the row was written. Only
   *  capacity rows carry one. */
  inside?: number | null;
  entered_at: string | null;
  alert_at: string | null;
  dwell_seconds: number | null;
  zone_index: number | null;
  zone_name: string | null;
  /** true when the backend saved a picture for this event */
  has_snapshot: boolean;
  acknowledged: boolean;
  acknowledged_at: string | null;
}

export interface EventList {
  count: number;
  total: number;
  events: AlertEvent[];
}

/** one row of /history: a zone event or a fire/smoke alert, told apart by
 *  kind (ids live in separate tables and can collide across them). Fire
 *  rows carry their own fields alongside the zone shape; zone rows leave
 *  them undefined. */
export interface HistoryRow extends AlertEvent {
  kind: "zone" | "fire";
  /** "fire" | "smoke" on fire rows, in backend storage case */
  alert_type?: string;
  severity?: string;
  confidence?: number | null;
  reported_at?: string | null;
  source_label?: string | null;
  false_alarm?: boolean;
  false_alarm_at?: string | null;
}

export interface HistoryList {
  count: number;
  total: number;
  rows: HistoryRow[];
}

/** Distinct workers and zones for the history dropdowns. */
export interface FilterOptions {
  workers: number[];
  zones: { index: number; name: string }[];
}

/** one day of the alert chart: incidents, not beeps. Fire and smoke ride
 *  along as their own series; both are optional so fixtures written before
 *  the fire series still compile, and missing means zero. */
export interface DailyDay {
  date: string;
  entry: number;
  dwell: number;
  capacity: number;
  fire?: number;
  smoke?: number;
}

export interface DailyCounts {
  days: DailyDay[];
  days_requested: number;
}

export interface ZonePayload {
  /** every danger area, one polygon each */
  zones: Point[][];
  /** each zone's name and rules, index aligned with `zones`. A zone that is not
   *  sent keeps what it had, so changing one card cannot reset the others. */
  zone_specs?: Partial<ZoneSpec>[];
  allowed_people: number;
  source?: string;
  machine_name?: string;
  mode?: RuleMode;
  dwell_seconds?: number;
  repeat_alert_seconds?: number;
  /** which point of a person has to be inside the zone */
  anchor?: AnchorPoint;
  /** detection confidence for the person detector, 0-1 */
  confidence?: number;
  /** the real frame size the points belong to, saved so a later camera with a
   *  different resolution can still scale the zone back up */
  frame_width?: number;
  frame_height?: number;
  /** side detection paths (Settings "Detection paths" card). Sent only when
   *  the card is used; an absent block keeps the saved values. */
  fire_smoke?: SidePathSettings;
  crowd?: SidePathSettings;
  ppe?: SidePathSettings;
  /** hazard-zone system switch. Same send-only-when-used rule. */
  hazard?: SidePathSettings;
  /** per-type fire/smoke switches (the shared block above stays as the
   *  legacy mirror). Same send-only-when-used rule. */
  fire?: SidePathSettings;
  smoke?: SidePathSettings;
  /** replay this start when the file ends. Sent on Start so the on-screen
   *  Loop box wins over the stored switch without waiting for the debounced
   *  save; absent keeps the stored value. */
  loop?: boolean;
  /** stored test-clip replay switch (the Loop video box). Sent by the
   *  debounced settings save; absent keeps the saved value. */
  loop_video?: boolean;
}
