# PORTING_NOTES.md

One-page summary of the core logic for porting this prototype to another
service. Only the four files below carry the safety-relevant behaviour;
`api/server.py` (FastAPI), `frontend/` (React) and `visualize.py` (drawing)
are presentation and can be replaced wholesale.

---

## 1. `detector/tracker.py` - detection and tracking

**Class `PersonTracker(model_name, confidence, person_class_id, imgsz,
detect_every_n, max_detect_long_side, openvino)`** - loads YOLO11n, tracks
with Ultralytics' built-in ByteTrack. Raises `DetectorError` when the model
cannot load or inference fails.

- **Model**: COCO `yolo11n.pt`, class filter `[person_class_id]` (0) only.
  Optional OpenVINO export lives in `<stem>_openvino_model/` next to the
  weights: exported once at the configured `imgsz` (static input shape, fp16),
  re-exported once if the baked size differs, smoke-tested with one blank
  frame; on any failure it falls back to the `.pt` path automatically.
- **ByteTrack setup**: `model.track(frame, persist=True, tracker="bytetrack.yaml",
  classes=[0], conf=confidence, imgsz=imgsz)` - one persistent tracker across
  frames, so ids stay stable.
- **Conf floor**: `confidence` is passed as `conf` every frame (default 0.25);
  it is read live, so changing it on a running pipeline takes effect next frame.
- **`track(frame, frame_index) -> list[TrackedPerson]`**: `TrackedPerson` is
  `(worker_id: int, bbox: (x1,y1,x2,y2), confidence: float)`, sorted by id.
  Runs only when `frame_index % detect_every_n == 0` (`should_detect()`); in
  between it returns a copy of the last result, so the zone clock and picture
  stay smooth while YOLO runs every Nth frame.
- **Adaptive idle cadence** (`adaptive_idle_n`, 0 = off): while people are
  visible YOLO runs on the `detect_every_n` schedule, but an empty scene is
  only looked at every `adaptive_idle_n`-th frame. The cadence is decided
  from what the last detection *reported* (stale-held tracks keep it fast),
  so the zone clock still ticks on every frame either way.
- **Stale-hold**: if boxes came back *without* track ids (first frames of a new
  track / an unmatched frame), the previous stable list is held for at most
  `MAX_IDLESS_HOLD = 3` detection frames, then dropped. Inventing an id would
  restart dwell timers and duplicate `ZONE_ENTRY`.
- **Downscale**: frames whose long side exceeds `max_detect_long_side` (960)
  are resized (INTER_AREA) before inference and boxes are scaled back.

## 2. `zone/danger_zone.py` - the safety logic (pure Python, no cv2/YOLO deps)

**Geometry**

- `point_in_polygon(point, polygon) -> bool` - horizontal ray casting.
- `point_in_polygon_margin(point, polygon, margin) -> bool` - inside **and**
  more than `margin` pixels from every edge (**hysteresis**, `ZONE_MARGIN_PX =
  5.0`; 0 = plain test) so box jitter on the boundary cannot flicker the count.
- Anchor points: `bottom_center(bbox)` = feet `((x1+x2)//2, y2)`, `center(bbox)`
  = box middle; `ANCHORS = {"bottom_center", "center"}`. The configured anchor
  is what gets tested against the polygon.
- Alternate counting: `ZoneRule.count_by = "overlap"` tests whether the bbox
  overlaps the zone by more than `overlap_threshold` (default 0.10) of the
  bbox area, instead of the foot point.
- `zone_problem(points, width, height) -> str | None` - validation: 3..32
  points (`MIN/MAX_ZONE_POINTS`), no duplicate corners, in-frame when the size
  is known, non-self-crossing, area >= 0.5% of the frame
  (`MIN_ZONE_AREA_RATIO = 0.005`). `drop_repeats()` merges doubled corners.

**State machine** - `ZoneMonitor`, one `Presence` per worker *per zone*
(overlapping zones each get their own clock):

- States: `OUTSIDE`, `INSIDE`, `ALERT` (per presence); aggregate `SAFE`/
  `ALERT` in `status()`.
- `update_all(detections, now, wall=None) -> (workers, events)` - called once
  per frame. `detections = {worker_id: (anchor_point, bbox, confidence)}`;
  `now` = video-clock seconds (dwell timing), `wall` = unix timestamp (stored).
- **Entry**: first inside frame -> `ZONE_ENTRY`, clock starts. `INSTANT` mode
  goes straight to `ALERT` and emits `DWELL_ALERT` once per entry.
  `TIMER` mode: `DWELL_ALERT` when `dwell >= zone's dwell_seconds`, then
  re-alerts every `repeat_alert_seconds` (0 = never).
- **Exit**: continuously outside for `EXIT_GRACE = 0.6 s` -> `ZONE_EXIT`
  (dwell frozen at the first outside frame). Back inside during the grace:
  nothing recorded, clock not restarted. Entering is always immediate.
- **Lost track**: a worker missing for `lost_timeout_frames = 15` frames gets
  one `ZONE_EXIT` per zone and is forgotten; below that its clocks keep
  running, so one dropped detection never clears an active alert.
- **Capacity rule**: per zone, `occupants > allowed_people` -> immediate
  `CAPACITY_ALERT` (no dwell), back within limit -> `CAPACITY_SAFE`; emitted
  only on state changes, `worker_id = -1`. Each zone counts only its own
  people against its own allowance.
- **Attribution**: the headline worker state comes from the alarming zone with
  the longest dwell, else the lowest zone index (`attributed_zone()`); the
  per-zone presences remain the truth.
- `reset_timing() -> list[int]` - after a zone is redrawn/switched: re-checks
  everyone, drops stale clocks, emits no events, returns changed ids.
- Read helpers: `summary()` (per worker), `zone_reports()` (per zone),
  `status()` (overall) - dicts for `/state`.

**Event types and fields** (`Event.to_dict()`):
`event_type` in `ZONE_ENTRY`, `DWELL_ALERT`, `ZONE_EXIT`, `CAPACITY_ALERT`,
`CAPACITY_SAFE`; fields `worker_id`, `machine_id`, `event_type`, `entered_at`
(wall), `alert_at` (wall), `dwell_seconds`, `zone_index`, `zone_name`.

**Rule object** `ZoneRule(name, points, allowed_people=0, dwell_seconds=5.0,
enabled=True, using_defaults=True, count_by="foot_point",
overlap_threshold=0.10)`; `using_defaults` makes a zone follow the global
allowance/dwell until it gets its own values.

## 3. `pipeline.py` - the frame loop

- `Pipeline(database).start(source, machine, points/zones, allowed_people,
  mode, dwell_seconds, repeat_alert_seconds, model, confidence,
  person_class_id, imgsz, detect_every_n, max_detect_long_side, openvino,
  anchor, zone_reference, realtime, loop, zone_specs, zone_margin_px)` -
  validates everything first, then builds `DangerZone` + `ZoneMonitor` +
  `PersonTracker` + `SnapshotWriter` and spawns one daemon thread. Raises
  `PipelineError`.
- **Frame loop (`_run`)** per frame: read -> video clock (`Timeline.update`) ->
  `tracker.track()` -> under a config lock: anchor function -> `update_all()`
  -> every event to `db.insert_event()`; events of type `DWELL_ALERT` or
  `CAPACITY_ALERT` additionally `snapshots.submit(event_id, jpeg)` (entry/exit
  rows get no picture) -> draw person boxes only (the zone is an SVG overlay in
  the browser) -> JPEG quality 80 -> publish as latest frame. Real-time pacing
  sleeps to the file's fps and drops at most `MAX_SKIP = 30` frames when behind;
  end of file rewinds and clears workers when `loop=True`.
- `DetectorError` skips the frame; `MAX_CONSECUTIVE_ERRORS = 30` in a row
  fails the pipeline.
- `apply_settings(...) -> list[int]` - live zone/rule/anchor/confidence change
  without reopening the source; validates before touching anything, then calls
  `reset_timing()`.
- `scale_zone_points/polygons(points, reference, actual)` - a zone drawn at one
  resolution is scaled (and clamped) onto the real frame size when the source
  opens, so stored zones survive a camera change.
- `state() -> dict` and `jpeg() -> bytes | None` - the `/state` and MJPEG feed.

## 4. `snapshots.py` - alert pictures

- `SnapshotWriter(database)`: daemon thread with a bounded queue
  `MAX_QUEUE = 32`; `submit(event_id, jpeg)` never blocks and drops (with a
  log line) when full, so the video thread never waits on disk.
- Writes `{event_id}.jpg` into `TRINETRA_SNAPSHOT_DIR` (env) or
  `data/snapshots/`, stores only the bare name via `db.set_snapshot_path`.
- `_prune()` after every write keeps the newest `KEEP = 300` files by mtime.
- `close(timeout=2.0)` drains and stops.

## 5. Default values

| Parameter | Code default | Shipped `config.yaml` |
| --- | --- | --- |
| model | `yolo11n.pt` | `yolo11n.pt` |
| confidence (conf floor) | 0.25 | 0.25 |
| person_class_id | 0 | 0 |
| imgsz | 320 | **640** |
| detect_every_n | 1 | 1 |
| max_detect_long_side | 960 | 960 |
| openvino | False | **true** |
| anchor | `bottom_center` (Pipeline) / `center` (config loader) | `center` |
| zone_margin_px (hysteresis) | 5.0 | 5 |
| mode | `TIMER` | `TIMER` |
| dwell_seconds | 5.0 (loader fallback 1) | 1 |
| allowed_people (capacity) | 0 | 0 (stored in SQLite) |
| repeat_alert_seconds | 10.0 | 10 |
| exit_grace_seconds | 0.6 | - |
| lost_timeout_frames | 15 | - |
| MAX_IDLESS_HOLD (stale-hold) | 3 detection frames | - |
| ZoneRule.dwell_seconds | 5.0 | per zone |
| overlap_threshold | 0.10 | per zone |
| zone validation | 3..32 points, >= 0.5% of frame | - |
| PERF_WINDOW / MAX_SKIP / MAX_CONSECUTIVE_ERRORS | 30 / 30 / 30 | - |
| SnapshotWriter MAX_QUEUE / KEEP | 32 / 300 | - |
| JPEG quality / stop join timeout | 80 / 5 s | - |

**Ordering rules worth keeping**: validate before mutating (start and
apply_settings); the zone step runs under its own lock so a settings change
never lands mid-frame; capacity alert immediately, dwell alert on video clock;
exit is debounced (0.6 s) but entry is immediate; alerts write a picture,
entry/exit rows do not.
