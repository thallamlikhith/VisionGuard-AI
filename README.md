# Trinetra AI (prototype)

Watches a video or webcam, detects **people** with YOLO, tracks them with
**ByteTrack**, gives each one a stable `Worker #id`, and warns when a worker
enters a **manually drawn danger zone polygon** around a machine and stays
there longer than a configurable dwell time.

> **Prototype only.** This is a warning / monitoring demo, *not* a certified
> safety system. See [Limitations](#limitations).

Flow:

```
Video / Webcam
  -> YOLO11n person detection (COCO class 0 only)
  -> ByteTrack tracking -> Worker #1, #2, #3 ...
  -> bottom-centre point of the bounding box
  -> point-in-polygon test against the danger zone drawn in the UI
  -> OUTSIDE / INSIDE / ALERT state + per-worker dwell timer
  -> capacity rule: people inside > allowed people -> immediate ALERT
  -> event stored in SQLite
  -> everything drawn on the video frame
  -> live video + controls in a React dashboard
```

Optional side paths, all off by default and switched per system in the UI
(see [Detection systems](#detection-systems-one-source-of-truth)):

```
  -> fire/smoke pass (fire_smoke_v2.pt, one pass, per-type Fire/Smoke switches)
  -> crowd metrics (person-tracker reuse, or crowd_v1.pt)
  -> PPE grading only (ppe_v1.pt, never alerts)
```

There are two ways to use it:

- **Web dashboard (recommended)** - React + Vite UI: draw the danger zone by
  clicking on the live video, set the allowed people count, watch workers and
  alerts in the browser. See [section 2.1](#21-web-dashboard-recommended).
- **Command line** - the original `app.py` loop with an OpenCV window, using the
  polygon from `config.yaml`.

---

## 1. Install

Python 3.11 or newer.

```bash
pip install -r requirements.txt
```

The YOLO weights (`yolo11n.pt`, ~5 MB) download automatically on the first
run. Make sure you are online for that first run, or copy the file next to
`app.py`.

---

## 2. Run

### 2.0 Fresh clone checklist

After `git pull` / a fresh clone, three things are **not** in the repo, all on
purpose (see `.gitignore`):

| Missing after clone | Where it lives | What you see | What to do |
| --- | --- | --- | --- |
| Danger zones | SQLite `data/trinetra.db` (per-source) | "No danger zone yet" | Draw one (§2.1 steps 3–5) or paste the sample from `config.yaml` |
| Test videos | `data/videos/*.mp4` | Videos card is empty, camera stopped | Settings → Videos (test inputs) → Upload Video (or copy an `.mp4` there and restart) |
| Built UI | `frontend/dist/` | — (dev server rebuilds it) | Nothing; `npm run dev` serves from source |

Side-path model weights (`detector/models/fire_smoke_v2.pt`,
`crowd_v1.pt`, `ppe_v1.pt`) are also git-ignored and are **not**
downloaded: copy them into `detector/models/` before switching those
systems on, or enabling a path fails fast with the missing file named.
The person model (`yolo11n.pt`, ~5 MB) downloads automatically on the
first run.

Concretely:

1. `pip install -r requirements.txt`, then `cd frontend && npm install`
   (re-run `npm install` whenever `package.json` changed after a pull, then
   restart the Vite server so its optimizer cache picks the new deps up).
2. Double-click **`start.bat`** (backend `8090` + frontend `5174`), or run the
   two terminals in §2.1 by hand.
3. Open <http://127.0.0.1:5174>. If the page looks unstyled or stale after a
   pull, hard-refresh (`Ctrl+Shift+R`) — the dev server serves the new files,
   the browser may still show the cached ones.
4. Press **Start camera** (webcam `0` by default). With no saved zone the video
   still plays so a zone can be drawn on top of the live picture.
5. If the stopped panel says **"Test video not found at \<path\>"**, the saved
   source points at a clip that is not on this machine: re-upload it under
   Settings → Videos (test inputs), or switch the source back to webcam `0`
   in Settings → Camera. (Pressing Start with a missing file is refused with
   the same message.)

No environment variables are required: every `TRINETRA_*` override
(`_DB`, `_CONFIG`, `_VIDEO_DIR`, `_DIST`, `_API`) falls back to the paths
above. There is no `.env` file; do not create one.

### 2.1 Web dashboard (recommended)

Double-click **`start.bat`**. It is the only launcher in the project and it
starts both servers for you:

| Server | Port | What it is |
| --- | --- | --- |
| backend | 8090 | FastAPI, the video, the detector and the tracker |
| frontend | 5174 | Vite dev server, the UI, proxying API calls to 8090 |

It installs the frontend packages on the first run, checks that both ports are
free and that the Node toolchain works, opens
<http://127.0.0.1:5174> once both answer, and stops the backend again if the
frontend fails to start. Close the two console windows to stop everything.

The dashboard is opened by starting Edge or Chrome **from its full path**, not
by handing the URL to Windows. The Windows way looks up whichever program is
registered as the `http` handler and runs that; if that program is a broken or
half-installed browser, Windows answers with a *"The application has failed to
start because its side-by-side configuration is incorrect"* dialog and no
browser at all. Naming the executable skips that lookup. If neither browser
exists the launcher prints `Open this link in your browser: <url>` and leaves
the servers running.

Set `NO_BROWSER=1` (or the older `TRINETRA_NO_BROWSER=1`) to start the
servers without opening anything, which is what the scripted checks use.

To run the two servers yourself instead:

**Terminal 1 - backend** (runs the video, detector and tracker):

```bash
pip install -r requirements.txt
uvicorn api.server:app --host 127.0.0.1 --port 8090
```

**Terminal 2 - frontend** (the UI, proxies API calls to port 8090):

```bash
cd frontend
npm install
npm run dev
```

Then open <http://127.0.0.1:5174>.

Using the dashboard (sidebar: Live Monitor, Alerts, History, Settings):

1. Settings → **Camera** → **Video source** dropdown: pick the webcam or an
   uploaded clip. The resolution line only appears when a real frame size is
   known. Upload clips under Settings → **Videos (test inputs)** → **Upload
   Video** (or drag & drop); each row shows name, size and date, and the
   selected clip is marked **In use** and cannot be deleted while selected.
2. Press **Start** on Live Monitor. The live video appears.
3. Click **Edit zones** (or **+ Add zone**) on the live video, pick a tool
   and draw: **Rectangle** (drag), **4-point** (click 4 corners — the zone
   is made on the 4th click), **Circle** (drag centre to edge), **Freehand**
   (draw one loop). A corner too close to the last one is refused with a
   note, and a self-crossing 4th corner is refused without losing the first
   three. **Escape** cancels the open shape, **Undo** takes the last corner
   back. Drag a corner handle to move it, double-click a handle to delete
   it, or click a **+** midpoint to add a corner there.
4. Settings → **Detection Defaults**: detection confidence, dwell limit
   (seconds) and allowed people (slider plus number field, kept in sync).
   Edits autosave — the **Applied** line confirms the backend stored them.
5. Press **Save & apply**. The zone and the count are stored and monitoring
   restarts with the new values.

### Detection systems (one source of truth)

Live Monitor → **Detection systems** panel and Settings → **Detection
Paths** card are two views of the same five stored switches: Hazard zone,
Fire, Smoke, Crowd, PPE. Toggling either side writes the same setting, so
the other side follows immediately; saving rides the existing settings
autosave and applies live to a running camera.

Tick-box behaviour:

- Unticking a system stops **new** results for it. Stored alerts stay
  visible in Alerts and History.
- Hazard zone off: the banner reads OFF ("Hazard monitoring off") instead
  of SAFE, and zone evaluation stops.
- Fire and Smoke are separate switches on **one shared pass**: one model,
  one cadence. The Target FPS field sits on the Fire row and is labelled
  "shared with Smoke".
- Crowd reports numbers only, never alerts. PPE is grading only ("Grading
  only, alerts not enabled").
- The Alerts sidebar badge counts unacknowledged zone alerts plus
  unacknowledged fire/smoke alerts: the sum of the two Active counts on the
  Alerts page.

Notes:

- With no zone the video still plays, so you can draw the zone on top of
  the live picture. Zones with more than 4 corners (drawn by older
  versions) still load, display, move and edit corner by corner.
- Video files play once and stop; tick Loop video (Live Monitor source
  picker or Settings Camera card) to replay a clip when it ends.
- The state panel shows SAFE / ALERT, the number of people inside, and each
  worker currently in the zone.

### 2.2 Command line

```bash
python app.py --source video.mp4
```

### Webcam

```bash
python app.py --source 0     # default camera
python app.py --source 1     # second camera
```

A plain number is treated as a camera index, anything else as a file path.

Press `q` (or `Esc`) to stop.

### API only

```bash
uvicorn api.server:app --port 8090
```

Then <http://127.0.0.1:8090/docs> for the API docs.

### Useful extra flags

| flag | what it does |
| --- | --- |
| `--config config.yaml` | use a different config file |
| `--db data/trinetra.db` | use a different SQLite file |
| `--model yolov8n.pt` | override the model |
| `--no-show` | do not open a window (headless / server) |
| `--save out.mp4` | write the annotated video to a file |
| `--max-frames 200` | stop after N frames (quick test) |

Example, processing a file without a GUI and saving the result:

```bash
python app.py --source video.mp4 --no-show --save out.mp4
```

---

## 3. Configure the danger zone

### In the dashboard

Draw it with any tool (Rectangle, 4-point, Circle, Freehand), drag corners
to adjust, **Save & apply**. The polygon
and the allowed people count are saved to SQLite and survive a restart, so
`config.yaml` is only the starting default.

### In config.yaml

The danger zone is a **polygon in pixel coordinates of the video frame**.

```yaml
machine:
  id: excavator_01
  name: Excavator 01

danger_zone:
  anchor: bottom_center        # or: center
  points:
    - [400, 200]
    - [700, 200]
    - [760, 500]
    - [350, 500]

rules:
  mode: TIMER                  # or: INSTANT
  dwell_seconds: 5
```

**To change the polygon**

1. Run `app.py` once and note the frame size it prints:
   `[video] video.mp4 - 1280x720`
2. Open the video in any image viewer, zoom in on the machine, and note the
   `(x, y)` pixel coordinates of its corners. `(0, 0)` is the top-left corner.
3. Replace the `points` list. Order does not matter, but 3 or more points are
   required and the shape should not self-intersect.

`app.py` prints a warning if the polygon is bigger than the frame:

```
[warn] danger zone reaches (760, 500) but the video is only 640x480 - update danger_zone.points in config.yaml
```

The shipped points are sized for a **1280x720** video.

**Which point of the box is tested** — `danger_zone.anchor`:

- `bottom_center` (default) — middle of the bottom edge, i.e. the worker's
  feet. Usually what you want for "is the person standing in this area".
- `center` — middle of the bounding box.

**To change the dwell time** — set `rules.dwell_seconds` (seconds, must be > 0):

```yaml
rules:
  mode: TIMER
  dwell_seconds: 10     # warn after 10 seconds inside
```

**To switch alert mode** — set `rules.mode`:

- `TIMER` — a worker is `INSIDE` at first, and only becomes `ALERT` once the
  dwell time exceeds `dwell_seconds`.
- `INSTANT` — a worker becomes `ALERT` the instant it enters the zone
  (`dwell_seconds` is ignored).

Other keys:

- `detector.confidence` — detection confidence threshold (default `0.35`).
- `detector.model` — any Ultralytics model; `yolo11n.pt` is the CPU-friendly
  default, `yolov8n.pt` also works.
- `runtime.repeat_alert_seconds` — re-alert interval for a worker that is
  still inside, so a long stay does not emit one event per frame (default 10).
- `runtime.show_zone_fill` — draw a translucent fill inside the polygon.

### Allowed people (capacity)

**Allowed people** is how many workers may stand in the danger zone at the same
time. The zone is over capacity when:

```
people inside  >  allowed people
```

- Default is `0`, so **one person inside is an immediate alert** — no dwell
  wait. The dwell timer only decides when a worker that is allowed in turns
  `INSIDE` into `ALERT`.
- `1` = one worker allowed, two triggers the alert.
- The capacity alert is written **once per transition**, not once per frame, so
  a busy zone does not flood the database. It clears when the count drops back
  to the allowed level.
- Change it in the UI and press **Save & apply**.

---

## 4. States

| state | meaning | box colour |
| --- | --- | --- |
| `OUTSIDE` | worker is outside the danger zone | green |
| `INSIDE` | worker is inside, dwell time not yet reached | amber |
| `ALERT` | worker is inside for longer than `dwell_seconds` | red |

Each worker gets its **own** entry time, so several workers can be inside the
zone at the same time without interfering. When a worker leaves the zone — or
disappears from the frame — its timer is reset. A full `ZONE_ENTRY` ->
`DWELL_ALERT` -> `ZONE_EXIT` sequence is written to SQLite for each visit.

---

## 5. Database

`data/trinetra.db`, three tables.

`events` - one row per visit:

| column | meaning |
| --- | --- |
| `id` | auto increment row id |
| `worker_id` | ByteTrack id of the worker |
| `machine_id` | from `machine.id` in the config |
| `event_type` | `ZONE_ENTRY`, `DWELL_ALERT`, `ZONE_EXIT`, `CAPACITY_ALERT` or `CAPACITY_SAFE` |
| `entered_at` | unix timestamp when the worker entered the zone |
| `alert_at` | unix timestamp of the alert (alerts only) |
| `dwell_seconds` | how long the worker had been inside |

`fire_alerts` - one row per fire/smoke alert (own table, so zone rows,
counts, history and the zone CSV are untouched):

| column | meaning |
| --- | --- |
| `id` | auto increment row id (its own id space — never compared with event ids) |
| `alert_type` | `fire` or `smoke` |
| `severity` | graded `RED` / `ORANGE` / `YELLOW` |
| `confidence` | detector confidence 0–1 |
| `bbox_xyxy` | alert box, JSON array |
| `captured_at` | video seconds of the analysed frame |
| `reported_at` | unix timestamp the alert was stored |
| `snapshot_path` / `clean_snapshot_path` | annotated / clean JPEG file names |
| `acknowledged` / `acknowledged_at` | ack state |
| `false_alarm` / `false_alarm_at` | hard-negative export flag (implies acked) |
| `source_label` | which source the row came from, stored at alert time (`NULL` on rows written before labels existed) |

`settings` - the zone and the rules saved from the UI (`zone_points`,
`allowed_people`, `source`). Values saved here override `config.yaml` on the
next start. Side paths persist here too, as JSON blocks: `hazard`,
`fire_smoke` (`enabled`, per-type `fire_enabled`/`smoke_enabled`,
`analysis_fps`), `crowd` (`enabled`, `analysis_fps`), `ppe` (`enabled`,
`analysis_fps`), plus `source_zones` (per-source polygons).

```bash
sqlite3 data/trinetra.db "SELECT * FROM events ORDER BY id DESC LIMIT 10;"
```

---

## 6. API

| endpoint | returns |
| --- | --- |
| `GET /health` | status and total event count |
| `GET /config` | machine, rules, saved polygon, allowed people, source |
| `POST /config` | save the polygon, allowed people, source and side-path switches |
| `POST /start` | start (or restart) monitoring with the given settings |
| `POST /stop` | stop monitoring |
| `GET /state` | live status, inside count, per-worker states, message |
| `GET /stream` | MJPEG live video with the zone and alerts drawn on it |
| `GET /events?limit=100&event_type=DWELL_ALERT` | newest events first |
| `GET /events/latest` | the single newest event |
| `GET /history` | merged zone + fire/smoke rows, newest first, with `total` |
| `GET /history/export.csv` | the merged history as CSV (zone columns plus `Source`) |
| `GET /fire-alerts` | fire/smoke alerts with acked/unacked totals |
| `POST /fire-alerts/{id}/ack` | acknowledge one fire/smoke alert (repeat acks succeed) |
| `POST /fire-alerts/{id}/false-alarm` | export the alert as a retraining hard negative (also acks it, idempotent) |
| `GET /fire-alerts/{id}/snapshot` | the annotated picture saved with the alert |
| `GET /workers` | live worker states |

`POST /start` restarts cleanly if monitoring is already running, so applying a
new zone always takes effect. Sending `"points": []` clears the zone.

```bash
curl http://127.0.0.1:8090/health
curl http://127.0.0.1:8090/config
curl -X POST http://127.0.0.1:8090/start -H "Content-Type: application/json" \
     -d '{"source":"0","allowed_people":0}'
curl http://127.0.0.1:8090/state
curl -X POST http://127.0.0.1:8090/stop
```

## 7. Tests

Backend (logic, database and every API endpoint — no camera, no video, no model
download):

```bash
python -m pytest tests/ -q
```

Frontend (the dashboard component: drawing, dragging, clearing, saving, and the
SAFE / ALERT display):

```bash
cd frontend
npm test
```

## 8. Project structure

```
Prototype/
├── app.py              command line video/webcam loop
├── pipeline.py         background video runner used by the API
├── config.py           loads and validates config.yaml
├── snapshots.py        alert snapshot (JPEG) writer
├── visualize.py        all frame drawing
├── config.yaml         machine, polygon, rules (defaults)
├── requirements.txt
├── README.md
├── PORTING_NOTES.md    core logic summary for porting to another service
├── start.bat           starts backend (8090) and frontend (5174)
├── yolo11n.pt          YOLO weights (auto-downloaded on first run, git-ignored)
├── detector/
│   └── tracker.py      YOLO person detection + ByteTrack
│   ├── fire_smoke_detector.py + fire_smoke_gate.py  fire/smoke pass (v2 weights)
│   ├── crowd_detector.py + crowd.py                 crowd metrics (v1 weights)
│   └── ppe_detector.py + ppe_risk.py                PPE grading (v1 weights)
├── services/
│   ├── registry.py     side-path worker registry
│   ├── heartbeat.py    side-path liveness
│   └── vision_alerts.py  alert emission for side paths
├── zone/
│   └── danger_zone.py  polygon test + worker state machine + capacity rule
├── database/
│   └── db.py           SQLite events, fire alerts and settings storage
├── api/
│   └── server.py       FastAPI: config, start/stop, state, MJPEG stream
├── frontend/           React + Vite + TypeScript dashboard
│   ├── src/App.tsx     dashboard and polygon editor
│   ├── src/App.test.tsx  UI tests
│   ├── src/api.ts      API client
│   ├── src/types.ts    shared types
│   ├── src/main.tsx    React entry point
│   ├── src/pages/      Live Monitor, Alerts, History, Settings
│   ├── src/components/ Detection panels/paths, evidence modal, layout
│   └── src/styles.css  plain CSS
├── dashboard/
│   └── index.html      simple plain HTML dashboard (served at /legacy)
├── tests/              pytest tests for the core logic and the API
│   └── e2e/            standalone Playwright scripts for the real UI
├── tools/              count_harness.py + BASELINE.md regression reference
└── data/
    ├── trinetra.db  events and settings (created on first run)
    ├── workers.json    snapshot path used by the CLI
    ├── snapshots/      alert JPEGs
    └── videos/         uploaded and test clips
```

---

## Limitations

This prototype is a **monitoring and warning demo**. Specifically:

- It uses a **2D camera view** only. There is no depth, no multi-camera
  fusion and no 3D reconstruction.
- The danger zone is a **manually configured polygon** in pixel coordinates.
  The machine itself is never detected or identified.
- It does **not** calculate real-world physical distance. The polygon is in
  image pixels, so its real-world size depends entirely on camera position
  and angle.
- It performs **no camera calibration**, no lens distortion correction and no
  perspective correction, so the zone is only valid for the exact viewpoint
  it was drawn on. Move the camera and the zone is wrong.
- It is **not a certified safety interlock**, not a safety-rated device and
  not compliant with any functional-safety standard. It must never be the
  only means of protecting anybody from a machine.
- Detection is a pretrained COCO "person" class. It does not know about
  hi-vis vests, helmets, roles or whether a person is authorised. It cannot
  distinguish a worker from a visitor.
- A single point (the bottom-centre of the box) decides inside/outside, so a
  worker straddling the boundary will flicker between `INSIDE` and `OUTSIDE`
  as the detector jitters, resetting the dwell timer. A production version
  would smooth the point, or use the fraction of the box inside the zone.
- ByteTrack can swap or lose IDs when people overlap or walk behind each
  other, which resets dwell timers.
- One camera view only: a worker hidden behind the machine is not detected.
- A camera pointing at a wall is fine, but low light, rain, masks, loose
  clothing and people only partially in frame reduce accuracy.
- Events are written to a local SQLite file with no retry queue, so events
  are lost if the process is killed mid-write or the disk is full.
- The zone is stored in **pixels of the video frame**. If the source resolution
  or aspect ratio changes, the drawn polygon no longer matches the scene and
  must be redrawn.
- The dashboard needs both the backend and the Vite dev server; there is no
  single-command production build or installer.
- The allowed people limit is a plain count of bounding boxes whose bottom
  centre is inside the polygon, so one person jittering on the boundary can
  push the count over the limit.

## Possible next steps

Not built, deliberately: automatic machine detection, camera calibration and
homography to metric coordinates, multiple zones per machine, per-role
permissions, zone hysteresis / smoothed point tests, alert delivery (email,
SMS, webhook), long-term event analytics, and RTSP/network camera support.
