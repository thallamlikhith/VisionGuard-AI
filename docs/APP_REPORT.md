# How Your Application Works — Features & Workflow Report

**Trinetra AI (prototype)** — AI video monitoring: watches a
camera/video, detects people with YOLO11n, tracks them with ByteTrack, and
alerts when a worker enters a drawn danger zone around a machine and stays too
long / too many are present.

> **Prototype only.** This is a warning / monitoring demo, *not* a certified
> safety system. See README → Limitations.

---

## 1. Architecture

```
Video/Webcam → YOLO11n person detection (COCO class 0)
            → ByteTrack tracking → stable Worker #1, #2, ...
            → anchor point of bounding box (feet / body center)
            → point-in-polygon test vs drawn danger zone(s)
            → OUTSIDE / INSIDE / ALERT + per-worker dwell timer
            → capacity rule (inside > allowed → immediate ALERT)
            → event stored in SQLite (+ JPEG snapshot for alerts)
            → React dashboard (live MJPEG stream, controls, alerts)
```

| Part | Tech | Port |
| --- | --- | --- |
| Backend | FastAPI + Ultralytics + OpenVINO (fp16 export, CPU) | 8090 |
| Frontend | React + Vite + TypeScript | 5173 |
| Database | SQLite (`data/trinetra.db`) | — |
| Launcher | `start.bat` (starts both, opens browser) | — |
| Also | CLI mode (`app.py`, OpenCV window) + REST API (`/docs`) | — |

All open source: YOLO11/Ultralytics (AGPL-3.0), ByteTrack (MIT), OpenVINO
(Apache-2.0), OpenCV (Apache-2.0), FastAPI/React (MIT), SQLite (public domain).

---

## 2. Main workflow (operator's day)

1. **Pick a source** — *Live camera* (0 = webcam, or a video file path) or
   *Test video* (drag & drop MP4/AVI/MOV/MKV/WebM, up to 500 MB, max 10
   clips; clips loop forever).
2. **Start** → live video appears (people boxed and labelled `#id  dwell s`
   even before any zone exists).
3. **Draw zones** — *Edit zones* toolbar: **Select / Rectangle / Polygon /
   Circle / Freehand**. Click to add corners, drag to move, double-click to
   delete; Undo/Clear; a coordinate table for pixel-precise entry.
   Validation: 3–32 points, no duplicate corners, no self-crossing,
   ≥0.5% of frame area, clamped in-frame. Zones **auto-scale** if saved
   against a different resolution.
4. **Set rules** (see §3) → **Save & apply** → stored in SQLite, survives
   restart.
5. **Monitor** — status banner SAFE/ALERT, per-zone cards with live dots
   (green/amber/red), people count `inside / allowed`, longest dwell vs
   limit, workers list, perf stats (fps/ms).
6. **Alerts fire** → stored with a snapshot picture → strip under video +
   **Alerts page**.
7. **Acknowledge** alerts (Active / Acknowledged tabs), view *evidence*
   photo in a modal.

---

## 3. Settings page (the rules)

### Global defaults

Apply to every zone that has no rule of its own:

| Setting | What it does |
| --- | --- |
| **Alert mode** | `TIMER` = "After a delay in the zone" · `INSTANT` = "As soon as they enter" (disables dwell fields) |
| **Default alert after (seconds)** | dwell time — how long someone may stand in a zone before it alarms |
| **Repeat alert every (seconds)** | re-alert for a long stay (0 = never); default 10 s — avoids event spam |
| **People allowed in each zone** | capacity — `inside > allowed` → immediate ALERT with no dwell wait; **0 = anyone inside alarms** |
| **Detect person at** | `Feet` (bottom-center of box, default) or `Body center` (for desk demos) |
| **Advanced** | machine name, camera source, **Remove all zones** (with confirm) |

Settings auto-save 0.5 s after you stop typing ("Applied" confirmation).

### Per-zone rules (each zone card, expandable)

- Zone **name** (shown on the picture and in alerts)
- **People allowed** — that zone's own capacity
- **Alert after (seconds)** — that zone's own dwell time
- **Enabled** on/off toggle
- **Delete zone**; inline *Rename* also available in Live view
- "Using the default rules" vs "This zone has its own rules" — typing any
  number detaches it from the global defaults (`using_defaults` flag)
- Live figures always visible: `People 2 / 3` and `Longest dwell 4.2s / 5.0s`
  (turn red when over)

---

## 4. Detection & safety logic

- **Model**: YOLO11n (`imgsz=640`, conf 0.25), optional **OpenVINO fp16 CPU
  export** (auto-exported, smoke-tested, falls back to PyTorch). Frames
  downscaled to ≤960 px long side before inference; boxes scaled back.
- **Tracking**: persistent ByteTrack → stable worker IDs; frames between
  detections reuse last tracks (`detect_every_n`); id-less frames held
  ≤3 detections (prevents duplicate ZONE_ENTRY). **Adaptive idle cadence**
  (`detector.adaptive_idle_n`, default 3): people visible → every frame;
  empty scene → every 3rd frame.
- **Zone test**: point-in-polygon (ray casting) with **5 px margin
  hysteresis** — jitter on the boundary can't flicker the count. Alternate
  counting: `overlap` mode (bbox overlap >10%).
- **States per worker**: `OUTSIDE` (green) → `INSIDE` (amber, dwell running)
  → `ALERT` (red, dwell exceeded). Each worker gets its own timer.
- **Capacity rule**: people inside > allowed → immediate `CAPACITY_ALERT`,
  one per transition (not per frame), clears automatically when back under
  the limit.
- **Exit grace 0.6 s**: a worker must be outside that long before
  `ZONE_EXIT` is written — a single missed detection (occlusion/jitter)
  doesn't reset timers or split one visit into two.
- **Multi-zone**: several polygons; overlapping zones each get an
  independent clock (`Presence` per worker-per-zone), so two zones alert
  independently.

---

## 5. Alerts & data

**Event types** in SQLite `events` table:

| event_type | meaning |
| --- | --- |
| `ZONE_ENTRY` | worker entered the zone |
| `DWELL_ALERT` | dwell time exceeded → alert |
| `ZONE_EXIT` | worker left (after the 0.6 s exit grace) |
| `CAPACITY_ALERT` | more people inside than allowed |
| `CAPACITY_SAFE` | back under the limit → alert clears |

Rows carry worker id, machine id, timestamps, dwell seconds and zone name.
Alert rows get a **JPEG snapshot** (written off the video thread, auto-pruned).

**API** (19 endpoints): `/health`, `/config` GET+POST, `/start`, `/stop`,
`/state`, `/stream` (MJPEG), `/snapshot`, `/events` (+`/latest`,
`/{id}/ack`, `/{id}/snapshot`), `/workers`, `/videos` GET+POST+DELETE.
Zones are saved **per source**, so each camera/video keeps its own zones.

---

## 6. Pages in the dashboard

| Page | Contents |
| --- | --- |
| **Live monitor** | source picker (stopped) → video stage, drawing toolbar, status banner, alert strip, zone cards, workers list, perf stats |
| **Settings** | the global rules from §3 + Advanced (machine name, camera source, remove all zones) |
| **Alerts** | Active / Acknowledged tabs, snapshot thumbnails, evidence modal, Acknowledge button |

---

## 7. Quality & verification

- 309 backend tests + 180 frontend tests + `npx tsc -b` clean + e2e scripts
- Regression harness: `python tools/count_harness.py` vs `tools/BASELINE.md`
- `PORTING_NOTES.md` documents the 4 safety-relevant files for porting

---

## 8. Current state / known gaps

- **Performance**: pipeline runs at ~41.5 ms/frame vs 41.7 ms frame budget
  (zero headroom) → planned: adaptive detect cadence, encode-on-demand,
  INT8 OpenVINO export, thread caps (target ~26 ms/frame).
- **Not a certified safety system**: no email/SMS/push alerts,
  single-camera, no PPE detection, RTSP live input not yet first-class.
