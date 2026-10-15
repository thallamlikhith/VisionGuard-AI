"""Read-only API for the danger zone events.

    uvicorn api.server:app --reload

The video processing itself runs separately in app.py.
"""

from __future__ import annotations

import asyncio
import csv
import io
import json
import logging
import os
import re
import sys
import threading
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, Literal, Optional

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import ConfigError, load_config  # noqa: E402
from database.db import Database, DatabaseError  # noqa: E402
from detector.fire_smoke import FIRE_SMOKE_WEIGHTS_FILE  # noqa: E402
from detector.ppe_risk import PPE_EMISSION_DEFERRED  # noqa: E402
from detector.tracker import LOG_FORMAT  # noqa: E402
from pipeline import Pipeline, PipelineError  # noqa: E402
from snapshots import event_snapshot_dir  # noqa: E402
from zone.danger_zone import (  # noqa: E402
    ANCHORS,
    CAPACITY_ALERT,
    DWELL_ALERT,
    ENTRY_ALERT,
    SAFE,
    DangerZone,
    ZoneError,
    drop_repeats,
    zone_problem,
)

# The detector logs which backend it actually loaded (and why OpenVINO was
# skipped). Without a handler those lines never appear, so set one up before
# anything else runs. uvicorn only configures its own loggers, so this does not
# duplicate or silence its access log.
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

DB_PATH = os.environ.get("TRINETRA_DB", os.path.join(ROOT, "data", "trinetra.db"))
SNAPSHOT_PATH = os.environ.get("TRINETRA_SNAPSHOT", os.path.join(ROOT, "data", "workers.json"))
CONFIG_PATH = os.environ.get("TRINETRA_CONFIG", os.path.join(ROOT, "config.yaml"))
DASHBOARD = os.path.join(ROOT, "dashboard", "index.html")
# the built React UI, served here as well as by the dev server, so a built
# frontend can be opened on this port without starting a second one
FRONTEND_DIST = os.environ.get("TRINETRA_DIST", os.path.join(ROOT, "frontend", "dist"))

# Uploaded test videos live here, one video + one json sidecar each. The folder
# is under data/ so it is already gitignored and is not shipped by accident.
VIDEO_DIR = os.environ.get("TRINETRA_VIDEO_DIR", os.path.join(ROOT, "data", "videos"))
# the file types a browser may upload. Checked on the name the browser sent,
# then the file has to survive an OpenCV open/read before it is kept.
VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".webm")
# one named ceiling, so the message the user reads and the code that enforces it
# cannot drift apart
MAX_VIDEO_BYTES = 500 * 1024 * 1024  # 500 MB
MAX_VIDEOS = 10


def _limits() -> dict:
    """Upload/history caps from config.yaml, falling back to current values.

    Missing keys (or an unreadable file) change nothing: every value falls
    back to the constant above, so a partial `limits:` block can never
    regress behavior.
    """
    try:
        cfg = load_config(CONFIG_PATH).get("limits") or {}
    except ConfigError:
        cfg = {}

    def _int(key: str, fallback: int) -> int:
        try:
            return int(cfg.get(key, fallback))
        except (TypeError, ValueError):
            return fallback

    return {
        "max_upload_size_mb": _int("max_upload_size_mb", MAX_VIDEO_BYTES // (1024 * 1024)),
        "max_videos": _int("max_videos", MAX_VIDEOS),
        "history_days": _int("history_days", 14),
    }
# uploads are copied a megabyte at a time, never read whole into memory
UPLOAD_CHUNK_BYTES = 1024 * 1024
_VIDEO_ID = re.compile(r"[0-9a-f]{32}\Z")

FRONTEND_NOT_BUILT = """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <title>Trinetra AI</title>
    <style>
      body { font: 16px/1.6 "Segoe UI", system-ui, sans-serif; margin: 4rem auto;
             max-width: 40rem; color: #1f2937; background: #f8fafc; padding: 0 1.5rem; }
      h1 { font-size: 1.4rem; }
      code { background: #e2e8f0; padding: 0.1rem 0.35rem; border-radius: 4px; }
      pre { background: #0f172a; color: #e2e8f0; padding: 0.9rem 1rem; border-radius: 8px; }
    </style>
  </head>
  <body>
    <h1>The dashboard has not been built yet</h1>
    <p>The API on this port is running. To get the interface, start both
       servers from the one launcher:</p>
    <pre>start.bat                backend on 8090 plus the UI on 5174</pre>
    <p>It installs the frontend packages on the first run. To build the UI
       into this port instead, use <code>npm --prefix frontend run build</code>.
       The old plain HTML dashboard is still at <a href="/legacy">/legacy</a>.</p>
  </body>
</html>
"""

# how often the MJPEG generator wakes up when there is no new frame. It must
# not sleep longer than one frame at the target rate, or the stream feels
# laggy on a fast pipeline.
STREAM_TARGET_FPS = 15.0
STREAM_POLL_S = min(0.05, 1.0 / STREAM_TARGET_FPS)

# A stored confidence below this admits marginal person boxes that appear and
# vanish frame to frame, so the merged value is floored here rather than trusted.
MIN_CONFIDENCE = 0.1

@asynccontextmanager
async def lifespan(_: FastAPI):
    clear_default_zone()
    cleanup_invalid_zone()
    # last, so it adopts whatever survived those two checks, and only once
    migrate_zones_to_sources()
    # Loading the YOLO weights costs seconds, so it starts here in the
    # background instead of inside the first Start press. Best effort: a
    # missing download or no GPU/CPU backend must never stop the server from
    # coming up - the real load in the video thread reports any failure.
    threading.Thread(target=_warm_detector, name="detector-warmup", daemon=True).start()
    yield


def _warm_detector() -> None:
    """Load the configured detector once so the first /start reuses it."""
    try:
        from detector.tracker import PersonTracker

        try:
            settings = load_config(CONFIG_PATH)
            detector = settings["detector"]
        except ConfigError:
            detector = {"model": "yolo11n.pt", "confidence": 0.25, "imgsz": 320}
        PersonTracker(
            model_name=str(detector.get("model", "yolo11n.pt")),
            confidence=float(detector.get("confidence", 0.25)),
            imgsz=int(detector.get("imgsz", 320)),
            openvino=bool(detector.get("openvino", False)),
        )
        logging.getLogger(__name__).info("detector pre-warmed")
        # fire/smoke pre-warm only when enabled (best effort like above: a
        # missing weights file must never stop the server from coming up).
        try:
            fs = ((settings if "settings" in locals() else {}) or {}).get("fire_smoke") or {}
            if fs.get("enabled"):
                from detector.fire_smoke_detector import FireSmokeDetector

                FireSmokeDetector(
                    model_path=str(fs.get("model", "detector/models/fire_smoke_v2.pt")),
                    confidence=float(fs.get("confidence", 0.35)),
                    imgsz=int(fs.get("imgsz", 640)),
                )
                logging.getLogger(__name__).info("fire/smoke detector pre-warmed")
        except Exception as exc:
            logging.getLogger(__name__).warning("fire/smoke pre-warm skipped: %s", exc)
        # crowd pre-warm only when enabled with the model source (best effort).
        try:
            cr = ((settings if "settings" in locals() else {}) or {}).get("crowd") or {}
            if cr.get("enabled") and str(cr.get("source", "reuse")).lower() == "model":
                from detector.crowd_detector import CrowdDetector

                CrowdDetector(
                    model_path=str(cr.get("model", "detector/models/crowd_v1.pt")),
                    confidence=float(cr.get("confidence", 0.35)),
                    imgsz=int(cr.get("imgsz", 640)),
                )
                logging.getLogger(__name__).info("crowd detector pre-warmed")
        except Exception as exc:
            logging.getLogger(__name__).warning("crowd pre-warm skipped: %s", exc)
        # ppe pre-warm only when enabled (best effort).
        try:
            pp = ((settings if "settings" in locals() else {}) or {}).get("ppe") or {}
            if pp.get("enabled"):
                from detector.ppe_detector import PPEDetector

                PPEDetector(
                    model_path=str(pp.get("model", "detector/models/ppe_v1.pt")),
                    confidence=float(pp.get("confidence", 0.15)),
                    iou_threshold=float(pp.get("iou", 0.45)),
                    imgsz=int(pp.get("imgsz", 960)),
                    tta=bool(pp.get("tta", True)),
                )
                logging.getLogger(__name__).info("ppe detector pre-warmed")
        except Exception as exc:
            logging.getLogger(__name__).warning("ppe pre-warm skipped: %s", exc)
    except Exception as exc:
        logging.getLogger(__name__).warning("detector pre-warm skipped: %s", exc)


app = FastAPI(title="Trinetra AI API", version="1.2.0", lifespan=lifespan)

_pipeline: Optional[Pipeline] = None


def get_pipeline() -> Pipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = Pipeline(read_db())
    return _pipeline


# --- request bodies -------------------------------------------------------


class ZoneRule(BaseModel):
    """One zone's own name and rules.

    `points` is accepted and ignored: the shape of a zone lives in `zone_polygons`
    and only there, so there is exactly one copy of it. A rule entry is allowed to
    leave the points out, which is how the UI sends the settings and rules of the
    cards it is editing while the shapes are being redrawn.
    """

    name: str = ""
    points: Optional[list[list[int]]] = None
    # deliberately not bounded here: a rejection from the model would come back
    # as a list naming a field and an index, and the UI shows `detail` as a
    # string. validate_zone_specs() checks these instead and can say which ZONE
    # is wrong, which is the thing the user has to act on.
    allowed_people: Optional[Any] = None
    dwell_seconds: Optional[Any] = None
    enabled: Optional[bool] = None
    # true while the zone is still following the global defaults. The UI sends it
    # back so it can tell "this zone has no rule of its own" apart from "this
    # zone happens to have the same numbers as the defaults".
    using_defaults: Optional[bool] = None
    count_by: Optional[str] = None
    overlap_threshold: Optional[Any] = None


class SidePathSettings(BaseModel):
    """One side detection path (fire/smoke, crowd, PPE): switch + cadence.

    Both fields are optional so "not sent" keeps the saved value, exactly
    like the zone rules below. The fps ceiling stops a typo from asking for
    hundreds of model analyses per second."""

    enabled: Optional[bool] = None
    analysis_fps: Optional[float] = Field(default=None, gt=0, le=30)


class ZoneConfig(BaseModel):
    """Danger zones + allowed people, drawn in the UI.

    All fields are optional so that "not sent" can be told apart from
    "sent as empty" (an empty zone list is a valid state = no zone).

    `zones` is the current shape: a list of polygons, several danger areas per
    machine. `points` is the older single polygon and is still read and
    written, so an older UI keeps working against a newer backend.

    `zone_specs` is index aligned with `zones`: the name and rules of each zone.
    It is stored next to the shapes rather than inside them, so the old
    `zone_polygons` value stays readable and a save made by an older UI still
    leaves a valid single zone behind.

    frame_width / frame_height are the size of the video the points were
    drawn against. The browser does not send them yet, so bounds checking
    falls back to the running pipeline's frame size and is skipped while
    neither is known.
    """

    points: Optional[list[list[int]]] = None
    zones: Optional[list[list[list[int]]]] = None
    zone_specs: Optional[list[ZoneRule]] = None
    allowed_people: Optional[int] = Field(default=None, ge=0, le=999)
    source: Optional[str] = None
    # replay a video file when it ends. Optional: when not sent, the STORED
    # loop_video switch decides for an uploaded test video (default off, so a
    # recorded run stops rather than silently starting over); any other file
    # plays once.
    loop: Optional[bool] = None
    # the stored replay switch for test clips, shared by the Live Monitor
    # source picker and the Settings Camera card. Optional so an unsent value
    # keeps the saved one, like every field above.
    loop_video: Optional[bool] = None
    machine_name: Optional[str] = None
    mode: Optional[Literal["TIMER", "INSTANT", "LIMIT"]] = None
    dwell_seconds: Optional[float] = Field(default=None, ge=0, le=3600)
    # 0 is a real value, not a missing one: it means the alert is not repeated
    repeat_alert_seconds: Optional[float] = Field(default=None, ge=0, le=3600)
    frame_width: Optional[int] = Field(default=None, ge=0)
    frame_height: Optional[int] = Field(default=None, ge=0)
    # which point of a person has to be inside the zone. Feet is right for a
    # machine on the ground, body centre for someone sitting at a desk.
    anchor: Optional[Literal["bottom_center", "center"]] = None
    # detection confidence, 0-1. Optional so an unsent value keeps the saved one.
    confidence: Optional[float] = Field(default=None, gt=0, le=1)
    # side detection paths (Settings "Detection paths" card). Optional so an
    # unsent block keeps the saved values, like every field above.
    fire_smoke: Optional[SidePathSettings] = None
    crowd: Optional[SidePathSettings] = None
    ppe: Optional[SidePathSettings] = None
    # hazard-zone system switch and per-type fire/smoke switches. Optional so
    # an unsent block keeps the saved values.
    hazard: Optional[SidePathSettings] = None
    fire: Optional[SidePathSettings] = None
    smoke: Optional[SidePathSettings] = None


def _frame_size_for(body: ZoneConfig) -> tuple[int, int]:
    """The frame size the incoming points should be validated against."""
    if body.frame_width and body.frame_height:
        return int(body.frame_width), int(body.frame_height)
    pipeline = _pipeline
    size = getattr(pipeline, "frame_size", (0, 0)) if pipeline is not None else (0, 0)
    if size[0] > 0 and size[1] > 0:
        return size[0], size[1]
    return 0, 0


def _zone_http_error(message: str) -> HTTPException:
    """422 with a plain-string detail.

    The browser reads body.detail straight into its error message, so it has
    to stay a string rather than an object.
    """
    return HTTPException(status_code=422, detail=message)


def clamp_zone_points(points: list, width: int, height: int) -> list:
    """Nudge every corner inside the frame, and only inside the frame.

    A zone drawn on a larger frame than the one that is playing is clamped
    rather than rejected: the corner that hangs off the edge is pulled to the
    edge and the shape that is left is still the one the operator drew. With no
    known frame size (0x0) the points are only turned into ints, so a caller
    that does not know the video yet is not clamped against a made-up picture.
    """
    if width <= 0 or height <= 0:
        return [(int(x), int(y)) for x, y in points]
    return [
        (min(width, max(0, int(x))), min(height, max(0, int(y))))
        for x, y in points
    ]


def validate_zone(points: list, width: int, height: int) -> None:
    """Raise a 422 naming the problem, or return quietly when the zone is fine.

    An empty list is allowed: it is how the UI clears the zone. Anything else
    has to be a polygon that could actually enclose somebody - the right
    number of corners, inside the frame, no repeated corners, not crossing over
    itself and big enough to be a real area. The same rules run in the UI, so
    a zone that reaches this point has already been checked.
    """
    if not points:
        return

    for index, point in enumerate(points):
        if len(point) != 2:
            raise _zone_http_error(
                f"Point {index} is invalid: {point}. Expected [x, y]."
            )

    problem = zone_problem(clamp_zone_points(points, width, height), width, height)
    if problem:
        raise _zone_http_error(problem)


def validate_zones(polygons: list, width: int, height: int) -> list[list[list[int]]]:
    """Check every polygon of a multi-zone save, or raise a 422 for the first
    one that is unusable.

    An empty list is allowed: it is how the UI clears all zones. The zone
    number is only added when there is more than one zone, so a single zone -
    which is what the older UI sends - still gets the plain message it always
    did.
    """
    if not polygons:
        return []

    numbered = len(polygons) > 1
    for index, polygon in enumerate(polygons):
        for corner, point in enumerate(polygon):
            if len(point) != 2:
                raise _zone_http_error(
                    f"Zone {index + 1}, point {corner} is invalid: {point}. Expected [x, y]."
                    if numbered
                    else f"Point {corner} is invalid: {point}. Expected [x, y]."
                )

    checked: list[list[list[int]]] = []
    for index, polygon in enumerate(polygons):
        points = clamp_zone_points(polygon, width, height)
        problem = zone_problem(points, width, height)
        if problem:
            raise _zone_http_error(f"Zone {index + 1}: {problem}" if numbered else problem)
        # what is stored is the shape after the repeat merge, so the zone on
        # screen, the one in the database and the one being monitored are one
        # and the same set of corners
        checked.append([[x, y] for x, y in drop_repeats(points)])
    return checked


def zones_from_body(body: ZoneConfig, current: Optional[list] = None) -> Optional[list]:
    """Which polygons this request means.

    `zones` wins when sent. An explicit `points` is one zone, which is how the
    older single-zone UI keeps working. When neither is sent the zones that are
    already stored are kept.
    """
    if body.zones is not None:
        return body.zones
    if body.points is not None:
        return [body.points] if body.points else []
    return current


def validate_zone_specs(specs: list[dict], zone_count: int) -> list[dict]:
    """Check each zone's rules, naming the zone in any message.

    The rules are bounded per zone rather than per request, so "allowed_people
    must be 0-999" has to say WHICH zone: with several cards on screen that is
    the difference between a message the user can act on and one they cannot.
    Extra entries past the last polygon are dropped rather than saved, because a
    rule with no shape is not a zone anybody can be standing in.
    """
    checked: list[dict] = []
    for index, spec in enumerate(specs[:zone_count]):
        name = spec.get("name") or f"Zone {index + 1}"
        allowed = spec.get("allowed_people", 0)
        dwell = spec.get("dwell_seconds", 5.0)
        try:
            allowed = int(allowed)
        except (TypeError, ValueError):
            raise _zone_http_error(f"{name}: allowed people must be a whole number, got {allowed}")
        try:
            dwell = float(dwell)
        except (TypeError, ValueError):
            raise _zone_http_error(f"{name}: dwell time must be a number, got {dwell}")
        if allowed < 0:
            raise _zone_http_error(f"{name}: allowed people cannot be negative")
        if dwell < 0:
            raise _zone_http_error(f"{name}: dwell time cannot be negative")
        checked.append(
            {
                "name": str(name),
                "allowed_people": allowed,
                "dwell_seconds": dwell,
                "enabled": bool(spec.get("enabled", True)),
                "using_defaults": bool(spec.get("using_defaults", False)),
            }
        )
    return checked


def int_zones(polygons: list) -> list[list[list[int]]]:
    """Normalise a zone list to plain ints, or raise a 400.

    /start does not re-run the full 422 validation of /config: the zone was
    already checked when it was saved, and the pipeline itself rejects a zone
    it cannot use, which has always been a 400 from this endpoint.
    """
    try:
        return [[[x, y] for x, y in polygon] for polygon in DangerZone(polygons).polygons]
    except (ZoneError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def iso(value: Optional[float]) -> Optional[str]:
    if value is None:
        return None
    return datetime.fromtimestamp(value, tz=timezone.utc).isoformat()


# only these three kinds are alerts an operator can act on; entry/exit rows are
# history and are never acked
ACKABLE_EVENTS = (ENTRY_ALERT, DWELL_ALERT, CAPACITY_ALERT)


def row_to_event(row: dict) -> dict:
    return {
        "id": row["id"],
        "worker_id": row["worker_id"],
        "machine_id": row["machine_id"],
        "event_type": row["event_type"],
        "reason": row.get("reason"),
        "entered_at": iso(row["entered_at"]),
        "alert_at": iso(row["alert_at"]),
        "dwell_seconds": row["dwell_seconds"],
        "zone_index": row.get("zone_index"),
        "zone_name": row.get("zone_name"),
        "inside": row.get("inside"),
        # the file name itself stays on the server; the UI only needs to know
        # whether there is a picture to fetch
        "has_snapshot": bool(row.get("snapshot_path")),
        "acknowledged": bool(row.get("acknowledged")),
        "acknowledged_at": iso(row.get("acknowledged_at")),
    }


def read_db() -> Database:
    try:
        return Database(DB_PATH)
    except DatabaseError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/health")
def health() -> dict:
    try:
        database = read_db()
        total = database.count()
    except HTTPException as exc:
        return {"status": "degraded", "database": str(exc.detail), "events": 0}
    return {"status": "ok", "database": os.path.basename(DB_PATH), "events": total}


@app.get("/events")
def events(
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    event_type: Optional[list[str]] = Query(
        None, description="repeatable: ZONE_ENTRY | ENTRY_ALERT | DWELL_ALERT | ZONE_EXIT | CAPACITY_ALERT"
    ),
    acknowledged: Optional[bool] = Query(None),
    worker_id: Optional[int] = Query(None),
    zone: Optional[str] = Query(None, description='zone index ("0") or zone name ("Kiln Platform")'),
    date_from: Optional[str] = Query(None, description="plant-local YYYY-MM-DD, inclusive"),
    date_to: Optional[str] = Query(None, description="plant-local YYYY-MM-DD, inclusive"),
) -> dict:
    """One page of history rows, newest first, plus the total the current
    filters match ignoring paging. Dates are plant-local days, both ends
    inclusive."""
    _check_history_dates(date_from, date_to)
    database = read_db()
    try:
        # every filter is bound as a parameter in list_events, so repeatable
        # event_type values cannot become SQL
        rows = database.list_events(
            limit=limit,
            offset=offset,
            event_types=event_type,
            acknowledged=acknowledged,
            worker_id=worker_id,
            zone=zone,
            date_from=date_from,
            date_to=date_to,
        )
        total = database.count_events(
            event_types=event_type,
            acknowledged=acknowledged,
            worker_id=worker_id,
            zone=zone,
            date_from=date_from,
            date_to=date_to,
        )
    except DatabaseError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"count": len(rows), "total": total, "events": [row_to_event(row) for row in rows]}


def _check_history_dates(date_from: Optional[str], date_to: Optional[str]) -> None:
    """422 on a date that is not a plant-local YYYY-MM-DD, before SQLite ever
    sees it."""
    for day in (date_from, date_to):
        if day is None or day == "":
            continue
        try:
            datetime.strptime(day.strip(), "%Y-%m-%d")
        except ValueError:
            raise HTTPException(
                status_code=422, detail=f"bad date {day!r}: expected YYYY-MM-DD"
            ) from None


EVENT_CSV_LABELS = {
    "ENTRY_ALERT": "Entry alert",
    "DWELL_ALERT": "Dwell alert",
    "CAPACITY_ALERT": "Capacity alert",
    "ZONE_ENTRY": "Zone entry",
    "ZONE_EXIT": "Zone exit",
    "CAPACITY_SAFE": "Capacity normal",
}


def _csv_worker(row: dict) -> str:
    """Who a CSV row is about, in the same words the UI uses."""
    if row.get("event_type") == "CAPACITY_ALERT":
        zone = row.get("zone_name") or "Zone"
        inside = row.get("inside")
        if inside is None:
            return str(zone)
        count = int(inside)
        return f"{zone} · {count} {'person' if count == 1 else 'people'}"
    worker_id = row.get("worker_id")
    if worker_id is None or int(worker_id) < 0:
        return ""
    return f"Worker #{int(worker_id)}"


@app.get("/events/export.csv")
def export_events_csv(
    event_type: Optional[list[str]] = Query(
        None, description="repeatable: ZONE_ENTRY | ENTRY_ALERT | DWELL_ALERT | ZONE_EXIT | CAPACITY_ALERT"
    ),
    acknowledged: Optional[bool] = Query(None),
    worker_id: Optional[int] = Query(None),
    zone: Optional[str] = Query(None, description='zone index ("0") or zone name ("Kiln Platform")'),
    date_from: Optional[str] = Query(None, description="plant-local YYYY-MM-DD, inclusive"),
    date_to: Optional[str] = Query(None, description="plant-local YYYY-MM-DD, inclusive"),
):
    """The filtered history as a CSV download: same filters as /events, every
    matching row, newest first. Times are plant-local (IST)."""
    from database.db import IST

    _check_history_dates(date_from, date_to)
    database = read_db()
    try:
        rows = database.list_events(
            limit=100000,
            event_types=event_type,
            acknowledged=acknowledged,
            worker_id=worker_id,
            zone=zone,
            date_from=date_from,
            date_to=date_to,
        )
    except DatabaseError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    def ist(stamp) -> str:
        if stamp is None:
            return ""
        try:
            moment = datetime.fromtimestamp(float(stamp), tz=IST)
        except (TypeError, ValueError, OSError):
            return ""
        return moment.strftime("%Y-%m-%d %H:%M:%S")

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(
        ["Time", "Worker", "Zone", "Event", "Dwell (s)", "People inside", "Acknowledged", "Acknowledged at"]
    )
    for row in rows:
        dwell = row.get("dwell_seconds")
        writer.writerow(
            [
                ist(row.get("alert_at") if row.get("alert_at") is not None else row.get("entered_at")),
                _csv_worker(row),
                row.get("zone_name") or "",
                EVENT_CSV_LABELS.get(row.get("event_type"), row.get("event_type") or ""),
                "" if dwell is None else dwell,
                row.get("inside") if row.get("inside") is not None else "",
                "Yes" if row.get("acknowledged") else "No",
                ist(row.get("acknowledged_at")),
            ]
        )
    today = datetime.now(tz=IST).date().isoformat()
    body = output.getvalue().encode("utf-8")
    return StreamingResponse(
        io.BytesIO(body),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="history_{today}.csv"'},
    )


# ---------------------------------------------------------------------------
# Merged history: zone events + fire/smoke alerts in one log.
#
# /events and /events/export.csv are untouched (zone rows only, exactly as
# before). /history merges both tables newest-first with one total, and
# /history/export.csv downloads the same merged rows. The type filter
# accepts the zone kinds plus FIRE and SMOKE; a zone value matches zone
# rows by index-or-name and fire rows by their stored source label.
# ---------------------------------------------------------------------------

HISTORY_EVENT_TYPES = (
    "ZONE_ENTRY | ENTRY_ALERT | DWELL_ALERT | ZONE_EXIT | CAPACITY_ALERT | FIRE | SMOKE"
)

FIRE_CSV_LABELS = {"FIRE": "Fire", "SMOKE": "Smoke"}


def _history_row_to_ui(row: dict) -> dict:
    """One merged row in UI shape: its table's shape plus "kind"."""
    if row.get("kind") == "fire":
        return {"kind": "fire", **row_to_fire_alert(row)}
    zone = dict(row)
    zone.pop("kind", None)
    return {"kind": "zone", **row_to_event(zone)}


@app.get("/history")
def history(
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    event_type: Optional[list[str]] = Query(None, description=f"repeatable: {HISTORY_EVENT_TYPES}"),
    acknowledged: Optional[bool] = Query(None),
    false_alarm: Optional[bool] = Query(None, description="true = flagged fire rows only"),
    worker_id: Optional[int] = Query(None),
    zone: Optional[str] = Query(None, description='zone index ("0"), zone name, or fire source label'),
    date_from: Optional[str] = Query(None, description="plant-local YYYY-MM-DD, inclusive"),
    date_to: Optional[str] = Query(None, description="plant-local YYYY-MM-DD, inclusive"),
) -> dict:
    """One page of merged history rows, newest first, plus the total the
    current filters match ignoring paging. Dates are plant-local days, both
    ends inclusive."""
    _check_history_dates(date_from, date_to)
    database = read_db()
    try:
        rows = database.list_history(
            limit=limit,
            offset=offset,
            event_types=event_type,
            acknowledged=acknowledged,
            false_alarm=false_alarm,
            worker_id=worker_id,
            zone=zone,
            date_from=date_from,
            date_to=date_to,
        )
        total = database.count_history(
            event_types=event_type,
            acknowledged=acknowledged,
            false_alarm=false_alarm,
            worker_id=worker_id,
            zone=zone,
            date_from=date_from,
            date_to=date_to,
        )
    except DatabaseError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"count": len(rows), "total": total, "rows": [_history_row_to_ui(row) for row in rows]}


@app.get("/history/export.csv")
def export_history_csv(
    event_type: Optional[list[str]] = Query(None, description=f"repeatable: {HISTORY_EVENT_TYPES}"),
    acknowledged: Optional[bool] = Query(None),
    false_alarm: Optional[bool] = Query(None, description="true = flagged fire rows only"),
    worker_id: Optional[int] = Query(None),
    zone: Optional[str] = Query(None, description='zone index ("0"), zone name, or fire source label'),
    date_from: Optional[str] = Query(None, description="plant-local YYYY-MM-DD, inclusive"),
    date_to: Optional[str] = Query(None, description="plant-local YYYY-MM-DD, inclusive"),
):
    """The filtered merged history as a CSV download: same filters as
    /history, every matching row, newest first. The zone columns are exactly
    the /events CSV columns with a Source column after Zone; fire rows fill
    Source with their stored label. Times are plant-local (IST)."""
    from database.db import IST

    _check_history_dates(date_from, date_to)
    database = read_db()
    try:
        rows = database.list_history(
            limit=100000,
            event_types=event_type,
            acknowledged=acknowledged,
            false_alarm=false_alarm,
            worker_id=worker_id,
            zone=zone,
            date_from=date_from,
            date_to=date_to,
        )
    except DatabaseError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    def ist(stamp) -> str:
        if stamp is None:
            return ""
        try:
            moment = datetime.fromtimestamp(float(stamp), tz=IST)
        except (TypeError, ValueError, OSError):
            return ""
        return moment.strftime("%Y-%m-%d %H:%M:%S")

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(
        ["Time", "Worker", "Zone", "Source", "Event", "Dwell (s)", "People inside", "Acknowledged", "Acknowledged at"]
    )
    for row in rows:
        if row.get("kind") == "fire":
            kind = str(row.get("alert_type") or "").upper()
            writer.writerow(
                [
                    ist(row.get("reported_at")),
                    "",
                    "",
                    row.get("source_label") or "",
                    FIRE_CSV_LABELS.get(kind, row.get("alert_type") or ""),
                    "",
                    "",
                    "Yes" if row.get("acknowledged") else "No",
                    ist(row.get("acknowledged_at")),
                ]
            )
            continue
        dwell = row.get("dwell_seconds")
        writer.writerow(
            [
                ist(row.get("alert_at") if row.get("alert_at") is not None else row.get("entered_at")),
                _csv_worker(row),
                row.get("zone_name") or "",
                "",
                EVENT_CSV_LABELS.get(row.get("event_type"), row.get("event_type") or ""),
                "" if dwell is None else dwell,
                row.get("inside") if row.get("inside") is not None else "",
                "Yes" if row.get("acknowledged") else "No",
                ist(row.get("acknowledged_at")),
            ]
        )
    today = datetime.now(tz=IST).date().isoformat()
    body = output.getvalue().encode("utf-8")
    return StreamingResponse(
        io.BytesIO(body),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="history_{today}.csv"'},
    )


@app.get("/events/filters")
def event_filters() -> dict:
    """Distinct workers and zones for the history dropdowns."""
    database = read_db()
    try:
        return database.filter_options()
    except DatabaseError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/events/latest")
def latest(event_type: Optional[str] = Query(None)) -> dict:
    database = read_db()
    try:
        row = database.latest_event(event_type=event_type)
    except DatabaseError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    if row is None:
        return {"event": None}
    return {"event": row_to_event(row)}


@app.get("/events/daily")
def events_daily(
    days: Optional[int] = Query(None, ge=1, le=31),
    zone: Optional[str] = Query(None),
    worker_id: Optional[int] = Query(None),
) -> dict:
    """One row per day for the last `days` days: entry vs dwell vs capacity
    vs fire vs smoke.

    Counts incidents, not beeps: a dwell visit that re-alerts every few
    seconds is one incident per person per zone visit (see
    Database.alert_incidents_by_day). Days are the plant's local dates, and
    days with no alerts come back as zeros rather than missing rows, so the
    chart never has to guess. `zone` is a zone index ("0") or a zone name
    ("Kiln Platform"); `worker_id` needs no explanation. An absent `days`
    reads the configured history window (14).
    """
    if days is None:
        days = _limits()["history_days"]
    database = read_db()
    try:
        counts = database.alert_incidents_by_day(days=days, zone=zone, worker_id=worker_id)
        fire_counts = database.fire_counts_by_day(days=days, zone=zone, worker_id=worker_id)
    except DatabaseError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    today = datetime.now().date()
    out = []
    for back in range(days - 1, -1, -1):
        day = (today - timedelta(days=back)).isoformat()
        out.append(
            {
                "date": day,
                "entry": int(counts.get((day, "ENTRY_ALERT"), 0)),
                "dwell": int(counts.get((day, "DWELL_ALERT"), 0)),
                "capacity": int(counts.get((day, "CAPACITY_ALERT"), 0)),
                "fire": int(fire_counts.get((day, "FIRE"), 0)),
                "smoke": int(fire_counts.get((day, "SMOKE"), 0)),
            }
        )
    return {"days": out, "days_requested": days}


@app.post("/events/ack-all")
def acknowledge_all_events(
    event_type: Optional[list[str]] = Query(
        None, description="repeatable alert kind to limit the ack to; entries and exits never ack"
    ),
    zone: Optional[str] = Query(None, description='zone index ("0") or zone name ("Kiln Platform")'),
) -> dict:
    """Mark every unacknowledged alert acknowledged in one query, all sharing
    one timestamp. Only DWELL / ENTRY / CAPACITY alerts are touched - zone
    entries, exits and return-to-safe rows stay as they are."""
    database = read_db()
    try:
        count = database.acknowledge_all(time.time(), event_types=event_type, zone=zone)
    except DatabaseError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"acknowledged": count}


@app.post("/events/{event_id}/ack")
def acknowledge_event(event_id: int) -> dict:
    """Mark one alert acknowledged.

    Only a real alert can be acked: an entry/exit or a CAPACITY_SAFE row is a
    record of something that happened, not a prompt, so refusing it keeps the
    Alerts page honest. A repeat ack succeeds and leaves the row as it is.
    """
    database = read_db()
    try:
        row = database.get_event(event_id)
    except DatabaseError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    if row is None:
        raise HTTPException(status_code=404, detail=f"no event with id {event_id}")
    if row["event_type"] not in ACKABLE_EVENTS:
        raise HTTPException(
            status_code=400,
            detail=f"{row['event_type']} is not an alert and cannot be acknowledged",
        )

    if not row.get("acknowledged"):
        try:
            database.acknowledge_event(event_id, time.time())
        except DatabaseError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
    updated = database.get_event(event_id)
    return {"event": row_to_event(updated if updated is not None else row)}


@app.get("/events/{event_id}/snapshot")
def event_snapshot(event_id: int) -> Response:
    """The picture saved with an alert, or 404 when there is none.

    The database holds a bare file name and it is always looked up inside the
    snapshot folder, so a stored value can never read a file from elsewhere.
    """
    database = read_db()
    try:
        row = database.get_event(event_id)
    except DatabaseError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    name = os.path.basename(str(row.get("snapshot_path") or "")) if row else ""
    if not name:
        raise HTTPException(status_code=404, detail="no snapshot for this event")

    path = os.path.join(event_snapshot_dir(), name)
    if not _inside(event_snapshot_dir(), name):
        raise HTTPException(status_code=404, detail="no snapshot for this event")
    try:
        with open(path, "rb") as handle:
            data = handle.read()
    except OSError as exc:
        raise HTTPException(status_code=404, detail="no snapshot for this event") from exc
    return Response(content=data, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


# ---------------------------------------------------------------------------
# Fire/smoke alerts.
#
# Own table, own routes: zone rows, counts, history and CSV exports behave
# exactly as before. Rows carry stable ids and their ack state, so the UI
# needs no sidecar files.
# ---------------------------------------------------------------------------


def row_to_fire_alert(row: dict) -> dict:
    """One fire/smoke row in UI shape (timestamps as ISO-8601)."""
    reported = row.get("reported_at")
    acked_at = row.get("acknowledged_at")
    return {
        "id": row["id"],
        "alert_type": row["alert_type"],
        "severity": row["severity"],
        "confidence": row["confidence"],
        "bbox_xyxy": row["bbox_xyxy"],
        "captured_at": row["captured_at"],
        "reported_at": (
            datetime.fromtimestamp(reported, tz=timezone.utc).isoformat()
            if reported is not None else None
        ),
        "has_snapshot": bool(row.get("snapshot_path")),
        "acknowledged": bool(row.get("acknowledged")),
        "acknowledged_at": (
            datetime.fromtimestamp(acked_at, tz=timezone.utc).isoformat()
            if acked_at is not None else None
        ),
        "false_alarm": bool(row.get("false_alarm")),
        "false_alarm_at": (
            datetime.fromtimestamp(fa_at, tz=timezone.utc).isoformat()
            if (fa_at := row.get("false_alarm_at")) is not None else None
        ),
        # Stored at alert time: the single source's label, so a merged log
        # can say where a fire row came from without a join.
        "source_label": row.get("source_label"),
    }


@app.get("/fire-alerts")
def fire_alerts(acknowledged: Optional[bool] = None, limit: int = 100,
                offset: int = 0) -> dict:
    """Fire/smoke alert rows, newest first, with true totals for both tabs."""
    database = read_db()
    try:
        rows = database.list_fire_alerts(acknowledged=acknowledged,
                                         limit=limit, offset=offset)
        unacked = database.count_fire_alerts(acknowledged=False)
        acked = database.count_fire_alerts(acknowledged=True)
    except DatabaseError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {
        "alerts": [row_to_fire_alert(row) for row in rows],
        "total": unacked + acked,
        "unacked": unacked,
        "acked": acked,
    }


@app.post("/fire-alerts/{alert_id}/ack")
def acknowledge_fire_alert(alert_id: int) -> dict:
    """Mark one fire/smoke alert acknowledged. A repeat ack succeeds."""
    database = read_db()
    try:
        row = database.get_fire_alert(alert_id)
    except DatabaseError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    if row is None:
        raise HTTPException(status_code=404, detail=f"no fire alert with id {alert_id}")
    if not row.get("acknowledged"):
        try:
            database.acknowledge_fire_alert(alert_id, time.time())
        except DatabaseError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
    updated = database.get_fire_alert(alert_id)
    return {"alert": row_to_fire_alert(updated if updated is not None else row)}


@app.get("/fire-alerts/{alert_id}/snapshot")
def fire_alert_snapshot(alert_id: int) -> Response:
    """The annotated picture saved with a fire/smoke alert, or 404.

    The database holds a bare file name and it is always looked up inside
    the fire snapshot folder, so a stored value can never read a file from
    elsewhere.
    """
    from snapshots import fire_snapshot_dir

    database = read_db()
    try:
        row = database.get_fire_alert(alert_id)
    except DatabaseError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    name = os.path.basename(str((row or {}).get("snapshot_path") or ""))
    if not name:
        raise HTTPException(status_code=404, detail="no snapshot for this alert")

    path = os.path.join(fire_snapshot_dir(), name)
    if not _inside(fire_snapshot_dir(), name):
        raise HTTPException(status_code=404, detail="no snapshot for this alert")
    try:
        with open(path, "rb") as handle:
            data = handle.read()
    except OSError as exc:
        raise HTTPException(status_code=404, detail="no snapshot for this alert") from exc
    return Response(content=data, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@app.post("/fire-alerts/{alert_id}/false-alarm")
def mark_fire_false_alarm(alert_id: int) -> dict:
    """Export one fire/smoke alert as a retraining hard negative.

    Copies the clean (pre-overlay) frame to ``data/hard_negatives/`` with a
    JSON sidecar, flags the row (which also acknowledges it), and answers
    with both file names. Idempotent: marking twice exports once — the
    second call answers ``duplicate: true`` with no new files. 404 for an
    unknown id, 409 with a plain message when the clean frame is gone
    (never stored, or pruned since); nothing here answers 500.
    """
    from snapshots import export_hard_negative, fire_snapshot_dir
    from snapshots import HardNegativeError

    database = read_db()
    try:
        row = database.get_fire_alert(alert_id)
    except DatabaseError as exc:
        raise HTTPException(status_code=409, detail=f"cannot read alert {alert_id}: {exc}") from exc
    if row is None:
        raise HTTPException(status_code=404, detail=f"no fire alert with id {alert_id}")
    if row.get("false_alarm"):
        return {"alert": row_to_fire_alert(row), "exported": None, "duplicate": True}

    clean_name = os.path.basename(str(row.get("clean_snapshot_path") or ""))
    if not clean_name:
        raise HTTPException(
            status_code=409,
            detail=f"no clean frame stored for alert {alert_id}: nothing to export",
        )
    clean_src = os.path.join(fire_snapshot_dir(), clean_name)
    if not _inside(fire_snapshot_dir(), clean_name) or not os.path.isfile(clean_src):
        raise HTTPException(
            status_code=409,
            detail=f"clean frame for alert {alert_id} is gone (pruned since the alert)",
        )

    metadata = {
        "alert_id": row["id"],
        "type": row["alert_type"],
        "captured_at": row["captured_at"],
        "reported_at": row["reported_at"],
        "box": row["bbox_xyxy"],
        "confidence": row["confidence"],
        "weights": FIRE_SMOKE_WEIGHTS_FILE,
    }
    try:
        jpg_name, json_name = export_hard_negative(
            row["alert_type"], row["reported_at"], int(row["id"]), clean_src, metadata
        )
    except HardNegativeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    try:
        marked = database.mark_fire_false_alarm(alert_id, time.time())
    except DatabaseError as exc:
        raise HTTPException(status_code=409, detail=f"cannot mark alert {alert_id}: {exc}") from exc
    if not marked:
        # lost a race with a concurrent mark: re-read, still idempotent
        row = database.get_fire_alert(alert_id) or row
        return {"alert": row_to_fire_alert(row), "exported": None, "duplicate": True}
    updated = database.get_fire_alert(alert_id) or row
    return {
        "alert": row_to_fire_alert(updated),
        "exported": {"image": jpg_name, "metadata": json_name},
        "duplicate": False,
    }


@app.get("/workers")
def workers() -> dict:
    """Live worker states, written by app.py to data/workers.json."""
    try:
        with open(SNAPSHOT_PATH, "r", encoding="utf-8") as handle:
            snapshot = json.load(handle)
    except FileNotFoundError:
        return {"running": False, "machine": None, "workers": []}
    except (OSError, json.JSONDecodeError):
        return {"running": False, "machine": None, "workers": []}
    return {"running": True, **snapshot}


def _inside(folder: str, name: str) -> bool:
    """True when `name` really sits inside `folder`.

    Used before a browser-supplied path is read, so "/assets/../config.yaml"
    cannot be used to fetch a file that is not part of the built UI.
    """
    if not name or os.path.isabs(name):
        return False
    target = os.path.normpath(os.path.join(folder, name))
    return os.path.commonpath([os.path.normpath(folder), target]) == os.path.normpath(folder)


@app.get("/legacy")
def dashboard() -> FileResponse:
    """The old plain HTML dashboard, kept so nothing that used "/" is lost."""
    if not os.path.isfile(DASHBOARD):
        raise HTTPException(status_code=404, detail="dashboard/index.html not found")
    return FileResponse(DASHBOARD)


@app.get("/", include_in_schema=False)
def frontend() -> Response:
    """The built React UI, when there is one, on the API port as well.

    The normal way to the dashboard is the Vite dev server that start.bat
    starts. This route serves `frontend/dist` when it has been built, so a
    production build can be opened without a second server.
    """
    index = os.path.join(FRONTEND_DIST, "index.html")
    if os.path.isfile(index):
        return FileResponse(index)
    return HTMLResponse(FRONTEND_NOT_BUILT)


@app.get("/assets/{name:path}", include_in_schema=False)
def frontend_asset(name: str) -> Response:
    """The hashed JS/CSS the built UI loads."""
    assets = os.path.join(FRONTEND_DIST, "assets")
    if not _inside(assets, name):
        raise HTTPException(status_code=404, detail="not found")
    target = os.path.normpath(os.path.join(assets, name))
    if not os.path.isfile(target):
        raise HTTPException(status_code=404, detail="not found")
    return FileResponse(target)


# ---------------------------------------------------------------------------
# Web UI support: settings, video stream, start/stop
# ---------------------------------------------------------------------------


ZONE_REFERENCE_KEY = "zone_reference"
ZONE_POLYGONS_KEY = "zone_polygons"
LEGACY_ZONE_KEY = "zone_points"
# names and rules, index aligned with ZONE_POLYGONS_KEY. Stored apart from the
# shapes on purpose: a database written before this key existed stays valid, and
# the shapes are never duplicated into a second place that could disagree.
ZONE_SPECS_KEY = "zone_specs"

# Zones belong to the source they were drawn on: a webcam and a video file have
# different frame sizes and different danger areas, and one must never be placed
# over the other. `source_zones` is a json object keyed by a stable source id,
# each entry holding that source's polygons, its rules and the frame size they
# were drawn against. The old global keys above mirror the ACTIVE source, so
# everything that used to read a single zone set keeps working unchanged.
SOURCE_ZONES_KEY = "source_zones"
# Set once the zones that predate per-source storage have been adopted, so the
# migration can never run a second time and overwrite a later save.
ZONES_MIGRATED_KEY = "zones_migrated"


def source_key(source: str) -> str:
    """A stable id for a video source: webcam index, or bare video file name.

    The camera field can be "0", " 0" or empty and mean the same camera, so only
    the index is kept. A video is keyed by its bare file name, never the full
    path, so moving the install folder or a different drive letter does not
    orphan the zones that belong to it.
    """
    text = str(source or "").strip()
    if not text:
        text = "0"
    if text.isdigit():
        return f"cam:{int(text)}"
    return f"video:{os.path.basename(text)}"


def _parse_source_zones(stored: dict) -> dict:
    """The per-source map, ignoring anything that is not a json object."""
    raw = stored.get(SOURCE_ZONES_KEY)
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _entry_polygons(entry: dict) -> list[list[list[int]]]:
    """The polygons of one source entry, as plain ints."""
    polygons = entry.get("polygons") if isinstance(entry, dict) else None
    if not isinstance(polygons, list):
        return []
    result: list[list[list[int]]] = []
    for polygon in polygons:
        if not polygon:
            continue
        try:
            result.append([[int(x), int(y)] for x, y in polygon])
        except (TypeError, ValueError):
            continue
    return result


def _entry_specs(entry: dict, polygons: list) -> list[dict]:
    """One rule dictionary per polygon, no more and no fewer."""
    specs = entry.get("specs") if isinstance(entry, dict) else None
    if not isinstance(specs, list):
        specs = []
    return [spec for spec in specs if isinstance(spec, dict)][: len(polygons)]


def _entry_reference(entry: dict) -> list[int]:
    """The frame size the entry was drawn on, or [0, 0] when it is unreadable."""
    reference = entry.get("reference") if isinstance(entry, dict) else None
    if isinstance(reference, (list, tuple)) and len(reference) == 2:
        try:
            return [int(reference[0]), int(reference[1])]
        except (TypeError, ValueError):
            pass
    return [0, 0]


def source_zones_for(stored: dict, source: str) -> tuple[list, list, list]:
    """The zones, rules and frame size saved for one source."""
    entry = _parse_source_zones(stored).get(source_key(source))
    if not isinstance(entry, dict):
        return [], [], [0, 0]
    polygons = _entry_polygons(entry)
    return polygons, _entry_specs(entry, polygons), _entry_reference(entry)

# The polygon config.yaml used to ship with, in the pixel coordinates it was
# written for. Anything equal to it (or the same shape on another frame size)
# is the old default and not a zone anybody drew.
OLD_DEFAULT_POLYGON = ((400, 200), (700, 200), (760, 500), (350, 500))
OLD_DEFAULT_STRAY_CORNER = (351, 228)
OLD_DEFAULT_FRAME = (1280, 720)
# a scaled or clamped copy can be off by a pixel or two from the exact numbers
OLD_DEFAULT_TOLERANCE = 2.0


def _is_old_default(polygon, reference: tuple[int, int]) -> bool:
    """True for the shipped default, however it was scaled or clamped."""
    try:
        points = [(int(x), int(y)) for x, y in polygon]
    except (TypeError, ValueError):
        return False
    if len(points) not in (len(OLD_DEFAULT_POLYGON), len(OLD_DEFAULT_POLYGON) + 1):
        return False

    width, height = reference if all(reference) else OLD_DEFAULT_FRAME
    try:
        width = max(1, int(width))
        height = max(1, int(height))
    except (TypeError, ValueError):
        width, height = OLD_DEFAULT_FRAME

    def scale(point: tuple[int, int]) -> tuple[float, float]:
        return (point[0] * width / OLD_DEFAULT_FRAME[0], point[1] * height / OLD_DEFAULT_FRAME[1])

    expected = [scale(point) for point in OLD_DEFAULT_POLYGON]
    if len(points) == len(expected) + 1:
        expected.append(scale(OLD_DEFAULT_STRAY_CORNER))

    # every corner has to sit on one of the default's corners, so a zone the
    # user really drew is left alone even when it shares a corner or two
    return all(
        any(abs(px - ex) <= OLD_DEFAULT_TOLERANCE and abs(py - ey) <= OLD_DEFAULT_TOLERANCE for ex, ey in expected)
        for px, py in points
    )


def stored_zones(stored: dict) -> list[list[list[int]]]:
    """The saved zones, whichever key they were written under.

    Newer saves use `zone_polygons` (a list of polygons); older ones only have
    `zone_points` (one polygon). Reading both is what keeps an existing
    database usable without a migration step.
    """
    raw = stored.get(ZONE_POLYGONS_KEY)
    if raw:
        try:
            polygons = json.loads(raw)
            if polygons and all(polygon and isinstance(polygon[0], (list, tuple)) for polygon in polygons):
                return [[[int(x), int(y)] for x, y in polygon] for polygon in polygons]
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

    legacy = stored.get(LEGACY_ZONE_KEY)
    if not legacy:
        return []
    try:
        points = json.loads(legacy)
        return [[[int(x), int(y)] for x, y in points]] if points else []
    except (json.JSONDecodeError, TypeError, ValueError):
        return []


def stored_zone_specs(stored: dict, polygons: list, allowed: int, dwell: float) -> list[dict]:
    """The saved rules for each zone, upgraded on read to match the saved shapes.

    The polygons are the one place a zone's shape is stored. A zone with no saved
    rule of its own is reported as `Zone N` following the global defaults, so a
    database saved before zones had names still opens as one working zone instead
    of as an error, and the UI can show which zones it is still borrowing.
    """
    saved: list[dict] = []
    raw = stored.get(ZONE_SPECS_KEY)
    if raw:
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                saved = [entry for entry in parsed if isinstance(entry, dict)]
        except (json.JSONDecodeError, TypeError, ValueError):
            saved = []

    specs: list[dict] = []
    for index, _polygon in enumerate(polygons):
        entry = saved[index] if index < len(saved) else {}
        name = entry.get("name")
        name = str(name).strip() if isinstance(name, str) else ""
        zone_allowed = entry.get("allowed_people")
        zone_dwell = entry.get("dwell_seconds")
        try:
            zone_allowed = int(zone_allowed)
        except (TypeError, ValueError):
            zone_allowed = None
        try:
            zone_dwell = float(zone_dwell)
        except (TypeError, ValueError):
            zone_dwell = None

        # The flag that was saved is what decides this, not the numbers: a zone that
        # was borrowing the defaults is stored WITH those numbers filled in, so
        # they cannot be used to tell it apart from a zone that has rules of its
        # own that happen to be identical.
        follows = entry.get("using_defaults")
        if follows is None:
            follows = zone_allowed is None or zone_dwell is None
        specs.append(
            {
                "name": name or f"Zone {index + 1}",
                "allowed_people": allowed if zone_allowed is None else zone_allowed,
                "dwell_seconds": dwell if zone_dwell is None else zone_dwell,
                "enabled": bool(entry.get("enabled", True)),
                "using_defaults": follows,
            }
        )
    return specs


def zone_specs_from_body(
    body: ZoneConfig, current: list[dict], allowed: int, dwell: float
) -> list[dict]:
    """Which rules this request means, one per zone the request is saving.

    `zone_specs` wins when sent. A field that is not sent inside it keeps the
    value that zone already has, so updating a single zone does not silently
    reset the others. An unsent list keeps the stored rules, reindexed onto the
    polygons being saved.

    A zone flagged `using_defaults` is stored with the global numbers filled in,
    because those are the numbers it is actually using. The flag is what says it
    is following them, so changing a global later still reaches it.
    """
    polygons = body.zones if body.zones is not None else body.points
    if body.zone_specs is None:
        # older UI: the shapes changed, the rules stay with their zone numbers
        specs = list(current[: len(polygons)] if polygons is not None else current)
    else:
        specs = []
        for index, incoming in enumerate(body.zone_specs):
            previous = current[index] if index < len(current) else {}
            specs.append(
                {
                    "name": (previous.get("name") or f"Zone {index + 1}")
                    if incoming.name is None or not incoming.name.strip()
                    else incoming.name.strip(),
                    "allowed_people": incoming.allowed_people
                    if incoming.allowed_people is not None
                    else previous.get("allowed_people", allowed),
                    "dwell_seconds": incoming.dwell_seconds
                    if incoming.dwell_seconds is not None
                    else previous.get("dwell_seconds", dwell),
                    "enabled": previous.get("enabled", True)
                    if incoming.enabled is None
                    else bool(incoming.enabled),
                    # sending neither number is a partial update, not a request to
                    # start following the defaults again: a zone that already has
                    # rules of its own keeps that decision until the user says
                    # otherwise by sending the flag
                    "using_defaults": incoming.using_defaults
                    if incoming.using_defaults is not None
                    else (
                        previous["using_defaults"]
                        if incoming.allowed_people is None
                        and incoming.dwell_seconds is None
                        and "using_defaults" in previous
                        else incoming.allowed_people is None and incoming.dwell_seconds is None
                    ),
                }
            )

    for spec in specs:
        if spec["using_defaults"]:
            spec["allowed_people"] = allowed
            spec["dwell_seconds"] = dwell
    return specs


def _zone_reference() -> tuple[int, int]:
    """The frame size the saved points were drawn against."""
    try:
        stored = read_db().all_settings().get(ZONE_REFERENCE_KEY)
    except HTTPException:
        return 0, 0
    if not stored:
        return 0, 0
    try:
        width, height = json.loads(stored)
        return int(width), int(height)
    except (json.JSONDecodeError, TypeError, ValueError):
        return 0, 0


def _zone_keys_hold_only_empties(stored: dict) -> bool:
    """True when every present zone key parses to an empty list.

    "[]" is the shape an emptied zone is saved in; it must read as empty,
    not as a stored zone that could not be read. Anything else truthy that
    yields no polygons (unparseable text, a non-list) stays unreadable.
    """
    seen = False
    for key in (ZONE_POLYGONS_KEY, LEGACY_ZONE_KEY):
        raw = stored.get(key)
        if not raw:
            continue
        seen = True
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, TypeError, ValueError):
            return False
        if not isinstance(parsed, list) or len(parsed) != 0:
            return False
    return seen


def cleanup_invalid_zone() -> Optional[str]:
    """One-time check on startup: drop stored zones that cannot be used.

    The camera is not open yet, so the only frame size available is the one the
    zone was saved against. A zone that is degenerate, has repeated corners or
    crosses over itself is cleared here rather than left in the database for
    the UI to discover later. Every polygon is checked, so one bad zone does
    not keep the good ones hostage. Returns what was cleared, or None.
    """
    try:
        database = read_db()
        stored = database.all_settings()
    except HTTPException:
        return None

    raw = stored.get(ZONE_POLYGONS_KEY) or stored.get(LEGACY_ZONE_KEY)
    if not raw:
        return None

    polygons = stored_zones(stored)
    if not polygons:
        # "[]" parses to nothing, which is an empty zone, not an unreadable
        # one: a present-but-empty key must neither warn nor rewrite. Only
        # a key that holds something other than an empty list falls through
        # to the unreadable path below (real zones are untouched either way,
        # they never reach this branch).
        if _zone_keys_hold_only_empties(stored):
            return None
        polygons = [["unreadable"]]

    reference = _zone_reference()
    problems: list[str] = []
    for index, polygon in enumerate(polygons):
        try:
            points = [(int(x), int(y)) for x, y in polygon]
        except (TypeError, ValueError):
            problems.append(f"Zone {index + 1} could not be read")
            continue
        problem = zone_problem(points, reference[0], reference[1])
        if problem:
            problems.append(f"Zone {index + 1}: {problem}")
    if not problems:
        return None

    try:
        database.set_setting(ZONE_POLYGONS_KEY, json.dumps([]))
        database.set_setting(LEGACY_ZONE_KEY, json.dumps([]))
        database.set_setting(ZONE_REFERENCE_KEY, json.dumps([0, 0]))
    except DatabaseError:
        return None

    message = f"Stored danger zone cleared on startup: {'; '.join(problems)}"
    logging.getLogger(__name__).warning(message)
    return message


def clear_default_zone() -> Optional[str]:
    """One-time cleanup on startup: forget the polygon config.yaml used to ship.

    config.yaml used to carry a four corner polygon as the danger zone, so
    every existing install has that exact shape sitting in SQLite - looking,
    to the operator, like a zone somebody had deliberately drawn. There is no
    way to tell it apart from real work by eye, so it is matched against the
    shipped numbers and dropped.

    An install on another camera will have the same polygon scaled to its own
    frame size, so the comparison is done against the scaled version too.

    Only the matched polygon is removed. An operator who has since drawn a
    real zone keeps it: the old default is dropped, not everything next to it.
    Returns what was cleared, or None when there was nothing to do.
    """
    try:
        database = read_db()
        stored = database.all_settings()
    except HTTPException:
        return None

    polygons = stored_zones(stored)
    if not polygons:
        return None

    reference = _zone_reference()
    kept = [polygon for polygon in polygons if not _is_old_default(polygon, reference)]
    if len(kept) == len(polygons):
        return None

    try:
        database.set_setting(ZONE_POLYGONS_KEY, json.dumps(kept))
        # the legacy key carries the first surviving polygon, so an older reader
        # is not handed the default that was just removed
        database.set_setting(LEGACY_ZONE_KEY, json.dumps(kept[0] if kept else []))
        if not kept:
            # nothing left to place on this camera, so there is no frame size
            database.set_setting(ZONE_REFERENCE_KEY, json.dumps([0, 0]))
    except DatabaseError:
        return None

    message = "Stored danger zone cleared on startup: it was the old default polygon"
    logging.getLogger(__name__).warning(message)
    return message


def migrate_zones_to_sources() -> Optional[str]:
    """One-time adoption of the zones that predate per-source storage.

    An install upgrading to this version has its zones in the old global keys
    and no `source_zones` at all. Those zones were drawn on the source that is
    currently saved, so they are filed under that source's id with the frame
    size they carry. `zones_migrated` is set in the same call, so a later save
    can never be overwritten by running this again. The global keys are left
    alone, because they are the mirror the older readers still use.

    The frame size is not rescaled here: the source is not open yet, and the
    stored reference IS the frame size those zones were last drawn on. The
    pipeline and the UI rescale from that reference when the source opens.
    """
    try:
        database = read_db()
        stored = database.all_settings()
    except HTTPException:
        return None

    if stored.get(ZONES_MIGRATED_KEY):
        return None

    zones = stored_zones(stored)
    reference = list(_zone_reference())
    allowed = int(stored.get("allowed_people", 0) or 0)
    dwell = float(stored.get("dwell_seconds", 0.0) or 0.0)
    specs = stored_zone_specs(stored, zones, allowed, dwell)
    key = source_key(stored.get("source", "0"))

    def adopt(entries: dict) -> dict:
        # read-modify-write under one lock: a save landing between the read and
        # the write cannot be lost
        if zones and key not in entries:
            entries[key] = {"polygons": zones, "specs": specs, "reference": reference}
        return entries

    try:
        database.update_json_setting(SOURCE_ZONES_KEY, adopt)
        database.set_setting(ZONES_MIGRATED_KEY, "1")
    except DatabaseError:
        return None

    logger = logging.getLogger(__name__)
    if zones:
        message = (
            f"Adopted {len(zones)} zone(s) for source {key} at {reference[0]}x{reference[1]}"
            " into per-source storage"
        )
        logger.info(message)
        return message
    logger.info("Per-source zone storage initialised (no existing zone to adopt)")
    return None


def load_settings(source: Optional[str] = None) -> dict:
    """Settings come from SQLite (set by the UI); config.yaml is the fallback."""
    try:
        settings = load_config(CONFIG_PATH)
    except ConfigError:
        settings = {
            "machine": {"id": "excavator_01", "name": "Excavator 01"},
            "rules": {"mode": "TIMER", "dwell_seconds": 1.0},
            "detector": {"model": "yolo11n.pt", "confidence": 0.25, "person_class_id": 0},
            "danger_zone": {"points": [], "anchor": "center"},
        }

    database = read_db()
    stored = database.all_settings()

    active = stored.get("source", "0")
    target = active if source is None else str(source)

    # Globals mirror the ACTIVE source. A request for a different source reads
    # that source's own entry, so selecting a video shows the video's zones
    # instead of the ones drawn on the camera that came before it.
    if source is not None and source_key(target) != source_key(active):
        zones, entry_specs, reference = source_zones_for(stored, target)
        spec_store = {ZONE_SPECS_KEY: json.dumps(entry_specs)} if entry_specs else {}
    else:
        zones = stored_zones(stored)
        reference = list(_zone_reference())
        spec_store = stored

    if zones:
        # the legacy single zone stays populated for anything still reading it
        settings["danger_zone"]["points"] = [tuple(point) for point in zones[0]]
    else:
        # No zone. Either nothing was ever stored, or the zones were saved as
        # empty on purpose because the user cleared them. Both mean the same
        # thing: there is no zone, and config.yaml must not put one back.
        settings["danger_zone"]["points"] = []
    if "allowed_people" in stored:
        settings["rules"]["allowed_people"] = int(stored["allowed_people"])
    settings["rules"].setdefault("allowed_people", 0)

    # the rules the UI can change are stored next to the other settings, so
    # they survive a reload instead of snapping back to config.yaml
    for key, cast in (
        ("mode", str),
        ("dwell_seconds", float),
        ("repeat_alert_seconds", float),
    ):
        if key in stored:
            settings["rules"][key] = cast(stored[key])
    settings["rules"].setdefault("mode", "TIMER")
    settings["rules"].setdefault("dwell_seconds", 5.0)
    settings["rules"].setdefault("repeat_alert_seconds", 10.0)

    # the machine name can be renamed from the UI and must survive a reload
    machine_name = stored.get("machine_name")
    if machine_name:
        settings["machine"] = dict(settings["machine"])
        settings["machine"]["name"] = machine_name

    # the anchor point is chosen in the UI; feet stay the default because that
    # is what a machine standing on the ground needs
    anchor = stored.get("anchor")
    if anchor in ANCHORS:
        settings["danger_zone"] = dict(settings["danger_zone"])
        settings["danger_zone"]["anchor"] = anchor
    settings["danger_zone"].setdefault("anchor", "center")

    # detection confidence can be tuned from the Source card; a stored value
    # overrides config.yaml the same way the other UI settings do
    stored_confidence = stored.get("confidence")
    if stored_confidence is not None:
        try:
            value = float(stored_confidence)
        except (TypeError, ValueError):
            value = None
        if value is not None and 0 < value <= 1:
            settings["detector"] = dict(settings["detector"])
            settings["detector"]["confidence"] = max(MIN_CONFIDENCE, value)

    # The rules of each zone, upgraded on read: a database saved before zones
    # had names of their own comes back as one working zone called "Zone 1"
    # instead of failing, which is what keeps an existing install usable.
    specs = stored_zone_specs(
        spec_store,
        zones,
        allowed=int(settings["rules"]["allowed_people"]),
        dwell=float(settings["rules"]["dwell_seconds"]),
    )

    # side detection paths (fire/smoke, crowd, PPE): the Settings card stores
    # switches + cadence next to the other rules, and stored values win over
    # config.yaml the same way confidence does above. Switches are "1"/"0";
    # anything unparseable falls back to config.yaml instead of breaking.
    def _stored_switch(key: str):
        raw = stored.get(key)
        if raw == "1":
            return True
        if raw == "0":
            return False
        return None

    def _stored_fps(key: str):
        try:
            value = float(stored[key])
        except (KeyError, TypeError, ValueError):
            return None
        return value if value > 0 else None

    side_blocks = {}
    for block_name in ("fire_smoke", "crowd", "ppe"):
        merged = dict(settings.get(block_name) or {})
        switch = _stored_switch(f"{block_name}_enabled")
        if switch is not None:
            merged["enabled"] = switch
        fps = _stored_fps(f"{block_name}_analysis_fps")
        if fps is not None:
            merged["analysis_fps"] = fps
        side_blocks[block_name] = merged
    fire_smoke_cfg = side_blocks["fire_smoke"]
    crowd_cfg = side_blocks["crowd"]
    ppe_cfg = side_blocks["ppe"]
    # hazard switch: on unless explicitly stored off (missing = on, so
    # existing installs behave as before).
    hazard_cfg = dict(settings.get("hazard") or {})
    _hazard_stored = _stored_switch("hazard_enabled")
    if _hazard_stored is not None:
        hazard_cfg["enabled"] = _hazard_stored
    hazard_cfg["enabled"] = bool(hazard_cfg.get("enabled", True))
    # per-type fire/smoke switches, ranked: stored per-type > file per-type
    # > shared switch (which already merged the stored legacy value above).
    # An absent file key arrives as None from load_config, so older files
    # with only the shared switch keep working unchanged.
    for _type_key in ("fire_enabled", "smoke_enabled"):
        _stored_type = _stored_switch(_type_key)
        if _stored_type is not None:
            fire_smoke_cfg[_type_key] = _stored_type
        elif fire_smoke_cfg.get(_type_key) is None:
            fire_smoke_cfg[_type_key] = bool(fire_smoke_cfg.get("enabled", False))
        else:
            fire_smoke_cfg[_type_key] = bool(fire_smoke_cfg[_type_key])

    return {
        "machine": settings["machine"],
        "rules": settings["rules"],
        "detector": settings["detector"],
        "zones": zones,
        "zone_specs": specs,
        # the first zone only, so an older UI keeps drawing what it always drew
        "zone_points": zones[0] if zones else [],
        "zone_reference": reference,
        "allowed_people": int(settings["rules"]["allowed_people"]),
        "anchor": settings["danger_zone"]["anchor"],
        "zone_margin_px": float(settings["danger_zone"].get("zone_margin_px", 5)),
        "source": target,
    # hazard switch (global zone-evaluation on/off; the zones themselves
    # stay per-source as before).
    "hazard": {
        "enabled": hazard_cfg["enabled"],
    },
    # test-clip replay switch, ticked from the Live Monitor source picker
    # or the Settings Camera card (one stored setting, both write it).
    # Default off: a clip plays once and stops, so a demo cannot flood
    # the Alerts page with one row per loop unasked.
    "loop_video": _stored_switch("loop_video") is True,
        # side paths: stored DB values already overlaid above; a config
        # without the block (or the ConfigError fallback) means disabled
        # with the operating-point defaults.
        "fire_smoke": {
            "enabled": bool(fire_smoke_cfg.get("enabled", False)),
            "fire_enabled": bool(fire_smoke_cfg.get("fire_enabled", False)),
            "smoke_enabled": bool(fire_smoke_cfg.get("smoke_enabled", False)),
            "fire_enabled": bool(fire_smoke_cfg.get("fire_enabled", False)),
            "smoke_enabled": bool(fire_smoke_cfg.get("smoke_enabled", False)),
            "model": str(fire_smoke_cfg.get(
                "model", "detector/models/fire_smoke_v2.pt")),
            "confidence": float(fire_smoke_cfg.get("confidence", 0.35)),
            "imgsz": int(fire_smoke_cfg.get("imgsz", 640)),
            "fire_seconds": float(fire_smoke_cfg.get("fire_seconds", 1.5)),
            "smoke_seconds": float(fire_smoke_cfg.get("smoke_seconds", 2.5)),
            "high_confidence": float(fire_smoke_cfg.get("high_confidence", 0.8)),
            "person_containment": float(fire_smoke_cfg.get("person_containment", 0.5)),
            "analysis_fps": float(fire_smoke_cfg.get("analysis_fps", 2.0)),
        },
        "crowd": {
            "enabled": bool(crowd_cfg.get("enabled", False)),
            "source": str(crowd_cfg.get("source", "reuse")),
            "model": str(crowd_cfg.get(
                "model", "detector/models/crowd_v1.pt")),
            "confidence": float(crowd_cfg.get("confidence", 0.35)),
            "imgsz": int(crowd_cfg.get("imgsz", 640)),
            "analysis_fps": float(crowd_cfg.get("analysis_fps", 2.0)),
        },
        "ppe": {
            "enabled": bool(ppe_cfg.get("enabled", False)),
            "model": str(ppe_cfg.get(
                "model", "detector/models/ppe_v1.pt")),
            "confidence": float(ppe_cfg.get("confidence", 0.15)),
            "iou": float(ppe_cfg.get("iou", 0.45)),
            "imgsz": int(ppe_cfg.get("imgsz", 960)),
            "tta": bool(ppe_cfg.get("tta", True)),
            "memory_seconds": float(ppe_cfg.get("memory_seconds", 4.0)),
            "grading": {
                "medium_ratio": float((ppe_cfg.get("grading") or {}).get("medium_ratio", 0.25)),
                "high_ratio": float((ppe_cfg.get("grading") or {}).get("high_ratio", 0.5)),
                "critical_ratio": float((ppe_cfg.get("grading") or {}).get("critical_ratio", 1.0)),
            },
            "analysis_fps": float(ppe_cfg.get("analysis_fps", 1.0)),
        },
        # upload + history caps (missing keys fall back to current values).
        "limits": {
            "max_upload_size_mb": int((settings.get("limits") or {}).get("max_upload_size_mb", 500)),
            "max_videos": int((settings.get("limits") or {}).get("max_videos", 10)),
            "history_days": int((settings.get("limits") or {}).get("history_days", 14)),
        },
    }


@app.get("/config")
def get_config(source: Optional[str] = None) -> dict:
    """Settings, and for `?source=` the zones saved for that source.

    Without a source the active source's zones come back, which is what a page
    loading fresh wants. The UI passes the source it just selected so it can
    swap in that source's zones before it starts drawing.
    """
    return load_settings(source)


@app.post("/config")
def save_config(body: ZoneConfig) -> dict:
    # a field that is not sent keeps its current value, so a partial update
    # (for example only the polygon) never silently resets the other settings
    current = load_settings()
    database = read_db()
    stored = database.all_settings()

    new_source = current["source"] if body.source is None else str(body.source)
    switching = source_key(new_source) != source_key(current["source"])

    # A save with no shapes of its own - the debounced settings save, and the
    # one that switches the Camera source field - only re-files what is already
    # stored. It must never rewrite a polygon and never be refused because of
    # one either.
    settings_only = body.zones is None and body.points is None

    # When only the source changes, the shapes that belong in the mirror are the
    # new source's own. Carrying the old source's zones over is exactly the bug
    # this per-source store exists to stop.
    if settings_only and switching:
        base_zones, base_specs, base_reference = source_zones_for(stored, new_source)
    else:
        base_zones = current["zones"]
        base_specs = current["zone_specs"]
        base_reference = list(_zone_reference())

    wanted = zones_from_body(body, base_zones)

    if settings_only:
        # The polygons are the target source's own - its entry when the source
        # changed, the mirror's when it did not - so they are checked and
        # clamped against the reference THEY were filed under, never against
        # the live frame: the pipeline keeps the picture of the source that is
        # playing until it is restarted, so _frame_size_for would hand over the
        # 640-wide camera's size for a video's 1920-wide zone and clamp it
        # into the wrong picture. A zone stored clean is byte-for-byte
        # unchanged by this - the clamp and the floor are no-ops against its
        # own reference - and one that cannot be used is refused with the same
        # 422 its own save would have got.
        width, height = int(base_reference[0]), int(base_reference[1])
    else:
        # 422 with the offending zone and point named, so the UI can show it as it is
        width, height = _frame_size_for(body)
        if (width, height) == (0, 0):
            # no live frame and none sent: check against the size the shapes belong to
            width, height = base_reference
    zones = validate_zones(wanted, width, height)

    allowed = current["allowed_people"] if body.allowed_people is None else int(body.allowed_people)

    mode = current["rules"]["mode"] if body.mode is None else body.mode
    dwell = current["rules"]["dwell_seconds"] if body.dwell_seconds is None else float(body.dwell_seconds)
    repeat = (
        current["rules"]["repeat_alert_seconds"]
        if body.repeat_alert_seconds is None
        else float(body.repeat_alert_seconds)
    )
    # an unsent anchor keeps the saved one, so a partial update cannot flip it
    anchor = current["anchor"] if body.anchor is None else body.anchor

    # One rule set per polygon, then each one checked with its zone named: with
    # several zones on screen, "allowed_people is invalid" tells the user
    # nothing about which card to fix.
    specs = zone_specs_from_body(body, base_specs, allowed, dwell)
    specs = validate_zone_specs(specs, len(zones))

    reference = (
        # A settings-only save files the shapes back under the reference they
        # are already filed under: frame dims riding along must not re-file
        # polygons drawn on another size under the wrong one.
        list(base_reference)
        if settings_only
        else [int(body.frame_width), int(body.frame_height)]
        if body.frame_width and body.frame_height
        else list(base_reference)
    )

    try:
        database.set_setting(ZONE_POLYGONS_KEY, json.dumps(zones))
        # mirrored into the old key so anything still reading a single zone
        # sees the first one instead of nothing
        database.set_setting(LEGACY_ZONE_KEY, json.dumps(zones[0] if zones else []))
        # rules next to the shapes, one entry per zone, so the two are always
        # the same length and a zone cannot end up with somebody else's rules
        database.set_setting(ZONE_SPECS_KEY, json.dumps(specs))
        # the frame size these shapes were drawn on, whether it was sent or kept
        database.set_setting(ZONE_REFERENCE_KEY, json.dumps(reference))
        database.set_setting("allowed_people", allowed)
        database.set_setting("source", new_source)
        # file the shapes under the source they were drawn on. Read-modify-write
        # under one lock, so two saves on different sources cannot overwrite
        # each other's entry.
        def store(entries: dict) -> dict:
            entries[source_key(new_source)] = {
                "polygons": zones,
                "specs": specs,
                "reference": reference,
            }
            return entries

        database.update_json_setting(SOURCE_ZONES_KEY, store)
        if body.machine_name is not None:
            database.set_setting("machine_name", str(body.machine_name))
        if body.mode is not None:
            database.set_setting("mode", mode)
        if body.dwell_seconds is not None:
            database.set_setting("dwell_seconds", dwell)
        if body.repeat_alert_seconds is not None:
            database.set_setting("repeat_alert_seconds", repeat)
        if body.anchor is not None:
            database.set_setting("anchor", anchor)
        # detection confidence is stored next to the other rules, so it survives
        # a reload instead of falling back to config.yaml on the next start
        if body.confidence is not None:
            database.set_setting("confidence", float(body.confidence))
        # side detection paths are stored the same way: switches as "1"/"0",
        # cadence as numbers. /start picks them up through load_settings;
        # a running camera gets them live through apply_settings below.
        for attr in ("fire_smoke", "crowd", "ppe"):
            block = getattr(body, attr, None)
            if block is None:
                continue
            if block.enabled is not None:
                database.set_setting(f"{attr}_enabled", "1" if block.enabled else "0")
            if block.analysis_fps is not None:
                database.set_setting(f"{attr}_analysis_fps", float(block.analysis_fps))
        # hazard switch and per-type fire/smoke switches, same storage shape.
        if body.hazard is not None and body.hazard.enabled is not None:
            database.set_setting("hazard_enabled", "1" if body.hazard.enabled else "0")
        if body.fire is not None and body.fire.enabled is not None:
            database.set_setting("fire_enabled", "1" if body.fire.enabled else "0")
        if body.smoke is not None and body.smoke.enabled is not None:
            database.set_setting("smoke_enabled", "1" if body.smoke.enabled else "0")
        # test-clip replay switch: the two tick boxes share this one stored
        # setting, so either one moves both.
        if body.loop_video is not None:
            database.set_setting("loop_video", "1" if body.loop_video else "0")
    except DatabaseError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    # If the camera is already running, push the new zones straight into it. The
    # capture stays open, so saving a redrawn zone never makes the video blink.
    flipped: list[int] = []
    pipeline = _pipeline
    if pipeline is not None and getattr(pipeline, "active", False):
        try:
            flipped = pipeline.apply_settings(
                zones=zones,
                zone_specs=specs,
                allowed_people=allowed,
                mode=mode,
                dwell_seconds=dwell,
                repeat_alert_seconds=repeat,
                anchor=anchor,
                confidence=body.confidence,
                zone_reference=(int(body.frame_width), int(body.frame_height))
                if body.frame_width and body.frame_height
                else None,
                fire_smoke_enabled=body.fire_smoke.enabled if body.fire_smoke else None,
                fire_smoke_analysis_fps=(
                    body.fire_smoke.analysis_fps if body.fire_smoke else None
                ),
                fire_enabled=body.fire.enabled if body.fire else None,
                smoke_enabled=body.smoke.enabled if body.smoke else None,
                hazard_enabled=body.hazard.enabled if body.hazard else None,
                crowd_enabled=body.crowd.enabled if body.crowd else None,
                crowd_analysis_fps=body.crowd.analysis_fps if body.crowd else None,
                ppe_enabled=body.ppe.enabled if body.ppe else None,
                ppe_analysis_fps=body.ppe.analysis_fps if body.ppe else None,
            )
        except PipelineError:
            # the zone is saved either way; the running camera just keeps the
            # zone it had, and /start can pick the new one up
            flipped = []

    return {
        "saved": True,
        "zones": zones,
        "zone_specs": specs,
        "points": zones[0] if zones else [],
        "allowed_people": allowed,
        "mode": mode,
        "dwell_seconds": dwell,
        "repeat_alert_seconds": repeat,
        "anchor": anchor,
        # how many workers the new zone moved in or out of the danger area
        "flipped": flipped,
    }


@app.post("/start")
def start_monitoring(body: ZoneConfig) -> dict:
    # read the settings for the source being started, so the zone reference and
    # the fallback shapes are that source's own and not the one running before
    settings = load_settings(body.source)

    # an explicit empty list clears the zone, so only fall back when not sent
    zones = int_zones(zones_from_body(body, settings["zones"]) or [])
    allowed = settings["allowed_people"] if body.allowed_people is None else int(body.allowed_people)
    # the rules travel with the shapes into the monitor, so /start cannot pick up
    # a zone from the database and give it the global numbers instead
    specs = validate_zone_specs(
        zone_specs_from_body(
            body,
            settings["zone_specs"],
            allowed,
            float(settings["rules"]["dwell_seconds"]),
        ),
        len(zones),
    )
    source = body.source or settings["source"]
    # An uploaded test clip replays only when the Loop video box is ticked
    # (the stored loop_video switch, default off); any other file plays once
    # and stops at the end. An explicit `loop` wins over both. The webcam is
    # unaffected either way: its read loop never rewinds (see _run).
    if body.loop is not None:
        loop = bool(body.loop)
    elif _is_test_video(source):
        loop = bool(settings.get("loop_video", False))
    else:
        loop = False
    # The Start button carries the rules on screen right now. They win over the
    # stored ones, so a dwell/mode/confidence change followed immediately by
    # Start cannot be lost to the debounced settings save still in flight.
    mode = body.mode or settings["rules"]["mode"]
    dwell = body.dwell_seconds if body.dwell_seconds is not None else float(settings["rules"]["dwell_seconds"])
    repeat = (
        body.repeat_alert_seconds
        if body.repeat_alert_seconds is not None
        else float(settings["rules"].get("repeat_alert_seconds", 10))
    )
    anchor = body.anchor or settings["anchor"]
    confidence = (
        float(body.confidence)
        if body.confidence is not None
        else float(settings["detector"]["confidence"])
    )
    machine = dict(settings["machine"])
    if body.machine_name:
        machine["name"] = body.machine_name

    pipeline = get_pipeline()
    if getattr(pipeline, "active", False):
        # A different source means a different capture: applying zones in place
        # keeps the OLD video playing, so the capture is restarted instead.
        # Without this, picking a test clip while running reported "Playing X"
        # while the webcam stayed on screen.
        current_source = getattr(pipeline, "source", "") or ""
        same = (
            _same_source(current_source, source)
            or source_key(current_source) == source_key(source)
        )
        if not same:
            pipeline.stop()
        else:
            # the camera is already open on this source: swap the zone in place
            # so the picture never blinks and the frame counter keeps climbing
            try:
                flipped = pipeline.apply_settings(
                    zones=zones,
                    zone_specs=specs,
                    allowed_people=allowed,
                    mode=mode,
                    dwell_seconds=float(dwell),
                    repeat_alert_seconds=float(repeat),
                    anchor=anchor,
                    confidence=confidence,
                    zone_reference=tuple(settings["zone_reference"]),
                    zone_margin_px=float(settings["zone_margin_px"]),
                )
            except PipelineError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            return {
                "running": True,
                "source": pipeline.source,
                "allowed_people": allowed,
                "anchor": anchor,
                "zones": zones,
                "zone_specs": specs,
                "points": zones[0] if zones else [],
                "flipped": flipped,
            }

    pipeline.stop()  # a stale thread from an earlier run is cleaned up first
    try:
        pipeline.start(
            source=source,
            machine=machine,
            zones=zones,
            zone_specs=specs,
            allowed_people=allowed,
            mode=mode,
            dwell_seconds=float(dwell),
            repeat_alert_seconds=float(repeat),
            model=settings["detector"]["model"],
            confidence=confidence,
            person_class_id=int(settings["detector"]["person_class_id"]),
            imgsz=int(settings["detector"].get("imgsz", 320)),
            detect_every_n=int(settings["detector"].get("detect_every_n", 1)),
            max_detect_long_side=int(settings["detector"].get("max_detect_long_side", 960)),
            openvino=bool(settings["detector"].get("openvino", False)),
            adaptive_idle_n=int(settings["detector"].get("adaptive_idle_n", 0)),
            person_confirm_conf=float(settings["detector"].get("person_confirm_conf", 0.4)),
            person_confirm_frames=int(settings["detector"].get("person_confirm_frames", 3)),
            person_instant_conf=float(settings["detector"].get("person_instant_conf", 0.6)),
            person_confirm_move_ratio=float(settings["detector"].get("person_confirm_move_ratio", 0.05)),
            anchor=anchor,
            zone_reference=tuple(settings["zone_reference"]),
            loop=loop,
            zone_margin_px=float(settings["zone_margin_px"]),
            hazard_enabled=bool(settings.get("hazard", {}).get("enabled", True)),
            fire_smoke_enabled=bool(settings.get("fire_smoke", {}).get("enabled", False)),
            fire_enabled=settings.get("fire_smoke", {}).get("fire_enabled", None),
            smoke_enabled=settings.get("fire_smoke", {}).get("smoke_enabled", None),
            fire_smoke_model=str(settings.get("fire_smoke", {}).get(
                "model", "detector/models/fire_smoke_v2.pt")),
            fire_smoke_confidence=float(settings.get("fire_smoke", {}).get("confidence", 0.35)),
            fire_smoke_imgsz=int(settings.get("fire_smoke", {}).get("imgsz", 640)),
            fire_smoke_fire_seconds=float(settings.get("fire_smoke", {}).get("fire_seconds", 1.5)),
            fire_smoke_smoke_seconds=float(settings.get("fire_smoke", {}).get("smoke_seconds", 2.5)),
            fire_smoke_analysis_fps=float(settings.get("fire_smoke", {}).get("analysis_fps", 2.0)),
            fire_high_confidence=float(settings.get("fire_smoke", {}).get("high_confidence", 0.8)),
            fire_person_containment=float(settings.get("fire_smoke", {}).get("person_containment", 0.5)),
            crowd_enabled=bool(settings.get("crowd", {}).get("enabled", False)),
            crowd_source=str(settings.get("crowd", {}).get("source", "reuse")),
            crowd_model=str(settings.get("crowd", {}).get(
                "model", "detector/models/crowd_v1.pt")),
            crowd_confidence=float(settings.get("crowd", {}).get("confidence", 0.35)),
            crowd_imgsz=int(settings.get("crowd", {}).get("imgsz", 640)),
            crowd_analysis_fps=float(settings.get("crowd", {}).get("analysis_fps", 2.0)),
            ppe_enabled=bool(settings.get("ppe", {}).get("enabled", False)),
            ppe_model=str(settings.get("ppe", {}).get(
                "model", "detector/models/ppe_v1.pt")),
            ppe_confidence=float(settings.get("ppe", {}).get("confidence", 0.15)),
            ppe_iou=float(settings.get("ppe", {}).get("iou", 0.45)),
            ppe_imgsz=int(settings.get("ppe", {}).get("imgsz", 960)),
            ppe_tta=bool(settings.get("ppe", {}).get("tta", True)),
            ppe_analysis_fps=float(settings.get("ppe", {}).get("analysis_fps", 1.0)),
            ppe_memory_seconds=float(settings.get("ppe", {}).get("memory_seconds", 4.0)),
            ppe_medium_ratio=float((settings.get("ppe", {}).get("grading") or {}).get("medium_ratio", 0.25)),
            ppe_high_ratio=float((settings.get("ppe", {}).get("grading") or {}).get("high_ratio", 0.5)),
            ppe_critical_ratio=float((settings.get("ppe", {}).get("grading") or {}).get("critical_ratio", 1.0)),
        )
    except PipelineError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {"running": True, "source": source, "allowed_people": allowed, "anchor": anchor,
            "zones": zones, "zone_specs": specs, "points": zones[0] if zones else []}


@app.post("/stop")
def stop_monitoring() -> dict:
    pipeline = _pipeline
    if pipeline is not None:
        pipeline.stop()
    return {"running": False}


@app.get("/state")
def state() -> dict:
    """Everything the UI needs to render the current situation."""
    started = time.perf_counter()

    def _timed(body: dict) -> dict:
        # the UI polls this four times a second with an 8 s budget, so
        # anything past ~2 s means a lock or the monitor is under unusual
        # load: log it here so the next occurrence names itself instead of
        # surfacing only as a client-side timeout toast.
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if elapsed_ms > 2000:
            logging.getLogger(__name__).warning(
                "/state slow: %.0f ms (pipeline locks or monitor under load)",
                elapsed_ms,
            )
        return body

    if _pipeline is None:
        settings = load_settings()
        # The same shape the running pipeline returns, built from the saved
        # rules. The UI has one set of fields to fill in whether or not the
        # camera is on, so a stopped dashboard shows every zone as empty rather
        # than as missing.
        return _timed({
            "running": False,
            "error": None,
            # a saved file source that no longer exists (a clip that was never
            # cloned, or has been deleted) is named here so the stopped panel
            # can say so instead of showing a blank camera
            "source_error": _missing_source_message(settings["source"]),
            "source": settings["source"],
            "frame_width": 0,
            "frame_height": 0,
            "frame_seq": 0,
            "machine": settings["machine"],
            "rules": settings["rules"],
            "zones": settings["zones"],
            "zone_specs": settings["zone_specs"],
            "zone_points": settings["zone_points"],
            "zone_reference": settings["zone_reference"],
            "allowed_people": settings["allowed_people"],
            "inside": 0,
            "status": SAFE,
            "capacity_alert": False,
            "worker_alert": False,
            "message": "",
            "workers": [],
            # hazard switch as configured while stopped (zones still report
            # their saved shapes below; nothing is evaluated).
            "hazard": settings["hazard"],
            # side detection paths report their configured values while
            # stopped (same shape as a running pipeline with nothing
            # analyzed yet), so the Settings card and the readouts render
            # before the first start.
            "fire_smoke": {**settings["fire_smoke"],
                           "confirming": {"fire": None, "smoke": None}},
            "crowd": settings["crowd"],
            "ppe": {**settings["ppe"], "deferred": PPE_EMISSION_DEFERRED},
            "zone_reports": [
                {
                    "index": index,
                    "name": spec["name"],
                    "inside": 0,
                    "allowed_people": spec["allowed_people"],
                    "dwell_seconds": 0.0,
                    "dwell_limit": spec["dwell_seconds"],
                    "alerting": False,
                    "enabled": spec["enabled"],
                }
                for index, spec in enumerate(settings["zone_specs"])
            ],
            "perf": None,
            "test_video": _is_test_video(settings["source"]),
        })
    body = _pipeline.state()
    body["test_video"] = _is_test_video(body.get("source", ""))
    if not body.get("running"):
        body["source_error"] = _missing_source_message(body.get("source", ""))
    else:
        body["source_error"] = None
    return _timed(body)


# ---------------------------------------------------------------------------
# Uploaded test videos
#
# A recorded or generated clip can stand in for the webcam. The file is copied
# into data/videos/ under a uuid name - the browser's name is kept only as a
# label - and everything shown about it (size, resolution, fps, duration) is
# read back from the real file, never taken from the upload request.
# ---------------------------------------------------------------------------


class VideoTooLarge(Exception):
    """Raised while streaming an upload past MAX_VIDEO_BYTES."""


def _sidecar_path(video_id: str) -> Optional[str]:
    """The json sidecar path for an id, or None when the id is not safe.

    The id arrives from the browser, so it is only ever joined to a path after
    it has been shown to be a 32-character hex uuid; `_inside` then proves the
    result really sits in the video folder.
    """
    if not _VIDEO_ID.match(video_id or ""):
        return None
    name = f"{video_id}.json"
    if not _inside(VIDEO_DIR, name):
        return None
    return os.path.join(VIDEO_DIR, name)


def _video_file_path(file_name: str) -> Optional[str]:
    """The video path for a stored file name, or None when it escapes the folder."""
    if not file_name or not _inside(VIDEO_DIR, file_name):
        return None
    return os.path.join(VIDEO_DIR, file_name)


def _read_sidecar(path: Optional[str]) -> Optional[dict]:
    if path is None:
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _load_videos() -> list[dict]:
    """Every stored video, newest first, with its real path as `source`.

    A sidecar whose video file has gone is skipped rather than reported, so the
    list can never offer a Use that the pipeline would then refuse.
    """
    if not os.path.isdir(VIDEO_DIR):
        return []
    found: list[dict] = []
    for entry in os.listdir(VIDEO_DIR):
        if not entry.endswith(".json"):
            continue
        sidecar = _sidecar_path(entry[:-5])
        meta = _read_sidecar(sidecar)
        if not meta:
            continue
        path = _video_file_path(str(meta.get("file", "")))
        if path is None or not os.path.isfile(path):
            continue
        meta = {**meta, "source": path}
        found.append(meta)
    found.sort(key=lambda item: item.get("created", ""), reverse=True)
    return found


def _remove_video(video_id: str, file_name: str) -> None:
    """Delete a half-written upload, ignoring anything that was never there."""
    for path in (_video_file_path(file_name), _sidecar_path(video_id)):
        if path and os.path.isfile(path):
            try:
                os.remove(path)
            except OSError:
                logging.warning("could not remove %s", path)


def _probe_video(path: str) -> Optional[dict]:
    """Read real facts off the file. None when OpenCV cannot open or decode it."""
    import cv2

    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            return None
        ok, frame = cap.read()
        if not ok or frame is None:
            return None
        height, width = frame.shape[:2]
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frames = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
        duration = (frames / fps) if fps > 0 else 0.0
        return {
            "width": int(width),
            "height": int(height),
            "fps": round(fps, 2),
            "duration": round(duration, 2),
        }
    finally:
        cap.release()


def _same_source(a: str, b: str) -> bool:
    try:
        return os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))
    except (OSError, ValueError):
        return False


def _is_test_video(source: str) -> bool:
    """True when `source` is one of the uploaded clips in data/videos/.

    A webcam index is never a test video, and neither is a file somewhere else
    on disk: only a path that resolves inside the upload folder counts.
    """
    if not source or str(source).isdigit():
        return False
    try:
        return os.path.normcase(os.path.dirname(os.path.abspath(source))) == os.path.normcase(
            os.path.abspath(VIDEO_DIR)
        )
    except (OSError, ValueError):
        return False


def _missing_source_message(source: str) -> Optional[str]:
    """Why the saved source cannot be opened, or None when it looks usable.

    A camera index is always "usable" here (there is nothing on disk to check),
    and an existing file needs no message. A path that no longer exists — e.g.
    a test clip from data/videos/ that was never cloned (data/ is gitignored)
    or has since been deleted — is reported with its path, so the UI can name
    the missing file instead of showing a blank stopped camera.
    """
    text = str(source or "").strip()
    if not text or text.isdigit():
        return None
    if os.path.isfile(text):
        return None
    if _is_test_video(text):
        return f"Test video not found at {text}"
    return f"Video file not found at {text}"


@app.get("/videos")
def list_videos() -> dict:
    limits = _limits()
    return {"videos": _load_videos(), "limit": limits["max_videos"],
            "max_mb": limits["max_upload_size_mb"]}


@app.post("/videos")
async def upload_video(file: UploadFile = File(...)) -> dict:
    original = os.path.basename(file.filename or "")
    ext = os.path.splitext(original)[1].lower()
    if ext not in VIDEO_EXTS:
        allowed = ", ".join(VIDEO_EXTS)
        raise HTTPException(
            status_code=415,
            detail=f"'{ext or original or 'file'}' is not a video. Allowed types: {allowed}",
        )

    if len(_load_videos()) >= _limits()["max_videos"]:
        raise HTTPException(
            status_code=409,
            detail=(
                f"the {_limits()['max_videos']} video limit is reached. "
                "Delete a video before uploading another one."
            ),
        )

    os.makedirs(VIDEO_DIR, exist_ok=True)
    video_id = uuid.uuid4().hex
    file_name = f"{video_id}{ext}"
    target = _video_file_path(file_name)
    assert target is not None  # a uuid + known extension is always inside

    limits = _limits()
    max_bytes = limits["max_upload_size_mb"] * 1024 * 1024
    size = 0
    try:
        with open(target, "wb") as handle:
            while True:
                chunk = await file.read(UPLOAD_CHUNK_BYTES)
                if not chunk:
                    break
                size += len(chunk)
                if size > max_bytes:
                    raise VideoTooLarge
                handle.write(chunk)
    except VideoTooLarge:
        _remove_video(video_id, file_name)
        raise HTTPException(
            status_code=413,
            detail=f"that video is larger than the {limits['max_upload_size_mb']} MB limit",
        )
    except OSError as exc:
        _remove_video(video_id, file_name)
        raise HTTPException(status_code=500, detail=f"could not save the upload: {exc}") from exc

    info = _probe_video(target)
    if info is None:
        _remove_video(video_id, file_name)
        raise HTTPException(
            status_code=422,
            detail="that file could not be read as a video - no frame could be decoded",
        )

    meta = {
        "id": video_id,
        "file": file_name,
        "name": original or f"{video_id}{ext}",
        "size_mb": round(size / (1024 * 1024), 2),
        "width": info["width"],
        "height": info["height"],
        "fps": info["fps"],
        "duration": info["duration"],
        "created": datetime.now(timezone.utc).isoformat(),
    }
    sidecar = _sidecar_path(video_id)
    assert sidecar is not None
    with open(sidecar, "w", encoding="utf-8") as handle:
        json.dump(meta, handle)
    try:
        os.utime(target, None)  # keep the sidecar and the video written together
    except OSError:
        pass
    return {**meta, "source": target}


@app.delete("/videos/{video_id}")
def delete_video(video_id: str) -> dict:
    sidecar = _sidecar_path(video_id)
    meta = _read_sidecar(sidecar)
    if not meta:
        raise HTTPException(status_code=404, detail="no such video")

    pipeline = get_pipeline()
    source = getattr(pipeline, "source", "") if getattr(pipeline, "active", False) else ""
    path = _video_file_path(str(meta.get("file", "")))
    if source and path and _same_source(source, path):
        raise HTTPException(
            status_code=409,
            detail="stop the camera before deleting the video it is playing",
        )

    _remove_video(video_id, str(meta.get("file", "")))
    # the clip is gone, so the zones drawn on it have nothing left to belong to
    try:
        reader = read_db()
        key = source_key(str(meta.get("file", "")))

        def forget(entries: dict) -> dict:
            entries.pop(key, None)
            return entries

        reader.update_json_setting(SOURCE_ZONES_KEY, forget)
    except DatabaseError:
        # a failed cleanup must not fail the delete the user asked for
        pass
    return {"deleted": video_id}


@app.get("/stream")
async def stream(request: Request) -> StreamingResponse:
    """Annotated frames as an MJPEG stream, read straight from the pipeline.

    The generator stops as soon as the browser goes away, so closing a tab
    actually frees the thread instead of leaving it polling forever.
    """
    pipeline = get_pipeline()

    async def frames():
        last_seq = -1
        idle = 0
        while True:
            if await request.is_disconnected():
                return
            jpeg = pipeline.jpeg()
            state = pipeline.state()  # one call per frame, it takes the lock
            seq = state["frame_seq"]

            if jpeg is not None and seq != last_seq:
                last_seq = seq
                idle = 0
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
                continue

            idle += 1
            # Nothing new. While stopped, replace the stale frame straight away
            # so it does not look like a live stream; otherwise wait a moment
            # in case the next frame is just about to arrive.
            if idle > 100 or (not state["running"] and jpeg is None):
                blank = _blank_frame(pipeline)
                if blank is None:
                    return
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + blank + b"\r\n"
                idle = 0
            # never block for longer than one frame at the target rate
            await asyncio.sleep(STREAM_POLL_S)

    return StreamingResponse(
        frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store"},
    )


@app.get("/snapshot")
def snapshot() -> Response:
    """One still frame, so a zone can be drawn on a picture that stands still.

    Drawing against the live MJPEG stream is unusable: the frame moves under
    the pointer, so every corner lands somewhere else. The browser grabs this
    single JPEG once, draws on top of it, and puts the stream back afterwards.
    """
    pipeline = get_pipeline()
    jpeg = pipeline.jpeg()
    if jpeg is None:
        raise HTTPException(status_code=404, detail="no video yet")
    state = pipeline.state()
    return Response(
        content=jpeg,
        media_type="image/jpeg",
        headers={
            "Cache-Control": "no-store",
            "X-Frame-Width": str(state["frame_width"]),
            "X-Frame-Height": str(state["frame_height"]),
        },
    )


def _blank_frame(pipeline: Pipeline) -> Optional[bytes]:
    """A placeholder image so the browser <img> has something to show.

    It is a plain dark rectangle on purpose. Drawing words on it would sit
    under the UI's own message about the camera, so the picture stays empty and
    the UI is the only thing that tells the user what to do next.
    """
    import cv2
    import numpy as np

    width, height = pipeline.frame_size
    if width <= 0 or height <= 0:
        width, height = 640, 360
    frame = np.full((height, width, 3), 24, dtype=np.uint8)
    success, jpeg = cv2.imencode(".jpg", frame)
    return jpeg.tobytes() if success else None
