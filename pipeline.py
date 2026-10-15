"""Background video pipeline used by the web UI.

This does NOT contain any new computer vision: detection and tracking come
from detector.tracker, the zone / state logic from zone.danger_zone, event
storage from database.db and the drawing from visualize - all unchanged.

The only thing added here is a background thread that owns the video source
and keeps the latest annotated frame available as a JPEG for the browser.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import deque
from typing import Optional

import cv2

from app import Timeline, open_source
from database.db import Database, DatabaseError
from detector.tracker import DetectorError, PersonTracker
from snapshots import SnapshotWriter
from visualize import draw_fire_box, draw_person_box, draw_ppe_box, draw_smoke_box
from zone.danger_zone import (
    ANCHORS,
    ALERT,
    CAPACITY_ALERT,
    DWELL_ALERT,
    ENTRY_ALERT,
    SAFE,
    DangerZone,
    ZoneError,
    ZoneMonitor,
    ZoneRule,
)

# the rolling window used for the /state "perf" averages
PERF_WINDOW = 30

LOG = logging.getLogger(__name__)


def _box_contained(inner, outers: list, threshold: float) -> bool:
    """True when `threshold` or more of `inner`'s area sits inside any box
    of `outers`. Bad boxes never match. A big real fire dwarfs any person
    box, so its containment reads near zero and it is kept."""
    try:
        x1, y1, x2, y2 = (float(v) for v in inner)
    except (TypeError, ValueError):
        return False
    area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if area <= 0:
        return False
    for outer in outers:
        try:
            ox1, oy1, ox2, oy2 = (float(v) for v in outer)
        except (TypeError, ValueError):
            continue
        overlap = max(0.0, min(x2, ox2) - max(x1, ox1)) * max(0.0, min(y2, oy2) - max(y1, oy1))
        if overlap / area >= threshold:
            return True
    return False


class PipelineError(Exception):
    """Raised when the pipeline cannot be started."""


def _clean_polygons(value) -> list[list[tuple[int, int]]]:
    """Accept a flat point list or a list of polygons and normalise both.

    Empty is fine: the camera can run with no zone drawn yet, people are just
    boxed and nothing alerts.
    """
    try:
        return DangerZone(value).polygons
    except ZoneError as exc:
        raise PipelineError(str(exc)) from exc
    except (TypeError, ValueError) as exc:
        raise PipelineError(f"the danger zone could not be read: {exc}") from exc


def _spec_value(entry, key: str, default=None):
    """One field of a zone rule, whether it arrived as a dict or as an object.

    The API hands `/start` and `apply_settings` plain dicts (`zone_specs` is a
    list of JSON objects), while the CLI and older callers pass objects. Reading
    a dict with `getattr` silently finds nothing, which is how every zone came
    back named "Zone N" and every per-zone allowance and dwell limit was quietly
    replaced by the global one: the rules reached the pipeline and were then
    thrown away one field at a time.
    """
    if isinstance(entry, dict):
        return entry.get(key, default)
    return getattr(entry, key, default)


def _clean_zone_specs(
    value, zone_count: int, allowed_people: int, dwell_seconds: float
) -> list:
    """One ZoneRule per polygon, or nothing when the caller sent no rules.

    Rules are matched up with the shapes by position, never by name: a zone can
    be renamed freely, and a rename must not move somebody else's allowance onto
    it. A zone with no rule of its own is left to follow the global defaults,
    which is what keeps a single-zone caller behaving exactly as it always did.
    """
    if not value:
        return []

    specs = []
    for index in range(zone_count):
        entry = value[index] if index < len(value) else None
        if entry is None:
            continue
        allowed = _spec_value(entry, "allowed_people")
        dwell = _spec_value(entry, "dwell_seconds")
        follows = bool(
            _spec_value(entry, "using_defaults", allowed is None and dwell is None)
        )
        name = _spec_value(entry, "name", "")
        # A zone that is still following the defaults is given the globals that
        # are in force right now, not the numbers it happened to be saved with:
        # the saved copy is there so the UI can show it, not to freeze it.
        count_by = str(_spec_value(entry, "count_by", "foot_point") or "foot_point")
        if count_by not in ("foot_point", "overlap"):
            count_by = "foot_point"
        overlap = _spec_value(entry, "overlap_threshold", 0.10)
        try:
            overlap = float(overlap)
        except Exception:
            overlap = 0.10
        if overlap < 0:
            overlap = 0
        if overlap > 1:
            overlap = 1
        specs.append(
            ZoneRule(
                name=str(name or f"Zone {index + 1}"),
                points=[],
                allowed_people=(
                    max(0, int(allowed_people))
                    if follows or allowed is None
                    else max(0, int(allowed))
                ),
                dwell_seconds=(
                    float(dwell_seconds)
                    if follows or dwell is None
                    else max(0.0, float(dwell))
                ),
                enabled=bool(_spec_value(entry, "enabled", True)),
                using_defaults=follows,
                count_by=count_by,
                overlap_threshold=overlap,
            )
        )
    return specs


def scale_zone_points(
    points: list, reference: tuple[int, int], actual: tuple[int, int]
) -> list[tuple[int, int]]:
    """Scale a zone drawn on `reference` to the `actual` frame size.

    A zone drawn on a 1280x720 camera still covers the right part of the
    picture on a 640x480 one, so it is scaled instead of being used unscaled.
    When either size is unknown the points are returned unchanged.
    """
    ref_width, ref_height = reference
    width, height = actual
    if ref_width <= 0 or ref_height <= 0 or width <= 0 or height <= 0:
        return [(int(x), int(y)) for x, y in points]
    if (ref_width, ref_height) == (width, height):
        return [(int(x), int(y)) for x, y in points]

    scale_x = width / ref_width
    scale_y = height / ref_height
    scaled: list[tuple[int, int]] = []
    for x, y in points:
        new_x = min(width, max(0, round(int(x) * scale_x)))
        new_y = min(height, max(0, round(int(y) * scale_y)))
        scaled.append((new_x, new_y))
    return scaled


def scale_zone_polygons(
    polygons: list, reference: tuple[int, int], actual: tuple[int, int]
) -> list[list[tuple[int, int]]]:
    """Scale every zone. Kept next to the single-zone version for symmetry."""
    return [scale_zone_points(polygon, reference, actual) for polygon in polygons]


class Pipeline:
    """Runs the video loop in a thread and publishes frame + state."""

    # never skip more than this many frames in one go, so a slow model cannot
    # fast-forward through the whole file at once
    MAX_SKIP = 30

    # a single bad frame must not kill the stream, but a model that fails on
    # 30 frames in a row is broken and the user needs to know
    MAX_CONSECUTIVE_ERRORS = 30

    def __init__(self, database: Database) -> None:
        self.db = database
        self._lock = threading.Lock()
        # held only for the zone/state step of a frame, so a zone change waits
        # for the frame in flight instead of landing in the middle of one
        self._config_lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

        self.running = False
        self.error: Optional[str] = None
        # True once a single-play file has reached its end by itself (never
        # for a camera, a user stop, or a failure): the UI shows a neutral
        # "Clip finished" instead of a blank stopped camera. Read under
        # self._lock with running, like error.
        self.finished_clip = False
        self.source: str = ""
        self.frame_size: tuple[int, int] = (0, 0)
        self._jpeg: Optional[bytes] = None
        self._frame_seq = 0

        self.monitor: Optional[ZoneMonitor] = None
        self.machine: dict = {"id": "", "name": ""}
        self.rules: dict = {"mode": "TIMER", "dwell_seconds": 5.0}
        # every danger area, each a list of corners. Several zones are allowed
        # and a person counts as inside when they are in ANY of them.
        self.zone_polygons: list[list[tuple[int, int]]] = []
        # each zone's own name and rules, index aligned with zone_polygons
        self.zone_specs: list[dict] = []
        # the frame size the saved points were drawn against; when the video
        # turns out to be a different size the zone is scaled, not rejected
        self.zone_reference: tuple[int, int] = (0, 0)
        self.tracker: Optional[PersonTracker] = None
        # saves the picture that goes with each alert, off the video thread
        self.snapshots: Optional[SnapshotWriter] = None
        self.anchor: str = "bottom_center"
        self.skipped_frames = 0
        self.detector_errors = 0
        # Hazard-zone system switch (default on). Off means no zone
        # evaluation, no zone events and no zone UI; the person tracker
        # still runs while crowd or PPE need its boxes (see tracker_needed).
        self.hazard_enabled: bool = True
        # Fire/smoke side path (additive, default off). The detector runs
        # when at least one of the per-type switches is on; each type is
        # filtered independently downstream, so either can run alone.
        self.fire_smoke = None
        self.fire_smoke_gate = None
        self.fire_smoke_emitter = None
        self.fire_smoke_camera_id: str = ""
        self.fire_smoke_analysis_fps: float = 2.0
        self.fire_smoke_confidence: float = 0.35
        self.fire_high_confidence: float = 0.8
        self.fire_person_containment: float = 0.5
        # per-type switches. True here only so a bare Pipeline() with a
        # detector attached directly (tests) behaves unfiltered; start()
        # always resolves both explicitly, defaulting to off.
        self.fire_enabled: bool = True
        self.smoke_enabled: bool = True
        # wall-time (monotonic) of the last analyzed frame; None means the
        # next frame analyzes immediately. Paced by elapsed time, never by
        # frame count, so the gate's seconds-to-frames math holds at any
        # video fps.
        self._fire_smoke_last_run: Optional[float] = None
        # person boxes of the latest loop frame, for the fire worker's
        # vest suppression (read under _config_lock; empty when nobody is
        # tracked, which disables suppression).
        self._last_people_boxes: list = []
        self.fire_smoke_stats: dict = {
            "analyzed": 0,
            "emitted": 0,
            "suppressed": 0,
            "last_types": [],
            "last_crossed": [],
        }
        # Crowd side path (additive, default off). The detector stays None
        # unless start() enables the "model" source; the "reuse" source
        # needs no model at all. Metrics only, never zone events.
        self.crowd_enabled: bool = False
        self.crowd_source: str = "reuse"
        self.crowd_detector = None
        self.crowd_analysis_fps: float = 2.0
        self.crowd_confidence: float = 0.35
        self._crowd_last_run: Optional[float] = None
        self.crowd_metrics: dict = {
            "person_count": 0,
            "coverage_fraction": 0.0,
            "density_per_megapixel": 0.0,
        }
        # PPE side path (additive, default off). Grade only, never events;
        # the detector stays None unless start() enables it.
        self.ppe_enabled: bool = False
        self.ppe_detector = None
        self.ppe_analysis_fps: float = 1.0
        self.ppe_confidence: float = 0.25
        self.ppe_iou: float = 0.45
        self.ppe_imgsz: int = 640
        self.ppe_tta: bool = False
        self.ppe_memory_seconds: float = 4.0
        self.ppe_medium_ratio: float = 0.25
        self.ppe_high_ratio: float = 0.5
        self.ppe_critical_ratio: float = 1.0
        self._ppe_last_run: Optional[float] = None
        # violation memory (class -> (video ts, det)): classes seen within
        # the memory window keep counting. Cleared on reset/replay/start.
        self._ppe_memory: dict = {}
        self.ppe_stats: dict = {
            "analyzed": 0,
            "grade": "none",
            "violation_ratio": 0.0,
            "violation_count": 0,
            "person_count": 0,
            "fall_count": 0,
            "other_count": 0,
            "violations_by_class": [],
        }
        # Background side-analysis workers (additive). Two threads, each with
        # its own latest-frame slot: "fire" runs fire/smoke alone so a long
        # PPE run can never delay a fire/smoke sample; "slow" runs
        # crowd-model vs PPE in priority order. Each thread exists only
        # while its paths are enabled; each slot holds at most one frame
        # (newest replaces stale, never queues). `_config_lock` guards
        # pick/store sections (never inference); the slot lock guards the
        # pointer swaps only, so the loop never waits.
        self._side_slot_lock = threading.Lock()
        self._fire_slot = None
        self._fire_event = threading.Event()
        self._fire_stop = threading.Event()
        self._fire_thread = None
        self._slow_slot = None
        self._slow_event = threading.Event()
        self._slow_stop = threading.Event()
        self._slow_thread = None
        self._side_errors = 0
        self._crowd_last_frame = None
        self._crowd_last_capture = None

        # rolling per-stage timings, exposed through state() as "perf"
        self._perf_lock = threading.Lock()
        self._perf: dict[str, deque] = {
            "inference": deque(maxlen=PERF_WINDOW),
            "draw": deque(maxlen=PERF_WINDOW),
            "encode": deque(maxlen=PERF_WINDOW),
            "frame": deque(maxlen=PERF_WINDOW),
        }

    def _reset_perf(self) -> None:
        with self._perf_lock:
            for samples in self._perf.values():
                samples.clear()

    @staticmethod
    def _warm_detector(label: str, detector) -> None:
        """One blank inference so the first loop does not pay cold-start.

        Model load is fast; the first real inference is not (one-time
        graph/thread/cache setup on the CPU, contended with the loop). A
        64x64 blank is enough to shake that out and costs a fraction of a
        second. Best effort: a failure only logs, it never breaks startup
        (the loop logs per-frame failures the same way).
        """
        try:
            import numpy as np

            detector.predict(np.zeros((64, 64, 3), dtype=np.uint8))
        except Exception as exc:
            LOG.debug("%s warm-up inference skipped: %s", label, exc)

    def perf(self) -> Optional[dict]:
        """Averages over the last PERF_WINDOW frames, or None before the first."""
        with self._perf_lock:
            frame = self._perf["frame"]
            if not frame:
                return None

            def average(stage: str) -> float:
                samples = self._perf[stage]
                return sum(samples) / len(samples) if samples else 0.0

            ms_per_frame = sum(frame) / len(frame)
            # fps is measured from the real frame cadence, so a skipped
            # inference frame still counts towards the frame rate
            return {
                "fps": round(1000.0 / ms_per_frame, 2) if ms_per_frame > 0 else 0.0,
                "ms_per_frame": round(ms_per_frame, 2),
                "inference_ms": round(average("inference"), 2),
                "draw_ms": round(average("draw"), 2),
                "encode_ms": round(average("encode"), 2),
            }

    # -- lifecycle --------------------------------------------------------

    def start(
        self,
        source: str,
        machine: dict,
        points: Optional[list] = None,
        allowed_people: int = 0,
        mode: str = "TIMER",
        dwell_seconds: float = 5.0,
        repeat_alert_seconds: float = 10.0,
        model: str = "yolo11n.pt",
        confidence: float = 0.25,
        person_class_id: int = 0,
        imgsz: int = 320,
        detect_every_n: int = 1,
        max_detect_long_side: int = 960,
        openvino: bool = False,
        adaptive_idle_n: int = 0,
        person_confirm_conf: float = 0.4,
        person_confirm_frames: int = 3,
        person_instant_conf: float = 0.6,
        person_confirm_move_ratio: float = 0.05,
        anchor: str = "bottom_center",
        zone_reference: tuple = (0, 0),
        realtime: bool = True,
        loop: bool = True,
        zones: Optional[list] = None,
        zone_specs: Optional[list] = None,
        zone_margin_px: float = 5.0,
        fire_smoke_enabled: bool = False,
        fire_smoke_model: str = "detector/models/fire_smoke_v2.pt",
        fire_smoke_confidence: float = 0.35,
        fire_smoke_imgsz: int = 640,
        fire_smoke_fire_seconds: float = 1.5,
        fire_smoke_smoke_seconds: float = 2.5,
        fire_smoke_analysis_fps: float = 2.0,
        fire_high_confidence: float = 0.8,
        fire_person_containment: float = 0.5,
        crowd_enabled: bool = False,
        crowd_source: str = "reuse",
        crowd_model: str = "detector/models/crowd_v1.pt",
        crowd_confidence: float = 0.35,
        crowd_imgsz: int = 640,
        crowd_analysis_fps: float = 2.0,
        ppe_enabled: bool = False,
        ppe_model: str = "detector/models/ppe_v1.pt",
        ppe_confidence: float = 0.15,
        ppe_iou: float = 0.45,
        ppe_imgsz: int = 960,
        ppe_tta: bool = True,
        ppe_analysis_fps: float = 1.0,
        ppe_memory_seconds: float = 4.0,
        ppe_medium_ratio: float = 0.25,
        ppe_high_ratio: float = 0.5,
        ppe_critical_ratio: float = 1.0,
        hazard_enabled: bool = True,
        fire_enabled: Optional[bool] = None,
        smoke_enabled: Optional[bool] = None,
    ) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                raise PipelineError("monitoring is already running")

            # `zones` is a list of polygons; `points` is the old single zone and
            # is still accepted so the CLI and older callers keep working
            try:
                polygons = _clean_polygons(zones if zones is not None else points)
            except ZoneError as exc:
                raise PipelineError(str(exc)) from exc

            # fail fast on a bad video path; an unavailable camera can only be
            # detected once the capture is actually opened in the thread
            source_text = str(source).strip()
            if not source_text:
                raise PipelineError("no video source given")
            if not source_text.isdigit() and not os.path.isfile(source_text):
                raise PipelineError(f"video file not found: {source_text}")

            self._stop.clear()
            self.error = None
            # already under self._lock here (it guards the whole start), so
            # this is a plain assignment, never a nested acquire: Lock is
            # not reentrant and would deadlock.
            self.finished_clip = False
            self.source = str(source)
            self.machine = dict(machine)
            self.rules = {"mode": mode, "dwell_seconds": float(dwell_seconds)}
            self.rules["repeat_alert_seconds"] = float(repeat_alert_seconds)
            self.zone_polygons = polygons
            # rules per zone, index aligned with the shapes. None means the
            # monitor gives every zone the global numbers, which is exactly how
            # a single zone behaves.
            self.zone_specs = _clean_zone_specs(zone_specs, len(polygons), allowed_people, dwell_seconds)
            self.zone_reference = (int(zone_reference[0]), int(zone_reference[1]))
            if str(anchor) not in ANCHORS:
                raise PipelineError(f"unsupported anchor point: {anchor}")
            self.anchor = str(anchor)

            try:
                zone = DangerZone(zones=polygons)
                self.monitor = ZoneMonitor(
                    zone=zone,
                    machine_id=self.machine.get("id", "machine"),
                    mode=mode,
                    dwell_seconds=dwell_seconds,
                    allowed_people=allowed_people,
                    repeat_alert_seconds=repeat_alert_seconds,
                    zone_rules=self.zone_specs or None,
                    zone_margin_px=zone_margin_px,
                )
            except ZoneError as exc:
                raise PipelineError(str(exc)) from exc

            try:
                self.tracker = PersonTracker(
                    model_name=model,
                    confidence=confidence,
                    person_class_id=person_class_id,
                    imgsz=imgsz,
                    detect_every_n=detect_every_n,
                    max_detect_long_side=max_detect_long_side,
                    openvino=openvino,
                    adaptive_idle_n=adaptive_idle_n,
                    person_confirm_conf=person_confirm_conf,
                    person_confirm_frames=person_confirm_frames,
                    person_instant_conf=person_instant_conf,
                    person_confirm_move_ratio=person_confirm_move_ratio,
                )
            except DetectorError as exc:
                raise PipelineError(str(exc)) from exc

            # a fresh writer per run: the old one is closed in stop(), so a
            # restart never leaks a thread or writes into a stale database
            self.hazard_enabled = bool(hazard_enabled)
            # Fire/smoke side path: the model loads when at least one per-type
            # switch is on (fail fast like the person tracker above). Heavy
            # imports stay inside this branch, so a disabled start imports
            # nothing new. An explicit per-type switch wins; otherwise each
            # type follows the shared switch, so older callers behave as before.
            if fire_enabled is None and smoke_enabled is None:
                fire_enabled = smoke_enabled = bool(fire_smoke_enabled)
            self.fire_enabled = bool(fire_enabled)
            self.smoke_enabled = bool(smoke_enabled)
            self.fire_high_confidence = float(fire_high_confidence)
            self.fire_person_containment = float(fire_person_containment)
            self.fire_smoke = None
            self.fire_smoke_gate = None
            self.fire_smoke_emitter = None
            self.fire_smoke_camera_id = ""
            self.fire_smoke_stats = {
                "analyzed": 0,
                "emitted": 0,
                "suppressed": 0,
                "last_types": [],
                "last_crossed": [],
            }
            self._fire_smoke_last_run = None
            if self.fire_enabled or self.smoke_enabled:
                if float(fire_smoke_analysis_fps) <= 0:
                    raise PipelineError("fire/smoke analysis_fps must be greater than 0")
                if float(fire_smoke_fire_seconds) <= 0 or float(fire_smoke_smoke_seconds) <= 0:
                    raise PipelineError("fire/smoke persistence seconds must be greater than 0")
                try:
                    from detector.fire_smoke_detector import FireSmokeDetector, FireSmokeError
                    from detector.fire_smoke_gate import FireSmokeGate
                    from services.vision_alerts import VisionAlertEmitter

                    self.fire_smoke = FireSmokeDetector(
                        model_path=str(fire_smoke_model),
                        confidence=float(fire_smoke_confidence),
                        imgsz=int(fire_smoke_imgsz),
                    )
                except FireSmokeError as exc:
                    raise PipelineError(str(exc)) from exc
                self.fire_smoke_gate = FireSmokeGate(
                    fire_seconds=float(fire_smoke_fire_seconds),
                    smoke_seconds=float(fire_smoke_smoke_seconds),
                    default_fps=float(fire_smoke_analysis_fps),
                )
                self.fire_smoke_camera_id = str(self.machine.get("id", "machine"))
                self.fire_smoke_gate.set_camera_fps(
                    self.fire_smoke_camera_id, float(fire_smoke_analysis_fps)
                )
                self.fire_smoke_analysis_fps = float(fire_smoke_analysis_fps)
                self.fire_smoke_confidence = float(fire_smoke_confidence)
                self.fire_smoke_emitter = VisionAlertEmitter()
                self._warm_detector("fire/smoke", self.fire_smoke)

            # Crowd side path: validated first, built only when enabled.
            # "reuse" needs no model (metrics come from the person tracker
            # below, the DMS zero-cost route); "model" loads crowd_v1.pt
            # lazily here and fails fast like the trackers above.
            self.crowd_enabled = bool(crowd_enabled)
            self.crowd_source = str(crowd_source or "reuse").lower()
            if self.crowd_source not in ("reuse", "model"):
                raise PipelineError("crowd source must be 'reuse' or 'model'")
            self.crowd_detector = None
            self.crowd_analysis_fps = float(crowd_analysis_fps)
            self.crowd_confidence = float(crowd_confidence)
            self._crowd_last_run = None
            self.crowd_metrics = {
                "person_count": 0,
                "coverage_fraction": 0.0,
                "density_per_megapixel": 0.0,
            }
            if self.crowd_enabled and self.crowd_source == "model":
                if float(crowd_analysis_fps) <= 0:
                    raise PipelineError("crowd analysis_fps must be greater than 0")
                try:
                    from detector.crowd_detector import CrowdDetector, CrowdError

                    self.crowd_detector = CrowdDetector(
                        model_path=str(crowd_model),
                        confidence=float(crowd_confidence),
                        imgsz=int(crowd_imgsz),
                    )
                    self._warm_detector("crowd", self.crowd_detector)
                except CrowdError as exc:
                    raise PipelineError(str(exc)) from exc

            # PPE side path: built only when enabled (fail fast like the
            # trackers above). Heavy imports stay inside this branch.
            self.ppe_enabled = bool(ppe_enabled)
            self.ppe_detector = None
            self.ppe_analysis_fps = float(ppe_analysis_fps)
            self.ppe_confidence = float(ppe_confidence)
            self.ppe_iou = float(ppe_iou)
            self.ppe_imgsz = int(ppe_imgsz)
            self.ppe_tta = bool(ppe_tta)
            self.ppe_memory_seconds = float(ppe_memory_seconds)
            self.ppe_medium_ratio = float(ppe_medium_ratio)
            self.ppe_high_ratio = float(ppe_high_ratio)
            self.ppe_critical_ratio = float(ppe_critical_ratio)
            self._ppe_last_run = None
            self._ppe_memory = {}
            self.ppe_stats = {
                "analyzed": 0,
                "grade": "none",
                "violation_ratio": 0.0,
                "violation_count": 0,
                "person_count": 0,
                "fall_count": 0,
                "other_count": 0,
                "violations_by_class": [],
            }
            if self.ppe_enabled:
                if float(ppe_analysis_fps) <= 0:
                    raise PipelineError("ppe analysis_fps must be greater than 0")
                try:
                    from detector.ppe_detector import PPEError, PPEDetector

                    self.ppe_detector = PPEDetector(
                        model_path=str(ppe_model),
                        confidence=float(ppe_confidence),
                        iou_threshold=float(ppe_iou),
                        imgsz=int(ppe_imgsz),
                        tta=bool(ppe_tta),
                    )
                    self._warm_detector("ppe", self.ppe_detector)
                except PPEError as exc:
                    raise PipelineError(str(exc)) from exc

            self.snapshots = SnapshotWriter(self.db)
            self.snapshots.start()

            self._thread = threading.Thread(target=self._run, args=(realtime, loop), daemon=True)
            self.running = True
            self._thread.start()

        # outside the instance lock (state() takes config->lock; never invert):
        # run the background worker iff a heavy path was enabled above.
        self._ensure_side_worker()

    def stop(self) -> bool:
        """Stop the video thread. Returns False if it is still alive.

        The thread reference is kept when the join times out (a camera that
        blocks in cap.read cannot be interrupted), so the pipeline never
        pretends it has stopped while frames are still being produced.
        """
        with self._lock:
            thread = self._thread
            self._stop.set()

        # outside the instance lock: the worker never takes it, and joining
        # under it would invert the state() lock order.
        self._stop_side_worker()

        if thread is not None and thread.is_alive():
            thread.join(timeout=5.0)

        still_alive = thread is not None and thread.is_alive()
        with self._lock:
            if not still_alive:
                self._thread = None
                self.running = False
                self._jpeg = None
                # an explicit stop is not a finished clip, even if the file
                # had just reached its end on its own moments earlier
                self.finished_clip = False
                # the picture is gone, so the size is unknown again: a stopped
                # pipeline must not report the frame of the source that has
                # ended, or the next start would scale zones against it
                self.frame_size = (0, 0)
                if self.snapshots is not None:
                    self.snapshots.close()
                    self.snapshots = None
        return not still_alive

    @property
    def active(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def zone_points(self) -> list[tuple[int, int]]:
        """The first zone only, for callers that predate multiple zones."""
        return self.zone_polygons[0] if self.zone_polygons else []

    # -- changing settings without restarting ----------------------------

    def apply_settings(
        self,
        points: Optional[list] = None,
        allowed_people: Optional[int] = None,
        mode: Optional[str] = None,
        dwell_seconds: Optional[float] = None,
        repeat_alert_seconds: Optional[float] = None,
        zone_reference: Optional[tuple] = None,
        zones: Optional[list] = None,
        zone_specs: Optional[list] = None,
        anchor: Optional[str] = None,
        confidence: Optional[float] = None,
        zone_margin_px: Optional[float] = None,
        fire_smoke_enabled: Optional[bool] = None,
        fire_smoke_analysis_fps: Optional[float] = None,
        fire_smoke_fire_seconds: Optional[float] = None,
        fire_smoke_smoke_seconds: Optional[float] = None,
        crowd_enabled: Optional[bool] = None,
        crowd_source: Optional[str] = None,
        crowd_analysis_fps: Optional[float] = None,
        ppe_enabled: Optional[bool] = None,
        ppe_analysis_fps: Optional[float] = None,
        hazard_enabled: Optional[bool] = None,
        fire_enabled: Optional[bool] = None,
        smoke_enabled: Optional[bool] = None,
    ) -> list[int]:
        """Apply new settings to the running pipeline without reopening the source.

        The capture stays open and the frame counter keeps climbing, so the
        browser never sees the video blink. Only the danger zone and the alert
        rules change; the worker list is kept so people are not forgotten.

        Anything not sent (None) keeps its current value, which is how a change
        to only the rules leaves the polygon alone. Returns the ids of the
        workers whose inside/outside status the new zone changed.
        """
        monitor = self.monitor
        if not self.active or monitor is None:
            raise PipelineError("monitoring is not running")

        if mode is not None and str(mode).upper() not in ("INSTANT", "TIMER", "LIMIT"):
            raise PipelineError(f"unsupported alert mode: {mode}")

        if anchor is not None and str(anchor) not in ANCHORS:
            raise PipelineError(f"unsupported anchor point: {anchor}")

        # fire/smoke live values are validated before anything is touched,
        # like every other setting above.
        if fire_smoke_analysis_fps is not None and float(fire_smoke_analysis_fps) <= 0:
            raise PipelineError("fire/smoke analysis_fps must be greater than 0")
        if fire_smoke_fire_seconds is not None and float(fire_smoke_fire_seconds) <= 0:
            raise PipelineError("fire/smoke fire_seconds must be greater than 0")
        if fire_smoke_smoke_seconds is not None and float(fire_smoke_smoke_seconds) <= 0:
            raise PipelineError("fire/smoke smoke_seconds must be greater than 0")

        # crowd live values are validated before anything is touched.
        if crowd_source is not None and str(crowd_source).lower() not in ("reuse", "model"):
            raise PipelineError("crowd source must be 'reuse' or 'model'")
        if crowd_analysis_fps is not None and float(crowd_analysis_fps) <= 0:
            raise PipelineError("crowd analysis_fps must be greater than 0")

        # ppe live values are validated before anything is touched.
        if ppe_analysis_fps is not None and float(ppe_analysis_fps) <= 0:
            raise PipelineError("ppe analysis_fps must be greater than 0")

        # the new zone is built (and therefore rejected) before anything is
        # touched, so a bad polygon can never leave the pipeline half changed
        new_zone: Optional[DangerZone] = None
        polygons: Optional[list] = None
        if zones is not None or points is not None:
            try:
                polygons = _clean_polygons(zones if zones is not None else points)
            except PipelineError as exc:
                raise PipelineError(str(exc)) from exc

            if polygons and any(0 < len(polygon) < 3 for polygon in polygons):
                raise PipelineError("a danger zone needs at least 3 points")

            if polygons:
                # points may have been drawn against another resolution, so they
                # are scaled onto the video that is actually running
                reference = tuple(zone_reference or self.zone_reference)
                polygons = scale_zone_polygons(polygons, reference, self.frame_size)
            try:
                new_zone = DangerZone(zones=polygons)
            except ZoneError as exc:
                raise PipelineError(str(exc)) from exc

        # The rules are built and rejected before anything is touched, so a bad
        # value cannot leave a running camera half changed.
        new_specs = _clean_zone_specs(
            zone_specs,
            len(polygons) if (new_zone is not None and polygons is not None) else len(self.zone_polygons),
            monitor.allowed_people,
            monitor.dwell_seconds,
        )

        with self._config_lock:
            if new_zone is not None:
                monitor.zone = new_zone
                self.zone_polygons = [list(polygon) for polygon in new_zone.polygons]
                if zone_reference:
                    self.zone_reference = (int(zone_reference[0]), int(zone_reference[1]))

            if allowed_people is not None:
                monitor.allowed_people = max(0, int(allowed_people))
            if mode is not None:
                monitor.mode = str(mode).upper()
            if dwell_seconds is not None:
                monitor.dwell_seconds = float(dwell_seconds)
            if repeat_alert_seconds is not None:
                monitor.repeat_alert_seconds = float(repeat_alert_seconds)
            if anchor is not None:
                self.anchor = str(anchor)
            if zone_margin_px is not None:
                monitor.zone_margin_px = max(0.0, float(zone_margin_px))

            # the tracker reads its confidence every frame, so a new value takes
            # effect on the next frame without rebuilding the detector
            if confidence is not None and self.tracker is not None:
                self.tracker.confidence = float(confidence)

            # Hazard switch: zone evaluation and zone events stop immediately
            # when off. Dwell clocks are dropped on either transition, so a
            # stale dwell can neither display nor fire after the change.
            if hazard_enabled is not None:
                self.hazard_enabled = bool(hazard_enabled)
                if monitor is not None:
                    monitor.reset_timing()

            # Fire/smoke live update: per-type switches share one model pass.
            # A legacy fire_smoke_enabled sets both types at once. Any toggle
            # resets the debounce + last result (below), so stale confirming
            # progress can neither display nor fire. Enabling mid-run builds
            # the detector here (blocks for the model load, once); dropping
            # both types drops the detector too — no model, no inference.
            if fire_smoke_enabled is not None and fire_enabled is None and smoke_enabled is None:
                fire_enabled = smoke_enabled = bool(fire_smoke_enabled)
            fire_toggled = fire_enabled is not None or smoke_enabled is not None
            if fire_enabled is not None:
                self.fire_enabled = bool(fire_enabled)
            if smoke_enabled is not None:
                self.smoke_enabled = bool(smoke_enabled)
            if fire_smoke_analysis_fps is not None:
                self.fire_smoke_analysis_fps = float(fire_smoke_analysis_fps)
                if self.fire_smoke_gate is not None:
                    self.fire_smoke_gate.set_camera_fps(
                        self.fire_smoke_camera_id, float(fire_smoke_analysis_fps)
                    )
            seconds_touched = (
                fire_smoke_fire_seconds is not None or fire_smoke_smoke_seconds is not None
            )
            if seconds_touched and self.fire_smoke_gate is not None:
                from detector.fire_smoke_gate import FireSmokeGate

                self.fire_smoke_gate = FireSmokeGate(
                    fire_seconds=(
                        float(fire_smoke_fire_seconds)
                        if fire_smoke_fire_seconds is not None
                        else float(self.fire_smoke_gate._fire_seconds)
                    ),
                    smoke_seconds=(
                        float(fire_smoke_smoke_seconds)
                        if fire_smoke_smoke_seconds is not None
                        else float(self.fire_smoke_gate._smoke_seconds)
                    ),
                    default_fps=self.fire_smoke_analysis_fps,
                )
                self.fire_smoke_gate.set_camera_fps(
                    self.fire_smoke_camera_id, self.fire_smoke_analysis_fps
                )
            if not (self.fire_enabled or self.smoke_enabled):
                self.fire_smoke = None
                self.fire_smoke_gate = None
                self.fire_smoke_emitter = None
                self._fire_smoke_last_run = None
            elif self.fire_smoke is None:
                from detector.fire_smoke_detector import FireSmokeDetector, FireSmokeError
                from detector.fire_smoke_gate import FireSmokeGate
                from services.vision_alerts import VisionAlertEmitter

                try:
                    self.fire_smoke = FireSmokeDetector(
                        confidence=self.fire_smoke_confidence,
                    )
                except FireSmokeError as exc:
                    raise PipelineError(str(exc)) from exc
                self._warm_detector("fire/smoke", self.fire_smoke)
                self.fire_smoke_gate = FireSmokeGate(
                    default_fps=self.fire_smoke_analysis_fps
                )
                self.fire_smoke_gate.set_camera_fps(
                    self.fire_smoke_camera_id or str(self.machine.get("id", "machine")),
                    self.fire_smoke_analysis_fps,
                )
                if not self.fire_smoke_camera_id:
                    self.fire_smoke_camera_id = str(self.machine.get("id", "machine"))
                self.fire_smoke_emitter = VisionAlertEmitter()
                self._fire_smoke_last_run = None
            if fire_toggled or seconds_touched:
                self._reset_fire_state()

            # Crowd live update (additive): disabling drops the side path;
            # source/fps apply immediately; switching to "model" mid-run
            # builds the detector here (blocks for the model load, once).
            if crowd_enabled is not None and not crowd_enabled:
                self.crowd_enabled = False
                self.crowd_detector = None
                self._crowd_last_run = None
                self._reset_crowd_state()
            if crowd_source is not None:
                self.crowd_source = str(crowd_source).lower()
                if self.crowd_source == "reuse":
                    self.crowd_detector = None
            if crowd_analysis_fps is not None:
                self.crowd_analysis_fps = float(crowd_analysis_fps)
            if crowd_enabled:
                self.crowd_enabled = True
                if self.crowd_source == "model" and self.crowd_detector is None:
                    from detector.crowd_detector import CrowdDetector, CrowdError

                    try:
                        self.crowd_detector = CrowdDetector(
                            confidence=self.crowd_confidence,
                        )
                    except CrowdError as exc:
                        raise PipelineError(str(exc)) from exc
                    self._warm_detector("crowd", self.crowd_detector)
                    self._crowd_last_run = None
                self._reset_crowd_state()

            # PPE live update (additive): disabling drops the side path; fps
            # applies immediately; enabling mid-run builds the detector here
            # (blocks for the model load, once).
            if ppe_enabled is not None and not ppe_enabled:
                self.ppe_enabled = False
                self.ppe_detector = None
                self._ppe_last_run = None
                self._reset_ppe_state()
            if ppe_analysis_fps is not None:
                self.ppe_analysis_fps = float(ppe_analysis_fps)
            if ppe_enabled:
                self.ppe_enabled = True
                if self.ppe_detector is None:
                    from detector.ppe_detector import PPEError, PPEDetector

                    try:
                        self.ppe_detector = PPEDetector(
                            confidence=self.ppe_confidence,
                            iou_threshold=self.ppe_iou,
                            imgsz=self.ppe_imgsz,
                            tta=self.ppe_tta,
                        )
                    except PPEError as exc:
                        raise PipelineError(str(exc)) from exc
                    self._warm_detector("ppe", self.ppe_detector)
                    self._ppe_last_run = None
                self._reset_ppe_state()

            # Global numbers first, so a zone still following the defaults picks
            # up the new value; then the zones that have rules of their own.
            monitor.apply_defaults(
                allowed_people=allowed_people,
                dwell_seconds=dwell_seconds,
                zone_rules=new_specs or None,
            )

            if zone_specs is not None or new_zone is not None:
                self.zone_specs = [
                    {
                        "name": rule.name,
                        "allowed_people": rule.allowed_people,
                        "dwell_seconds": rule.dwell_seconds,
                        "enabled": rule.enabled,
                        "using_defaults": rule.using_defaults,
                    }
                    for rule in monitor.zones
                ]

            self.rules = {
                "mode": monitor.mode,
                "dwell_seconds": monitor.dwell_seconds,
                "repeat_alert_seconds": monitor.repeat_alert_seconds,
            }

            # whoever the new zone moved in or out starts timing from zero, so a
            # stale dwell clock cannot fire an alert right after the change. This
            # is also what silently clears a zone that was switched off, which is
            # the right behaviour: it is like redrawing it.
            if new_zone is not None or zone_specs is not None:
                changed = monitor.reset_timing()
            else:
                changed = []

        # outside the config lock (it may join a stopping worker): reconcile
        # the background worker with the new flags.
        self._ensure_side_worker()
        return changed

    # -- fire/smoke side path (additive, default off) ---------------------

    @staticmethod
    def _analysis_due(analysis_fps: float, last_run, now_m: float) -> bool:
        """Peek whether a time-paced analysis is due (never consumes)."""
        if last_run is None:
            return True
        interval = 1.0 / analysis_fps if analysis_fps > 0 else 0
        if interval <= 0:
            return True
        return (now_m - last_run) >= interval

    def tracker_needed(self) -> bool:
        """True when anything needs person boxes this frame.

        The tracker runs for hazard evaluation, for crowd-reuse metrics, and
        for the PPE person count — never for its own sake. Read lock-free on
        plain flags (worst case one frame uses a just-changed value).
        """
        return bool(
            self.hazard_enabled
            or (self.crowd_enabled and self.crowd_source == "reuse")
            or self.ppe_enabled
        )

    def _maybe_run_fire_smoke(self, frame, now_m: float, frame_index=None):
        """Analyze at most at `fire_smoke_analysis_fps`, paced by wall time.

        Returns None when disabled or when this frame is skipped by cadence,
        [] when analyzed with nothing crossing, or the crossed alert types.
        `now_m` is the pacing clock: the video capture timestamp in the
        loop/worker, injectable for tests. `frame_index` (when known) is
        stamped onto the result alongside the capture timestamp.
        """
        with self._config_lock:
            detector = self.fire_smoke
            gate = self.fire_smoke_gate
            if detector is None or gate is None:
                return None
            if not self._analysis_due(self.fire_smoke_analysis_fps,
                                      self._fire_smoke_last_run, now_m):
                return None
            self._fire_smoke_last_run = now_m
        return self._run_fire_smoke(frame, capture_ts=now_m, frame_index=frame_index)

    def _run_fire_smoke(self, frame, capture_ts=None, frame_index=None) -> list:
        """One analyzed frame: predict -> gate -> emit. Never raises.

        The gate and the stats use `capture_ts` (video time of the analyzed
        frame), never completion time. Models are used by at most one thread
        (the side worker in the loop); shared state is guarded.
        """
        from detector.fire_smoke_gate import build_vision_alert_payload, extract_fire_smoke

        with self._config_lock:
            detector = self.fire_smoke
            confidence = self.fire_smoke_confidence
            want = {"fire"} if self.fire_enabled else set()
            if self.smoke_enabled:
                want.add("smoke")
        if detector is None or not want:
            return []
        try:
            raw = detector.predict(
                frame, confidence_threshold=confidence
            )
            kept = extract_fire_smoke(raw)
        except Exception as exc:
            LOG.warning("fire/smoke inference skipped: %s", exc)
            return []
        # an unticked type is dropped before the gate ever sees it: no
        # counting, no alerts, no boxes, no confirming progress for it.
        kept = [d for d in kept if d["vision_alert_type"] in want]
        kept = self._drop_person_contained_fire(kept)
        present = sorted({d["vision_alert_type"] for d in kept})
        with self._config_lock:
            gate = self.fire_smoke_gate
            camera_id = self.fire_smoke_camera_id
            emitter = self.fire_smoke_emitter
            if gate is None:
                return []
            try:
                crossed = gate.update(camera_id, present)
            except Exception as exc:
                LOG.warning("fire/smoke gate skipped: %s", exc)
                return []
            payloads = []
            if crossed and emitter is not None:
                by_type = {}
                for det in kept:  # kept is confidence-sorted, so first per type wins
                    by_type.setdefault(det["vision_alert_type"], det)
                for alert_type in crossed:
                    rep = by_type.get(alert_type)
                    if rep is None:
                        continue
                    payloads.append(build_vision_alert_payload(
                        camera=camera_id,
                        vision_alert_type=alert_type,
                        confidence=rep["confidence"],
                        bbox_xyxy=list(rep["bbox_xyxy"]),
                        track_id=None,
                    ))
        emitted = 0
        for payload in payloads:
            try:
                emitter.emit(payload)
                emitted += 1
            except Exception as exc:
                LOG.warning("fire/smoke emit skipped: %s", exc)
        alert_ids = self._record_fire_alerts(frame, crossed, by_type if crossed else {},
                                              capture_ts)
        with self._config_lock:
            if self.fire_smoke_gate is None:
                return []
            self.fire_smoke_stats["analyzed"] += 1
            self.fire_smoke_stats["emitted"] += emitted
            self.fire_smoke_stats["last_types"] = present
            self.fire_smoke_stats["last_crossed"] = list(crossed)
            self.fire_smoke_stats["last_frame_index"] = frame_index
            self.fire_smoke_stats["last_capture_ts"] = capture_ts
            self.fire_smoke_stats["last_boxes"] = [
                {"vision_alert_type": d["vision_alert_type"],
                 "bbox_xyxy": list(d["bbox_xyxy"]),
                 "confidence": d["confidence"]}
                for d in kept[:20]
            ]
            return list(crossed)

    def _drop_person_contained_fire(self, dets: list) -> list:
        """Drop FIRE detections explained by a tracked worker.

        Hi-vis vests read as flames for a frame or two (measured 100%
        containment, PPE.mp4 f103-118); a box that far inside a person box
        never reaches the gate, the boxes, or the alerts. Smoke is
        untouched. Never raises; without tracked people everything is kept.
        """
        fires = [d for d in dets
                 if str(d.get("vision_alert_type", "")).strip().casefold() == "fire"]
        if not fires:
            return dets
        with self._config_lock:
            people = [list(box) for box in self._last_people_boxes]
            containment = float(self.fire_person_containment)
        if not people:
            return dets
        out = []
        dropped = 0
        for det in dets:
            if (str(det.get("vision_alert_type", "")).strip().casefold() == "fire"
                    and _box_contained(det.get("bbox_xyxy"), people,
                                       containment)):
                dropped += 1
                continue
            out.append(det)
        if dropped:
            with self._config_lock:
                self.fire_smoke_stats["suppressed"] = \
                    self.fire_smoke_stats.get("suppressed", 0) + dropped
            LOG.debug("fire suppressed by person overlap: %d box(es)", dropped)
        return out

    def _record_fire_alerts(self, frame, crossed: list, by_type: dict,
                            capture_ts) -> list:
        """Store one DB row + annotated + clean snapshots per crossed type.

        Best effort and never raising: a full disk or a locked database must
        not lose the detection (it was still emitted above). Returns the new
        row ids. `frame` is the worker's own clean pre-overlay copy, which is
        what the clean snapshot stores verbatim for hard-negative retraining.
        """
        import time as _time

        from detector.fire_smoke_gate import grade_fire_smoke
        from snapshots import write_fire_clean_snapshot, write_fire_snapshot
        from visualize import draw_fire_box, draw_smoke_box

        try:
            import cv2

            success, clean_jpeg = cv2.imencode(
                ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            clean_bytes = bytes(clean_jpeg.tobytes()) if success else b""
        except Exception as exc:
            LOG.warning("fire clean snapshot encode skipped: %s", exc)
            clean_bytes = b""

        ids: list = []
        for alert_type in crossed:
            rep = by_type.get(alert_type)
            if rep is None:
                continue
            try:
                import cv2

                annotated = frame.copy()
                label = f"{alert_type.upper()} {float(rep['confidence']):.2f}"
                if alert_type == "fire":
                    draw_fire_box(annotated, rep["bbox_xyxy"], label)
                else:
                    draw_smoke_box(annotated, rep["bbox_xyxy"], label)
                success, jpeg = cv2.imencode(
                    ".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                jpeg_bytes = bytes(jpeg.tobytes()) if success else b""
            except Exception as exc:
                LOG.warning("fire snapshot encode skipped: %s", exc)
                jpeg_bytes = b""
            try:
                with self._config_lock:
                    high_confidence = float(self.fire_high_confidence)
                severity, _risk = grade_fire_smoke(alert_type, float(rep["confidence"]),
                                                   high_confidence=high_confidence)
                alert_id = self.db.insert_fire_alert(
                    alert_type=alert_type,
                    severity=severity,
                    confidence=float(rep["confidence"]),
                    bbox_xyxy=list(rep["bbox_xyxy"]),
                    captured_at=capture_ts,
                    reported_at=_time.time(),
                    # One running source only, so the machine it watches is
                    # the label: stored now, shown in every UI/API surface.
                    source_label=str(
                        self.machine.get("name") or self.machine.get("id") or ""
                    ) or None,
                )
            except Exception as exc:
                LOG.warning("fire alert row skipped: %s", exc)
                continue
            if jpeg_bytes:
                try:
                    name = write_fire_snapshot(alert_id, jpeg_bytes)
                    if name:
                        self.db.set_fire_snapshot_path(alert_id, name)
                except Exception as exc:
                    LOG.warning("fire snapshot skipped: %s", exc)
            if clean_bytes:
                try:
                    clean_name = write_fire_clean_snapshot(alert_id, clean_bytes)
                    if clean_name:
                        self.db.set_fire_clean_snapshot_path(alert_id, clean_name)
                except Exception as exc:
                    LOG.warning("fire clean snapshot skipped: %s", exc)
            ids.append(alert_id)
        return ids

    def _fire_boxes_for_draw(self) -> list:
        """Latest analyzed fire/smoke boxes for the MJPEG overlay (copies).

        Empty unless the path is enabled. The boxes are from the latest
        analysis, so up to one cadence older than the person boxes.
        """
        with self._config_lock:
            if self.fire_smoke is None:
                return []
            return [dict(box) for box in self.fire_smoke_stats.get("last_boxes", [])]

    def _reset_for_replay(self) -> None:
        """Restart pacing stamps + fire/smoke debounce for a looped replay.

        The capture clock resets to ~0 on replay while the last-run stamps
        persist, so without this no time-paced analysis (fire/smoke, PPE,
        crowd-model) is ever due again after the first loop. The gate
        debounce resets too: a replayed clip is a new pass, consistent with
        monitor.workers.clear() above, so a still-visible fire re-confirms
        and re-fires one alert per loop — the same as zone alerts, whose
        dwell re-accumulates for the fresh workers every loop. Each row is
        timestamped, so the alert list shows one entry per pass, never
        duplicates of a single moment. Only the replay branch calls this
        (it takes _config_lock itself, so callers must not hold it):
        a webcam clock never goes backwards and a source switch starts a
        fresh pipeline (stamps None, new gate), so both are unaffected.
        """
        with self._config_lock:
            self._reset_fire_state()
            self._ppe_last_run = None
            # a replayed clip is a new pass: the video clock resets, so
            # remembered violations would read as future-dated and leak in
            self._ppe_memory = {}
            self._crowd_last_run = None

    def _reset_fire_state(self) -> None:
        """Clear fire/smoke debounce + last result (call with _config_lock).

        Used whenever the pair toggles or its windows change, so a stale
        confirming count can neither display nor fire. Re-registers the
        camera cadence when the gate still exists (reset drops it).
        """
        if self.fire_smoke_gate is not None:
            self.fire_smoke_gate.reset_camera(self.fire_smoke_camera_id)
            self.fire_smoke_gate.set_camera_fps(
                self.fire_smoke_camera_id, self.fire_smoke_analysis_fps
            )
        self._fire_smoke_last_run = None
        self.fire_smoke_stats["last_types"] = []
        self.fire_smoke_stats["last_crossed"] = []
        self.fire_smoke_stats["last_boxes"] = []
        self.fire_smoke_stats["last_frame_index"] = None
        self.fire_smoke_stats["last_capture_ts"] = None

    def _reset_crowd_state(self) -> None:
        """Zero crowd metrics + stamps (call with _config_lock)."""
        self._crowd_last_run = None
        self.crowd_metrics = {
            "person_count": 0,
            "coverage_fraction": 0.0,
            "density_per_megapixel": 0.0,
        }
        self._crowd_last_frame = None
        self._crowd_last_capture = None

    def _reset_ppe_state(self) -> None:
        """Zero PPE grade stats (call with _config_lock)."""
        self._ppe_last_run = None
        self._ppe_memory = {}
        self.ppe_stats = {
            "analyzed": 0,
            "grade": "none",
            "violation_ratio": 0.0,
            "violation_count": 0,
            "person_count": 0,
            "fall_count": 0,
            "other_count": 0,
            "violations_by_class": [],
            "last_boxes": [],
            "last_frame_index": None,
            "last_capture_ts": None,
        }

    def _ppe_boxes_for_draw(self) -> list:
        """Latest analyzed PPE violation boxes for the MJPEG overlay (copies).

        Empty unless the path is enabled. Grading stays grading-only: drawing
        reads the stored boxes and changes no counts, rows or alerts.
        """
        with self._config_lock:
            if not self.ppe_enabled or self.ppe_detector is None:
                return []
            return [dict(box) for box in self.ppe_stats.get("last_boxes", [])]

    def _confirming_for_state(self):
        """Gate progress filtered to ticked types, for state().

        Call with _config_lock already held (state() holds it). Unticked
        types always read None, so confirming progress can never display
        for a system that is off.
        """
        gate = self.fire_smoke_gate
        if gate is None:
            return None
        progress = gate.progress(self.fire_smoke_camera_id)
        return {
            "fire": progress.get("fire") if self.fire_enabled else None,
            "smoke": progress.get("smoke") if self.smoke_enabled else None,
        }

    def fire_smoke_progress(self) -> Optional[dict]:
        """Per-type ``{samples, required}`` while the gate is confirming.

        Entries read ``None`` when there is nothing to confirm (no samples
        yet, or already crossed and suppressed) and the whole result reads
        ``None`` when the path is off. Read under the existing lock.
        """
        with self._config_lock:
            gate = self.fire_smoke_gate
            if gate is None:
                return None
            return gate.progress(self.fire_smoke_camera_id)

    # -- crowd side path (additive, default off) ---------------------------

    def _crowd_reuse_metrics(self, people, width: int, height: int) -> dict:
        """Metrics from the person tracker's boxes (zero-cost route)."""
        from detector.crowd import calculate_crowd_metrics

        dets = []
        for person in people or ():
            box = getattr(person, "bbox", None)
            if box is None and isinstance(person, dict):
                box = person.get("bbox_xyxy", person.get("bounding_box", person.get("bbox")))
            if box is None:
                continue
            try:
                dets.append({"class_name": "person", "bbox_xyxy": [float(v) for v in box]})
            except (TypeError, ValueError):
                continue
        return calculate_crowd_metrics(dets, width, height)

    def _maybe_run_crowd(self, frame, people, width: int, height: int, now_m: float,
                         frame_index=None):
        """Update crowd metrics; returns the metrics dict or None when off.

        "reuse" derives metrics from this frame's tracker boxes every frame
        (no model, ~free); "model" runs crowd_v1.pt at most at
        `crowd_analysis_fps`, paced by wall time like fire/smoke.
        Annotation only: the return value is stored for /state, never turned
        into zone events. `now_m` is the pacing clock (capture timestamp in
        the loop, injectable); `frame_index` is stamped when known.
        """
        with self._config_lock:
            enabled = self.crowd_enabled
            source = self.crowd_source
            detector = self.crowd_detector
        if not enabled:
            return None
        if source == "reuse" or detector is None:
            metrics = self._crowd_reuse_metrics(people, width, height)
            with self._config_lock:
                self.crowd_metrics = metrics
                self._crowd_last_frame = frame_index
                self._crowd_last_capture = now_m
            return dict(metrics)
        with self._config_lock:
            if not self._analysis_due(self.crowd_analysis_fps,
                                      self._crowd_last_run, now_m):
                return dict(self.crowd_metrics)
            self._crowd_last_run = now_m
        return self._run_crowd_model(frame, width, height,
                                     capture_ts=now_m, frame_index=frame_index)

    def _run_crowd_model(self, frame, width: int, height: int,
                         capture_ts=None, frame_index=None) -> dict:
        """One dedicated crowd_v1.pt pass. Never raises."""
        from detector.crowd import calculate_crowd_metrics

        with self._config_lock:
            detector = self.crowd_detector
            confidence = self.crowd_confidence
        if detector is None:
            with self._config_lock:
                return dict(self.crowd_metrics)
        try:
            dets = detector.predict(
                frame, confidence_threshold=confidence
            )
            metrics = calculate_crowd_metrics(dets, width, height)
        except Exception as exc:
            LOG.warning("crowd inference skipped: %s", exc)
            with self._config_lock:
                return dict(self.crowd_metrics)
        with self._config_lock:
            self.crowd_metrics = metrics
            self._crowd_last_frame = frame_index
            self._crowd_last_capture = capture_ts
            return dict(metrics)

    # -- PPE side path (additive, default off) -------------------------------

    def _maybe_run_ppe(self, frame, people, now_m: float, frame_index=None):
        """Grade at most at `ppe_analysis_fps`, paced by wall time.

        Returns the stats dict, or None when disabled. Person count is
        reused from this frame's tracker boxes (the DMS zero-cost-route
        principle: the tracker is the person authority here); violations,
        falls and compliant gear come from the PPE model pass. Grade only:
        stored for /state, never emitted, never zone events. `frame_index`
        is stamped when known.
        """
        with self._config_lock:
            enabled = self.ppe_enabled
            detector = self.ppe_detector
            if not enabled or detector is None:
                return None
            if not self._analysis_due(self.ppe_analysis_fps, self._ppe_last_run, now_m):
                return dict(self.ppe_stats)
            self._ppe_last_run = now_m
        return self._run_ppe(frame, people, capture_ts=now_m, frame_index=frame_index)

    def _run_ppe(self, frame, people, capture_ts=None, frame_index=None) -> dict:
        """One PPE pass + grade. Never raises."""
        from detector.ppe_risk import VIOLATION_CLASSES, grade_frame

        violation_names = {str(name).strip().casefold() for name in VIOLATION_CLASSES}

        with self._config_lock:
            detector = self.ppe_detector
            confidence = self.ppe_confidence
        if detector is None:
            with self._config_lock:
                return dict(self.ppe_stats)
        try:
            dets = detector.predict(
                frame, confidence_threshold=confidence
            )
        except Exception as exc:
            LOG.warning("ppe inference skipped: %s", exc)
            with self._config_lock:
                return dict(self.ppe_stats)
        person_count = 0
        for person in people or ():
            box = getattr(person, "bbox", None)
            if box is None and isinstance(person, dict):
                box = person.get("bbox_xyxy", person.get("bounding_box", person.get("bbox")))
            if box is not None:
                person_count += 1
        others = [d for d in dets
                  if str(d.get("class_name", "")).strip().casefold() != "person"]
        others.extend({"class_name": "Person", "confidence": 1.0}
                      for _ in range(person_count))
        # Violation memory: a class seen within the window counts and draws
        # even when this frame missed it, so flickering NO-* reads steadily
        # on the tile (the DMS debounces emission the same way). Only with a
        # real timestamp — timeless calls grade live-only — and only classes
        # absent from this frame, so nothing is ever double-counted.
        if capture_ts is not None:
            with self._config_lock:
                try:
                    now_ts = float(capture_ts)
                except (TypeError, ValueError):
                    now_ts = None
                memory_window = float(self.ppe_memory_seconds)
                if now_ts is not None:
                    now_classes = {
                        str(d.get("class_name", "")).strip()
                        for d in dets
                        if str(d.get("class_name", "")).strip().casefold()
                        in violation_names
                    }
                    for det in dets:
                        name = str(det.get("class_name", "")).strip()
                        if (name.casefold() in violation_names
                                and det.get("bbox_xyxy") is not None):
                            self._ppe_memory[name] = (
                                now_ts,
                                {"class_name": det.get("class_name"),
                                 "confidence": float(det.get("confidence", 0.0)),
                                 "bbox_xyxy": [float(v) for v in det["bbox_xyxy"]]},
                            )
                    for name in list(self._ppe_memory):
                        seen_ts, _det = self._ppe_memory[name]
                        age = now_ts - float(seen_ts)
                        if not (0 <= age <= memory_window):
                            del self._ppe_memory[name]
                        elif name not in now_classes:
                            others.append(dict(self._ppe_memory[name][1]))
        with self._config_lock:
            medium_ratio = float(self.ppe_medium_ratio)
            high_ratio = float(self.ppe_high_ratio)
            critical_ratio = float(self.ppe_critical_ratio)
        try:
            result = grade_frame(others, medium_ratio=medium_ratio,
                                 high_ratio=high_ratio,
                                 critical_ratio=critical_ratio)
        except Exception as exc:
            LOG.warning("ppe grade skipped: %s", exc)
            with self._config_lock:
                return dict(self.ppe_stats)
        with self._config_lock:
            self.ppe_stats = {
                "analyzed": self.ppe_stats.get("analyzed", 0) + 1,
                "grade": result.grade,
                "violation_ratio": result.violation_ratio,
                "violation_count": result.violation_count,
                "person_count": result.person_count,
                "fall_count": result.fall_count,
                "other_count": result.other_count,
                # aggregate counts per violation type, no per-worker
                # attribution (lists, so /state stays JSON-serializable)
                "violations_by_class": [[name, count] for name, count in result.violations_by_class],
                # violation detections for the burned-in overlay (copies of
                # the model output, same shape as the fire last_boxes). Built
                # from the graded set, so remembered classes draw too and the
                # tile reads the aggregates above: tile and boxes update on
                # the same analysis. Grading-only: no rows or alerts.
                "last_boxes": [
                    {"class_name": str(det.get("class_name")),
                     "confidence": float(det.get("confidence", 0.0)),
                     "bbox_xyxy": [float(v) for v in det["bbox_xyxy"]]}
                    for det in others
                    if str(det.get("class_name", "")).strip().casefold()
                    in violation_names
                    and det.get("bbox_xyxy") is not None
                ],
                "last_frame_index": frame_index,
                "last_capture_ts": capture_ts,
            }
            return dict(self.ppe_stats)

    def _run_side_paths(self, frame, people, width: int, height: int,
                        now_m: float, frame_index=None):
        """Run due side analyses with at most one heavy model per frame.

        Priority: fire/smoke, crowd-model, PPE. The winner consumes its
        slot; losers stay due and run on a later frame (at most ~1 video
        frame late, which the persistence gates absorb — they only ever
        see analyzed frames). Cheap paths (crowd reuse) always run.
        `now_m` is the capture timestamp (video time) in the loop;
        `frame_index` (when known) is stamped onto results. Never raises.
        """
        try:
            if self.crowd_enabled and (
                self.crowd_source == "reuse" or self.crowd_detector is None
            ):
                self._maybe_run_crowd(frame, people, width, height, now_m,
                                       frame_index=frame_index)
            heavy = None
            if self.fire_smoke is not None and self._analysis_due(
                self.fire_smoke_analysis_fps, self._fire_smoke_last_run, now_m
            ):
                heavy = "fire_smoke"
            elif (
                self.crowd_enabled
                and self.crowd_source == "model"
                and self.crowd_detector is not None
                and self._analysis_due(
                    self.crowd_analysis_fps, self._crowd_last_run, now_m
                )
            ):
                heavy = "crowd"
            elif (
                self.ppe_enabled
                and self.ppe_detector is not None
                and self._analysis_due(
                    self.ppe_analysis_fps, self._ppe_last_run, now_m
                )
            ):
                heavy = "ppe"
            if heavy == "fire_smoke":
                self._maybe_run_fire_smoke(frame, now_m, frame_index=frame_index)
            elif heavy == "crowd":
                self._maybe_run_crowd(frame, people, width, height, now_m,
                                       frame_index=frame_index)
            elif heavy == "ppe":
                self._maybe_run_ppe(frame, people, now_m, frame_index=frame_index)
            return heavy
        except Exception as exc:  # never let the side paths stop the video
            LOG.warning("side-path step failed: %s", exc)
            return None

    # -- background side-analysis workers (additive) -------------------------

    def _fire_on_locked(self) -> bool:
        """True when fire/smoke analysis should run (call with _config_lock).

        The model pass is skipped entirely unless at least one per-type
        switch is on; the detector existing alone is not enough."""
        return self.fire_smoke is not None and (
            self.fire_enabled or self.smoke_enabled
        )

    def _slow_on_locked(self) -> bool:
        """True when crowd-model or PPE is enabled (call with _config_lock)."""
        return (
            (
                self.crowd_enabled
                and self.crowd_source == "model"
                and self.crowd_detector is not None
            )
            or (self.ppe_enabled and self.ppe_detector is not None)
        )

    def _heavy_paths_on_locked(self) -> bool:
        """True when any heavy model path is enabled (call with _config_lock)."""
        return self._fire_on_locked() or self._slow_on_locked()

    def _run_slow_paths(self, frame, people, width: int, height: int,
                        now_m: float, frame_index=None):
        """Run the slow worker's due analyses: at most one per grabbed frame.

        Priority: crowd-model, then PPE. Cheap crowd-reuse always runs (it
        is idempotent with the loop's inline computation). `now_m` is the
        grabbed frame's capture timestamp; `frame_index` is stamped onto
        results. Never raises. (Fire/smoke has its own worker and thread
        and never waits behind these.)
        """
        try:
            if self.crowd_enabled and (
                self.crowd_source == "reuse" or self.crowd_detector is None
            ):
                self._maybe_run_crowd(frame, people, width, height, now_m,
                                       frame_index=frame_index)
            slow = None
            if (
                self.crowd_enabled
                and self.crowd_source == "model"
                and self.crowd_detector is not None
                and self._analysis_due(
                    self.crowd_analysis_fps, self._crowd_last_run, now_m
                )
            ):
                slow = "crowd"
            elif (
                self.ppe_enabled
                and self.ppe_detector is not None
                and self._analysis_due(
                    self.ppe_analysis_fps, self._ppe_last_run, now_m
                )
            ):
                slow = "ppe"
            if slow == "crowd":
                self._maybe_run_crowd(frame, people, width, height, now_m,
                                       frame_index=frame_index)
            elif slow == "ppe":
                self._maybe_run_ppe(frame, people, now_m, frame_index=frame_index)
            return slow
        except Exception as exc:  # never let the side paths stop the video
            LOG.warning("slow-path step failed: %s", exc)
            return None

    def _publish_side_frame(self, frame, people, width: int, height: int,
                            frame_index: int, capture_ts: float) -> bool:
        """Newest-only handoff for the workers. Never blocks on a worker.

        One clean pre-overlay copy is shared by however many slots are
        active (workers only read it): the fire slot when fire/smoke is on,
        the slow slot when crowd-model or PPE is on. Each slot holds at most
        one frame — publishing replaces a stale one, so analyses can never
        queue up. Returns False (publishing nothing) when no heavy path is
        enabled.
        """
        with self._config_lock:
            want_fire = self._fire_on_locked()
            want_slow = self._slow_on_locked()
        if not want_fire and not want_slow:
            return False
        boxes = []
        if want_slow:
            for person in people or ():
                box = getattr(person, "bbox", None)
                if box is None and isinstance(person, dict):
                    box = person.get("bbox_xyxy", person.get("bounding_box",
                                                             person.get("bbox")))
                if box is None:
                    continue
                try:
                    boxes.append([float(v) for v in box])
                except (TypeError, ValueError):
                    continue
        with self._side_slot_lock:
            if want_fire or want_slow:
                shared = frame.copy()
            if want_fire:
                self._fire_slot = (shared, capture_ts, frame_index)
            if want_slow:
                self._slow_slot = (shared, boxes, width, height,
                                   frame_index, capture_ts)
        if want_fire:
            self._fire_event.set()
        if want_slow:
            self._slow_event.set()
        return True

    def _grab_slot(self, kind: str):
        """Swap out a worker's latest frame (None when empty)."""
        with self._side_slot_lock:
            if kind == "fire":
                job, self._fire_slot = self._fire_slot, None
            else:
                job, self._slow_slot = self._slow_slot, None
            return job

    def _fire_run(self) -> None:
        """Fire worker: latest frame only, fire/smoke analysis per due grab."""
        try:
            while not self._fire_stop.is_set():
                self._fire_event.wait(timeout=0.05)
                self._fire_event.clear()
                if self._fire_stop.is_set():
                    break
                job = self._grab_slot("fire")
                if job is None:
                    continue
                frame, capture_ts, frame_index = job
                try:
                    self._maybe_run_fire_smoke(frame, capture_ts,
                                               frame_index=frame_index)
                except Exception as exc:  # one bad frame never kills the worker
                    with self._config_lock:
                        self._side_errors += 1
                    LOG.warning("fire worker job failed (%s): %s",
                                type(exc).__name__, exc)
        except Exception as exc:  # the worker thread never dies silently
            LOG.warning("fire worker stopped unexpectedly (%s): %s",
                        type(exc).__name__, exc)

    def _slow_run(self) -> None:
        """Slow worker: latest frame only, crowd-vs-PPE in priority order."""
        try:
            while not self._slow_stop.is_set():
                self._slow_event.wait(timeout=0.05)
                self._slow_event.clear()
                if self._slow_stop.is_set():
                    break
                job = self._grab_slot("slow")
                if job is None:
                    continue
                frame, boxes, width, height, frame_index, capture_ts = job
                try:
                    people = [{"class_name": "person", "bbox_xyxy": box}
                              for box in boxes]
                    self._run_slow_paths(frame, people, width, height, capture_ts,
                                         frame_index=frame_index)
                except Exception as exc:  # one bad frame never kills the worker
                    with self._config_lock:
                        self._side_errors += 1
                    LOG.warning("slow worker job failed (%s): %s",
                                type(exc).__name__, exc)
        except Exception as exc:  # the worker thread never dies silently
            LOG.warning("slow worker stopped unexpectedly (%s): %s",
                        type(exc).__name__, exc)

    def _ensure_side_worker(self) -> None:
        """Start each worker iff its paths are on; stop it iff none is."""
        with self._config_lock:
            want_fire = self._fire_on_locked()
            want_slow = self._slow_on_locked()
            fire_thread = self._fire_thread
            slow_thread = self._slow_thread
            fire_alive = fire_thread is not None and fire_thread.is_alive()
            slow_alive = slow_thread is not None and slow_thread.is_alive()
        if want_fire and not fire_alive:
            self._start_fire_worker()
        elif not want_fire and fire_alive:
            self._stop_fire_worker()
        if want_slow and not slow_alive:
            self._start_slow_worker()
        elif not want_slow and slow_alive:
            self._stop_slow_worker()

    def _start_fire_worker(self) -> None:
        with self._config_lock:
            thread = self._fire_thread
            if thread is not None and thread.is_alive():
                return
            self._fire_stop.clear()
            self._fire_event.clear()
            self._fire_slot = None
            self._fire_thread = threading.Thread(
                target=self._fire_run, name="side-fire", daemon=True
            )
            self._fire_thread.start()

    def _start_slow_worker(self) -> None:
        with self._config_lock:
            thread = self._slow_thread
            if thread is not None and thread.is_alive():
                return
            self._slow_stop.clear()
            self._slow_event.clear()
            self._slow_slot = None
            self._slow_thread = threading.Thread(
                target=self._slow_run, name="side-slow", daemon=True
            )
            self._slow_thread.start()

    def _stop_fire_worker(self, timeout: float = 5.0) -> bool:
        with self._config_lock:
            thread = self._fire_thread
        if thread is None:
            return True
        self._fire_stop.set()
        self._fire_event.set()  # wake it so it sees the stop promptly
        thread.join(timeout=timeout)
        alive = thread.is_alive()
        if alive:
            LOG.warning("fire worker did not stop in %.1fs", timeout)
        else:
            with self._config_lock:
                if self._fire_thread is thread:
                    self._fire_thread = None
                    self._fire_slot = None
        return not alive

    def _stop_slow_worker(self, timeout: float = 5.0) -> bool:
        with self._config_lock:
            thread = self._slow_thread
        if thread is None:
            return True
        self._slow_stop.set()
        self._slow_event.set()  # wake it so it sees the stop promptly
        thread.join(timeout=timeout)
        alive = thread.is_alive()
        if alive:
            LOG.warning("slow worker did not stop in %.1fs", timeout)
        else:
            with self._config_lock:
                if self._slow_thread is thread:
                    self._slow_thread = None
                    self._slow_slot = None
        return not alive

    def _stop_side_worker(self, timeout: float = 5.0) -> bool:
        """Stop both workers. Kept so stop()/apply() touch one helper."""
        fire_ok = self._stop_fire_worker(timeout=timeout)
        slow_ok = self._stop_slow_worker(timeout=timeout)
        return fire_ok and slow_ok

    # -- the loop ---------------------------------------------------------

    def _run(self, realtime: bool, loop: bool = True) -> None:
        try:
            cap, is_camera = open_source(self.source)
        except ValueError as exc:
            self._fail(str(exc))
            return

        try:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            self.frame_size = (width, height)

            timeline = Timeline(cap, is_camera)
            monitor = self.monitor

            # This is the first moment the real frame size is known. A zone
            # saved against a different resolution is scaled to fit instead of
            # being used unscaled, and the monitor is rebuilt around it.
            if width > 0 and height > 0 and self.zone_polygons:
                scaled = scale_zone_polygons(
                    self.zone_polygons, self.zone_reference, (width, height)
                )
                if scaled != self.zone_polygons:
                    LOG.warning(
                        "danger zone drawn on %dx%d rescaled onto the %dx%d video",
                        self.zone_reference[0],
                        self.zone_reference[1],
                        width,
                        height,
                    )
                    self.zone_polygons = scaled
                    # the per-zone rules ride along: a zone drawn on 1280x720
                    # still has to be the Kiln Platform with the same allowance
                    # once it has been scaled onto a 640x480 camera
                    monitor = ZoneMonitor(
                        zone=DangerZone(zones=scaled),
                        machine_id=self.machine.get("id", "machine"),
                        mode=monitor.mode,
                        dwell_seconds=monitor.dwell_seconds,
                        allowed_people=monitor.allowed_people,
                        repeat_alert_seconds=monitor.repeat_alert_seconds,
                        zone_rules=monitor.zones or None,
                        zone_margin_px=monitor.zone_margin_px,
                    )
                    self.monitor = monitor

            frame_index = 0
            # when the file is replayed, cap.grab() past the end would spin
            started = time.perf_counter()
            self.skipped_frames = 0
            self.detector_errors = 0
            self._reset_perf()

            while not self._stop.is_set():
                frame_started = time.perf_counter()
                ok, frame = cap.read()
                if not ok:
                    if not is_camera and loop:
                        # a video file is replayed so the demo keeps running;
                        # worker state is cleared so the loop starts clean
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        monitor.workers.clear()
                        frame_index = 0
                        started = time.perf_counter()
                        # a replayed clip is a new pass: the capture clock
                        # resets to ~0 while the pacing stamps persist, so
                        # without this no time-paced analysis is ever due
                        # again after the first loop (see _reset_for_replay)
                        self._reset_for_replay()
                        continue
                    if not is_camera and not loop and not self._stop.is_set():
                        # single play reached the end by itself: a clean
                        # finish, not an error and not a stop. The UI shows
                        # "Clip finished" and Start works again.
                        with self._lock:
                            self.finished_clip = True
                    break

                now = timeline.update(cap, frame_index)

                # the person tracker runs only while something needs people
                # (hazard evaluation, crowd-reuse metrics, or PPE grading);
                # otherwise the frame skips inference entirely and draws
                # nothing. The error streak is untouched by skipped frames.
                run_tracker = self.tracker_needed()
                t0 = time.perf_counter()
                if not run_tracker:
                    people = []
                    inference_ms = 0.0
                else:
                    try:
                        people = self.tracker.track(frame, frame_index)
                    except DetectorError as exc:
                        # One bad frame is not worth stopping the video for, so the
                        # frame is skipped. Only a model that keeps failing is
                        # treated as fatal, otherwise a transient error would end
                        # the stream in the middle of a shift.
                        inference_ms = (time.perf_counter() - t0) * 1000.0
                        self.detector_errors += 1
                        with self._perf_lock:
                            self._perf["inference"].append(inference_ms)
                        if self.detector_errors >= self.MAX_CONSECUTIVE_ERRORS:
                            self._fail(
                                f"YOLO failed {self.detector_errors} times in a row: {exc}"
                            )
                            return
                        frame_index += 1
                        self.error = None
                        continue
                    if self.detector_errors:
                        # one good frame clears the streak
                        self.detector_errors = 0
                        self.error = None
                    inference_ms = (time.perf_counter() - t0) * 1000.0

                # the zone step runs under the config lock so a zone change
                # cannot land in the middle of a frame: it either waits for this
                # update_all() to finish or is applied before the next one.
                # the anchor is read here too, so switching Feet for Body centre
                # takes effect on a camera that is already running. With the
                # hazard system off there is no evaluation, no events and no
                # snapshots: workers stays empty and the UI shows "off".
                with self._config_lock:
                    anchor_fn = ANCHORS.get(self.anchor, ANCHORS["bottom_center"])
                    if self.hazard_enabled:
                        detections = {
                            person.worker_id: (anchor_fn(person.bbox), person.bbox, person.confidence)
                            for person in people
                        }
                        workers, events = monitor.update_all(detections, now)
                    else:
                        workers, events = [], []
                alert_ids: list[int] = []
                for event in events:
                    try:
                        event_id = self.db.insert_event(event)
                    except DatabaseError:
                        continue  # logging must not stop the video
                    # only the three alert kinds keep a picture: entry/exit rows
                    # are far more common and a JPEG each would fill the disk
                    if event_id and event.event_type in (ENTRY_ALERT, DWELL_ALERT, CAPACITY_ALERT):
                        alert_ids.append(event_id)

                # Side paths (all default off): hand the latest frame to the
                # background worker (slot of 1, never blocks) and run the
                # cheap crowd-reuse inline. Heavy analyses run in the worker;
                # the loop never waits for them. `now` is the video capture
                # timestamp; results are stamped with it, never wall time.
                try:
                    if self.crowd_enabled and (
                        self.crowd_source == "reuse" or self.crowd_detector is None
                    ):
                        with self._config_lock:
                            self.crowd_metrics = self._crowd_reuse_metrics(
                                people, width, height
                            )
                            self._crowd_last_frame = frame_index
                            self._crowd_last_capture = now
                    self._publish_side_frame(
                        frame, people, width, height, frame_index, now
                    )
                except Exception as exc:  # never let the side paths stop the video
                    LOG.warning("side-path step failed: %s", exc)

                # NOTE: no data/workers.json write here. That file is the CLI's
                # contract with the old dashboard; the web UI reads /state, and
                # writing a file every 5th frame only added disk I/O and could
                # race with app.py.

                # the zone is NOT painted on the JPEG: the browser draws it as an
                # SVG overlay on top of the stream, so painting it here as well
                # would show it twice
                draw_started = time.perf_counter()
                # the web frame stays clean: just the person boxes with a small
                # "#id  dwell s" label. The zone, the capacity strip, the dwell
                # alert and the machine details are HTML around the video, so
                # burning them into every JPEG only made the picture harder to
                # read and cost draw time on each frame.
                #
                # Red is reserved for an active alert: a capacity breach alerts
                # the zone while the people inside are still INSIDE, so their
                # boxes are forced red here instead of staying amber.
                alert_zones = {
                    index for index, on in monitor.capacity_by_zone.items() if on
                }
                # one label registry per frame: person, PPE, fire and smoke
                # labels all claim space from it, so a crowded frame cascades
                # labels down instead of painting them over each other.
                placed: list = []
                # person boxes for the fire worker's vest suppression,
                # snapshotted fresh every frame under the same lock it reads.
                # Bad boxes are skipped, never fatal: drawing must not fail.
                with self._config_lock:
                    people_boxes = []
                    for worker in workers:
                        try:
                            box = getattr(worker, "bbox", None)
                            if box is None:
                                continue
                            people_boxes.append([float(v) for v in box])
                        except (TypeError, ValueError):
                            continue
                    self._last_people_boxes = people_boxes
                for worker in workers:
                    in_alert_zone = worker.state == ALERT or any(
                        index in alert_zones for index in worker.presences
                    )
                    draw_person_box(frame, worker, alert=in_alert_zone, placed=placed)
                # PPE violation boxes, burned in only while that path is on
                # (the helper returns [] otherwise). They sit between person
                # and fire/smoke so every layer stays visible. Grading-only:
                # drawing changes no counts, rows or alerts.
                for box in self._ppe_boxes_for_draw():
                    label = f"{box['class_name'].replace('-', ' ').upper()} {box['confidence']:.2f}"
                    draw_ppe_box(frame, box["bbox_xyxy"], label, placed=placed)
                # fire/smoke boxes, burned in only while that path is on (the
                # helper returns [] otherwise). They trail the person boxes
                # by up to one analysis cadence.
                for box in self._fire_boxes_for_draw():
                    label = f"{box['vision_alert_type'].upper()} {box['confidence']:.2f}"
                    if box["vision_alert_type"] == "fire":
                        draw_fire_box(frame, box["bbox_xyxy"], label, placed=placed)
                    else:
                        draw_smoke_box(frame, box["bbox_xyxy"], label, placed=placed)
                draw_ms = (time.perf_counter() - draw_started) * 1000.0

                encode_started = time.perf_counter()
                success, jpeg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                encode_ms = (time.perf_counter() - encode_started) * 1000.0
                if success:
                    with self._lock:
                        self._jpeg = jpeg.tobytes()
                        self._frame_seq = frame_index
                    if alert_ids and self.snapshots is not None:
                        # the frame the alert was decided on, queued for the
                        # writer thread so the video never waits on the disk
                        for event_id in alert_ids:
                            self.snapshots.submit(event_id, self._jpeg)

                frame_index += 1

                with self._perf_lock:
                    self._perf["inference"].append(inference_ms)
                    self._perf["draw"].append(draw_ms)
                    self._perf["encode"].append(encode_ms)
                    self._perf["frame"].append((time.perf_counter() - frame_started) * 1000.0)

                if realtime and not is_camera:
                    # Play the file at its own speed so the dwell timer matches
                    # wall clock. Only the *remaining* time is slept, and when
                    # processing took longer than the frame budget the next
                    # frame is skipped with grab() instead of adding delay, so
                    # a slow machine drops frames rather than lagging behind.
                    deadline = started + frame_index / fps
                    remaining = deadline - time.perf_counter()
                    if remaining > 0:
                        time.sleep(remaining)
                    else:
                        # behind schedule: drop frames instead of lagging.
                        # grab() advances without decoding, and returns False
                        # at the end of the file (the loop handler then rewinds)
                        skipped = 0
                        while time.perf_counter() < deadline and skipped < self.MAX_SKIP:
                            if not cap.grab():
                                break
                            skipped += 1
                            frame_index += 1
                        self.skipped_frames += skipped
        except Exception as exc:  # never let the thread die silently
            self._fail(f"{type(exc).__name__}: {exc}")
        finally:
            cap.release()
            with self._lock:
                self.running = False
                # drop the last frame so the browser shows "no video" instead of
                # a still image that looks like a live stream
                self._jpeg = None

    def _fail(self, message: str) -> None:
        with self._lock:
            self.error = message
            self.running = False

    # -- published state --------------------------------------------------

    def jpeg(self) -> Optional[bytes]:
        with self._lock:
            return self._jpeg

    def state(self) -> dict:
        perf = self.perf()
        # the config lock is taken first so /state never mixes the old zone with
        # the new rules; it is held for the short zone step of a frame only
        from detector.ppe_risk import PPE_EMISSION_DEFERRED
        with self._config_lock:
            with self._lock:
                running = self.active
                finished_clip = self.finished_clip
                monitor = self.monitor
                status = monitor.status() if monitor else {}
            return {
                "running": running,
                "error": self.error,
                # True after a single-play file ended by itself: the stopped
                # panel says "Clip finished" instead of "Camera stopped".
                "finished_clip": finished_clip,
                "source": self.source,
                "frame_width": self.frame_size[0],
                "frame_height": self.frame_size[1],
                "frame_seq": self._frame_seq,
                "machine": self.machine,
                "rules": self.rules,
                "zone_points": [list(point) for point in monitor.zone.points] if monitor else [],
                "zone_polygons": (
                    [[list(point) for point in polygon] for polygon in monitor.zone.polygons]
                    if monitor
                    else []
                ),
                "zone_specs": [
                    {
                        "name": rule.name,
                        "allowed_people": rule.allowed_people,
                        "dwell_seconds": rule.dwell_seconds,
                        "enabled": rule.enabled,
                        "using_defaults": rule.using_defaults,
                    }
                    for rule in (monitor.zones if monitor else [])
                ],
                "zone_reports": monitor.zone_reports() if monitor else [],
                "allowed_people": monitor.allowed_people if monitor else 0,
                "inside": 0 if not self.hazard_enabled else (monitor.inside_count if monitor else 0),
                # Hazard off is its own status, never "SAFE": green must not
                # claim a clear zone when nothing is being evaluated. No
                # events, snapshots or alerts exist while off (the loop
                # skips evaluation), and workers stays empty.
                "status": "off" if not self.hazard_enabled else status.get("status", SAFE),
                "capacity_alert": False if not self.hazard_enabled else status.get("capacity_alert", False),
                "worker_alert": False if not self.hazard_enabled else status.get("worker_alert", False),
                "message": "Hazard monitoring off" if not self.hazard_enabled else status.get("message", ""),
                "workers": [] if not self.hazard_enabled else (monitor.summary() if monitor else []),
                "hazard": {"enabled": bool(self.hazard_enabled)},
                "fire_smoke": (
                    {
                        "enabled": True,
                        "fire_enabled": bool(self.fire_enabled),
                        "smoke_enabled": bool(self.smoke_enabled),
                        "analyzed": self.fire_smoke_stats.get("analyzed", 0),
                        "emitted": self.fire_smoke_stats.get("emitted", 0),
                        "last_types": list(self.fire_smoke_stats.get("last_types", [])),
                        "last_crossed": list(self.fire_smoke_stats.get("last_crossed", [])),
                        "last_frame_index": self.fire_smoke_stats.get("last_frame_index"),
                        "last_capture_ts": self.fire_smoke_stats.get("last_capture_ts"),
                        "confirming": self._confirming_for_state(),
                    }
                    if self.fire_smoke is not None
                    else {"enabled": False, "fire_enabled": bool(self.fire_enabled),
                          "smoke_enabled": bool(self.smoke_enabled)}
                ),
                "crowd": (
                    {
                        "enabled": True,
                        "source": self.crowd_source,
                        "metrics": dict(self.crowd_metrics),
                        "last_frame_index": self._crowd_last_frame,
                        "last_capture_ts": self._crowd_last_capture,
                    }
                    if self.crowd_enabled
                    else {"enabled": False}
                ),
                "ppe": (
                    {
                        "enabled": True,
                        "grade": self.ppe_stats.get("grade", "none"),
                        "analyzed": self.ppe_stats.get("analyzed", 0),
                        "stats": {
                            key: self.ppe_stats.get(key)
                            for key in ("violation_ratio", "violation_count",
                                        "person_count", "fall_count", "other_count",
                                        "violations_by_class")
                        },
                        "last_frame_index": self.ppe_stats.get("last_frame_index"),
                        "last_capture_ts": self.ppe_stats.get("last_capture_ts"),
                        "deferred": PPE_EMISSION_DEFERRED,
                    }
                    if self.ppe_enabled
                    else {"enabled": False, "deferred": PPE_EMISSION_DEFERRED}
                ),
                "perf": perf,
            }
