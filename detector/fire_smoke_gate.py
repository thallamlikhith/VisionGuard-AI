"""Fire/smoke streaming gate + payload (copied from DMS AI_services).

Source: DMS ``AI_services/app/vision/streaming/fire_smoke.py`` (read-only).
Adapted: the DMS ``app.shared.core.config.settings`` import is replaced by
local Prototype settings (``services.vision_settings``): same default
values (fire 1.5 s, smoke 2.5 s, analysis fps 1.0), overridable by
environment / ``config.yaml`` ``fire_smoke`` block. No backend, registry or
heartbeat import remains.

Preserved exactly:
- canonical classes ``fire``/``smoke`` (case-insensitive)
- severity/risk table: fire high/RED else ORANGE, smoke high/ORANGE else
  YELLOW, HIGH_CONFIDENCE = 0.8
- payload keys: camera, vision_alert_type, severity, risk_level,
  confidence, bbox_xyxy, track_id, reported_at (never a dedup key)
- gate semantics: consecutive-frame debounce per (camera, type), first
  crossing emits once, continuous presence suppressed, first negative
  frame re-arms; persistence mode ``ceil(seconds * fps)`` min 1, legacy
  fixed-count ``threshold`` mode kept.
"""

from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Optional

logger = logging.getLogger(__name__)

__all__ = [
    "FIRE_CLASS",
    "SMOKE_CLASS",
    "FIRE_SMOKE_MODEL_KEY",
    "VISION_ALERT_TYPES",
    "SEVERITY_BY_TYPE",
    "HIGH_CONFIDENCE",
    "FireSmokeGate",
    "extract_fire_smoke",
    "grade_fire_smoke",
    "build_vision_alert_payload",
    "build_fire_smoke_predict_fn",
]

FIRE_CLASS = "fire"
SMOKE_CLASS = "smoke"
FIRE_SMOKE_MODEL_KEY = "FIRE_SMOKE"
VISION_ALERT_TYPES = ("fire", "smoke")
HIGH_CONFIDENCE = 0.8
SEVERITY_BY_TYPE = {
    "fire": {"high": "RED", "base": "ORANGE", "risk_level": "high"},
    "smoke": {"high": "ORANGE", "base": "YELLOW", "risk_level": "medium"},
}

# Operating point (must stay exactly as in DMS regression report):
# conf 0.35, imgsz 640, fire gate 1.5 s, smoke gate 2.5 s.
FIRE_SMOKE_CONF_THRESHOLD = 0.35
FIRE_SMOKE_IMGSZ = 640


def _default_settings() -> tuple[float, float, float]:
    """Return (fire_seconds, smoke_seconds, analysis_fps) from local config.

    Precedence: ``services.vision_settings`` when importable (reads
    ``config.yaml`` + env), else the DMS plan defaults. Never imports DMS.
    """
    try:
        from services.vision_settings import get_vision_settings

        current = get_vision_settings()
        return (
            float(current["fire_seconds"]),
            float(current["smoke_seconds"]),
            float(current["analysis_fps"]),
        )
    except Exception:
        return (1.5, 2.5, 1.0)


def _source_fields(det: Any) -> Optional[tuple[str, float, list, Optional[int]]]:
    if isinstance(det, dict):
        class_name = det.get("class_name")
        confidence = det.get("confidence")
        bbox = det.get("bbox_xyxy", det.get("bounding_box"))
        track_id = det.get("track_id")
    else:
        class_name = getattr(det, "class_name", None)
        confidence = getattr(det, "confidence", None)
        bbox = getattr(det, "bbox_xyxy", None)
        if bbox is None:
            bbox = getattr(det, "bounding_box", None)
        track_id = getattr(det, "track_id", None)
    if class_name is None or confidence is None or bbox is None:
        return None
    try:
        conf = float(confidence)
    except (TypeError, ValueError):
        return None
    if not 0.0 <= conf <= 1.0:
        return None
    if isinstance(bbox, dict):
        try:
            box = [float(bbox["x1"]), float(bbox["y1"]),
                   float(bbox["x2"]), float(bbox["y2"])]
        except (KeyError, TypeError, ValueError):
            return None
    else:
        try:
            box = [float(v) for v in bbox]
        except (TypeError, ValueError):
            return None
        if len(box) != 4:
            return None
    x1, y1, x2, y2 = box
    if x2 <= x1 or y2 <= y1:
        return None
    norm_track: Optional[int] = None
    if track_id is not None:
        try:
            norm_track = int(track_id)
        except (TypeError, ValueError):
            norm_track = None
    return str(class_name), conf, box, norm_track


def extract_fire_smoke(detections: Iterable[Any]) -> list[dict]:
    """Keep canonical fire/smoke detections as plain dicts (source pixels)."""
    kept: list[dict] = []
    for det in detections or ():
        fields = _source_fields(det)
        if fields is None:
            continue
        class_name, confidence, box, track_id = fields
        folded = class_name.strip().casefold()
        if folded == FIRE_CLASS:
            alert_type = "fire"
        elif folded == SMOKE_CLASS:
            alert_type = "smoke"
        else:
            continue
        kept.append({
            "vision_alert_type": alert_type,
            "class_name": class_name.strip(),
            "confidence": confidence,
            "bbox_xyxy": box,
            "track_id": track_id,
        })
    kept.sort(key=lambda d: d["confidence"], reverse=True)
    return kept


def grade_fire_smoke(vision_alert_type: str, confidence: float,
                     high_confidence: float = HIGH_CONFIDENCE) -> tuple[str, str]:
    """Initial ``(severity, risk_level)`` grade (see module docstring table)."""
    table = SEVERITY_BY_TYPE[vision_alert_type]
    severity = table["high"] if confidence >= high_confidence else table["base"]
    return severity, table["risk_level"]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_vision_alert_payload(
    *,
    camera: str,
    vision_alert_type: str,
    confidence: float,
    bbox_xyxy: list,
    track_id: Optional[int] = None,
    reported_at: Optional[str] = None,
) -> dict:
    """Build exactly the keys the DMS intake serializer accepts (no dedup)."""
    severity, risk_level = grade_fire_smoke(vision_alert_type, float(confidence))
    return {
        "camera": str(camera),
        "vision_alert_type": vision_alert_type,
        "severity": severity,
        "risk_level": risk_level,
        "confidence": float(confidence),
        "bbox_xyxy": [float(v) for v in bbox_xyxy],
        "track_id": track_id,
        "reported_at": reported_at or utc_now_iso(),
    }


class FireSmokeGate:
    """Consecutive-frame debounce per ``(camera, alert_type)`` (DMS semantics)."""

    def __init__(
        self,
        threshold: Optional[int] = None,
        *,
        fire_seconds: Optional[float] = None,
        smoke_seconds: Optional[float] = None,
        default_fps: Optional[float] = None,
    ) -> None:
        default_fire, default_smoke, default_analysis_fps = _default_settings()
        if threshold is not None:
            if int(threshold) < 1:
                raise ValueError(f"detection threshold must be >= 1, got {threshold!r}")
            self._threshold: Optional[int] = int(threshold)
        else:
            self._threshold = None
        fire_want = default_fire if fire_seconds is None else fire_seconds
        smoke_want = default_smoke if smoke_seconds is None else smoke_seconds
        for label, value in (("fire_seconds", fire_want),
                             ("smoke_seconds", smoke_want)):
            if float(value) <= 0:
                raise ValueError(f"{label} must be > 0, got {value!r}")
        self._fire_seconds = float(fire_want)
        self._smoke_seconds = float(smoke_want)
        fps_want = default_analysis_fps if default_fps is None else default_fps
        if float(fps_want) <= 0:
            raise ValueError(f"default_fps must be > 0, got {fps_want!r}")
        self._default_fps = float(fps_want)
        self._lock = threading.Lock()
        self._counts: dict[tuple[str, str], int] = {}
        self._emitted: set[tuple[str, str]] = set()
        self._fps: dict[str, float] = {}
        self._persistence: dict[str, tuple[Optional[float], Optional[float]]] = {}

    @property
    def threshold(self) -> Optional[int]:
        return self._threshold

    def set_camera_fps(self, camera_id: str, fps: float) -> None:
        fps_value = float(fps)
        if fps_value <= 0:
            raise ValueError(f"analysis fps must be > 0, got {fps!r}")
        with self._lock:
            self._fps[str(camera_id)] = fps_value

    def set_camera_persistence(
        self,
        camera_id: str,
        *,
        fire_seconds: Optional[float] = None,
        smoke_seconds: Optional[float] = None,
    ) -> None:
        for label, value in (("fire_seconds", fire_seconds),
                             ("smoke_seconds", smoke_seconds)):
            if value is not None and float(value) <= 0:
                raise ValueError(f"{label} must be > 0, got {value!r}")
        with self._lock:
            if fire_seconds is None and smoke_seconds is None:
                self._persistence.pop(str(camera_id), None)
            else:
                self._persistence[str(camera_id)] = (
                    float(fire_seconds) if fire_seconds is not None else None,
                    float(smoke_seconds) if smoke_seconds is not None else None,
                )

    def required_frames(self, camera_id: str, alert_type: str) -> int:
        return self._required_locked(str(camera_id), str(alert_type), _lock_held=False)

    def _required_locked(self, camera_id: str, alert_type: str,
                         *, _lock_held: bool = True) -> int:
        def need() -> int:
            if self._threshold is not None:
                return self._threshold
            fps = self._fps.get(camera_id, self._default_fps)
            override = self._persistence.get(camera_id, (None, None))
            if alert_type == "fire":
                seconds = override[0] if override[0] is not None else self._fire_seconds
            else:
                seconds = override[1] if override[1] is not None else self._smoke_seconds
            return max(1, math.ceil(float(seconds) * float(fps)))

        if _lock_held:
            return need()
        with self._lock:
            return need()

    def update(self, camera_id: str, alert_types_present: Iterable[str]) -> list[str]:
        present = {str(t) for t in alert_types_present}
        crossed: list[str] = []
        with self._lock:
            for alert_type in VISION_ALERT_TYPES:
                key = (camera_id, alert_type)
                if alert_type in present:
                    count = self._counts.get(key, 0) + 1
                    self._counts[key] = count
                    if count >= self._required_locked(camera_id, alert_type) \
                            and key not in self._emitted:
                        self._emitted.add(key)
                        crossed.append(alert_type)
                else:
                    self._counts.pop(key, None)
                    self._emitted.discard(key)
        return crossed

    def progress(self, camera_id: str) -> dict:
        """Per-type ``{samples, required}`` for the confirming indicator.

        ``samples`` is the current consecutive count (0 when nothing is
        present); an entry reads ``None`` while there is nothing to confirm
        (no samples yet, or already crossed and suppressed). Read under the
        gate lock so a worker update cannot land mid-read.
        """
        camera_id = str(camera_id)
        with self._lock:
            out = {}
            for alert_type in VISION_ALERT_TYPES:
                key = (camera_id, alert_type)
                samples = self._counts.get(key, 0)
                if samples <= 0 or key in self._emitted:
                    out[alert_type] = None
                    continue
                required = self._required_locked(camera_id, alert_type)
                out[alert_type] = {"samples": samples, "required": required}
            return out

    def reset_camera(self, camera_id: str) -> None:
        with self._lock:
            for alert_type in VISION_ALERT_TYPES:
                self._counts.pop((camera_id, alert_type), None)
                self._emitted.discard((camera_id, alert_type))
            self._fps.pop(camera_id, None)
            self._persistence.pop(camera_id, None)


@dataclass(frozen=True)
class FireSmokeBatchResult:
    camera_id: str
    kept: tuple = ()


def build_fire_smoke_predict_fn(
    *,
    confidence_threshold: float = FIRE_SMOKE_CONF_THRESHOLD,
    loader: Any = None,
    model_name: Optional[str] = None,
) -> Callable[[list], list]:
    """Batch predict fn over the local fire-smoke model (lazy heavy imports)."""
    def _loader() -> Any:
        if loader is not None:
            return loader
        from detector.fire_smoke_detector import get_fire_smoke_detector

        return get_fire_smoke_detector()

    def predict_fn(images: list) -> list:
        model = _loader()
        name = model_name  # reserved: single local model today
        _ = name
        return [model.predict(image, confidence_threshold=confidence_threshold)
                for image in images]

    return predict_fn
