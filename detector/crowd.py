"""Crowd-model capability and metric helpers (copied from DMS AI_services).

Source: DMS ``AI_services/app/vision/crowd.py`` (read-only copy).
Adapted: the DMS ``RawDetection`` / ``CrowdMetricsSchema`` imports are
removed so this module has no DMS dependency; ``calculate_crowd_metrics``
accepts any detection shape (dicts with ``bbox_xyxy`` / ``bounding_box`` /
``bbox``, ``TrackedPerson``-like objects, or DMS-shaped objects) and
returns a plain dict with the same three fields.

Preserved exactly:
- weights file ``crowd_v1.pt``
- class filter: ``class_name.casefold() == "person"`` (matched
  case-insensitively, no fixed class_id — ``crowd_v1.pt`` emits lowercase
  ``person``)
- metrics: person_count, coverage_fraction (bbox area / image area, capped
  at 1.0, rounded to 4), density_per_megapixel (count / MP, rounded to 2)
"""

from __future__ import annotations

from typing import Any

CROWD_MODEL_NAME = "crowd"
CROWD_MODEL_VERSION = "1.0.0"
CROWD_WEIGHTS_FILE = "crowd_v1.pt"
CROWD_CLASSES = ("person",)
PERSON_CLASS_NAME = "person"


def _class_name_of(det: Any) -> str:
    if isinstance(det, dict):
        name = det.get("class_name", det.get("source_class", ""))
        return str(name or "")
    for attr in ("class_name", "source_class"):
        name = getattr(det, attr, None)
        if isinstance(name, str) and name:
            return name
    return ""


def _bbox_of(det: Any):
    """Return (x1, y1, x2, y2) floats, or None when unusable."""
    box = None
    if isinstance(det, dict):
        box = det.get("bbox_xyxy", det.get("bounding_box", det.get("bbox")))
    else:
        box = getattr(det, "bbox_xyxy", None)
        if box is None:
            box = getattr(det, "bounding_box", None)
        if box is None:
            box = getattr(det, "bbox", None)
    if box is None:
        return None
    try:
        x1, y1, x2, y2 = (float(v) for v in box)
    except (TypeError, ValueError):
        return None
    if x2 < x1 or y2 < y1:
        return None
    return (x1, y1, x2, y2)


def is_person_detection(det: Any) -> bool:
    """True when the detection is the person class (case-insensitive)."""
    try:
        return _class_name_of(det).strip().casefold() == PERSON_CLASS_NAME
    except Exception:
        return False


def calculate_crowd_metrics(
    detections: list,
    image_width: int,
    image_height: int,
) -> dict:
    """Count and image-space density from post-NMS person boxes.

    Only person-named detections contribute; image area of 0 is guarded to
    a minimum of 1 px. Returns a plain dict (same fields as the DMS
    ``CrowdMetricsSchema``).
    """
    boxes = []
    for det in detections or ():
        if not is_person_detection(det):
            continue
        box = _bbox_of(det)
        if box is None:
            continue
        boxes.append(box)
    image_area = max(1, int(image_width) * int(image_height))
    coverage = min(1.0, sum((x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in boxes) / image_area)
    megapixels = image_area / 1_000_000
    return {
        "person_count": len(boxes),
        "coverage_fraction": round(coverage, 4),
        "density_per_megapixel": round(len(boxes) / megapixels, 2),
    }
