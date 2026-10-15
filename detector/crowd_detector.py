"""YOLO crowd (person) detector wrapper (Prototype-local, DMS-compatible).

Uses the copied ``detector/models/crowd_v1.pt`` weights at the DMS
operating point (conf 0.35, imgsz 640). The weights carry the full COCO
table with id 0 = ``person``; only person-named boxes are returned
(matched case-insensitively, like DMS ``person_detector.py`` — no fixed
class_id is assumed). Lazy: ultralytics imports only when constructed,
so the person/zone loop imports nothing new while this path is off.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

LOG = logging.getLogger(__name__)

DEFAULT_CONF = 0.35
DEFAULT_IMGSZ = 640
WEIGHTS_FILENAME = "crowd_v1.pt"


def weights_path() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "models", WEIGHTS_FILENAME)


class CrowdError(Exception):
    """Raised when the crowd model cannot load or has no person class."""


def check_person_class(names: dict) -> None:
    """Verify the weights table contains a person class (any id, any case)."""
    try:
        values = [str(v).strip().casefold() for v in dict(names).values()]
    except Exception as exc:
        raise CrowdError(f"unreadable crowd class table: {exc}") from exc
    if "person" not in values:
        raise CrowdError(f"crowd weights have no person class, got {dict(names)!r}")


class CrowdDetector:
    """Thin YOLO wrapper returning person-only detection dicts."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence: float = DEFAULT_CONF,
        imgsz: int = DEFAULT_IMGSZ,
    ) -> None:
        from detector.model_cache import load_yolo

        path = model_path or weights_path()
        if not os.path.isfile(path):
            raise CrowdError(
                f"crowd weights not found: {path} "
                "(copy crowd_v1.pt into detector/models/)"
            )
        self.confidence = float(confidence)
        self.imgsz = int(imgsz)
        try:
            self._model, cached = load_yolo(path, self.imgsz)
        except Exception as exc:
            raise CrowdError(f"could not load crowd model: {exc}") from exc
        check_person_class(getattr(self._model, "names", {}))
        LOG.info("crowd detector ready%s: model=%s conf=%.2f imgsz=%d",
                 " (cached)" if cached else "", path, self.confidence, self.imgsz)

    def predict(
        self, image: Any, confidence_threshold: Optional[float] = None
    ) -> list[dict]:
        """Return person-only ``[{class_name, confidence, bbox_xyxy}]`` dicts."""
        conf = DEFAULT_CONF if confidence_threshold is None else float(confidence_threshold)
        results = self._model.predict(image, conf=conf, imgsz=self.imgsz, verbose=False)
        own = {int(k): str(v) for k, v in self._model.names.items()}
        out: list[dict] = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None or len(boxes) == 0:
                continue
            for box in boxes:
                try:
                    class_id = int(box.cls[0])
                    class_name = own.get(class_id, str(class_id))
                    if class_name.strip().casefold() != "person":
                        continue
                    confidence = float(box.conf[0])
                    x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
                except Exception:
                    continue
                out.append({
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": confidence,
                    "bbox_xyxy": [x1, y1, x2, y2],
                    "track_id": None,
                })
        return out


_detector: Optional[CrowdDetector] = None


def get_crowd_detector() -> CrowdDetector:
    """Process-wide singleton (lazy so imports stay light)."""
    global _detector
    if _detector is None:
        _detector = CrowdDetector()
    return _detector
