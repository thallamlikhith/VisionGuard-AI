"""YOLO fire/smoke detector wrapper (Prototype-local, DMS-compatible).

Uses the copied ``detector/models/fire_smoke_v2.pt`` weights at the DMS
operating point (conf 0.35, imgsz 640). Verifies the weights' own class
table is id 0 = smoke, id 1 = fire on load and refuses to run when it is
not, so a swapped export can never invert severity downstream.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

LOG = logging.getLogger(__name__)

EXPECTED_NAMES = {0: "smoke", 1: "fire"}
DEFAULT_CONF = 0.35
DEFAULT_IMGSZ = 640
WEIGHTS_FILENAME = "fire_smoke_v2.pt"


def weights_path() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "models", WEIGHTS_FILENAME)


class FireSmokeError(Exception):
    """Raised when the fire/smoke model cannot load or has a bad class table."""


def check_class_table(names: dict) -> None:
    """Verify id 0 = smoke, id 1 = fire (case-insensitive)."""
    try:
        got = {int(k): str(v).strip().casefold() for k, v in dict(names).items()}
    except Exception as exc:
        raise FireSmokeError(f"unreadable fire/smoke class table: {exc}") from exc
    if got.get(0) != "smoke" or got.get(1) != "fire":
        raise FireSmokeError(
            f"fire/smoke class order must be 0=smoke, 1=fire, got {got!r}"
        )


class FireSmokeDetector:
    """Thin YOLO wrapper returning DMS-shaped detection dicts."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence: float = DEFAULT_CONF,
        imgsz: int = DEFAULT_IMGSZ,
    ) -> None:
        from detector.model_cache import load_yolo

        path = model_path or weights_path()
        if not os.path.isfile(path):
            raise FireSmokeError(
                f"fire/smoke weights not found: {path} "
                "(copy fire_smoke_v2.pt into detector/models/)"
            )
        self.confidence = float(confidence)
        self.imgsz = int(imgsz)
        try:
            self._model, cached = load_yolo(path, self.imgsz)
        except Exception as exc:
            raise FireSmokeError(f"could not load fire/smoke model: {exc}") from exc
        check_class_table(getattr(self._model, "names", {}))
        LOG.info("fire/smoke detector ready%s: model=%s conf=%.2f imgsz=%d",
                 " (cached)" if cached else "", path, self.confidence, self.imgsz)

    def predict(
        self, image: Any, confidence_threshold: Optional[float] = None
    ) -> list[dict]:
        """Return ``[{class_name, confidence, bbox_xyxy, track_id}]`` dicts."""
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


_detector: Optional[FireSmokeDetector] = None


def get_fire_smoke_detector() -> FireSmokeDetector:
    """Process-wide singleton (lazy so imports stay light)."""
    global _detector
    if _detector is None:
        _detector = FireSmokeDetector()
    return _detector
