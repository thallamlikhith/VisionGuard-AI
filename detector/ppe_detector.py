"""YOLO PPE detector wrapper (Prototype-local, DMS-compatible).

Lean port of DMS ``app/vision/ppe_detection/services/vision/detectors/
{yolo, ppe}.py`` minus the service layer: no VisionService, no registry,
no manager, no HTTP router. Uses the copied ``detector/models/ppe_v1.pt``
weights (YOLOv8m, 14 classes) at the DMS operating point (conf 0.25,
IoU 0.45). Lazy: ultralytics/torch import only when constructed.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

LOG = logging.getLogger(__name__)

DEFAULT_CONF = 0.25
DEFAULT_IOU = 0.45
#: Inference size. The model default (640); the app runs the PPE pass at 960
#: (see config.yaml), which catches small/distant hardhats the 640 pass
#: misses, at ~2x the inference cost.
DEFAULT_IMGSZ = 640
WEIGHTS_FILENAME = "ppe_v1.pt"

#: The 14-class vocabulary as trained (verified against best.pt names table).
EXPECTED_CLASSES = {
    0: "Fall-Detected",
    1: "Gloves",
    2: "Goggles",
    3: "Hardhat",
    4: "Ladder",
    5: "Mask",
    6: "NO-Gloves",
    7: "NO-Goggles",
    8: "NO-Hardhat",
    9: "NO-Mask",
    10: "NO-Safety Vest",
    11: "Person",
    12: "Safety Cone",
    13: "Safety Vest",
}
MODEL_VERSION = "ppe_v1.0"


def weights_path() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "models", WEIGHTS_FILENAME)


class PPEError(Exception):
    """Raised when the PPE model cannot load or has a bad class table."""


def check_class_table(names: dict) -> None:
    """Verify the 14-class PPE vocabulary (exact ids and names)."""
    try:
        got = {int(k): str(v) for k, v in dict(names).items()}
    except Exception as exc:
        raise PPEError(f"unreadable PPE class table: {exc}") from exc
    if got != EXPECTED_CLASSES:
        raise PPEError(
            "PPE class table must be the 14-class vocabulary "
            f"{EXPECTED_CLASSES!r}, got {got!r}"
        )


class PPEDetector:
    """Thin YOLO wrapper returning DMS-shaped detection dicts."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence: float = DEFAULT_CONF,
        iou_threshold: float = DEFAULT_IOU,
        device: str = "",
        imgsz: int = DEFAULT_IMGSZ,
        tta: bool = False,
    ) -> None:
        from detector.model_cache import load_yolo

        path = model_path or weights_path()
        if not os.path.isfile(path):
            raise PPEError(
                f"PPE weights not found: {path} "
                "(copy best.pt into detector/models/ as ppe_v1.pt)"
            )
        self.confidence = float(confidence)
        self.iou_threshold = float(iou_threshold)
        self.device = str(device or "")
        self.imgsz = int(imgsz)
        # test-time augmentation (flips + multi-scale): lifts borderline
        # detections (a 0.17 vest reads 0.2+) at ~3-4x inference cost. Off
        # by default so golden baselines and light machines are unaffected;
        # the app enables it (see config.yaml).
        self.tta = bool(tta)
        try:
            # shared with other starts on the same (weights, size): the
            # compiled predictor is reused, so the size is part of the key.
            self._model, cached = load_yolo(path, self.imgsz)
        except Exception as exc:
            raise PPEError(f"could not load PPE model: {exc}") from exc
        check_class_table(getattr(self._model, "names", {}))
        LOG.info("PPE detector ready%s: model=%s conf=%.2f iou=%.2f imgsz=%d tta=%s",
                 " (cached)" if cached else "",
                 path, self.confidence, self.iou_threshold, self.imgsz,
                 self.tta)

    def predict(
        self, image: Any, confidence_threshold: Optional[float] = None
    ) -> list[dict]:
        """Return ``[{class_id, class_name, confidence, bbox_xyxy}]`` dicts."""
        conf = DEFAULT_CONF if confidence_threshold is None else float(confidence_threshold)
        kwargs: dict = dict(conf=conf, iou=self.iou_threshold, imgsz=self.imgsz, verbose=False)
        if self.tta:
            kwargs["augment"] = True
        if self.device:
            kwargs["device"] = self.device
        results = self._model.predict(image, **kwargs)
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

    def version(self) -> str:
        return MODEL_VERSION


_detector: Optional[PPEDetector] = None


def get_ppe_detector() -> PPEDetector:
    """Process-wide singleton (lazy so imports stay light)."""
    global _detector
    if _detector is None:
        _detector = PPEDetector()
    return _detector
