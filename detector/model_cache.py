"""Process-wide cache of loaded YOLO weights for the side detectors.

Fire/smoke, crowd and PPE each wrap one read-only YOLO model. Loading the
weights (and compiling the graph on the first predict) costs several seconds,
and the pipeline built a fresh detector on every POST /start - so every Start
press and every video switch paid the load again, then stalled the first
analysis while the graph compiled. The weights are read-only during inference,
the per-call ``conf`` and ``imgsz`` are passed to ``predict`` each time, and
only one pipeline run is alive at a time, so sharing a single model object
across sequential runs is safe.

Keyed by ``(YOLO class, resolved path, imgsz)``. The YOLO *class* is part of
the key so monkeypatched fakes in tests never get a stale cached object.
"""

from __future__ import annotations

import os
import threading
from typing import Any

_MODEL_CACHE: dict[tuple, Any] = {}
_LOCK = threading.Lock()


def load_yolo(path: str, imgsz: int) -> tuple[Any, bool]:
    """Return ``(model, cached)``, reusing a process-wide object when possible.

    The cached object keeps its compiled predictor: unlike the person tracker,
    these detectors have no per-run session state to reset, so the compile cost
    is paid once and reused. Raises whatever ``YOLO`` raises on a bad path.
    """
    from ultralytics import YOLO

    key = (YOLO, os.path.abspath(str(path)), int(imgsz))
    with _LOCK:
        hit = _MODEL_CACHE.get(key)
    if hit is not None:
        return hit, True
    model = YOLO(path)
    with _LOCK:
        # a concurrent first load wins; both are equivalent read-only weights
        model = _MODEL_CACHE.setdefault(key, model)
    return model, False


def clear_cache() -> None:
    """Drop every cached model (tests use this between monkeypatched fakes)."""
    with _LOCK:
        _MODEL_CACHE.clear()
