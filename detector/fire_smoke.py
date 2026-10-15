"""Fire-and-smoke model capability definitions (copied from DMS AI_services).

Source: DMS ``AI_services/app/vision/fire_smoke.py`` (read-only copy).
Adapted: DMS ``app.vision.schemas`` import removed so this module has no
DMS dependency; ``is_fire_smoke_detection`` accepts any object/dict with
a ``class_name`` instead of the DMS ``RawDetection`` type.

Preserved exactly:
- weights file ``fire_smoke_v2.pt`` (v1 NOT copied)
- class order 0 = smoke, 1 = fire (``FIRE_SMOKE_CLASSES`` mirrors the
  weights' own ``model.names`` order; swapping it inverts severity)
"""

from __future__ import annotations

from typing import Any

FIRE_SMOKE_MODEL_NAME = "fire-smoke"
FIRE_SMOKE_MODEL_VERSION = "2.0.0"
FIRE_SMOKE_WEIGHTS_FILE = "fire_smoke_v2.pt"
#: Model-index order (NOT alphabetical): v2 emits id 0 = smoke, id 1 = fire.
FIRE_SMOKE_CLASSES = ("smoke", "fire")
FIRE_SMOKE_DISASTER_TYPES = ("fire", "smoke")


def _class_name_of(detection: Any) -> str:
    if isinstance(detection, dict):
        return str(detection.get("class_name", ""))
    return str(getattr(detection, "class_name", ""))


def is_fire_smoke_detection(detection: Any) -> bool:
    """Return True only for canonical classes emitted by the fire/smoke model."""
    try:
        return _class_name_of(detection).casefold() in FIRE_SMOKE_CLASSES
    except Exception:
        return False
