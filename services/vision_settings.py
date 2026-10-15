"""Prototype-local vision settings (replaces DMS shared config import).

Defaults are pinned to the DMS plan values so the copied gate behaves
identically here:
- fire persistence 1.5 s, smoke persistence 2.5 s
- analysis fps 1.0, registry poll 30 s, heartbeat 30 s interval
- fire/smoke detector conf 0.35, imgsz 640, model fire_smoke_v2.pt

Overrides (all optional): ``config.yaml`` ``fire_smoke:`` block and the
``VISION_*`` / ``DMS_BACKEND_URL`` / ``VISION_SERVICE_TOKEN`` env vars.
No DMS import; no network; no heavy imports.
"""

from __future__ import annotations

import os
from typing import Any

DEFAULTS: dict[str, Any] = {
    "enabled": False,
    "model": "detector/models/fire_smoke_v2.pt",
    "confidence": 0.35,
    "imgsz": 640,
    "fire_seconds": 1.5,
    "smoke_seconds": 2.5,
    "analysis_fps": 2.0,
    "registry_poll_seconds": 30.0,
    "heartbeat_interval_seconds": 30.0,
    "backend_url": "",
    "service_token": "",
    "camera_file": "",
}


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def get_vision_settings(config_path: str = "config.yaml") -> dict[str, Any]:
    """Merge defaults + ``config.yaml`` fire_smoke block + env overrides."""
    current = dict(DEFAULTS)
    try:
        import yaml  # local dep already used by config.py

        if os.path.isfile(config_path):
            with open(config_path, "r", encoding="utf-8") as handle:
                raw = yaml.safe_load(handle) or {}
            block = (raw.get("fire_smoke") or {}) if isinstance(raw, dict) else {}
            for key in ("enabled", "model", "confidence", "imgsz",
                        "fire_seconds", "smoke_seconds", "analysis_fps",
                        "registry_poll_seconds", "heartbeat_interval_seconds",
                        "backend_url", "camera_file"):
                if key in block and block[key] is not None:
                    current[key] = block[key]
    except Exception:
        pass
    current["fire_seconds"] = _env_float("VISION_FIRE_PERSISTENCE_SECONDS",
                                         float(current["fire_seconds"]))
    current["smoke_seconds"] = _env_float("VISION_SMOKE_PERSISTENCE_SECONDS",
                                          float(current["smoke_seconds"]))
    current["analysis_fps"] = _env_float("VISION_ANALYSIS_FPS",
                                         float(current["analysis_fps"]))
    current["registry_poll_seconds"] = _env_float(
        "VISION_REGISTRY_POLL_SECONDS", float(current["registry_poll_seconds"]))
    if os.environ.get("DMS_BACKEND_URL"):
        current["backend_url"] = os.environ["DMS_BACKEND_URL"]
    if os.environ.get("VISION_SERVICE_TOKEN"):
        current["service_token"] = os.environ["VISION_SERVICE_TOKEN"]
    if os.environ.get("VISION_CAMERA_FILE"):
        current["camera_file"] = os.environ["VISION_CAMERA_FILE"]
    return current


def backend_configured(settings: dict[str, Any] | None = None) -> bool:
    """True only when a backend URL + service token are both present."""
    current = settings or get_vision_settings()
    return bool(current.get("backend_url")) and bool(current.get("service_token"))
