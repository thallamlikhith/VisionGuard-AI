"""Loading and validation of config.yaml.

Kept deliberately small: read the YAML file, check the few values we depend
on, return a plain dict. No pydantic, no schema framework.
"""

from __future__ import annotations

import os
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = "config.yaml"

VALID_MODES = ("INSTANT", "TIMER", "LIMIT")
VALID_ANCHORS = ("bottom_center", "center")


class ConfigError(Exception):
    """Raised when config.yaml is missing, unreadable or invalid."""


def _require(data: dict, key: str) -> Any:
    if key not in data or data[key] is None:
        raise ConfigError(f"config.yaml: missing required key '{key}'")
    return data[key]


def _validate_points(raw: Any) -> list[tuple[int, int]]:
    # An empty zone is valid, not an error: there is no default polygon any
    # more, so "not set up yet" has to be a state the app can start in.
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ConfigError("config.yaml: 'danger_zone.points' must be a list of [x, y] points")
    if not raw:
        return []
    if len(raw) < 3:
        raise ConfigError("config.yaml: 'danger_zone.points' needs at least 3 [x, y] points")

    points: list[tuple[int, int]] = []
    for item in raw:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ConfigError(f"config.yaml: invalid point {item!r}, expected [x, y]")
        try:
            points.append((int(item[0]), int(item[1])))
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"config.yaml: point {item!r} is not numeric") from exc
    return points


def load_config(path: str = DEFAULT_CONFIG_PATH) -> dict:
    """Read config.yaml and return a validated settings dict."""
    if not os.path.isfile(path):
        raise ConfigError(f"config file not found: {path}")

    try:
        with open(path, "r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise ConfigError(f"could not parse {path}: {exc}") from exc
    except OSError as exc:
        raise ConfigError(f"could not read {path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: expected a YAML mapping at the top level")

    machine = _require(raw, "machine")
    danger_zone = _require(raw, "danger_zone")
    rules = raw.get("rules") or {}
    detector = raw.get("detector") or {}
    runtime = raw.get("runtime") or {}
    hazard = raw.get("hazard") or {}
    fire_smoke = raw.get("fire_smoke") or {}
    crowd = raw.get("crowd") or {}
    ppe = raw.get("ppe") or {}
    limits = raw.get("limits") or {}
    ppe_grading = ppe.get("grading") or {}
    # per-type fire/smoke switches. Absent keys stay None (rather than
    # resolving here) so stored settings downstream can rank: stored
    # per-type > file per-type > stored-or-file shared switch > off.
    _fire_cfg = fire_smoke.get("fire_enabled", None)
    _smoke_cfg = fire_smoke.get("smoke_enabled", None)
    fire_enabled = None if _fire_cfg is None else bool(_fire_cfg)
    smoke_enabled = None if _smoke_cfg is None else bool(_smoke_cfg)

    mode = str(rules.get("mode", "TIMER")).upper()
    if mode not in VALID_MODES:
        raise ConfigError(f"config.yaml: rules.mode must be one of {VALID_MODES}, got {mode!r}")

    dwell_seconds = float(rules.get("dwell_seconds", 1))
    if dwell_seconds <= 0:
        raise ConfigError("config.yaml: rules.dwell_seconds must be greater than 0")

    anchor = str(danger_zone.get("anchor", "center")).lower()
    if anchor not in VALID_ANCHORS:
        raise ConfigError(f"config.yaml: danger_zone.anchor must be one of {VALID_ANCHORS}")

    zone_margin_px = float(danger_zone.get("zone_margin_px", 5))
    if zone_margin_px < 0:
        raise ConfigError("config.yaml: danger_zone.zone_margin_px cannot be negative")

    confidence = float(detector.get("confidence", 0.25))
    if not 0 < confidence <= 1:
        raise ConfigError("config.yaml: detector.confidence must be between 0 and 1")

    imgsz = int(detector.get("imgsz", 320))
    if imgsz <= 0:
        raise ConfigError("config.yaml: detector.imgsz must be greater than 0")

    detect_every_n = int(detector.get("detect_every_n", 1))
    if detect_every_n < 1:
        raise ConfigError("config.yaml: detector.detect_every_n must be 1 or more")

    # 0 (or 1) = off: YOLO runs on the fixed detect_every_n schedule. 2 or
    # more = an empty scene is only looked at every Nth frame, while people
    # around still means every detect_every_n-th frame.
    adaptive_idle_n = int(detector.get("adaptive_idle_n", 0))
    if adaptive_idle_n < 0:
        raise ConfigError("config.yaml: detector.adaptive_idle_n cannot be negative")

    max_detect_long_side = int(detector.get("max_detect_long_side", 960))
    if max_detect_long_side < 0:
        raise ConfigError("config.yaml: detector.max_detect_long_side must be >= 0")

    # person-track confirmation gate (see config.yaml): thresholds only, the
    # gate itself lives in detector.tracker. Every key falls back to the
    # measured default, so an older config.yaml without them behaves exactly
    # like the tuned one.
    person_confirm_conf = float(detector.get("person_confirm_conf", 0.4))
    if not 0 < person_confirm_conf <= 1:
        raise ConfigError("config.yaml: detector.person_confirm_conf must be between 0 and 1")

    person_confirm_frames = int(detector.get("person_confirm_frames", 3))
    if person_confirm_frames < 1:
        raise ConfigError("config.yaml: detector.person_confirm_frames must be 1 or more")

    person_instant_conf = float(detector.get("person_instant_conf", 0.6))
    if not 0 < person_instant_conf <= 1:
        raise ConfigError("config.yaml: detector.person_instant_conf must be between 0 and 1")

    person_confirm_move_ratio = float(detector.get("person_confirm_move_ratio", 0.05))
    if person_confirm_move_ratio < 0:
        raise ConfigError("config.yaml: detector.person_confirm_move_ratio cannot be negative")

    crowd_source = str(crowd.get("source", "reuse")).lower()
    if crowd_source not in ("reuse", "model"):
        raise ConfigError("config.yaml: crowd.source must be 'reuse' or 'model'")

    return {
        "config_path": os.path.abspath(path),
        "machine": {
            "id": str(_require(machine, "id")),
            "name": str(machine.get("name", machine["id"])),
        },
        "danger_zone": {
            # no stored zone and nothing in this file simply means no zone
            "points": _validate_points(danger_zone.get("points")),
            "anchor": anchor,
            "zone_margin_px": zone_margin_px,
        },
        "rules": {
            "mode": mode,
            "dwell_seconds": dwell_seconds,
        },
        "detector": {
            "model": str(detector.get("model", "yolo11n.pt")),
            "confidence": confidence,
            "person_class_id": int(detector.get("person_class_id", 0)),
            "imgsz": imgsz,
            "detect_every_n": detect_every_n,
            "adaptive_idle_n": adaptive_idle_n,
            "max_detect_long_side": max_detect_long_side,
            "openvino": bool(detector.get("openvino", False)),
            "person_confirm_conf": person_confirm_conf,
            "person_confirm_frames": person_confirm_frames,
            "person_instant_conf": person_instant_conf,
            "person_confirm_move_ratio": person_confirm_move_ratio,
        },
        "runtime": {
            "repeat_alert_seconds": float(runtime.get("repeat_alert_seconds", 10)),
            "show_zone_fill": bool(runtime.get("show_zone_fill", True)),
        },
        "hazard": {
            "enabled": bool(hazard.get("enabled", True)),
        },
        "fire_smoke": {
            "enabled": bool(fire_smoke.get("enabled", False)),
            "fire_enabled": fire_enabled,
            "smoke_enabled": smoke_enabled,
            "model": str(fire_smoke.get("model", "detector/models/fire_smoke_v2.pt")),
            "confidence": float(fire_smoke.get("confidence", 0.35)),
            "imgsz": int(fire_smoke.get("imgsz", 640)),
            "fire_seconds": float(fire_smoke.get("fire_seconds", 1.5)),
            "smoke_seconds": float(fire_smoke.get("smoke_seconds", 2.5)),
            "high_confidence": float(fire_smoke.get("high_confidence", 0.8)),
            "person_containment": float(fire_smoke.get("person_containment", 0.5)),
            "analysis_fps": float(fire_smoke.get("analysis_fps", 2.0)),
            "camera_file": str(fire_smoke.get("camera_file", "")),
            "backend_url": str(fire_smoke.get("backend_url", "")),
            "service_token": str(fire_smoke.get("service_token", "")),
        },
        "crowd": {
            "enabled": bool(crowd.get("enabled", False)),
            "source": crowd_source,
            "model": str(crowd.get("model", "detector/models/crowd_v1.pt")),
            "confidence": float(crowd.get("confidence", 0.35)),
            "imgsz": int(crowd.get("imgsz", 640)),
            "analysis_fps": float(crowd.get("analysis_fps", 2.0)),
        },
        "ppe": {
            "enabled": bool(ppe.get("enabled", False)),
            "model": str(ppe.get("model", "detector/models/ppe_v1.pt")),
            "confidence": float(ppe.get("confidence", 0.15)),
            "iou": float(ppe.get("iou", 0.45)),
            "imgsz": int(ppe.get("imgsz", 960)),
            "tta": bool(ppe.get("tta", True)),
            "memory_seconds": float(ppe.get("memory_seconds", 4.0)),
            "grading": {
                "medium_ratio": float(ppe_grading.get("medium_ratio", 0.25)),
                "high_ratio": float(ppe_grading.get("high_ratio", 0.5)),
                "critical_ratio": float(ppe_grading.get("critical_ratio", 1.0)),
            },
            "analysis_fps": float(ppe.get("analysis_fps", 1.0)),
        },
        "limits": {
            # sparse on purpose: only explicitly set keys come back, so the
            # server fallbacks (and their monkeypatched test doubles) apply
            # to anything missing. A filled block here would shadow them.
            key: int(value)
            for key, value in (
                ("max_upload_size_mb", limits.get("max_upload_size_mb")),
                ("max_videos", limits.get("max_videos")),
                ("history_days", limits.get("history_days")),
            )
            if value is not None
        },
    }
