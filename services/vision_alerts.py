"""Vision-alert emitter stub (Prototype-local, DMS intake compatible).

The DMS backend intake ``POST /api/v1/mining-vision/vision-alerts/`` is not
copied. Instead: when ``DMS_BACKEND_URL`` + ``VISION_SERVICE_TOKEN`` are set,
``post_json``-compatible HTTP clients can be injected; otherwise payloads
built by ``detector.fire_smoke_gate.build_vision_alert_payload`` are
appended to ``data/vision_alerts.jsonl``. Never raises; outcome mapping
mirrors the DMS vision emitter (202 sent, 400 rejected, 403 denied +
backoff, else failed).
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger(__name__)

VISION_ALERTS_PATH = "/api/v1/mining-vision/vision-alerts/"
AUTH_BACKOFF_SECONDS = 300.0


class VisionAlertOutcome(str, Enum):
    SENT = "sent"
    REJECTED = "rejected"
    DENIED = "denied"
    FAILED = "failed"
    SKIPPED = "skipped"


class LocalAlertSink:
    """File stub for vision alerts (no network)."""

    def __init__(self, path: str = "data/vision_alerts.jsonl") -> None:
        self.path = path
        self.base_url = "local"

    def post_json(self, path: str, payload: dict) -> int | None:
        try:
            directory = os.path.dirname(os.path.abspath(self.path))
            os.makedirs(directory, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload) + "\n")
        except OSError as exc:
            logger.warning("local alert sink write failed: %s", exc)
            return None
        return 202


class VisionAlertEmitter:
    def __init__(self, http: Any = None, *,
                 clock: Callable[[], float] = time.monotonic,
                 auth_backoff_seconds: float = AUTH_BACKOFF_SECONDS) -> None:
        self._http = http or LocalAlertSink()
        self._clock = clock
        self._backoff = float(auth_backoff_seconds)
        self._auth_next_attempt = 0.0
        self._lock = threading.Lock()

    @property
    def auth_backoff_active(self) -> bool:
        return self._clock() < self._auth_next_attempt

    def emit(self, payload: dict) -> VisionAlertOutcome:
        if self.auth_backoff_active:
            return VisionAlertOutcome.SKIPPED
        try:
            status = self._http.post_json(VISION_ALERTS_PATH, payload)
        except Exception:
            return VisionAlertOutcome.FAILED
        if status == 202:
            return VisionAlertOutcome.SENT
        if status == 400:
            return VisionAlertOutcome.REJECTED
        if status == 403:
            with self._lock:
                self._auth_next_attempt = self._clock() + self._backoff
            return VisionAlertOutcome.DENIED
        return VisionAlertOutcome.FAILED
