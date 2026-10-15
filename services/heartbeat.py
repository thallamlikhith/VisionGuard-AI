"""Heartbeat reporter (copied semantics from DMS, Prototype-local stub).

Source: DMS ``AI_services/app/vision/streaming/heartbeat.py``
``CameraHeartbeatReporter`` outcome mapping + backoff + sanitizer
(read-only copy). Adapted: the DMS service-token HTTP client is replaced
by an injectable ``http`` object; when no backend is configured the
``LocalHeartbeatSink`` file stub (``data/heartbeats.jsonl``) is used
instead of network POSTs. No DMS backend code is copied.

Rules kept: beats only for flowing workers (enforced by the pump caller),
401/403 -> DENIED + backoff, 404 -> UNRESOLVED, transport error -> FAILED
(never raises), payload exactly {timestamp, timeout}.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

__all__ = [
    "HEARTBEAT_PATH_TEMPLATE",
    "DEFAULT_HEARTBEAT_TIMEOUT_SECONDS",
    "HEARTBEAT_INTERVAL_SECONDS",
    "HEARTBEAT_FRESHNESS_SECONDS",
    "AUTH_BACKOFF_SECONDS",
    "HeartbeatOutcome",
    "sanitize_error_detail",
    "utc_isoformat",
    "CameraHeartbeatReporter",
    "LocalHeartbeatSink",
    "default_reporter",
]

HEARTBEAT_PATH_TEMPLATE = "/api/v1/mining-cameras/feed/{camera_pk}/heartbeat/"
DEFAULT_HEARTBEAT_TIMEOUT_SECONDS = 300
HEARTBEAT_INTERVAL_SECONDS = 30.0
HEARTBEAT_FRESHNESS_SECONDS = 60.0
AUTH_BACKOFF_SECONDS = 300.0
FLOWING_STATES = frozenset({"running"})


class HeartbeatOutcome(str, Enum):
    SENT = "sent"
    REJECTED = "rejected"
    DENIED = "denied"
    FAILED = "failed"
    UNRESOLVED = "unresolved"


def redact_credentials(text: Any) -> str:
    import re

    return re.sub(r"://[^/@\s]*:[^/@\s]*@", "://***@", str(text))


def sanitize_error_detail(text: Any) -> str:
    cleaned = redact_credentials(str(text))
    return " ".join(cleaned.split())[:300]


def utc_isoformat(wall_seconds: float) -> str:
    return datetime.fromtimestamp(float(wall_seconds), tz=timezone.utc).isoformat()


class LocalHeartbeatSink:
    """File stub: appends beats to ``data/heartbeats.jsonl`` (no network)."""

    def __init__(self, path: str = "data/heartbeats.jsonl") -> None:
        self.path = path
        self.base_url = "local"

    def post_json(self, path: str, payload: dict) -> int:
        try:
            directory = os.path.dirname(os.path.abspath(self.path))
            os.makedirs(directory, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps({"path": path, **payload}) + "\n")
        except OSError as exc:
            logger.warning("local heartbeat sink write failed: %s", exc)
            return 500
        return 200


class CameraHeartbeatReporter:
    def __init__(
        self,
        http: Any = None,
        *,
        clock: Callable[[], float] = time.monotonic,
        auth_backoff_seconds: float = AUTH_BACKOFF_SECONDS,
    ) -> None:
        self._http = http or LocalHeartbeatSink()
        self._clock = clock
        self._backoff = float(auth_backoff_seconds)
        self._auth_next_attempt = 0.0
        self._lock = threading.Lock()
        self._beats_sent = 0
        self._denied = 0
        self._failed = 0
        self._rejected = 0
        self._unresolved: set[str] = set()
        self._last_error = ""
        self._last_beat_at: Optional[float] = None

    @property
    def auth_backoff_active(self) -> bool:
        return self._clock() < self._auth_next_attempt

    def beat(self, camera_pk: Any, timestamp_iso: str, *,
             timeout: int = DEFAULT_HEARTBEAT_TIMEOUT_SECONDS) -> HeartbeatOutcome:
        pk = str(camera_pk)
        with self._lock:
            if pk in self._unresolved:
                return HeartbeatOutcome.UNRESOLVED
        if self.auth_backoff_active:
            return HeartbeatOutcome.DENIED
        path = HEARTBEAT_PATH_TEMPLATE.format(camera_pk=pk)
        payload = {"timestamp": str(timestamp_iso), "timeout": int(timeout)}
        try:
            status = self._http.post_json(path, payload)
        except Exception as exc:  # never raise
            with self._lock:
                self._failed += 1
                self._last_error = sanitize_error_detail(exc)
            return HeartbeatOutcome.FAILED
        return self._interpret(pk, status)

    def _interpret(self, pk: str, status: Optional[int]) -> HeartbeatOutcome:
        if status == 200:
            with self._lock:
                self._beats_sent += 1
                self._last_beat_at = self._clock()
                self._last_error = ""
            return HeartbeatOutcome.SENT
        if status in (401, 403):
            with self._lock:
                self._auth_next_attempt = self._clock() + self._backoff
                self._denied += 1
                self._last_error = f"heartbeat denied ({status})"
            return HeartbeatOutcome.DENIED
        if status == 404:
            with self._lock:
                self._unresolved.add(pk)
                self._last_error = f"camera pk {pk}: no such camera (404)"
            return HeartbeatOutcome.UNRESOLVED
        if status is None:
            with self._lock:
                self._failed += 1
                self._last_error = "heartbeat transport failed"
            return HeartbeatOutcome.FAILED
        with self._lock:
            self._rejected += 1
            self._last_error = f"heartbeat rejected (status {status})"
        return HeartbeatOutcome.REJECTED

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "beats_sent": self._beats_sent,
                "denied": self._denied,
                "failed": self._failed,
                "rejected": self._rejected,
                "unresolved": sorted(self._unresolved),
                "backoff_active": self._clock() < self._auth_next_attempt,
                "last_error": self._last_error,
                "last_beat_at": self._last_beat_at,
            }


def default_reporter(http: Any = None) -> CameraHeartbeatReporter:
    """Local sink unless a backend URL + token are configured (then caller
    injects its own HTTP client; no backend code lives here)."""
    if http is not None:
        return CameraHeartbeatReporter(http=http)
    try:
        from services.vision_settings import backend_configured

        if backend_configured():
            logger.info("backend heartbeat configured; inject an HTTP client")
    except Exception:
        pass
    return CameraHeartbeatReporter(http=LocalHeartbeatSink())
