"""Camera registry source (copied logic from DMS, Prototype-local).

Source: DMS ``AI_services/app/vision/streaming/registry_poller.py``
``CameraRecord`` / ``parse_camera_entry`` / ``FileRegistrySource``
(read-only copy). Adapted: DMS settings/error-tracker/capture imports
replaced by stdlib + ``services.vision_settings``; the backend HTTP source
is replaced by a config switch (local file when ``VISION_CAMERA_FILE`` /
``fire_smoke.camera_file`` is set, otherwise an explicit
``RegistryUnavailable`` telling the caller backend mode needs a URL —
no DMS backend code is copied).

Credential hygiene kept: stream refs never enter logs/status; the
fingerprint stores a truncated SHA-256 digest only.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

__all__ = [
    "REGISTRY_PATH",
    "DEFAULT_HTTP_TIMEOUT_SECONDS",
    "AUTH_BACKOFF_SECONDS",
    "RegistryError",
    "RegistryAuthError",
    "RegistryUnavailable",
    "CameraRecord",
    "RegistrySource",
    "BackendRegistrySource",
    "FileRegistrySource",
    "RegistryPoller",
    "parse_camera_entry",
    "redact_credentials",
]

REGISTRY_PATH = "/api/v1/mining-cameras/camera-service-config/"
DEFAULT_HTTP_TIMEOUT_SECONDS = 3.0
AUTH_BACKOFF_SECONDS = 300.0


class RegistryError(RuntimeError):
    pass


class RegistryAuthError(RegistryError):
    pass


class RegistryUnavailable(RegistryError):
    pass


def redact_credentials(text: Any) -> str:
    """Redact ``user:pass@`` credential blocks from URLs for logs/status."""
    import re

    cleaned = str(text)
    return re.sub(r"://[^/@\s]*:[^/@\s]*@", "://***@", cleaned)


@dataclass(frozen=True)
class CameraRecord:
    camera_id: str
    code: str
    name: str = ""
    mine_id: Optional[Any] = None
    rtsp_url: str = ""
    video_file: str = ""
    detectors: tuple = ("PERSON",)
    analysis_fps: float = 1.0
    confidence_threshold: Optional[float] = None
    frame_width: Optional[int] = None
    frame_height: Optional[int] = None

    def stream_ref(self) -> str:
        return self.video_file or self.rtsp_url

    def fingerprint(self) -> tuple:
        digest = hashlib.sha256(self.stream_ref().encode("utf-8")).hexdigest()[:16]
        return (
            digest,
            float(self.analysis_fps),
            tuple(self.detectors),
            self.frame_width,
            self.frame_height,
        )


def _entry_error(index: Any, reason: str) -> None:
    logger.warning("skipping camera entry %r: %s", index, reason)


def parse_camera_entry(entry: Any, index: Any = "?") -> Optional[CameraRecord]:
    if not isinstance(entry, dict):
        _entry_error(index, f"not an object ({type(entry).__name__})")
        return None
    raw_id = entry.get("id")
    if raw_id is None or isinstance(raw_id, bool):
        _entry_error(index, "missing id")
        return None
    camera_id = str(raw_id)
    code = entry.get("code")
    if not isinstance(code, str) or not code.strip():
        _entry_error(index, f"camera {camera_id}: missing code")
        return None
    detectors = entry.get("detectors")
    if isinstance(detectors, str):
        detectors = [detectors]
    if not isinstance(detectors, (list, tuple)) or not detectors:
        _entry_error(index, f"camera {camera_id}: detectors must be a non-empty list")
        return None
    clean_detectors = []
    for detector in detectors:
        if not isinstance(detector, str) or not detector.strip():
            _entry_error(index, f"camera {camera_id}: bad detector value {detector!r}")
            return None
        clean_detectors.append(detector.strip().upper())
    try:
        fps = float(entry.get("analysis_fps", 1.0))
    except (TypeError, ValueError):
        _entry_error(index, f"camera {camera_id}: bad analysis_fps")
        return None
    if fps <= 0:
        _entry_error(index, f"camera {camera_id}: analysis_fps must be > 0")
        return None
    rtsp_url = entry.get("rtsp_url") or ""
    if not isinstance(rtsp_url, str):
        _entry_error(index, f"camera {camera_id}: bad rtsp_url")
        return None
    video_path = entry.get("video_path") or entry.get("video_file") or ""
    if not isinstance(video_path, str):
        _entry_error(index, f"camera {camera_id}: bad video_path")
        return None
    if not rtsp_url.strip() and not video_path.strip():
        _entry_error(index, f"camera {camera_id}: no rtsp_url (and no dev video_path)")
        return None
    confidence = entry.get("confidence_threshold")
    if confidence is not None:
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            _entry_error(index, f"camera {camera_id}: bad confidence_threshold")
            return None
        if not 0.0 <= confidence <= 1.0:
            _entry_error(index, f"camera {camera_id}: confidence out of range")
            return None
    frame_size = {}
    for key in ("frame_width", "frame_height"):
        value = entry.get(key)
        if value is None:
            frame_size[key] = None
        else:
            try:
                ivalue = int(value)
            except (TypeError, ValueError):
                _entry_error(index, f"camera {camera_id}: bad {key}")
                return None
            if ivalue <= 0:
                _entry_error(index, f"camera {camera_id}: {key} must be positive")
                return None
            frame_size[key] = ivalue
    mine_id = entry.get("mine_id")
    if isinstance(mine_id, bool):
        _entry_error(index, f"camera {camera_id}: bad mine_id")
        return None
    name = entry.get("name") or ""
    return CameraRecord(
        camera_id=camera_id,
        code=code.strip(),
        name=name if isinstance(name, str) else "",
        mine_id=mine_id,
        rtsp_url=rtsp_url.strip(),
        video_file=video_path.strip(),
        detectors=tuple(clean_detectors),
        analysis_fps=fps,
        confidence_threshold=confidence,
        frame_width=frame_size["frame_width"],
        frame_height=frame_size["frame_height"],
    )


@runtime_checkable
class RegistrySource(Protocol):
    def fetch(self) -> list[CameraRecord]: ...
    def describe(self) -> str: ...


class BackendRegistrySource:
    """Config switch stub: backend mode needs DMS_BACKEND_URL + token.

    No DMS backend code is copied. ``fetch`` raises ``RegistryUnavailable``
    with setup instructions unless both are configured, in which case it
    performs a plain stdlib GET with ``X-Service-Token`` (same path/header
    as DMS, no other backend logic).
    """

    def __init__(self, base_url: str = "", token: str = "",
                 *, timeout: float = DEFAULT_HTTP_TIMEOUT_SECONDS) -> None:
        from services.vision_settings import get_vision_settings

        current = get_vision_settings()
        self._base_url = (base_url or str(current.get("backend_url", ""))).rstrip("/")
        self._token = token or str(current.get("service_token", ""))
        self._timeout = float(timeout)

    def describe(self) -> str:
        return f"backend {self._base_url}{REGISTRY_PATH}" if self._base_url else "backend (unconfigured)"

    def fetch(self) -> list[CameraRecord]:
        import urllib.error
        import urllib.request

        if not self._base_url or not self._token:
            raise RegistryUnavailable(
                "backend registry unconfigured: set DMS_BACKEND_URL and "
                "VISION_SERVICE_TOKEN env vars, or use FileRegistrySource with "
                "VISION_CAMERA_FILE pointing at a local cameras JSON file."
            )
        url = f"{self._base_url}{REGISTRY_PATH}"
        request = urllib.request.Request(url, method="GET", headers={
            "Accept": "application/json", "X-Service-Token": self._token})
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                status = int(response.status)
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if int(exc.code) == 403:
                raise RegistryAuthError("service token rejected (403)") from None
            raise RegistryUnavailable(f"registry status {exc.code}") from None
        except Exception as exc:
            raise RegistryUnavailable(f"backend unreachable: {exc}") from None
        if status == 403:
            raise RegistryAuthError("service token rejected (403)")
        if status != 200 or not isinstance(body, dict) or not isinstance(body.get("data"), list):
            raise RegistryUnavailable("malformed registry body (no 'data' list)")
        return [r for r in (parse_camera_entry(e, i)
                for i, e in enumerate(body["data"])) if r is not None]


class FileRegistrySource:
    """Local ``{"cameras": [...]}`` file (also ``{"data": [...]}`` or bare list)."""

    def __init__(self, path: Any) -> None:
        self._path = str(path)

    def describe(self) -> str:
        return f"file {self._path}"

    def fetch(self) -> list[CameraRecord]:
        try:
            raw = json.loads(Path(self._path).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise RegistryUnavailable(f"cannot read camera file {self._path}: {exc}") from None
        entries = raw["cameras"] if isinstance(raw, dict) and "cameras" in raw else (
            raw["data"] if isinstance(raw, dict) and "data" in raw else raw)
        if not isinstance(entries, list):
            raise RegistryUnavailable(f"camera file {self._path} has no camera list")
        records = []
        for index, entry in enumerate(entries):
            record = parse_camera_entry(entry, index)
            if record is not None:
                records.append(record)
        return records


def default_registry_source(camera_file: str = "") -> RegistrySource:
    """File source when a camera file is configured, else backend stub."""
    from services.vision_settings import get_vision_settings

    path = camera_file or str(get_vision_settings().get("camera_file", ""))
    if path:
        return FileRegistrySource(path)
    return BackendRegistrySource()


class RegistryPoller:
    """Poll → reconcile stub. Never raises into the caller; backend optional."""

    def __init__(
        self,
        source: RegistrySource,
        supervisor: Any = None,
        *,
        poll_seconds: Optional[float] = None,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
        auth_backoff_seconds: float = AUTH_BACKOFF_SECONDS,
    ) -> None:
        from services.vision_settings import get_vision_settings

        configured = float(get_vision_settings().get("registry_poll_seconds", 30.0))
        self._source = source
        self._supervisor = supervisor
        self._poll_seconds = float(poll_seconds if poll_seconds is not None else configured)
        if self._poll_seconds <= 0:
            raise ValueError("poll interval must be > 0")
        self._clock = clock
        self._sleeper = sleeper
        self._auth_backoff = float(auth_backoff_seconds)
        self._auth_next_attempt = 0.0
        self._fingerprints: dict[str, tuple] = {}
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._last_error = ""
        self._polls = 0
        self._cameras_seen = 0
        self._thread = threading.Thread(target=self._run,
                                        name="registry-poller", daemon=True)
        self._started = False

    def start(self) -> None:
        if self._started:
            raise RegistryError("registry poller is already started")
        self._started = True
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> bool:
        self._stop.set()
        if self._started:
            self._thread.join(timeout=timeout)
        return not self._thread.is_alive()

    def poll_once(self) -> dict[str, int]:
        return self._poll()

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "source": self._source.describe(),
                "poll_seconds": self._poll_seconds,
                "polls": self._polls,
                "last_error": self._last_error,
                "cameras_seen": self._cameras_seen,
                "auth_backoff_active": self._clock() < self._auth_next_attempt,
            }

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self._poll()
            except Exception as exc:  # never die
                logger.warning("registry poll failed: %s: %s",
                               type(exc).__name__, exc)
            remaining = self._poll_seconds
            while remaining > 0 and not self._stop.is_set():
                step = min(0.5, remaining)
                self._sleeper(step)
                remaining -= step

    def _poll(self) -> dict[str, int]:
        outcome = {"added": 0, "removed": 0, "restarted": 0, "kept": 0}
        if self._clock() < self._auth_next_attempt:
            return outcome
        try:
            records = self._source.fetch()
        except RegistryAuthError as exc:
            self._auth_next_attempt = self._clock() + self._auth_backoff
            with self._lock:
                self._last_error = str(exc)
            return outcome
        except RegistryError as exc:
            with self._lock:
                self._last_error = str(exc)
            return outcome
        with self._lock:
            self._polls += 1
            self._cameras_seen = len(records)
            self._last_error = ""
        seen = {r.camera_id: r.fingerprint() for r in records}
        for camera_id, fingerprint in seen.items():
            if camera_id not in self._fingerprints:
                outcome["added"] += 1
            elif self._fingerprints[camera_id] != fingerprint:
                outcome["restarted"] += 1
            else:
                outcome["kept"] += 1
        for camera_id in list(self._fingerprints):
            if camera_id not in seen:
                outcome["removed"] += 1
        self._fingerprints = seen
        return outcome
