"""Writes the picture that goes with an alert.

The video thread cannot stop to touch the disk, so an alert JPEG is handed to
a background thread through a small bounded queue. The database stores only the
bare file name; the route that serves it joins that name against the snapshot
directory, so a value from the database can never point outside it.
"""

from __future__ import annotations

import logging
import os
import queue
import threading
from typing import Optional

ROOT = os.path.dirname(os.path.abspath(__file__))


def event_snapshot_dir() -> str:
    """The one directory alert snapshots live in, for the writer and the route."""
    return os.environ.get(
        "TRINETRA_SNAPSHOT_DIR", os.path.join(ROOT, "data", "snapshots")
    )


def fire_snapshot_dir() -> str:
    """The one directory fire/smoke alert snapshots live in."""
    return os.environ.get(
        "TRINETRA_FIRE_SNAPSHOT_DIR", os.path.join(ROOT, "data", "fire_snapshots")
    )


def hard_negatives_dir() -> str:
    """The one directory false-alarm retraining exports live in."""
    return os.environ.get(
        "TRINETRA_HARD_NEGATIVES_DIR", os.path.join(ROOT, "data", "hard_negatives")
    )


def write_fire_snapshot(alert_id: int, jpeg: bytes, keep: int = 300) -> Optional[str]:
    """Save one fire/smoke alert JPEG synchronously; returns the bare name.

    Fire alerts are rare (at most one per presence episode and paced by the
    background worker), so unlike zone alerts this needs no queue thread.
    Keeps the newest `keep` pictures; never raises. The database stores only
    the bare name; the route joins it against `fire_snapshot_dir()`, so a
    value from the database can never point outside it.
    """
    log = logging.getLogger(__name__)
    if not jpeg:
        return None
    try:
        os.makedirs(fire_snapshot_dir(), exist_ok=True)
    except OSError as exc:
        log.warning("could not create fire snapshot folder %s: %s",
                    fire_snapshot_dir(), exc)
        return None
    name = f"{int(alert_id)}.jpg"
    try:
        with open(os.path.join(fire_snapshot_dir(), name), "wb") as handle:
            handle.write(bytes(jpeg))
    except OSError as exc:
        log.warning("could not write fire snapshot %s: %s", name, exc)
        return None
    try:
        names = [n for n in os.listdir(fire_snapshot_dir()) if n.endswith(".jpg")]
    except OSError:
        return name
    if len(names) > keep:
        try:
            names.sort(
                key=lambda n: os.path.getmtime(os.path.join(fire_snapshot_dir(), n)),
                reverse=True,
            )
        except OSError:
            return name
        for stale in names[keep:]:
            try:
                os.remove(os.path.join(fire_snapshot_dir(), stale))
            except OSError as exc:
                log.warning("could not prune fire snapshot %s: %s", stale, exc)
    return name


def write_fire_clean_snapshot(alert_id: int, jpeg: bytes, keep: int = 300) -> Optional[str]:
    """Save the clean (pre-overlay) frame for one fire/smoke alert.

    Same hygiene and prune behaviour as the annotated snapshot: bare
    ``<id>_clean.jpg`` name, newest `keep` pictures kept, never raises.
    """
    log = logging.getLogger(__name__)
    if not jpeg:
        return None
    try:
        os.makedirs(fire_snapshot_dir(), exist_ok=True)
    except OSError as exc:
        log.warning("could not create fire snapshot folder %s: %s",
                    fire_snapshot_dir(), exc)
        return None
    name = f"{int(alert_id)}_clean.jpg"
    try:
        with open(os.path.join(fire_snapshot_dir(), name), "wb") as handle:
            handle.write(bytes(jpeg))
    except OSError as exc:
        log.warning("could not write fire clean snapshot %s: %s", name, exc)
        return None
    try:
        names = [n for n in os.listdir(fire_snapshot_dir()) if n.endswith(".jpg")]
    except OSError:
        return name
    if len(names) > keep:
        try:
            names.sort(
                key=lambda n: os.path.getmtime(os.path.join(fire_snapshot_dir(), n)),
                reverse=True,
            )
        except OSError:
            return name
        for stale in names[keep:]:
            try:
                os.remove(os.path.join(fire_snapshot_dir(), stale))
            except OSError as exc:
                log.warning("could not prune fire snapshot %s: %s", stale, exc)
    return name


class HardNegativeError(Exception):
    """The clean frame cannot be exported (missing, unreadable, unwritable)."""


def export_hard_negative(alert_type: str, reported_at, alert_id: int,
                         clean_src_path: str, metadata: dict) -> tuple:
    """Copy a clean frame into data/hard_negatives with its JSON sidecar.

    `metadata` is written verbatim as ``<same name>.json`` next to the
    picture. Returns ``(jpg_name, json_name)`` (bare names). The file name
    is ``<type>_<utc>.jpg`` from the alert's report time, suffixed with the
    alert id when that name is already taken by another alert. Raises
    HardNegativeError (never anything else) when the source is missing or
    nothing can be written.
    """
    import json
    from datetime import datetime, timezone

    try:
        reported = float(reported_at) if reported_at is not None else None
    except (TypeError, ValueError):
        reported = None
    if reported is not None:
        stamp = datetime.fromtimestamp(reported, tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    else:
        stamp = "unknowntime"
    try:
        os.makedirs(hard_negatives_dir(), exist_ok=True)
    except OSError as exc:
        raise HardNegativeError(f"could not create {hard_negatives_dir()}: {exc}") from exc
    base = f"{str(alert_type)}_{stamp}"
    jpg_name = f"{base}.jpg"
    if os.path.exists(os.path.join(hard_negatives_dir(), jpg_name)):
        jpg_name = f"{base}_{int(alert_id)}.jpg"
    json_name = jpg_name[: -len(".jpg")] + ".json"
    try:
        with open(clean_src_path, "rb") as source:
            pixels = source.read()
        if not pixels:
            raise HardNegativeError(f"clean frame is empty: {clean_src_path}")
        with open(os.path.join(hard_negatives_dir(), jpg_name), "wb") as handle:
            handle.write(pixels)
        with open(os.path.join(hard_negatives_dir(), json_name), "w",
                   encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=1)
    except OSError as exc:
        raise HardNegativeError(f"could not export {jpg_name}: {exc}") from exc
    return jpg_name, json_name


class SnapshotWriter:
    """A daemon thread that saves alert JPEGs without blocking the video."""

    # small on purpose: a burst longer than this loses the extras rather than
    # growing without bound or making the video thread wait
    MAX_QUEUE = 32
    # how many alert pictures to keep; older ones are deleted as new ones land,
    # so a long-running camera cannot fill the disk with evidence
    KEEP = 300

    def __init__(self, database) -> None:
        self._db = database
        self._queue: "queue.Queue[Optional[tuple]]" = queue.Queue(maxsize=self.MAX_QUEUE)
        self._thread: Optional[threading.Thread] = None
        self.directory = event_snapshot_dir()

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        try:
            os.makedirs(self.directory, exist_ok=True)
        except OSError as exc:
            logging.getLogger(__name__).warning(
                "could not create snapshot folder %s: %s", self.directory, exc
            )
        self._thread = threading.Thread(target=self._run, name="snapshot-writer", daemon=True)
        self._thread.start()

    def submit(self, event_id: int, jpeg: bytes) -> None:
        """Queue one frame for the event. Never raises and never blocks."""
        if not jpeg:
            return
        try:
            self._queue.put_nowait((int(event_id), bytes(jpeg)))
        except queue.Full:
            logging.getLogger(__name__).warning(
                "snapshot queue is full - dropped the picture for event %s", event_id
            )

    def close(self, timeout: float = 2.0) -> None:
        """Stop the thread and drain what is left, then let it die."""
        thread = self._thread
        if thread is None:
            return
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            # the writer will drain the queue and then see the thread stop
            pass
        thread.join(timeout)
        self._thread = None

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    return
                self._write(*item)
            finally:
                self._queue.task_done()

    def _write(self, event_id: int, jpeg: bytes) -> None:
        name = f"{event_id}.jpg"
        path = os.path.join(self.directory, name)
        try:
            with open(path, "wb") as handle:
                handle.write(jpeg)
        except OSError as exc:
            logging.getLogger(__name__).warning("could not write snapshot %s: %s", path, exc)
            return
        try:
            self._db.set_snapshot_path(event_id, name)
        except Exception as exc:  # the file is saved; a DB hiccup must not kill the thread
            logging.getLogger(__name__).warning(
                "could not record snapshot for event %s: %s", event_id, exc
            )
        self._prune()

    def _prune(self) -> None:
        """Keep only the newest KEEP pictures.

        Listing the folder after every write is cheap for a handful of files and
        means a camera left running for weeks still cannot grow the folder
        without bound. A file that cannot be removed is skipped rather than
        allowed to end the writer thread.
        """
        log = logging.getLogger(__name__)
        try:
            names = [n for n in os.listdir(self.directory) if n.endswith(".jpg")]
        except OSError as exc:
            log.warning("could not list snapshot folder %s: %s", self.directory, exc)
            return
        if len(names) <= self.KEEP:
            return
        try:
            names.sort(
                key=lambda n: os.path.getmtime(os.path.join(self.directory, n)), reverse=True
            )
        except OSError:
            return
        for name in names[self.KEEP :]:
            try:
                os.remove(os.path.join(self.directory, name))
            except OSError as exc:
                log.warning("could not prune snapshot %s: %s", name, exc)