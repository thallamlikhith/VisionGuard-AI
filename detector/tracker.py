"""YOLO person detection + ByteTrack tracking.

Only COCO class 0 ("person") is processed, and tracking is delegated to
Ultralytics' built in ByteTrack implementation.

On an Intel CPU an OpenVINO export of the same weights is usually a lot
faster than the PyTorch path. It is optional: when it is missing or fails,
the .pt model is used and the app keeps working.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from typing import Optional

import numpy as np
from ultralytics import YOLO

LOG = logging.getLogger(__name__)

# Process-wide cache of loaded YOLO weights, keyed by (YOLO class, model
# name, openvino flag, imgsz). Loading the weights (and verifying the
# OpenVINO export with a smoke inference) costs seconds, and it used to run
# inside every POST /start - so every Start press and every video switch
# paid the full load again. The weights are read-only during inference, so
# sharing one object across sequential runs is safe; only one pipeline run
# is ever alive at a time (start refuses while the old thread lives).
# The YOLO *class* is part of the key so monkeypatched fakes in tests never
# get a stale cached object from another test.
_MODEL_CACHE: dict[tuple, tuple] = {}
_MODEL_CACHE_LOCK = threading.Lock()

# --- person-track confirmation gate -----------------------------------------
# A raw ByteTrack id is NOT a person yet: the COCO person head fires on
# small static objects (measured: a fire extinguisher read as a person for
# 28 frames at conf 0.25-0.33, never moving more than ~1 px). A track is
# only reported once it has proven itself, by EITHER route:
#
#   conf route   - PERSON_CONFIRM_FRAMES frames at/above PERSON_CONFIRM_CONF
#                  (measured gap: junk lifetime-max 0.333, real tracks 0.414+)
#   instant route- one frame at/above PERSON_INSTANT_CONF (a clear worker;
#                  strictly between the must-reject single spike 0.45 and the
#                  must-show still worker 0.9, and above every measured junk
#                  frame, so no observed false track ever takes it)
#   motion route - the box centre moved PERSON_CONFIRM_MOVE_RATIO of the
#                  track's own box height away from where it was first seen
#                  (ratio, never pixels, so near and far cameras behave alike)
#
# Whichever route fires first latches for the life of the id, so a worker
# who stops moving (or whose confidence dips for a few frames) stays
# reported. Until then the id is withheld from EVERY consumer - overlay,
# zone occupancy, dwell, crowd reuse and the PPE person count all read
# track(), so one gate filters them all with no per-consumer change.
#
# Tuning surface is config.yaml (detector.person_confirm_*); these module
# values are the measured defaults and the fallback when no config is read
# (tests, standalone use).
PERSON_CONFIRM_CONF = 0.4
PERSON_CONFIRM_FRAMES = 3
PERSON_INSTANT_CONF = 0.8
PERSON_CONFIRM_MOVE_RATIO = 0.05
# ids unseen for this many track() calls are forgotten (memory bound only;
# ByteTrack itself drops lost ids far sooner, so a live id never hits it)
PERSON_GATE_PRUNE_CALLS = 300
# suppression is logged at DEBUG on the 1st held frame and every Nth after,
# so a flood of held frames cannot spam the log but a wrongly held real
# worker is still visible. Release (held, then qualified) and prune (id
# forgotten) are logged once each with the same details.
PERSON_GATE_LOG_EVERY = 60

# one place for the format, so the CLI and the API print the same lines
LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
LOG_LEVEL = logging.INFO


def configure_logging(level: int = LOG_LEVEL) -> None:
    """Send log records to stderr with a time and level.

    Called by both entry points (app.py and api/server.py) so the line saying
    which detector backend is in use is actually visible. basicConfig does
    nothing when the root logger already has a handler, so it is safe to call
    twice and it does not touch pytest's or uvicorn's own handlers.
    """
    logging.basicConfig(level=level, format=LOG_FORMAT)


class DetectorError(Exception):
    """Raised when the model cannot be loaded or inference fails."""


@dataclass
class TrackedPerson:
    worker_id: int
    bbox: tuple[int, int, int, int]
    confidence: float


@dataclass
class _PersonGateEntry:
    """Confirmation state of one raw ByteTrack id (see gate constants above)."""

    frames_seen: int = 0
    frames_above_conf: int = 0
    max_conf: float = 0.0
    first_cx: float = 0.0
    first_cy: float = 0.0
    excursion_px: float = 0.0
    qualified: bool = False
    last_call: int = 0
    # frames seen-but-not-reported. Stops growing once qualified; reported
    # on release and on prune so a held-then-forgotten id leaves a trace.
    held_frames: int = 0


class PersonTracker:
    # detection frames an id-less box list is held for before it is dropped
    MAX_IDLESS_HOLD = 3

    def __init__(
        self,
        model_name: str = "yolo11n.pt",
        confidence: float = 0.25,
        person_class_id: int = 0,
        imgsz: int = 320,
        detect_every_n: int = 1,
        max_detect_long_side: int = 960,
        openvino: bool = False,
        adaptive_idle_n: int = 0,
        person_confirm_conf: float = PERSON_CONFIRM_CONF,
        person_confirm_frames: int = PERSON_CONFIRM_FRAMES,
        person_instant_conf: float = PERSON_INSTANT_CONF,
        person_confirm_move_ratio: float = PERSON_CONFIRM_MOVE_RATIO,
        person_gate_enabled: bool = True,
    ) -> None:
        self.person_class_id = int(person_class_id)
        self.confidence = float(confidence)
        self.imgsz = int(imgsz)
        self.detect_every_n = max(1, int(detect_every_n))
        self.max_detect_long_side = max(0, int(max_detect_long_side))
        # Confirmation gate (see module constants): the person-specific bar a
        # raw track id must clear before it is shown or counted. Deliberately
        # separate from `confidence`, which is the YOLO detection floor - the
        # PPE tuning (0.15/960/TTA/memory) lives in its own detector and never
        # reaches these. person_gate_enabled=False restores the pre-gate
        # behaviour (every ByteTrack id reported) for before/after comparison.
        self.person_confirm_conf = float(person_confirm_conf)
        self.person_confirm_frames = max(1, int(person_confirm_frames))
        self.person_instant_conf = float(person_instant_conf)
        self.person_confirm_move_ratio = max(0.0, float(person_confirm_move_ratio))
        self.person_gate_enabled = bool(person_gate_enabled)
        self._person_gate: dict[int, _PersonGateEntry] = {}
        self._calls = 0
        # Adaptive scheduling: while people are visible YOLO runs at the
        # configured detect_every_n, but an empty scene only needs a look
        # every adaptive_idle_n frames. 0 (or 1) disables it, so the fixed
        # detect_every_n schedule is used exactly as before.
        self.adaptive_idle_n = max(0, int(adaptive_idle_n))
        # frames since the last real inference, and the cadence in force now.
        # Counting calls instead of frame_index because the cadence itself
        # changes over time, so a fixed modulo no longer lines up.
        self._since_detect = 0
        self._cadence = self.detect_every_n
        self._last_people: list[TrackedPerson] = []
        # how many detection frames in a row came back with boxes but no ids.
        # The previous tracks are held for a few of those and then dropped.
        self._idless_streak = 0

        self.model, self.backend = self._load_model(model_name, openvino=openvino)
        LOG.info(
            "detector ready: backend=%s model=%s imgsz=%d detect_every_n=%d max_long_side=%d adaptive_idle_n=%d",
            self.backend,
            model_name,
            self.imgsz,
            self.detect_every_n,
            self.max_detect_long_side,
            self.adaptive_idle_n,
        )
        # Every start behaves like the strict cold case: burn ByteTrack's
        # frame 1 with a blank frame (see _burn_first_frame), including the
        # cache-hit path above, whose predictor reset would otherwise leave
        # frame 1 for the first real frame.
        self._burn_first_frame()

    def _burn_first_frame(self) -> None:
        """Run one blank frame through the model with persist=False.

        ByteTrack activates tracks seen on its frame 1 without needing a
        re-match, so whatever junk the detector reports on the first real
        frame would instantly become workers. Burning frame 1 with blanks
        forces every real frame down the confirm path instead: a box has to
        be re-matched on a later frame before it is reported, which is what
        keeps flickering false persons (flames, machinery) out while stable
        workers confirm within a frame or two. Best effort: a failure only
        logs, it never breaks startup (the first frame then behaves like
        the old lax case, exactly as before this burn existed).
        """
        try:
            blank = np.zeros((64, 64, 3), dtype=np.uint8)
            self.model.track(
                blank,
                persist=False,
                verbose=False,
                classes=[self.person_class_id],
                conf=self.confidence,
                imgsz=self.imgsz,
            )
        except Exception as exc:
            LOG.warning(
                "first-frame burn failed (%s): the first real frame may "
                "instant-activate tracks",
                exc,
            )

    # -- model loading ----------------------------------------------------

    @staticmethod
    def _openvino_dir(model_name: str) -> str:
        """Where the export of model_name lives: next to the .pt weights.

        Anchored to the weights instead of the current directory, so the
        export is found again no matter where the app was started from.
        """
        stem, extension = os.path.splitext(model_name)
        if extension:
            return f"{stem}_openvino_model"
        # a bare model name: ultralytics resolves it from the working directory
        resolved = os.path.abspath(YOLO(model_name).ckpt_path)
        return f"{os.path.splitext(resolved)[0]}_openvino_model"

    def _export_openvino(self, model_name: str, directory: str) -> bool:
        """Export the .pt weights for OpenVINO at the size we actually run.

        Ultralytics bakes a *static* input shape into the export. An export
        made at 640 and then used at imgsz=320 is not resized by OpenVINO, it
        is fed 640 sized input and wasted work, so the export has to match the
        configured imgsz. half=True gives an fp16 CPU model, which is faster
        again on this hardware.
        """
        try:
            YOLO(model_name).export(
                format="openvino", imgsz=self.imgsz, half=True, verbose=False
            )
        except Exception as exc:
            LOG.warning("OpenVINO export failed: %s - using %s", exc, model_name)
            return False
        return os.path.isdir(directory)

    def _exported_imgsz(self, directory: str) -> Optional[int]:
        """The input size of an existing export, or None if it cannot be read."""
        xml_files = [name for name in os.listdir(directory) if name.endswith(".xml")]
        if not xml_files:
            return None
        try:
            import openvino as ov

            model = ov.Core().read_model(os.path.join(directory, xml_files[0]))
            shape = model.input(0).get_partial_shape()
            if len(shape) == 4 and shape[2].is_static:
                # an OpenVINO Dimension does not cast with int(); get_length()
                # returns the concrete size the export was baked with
                return int(shape[2].get_length())
        except Exception:
            # openvino missing or an unreadable model: leave the decision to
            # the load below, which will fall back to the .pt weights
            return None
        return None

    def _openvino_usable(self, model: YOLO, directory: str, model_name: str) -> bool:
        """Check the export really runs here before committing to it.

        YOLO(directory) builds its backend lazily: on a machine without the
        openvino package it still constructs happily and reports
        backend=openvino, then fails on the first frame - long after the
        video thread has started. One blank frame is enough to find out now,
        so the fallback happens at startup with a clear message.
        """
        try:
            import openvino  # noqa: F401
        except ImportError as exc:
            LOG.warning(
                "the openvino package is not installed (%s), so the export in %s "
                "cannot be used - using %s",
                exc,
                directory,
                model_name,
            )
            return False

        try:
            model.track(
                np.zeros((64, 64, 3), dtype=np.uint8),
                persist=False,
                verbose=False,
                classes=[self.person_class_id],
                conf=self.confidence,
                imgsz=self.imgsz,
            )
        except Exception as exc:
            LOG.warning(
                "the OpenVINO export in %s could not run (%s) - using %s",
                directory,
                exc,
                model_name,
            )
            return False
        return True

    def _load_openvino(self, model_name: str) -> tuple[Optional[YOLO], str]:
        """Load the OpenVINO export, exporting it once if needed.

        Returns (model, "openvino") on success and (None, "") when OpenVINO
        is unavailable, the export fails, or the exported model cannot be
        loaded or run. The caller then falls back to the .pt weights.
        """
        directory = self._openvino_dir(model_name)

        if not os.path.isdir(directory):
            LOG.info("OpenVINO export not found, creating %s (one time)", directory)
            if not self._export_openvino(model_name, directory):
                return None, ""
        else:
            # an export made for another imgsz still loads, but it runs at its
            # baked in size, so it is replaced once for the size we want
            exported = self._exported_imgsz(directory)
            if exported is not None and exported != self.imgsz:
                LOG.info(
                    "OpenVINO export in %s is imgsz=%d but %d is configured - re-exporting once",
                    directory,
                    exported,
                    self.imgsz,
                )
                if not self._export_openvino(model_name, directory):
                    return None, ""

        try:
            model = YOLO(directory)
        except Exception as exc:
            LOG.warning(
                "could not load the OpenVINO model in %s: %s - using %s",
                directory,
                exc,
                model_name,
            )
            return None, ""

        if not self._openvino_usable(model, directory, model_name):
            return None, ""
        return model, "openvino"

    def _load_model(self, model_name: str, openvino: bool = False) -> tuple[YOLO, str]:
        key = (YOLO, str(model_name), bool(openvino), int(self.imgsz))
        with _MODEL_CACHE_LOCK:
            hit = _MODEL_CACHE.get(key)
        if hit is not None:
            model, backend = hit
            # The weights are shared but the tracking session is not: dropping
            # the predictor forces a fresh one (and fresh ByteTrack ids) on
            # the next track() call, so a new video never inherits the old
            # video's tracks. Ultralytics itself resets the predictor the
            # same way when switching tasks.
            try:
                model.predictor = None
            except Exception:
                pass
            LOG.info(
                "detector ready (cached): backend=%s model=%s imgsz=%d",
                backend,
                model_name,
                self.imgsz,
            )
            return model, backend

        if openvino:
            model, backend = self._load_openvino(model_name)
            if model is not None:
                with _MODEL_CACHE_LOCK:
                    _MODEL_CACHE.setdefault(key, (model, backend))
                return model, backend

        try:
            loaded = YOLO(model_name), "pytorch"
        except Exception as exc:  # ultralytics raises many different types
            raise DetectorError(
                f"could not load YOLO model '{model_name}': {exc}\n"
                "The weights are downloaded on first use - check your internet connection."
            ) from exc
        with _MODEL_CACHE_LOCK:
            _MODEL_CACHE.setdefault(key, loaded)
        return loaded

    def should_detect(self, frame_index: int) -> bool:
        """True when this frame is one of the frames YOLO actually runs on."""
        return frame_index % self.detect_every_n == 0

    def _schedule_next(self, reported: list[TrackedPerson]) -> None:
        """Pick the cadence for the coming frames from what was just seen.

        A no-op unless adaptive scheduling is enabled. People around means
        the configured detect_every_n (a new entry must be seen at once);
        an empty scene means adaptive_idle_n (nothing to miss). The decision
        is made from what is *reported*, so id-less frames holding stale
        tracks keep the fast cadence instead of going idle on a technicality.
        """
        if self.adaptive_idle_n <= 1:
            return
        self._cadence = self.detect_every_n if reported else self.adaptive_idle_n

    def track(self, frame, frame_index: int = 0) -> list[TrackedPerson]:
        """Detect + track people in one frame. Returns [] when nothing is found.

        With detect_every_n > 1 the frames in between reuse the last result
        instead of running YOLO again, which is where most of the time goes.
        The returned objects are the same as the previous frame's, so callers
        see a stable picture and the zone timer keeps advancing.
        """
        if self.adaptive_idle_n > 1:
            self._since_detect += 1
            if self._since_detect < self._cadence:
                return list(self._last_people)
            self._since_detect = 0
        elif not self.should_detect(frame_index):
            return list(self._last_people)

        # Optionally run detection on a downscaled frame to speed up; scale boxes back up
        detect_frame = frame
        scale = 1.0
        if self.max_detect_long_side > 0:
            h, w = frame.shape[:2] if hasattr(frame, "shape") else (0, 0)
            long_side = max(h, w) if h > 0 and w > 0 else 0
            if long_side > self.max_detect_long_side:
                scale = self.max_detect_long_side / long_side
                new_w = int(round(w * scale))
                new_h = int(round(h * scale))
                import cv2

                detect_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        self._last_scale = scale

        try:
            results = self.model.track(
                detect_frame,
                persist=True,
                tracker="bytetrack.yaml",
                classes=[self.person_class_id],
                conf=self.confidence,
                imgsz=self.imgsz,
                verbose=False,
            )
        except Exception as exc:
            raise DetectorError(f"YOLO inference failed: {exc}") from exc

        people: list[TrackedPerson] = []
        if not results:
            self._last_people = people
            self._schedule_next(people)
            return people

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            self._idless_streak = 0
            self._last_people = people
            self._schedule_next(people)
            return people  # nothing detected

        if boxes.id is None:
            # ByteTrack has not assigned an id to this frame's boxes yet (the
            # first frames of a new track, or a frame it could not match), so
            # there is no stable worker_id to report. An invented id would make
            # the zone logic see a brand new worker, restart its dwell timer and
            # write a second ZONE_ENTRY once the real id arrives. The previous
            # stable tracks are held instead, but only for MAX_IDLESS_HOLD
            # frames: after that the ids are not coming and holding them would
            # pin a stale count on screen. With no previous track this is just
            # empty, as before.
            self._idless_streak += 1
            if self._idless_streak <= self.MAX_IDLESS_HOLD:
                held = list(self._last_people)
                self._schedule_next(held)
                return held
            self._last_people = []
            self._schedule_next(people)
            return people

        ids = boxes.id.tolist()
        coords = boxes.xyxy.tolist()
        confidences = boxes.conf.tolist() if boxes.conf is not None else [0.0] * len(ids)

        scale = getattr(self, "_last_scale", 1.0)
        for track_id, (x1, y1, x2, y2), score in zip(ids, coords, confidences):
            if track_id is None:
                continue
            bx1 = int(round(x1 / scale))
            by1 = int(round(y1 / scale))
            bx2 = int(round(x2 / scale))
            by2 = int(round(y2 / scale))
            people.append(
                TrackedPerson(
                    worker_id=int(track_id),
                    bbox=(bx1, by1, bx2, by2),
                    confidence=float(score),
                )
            )

        people.sort(key=lambda person: person.worker_id)
        people = self._confirm_people(people)
        self._idless_streak = 0
        self._last_people = people
        self._schedule_next(people)
        return people

    def _confirm_people(self, people: list[TrackedPerson]) -> list[TrackedPerson]:
        """Withhold unproven track ids (see PERSON_CONFIRM_* above).

        Runs on inference frames only; every other path (frame reuse, idless
        hold) replays self._last_people, which is already filtered. Never
        raises: a bad box only forfeits the motion route for that frame.
        """
        self._calls += 1
        if not self.person_gate_enabled:
            return people
        kept: list[TrackedPerson] = []
        for person in people:
            entry = self._person_gate.get(person.worker_id)
            if entry is None:
                try:
                    x1, y1, x2, y2 = person.bbox
                    first_cx, first_cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                except (TypeError, ValueError):
                    first_cx, first_cy = 0.0, 0.0
                entry = _PersonGateEntry(
                    first_cx=first_cx, first_cy=first_cy, last_call=self._calls
                )
                self._person_gate[person.worker_id] = entry
            entry.frames_seen += 1
            entry.last_call = self._calls
            entry.max_conf = max(entry.max_conf, person.confidence)
            if person.confidence >= self.person_confirm_conf:
                entry.frames_above_conf += 1
            try:
                x1, y1, x2, y2 = person.bbox
                excursion = (
                    ((x1 + x2) / 2.0 - entry.first_cx) ** 2
                    + ((y1 + y2) / 2.0 - entry.first_cy) ** 2
                ) ** 0.5
                entry.excursion_px = max(entry.excursion_px, excursion)
                height = y2 - y1
            except (TypeError, ValueError):
                height = 0
            if not entry.qualified:
                route = ""
                if person.confidence >= self.person_instant_conf:
                    route = "instant"
                elif entry.frames_above_conf >= self.person_confirm_frames:
                    route = "confirm"
                elif (
                    height > 0
                    and entry.excursion_px >= self.person_confirm_move_ratio * height
                ):
                    route = "motion"
                if route:
                    entry.qualified = True
                    if entry.held_frames:
                        LOG.debug(
                            "person gate released track id=%d: held=%d frames "
                            "max_conf=%.2f qualified=True route=%s",
                            person.worker_id,
                            entry.held_frames,
                            entry.max_conf,
                            route,
                        )
            if entry.qualified:
                kept.append(person)
            else:
                entry.held_frames += 1
                if entry.frames_seen == 1 or entry.frames_seen % PERSON_GATE_LOG_EVERY == 0:
                    LOG.debug(
                        "person gate holding track id=%d: max_conf=%.2f "
                        "frames_seen=%d frames_above_conf=%d excursion=%.1fpx",
                        person.worker_id,
                        entry.max_conf,
                        entry.frames_seen,
                        entry.frames_above_conf,
                        entry.excursion_px,
                    )
        # memory bound only; a live id is seen far more often than this
        stale = [
            worker_id
            for worker_id, entry in self._person_gate.items()
            if self._calls - entry.last_call > PERSON_GATE_PRUNE_CALLS
        ]
        for worker_id in stale:
            entry = self._person_gate.pop(worker_id)
            LOG.debug(
                "person gate pruned track id=%d: held=%d frames "
                "max_conf=%.2f qualified=%s",
                worker_id,
                entry.held_frames,
                entry.max_conf,
                entry.qualified,
            )
        return kept
