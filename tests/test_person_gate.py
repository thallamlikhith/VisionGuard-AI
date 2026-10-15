"""Regression tests for the person-track confirmation gate.

A raw ByteTrack id is not a person yet: the COCO person head fires on small
static objects (a fire extinguisher held a track for 28 frames at conf
0.25-0.33 without moving a pixel). PersonTracker.withhold such ids until
they qualify by confidence (3 frames >= 0.4, or one frame >= 0.6) or by
motion (centre moved 5% of the box height), latched per id.

The real YOLO weights are never loaded here - model loading is monkeypatched
exactly like tests/test_detector.py, so these stay fast and offline.
"""

import pytest

from detector import tracker as tracker_module
from detector.tracker import (
    PERSON_CONFIRM_CONF,
    PERSON_CONFIRM_FRAMES,
    PERSON_CONFIRM_MOVE_RATIO,
    PERSON_INSTANT_CONF,
    PersonTracker,
)


class FakeResults:
    def __init__(self, people):
        self.boxes = _Boxes(people)


class _Array(list):
    def tolist(self):
        return list(self)


class _Boxes:
    def __init__(self, people):
        self.id = _Array([p["id"] for p in people]) if people else None
        self.xyxy = _Array([p["box"] for p in people]) if people else None
        self.conf = _Array([p["conf"] for p in people]) if people else None

    def __len__(self):
        return len(self.xyxy) if self.xyxy is not None else 0


class ScriptedModel:
    """Plays back one result list per track() call (burn consumes the first).

    Patched in as the YOLO *class*: PersonTracker calls YOLO(model_name),
    which lands here via __call__, exactly like test_detector.ScriptedModel.
    A fresh instance per test keeps the model cache from leaking state.
    """

    def __init__(self, script):
        self.script = script
        self.calls = 0

    def __call__(self, *args, **kwargs):
        return self

    def track(self, frame, **kwargs):
        index = min(self.calls, len(self.script) - 1)
        self.calls += 1
        return [FakeResults(self.script[index])]


def make_tracker(monkeypatch, script, **kwargs):
    monkeypatch.setattr(tracker_module, "YOLO", ScriptedModel(script))
    return PersonTracker("yolo11n.pt", detect_every_n=1, **kwargs)


def box(x1, y1, x2, y2):
    return [x1, y1, x2, y2]


def test_gate_constants_match_the_measured_gap():
    """The bar sits between the junk (lifetime-max 0.333) and real tracks."""
    assert PERSON_CONFIRM_CONF == 0.4
    assert PERSON_CONFIRM_FRAMES == 3
    assert PERSON_CONFIRM_MOVE_RATIO == 0.05
    # the instant route is strictly between the must-reject single spike
    # (0.45) and the must-show still worker (0.9)
    assert 0.45 < PERSON_INSTANT_CONF <= 0.9


def test_static_low_conf_object_is_never_reported(monkeypatch):
    """The extinguisher replica: conf 0.33, ~1px jitter, 10 frames."""
    frames = [
        [{"id": 17, "box": box(94 + (i % 2), 398, 141 + (i % 2), 514), "conf": 0.33}]
        for i in range(10)
    ]
    tracker = make_tracker(monkeypatch, [[]] + frames)
    seen = [tracker.track(object(), i) for i in range(10)]
    assert all(people == [] for people in seen)


def test_moving_low_conf_person_is_reported_and_stays_reported(monkeypatch):
    """Conf 0.35 never reaches either confidence bar, but the walk qualifies
    it; once latched it stays reported after it stops moving."""
    h, w = 200, 90  # 5% of height = 10px
    moving = [
        [{"id": 15, "box": box(100 + 5 * i, 100, 100 + 5 * i + w, 100 + h), "conf": 0.35}]
        for i in range(4)  # excursions 0, 5, 10, 15px -> qualifies on frame 3
    ]
    stopped = [
        [{"id": 15, "box": box(120, 100, 120 + w, 100 + h), "conf": 0.35}]
        for _ in range(4)
    ]
    tracker = make_tracker(monkeypatch, [[]] + moving + stopped)
    seen = [tracker.track(object(), i) for i in range(8)]
    assert seen[0] == []  # excursion 0, conf below every bar
    assert seen[1] == []  # excursion 5px < 10px
    assert [p.worker_id for p in seen[2]] == [15]  # excursion 10px qualifies
    assert all([p.worker_id for p in frame] == [15] for frame in seen[3:])
    assert tracker._person_gate[15].qualified  # latched, not re-proven


def test_still_high_conf_person_is_reported_on_frame_one(monkeypatch):
    """A clear worker standing still is never held: one 0.9 frame qualifies."""
    person = [{"id": 15, "box": box(400, 200, 523, 517), "conf": 0.9}]
    tracker = make_tracker(monkeypatch, [[], person])
    first = tracker.track(object(), 0)
    assert [p.worker_id for p in first] == [15]


def test_single_mid_conf_spike_on_static_object_does_not_qualify(monkeypatch):
    """One 0.45 frame is below the instant bar and alone on the confirm bar."""
    spike = [[{"id": 17, "box": box(94, 398, 141, 514), "conf": 0.45}]]
    tracker = make_tracker(monkeypatch, [[]] + spike + [[], []])
    assert tracker.track(object(), 0) == []
    assert tracker.track(object(), 1) == []
    assert tracker.track(object(), 2) == []
    # ...but three such frames in a row are a real detection, not a spike
    tracker2 = make_tracker(monkeypatch, [[]] + spike + spike + spike)
    tracker2.track(object(), 0)
    tracker2.track(object(), 1)
    assert [p.worker_id for p in tracker2.track(object(), 2)] == [17]


def test_latch_persists_for_the_track_lifetime(monkeypatch, caplog):
    """Qualified by motion, then conf collapses to 0.2 and it stops: still
    reported, because the latch - not each frame - carries it."""
    import logging

    qualified = [
        [{"id": 15, "box": box(100 + 30 * i, 100, 190 + 30 * i, 300), "conf": 0.5}]
        for i in range(3)  # excursion hits 30px on frame 2: motion qualifies
    ]
    faded = [
        [{"id": 15, "box": box(190, 100, 280, 300), "conf": 0.2}]
        for _ in range(4)
    ]
    tracker = make_tracker(monkeypatch, [[]] + qualified + faded)
    with caplog.at_level(logging.DEBUG, logger="detector.tracker"):
        seen = [tracker.track(object(), i) for i in range(7)]
    assert seen[0] == []  # first sighting never qualifies by itself
    assert all([p.worker_id for p in frame] == [15] for frame in seen[1:])
    assert tracker._person_gate[15].qualified


def test_suppression_is_logged_with_track_details(monkeypatch, caplog):
    """A held track leaves a rate-limited DEBUG line with id, max conf,
    frames seen and displacement, so a missed real person is visible."""
    import logging

    frames = [
        [{"id": 17, "box": box(94, 398, 141, 514), "conf": 0.33}]
        for _ in range(3)
    ]
    tracker = make_tracker(monkeypatch, [[]] + frames)
    with caplog.at_level(logging.DEBUG, logger="detector.tracker"):
        tracker.track(object(), 0)
    messages = [r.getMessage() for r in caplog.records]
    assert any(
        "id=17" in m and "max_conf=0.33" in m and "frames_seen=1" in m
        and "excursion=" in m
        for m in messages
    )


def test_gate_can_be_disabled_for_before_after_comparison(monkeypatch):
    """person_gate_enabled=False restores the pre-gate behaviour exactly."""
    frames = [
        [{"id": 17, "box": box(94, 398, 141, 514), "conf": 0.33}]
        for _ in range(3)
    ]
    tracker = make_tracker(monkeypatch, [[]] + frames, person_gate_enabled=False)
    assert [p.worker_id for p in tracker.track(object(), 0)] == [17]


def test_release_is_logged_with_route_and_held_frames(monkeypatch, caplog):
    """A track held for 2 frames then qualified by motion logs its release:
    id, frames held, max conf, qualified flag and the winning route."""
    import logging

    from detector.tracker import PERSON_CONFIRM_MOVE_RATIO

    h = 200
    script = [
        [],
        [{"id": 15, "box": box(100, 100, 190, 100 + h), "conf": 0.35}],
        [{"id": 15, "box": box(104, 100, 194, 100 + h), "conf": 0.35}],
        [{"id": 15, "box": box(120, 100, 210, 100 + h), "conf": 0.35}],
    ]
    need = PERSON_CONFIRM_MOVE_RATIO * h
    assert 4 < need <= 20  # frame 2 still held, frame 3 released by motion
    tracker = make_tracker(monkeypatch, script)
    with caplog.at_level(logging.DEBUG, logger="detector.tracker"):
        assert tracker.track(object(), 0) == []
        assert tracker.track(object(), 1) == []
        assert [p.worker_id for p in tracker.track(object(), 2)] == [15]
    messages = [r.getMessage() for r in caplog.records]
    assert any(
        "released track id=15" in m and "held=2 frames" in m
        and "max_conf=0.35" in m and "qualified=True" in m and "route=motion" in m
        for m in messages
    )


def test_prune_is_logged_with_qualified_flag(monkeypatch, caplog):
    """An id unseen for PERSON_GATE_PRUNE_CALLS is forgotten with one DEBUG
    line saying whether it ever qualified."""
    import logging

    from detector import tracker as tracker_module

    monkeypatch.setattr(tracker_module, "PERSON_GATE_PRUNE_CALLS", 3)
    script = [
        [],
        [{"id": 17, "box": box(94, 398, 141, 514), "conf": 0.33}],
        [{"id": 9, "box": box(400, 200, 523, 517), "conf": 0.9}],
        [{"id": 9, "box": box(400, 200, 523, 517), "conf": 0.9}],
        [{"id": 9, "box": box(400, 200, 523, 517), "conf": 0.9}],
        [{"id": 9, "box": box(400, 200, 523, 517), "conf": 0.9}],
    ]
    tracker = make_tracker(monkeypatch, script)
    with caplog.at_level(logging.DEBUG, logger="detector.tracker"):
        for i in range(5):
            tracker.track(object(), i)
    messages = [r.getMessage() for r in caplog.records]
    assert any(
        "pruned track id=17" in m and "held=1 frames" in m
        and "max_conf=0.33" in m and "qualified=False" in m
        for m in messages
    )
