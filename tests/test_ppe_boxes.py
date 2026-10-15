"""PPE overlay boxes: violations only, ticked only, grading-only (no rows)."""

import time

import cv2
import numpy as np
import pytest

from database.db import Database
from pipeline import Pipeline
from zone.danger_zone import Worker


class StubPerson:
    def __init__(self, *_args, **_kwargs):
        pass

    def track(self, frame, frame_index):
        return [Worker(worker_id=1, bbox=(0, 0, 64, 48), confidence=0.9)]


class MixedPPE:
    """One violation, one compliant box, one person box per analysis."""

    def __init__(self, *args, **kwargs):
        pass

    def predict(self, frame, confidence_threshold=None):
        return [
            {"class_name": "NO-Hardhat", "confidence": 0.8,
             "bbox_xyxy": [0, 0, 10, 10]},
            {"class_name": "Vest", "confidence": 0.9,
             "bbox_xyxy": [20, 20, 40, 40]},
            {"class_name": "Person", "confidence": 0.95,
             "bbox_xyxy": [0, 0, 64, 48]},
        ]


def write_clip(path):
    video = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
    assert video.isOpened()
    for _ in range(10):
        video.write(np.full((48, 64, 3), 40, dtype=np.uint8))
    video.release()
    return str(path)


def run_clip(pipe, clip, ppe_enabled):
    pipe.start(
        source=clip,
        machine={"id": "m", "name": "m"},
        points=[],
        allowed_people=0,
        ppe_enabled=ppe_enabled,
        realtime=False,
        loop=False,
    )
    deadline = time.monotonic() + 20.0
    while pipe.active and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not pipe.active, "ppe clip did not finish"


def test_ppe_boxes_cover_only_violations_while_ticked(tmp_path, monkeypatch):
    monkeypatch.setattr("pipeline.PersonTracker", StubPerson)
    monkeypatch.setattr("detector.ppe_detector.PPEDetector", MixedPPE)
    db = Database(str(tmp_path / "ppe-box.db"))
    pipe = Pipeline(db)
    clip = write_clip(tmp_path / "ppe-box.mp4")
    try:
        run_clip(pipe, clip, ppe_enabled=True)
        boxes = pipe._ppe_boxes_for_draw()
        assert [box["class_name"] for box in boxes] == ["NO-Hardhat"]
        # the grade still counts the same single violation
        assert pipe.ppe_stats["violation_count"] == 1
        # grading-only: nothing written anywhere
        assert db.list_events() == []
        assert db.list_fire_alerts() == []
    finally:
        pipe.stop()


def test_ppe_boxes_absent_when_unticked(tmp_path, monkeypatch):
    monkeypatch.setattr("pipeline.PersonTracker", StubPerson)
    monkeypatch.setattr("detector.ppe_detector.PPEDetector", MixedPPE)
    pipe = Pipeline(Database(str(tmp_path / "ppe-off.db")))
    clip = write_clip(tmp_path / "ppe-off.mp4")
    try:
        run_clip(pipe, clip, ppe_enabled=False)
        assert pipe._ppe_boxes_for_draw() == []
    finally:
        pipe.stop()


def test_ppe_operating_point_reaches_the_detector(tmp_path, monkeypatch):
    """The tuned pass (conf 0.15, imgsz 960) is what start() builds, so the
    flicker fix is live wherever the defaults flow."""
    seen = {}

    class CapturingPPE:
        def __init__(self, *args, **kwargs):
            seen.update(kwargs)

        def predict(self, frame, confidence_threshold=None):
            return []

    monkeypatch.setattr("pipeline.PersonTracker", StubPerson)
    monkeypatch.setattr("detector.ppe_detector.PPEDetector", CapturingPPE)
    pipe = Pipeline(Database(str(tmp_path / "ppe-op.db")))
    clip = write_clip(tmp_path / "ppe-op.mp4")
    try:
        run_clip(pipe, clip, ppe_enabled=True)
        assert seen.get("confidence") == pytest.approx(0.15)
        assert seen.get("imgsz") == 960
        assert seen.get("tta") is True
    finally:
        pipe.stop()
