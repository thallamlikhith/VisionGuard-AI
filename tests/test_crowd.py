"""Crowd side-path tests (Stage 2): pure metrics + pipeline wiring.

No model inference here except the class-table check (weights load only);
cadence uses fake detectors and plain timestamps. Annotation only: crowd
never writes zone events — reuse/model routes only fill `state()["crowd"]`.
"""

import time

import pytest

from database.db import Database
from detector.crowd import (
    CROWD_CLASSES,
    CROWD_MODEL_NAME,
    CROWD_WEIGHTS_FILE,
    calculate_crowd_metrics,
    is_person_detection,
)
from pipeline import Pipeline, PipelineError


def det(name, box=(0, 0, 10, 10)):
    return {"class_name": name, "confidence": 0.9, "bbox_xyxy": list(box)}


def test_constants_preserved():
    assert CROWD_MODEL_NAME == "crowd"
    assert CROWD_WEIGHTS_FILE == "crowd_v1.pt"
    assert CROWD_CLASSES == ("person",)


def test_metrics_math():
    out = calculate_crowd_metrics(
        [det("person", (0, 0, 50, 50)), det("Person", (50, 50, 100, 100))], 100, 100
    )
    assert out["person_count"] == 2
    assert out["coverage_fraction"] == pytest.approx(0.5)
    assert out["density_per_megapixel"] == pytest.approx(200.0)


def test_metrics_ignore_non_person_and_bad_boxes():
    out = calculate_crowd_metrics(
        [
            det("person", (0, 0, 10, 10)),
            det("PERSON", (0, 0, 10, 10)),
            det("car", (0, 0, 90, 90)),
            det("Personx", (0, 0, 90, 90)),
            {"class_name": "person", "confidence": 0.9},  # no box
            det("person", (20, 20, 5, 5)),  # degenerate
        ],
        100,
        100,
    )
    assert out["person_count"] == 2
    assert out["coverage_fraction"] == pytest.approx(0.02)


def test_metrics_caps_and_guards():
    full = calculate_crowd_metrics([det("person", (0, 0, 200, 200))], 100, 100)
    assert full["coverage_fraction"] == 1.0
    empty = calculate_crowd_metrics([], 0, 0)
    assert empty == {"person_count": 0, "coverage_fraction": 0.0,
                     "density_per_megapixel": 0.0}


def test_metrics_accept_all_shapes():
    class Obj:
        def __init__(self, name, box):
            self.class_name = name
            self.confidence = 0.9
            self.bbox_xyxy = box

    dets = [
        {"class_name": "person", "confidence": 0.9, "bounding_box": (0, 0, 10, 10)},
        {"class_name": "Person", "confidence": 0.9, "bbox": [10, 10, 20, 20]},
        Obj("person", (20, 20, 30, 30)),
    ]
    assert is_person_detection(dets[0])
    assert not is_person_detection({"class_name": "car"})
    out = calculate_crowd_metrics(dets, 100, 100)
    assert out["person_count"] == 3


def test_crowd_weights_have_person_class():
    from ultralytics import YOLO

    from detector.crowd_detector import check_person_class

    model = YOLO("detector/models/crowd_v1.pt")
    check_person_class(model.names)  # must not raise
    with pytest.raises(Exception):
        check_person_class({0: "car", 1: "dog"})


def test_disabled_side_path_never_runs(tmp_path):
    pipe = Pipeline(Database(str(tmp_path / "crowd-off.db")))
    assert pipe.crowd_enabled is False
    assert pipe.crowd_detector is None
    assert pipe._maybe_run_crowd(None, [], 320, 180, 100.0) is None
    assert pipe.state()["crowd"] == {"enabled": False}


def test_reuse_route_uses_tracker_boxes_every_frame(tmp_path):
    from detector.tracker import TrackedPerson

    pipe = Pipeline(Database(str(tmp_path / "crowd-reuse.db")))
    pipe.crowd_enabled = True
    pipe.crowd_source = "reuse"
    people = [TrackedPerson(worker_id=1, bbox=(0, 0, 50, 50), confidence=0.9),
              TrackedPerson(worker_id=2, bbox=(50, 50, 100, 100), confidence=0.8)]
    first = pipe._maybe_run_crowd(None, people, 100, 100, 100.0)
    second = pipe._maybe_run_crowd(None, people, 100, 100, 100.01)
    assert first["person_count"] == 2 and second["person_count"] == 2
    assert first["coverage_fraction"] == pytest.approx(0.5)
    assert pipe.crowd_detector is None  # reuse needs no model
    state = pipe.state()["crowd"]
    assert state["enabled"] is True and state["source"] == "reuse"
    assert state["metrics"]["person_count"] == 2


class FakeCrowdDetector:
    def __init__(self):
        self.calls = 0

    def predict(self, frame, confidence_threshold=None):
        self.calls += 1
        return [{"class_name": "person", "confidence": 0.8,
                 "bbox_xyxy": [0, 0, 20, 20], "track_id": None}]


def test_model_route_is_time_paced(tmp_path):
    pipe = Pipeline(Database(str(tmp_path / "crowd-model.db")))
    pipe.crowd_enabled = True
    pipe.crowd_source = "model"
    pipe.crowd_detector = FakeCrowdDetector()
    pipe.crowd_analysis_fps = 2.0
    assert pipe._maybe_run_crowd(None, [], 100, 100, 200.0)["person_count"] == 1
    assert pipe.crowd_detector.calls == 1
    # 0.1 s later: skipped by elapsed time, previous metrics repeated
    assert pipe._maybe_run_crowd(None, [], 100, 100, 200.1)["person_count"] == 1
    assert pipe.crowd_detector.calls == 1
    # 0.6 s later: analyzed again
    pipe._maybe_run_crowd(None, [], 100, 100, 200.6)
    assert pipe.crowd_detector.calls == 2


def test_start_rejects_bad_crowd_source(tmp_path, monkeypatch):
    import cv2
    import numpy as np

    from pipeline import Pipeline as RealPipeline

    class StubTracker:
        def __init__(self, *_a, **_k):
            pass

        def track(self, frame, frame_index):
            return []

    monkeypatch.setattr("pipeline.PersonTracker", StubTracker)
    clip = str(tmp_path / "src.mp4")
    video = cv2.VideoWriter(clip, cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
    assert video.isOpened()
    for _ in range(5):
        video.write(np.zeros((48, 64, 3), dtype=np.uint8))
    video.release()

    pipe = RealPipeline(Database(str(tmp_path / "src.db")))
    try:
        with pytest.raises(PipelineError, match="crowd source"):
            pipe.start(
                source=clip,
                machine={"id": "m", "name": "m"},
                points=[],
                crowd_enabled=True,
                crowd_source="bogus",
            )
    finally:
        pipe.stop()


def test_config_crowd_defaults_off():
    import os

    from config import load_config

    settings = load_config(os.path.join(os.path.dirname(__file__), os.pardir, "config.yaml"))
    assert settings["crowd"]["enabled"] is False
    assert settings["crowd"]["source"] == "reuse"
    assert settings["crowd"]["confidence"] == pytest.approx(0.35)
    assert settings["crowd"]["analysis_fps"] == pytest.approx(2.0)
