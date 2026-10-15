"""PPE cadence, stagger, reuse-person wiring (Stage 3).

Fake detectors + plain timestamps: no model, no video. The golden test
covers the real weights; this file covers the pipeline contract:
1.0 Hz wall-paced cadence, at-most-one heavy analysis per frame with
fire > crowd > PPE priority, tracker-reused person counts, grade-only.
"""

import pytest

from database.db import Database
from detector.fire_smoke_gate import FireSmokeGate
from detector.ppe_risk import PPE_EMISSION_DEFERRED
from detector.tracker import TrackedPerson
from pipeline import Pipeline


class FakeFire:
    def __init__(self):
        self.calls = 0

    def predict(self, frame, confidence_threshold=None):
        self.calls += 1
        return [{"class_name": "Fire", "confidence": 0.9,
                 "bbox_xyxy": [5, 5, 20, 20], "track_id": None}]


class FakeCrowd:
    def __init__(self):
        self.calls = 0

    def predict(self, frame, confidence_threshold=None):
        self.calls += 1
        return [{"class_name": "person", "confidence": 0.8,
                 "bbox_xyxy": [0, 0, 20, 20], "track_id": None}]


class FakePPE:
    def __init__(self, dets=None):
        self.calls = 0
        self.dets = dets if dets is not None else [
            {"class_name": "NO-Hardhat", "confidence": 0.8,
             "bbox_xyxy": [0, 0, 10, 10], "track_id": None}]

    def predict(self, frame, confidence_threshold=None):
        self.calls += 1
        return [dict(d) for d in self.dets]


class FakeEmitter:
    def emit(self, payload):
        return "sent"


def make_pipe(tmp_path):
    pipe = Pipeline(Database(str(tmp_path / "ppe.db")))
    pipe.fire_smoke = FakeFire()
    pipe.fire_smoke_gate = FireSmokeGate(fire_seconds=1.5, smoke_seconds=2.5,
                                         default_fps=2.0)
    pipe.fire_smoke_camera_id = "m"
    pipe.fire_smoke_gate.set_camera_fps("m", 2.0)
    pipe.fire_smoke_analysis_fps = 2.0
    pipe.fire_smoke_confidence = 0.35
    pipe.fire_smoke_emitter = FakeEmitter()
    pipe.crowd_enabled = True
    pipe.crowd_source = "model"
    pipe.crowd_detector = FakeCrowd()
    pipe.crowd_analysis_fps = 2.0
    pipe.ppe_enabled = True
    pipe.ppe_detector = FakePPE()
    pipe.ppe_analysis_fps = 1.0
    pipe.ppe_confidence = 0.25
    return pipe


def people(n=2):
    return [TrackedPerson(worker_id=i + 1, bbox=(i * 60, 0, i * 60 + 40, 100),
                          confidence=0.9) for i in range(n)]


def test_stagger_never_runs_two_heavy_analyses_on_one_frame(tmp_path):
    pipe = make_pipe(tmp_path)
    winners = [pipe._run_side_paths(None, people(), 100, 100, 100.0 + k / 25.0)
               for k in (0, 1, 2, 3)]
    # t=100.00 all due -> fire; t=100.04 fire not due -> crowd;
    # t=100.08 fire/crowd not due -> ppe; t=100.12 none due.
    assert winners == ["fire_smoke", "crowd", "ppe", None]
    assert (pipe.fire_smoke.calls, pipe.crowd_detector.calls,
            pipe.ppe_detector.calls) == (1, 1, 1)


def test_stagger_keeps_priority_on_later_frames(tmp_path):
    pipe = make_pipe(tmp_path)
    pipe._run_side_paths(None, people(), 100, 100, 100.0)  # fire wins
    # t=100.5: fire due again (and crowd/ppe long due) -> fire wins by priority
    assert pipe._run_side_paths(None, people(), 100, 100, 100.5) == "fire_smoke"
    assert (pipe.fire_smoke.calls, pipe.crowd_detector.calls,
            pipe.ppe_detector.calls) == (2, 0, 0)


def test_ppe_cadence_is_one_hertz(tmp_path):
    pipe = make_pipe(tmp_path)
    pipe.fire_smoke = None  # isolate ppe
    pipe.crowd_enabled = False
    assert pipe._maybe_run_ppe(None, people(), 300.0)["analyzed"] == 1
    assert pipe.ppe_detector.calls == 1
    assert pipe._maybe_run_ppe(None, people(), 300.5)["analyzed"] == 1
    assert pipe.ppe_detector.calls == 1  # skipped by elapsed time
    assert pipe._maybe_run_ppe(None, people(), 301.0)["analyzed"] == 2
    assert pipe.ppe_detector.calls == 2


def test_ppe_reuses_tracker_person_count(tmp_path):
    pipe = make_pipe(tmp_path)
    out = pipe._run_ppe(None, people(2))
    assert out["person_count"] == 2  # from the tracker, not the PPE pass
    assert out["violation_count"] == 1
    assert out["violation_ratio"] == pytest.approx(0.5)
    assert out["grade"] == "high"
    assert pipe.state()["ppe"]["grade"] == "high"


def test_ppe_model_person_boxes_do_not_inflate_count(tmp_path):
    pipe = make_pipe(tmp_path)
    pipe.ppe_detector = FakePPE(dets=[
        {"class_name": "Person", "confidence": 0.9, "bbox_xyxy": [0, 0, 9, 9]},
        {"class_name": "Person", "confidence": 0.9, "bbox_xyxy": [9, 9, 18, 18]},
        {"class_name": "Person", "confidence": 0.9, "bbox_xyxy": [18, 18, 27, 27]},
        {"class_name": "NO-Mask", "confidence": 0.8, "bbox_xyxy": [0, 0, 5, 5]},
    ])
    out = pipe._run_ppe(None, people(1))
    assert out["person_count"] == 1
    assert out["violation_count"] == 1
    assert out["grade"] == "critical"


def test_violation_memory_bridges_a_gap(tmp_path):
    """A class seen within the window counts and draws even when the frame
    missed it; past the window it drops. Flicker smoothing, DMS-style."""
    pipe = make_pipe(tmp_path)
    vest = {"class_name": "NO-Safety Vest", "confidence": 0.2,
            "bbox_xyxy": [0, 0, 10, 10]}
    pipe.ppe_detector = FakePPE(dets=[vest])
    out = pipe._run_ppe(None, people(1), capture_ts=100.0)
    assert out["violation_count"] == 1
    # next analysis sees nothing, but the class is remembered
    pipe.ppe_detector = FakePPE(dets=[])
    out = pipe._run_ppe(None, people(1), capture_ts=102.0)
    assert out["violation_count"] == 1
    assert [b["class_name"] for b in pipe._ppe_boxes_for_draw()] == ["NO-Safety Vest"]
    # past the window it drops
    out = pipe._run_ppe(None, people(1), capture_ts=110.0)
    assert out["violation_count"] == 0
    assert pipe._ppe_boxes_for_draw() == []


def test_memory_never_double_counts(tmp_path):
    pipe = make_pipe(tmp_path)
    pipe._run_ppe(None, people(1), capture_ts=100.0)
    # model sees NO-Hardhat again while memory holds it: still one
    out = pipe._run_ppe(None, people(1), capture_ts=101.0)
    assert out["violation_count"] == 1


def test_timeless_calls_skip_memory(tmp_path):
    pipe = make_pipe(tmp_path)
    pipe._run_ppe(None, people(1), capture_ts=100.0)
    assert pipe._ppe_memory
    # no timestamp: live-only grading, memory untouched and unused
    pipe.ppe_detector = FakePPE(dets=[])
    out = pipe._run_ppe(None, people(1))
    assert out["violation_count"] == 0
    assert pipe._ppe_memory


def test_replay_clears_violation_memory(tmp_path):
    pipe = make_pipe(tmp_path)
    pipe._run_ppe(None, people(1), capture_ts=100.0)
    assert pipe._ppe_memory
    pipe._reset_for_replay()
    assert pipe._ppe_memory == {}


def test_ppe_is_grade_only(tmp_path):
    assert PPE_EMISSION_DEFERRED is True
    pipe = make_pipe(tmp_path)
    assert getattr(pipe, "ppe_emitter", None) is None  # no emit path exists
    pipe._run_ppe(None, people(1))
    assert pipe.state()["ppe"]["stats"]["violation_count"] == 1


def test_disabled_ppe_never_runs(tmp_path):
    pipe = Pipeline(Database(str(tmp_path / "ppe-off.db")))
    assert pipe.ppe_enabled is False
    assert pipe.ppe_detector is None
    assert pipe._maybe_run_ppe(None, people(), 100.0) is None
    assert pipe._run_side_paths(None, people(), 100, 100, 100.0) is None
    assert pipe.state()["ppe"] == {"enabled": False, "deferred": True}


def test_config_ppe_defaults_off_1hz():
    import os

    from config import load_config

    settings = load_config(os.path.join(os.path.dirname(__file__), os.pardir, "config.yaml"))
    assert settings["ppe"]["enabled"] is False
    assert settings["ppe"]["analysis_fps"] == pytest.approx(1.0)
    # tuned pass (Oct 2026): catches the flickering NO-Hardhat frames
    assert settings["ppe"]["confidence"] == pytest.approx(0.15)
    assert settings["ppe"]["imgsz"] == 960


def test_ppe_stats_carry_per_class_counts_and_fall_count(tmp_path):
    """Aggregate violation counts per type (no per-worker attribution)."""
    pipe = make_pipe(tmp_path)
    pipe.ppe_detector = FakePPE(dets=[
        {"class_name": "NO-Hardhat", "confidence": 0.8, "bbox_xyxy": [0, 0, 10, 10]},
        {"class_name": "NO-Mask", "confidence": 0.7, "bbox_xyxy": [0, 0, 10, 10]},
        {"class_name": "NO-Hardhat", "confidence": 0.6, "bbox_xyxy": [0, 0, 10, 10]},
        {"class_name": "Fall-Detected", "confidence": 0.9, "bbox_xyxy": [0, 0, 10, 10]},
        {"class_name": "Hardhat", "confidence": 0.9, "bbox_xyxy": [0, 0, 10, 10]},
    ])
    out = pipe._run_ppe(None, people(2))
    assert out["violation_count"] == 3
    assert out["violations_by_class"] == [["NO-Hardhat", 2], ["NO-Mask", 1]]
    assert out["fall_count"] == 1  # reported, never graded
    assert out["other_count"] == 1  # compliant gear, not a violation
    assert out["person_count"] == 2  # reused from the tracker
    assert out["grade"] == "critical"  # 3 violations / 2 people
    stats = pipe.state()["ppe"]["stats"]
    assert stats["violations_by_class"] == [["NO-Hardhat", 2], ["NO-Mask", 1]]
    assert stats["fall_count"] == 1
