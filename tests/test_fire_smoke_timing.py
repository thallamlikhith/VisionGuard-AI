"""Fire/smoke time-paced analysis (Stage 1).

Video frames arrive at 25 fps; analysis runs at 2.0 Hz paced by WALL
(monotonic) time — skips are decided by elapsed seconds, never by frame
count — so the gate's seconds-to-frames conversion stays true:
fire 1.5 s -> ceil(1.5*2) = 3 analyzed samples, smoke 2.5 s ->
ceil(2.5*2) = 5 samples.

Fencepost note: the first analyzed sample (t=0, presence just appeared)
already counts as sample 1, so with a 0.5 s cadence fire crosses at ~1.0 s
of continuous presence (inside the 1.5 s window) and smoke at ~2.0 s
(inside the 2.5 s window). The test pins the sample counts (3 / 5) and the
time bounds, not exact equality with the window.

No model, no video, no clock: fake detector + stepped fake monotonic time.
"""

import pytest

from database.db import Database
from detector.fire_smoke_gate import FireSmokeGate
from pipeline import Pipeline

VIDEO_FPS = 25.0
ANALYSIS_FPS = 2.0
T0 = 100.0


class FakeDetector:
    """Constant fire (or smoke) on every analyzed frame."""

    def __init__(self, kind="fire", confidence=0.9):
        self.kind = kind
        self.confidence = confidence
        self.calls = 0

    def predict(self, frame, confidence_threshold=None):
        self.calls += 1
        name = "Fire" if self.kind == "fire" else "Smoke"
        return [{
            "class_name": name,
            "confidence": self.confidence,
            "bbox_xyxy": [5, 5, 20, 20],
            "track_id": None,
        }]


class FakeEmitter:
    def __init__(self):
        self.payloads = []

    def emit(self, payload):
        self.payloads.append(payload)
        return "sent"


def make_pipe(tmp_path, kind="fire"):
    pipe = Pipeline(Database(str(tmp_path / f"timing-{kind}.db")))
    pipe.fire_smoke = FakeDetector(kind)
    pipe.fire_smoke_gate = FireSmokeGate(
        fire_seconds=1.5, smoke_seconds=2.5, default_fps=ANALYSIS_FPS
    )
    pipe.fire_smoke_camera_id = "cam-1"
    pipe.fire_smoke_gate.set_camera_fps("cam-1", ANALYSIS_FPS)
    pipe.fire_smoke_analysis_fps = ANALYSIS_FPS
    pipe.fire_smoke_confidence = 0.35
    pipe.fire_smoke_emitter = FakeEmitter()
    return pipe


def drive(pipe, seconds=6.0):
    """Step fake 25 fps frames; return (time, k, analyzed_count) of first cross."""
    first = (None, None, None)
    n = int(seconds * VIDEO_FPS)
    for k in range(n):
        before = pipe.fire_smoke_stats["analyzed"]
        crossed = pipe._maybe_run_fire_smoke(None, T0 + k / VIDEO_FPS)
        if crossed and first[0] is None:
            first = (T0 + k / VIDEO_FPS, k, before + 1)
    return first


def test_gate_windows_are_three_and_five_samples():
    gate = FireSmokeGate(fire_seconds=1.5, smoke_seconds=2.5, default_fps=2.0)
    assert gate.required_frames("any", "fire") == 3  # ceil(1.5 * 2.0)
    assert gate.required_frames("any", "smoke") == 5  # ceil(2.5 * 2.0)


def test_fire_alerts_after_three_time_paced_samples(tmp_path):
    pipe = make_pipe(tmp_path, "fire")
    cross_t, cross_k, analyzed_at_cross = drive(pipe, seconds=6.0)
    assert cross_t is not None, "fire never crossed in 6 s of presence"
    elapsed = cross_t - T0
    assert 1.0 <= elapsed < 1.5, f"fire crossed at {elapsed:.2f}s, want within [1.0, 1.5)"
    assert analyzed_at_cross == 3, "fire must cross on the 3rd analyzed sample"
    # ~12 analyses in 6 s at 2 Hz (first immediate, then every ~0.5 s)
    assert 11 <= pipe.fire_smoke_stats["analyzed"] <= 13
    # one emission per continuous episode, however long it runs
    assert len(pipe.fire_smoke_emitter.payloads) == 1
    payload = pipe.fire_smoke_emitter.payloads[0]
    assert set(payload) == {"camera", "vision_alert_type", "severity", "risk_level",
                            "confidence", "bbox_xyxy", "track_id", "reported_at"}
    assert payload["vision_alert_type"] == "fire"
    assert (payload["severity"], payload["risk_level"]) == ("RED", "high")
    assert "dedup_key" not in payload


def test_smoke_alerts_after_five_time_paced_samples(tmp_path):
    pipe = make_pipe(tmp_path, "smoke")
    cross_t, cross_k, analyzed_at_cross = drive(pipe, seconds=6.0)
    assert cross_t is not None, "smoke never crossed in 6 s of presence"
    elapsed = cross_t - T0
    assert 2.0 <= elapsed < 2.5, f"smoke crossed at {elapsed:.2f}s, want within [2.0, 2.5)"
    assert analyzed_at_cross == 5, "smoke must cross on the 5th analyzed sample"
    assert len(pipe.fire_smoke_emitter.payloads) == 1
    payload = pipe.fire_smoke_emitter.payloads[0]
    assert payload["vision_alert_type"] == "smoke"
    assert (payload["severity"], payload["risk_level"]) == ("ORANGE", "medium")


class EmptyDetector:
    """Sees nothing on every analyzed frame."""

    def predict(self, frame, confidence_threshold=None):
        return []


def test_negative_frame_rearms_after_crossing(tmp_path):
    pipe = make_pipe(tmp_path, "fire")
    cross_t, _, _ = drive(pipe, seconds=2.0)
    assert cross_t is not None
    assert len(pipe.fire_smoke_emitter.payloads) == 1
    fire = pipe.fire_smoke
    # one analyzed clean frame re-arms the gate
    pipe.fire_smoke = EmptyDetector()
    pipe._fire_smoke_last_run = T0  # force the next call to analyze
    assert pipe._maybe_run_fire_smoke(None, T0 + 100.0) == []
    assert len(pipe.fire_smoke_emitter.payloads) == 1
    # presence again needs a full fresh window: 3 more analyzed samples
    pipe.fire_smoke = fire
    calls_before = fire.calls
    for i in range(1, 4):
        pipe._fire_smoke_last_run = T0 + 100.0 + (i - 1) * 0.6
        crossed = pipe._maybe_run_fire_smoke(None, T0 + 100.0 + i * 0.6)
        assert crossed == ([] if i < 3 else ["fire"]), f"sample {i}: {crossed}"
    assert fire.calls == calls_before + 3
    assert len(pipe.fire_smoke_emitter.payloads) == 2


def test_disabled_side_path_never_runs(tmp_path):
    pipe = Pipeline(Database(str(tmp_path / "off.db")))
    assert pipe.fire_smoke is None
    assert pipe.fire_smoke_gate is None
    assert pipe._maybe_run_fire_smoke(None, T0) is None
    assert pipe._maybe_run_fire_smoke(None, T0 + 5.0) is None
    assert pipe.state()["fire_smoke"]["enabled"] is False


def test_loop_replay_resumes_analyses_and_refires_alerts(tmp_path):
    """A replayed clip is a new pass: stamps reset, analyses resume at once,
    and a still-present fire re-confirms and re-fires (one row per loop)."""
    pipe = make_pipe(tmp_path, "fire")
    cross_t, _, _ = drive(pipe, seconds=2.0)
    assert cross_t is not None
    assert len(pipe.fire_smoke_emitter.payloads) == 1
    analyzed_loop1 = pipe.fire_smoke_stats["analyzed"]
    # stale stamps as the end of an 8 s loop leaves them
    pipe._fire_smoke_last_run = T0 + 7.9
    pipe._ppe_last_run = T0 + 7.9
    pipe._crowd_last_run = T0 + 7.9
    # without the reset, video time 0..8 of the next loop is never due
    assert pipe._maybe_run_fire_smoke(None, 0.4) is None
    # loop restart
    pipe._reset_for_replay()
    assert pipe._fire_smoke_last_run is None
    assert pipe._ppe_last_run is None
    assert pipe._crowd_last_run is None
    assert pipe.fire_smoke_gate.progress("cam-1") == {"fire": None, "smoke": None}
    # analyses resume at once on the new pass ...
    pipe._maybe_run_fire_smoke(None, 0.0)
    assert pipe.fire_smoke_stats["analyzed"] == analyzed_loop1 + 1
    # ... and the gate re-crosses after a fresh window: one row per loop
    t = 0.0
    while t < 2.0 and len(pipe.fire_smoke_emitter.payloads) < 2:
        t += 0.5
        pipe._maybe_run_fire_smoke(None, t)
    assert len(pipe.fire_smoke_emitter.payloads) == 2
