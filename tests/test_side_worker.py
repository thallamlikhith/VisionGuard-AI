"""Background side-analysis worker tests (all heavy paths).

Deterministic: fake detectors, plain capture timestamps, manual worker
steps (no sleeps) except the lifecycle test, which polls a real thread
with a generous timeout. Proves the worker contract:

- no thread unless a heavy path is on; the loop publishes nothing then
- slot of 1: rapid publishes keep only the newest frame (no queue buildup)
- one heavy analysis per grabbed frame, priority fire > crowd > PPE
- results stamped with capture ts + frame index, never completion time
- fire needs its full video-time window however completions bunch up;
  no emission burst after a long stall
"""

import threading
import time

import numpy as np
import pytest

from database.db import Database
from detector.fire_smoke_gate import FireSmokeGate
from pipeline import Pipeline

W, H = 64, 48


def blank():
    return np.zeros((H, W, 3), dtype=np.uint8)


class FakeFire:
    def __init__(self, kind="fire", conf=0.9):
        self.kind, self.conf, self.calls = kind, conf, 0

    def predict(self, frame, confidence_threshold=None):
        self.calls += 1
        name = "Fire" if self.kind == "fire" else "Smoke"
        return [{"class_name": name, "confidence": self.conf,
                 "bbox_xyxy": [5, 5, 20, 20], "track_id": None}]


class FakeCrowd:
    def __init__(self):
        self.calls = 0

    def predict(self, frame, confidence_threshold=None):
        self.calls += 1
        return [{"class_name": "person", "confidence": 0.8,
                 "bbox_xyxy": [0, 0, 20, 20], "track_id": None}]


class FakePPE:
    def __init__(self):
        self.calls = 0

    def predict(self, frame, confidence_threshold=None):
        self.calls += 1
        return [{"class_name": "NO-Hardhat", "confidence": 0.8,
                 "bbox_xyxy": [0, 0, 10, 10], "track_id": None}]


class FakeEmitter:
    def __init__(self):
        self.payloads = []

    def emit(self, payload):
        self.payloads.append(payload)
        return "sent"


def arm_fire(pipe, kind="fire", fps=2.0):
    pipe.fire_smoke = FakeFire(kind)
    pipe.fire_smoke_gate = FireSmokeGate(fire_seconds=1.5, smoke_seconds=2.5,
                                         default_fps=fps)
    pipe.fire_smoke_camera_id = "m"
    pipe.fire_smoke_gate.set_camera_fps("m", fps)
    pipe.fire_smoke_analysis_fps = fps
    pipe.fire_smoke_confidence = 0.35
    pipe.fire_smoke_emitter = FakeEmitter()


def arm_all(pipe):
    arm_fire(pipe)
    pipe.crowd_enabled = True
    pipe.crowd_source = "model"
    pipe.crowd_detector = FakeCrowd()
    pipe.crowd_analysis_fps = 2.0
    pipe.ppe_enabled = True
    pipe.ppe_detector = FakePPE()
    pipe.ppe_analysis_fps = 1.0


def grab_fire(pipe):
    """Deterministically run one fire-worker iteration (no thread)."""
    with pipe._side_slot_lock:
        job = pipe._fire_slot
        pipe._fire_slot = None
    if job is None:
        return None
    frame, capture_ts, frame_index = job
    return pipe._maybe_run_fire_smoke(frame, capture_ts, frame_index=frame_index)


def grab_slow(pipe):
    """Deterministically run one slow-worker iteration (no thread)."""
    with pipe._side_slot_lock:
        job = pipe._slow_slot
        pipe._slow_slot = None
    if job is None:
        return None
    frame, boxes, width, height, frame_index, capture_ts = job
    people = [{"class_name": "person", "bbox_xyxy": box} for box in boxes]
    return pipe._run_slow_paths(frame, people, width, height, capture_ts,
                                frame_index=frame_index)


def test_no_thread_and_no_publish_when_all_off(tmp_path):
    pipe = Pipeline(Database(str(tmp_path / "w-off.db")))
    assert pipe._fire_thread is None
    assert pipe._slow_thread is None
    assert pipe._publish_side_frame(blank(), [], W, H, 0, 100.0) is False
    pipe._ensure_side_worker()
    assert pipe._fire_thread is None
    assert pipe._slow_thread is None
    assert pipe.state()["fire_smoke"]["enabled"] is False


def test_slot_keeps_only_the_newest_frame(tmp_path):

    pipe = Pipeline(Database(str(tmp_path / "w-slot.db")))
    arm_fire(pipe)
    for k in range(150):  # 6 s of video published with no worker step
        assert pipe._publish_side_frame(blank(), [], W, H, k, 100.0 + k / 25.0) is True
    with pipe._side_slot_lock:
        slot = pipe._fire_slot
        assert pipe._slow_slot is None  # fire-only: slow slot untouched
    assert slot[1] == pytest.approx(100.0 + 149 / 25.0)
    assert slot[2] == 149
    grab_fire(pipe)  # exactly one analysis for the whole backlog, no burst
    assert pipe.fire_smoke.calls == 1


def test_fire_worker_never_waits_behind_slow_paths(tmp_path):
    """Split workers: fire runs per grab even while crowd/PPE are also due."""
    pipe = Pipeline(Database(str(tmp_path / "w-prio.db")))
    arm_all(pipe)
    pipe._publish_side_frame(blank(), [], W, H, 0, 100.0)
    grab_fire(pipe)
    assert pipe.fire_smoke.calls == 1  # fire does not wait for slow paths
    grab_slow(pipe)
    assert (pipe.crowd_detector.calls, pipe.ppe_detector.calls) == (1, 0)
    pipe._publish_side_frame(blank(), [], W, H, 1, 100.04)
    grab_fire(pipe)  # fire not due yet (0.04 s < 0.5 s)
    assert pipe.fire_smoke.calls == 1
    grab_slow(pipe)  # crowd not due -> PPE runs: stagger preserved on slow
    assert (pipe.crowd_detector.calls, pipe.ppe_detector.calls) == (1, 1)


def test_fire_needs_full_video_window_despite_bunched_completions(tmp_path):
    """Five grabs completing back-to-back still need 3 capture-spaced samples."""
    pipe = Pipeline(Database(str(tmp_path / "w-fire.db")))
    arm_fire(pipe)
    crossed_at = None
    for k, t in enumerate([100.0, 100.5, 101.0, 101.5, 102.0]):
        pipe._publish_side_frame(blank(), [], W, H, k, t)
        grab_fire(pipe)
        if pipe.fire_smoke_emitter.payloads and crossed_at is None:
            crossed_at = t
    assert pipe.fire_smoke.calls == 5
    assert crossed_at == pytest.approx(101.0)  # 3rd sample, ~1.0 s of video
    assert crossed_at - 100.0 >= 1.0  # never earlier than the window allows
    assert len(pipe.fire_smoke_emitter.payloads) == 1
    stats = pipe.fire_smoke_stats
    assert stats["last_capture_ts"] == pytest.approx(102.0)  # stamped, not wall
    assert stats["last_frame_index"] == 4


def test_smoke_needs_five_samples_over_video_time(tmp_path):
    pipe = Pipeline(Database(str(tmp_path / "w-smoke.db")))
    arm_fire(pipe, kind="smoke")
    crossed_at = None
    for k in range(7):
        t = 100.0 + k * 0.5
        pipe._publish_side_frame(blank(), [], W, H, k, t)
        grab_fire(pipe)
        if pipe.fire_smoke_emitter.payloads and crossed_at is None:
            crossed_at = t
    assert pipe.fire_smoke.calls == 7
    assert crossed_at == pytest.approx(102.0)  # 5th sample, ~2.0 s of video
    assert len(pipe.fire_smoke_emitter.payloads) == 1


def test_no_burst_after_long_ppe_stall(tmp_path):
    """Worker busy elsewhere 0.5 -> 6.0 s: resume analyses, single emission."""
    pipe = Pipeline(Database(str(tmp_path / "w-stall.db")))
    arm_fire(pipe)
    for k, t in ((0, 100.0), (1, 100.5)):  # count 2/3, then the worker stalls
        pipe._publish_side_frame(blank(), [], W, H, k, t)
        grab_fire(pipe)
    assert pipe.fire_smoke.calls == 2
    assert pipe.fire_smoke_emitter.payloads == []
    for k, t in ((2, 106.0), (3, 106.5), (4, 107.0)):  # resume after the stall
        pipe._publish_side_frame(blank(), [], W, H, k, t)
        grab_fire(pipe)
    assert pipe.fire_smoke.calls == 5  # one analysis per grab, no backlog burst
    assert len(pipe.fire_smoke_emitter.payloads) == 1  # one episode, one emission


def test_results_carry_frame_index_and_capture_ts(tmp_path):
    pipe = Pipeline(Database(str(tmp_path / "w-stamps.db")))
    arm_all(pipe)
    pipe._publish_side_frame(blank(), [], W, H, 42, 107.5)
    grab_fire(pipe)  # fire wins its own worker
    assert pipe.fire_smoke_stats["last_frame_index"] == 42
    assert pipe.fire_smoke_stats["last_capture_ts"] == pytest.approx(107.5)
    assert pipe.state()["fire_smoke"]["last_capture_ts"] == pytest.approx(107.5)
    pipe.fire_smoke = None  # slow worker's turn on the next grab
    pipe._publish_side_frame(blank(), [{"bbox_xyxy": [0, 0, 9, 9]}], W, H, 43, 107.6)
    grab_slow(pipe)
    assert pipe._crowd_last_frame == 43
    assert pipe._crowd_last_capture == pytest.approx(107.6)
    pipe.crowd_enabled = False  # ppe's turn next grab
    pipe._publish_side_frame(blank(), [{"bbox_xyxy": [0, 0, 9, 9]}], W, H, 44, 107.7)
    grab_slow(pipe)
    assert pipe.ppe_stats["last_frame_index"] == 44
    assert pipe.ppe_stats["last_capture_ts"] == pytest.approx(107.7)
    assert pipe.state()["ppe"]["last_capture_ts"] == pytest.approx(107.7)


def test_worker_thread_lifecycle(tmp_path):
    pipe = Pipeline(Database(str(tmp_path / "w-life.db")))
    arm_fire(pipe)
    before = {t.name for t in threading.enumerate()}
    pipe._start_fire_worker()
    assert pipe._fire_thread is not None and pipe._fire_thread.is_alive()
    pipe._publish_side_frame(blank(), [], W, H, 0, 200.0)
    deadline = time.monotonic() + 5.0
    while pipe.fire_smoke.calls == 0 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert pipe.fire_smoke.calls >= 1, "worker never picked up the frame"
    assert pipe._stop_side_worker() is True
    assert pipe._fire_thread is None
    pipe._stop_side_worker()  # idempotent
    leftovers = {t.name for t in threading.enumerate() if t.is_alive()} - before
    assert "side-fire" not in leftovers
    assert "side-slow" not in leftovers
    # restart works
    pipe._start_fire_worker()
    assert pipe._fire_thread.is_alive()
    assert pipe._stop_side_worker() is True


def test_per_path_threads_start_and_stop_independently(tmp_path):
    pipe = Pipeline(Database(str(tmp_path / "w-split.db")))
    arm_all(pipe)
    pipe._ensure_side_worker()
    assert pipe._fire_thread.is_alive()
    assert pipe._slow_thread.is_alive()
    # disabling fire stops only the fire thread
    pipe.fire_smoke = None
    pipe._ensure_side_worker()
    assert pipe._fire_thread is None
    assert pipe._slow_thread.is_alive()
    # disabling the slow paths stops only the slow thread
    pipe.crowd_enabled = False
    pipe.ppe_enabled = False
    pipe._ensure_side_worker()
    assert pipe._slow_thread is None
    leftovers = {t.name for t in threading.enumerate() if t.is_alive()}
    assert "side-fire" not in leftovers
    assert "side-slow" not in leftovers


def test_long_ppe_run_never_delays_fire(tmp_path):
    """PPE blocked mid-analysis: fire samples keep flowing on schedule."""
    release = threading.Event()

    class BlockingPPE(FakePPE):
        def __init__(self):
            super().__init__()
            self.entered = 0

        def predict(self, frame, confidence_threshold=None):
            self.entered += 1
            assert release.wait(timeout=10.0), "ppe gate never released"
            return FakePPE.predict(self, frame, confidence_threshold)

    pipe = Pipeline(Database(str(tmp_path / "w-block.db")))
    arm_fire(pipe)
    pipe.ppe_enabled = True
    pipe.ppe_detector = BlockingPPE()
    pipe._ensure_side_worker()
    assert pipe._fire_thread.is_alive() and pipe._slow_thread.is_alive()
    try:
        # slow worker grabs a frame and blocks inside PPE...
        pipe._publish_side_frame(blank(), [], W, H, 0, 300.0)
        deadline = time.monotonic() + 5.0
        while pipe.ppe_detector.entered == 0 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert pipe.ppe_detector.entered >= 1, "slow worker never started PPE"
        # ...while fire keeps sampling on its own thread and cadence: publish
        # the next frame only after the previous one was analyzed, so every
        # capture is sampled exactly once (slot replacement can't skip any).
        deadline = time.monotonic() + 5.0
        while pipe.fire_smoke.calls < 1 and time.monotonic() < deadline:
            time.sleep(0.01)
        for k in range(1, 4):
            pipe._publish_side_frame(blank(), [], W, H, k, 300.0 + k * 0.5)
            want = pipe.fire_smoke.calls + 1
            deadline = time.monotonic() + 5.0
            while pipe.fire_smoke.calls < want and time.monotonic() < deadline:
                time.sleep(0.01)
        assert pipe.fire_smoke.calls == 4, (
            f"fire stalled behind blocked PPE: {pipe.fire_smoke.calls} calls"
        )
        assert len(pipe.fire_smoke_emitter.payloads) == 1  # crossed mid-block
    finally:
        release.set()
        assert pipe._stop_side_worker() is True
    assert pipe._fire_thread is None and pipe._slow_thread is None


def test_full_loop_lifecycle_with_worker(tmp_path, monkeypatch):
    """Real loop + real thread: worker lives during the run, dies on stop."""
    import cv2

    monkeypatch.setattr("pipeline.PersonTracker", _StubTracker)
    clip = str(tmp_path / "life.mp4")
    video = cv2.VideoWriter(clip, cv2.VideoWriter_fourcc(*"mp4v"), 10, (W, H))
    assert video.isOpened()
    for _ in range(5):
        video.write(np.zeros((H, W, 3), dtype=np.uint8))
    video.release()

    pipe = Pipeline(Database(str(tmp_path / "life.db")))
    try:
        pipe.start(source=clip, machine={"id": "m", "name": "m"}, points=[],
                   fire_smoke_enabled=True, realtime=False, loop=False)
        assert pipe._fire_thread is not None  # ensured on start (real detector)
        assert pipe._slow_thread is None  # nothing slow enabled
        deadline = time.monotonic() + 15.0
        while pipe.active and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not pipe.active
    finally:
        assert pipe.stop() is True
    assert pipe._fire_thread is None  # no leaked thread


class _StubTracker:
    def __init__(self, *_a, **_k):
        pass

    def track(self, frame, frame_index):
        return []


def test_published_slot_is_a_clean_pre_overlay_copy(tmp_path):
    """Drawing on the loop frame after publish must not reach the slot."""
    import cv2

    pipe = Pipeline(Database(str(tmp_path / "w-clean.db")))
    arm_fire(pipe)
    frame = blank()
    assert pipe._publish_side_frame(frame, [], W, H, 7, 100.0) is True
    # the loop draws its overlays afterwards: bright orange + red boxes
    cv2.rectangle(frame, (5, 5), (30, 30), (0, 140, 255), -1)  # orange (fire color)
    cv2.rectangle(frame, (35, 5), (60, 40), (0, 0, 255), -1)  # red (alert color)
    assert int(frame.max()) == 255  # sanity: the loop frame really is drawn on
    with pipe._side_slot_lock:
        slot_frame = pipe._fire_slot[0]
    assert slot_frame is not frame
    assert int(slot_frame.max()) == 0, "slot must hold the clean pre-overlay copy"


def test_worker_analyses_clean_frame_despite_late_overlay(tmp_path):
    """End to end: overlay drawn after publish never reaches the detector."""
    import cv2

    class RecordingFire(FakeFire):
        def __init__(self):
            super().__init__()
            self.seen_max = None

        def predict(self, frame, confidence_threshold=None):
            self.seen_max = int(frame.max())
            return super().predict(frame, confidence_threshold)

    pipe = Pipeline(Database(str(tmp_path / "w-cleanjob.db")))
    arm_fire(pipe)
    recorder = RecordingFire()
    pipe.fire_smoke = recorder
    frame = blank()
    pipe._publish_side_frame(frame, [], W, H, 3, 100.0)
    cv2.rectangle(frame, (0, 0), (W - 1, H - 1), (0, 140, 255), -1)
    with pipe._side_slot_lock:
        job = pipe._fire_slot
        pipe._fire_slot = None
    frame_j, capture_j, index_j = job
    pipe._maybe_run_fire_smoke(frame_j, capture_j, frame_index=index_j)
    assert recorder.calls == 1
    assert recorder.seen_max == 0, "worker must analyze the clean camera frame"
