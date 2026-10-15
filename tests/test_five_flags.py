"""All 32 combinations of the five detection-system flags (backend stage).

hazard x fire x smoke x crowd x ppe, each run through a real (short,
stubbed-detector) pipeline loop: what runs, what /state reports, what gets
drawn, and that unticked systems create no alerts and no DB rows. Crowd uses
the reuse source throughout (the model route is covered elsewhere); smoke
detections never appear (the fake emits fire only), which is exactly what
pins the fire/smoke type split.
"""

import itertools
import time

import cv2
import numpy as np
import pytest

from database.db import Database
from pipeline import Pipeline

WIDTH, HEIGHT = 64, 48
FRAMES = 30
FPS = 10.0
ZONE = [[0, 0], [WIDTH, 0], [WIDTH, HEIGHT], [0, HEIGHT]]


class Person:
    def __init__(self, worker_id=1, bbox=None, confidence=0.9):
        self.worker_id = worker_id
        self.bbox = bbox or (30, 0, WIDTH, HEIGHT)
        self.confidence = confidence


class StubTracker:
    """One person on the right half of every frame.

    Deliberately clear of the fake fire box on the left: a FIRE detection
    fully inside a tracked worker is suppressed as a hi-vis vest, so the
    person must stand aside for the flag matrix to exercise fire rows.
    """

    def __init__(self, *_args, **_kwargs):
        self.frames = 0

    def track(self, frame, frame_index):
        self.frames += 1
        return [Person()]


class FakeFire:
    """Constant fire (never smoke): pins the type split."""

    kinds = ("fire",)

    def __init__(self, *args, **kwargs):
        self.calls = 0

    def predict(self, frame, confidence_threshold=None):
        self.calls += 1
        return [{"class_name": "Fire", "confidence": 0.9,
                 "bbox_xyxy": [5, 5, 20, 20], "track_id": None}]


class FakeCrowd:
    def __init__(self, *args, **kwargs):
        self.calls = 0

    def predict(self, frame, confidence_threshold=None):
        self.calls += 1
        return [{"class_name": "person", "confidence": 0.8,
                 "bbox_xyxy": [0, 0, 20, 20], "track_id": None}]


class FakePPE:
    def __init__(self, *args, **kwargs):
        self.calls = 0

    def predict(self, frame, confidence_threshold=None):
        self.calls += 1
        return [{"class_name": "NO-Hardhat", "confidence": 0.8,
                 "bbox_xyxy": [0, 0, 10, 10], "track_id": None}]


class FakeEmitter:
    def emit(self, payload):
        return "sent"


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("clips") / "flags.mp4")
    video = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT))
    assert video.isOpened()
    for _ in range(FRAMES):
        video.write(np.full((HEIGHT, WIDTH, 3), 40, dtype=np.uint8))
    video.release()
    return path


@pytest.fixture()
def fakes(monkeypatch, tmp_path):
    # snapshot writers (zone + fire) point at tmp so loop runs never touch
    # the repo's data/ folder
    monkeypatch.setenv("TRINETRA_SNAPSHOT_DIR", str(tmp_path / "snaps"))
    monkeypatch.setenv("TRINETRA_FIRE_SNAPSHOT_DIR", str(tmp_path / "fire-snaps"))
    monkeypatch.setattr("pipeline.PersonTracker", StubTracker)
    monkeypatch.setattr("detector.fire_smoke_detector.FireSmokeDetector", FakeFire)
    monkeypatch.setattr("detector.crowd_detector.CrowdDetector", FakeCrowd)
    monkeypatch.setattr("detector.ppe_detector.PPEDetector", FakePPE)
    monkeypatch.setattr("services.vision_alerts.VisionAlertEmitter", FakeEmitter)
    calls = []
    monkeypatch.setattr("pipeline.draw_person_box",
                        lambda *a, **k: calls.append("person"))
    monkeypatch.setattr("pipeline.draw_ppe_box",
                        lambda *a, **k: calls.append("ppe"))
    monkeypatch.setattr("pipeline.draw_fire_box",
                        lambda *a, **k: calls.append("fire"))
    monkeypatch.setattr("pipeline.draw_smoke_box",
                        lambda *a, **k: calls.append("smoke"))
    return calls


def drain(pipe, timeout=5.0):
    """Wait until both worker slots are empty and quiet."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with pipe._side_slot_lock:
            pending = pipe._fire_slot is not None or pipe._slow_slot is not None
        if not pending:
            time.sleep(0.2)
            with pipe._side_slot_lock:
                if pipe._fire_slot is None and pipe._slow_slot is None:
                    return
        time.sleep(0.05)


def run_combo(tmp_path, clip, hazard, fire, smoke, crowd, ppe):
    db = Database(str(tmp_path / f"flags-{hazard}-{fire}-{smoke}-{crowd}-{ppe}.db"))
    # stored history predates the run: it must survive it untouched below
    db.insert_event({"worker_id": 1, "machine_id": "m", "event_type": "DWELL_ALERT",
                     "dwell_seconds": 5.0})
    db.insert_fire_alert(alert_type="smoke", severity="YELLOW", confidence=0.5,
                         bbox_xyxy=[1, 1, 2, 2], captured_at=0.5, reported_at=1700000000.0)
    pipe = Pipeline(db)
    pipe.start(
        source=clip,
        machine={"id": "m", "name": "m"},
        zones=[ZONE],
        allowed_people=0,
        hazard_enabled=hazard,
        fire_smoke_enabled=False,
        fire_enabled=fire,
        smoke_enabled=smoke,
        crowd_enabled=crowd,
        ppe_enabled=ppe,
        realtime=False,
        loop=False,
    )
    deadline = time.monotonic() + 20.0
    while pipe.active and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not pipe.active, "loop did not finish"
    drain(pipe)
    state = pipe.state()
    events = db.list_events()
    fire_rows = db.list_fire_alerts()
    pipe.stop()
    return pipe, state, events, fire_rows


COMBOS = list(itertools.product([False, True], repeat=5))


@pytest.mark.parametrize("hazard,fire,smoke,crowd,ppe", COMBOS)
def test_every_combination(tmp_path, clip, fakes, hazard, fire, smoke, crowd, ppe):
    calls = fakes
    pipe, state, events, fire_rows = run_combo(
        tmp_path, clip, hazard, fire, smoke, crowd, ppe)

    # --- what runs -------------------------------------------------------
    tracker_should_run = hazard or crowd or ppe  # reuse source always here
    assert (pipe.tracker.frames > 0) is tracker_should_run
    # person boxes are drawn from zone workers, so they follow the hazard
    # switch even while the tracker runs silently for crowd/PPE
    assert ("person" in calls) is hazard
    # fire model runs iff a fire type is ticked (smoke-only still loads it:
    # one shared pass, filtered downstream)
    assert (pipe.fire_smoke is not None) is (fire or smoke)
    # --- /state ----------------------------------------------------------
    if hazard:
        assert state["status"] in ("SAFE", "ALERT")
        assert state["message"] != "Hazard monitoring off"
    else:
        assert state["status"] == "off"
        assert state["message"] == "Hazard monitoring off"
        assert state["workers"] == []
        assert state["inside"] == 0
        assert state["capacity_alert"] is False
        assert state["worker_alert"] is False
    assert state["hazard"] == {"enabled": hazard}
    assert state["fire_smoke"]["fire_enabled"] is fire
    assert state["fire_smoke"]["smoke_enabled"] is smoke
    assert state["crowd"]["enabled"] is crowd
    assert state["ppe"]["enabled"] is ppe
    if crowd:
        assert state["crowd"]["metrics"]["person_count"] > 0
    if ppe:
        assert state["ppe"]["analyzed"] > 0
    # --- what gets drawn --------------------------------------------------
    assert ("fire" in calls) is fire  # fake emits fire only; smoke never drawn
    assert "smoke" not in calls
    # --- no alerts or rows from unticked systems ---------------------------
    # the single seeded DWELL row (worker 1) predates the run; only hazard
    # may add rows (capacity alerts on the full-frame zone)
    if hazard:
        assert len(events) >= 1
    else:
        assert len(events) == 1  # exactly the seeded row
    new_fire = [r for r in fire_rows if r["captured_at"] != 0.5]
    if fire:
        assert len(new_fire) >= 1
        assert {r["alert_type"] for r in new_fire} == {"fire"}
    else:
        assert new_fire == []
    # --- stored history survives, whatever is ticked -------------------------
    assert len(events) >= 1 and len(fire_rows) >= 1
    assert any(e["worker_id"] == 1 for e in events)
    assert any(r["captured_at"] == 0.5 for r in fire_rows)


def test_draw_order_person_ppe_fire_smoke(tmp_path, clip, fakes, monkeypatch):
    """Burned-in order on one frame is person, then PPE, then fire, then smoke."""
    calls = fakes

    class BothKinds(FakeFire):
        def predict(self, frame, confidence_threshold=None):
            self.calls += 1
            return [
                {"class_name": "Fire", "confidence": 0.9,
                 "bbox_xyxy": [5, 5, 20, 20], "track_id": None},
                {"class_name": "Smoke", "confidence": 0.8,
                 "bbox_xyxy": [25, 25, 40, 40], "track_id": None},
            ]

    monkeypatch.setattr("detector.fire_smoke_detector.FireSmokeDetector", BothKinds)
    db = Database(str(tmp_path / "order.db"))
    pipe = Pipeline(db)
    pipe.start(
        source=clip,
        machine={"id": "m", "name": "m"},
        zones=[ZONE],
        allowed_people=0,
        hazard_enabled=True,
        fire_smoke_enabled=False,
        fire_enabled=True,
        smoke_enabled=True,
        ppe_enabled=True,
        realtime=False,
        loop=False,
    )
    try:
        deadline = time.monotonic() + 20.0
        while pipe.active and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not pipe.active
    finally:
        pipe.stop()
    assert "person" in calls and "ppe" in calls
    assert "fire" in calls and "smoke" in calls
    # every layer trails the previous one, per frame: person, PPE, fire,
    # smoke. The first frames are still warming up (one heavy model per
    # frame, fire first), so the order is pinned past the first PPE box,
    # from where every frame draws all four.
    steady = calls[calls.index("ppe"):]
    for index, name in enumerate(steady):
        if index == 0:
            continue
        if name == "ppe":
            assert steady[index - 1] == "person"
        if name == "fire":
            assert steady[index - 1] == "ppe"
        if name == "smoke":
            assert steady[index - 1] == "fire"


def test_fire_on_a_worker_is_suppressed_as_a_vest(tmp_path, clip, fakes, monkeypatch):
    """A FIRE box fully inside a tracked worker never reaches the gate: no
    confirming, no rows, no burned-in boxes — but the counter records it."""
    calls = fakes

    class FullFramePerson:
        def __init__(self, *_args, **_kwargs):
            pass

        def track(self, frame, frame_index):
            return [Person(bbox=(0, 0, WIDTH, HEIGHT))]

    monkeypatch.setattr("pipeline.PersonTracker", FullFramePerson)
    db = Database(str(tmp_path / "suppress.db"))
    pipe = Pipeline(db)
    pipe.start(
        source=clip,
        machine={"id": "m", "name": "m"},
        zones=[ZONE],
        allowed_people=0,
        hazard_enabled=True,
        fire_smoke_enabled=False,
        fire_enabled=True,
        smoke_enabled=False,
        realtime=False,
        loop=False,
    )
    try:
        deadline = time.monotonic() + 20.0
        while pipe.active and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not pipe.active
    finally:
        pipe.stop()
    assert pipe.fire_smoke_stats["analyzed"] > 0
    assert pipe.fire_smoke_stats["emitted"] == 0
    assert pipe.fire_smoke_stats["suppressed"] > 0
    assert db.list_fire_alerts() == []
    assert "fire" not in calls


def test_zones_save_and_apply_while_hazard_off(tmp_path, monkeypatch):
    """Drawing zones never needs the hazard system running."""
    from fastapi.testclient import TestClient

    import api.server as server
    from api.server import app as fastapi_app

    db_path = str(tmp_path / "zones-off.db")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "machine:\n  id: excavator_01\n  name: Excavator 01\n"
        "danger_zone:\n  points: []\n"
        "rules:\n  mode: TIMER\n  dwell_seconds: 5\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(server, "DB_PATH", db_path)
    monkeypatch.setattr(server, "CONFIG_PATH", str(config_path))
    monkeypatch.setattr(server, "SNAPSHOT_PATH", str(tmp_path / "workers.json"))
    monkeypatch.setattr(server, "_pipeline", None)
    test_client = TestClient(fastapi_app)

    zone = [[5, 5], [60, 5], [60, 40], [5, 40]]
    body = test_client.post("/config", json={
        "hazard": {"enabled": False},
        "zones": [zone],
        "frame_width": 64,
        "frame_height": 48,
    }).json()
    assert body["saved"] is True
    assert body["zones"] == [zone]
    saved = test_client.get("/config").json()
    assert saved["zones"] == [zone]
    assert saved["hazard"] == {"enabled": False}

    class FakePipeline:
        active = True
        source = "0"
        frame_size = (64, 48)

        def __init__(self):
            self.applied = {}

        def apply_settings(self, **kwargs):
            self.applied.update(kwargs)
            return []

        def state(self):
            return {}

    fake = FakePipeline()
    monkeypatch.setattr(server, "_pipeline", fake)
    test_client.post("/config", json={
        "hazard": {"enabled": False},
        "zones": [zone],
        "frame_width": 64,
        "frame_height": 48,
    })
    assert fake.applied["zones"] == [zone]
    assert fake.applied["hazard_enabled"] is False
