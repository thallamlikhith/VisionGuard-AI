"""Config cleanup: selected behavior values live in config.yaml now.

Small and targeted: supplied values surface everywhere, missing keys fall
back to the exact current defaults, behavior with the default config is
unchanged.
"""

import pytest
from fastapi.testclient import TestClient

import api.server as server
from api.server import app as fastapi_app
from database.db import Database
from pipeline import Pipeline


@pytest.fixture()
def client(tmp_path, monkeypatch):
    db_path = str(tmp_path / "cleanup.db")
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
    return TestClient(fastapi_app), db_path, config_path


def _with_limits(config_path, text):
    current = config_path.read_text(encoding="utf-8")
    config_path.write_text(current + text, encoding="utf-8")


def test_loader_reads_supplied_values(tmp_path):
    from config import load_config

    path = tmp_path / "full.yaml"
    path.write_text(
        "machine:\n  id: m\n  name: M\n"
        "danger_zone:\n  points: []\n"
        "rules:\n  mode: TIMER\n  dwell_seconds: 5\n"
        "fire_smoke:\n  high_confidence: 0.9\n  person_containment: 0.7\n"
        "ppe:\n  memory_seconds: 9.0\n"
        "  grading:\n    medium_ratio: 0.3\n    high_ratio: 0.6\n    critical_ratio: 0.9\n"
        "limits:\n  max_upload_size_mb: 100\n  max_videos: 3\n  history_days: 7\n",
        encoding="utf-8",
    )
    settings = load_config(str(path))
    assert settings["fire_smoke"]["high_confidence"] == pytest.approx(0.9)
    assert settings["fire_smoke"]["person_containment"] == pytest.approx(0.7)
    assert settings["ppe"]["memory_seconds"] == pytest.approx(9.0)
    assert settings["ppe"]["grading"]["medium_ratio"] == pytest.approx(0.3)
    assert settings["ppe"]["grading"]["high_ratio"] == pytest.approx(0.6)
    assert settings["ppe"]["grading"]["critical_ratio"] == pytest.approx(0.9)
    assert settings["limits"] == {"max_upload_size_mb": 100, "max_videos": 3,
                                  "history_days": 7}


def test_loader_missing_keys_fall_back_to_current_values(client):
    test_client, _, _ = client
    body = test_client.get("/config").json()
    assert body["fire_smoke"]["high_confidence"] == pytest.approx(0.8)
    assert body["fire_smoke"]["person_containment"] == pytest.approx(0.5)
    assert body["ppe"]["memory_seconds"] == pytest.approx(4.0)
    assert body["ppe"]["grading"]["medium_ratio"] == pytest.approx(0.25)
    assert body["ppe"]["grading"]["high_ratio"] == pytest.approx(0.5)
    assert body["ppe"]["grading"]["critical_ratio"] == pytest.approx(1.0)
    assert body["limits"] == {"max_upload_size_mb": 500, "max_videos": 10,
                              "history_days": 14}


def test_start_forwards_the_new_values(client, monkeypatch):
    test_client, _, _ = client
    _with_limits(
        client[2],
        "fire_smoke:\n  high_confidence: 0.9\n  person_containment: 0.7\n"
        "ppe:\n  memory_seconds: 9.0\n"
        "  grading:\n    medium_ratio: 0.3\n    high_ratio: 0.6\n    critical_ratio: 0.9\n",
    )
    captured = {}

    class FakePipeline:
        active = False
        source = ""

        def stop(self):
            return True

        def start(self, **kwargs):
            captured.update(kwargs)

        def state(self):
            return {}

    monkeypatch.setattr(server, "_pipeline", FakePipeline())
    test_client.post("/start", json={})
    assert captured["fire_high_confidence"] == pytest.approx(0.9)
    assert captured["fire_person_containment"] == pytest.approx(0.7)
    assert captured["ppe_memory_seconds"] == pytest.approx(9.0)
    assert captured["ppe_medium_ratio"] == pytest.approx(0.3)
    assert captured["ppe_high_ratio"] == pytest.approx(0.6)
    assert captured["ppe_critical_ratio"] == pytest.approx(0.9)


def test_upload_honors_configured_size_limit(client):
    test_client, _, _ = client
    _with_limits(client[2], "limits:\n  max_upload_size_mb: 0\n")
    response = test_client.post(
        "/videos",
        files={"file": ("tiny.mp4", b"0" * 64, "video/mp4")},
    )
    assert response.status_code == 413


def test_daily_honors_configured_history_days(client):
    from datetime import datetime

    test_client, db_path = client[0], client[1]
    _with_limits(client[2], "limits:\n  history_days: 7\n")
    database = Database(db_path)
    now = datetime.now().replace(hour=12, minute=0, second=0, microsecond=0).timestamp()
    database.insert_event({"worker_id": 1, "machine_id": "excavator_01",
                           "event_type": "DWELL_ALERT", "entered_at": now,
                           "alert_at": now, "dwell_seconds": 6.0,
                           "zone_index": 0, "zone_name": "Kiln"})
    body = test_client.get("/events/daily").json()
    assert len(body["days"]) == 7


class _FakePPE:
    def __init__(self, dets):
        self.dets = dets

    def predict(self, frame, confidence_threshold=None):
        return [dict(d) for d in self.dets]


def _person():
    from detector.tracker import TrackedPerson

    return [TrackedPerson(worker_id=1, bbox=(0, 0, 40, 100), confidence=0.9)]


def test_configured_containment_is_used(tmp_path):
    pipe = Pipeline(Database(str(tmp_path / "contain.db")))
    pipe._last_people_boxes = [[0, 0, 100, 100]]
    # half of this box sits inside the person box: containment 0.5
    fire = {"vision_alert_type": "fire", "confidence": 0.9,
            "bbox_xyxy": [50, 10, 150, 60]}
    assert pipe._drop_person_contained_fire([fire]) == []
    pipe.fire_person_containment = 0.9
    assert pipe._drop_person_contained_fire([fire]) == [fire]


def test_configured_memory_window_is_used(tmp_path):
    pipe = Pipeline(Database(str(tmp_path / "mem.db")))
    pipe.ppe_enabled = True
    vest = {"class_name": "NO-Safety Vest", "confidence": 0.2,
            "bbox_xyxy": [0, 0, 10, 10]}
    pipe.ppe_detector = _FakePPE([vest])
    pipe._run_ppe(None, _person(), capture_ts=100.0)
    pipe.ppe_detector = _FakePPE([])
    pipe.ppe_memory_seconds = 100.0
    assert pipe._run_ppe(None, _person(), capture_ts=150.0)["violation_count"] == 1
    pipe.ppe_memory_seconds = 1.0
    assert pipe._run_ppe(None, _person(), capture_ts=200.0)["violation_count"] == 0


def test_configured_grading_ratios_are_used(tmp_path):
    pipe = Pipeline(Database(str(tmp_path / "grade.db")))
    pipe.ppe_enabled = True
    pipe.ppe_detector = _FakePPE([
        {"class_name": "NO-Hardhat", "confidence": 0.8, "bbox_xyxy": [0, 0, 10, 10]}])
    pipe.ppe_medium_ratio = 0.9
    pipe.ppe_high_ratio = 0.95
    pipe.ppe_critical_ratio = 1.5
    # ratio 1.0 would be critical by default; with a 1.5 critical line it is high
    out = pipe._run_ppe(None, _person(), capture_ts=100.0)
    assert out["grade"] == "high"
