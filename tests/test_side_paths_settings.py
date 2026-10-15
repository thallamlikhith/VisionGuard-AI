"""Tests for the side detection paths settings plumbing (Phase 1).

Stored DB values win over config.yaml (mirroring confidence); /config
persists them and applies them live to a running camera; /start forwards
the merged values; the stopped /state carries the same keys so the UI
renders before the first start.
"""

import pytest
from fastapi.testclient import TestClient

import api.server as server
from api.server import app as fastapi_app
from database.db import Database


@pytest.fixture()
def client(tmp_path, monkeypatch):
    db_path = str(tmp_path / "paths.db")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "machine:\n  id: excavator_01\n  name: Excavator 01\n"
        "danger_zone:\n  points: [[10, 10], [60, 10], [60, 60]]\n"
        "rules:\n  mode: TIMER\n  dwell_seconds: 5\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(server, "DB_PATH", db_path)
    monkeypatch.setattr(server, "CONFIG_PATH", str(config_path))
    monkeypatch.setattr(server, "SNAPSHOT_PATH", str(tmp_path / "workers.json"))
    monkeypatch.setattr(server, "_pipeline", None)
    return TestClient(fastapi_app), db_path


def test_config_reports_paths_off_by_default(client):
    test_client, _ = client
    body = test_client.get("/config").json()
    assert body["fire_smoke"]["enabled"] is False
    assert body["fire_smoke"]["analysis_fps"] == pytest.approx(2.0)
    assert body["crowd"]["enabled"] is False
    assert body["crowd"]["source"] == "reuse"
    assert body["ppe"]["enabled"] is False
    assert body["ppe"]["analysis_fps"] == pytest.approx(1.0)


def test_config_yaml_values_used_without_stored(client, tmp_path):
    test_client, db_path = client
    Database(db_path).set_setting("fire_smoke_enabled", "1")
    Database(db_path).set_setting("fire_smoke_analysis_fps", "3")
    Database(db_path).set_setting("ppe_enabled", "1")
    body = test_client.get("/config").json()
    assert body["fire_smoke"]["enabled"] is True
    assert body["fire_smoke"]["analysis_fps"] == pytest.approx(3.0)
    assert body["ppe"]["enabled"] is True
    # untouched paths keep config.yaml values
    assert body["crowd"]["enabled"] is False


def test_garbage_stored_values_fall_back_to_config(client):
    test_client, db_path = client
    Database(db_path).set_setting("fire_smoke_enabled", "yes")
    Database(db_path).set_setting("fire_smoke_analysis_fps", "-2")
    Database(db_path).set_setting("crowd_analysis_fps", "not-a-number")
    body = test_client.get("/config").json()
    assert body["fire_smoke"]["enabled"] is False
    assert body["fire_smoke"]["analysis_fps"] == pytest.approx(2.0)
    assert body["crowd"]["analysis_fps"] == pytest.approx(2.0)


def test_saving_paths_persists_and_applies_live(client, monkeypatch):
    test_client, db_path = client

    class FakePipeline:
        active = True
        source = "0"
        frame_size = (640, 480)

        def __init__(self):
            self.applied = {}

        def apply_settings(self, **kwargs):
            self.applied.update(kwargs)
            return []

        def state(self):
            return {}

    fake = FakePipeline()
    monkeypatch.setattr(server, "_pipeline", fake)

    response = test_client.post("/config", json={
        "fire_smoke": {"enabled": True, "analysis_fps": 3},
        "ppe": {"enabled": True},
    })
    assert response.status_code == 200
    assert fake.applied["fire_smoke_enabled"] is True
    assert fake.applied["fire_smoke_analysis_fps"] == pytest.approx(3.0)
    assert fake.applied["ppe_enabled"] is True
    # absent blocks arrive as None, so the running camera keeps them
    # (the same "not sent keeps current value" contract as confidence)
    assert fake.applied["crowd_enabled"] is None
    assert fake.applied["ppe_analysis_fps"] is None

    stored = Database(db_path).all_settings()
    assert stored["fire_smoke_enabled"] == "1"
    assert stored["fire_smoke_analysis_fps"] == "3.0"
    assert stored["ppe_enabled"] == "1"
    assert test_client.get("/config").json()["fire_smoke"]["enabled"] is True


def test_disabling_a_path_applies_live(client, monkeypatch):
    test_client, _ = client

    class FakePipeline:
        active = True
        source = "0"
        frame_size = (640, 480)

        def __init__(self):
            self.applied = {}

        def apply_settings(self, **kwargs):
            self.applied.update(kwargs)
            return []

        def state(self):
            return {}

    fake = FakePipeline()
    monkeypatch.setattr(server, "_pipeline", fake)
    test_client.post("/config", json={"crowd": {"enabled": False}})
    assert fake.applied["crowd_enabled"] is False


def test_start_forwards_stored_paths_to_the_pipeline(client, monkeypatch):
    test_client, db_path = client
    Database(db_path).set_setting("fire_smoke_enabled", "1")
    Database(db_path).set_setting("ppe_analysis_fps", "1.5")

    captured = {}

    class FakePipeline:
        def stop(self):
            return True

        def start(self, **kwargs):
            captured.update(kwargs)

        def state(self):
            return {}

    monkeypatch.setattr(server, "_pipeline", FakePipeline())
    test_client.post("/start", json={})
    assert captured["fire_smoke_enabled"] is True
    assert captured["ppe_analysis_fps"] == pytest.approx(1.5)
    assert captured["crowd_enabled"] is False


def test_stopped_state_carries_side_keys_and_deferred_flag(client):
    test_client, _ = client
    body = test_client.get("/state").json()
    assert body["running"] is False
    assert body["fire_smoke"]["enabled"] is False
    assert body["crowd"]["enabled"] is False
    assert body["crowd"]["source"] == "reuse"
    assert body["ppe"]["enabled"] is False
    assert body["ppe"]["deferred"] is True


def test_invalid_fps_is_rejected(client):
    test_client, _ = client
    assert test_client.post("/config", json={"ppe": {"analysis_fps": -1}}).status_code == 422
    assert test_client.post("/config", json={"ppe": {"analysis_fps": 99}}).status_code == 422
    assert test_client.post("/config", json={"fire_smoke": {"analysis_fps": 0}}).status_code == 422


def test_hazard_defaults_on_and_fire_smoke_split_follows_legacy(client):
    test_client, db_path = client
    body = test_client.get("/config").json()
    assert body["hazard"] == {"enabled": True}
    assert body["fire_smoke"]["fire_enabled"] is False
    assert body["fire_smoke"]["smoke_enabled"] is False

    Database(db_path).set_setting("fire_smoke_enabled", "1")
    body = test_client.get("/config").json()
    assert body["fire_smoke"]["fire_enabled"] is True
    assert body["fire_smoke"]["smoke_enabled"] is True

    # an explicit per-type value narrows the legacy switch, not the reverse
    Database(db_path).set_setting("fire_enabled", "0")
    body = test_client.get("/config").json()
    assert body["fire_smoke"]["fire_enabled"] is False
    assert body["fire_smoke"]["smoke_enabled"] is True

    Database(db_path).set_setting("hazard_enabled", "0")
    assert test_client.get("/config").json()["hazard"] == {"enabled": False}
    Database(db_path).set_setting("hazard_enabled", "maybe")
    assert test_client.get("/config").json()["hazard"] == {"enabled": True}


def test_saving_hazard_and_split_persists_and_applies_live(client, monkeypatch):
    test_client, db_path = client

    class FakePipeline:
        active = True
        source = "0"
        frame_size = (640, 480)

        def __init__(self):
            self.applied = {}

        def apply_settings(self, **kwargs):
            self.applied.update(kwargs)
            return []

        def state(self):
            return {}

    fake = FakePipeline()
    monkeypatch.setattr(server, "_pipeline", fake)
    response = test_client.post("/config", json={
        "hazard": {"enabled": False},
        "fire": {"enabled": True},
        "smoke": {"enabled": False},
    })
    assert response.status_code == 200
    assert fake.applied["hazard_enabled"] is False
    assert fake.applied["fire_enabled"] is True
    assert fake.applied["smoke_enabled"] is False
    stored = Database(db_path).all_settings()
    assert stored["hazard_enabled"] == "0"
    assert stored["fire_enabled"] == "1"
    assert stored["smoke_enabled"] == "0"
    body = test_client.get("/config").json()
    assert body["hazard"] == {"enabled": False}
    assert body["fire_smoke"]["fire_enabled"] is True
    assert body["fire_smoke"]["smoke_enabled"] is False


def test_start_forwards_hazard_and_split_flags(client, monkeypatch):
    test_client, db_path = client
    Database(db_path).set_setting("hazard_enabled", "0")
    Database(db_path).set_setting("fire_enabled", "1")

    captured = {}

    class FakePipeline:
        def stop(self):
            return True

        def start(self, **kwargs):
            captured.update(kwargs)

        def state(self):
            return {}

    monkeypatch.setattr(server, "_pipeline", FakePipeline())
    test_client.post("/start", json={})
    assert captured["hazard_enabled"] is False
    assert captured["fire_enabled"] is True
    assert captured["smoke_enabled"] is False


def test_stopped_state_carries_hazard_and_split_keys(client):
    test_client, _ = client
    body = test_client.get("/state").json()
    assert body["running"] is False
    assert body["hazard"] == {"enabled": True}
    assert body["fire_smoke"]["fire_enabled"] is False
    assert body["fire_smoke"]["smoke_enabled"] is False
    assert body["status"] != "off"  # off is a running-pipeline status only
