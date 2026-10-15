"""Tests for Phase 3 false-alarm hard-negative export.

Covers the fire_alerts migration on Phase 2 databases, the clean-frame
store, the export files + JSON content, idempotent marking, 404/409 paths
and path hygiene. Zone rows and counts are untouched throughout.
"""

import json
import sqlite3

import pytest
from fastapi.testclient import TestClient

import api.server as server
from api.server import app as fastapi_app
from database.db import Database
from pipeline import Pipeline


@pytest.fixture()
def db(tmp_path):
    return Database(str(tmp_path / "fa.db"))


@pytest.fixture()
def client(tmp_path, monkeypatch):
    db_path = str(tmp_path / "fa-ui.db")
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


def make_alert(db, **over):
    base = dict(alert_type="fire", severity="ORANGE", confidence=0.76,
                bbox_xyxy=[1.0, 2.0, 30.0, 40.0], captured_at=1.0,
                reported_at=1700000000.0)
    base.update(over)
    return db.insert_fire_alert(**base)


# --- migration ---------------------------------------------------------------


def test_phase2_db_gains_new_columns_without_losing_rows(tmp_path):
    """A database whose fire_alerts table predates Phase 3 keeps its rows."""
    path = str(tmp_path / "phase2.db")
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE fire_alerts (id INTEGER PRIMARY KEY AUTOINCREMENT,"
        " alert_type TEXT NOT NULL, severity TEXT NOT NULL,"
        " confidence REAL NOT NULL DEFAULT 0, bbox_xyxy TEXT,"
        " captured_at REAL, reported_at REAL, snapshot_path TEXT,"
        " acknowledged INTEGER NOT NULL DEFAULT 0, acknowledged_at REAL)")
    conn.execute(
        "INSERT INTO fire_alerts (alert_type, severity, confidence,"
        " captured_at, reported_at) VALUES ('smoke', 'YELLOW', 0.5, 2.5, 1700000000.0)")
    conn.execute("CREATE TABLE settings (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    conn.commit()
    conn.close()

    db = Database(path)
    rows = db.list_fire_alerts()
    assert len(rows) == 1 and rows[0]["alert_type"] == "smoke"
    assert rows[0]["false_alarm"] is False
    assert rows[0]["false_alarm_at"] is None
    assert rows[0]["clean_snapshot_path"] is None
    assert db.mark_fire_false_alarm(rows[0]["id"], 1700000001.0) is True
    assert db.mark_fire_false_alarm(rows[0]["id"], 1700000002.0) is False
    marked = db.get_fire_alert(rows[0]["id"])
    assert marked["false_alarm"] is True
    assert marked["acknowledged"] is True  # marking acknowledges too


def test_mark_unknown_id_is_not_an_error(db):
    assert db.mark_fire_false_alarm(999, 1700000000.0) is False


# --- export helpers -------------------------------------------------------------


def test_export_writes_jpg_plus_json_with_exact_metadata(tmp_path, monkeypatch):
    import snapshots

    snap_dir = tmp_path / "snaps"
    snap_dir.mkdir()
    out_dir = tmp_path / "hn"
    monkeypatch.setattr(snapshots, "fire_snapshot_dir", lambda: str(snap_dir))
    monkeypatch.setattr(snapshots, "hard_negatives_dir", lambda: str(out_dir))
    from snapshots import export_hard_negative, write_fire_clean_snapshot

    assert write_fire_clean_snapshot(7, b"\xff\xd8clean") == "7_clean.jpg"
    assert (snap_dir / "7_clean.jpg").read_bytes() == b"\xff\xd8clean"

    metadata = {"alert_id": 7, "type": "smoke", "captured_at": 2.5,
                "reported_at": 1700000000.0, "box": [1.0, 2.0, 3.0, 4.0],
                "confidence": 0.87, "weights": "fire_smoke_v2.pt"}
    jpg, js = export_hard_negative("smoke", 1700000000.0, 7,
                                   str(snap_dir / "7_clean.jpg"), metadata)
    assert jpg.startswith("smoke_") and jpg.endswith(".jpg")
    assert js == jpg[: -len(".jpg")] + ".json"
    assert (out_dir / jpg).read_bytes() == b"\xff\xd8clean"
    assert json.loads((out_dir / js).read_text()) == metadata


def test_export_rejects_missing_source(tmp_path, monkeypatch):
    import snapshots
    from snapshots import HardNegativeError, export_hard_negative

    out_dir = tmp_path / "hn"
    monkeypatch.setattr(snapshots, "hard_negatives_dir", lambda: str(out_dir))
    with pytest.raises(HardNegativeError):
        export_hard_negative("fire", 1700000000.0, 1,
                             str(tmp_path / "nope.jpg"), {"a": 1})


# --- routes -----------------------------------------------------------------------


def test_false_alarm_exports_marks_and_acks_idempotently(client, tmp_path, monkeypatch):
    import snapshots

    test_client, db_path = client
    snap_dir = tmp_path / "snaps"
    snap_dir.mkdir()
    out_dir = tmp_path / "hn"
    monkeypatch.setattr(snapshots, "fire_snapshot_dir", lambda: str(snap_dir))
    monkeypatch.setattr(snapshots, "hard_negatives_dir", lambda: str(out_dir))

    db = Database(db_path)
    alert_id = make_alert(db)
    (snap_dir / f"{alert_id}_clean.jpg").write_bytes(b"\xff\xd8clean")
    db.set_fire_clean_snapshot_path(alert_id, f"{alert_id}_clean.jpg")

    body = test_client.post(f"/fire-alerts/{alert_id}/false-alarm").json()
    assert body["duplicate"] is False
    assert body["alert"]["false_alarm"] is True
    assert body["alert"]["acknowledged"] is True
    assert body["exported"]["image"].startswith("fire_")
    assert body["exported"]["metadata"].endswith(".json")
    exported = sorted(p.name for p in out_dir.iterdir())
    assert exported == sorted([body["exported"]["image"], body["exported"]["metadata"]])
    sidecar = json.loads((out_dir / body["exported"]["metadata"]).read_text())
    assert sidecar["alert_id"] == alert_id
    assert sidecar["type"] == "fire"
    assert sidecar["weights"] == "fire_smoke_v2.pt"
    assert sidecar["box"] == [1.0, 2.0, 30.0, 40.0]
    assert (out_dir / body["exported"]["image"]).read_bytes() == b"\xff\xd8clean"

    again = test_client.post(f"/fire-alerts/{alert_id}/false-alarm").json()
    assert again["duplicate"] is True and again["exported"] is None
    assert sorted(p.name for p in out_dir.iterdir()) == exported  # no new files

    assert test_client.post("/fire-alerts/999/false-alarm").status_code == 404


def test_false_alarm_without_clean_frame_is_409_not_500(client):
    test_client, db_path = client
    db = Database(db_path)
    alert_id = make_alert(db)  # no clean snapshot stored
    response = test_client.post(f"/fire-alerts/{alert_id}/false-alarm")
    assert response.status_code == 409
    assert "clean frame" in response.json()["detail"]
    assert db.get_fire_alert(alert_id)["false_alarm"] is False

    dangling = make_alert(db)
    db.set_fire_clean_snapshot_path(dangling, "99_clean.jpg")  # file never written
    assert test_client.post(f"/fire-alerts/{dangling}/false-alarm").status_code == 409


def test_false_alarm_rejects_escape_paths(client, tmp_path):
    test_client, db_path = client
    db = Database(db_path)
    alert_id = make_alert(db)
    db.set_fire_clean_snapshot_path(alert_id, "../../evil.jpg")
    response = test_client.post(f"/fire-alerts/{alert_id}/false-alarm")
    assert response.status_code == 409
    outside = tmp_path / "evil.jpg"
    assert not outside.exists()
    db.set_fire_clean_snapshot_path(alert_id, "/etc/passwd")
    assert test_client.post(f"/fire-alerts/{alert_id}/false-alarm").status_code == 409


# --- pipeline stores the clean frame ----------------------------------------------


def test_pipeline_stores_clean_frame_at_alert_time(tmp_path, monkeypatch):
    import numpy as np

    monkeypatch.setenv("TRINETRA_FIRE_SNAPSHOT_DIR", str(tmp_path / "snaps"))
    pipe = Pipeline(Database(str(tmp_path / "clean.db")))
    frame = np.zeros((48, 64, 3), dtype=np.uint8)

    class FakeDetector:
        def predict(self, frame, confidence_threshold=None):
            return [{"class_name": "Fire", "confidence": 0.9,
                     "bbox_xyxy": [5, 5, 20, 20], "track_id": None}]

    class FakeEmitter:
        def emit(self, payload):
            return "sent"

    from detector.fire_smoke_gate import FireSmokeGate as Gate
    pipe.fire_smoke = FakeDetector()
    pipe.fire_smoke_gate = Gate(fire_seconds=1.5, smoke_seconds=2.5, default_fps=2.0)
    pipe.fire_smoke_camera_id = "m"
    pipe.fire_smoke_gate.set_camera_fps("m", 2.0)
    pipe.fire_smoke_analysis_fps = 2.0
    pipe.fire_smoke_confidence = 0.35
    pipe.fire_smoke_emitter = FakeEmitter()

    pipe._maybe_run_fire_smoke(frame, 100.0)
    pipe._maybe_run_fire_smoke(frame, 100.5)
    pipe._maybe_run_fire_smoke(frame, 101.0)  # crosses
    rows = pipe.db.list_fire_alerts()
    assert len(rows) == 1
    assert rows[0]["clean_snapshot_path"] == f"{rows[0]['id']}_clean.jpg"
    assert (tmp_path / "snaps" / rows[0]["clean_snapshot_path"]).exists()
