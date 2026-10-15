"""Tests for Phase 2 fire/smoke alerts (SQLite table, routes, progress).

Zone rows, counts and history are covered elsewhere and must not move;
these tests pin the fire/smoke side only: migration on old and fresh DBs,
CRUD + ack, route shapes + 404s + path hygiene, gate progress semantics,
and burned-in box colours.
"""

import json
import sqlite3

import pytest
from fastapi.testclient import TestClient

import api.server as server
from api.server import app as fastapi_app
from database.db import Database
from detector.fire_smoke_gate import FireSmokeGate
from pipeline import Pipeline


@pytest.fixture()
def db(tmp_path):
    return Database(str(tmp_path / "fire.db"))


@pytest.fixture()
def client(tmp_path, monkeypatch):
    db_path = str(tmp_path / "fire-ui.db")
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


# --- migration ------------------------------------------------------------


def test_fresh_db_has_fire_table(db):
    assert make_alert(db) == 1
    assert db.count_fire_alerts() == 1


def test_existing_db_without_fire_table_is_upgraded(tmp_path):
    """A database from before Phase 2 (events + settings only) gains the
    table on open, keeping its rows."""
    path = str(tmp_path / "old.db")
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE events (id INTEGER PRIMARY KEY AUTOINCREMENT,"
                 " worker_id INTEGER NOT NULL, machine_id TEXT NOT NULL,"
                 " event_type TEXT NOT NULL, dwell_seconds REAL NOT NULL DEFAULT 0)")
    conn.execute("INSERT INTO events (worker_id, machine_id, event_type)"
                 " VALUES (1, 'm', 'ZONE_ENTRY')")
    conn.execute("CREATE TABLE settings (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    conn.commit()
    conn.close()

    db = Database(path)
    assert db.count() == 1  # old rows survive
    assert db.count_fire_alerts() == 0
    assert make_alert(db) == 1
    # reopening is a no-op, not a second migration
    assert Database(path).count_fire_alerts() == 1


# --- store -----------------------------------------------------------------


def test_insert_list_ack_roundtrip(db):
    assert db.list_fire_alerts() == []
    first = make_alert(db)
    second = make_alert(db, alert_type="smoke", severity="YELLOW", confidence=0.5)
    rows = db.list_fire_alerts()
    assert [r["id"] for r in rows] == [second, first]  # newest first
    assert rows[0]["alert_type"] == "smoke"
    assert rows[0]["bbox_xyxy"] == [1.0, 2.0, 30.0, 40.0]
    assert rows[0]["acknowledged"] is False
    assert db.get_fire_alert(999) is None
    assert db.count_fire_alerts(acknowledged=False) == 2

    assert db.acknowledge_fire_alert(first, 1700000001.0) is True
    assert db.acknowledge_fire_alert(999, 1700000001.0) is False
    assert db.get_fire_alert(first)["acknowledged"] is True
    assert db.count_fire_alerts(acknowledged=False) == 1
    assert db.count_fire_alerts(acknowledged=True) == 1
    assert [r["id"] for r in db.list_fire_alerts(acknowledged=True)] == [first]


def test_zone_counts_untouched_by_fire_rows(db):
    make_alert(db)
    assert db.count() == 0
    assert db.list_events() == []


# --- routes -----------------------------------------------------------------


def test_fire_alert_routes_list_ack_and_404(client):
    test_client, db_path = client
    assert test_client.get("/fire-alerts").json() == {
        "alerts": [], "total": 0, "unacked": 0, "acked": 0,
    }
    db = Database(db_path)
    first = make_alert(db)
    make_alert(db, alert_type="smoke")

    body = test_client.get("/fire-alerts").json()
    assert body["total"] == 2 and body["unacked"] == 2 and body["acked"] == 0
    row = body["alerts"][0]
    assert row["alert_type"] == "smoke" and row["has_snapshot"] is False
    assert row["reported_at"].endswith("+00:00")

    acked = test_client.post(f"/fire-alerts/{first}/ack").json()["alert"]
    assert acked["acknowledged"] is True and acked["acknowledged_at"] is not None
    # repeat ack succeeds
    assert test_client.post(f"/fire-alerts/{first}/ack").json()["alert"]["acknowledged"] is True
    assert test_client.get("/fire-alerts").json()["unacked"] == 1

    assert test_client.post("/fire-alerts/999/ack").status_code == 404
    assert test_client.get("/fire-alerts/999/snapshot").status_code == 404


def test_fire_snapshot_serves_file_and_rejects_escapes(client, tmp_path, monkeypatch):
    import snapshots

    test_client, db_path = client
    snap_dir = tmp_path / "fire-snaps"
    snap_dir.mkdir()
    monkeypatch.setattr(snapshots, "fire_snapshot_dir", lambda: str(snap_dir))

    db = Database(db_path)
    alert_id = make_alert(db)
    (snap_dir / f"{alert_id}.jpg").write_bytes(b"\xff\xd8fakejpeg")
    db.set_fire_snapshot_path(alert_id, f"{alert_id}.jpg")

    response = test_client.get(f"/fire-alerts/{alert_id}/snapshot")
    assert response.status_code == 200
    assert response.content == b"\xff\xd8fakejpeg"

    # a stored value pointing outside the folder can never be served
    evil = make_alert(db)
    db.set_fire_snapshot_path(evil, "../../evil.jpg")
    assert test_client.get(f"/fire-alerts/{evil}/snapshot").status_code == 404
    db.set_fire_snapshot_path(evil, "/etc/passwd")
    assert test_client.get(f"/fire-alerts/{evil}/snapshot").status_code == 404


# --- progress / confirming ----------------------------------------------------


def test_gate_progress_counts_until_crossing_then_goes_quiet():
    gate = FireSmokeGate()
    gate.set_camera_fps("c", 2.0)  # fire: 3 samples, smoke: 5
    assert gate.progress("c") == {"fire": None, "smoke": None}
    assert gate.update("c", ["fire", "smoke"]) == []
    assert gate.progress("c") == {"fire": {"samples": 1, "required": 3},
                                  "smoke": {"samples": 1, "required": 5}}
    assert gate.update("c", ["fire", "smoke"]) == []
    assert gate.update("c", ["fire", "smoke"]) == ["fire"]
    assert gate.progress("c") == {"fire": None, "smoke": {"samples": 3, "required": 5}}
    assert gate.update("c", []) == []
    assert gate.progress("c") == {"fire": None, "smoke": None}


def test_pipeline_progress_none_when_off_and_state_carries_confirming(tmp_path):
    pipe = Pipeline(Database(str(tmp_path / "prog.db")))
    assert pipe.fire_smoke_progress() is None
    assert pipe.state()["fire_smoke"]["enabled"] is False


def test_pipeline_progress_flows_into_state(tmp_path, monkeypatch):
    import numpy as np

    from detector.fire_smoke_gate import FireSmokeGate as Gate

    monkeypatch.setenv("TRINETRA_FIRE_SNAPSHOT_DIR", str(tmp_path / "fire-snaps"))
    pipe = Pipeline(Database(str(tmp_path / "prog2.db")))
    frame = np.zeros((48, 64, 3), dtype=np.uint8)

    class FakeDetector:
        def predict(self, frame, confidence_threshold=None):
            return [{"class_name": "Fire", "confidence": 0.9,
                     "bbox_xyxy": [5, 5, 20, 20], "track_id": None}]

    class FakeEmitter:
        def emit(self, payload):
            return "sent"

    pipe.fire_smoke = FakeDetector()
    pipe.fire_smoke_gate = Gate(fire_seconds=1.5, smoke_seconds=2.5, default_fps=2.0)
    pipe.fire_smoke_camera_id = "m"
    pipe.fire_smoke_gate.set_camera_fps("m", 2.0)
    pipe.fire_smoke_analysis_fps = 2.0
    pipe.fire_smoke_confidence = 0.35
    pipe.fire_smoke_emitter = FakeEmitter()

    assert pipe.fire_smoke_progress() == {"fire": None, "smoke": None}
    pipe._maybe_run_fire_smoke(frame, 100.0)
    pipe._maybe_run_fire_smoke(frame, 100.5)
    assert pipe.fire_smoke_progress() == {"fire": {"samples": 2, "required": 3},
                                          "smoke": None}
    assert pipe.state()["fire_smoke"]["confirming"] == {
        "fire": {"samples": 2, "required": 3}, "smoke": None}
    pipe._maybe_run_fire_smoke(frame, 101.0)  # crosses
    assert pipe.fire_smoke_progress() == {"fire": None, "smoke": None}
    # a row and a snapshot were recorded for the crossing
    rows = pipe.db.list_fire_alerts()
    assert len(rows) == 1 and rows[0]["alert_type"] == "fire"
    assert rows[0]["snapshot_path"] and rows[0]["snapshot_path"].endswith(".jpg")


# --- boxes ---------------------------------------------------------------------


def test_fire_and_smoke_boxes_differ_from_person_zone_colours():
    import numpy as np

    from visualize import (ALERT_ZONE_COLOR, COLORS, FIRE_BOX_COLOR, SMOKE_BOX_COLOR,
                           ZONE_COLOR, draw_fire_box, draw_smoke_box)
    from zone.danger_zone import ALERT, INSIDE, OUTSIDE

    taken = {COLORS[OUTSIDE], COLORS[INSIDE], COLORS[ALERT], ZONE_COLOR, ALERT_ZONE_COLOR}
    assert FIRE_BOX_COLOR not in taken
    assert SMOKE_BOX_COLOR not in taken
    assert FIRE_BOX_COLOR != SMOKE_BOX_COLOR

    frame = np.zeros((60, 80, 3), dtype=np.uint8)
    draw_fire_box(frame, (10, 10, 30, 30), "FIRE 0.90")
    assert int(frame.max()) > 0  # something was painted
    before = frame.copy()
    draw_smoke_box(frame, (40, 10, 70, 30))
    assert not (frame == before).all()  # second box paints too
    draw_fire_box(frame, (5, 5, 1, 1))  # degenerate: no-op, never raises
    draw_fire_box(frame, None)  # noqa - unusable: no-op, never raises


def test_ppe_box_is_magenta_and_never_an_alert_colour():
    import numpy as np

    from visualize import (ALERT_ZONE_COLOR, COLORS, FIRE_BOX_COLOR, PPE_BOX_COLOR,
                           SMOKE_BOX_COLOR, ZONE_COLOR, draw_ppe_box)
    from zone.danger_zone import ALERT, INSIDE, OUTSIDE

    taken = {COLORS[OUTSIDE], COLORS[INSIDE], COLORS[ALERT], ZONE_COLOR, ALERT_ZONE_COLOR,
             FIRE_BOX_COLOR, SMOKE_BOX_COLOR}
    assert PPE_BOX_COLOR not in taken

    frame = np.zeros((80, 200, 3), dtype=np.uint8)
    draw_ppe_box(frame, (10, 10, 60, 60), "NO HARDHAT 0.71")
    # top-edge border pixel wears the magenta, deep interior stays background
    assert (frame[10, 40] == np.array(PPE_BOX_COLOR)).all()
    assert (frame[50, 40] == np.zeros(3, dtype=np.uint8)).all()
    draw_ppe_box(frame, (5, 5, 1, 1))  # degenerate: no-op, never raises
    draw_ppe_box(frame, None)  # noqa - unusable: no-op, never raises


def test_box_contained_thresholds():
    from pipeline import _box_contained

    assert _box_contained([0, 0, 10, 10], [[0, 0, 10, 10]], 0.5) is True
    # a quarter inside stays kept
    assert _box_contained([0, 0, 10, 10], [[5, 5, 20, 20]], 0.5) is False
    assert _box_contained([0, 0, 10, 10], [], 0.5) is False
    assert _box_contained(None, [[0, 0, 10, 10]], 0.5) is False
    assert _box_contained([5, 5, 1, 1], [[0, 0, 10, 10]], 0.5) is False


def test_drop_keeps_smoke_and_uncovered_fire(tmp_path):
    pipe = Pipeline(Database(str(tmp_path / "drop.db")))
    pipe._last_people_boxes = [[0, 0, 100, 100]]
    dets = [
        {"vision_alert_type": "fire", "confidence": 0.9, "bbox_xyxy": [10, 10, 20, 20]},
        {"vision_alert_type": "fire", "confidence": 0.9, "bbox_xyxy": [200, 200, 400, 400]},
        {"vision_alert_type": "smoke", "confidence": 0.8, "bbox_xyxy": [10, 10, 20, 20]},
    ]
    kept = pipe._drop_person_contained_fire(dets)
    assert [d["bbox_xyxy"] for d in kept] == [[200, 200, 400, 400], [10, 10, 20, 20]]
    assert pipe.fire_smoke_stats["suppressed"] == 1
    # nobody tracked: everything is kept, smoke is never filtered
    pipe._last_people_boxes = []
    assert pipe._drop_person_contained_fire(dets) == dets


def test_boxes_drawn_only_when_enabled(tmp_path):
    pipe = Pipeline(Database(str(tmp_path / "draw.db")))
    assert pipe._fire_boxes_for_draw() == []
    assert pipe._ppe_boxes_for_draw() == []
