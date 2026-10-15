"""Tests for the merged History log (Stage D): source labels + /history.

Zone routes (/events, /events/export.csv, /events/daily shapes) are pinned
by their own suites and must not move; these tests pin the merged side:
//history merges zone + fire rows newest-first with one total, the CSV
//gains a Source column, and fire rows carry the label stored at alert time.
"""

import sqlite3
import time

import pytest
from fastapi.testclient import TestClient

import api.server as server
from api.server import app as fastapi_app
from database.db import Database, DatabaseError


@pytest.fixture()
def db(tmp_path):
    return Database(str(tmp_path / "history.db"))


@pytest.fixture()
def client(tmp_path, monkeypatch):
    db_path = str(tmp_path / "history-ui.db")
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


def zone_event(db, **over):
    base = dict(worker_id=3, machine_id="m", event_type="DWELL_ALERT",
                entered_at=1700000000.0, alert_at=1700000005.0,
                dwell_seconds=6.0, zone_index=0, zone_name="Kiln")
    base.update(over)
    return db.insert_event(base)


def fire_alert(db, **over):
    base = dict(alert_type="fire", severity="ORANGE", confidence=0.76,
                bbox_xyxy=[1.0, 2.0, 30.0, 40.0], captured_at=1.0,
                reported_at=1700000010.0, source_label="Kiln Cam")
    base.update(over)
    return db.insert_fire_alert(**base)


# --- source_label storage -------------------------------------------------


def test_insert_without_label_stores_null(db):
    row_id = db.insert_fire_alert(alert_type="smoke", severity="YELLOW",
                                  confidence=0.5, bbox_xyxy=None,
                                  captured_at=None, reported_at=1700000000.0)
    assert db.get_fire_alert(row_id)["source_label"] is None


def test_insert_with_label_round_trips(db):
    row_id = fire_alert(db, source_label="Kiln Cam")
    assert db.get_fire_alert(row_id)["source_label"] == "Kiln Cam"


def test_old_db_without_source_label_column_is_upgraded(tmp_path):
    """A fire_alerts table from before the label column gains it on open,
    keeping its rows; new rows then carry labels."""
    path = str(tmp_path / "old-fire.db")
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE events (id INTEGER PRIMARY KEY AUTOINCREMENT,"
                 " worker_id INTEGER NOT NULL, machine_id TEXT NOT NULL,"
                 " event_type TEXT NOT NULL, dwell_seconds REAL NOT NULL DEFAULT 0)")
    conn.execute("CREATE TABLE settings (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    conn.execute("CREATE TABLE fire_alerts (id INTEGER PRIMARY KEY AUTOINCREMENT,"
                 " alert_type TEXT NOT NULL, severity TEXT NOT NULL,"
                 " confidence REAL NOT NULL DEFAULT 0, bbox_xyxy TEXT,"
                 " captured_at REAL, reported_at REAL, snapshot_path TEXT,"
                 " clean_snapshot_path TEXT, acknowledged INTEGER NOT NULL DEFAULT 0,"
                 " acknowledged_at REAL, false_alarm INTEGER NOT NULL DEFAULT 0,"
                 " false_alarm_at REAL)")
    conn.execute("INSERT INTO fire_alerts (alert_type, severity, confidence, reported_at)"
                 " VALUES ('fire', 'ORANGE', 0.8, 1700000000.0)")
    conn.commit()
    conn.close()

    db = Database(path)
    old = db.get_fire_alert(1)
    assert old["source_label"] is None  # upgraded row reads back clean
    assert db.insert_fire_alert(alert_type="smoke", severity="YELLOW",
                                confidence=0.5, bbox_xyxy=None, captured_at=None,
                                reported_at=1700000001.0,
                                source_label="Kiln Cam") == 2
    assert Database(path).get_fire_alert(2)["source_label"] == "Kiln Cam"


def test_pipeline_record_stores_machine_name(tmp_path, monkeypatch):
    """The alert-time label is the machine the running source watches."""
    import numpy as np

    from pipeline import Pipeline

    monkeypatch.setenv("TRINETRA_FIRE_SNAPSHOT_DIR", str(tmp_path / "fire-snaps"))
    monkeypatch.setenv("TRINETRA_HARD_NEGATIVES_DIR", str(tmp_path / "hardneg"))
    db = Database(str(tmp_path / "pipe.db"))
    pipe = Pipeline(db)
    pipe.machine = {"id": "excavator_01", "name": "Kiln Cam"}
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    ids = pipe._record_fire_alerts(
        frame, ["fire"],
        {"fire": {"confidence": 0.9, "bbox_xyxy": [1.0, 1.0, 10.0, 10.0]}},
        capture_ts=1700000000.0,
    )
    assert len(ids) == 1
    assert db.get_fire_alert(ids[0])["source_label"] == "Kiln Cam"


# --- merged listing -------------------------------------------------------


def test_merge_orders_newest_first_with_kinds(db):
    zone_event(db)  # alert_at 1700000005
    fire_alert(db)  # reported 1700000010 -> first
    rows = db.list_history()
    assert [(r["kind"], r["id"]) for r in rows] == [("fire", 1), ("zone", 1)]
    assert rows[0]["source_label"] == "Kiln Cam"
    assert rows[1]["event_type"] == "DWELL_ALERT"


def test_count_matches_paged_lists(db):
    for n in range(5):
        zone_event(db, alert_at=1700000000.0 + n)
        fire_alert(db, alert_type="smoke", reported_at=1700000000.5 + n)
    assert db.count_history() == 10
    page1 = db.list_history(limit=4, offset=0)
    page2 = db.list_history(limit=4, offset=4)
    page3 = db.list_history(limit=4, offset=8)
    assert len(page1) + len(page2) + len(page3) == db.count_history() == 10


def test_type_filter_splits_tables(db):
    zone_event(db)
    fire_alert(db)
    fire_only = db.list_history(event_types=["FIRE"])
    assert [r["kind"] for r in fire_only] == ["fire"]
    assert db.count_history(event_types=["FIRE"]) == 1
    zone_only = db.list_history(event_types=["DWELL_ALERT"])
    assert [r["kind"] for r in zone_only] == ["zone"]
    assert db.count_history(event_types=["DWELL_ALERT"]) == 1
    both = db.list_history(event_types=["DWELL_ALERT", "FIRE"])
    assert sorted(r["kind"] for r in both) == ["fire", "zone"]


def test_zone_filter_matches_source_label(db):
    zone_event(db, zone_name="Kiln", zone_index=0)
    fire_alert(db, source_label="Kiln")
    fire_alert(db, alert_type="smoke", source_label="Crusher", reported_at=1700000020.0)
    rows = db.list_history(zone="Kiln")
    assert sorted((r["kind"], r.get("source_label") or r.get("zone_name")) for r in rows) == [
        ("fire", "Kiln"), ("zone", "Kiln"),
    ]
    assert db.count_history(zone="Kiln") == 2
    # a numeric index never matches a fire row
    index_rows = db.list_history(zone="0")
    assert [r["kind"] for r in index_rows] == ["zone"]


def test_worker_filter_excludes_fire(db):
    zone_event(db, worker_id=3)
    fire_alert(db)
    rows = db.list_history(worker_id=3)
    assert [r["kind"] for r in rows] == ["zone"]
    assert db.list_history(worker_id=9) == []
    assert db.count_history(worker_id=3) == 1


def test_acknowledged_filter_applies_to_both(db):
    zone_event(db)
    fire_alert(db)
    db.acknowledge_event(1, 1700000100.0)
    assert [r["kind"] for r in db.list_history(acknowledged=True)] == ["zone"]
    assert [r["kind"] for r in db.list_history(acknowledged=False)] == ["fire"]
    assert db.count_history(acknowledged=True) == 1


def test_false_alarm_filter(db):
    zone_event(db)
    flagged = fire_alert(db)
    plain = fire_alert(db, alert_type="smoke", reported_at=1700000020.0)
    db.mark_fire_false_alarm(flagged, 1700000100.0)
    only_flagged = db.list_history(false_alarm=True)
    assert [(r["kind"], r["id"]) for r in only_flagged] == [("fire", flagged)]
    rest = db.list_history(false_alarm=False)
    assert sorted([(r["kind"], r["id"]) for r in rest]) == [("fire", plain), ("zone", 1)]
    assert db.count_history(false_alarm=True) == 1


def test_dates_bound_both_tables(db):
    zone_event(db, alert_at=1700000005.0, entered_at=1700000000.0)
    fire_alert(db, reported_at=1700000010.0)
    # the stamps are 2023-11-15 ~03:4x in plant-local (IST) time
    assert db.count_history(date_from="2023-11-15", date_to="2023-11-15") == 2
    assert db.count_history(date_from="2023-11-16") == 0
    with pytest.raises(DatabaseError):
        db.list_history(date_from="not-a-date")


# --- routes ---------------------------------------------------------------


def test_history_route_shape_and_total(client):
    http, db_path = client
    database = Database(db_path)
    zone_event(database)
    fire_alert(database)
    body = http.get("/history").json()
    assert body["total"] == 2
    assert body["count"] == 2
    assert [r["kind"] for r in body["rows"]] == ["fire", "zone"]
    assert body["rows"][0]["source_label"] == "Kiln Cam"
    assert body["rows"][0]["alert_type"] == "fire"


def test_history_route_filters_and_bad_date(client):
    http, db_path = client
    database = Database(db_path)
    zone_event(database)
    fire_alert(database, source_label="Kiln")
    assert http.get("/history", params={"event_type": "FIRE"}).json()["total"] == 1
    assert http.get("/history", params={"zone": "Kiln"}).json()["total"] == 2
    assert http.get("/history", params={"worker_id": 3}).json()["total"] == 1
    assert http.get("/history", params={"date_from": "nope"}).status_code == 422


def test_history_csv_has_source_column(client):
    http, db_path = client
    database = Database(db_path)
    zone_event(database)
    fire_alert(database)
    response = http.get("/history/export.csv")
    assert response.status_code == 200
    lines = response.text.strip().splitlines()
    assert lines[0] == ("Time,Worker,Zone,Source,Event,Dwell (s),People inside,"
                        "Acknowledged,Acknowledged at")
    assert len(lines) == 3  # header + one row per table
    assert "Kiln Cam" in response.text
    assert "Fire" in response.text


def test_zone_csv_unchanged_without_source(client):
    """The zone-only export keeps its exact columns: no Source here."""
    http, db_path = client
    database = Database(db_path)
    zone_event(database)
    fire_alert(database)
    response = http.get("/events/export.csv")
    assert response.status_code == 200
    lines = response.text.strip().splitlines()
    assert lines[0] == ("Time,Worker,Zone,Event,Dwell (s),People inside,"
                        "Acknowledged,Acknowledged at")
    assert len(lines) == 2  # the fire row is not a zone row
    assert "Kiln Cam" not in response.text


def test_daily_gains_fire_and_smoke_series(client):
    http, db_path = client
    database = Database(db_path)
    now = time.time()
    database.insert_event({"worker_id": 3, "machine_id": "m", "event_type": "DWELL_ALERT",
                           "entered_at": now - 5, "alert_at": now,
                           "dwell_seconds": 6.0, "zone_index": 0, "zone_name": "Kiln"})
    database.insert_fire_alert(alert_type="fire", severity="ORANGE", confidence=0.8,
                               bbox_xyxy=None, captured_at=None, reported_at=now,
                               source_label="Kiln Cam")
    body = http.get("/events/daily", params={"days": 1}).json()
    assert body["days"][0]["fire"] == 1
    assert body["days"][0]["smoke"] == 0
    assert body["days"][0]["dwell"] == 1
