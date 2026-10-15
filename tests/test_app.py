"""Tests for config loading, the database layer and the API."""

import os

import pytest
import yaml
from fastapi.testclient import TestClient

from api import server
from api.server import app as fastapi_app
from config import ConfigError, load_config
from database.db import Database, DatabaseError
from zone.danger_zone import CAPACITY_ALERT, CAPACITY_SAFE, DWELL_ALERT, ENTRY_ALERT, ZONE_ENTRY, ZONE_EXIT

BASE_CONFIG = {
    "machine": {"id": "excavator_01", "name": "Excavator 01"},
    "danger_zone": {"points": [[400, 200], [700, 200], [760, 500], [350, 500]]},
    "rules": {"mode": "TIMER", "dwell_seconds": 5},
}


def write_config(tmp_path, data):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return str(path)


# -- config ----------------------------------------------------------------

def test_load_config_reads_example(tmp_path):
    settings = load_config(write_config(tmp_path, BASE_CONFIG))
    assert settings["machine"]["name"] == "Excavator 01"
    assert settings["rules"]["mode"] == "TIMER"
    assert settings["danger_zone"]["points"][0] == (400, 200)
    assert settings["detector"]["model"] == "yolo11n.pt"


def test_an_empty_zone_is_valid(tmp_path):
    """A machine nobody has set up yet has to be able to start."""
    settings = load_config(write_config(tmp_path, {**BASE_CONFIG, "danger_zone": {"points": []}}))

    assert settings["danger_zone"]["points"] == []
    assert settings["danger_zone"]["anchor"] == "center"


def test_a_missing_zone_is_valid(tmp_path):
    data = {**BASE_CONFIG}
    del data["danger_zone"]["points"]
    settings = load_config(write_config(tmp_path, data))

    assert settings["danger_zone"]["points"] == []


def test_the_shipped_config_has_no_default_zone():
    """config.yaml must not hand a fresh install a zone nobody drew."""
    settings = load_config(os.path.join(os.path.dirname(__file__), os.pardir, "config.yaml"))

    assert settings["danger_zone"]["points"] == []


def test_defaults_are_applied(tmp_path):
    settings = load_config(write_config(tmp_path, BASE_CONFIG))
    assert settings["danger_zone"]["anchor"] == "center"
    assert settings["detector"]["confidence"] == 0.25
    assert settings["detector"]["person_class_id"] == 0


def test_person_gate_config_defaults_to_measured_values(tmp_path):
    """An older config.yaml without the gate keys behaves like the tuned one."""
    detector = load_config(write_config(tmp_path, BASE_CONFIG))["detector"]
    assert detector["person_confirm_conf"] == 0.4
    assert detector["person_confirm_frames"] == 3
    assert detector["person_instant_conf"] == 0.6
    assert detector["person_confirm_move_ratio"] == 0.05


def test_person_gate_config_values_are_honored(tmp_path):
    data = {**BASE_CONFIG, "detector": {"person_confirm_conf": 0.5,
                                        "person_confirm_frames": 5,
                                        "person_instant_conf": 0.7,
                                        "person_confirm_move_ratio": 0.1}}
    detector = load_config(write_config(tmp_path, data))["detector"]
    assert detector["person_confirm_conf"] == 0.5
    assert detector["person_confirm_frames"] == 5
    assert detector["person_instant_conf"] == 0.7
    assert detector["person_confirm_move_ratio"] == 0.1


@pytest.mark.parametrize(
    "detector",
    [
        {"person_confirm_conf": 0},
        {"person_confirm_conf": 1.5},
        {"person_confirm_frames": 0},
        {"person_instant_conf": 0},
        {"person_confirm_move_ratio": -0.1},
    ],
)
def test_invalid_person_gate_config_is_rejected(tmp_path, detector):
    with pytest.raises(ConfigError):
        load_config(write_config(tmp_path, {**BASE_CONFIG, "detector": detector}))


def test_missing_config_file():
    with pytest.raises(ConfigError, match="not found"):
        load_config("does_not_exist.yaml")


@pytest.mark.parametrize(
    "mutation",
    [
        {"rules": {"mode": "SOMETIMES", "dwell_seconds": 5}},
        {"rules": {"mode": "TIMER", "dwell_seconds": 0}},
        {"danger_zone": {"points": [[0, 0], [1, 1]]}},
        {"danger_zone": {"points": [["a", 1], [1, 1], [2, 2]]}},
    ],
)
def test_invalid_config_is_rejected(tmp_path, mutation):
    data = {**BASE_CONFIG, **mutation}
    with pytest.raises(ConfigError):
        load_config(write_config(tmp_path, data))


# -- database --------------------------------------------------------------

def test_events_round_trip(tmp_path):
    database = Database(str(tmp_path / "test.db"))
    assert database.count() == 0

    database.insert_event(
        {"worker_id": 1, "machine_id": "excavator_01", "event_type": ZONE_ENTRY, "entered_at": 100.0, "dwell_seconds": 0.0}
    )
    database.insert_event(
        {
            "worker_id": 1,
            "machine_id": "excavator_01",
            "event_type": DWELL_ALERT,
            "entered_at": 100.0,
            "alert_at": 105.0,
            "dwell_seconds": 5.0,
        }
    )

    assert database.count() == 2
    assert [row["event_type"] for row in database.list_events()] == [DWELL_ALERT, ZONE_ENTRY]
    assert database.latest_event()["event_type"] == DWELL_ALERT
    assert database.latest_event(event_type=ZONE_ENTRY)["event_type"] == ZONE_ENTRY
    assert len(database.list_events(limit=1)) == 1


def test_capacity_inside_count_round_trips(tmp_path):
    """The crowd size on a capacity row survives the database, old rows read None."""
    database = Database(str(tmp_path / "test.db"))
    database.insert_event(
        {
            "worker_id": -1,
            "machine_id": "excavator_01",
            "event_type": CAPACITY_ALERT,
            "entered_at": 100.0,
            "alert_at": 100.0,
            "dwell_seconds": 0.0,
            "zone_index": 0,
            "zone_name": "Zone 1",
            "reason": "CAPACITY",
            "inside": 2,
        }
    )

    row = database.latest_event()
    assert row["inside"] == 2


def test_database_error_for_unwritable_path(tmp_path):
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory", encoding="utf-8")
    with pytest.raises(DatabaseError):
        Database(str(blocker / "nested" / "test.db"))


def test_database_handles_bad_payload(tmp_path):
    database = Database(str(tmp_path / "test.db"))
    with pytest.raises(DatabaseError):
        database.insert_event({"worker_id": "abc"})


# -- api -------------------------------------------------------------------

@pytest.fixture()
def client(tmp_path, monkeypatch):
    db_path = str(tmp_path / "api.db")
    snapshot = tmp_path / "workers.json"
    monkeypatch.setattr("api.server.DB_PATH", db_path)
    monkeypatch.setattr("api.server.SNAPSHOT_PATH", str(snapshot))
    return TestClient(fastapi_app), snapshot


def test_health_and_empty_events(client):
    test_client, _ = client
    assert test_client.get("/health").json()["status"] == "ok"

    body = test_client.get("/events").json()
    assert body["events"] == []
    assert test_client.get("/events/latest").json() == {"event": None}


def test_api_returns_events(client):
    test_client, _ = client
    from api import server

    Database(server.DB_PATH).insert_event(
        {
            "worker_id": 7,
            "machine_id": "excavator_01",
            "event_type": DWELL_ALERT,
            "entered_at": 1700000000.0,
            "alert_at": 1700000005.0,
            "dwell_seconds": 5.0,
        }
    )

    latest = test_client.get("/events/latest").json()["event"]
    assert latest["worker_id"] == 7
    assert latest["event_type"] == DWELL_ALERT
    assert latest["dwell_seconds"] == 5.0
    assert latest["alert_at"].startswith("2023-11-14")

    assert test_client.get("/events?limit=1").json()["count"] == 1
    assert test_client.get("/events?event_type=ZONE_ENTRY").json()["count"] == 0


def test_api_returns_capacity_inside_count(client):
    test_client, _ = client
    from api import server

    Database(server.DB_PATH).insert_event(
        {
            "worker_id": -1,
            "machine_id": "excavator_01",
            "event_type": CAPACITY_ALERT,
            "entered_at": 1700000000.0,
            "alert_at": 1700000000.0,
            "dwell_seconds": 0.0,
            "zone_index": 0,
            "zone_name": "Zone 1",
            "reason": "CAPACITY",
            "inside": 2,
        }
    )

    latest = test_client.get("/events/latest").json()["event"]
    assert latest["inside"] == 2


def test_workers_endpoint_without_snapshot(client):
    test_client, _ = client
    assert test_client.get("/workers").json() == {"running": False, "machine": None, "workers": []}


def test_workers_endpoint_with_snapshot(client):
    test_client, snapshot = client
    snapshot.write_text(
        '{"machine": {"id": "excavator_01", "name": "Excavator 01"},'
        ' "rules": {"mode": "TIMER", "dwell_seconds": 5},'
        ' "workers": [{"worker_id": 3, "state": "ALERT", "dwell_seconds": 6.1}]}',
        encoding="utf-8",
    )
    body = test_client.get("/workers").json()
    assert body["running"] is True
    assert body["workers"][0]["state"] == "ALERT"


def test_legacy_dashboard_is_served():
    test_client = TestClient(fastapi_app)
    response = test_client.get("/legacy")
    assert response.status_code == 200
    assert "Trinetra AI" in response.text


def test_root_serves_the_built_ui(tmp_path, monkeypatch):
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<html>Trinetra AI</html>", encoding="utf-8")
    (dist / "assets" / "index-abc.js").write_text("console.log(1)", encoding="utf-8")
    monkeypatch.setattr(server, "FRONTEND_DIST", str(dist))

    test_client = TestClient(fastapi_app)
    assert "Trinetra AI" in test_client.get("/").text
    assert test_client.get("/assets/index-abc.js").text == "console.log(1)"


def test_root_explains_a_ui_that_was_never_built(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "FRONTEND_DIST", str(tmp_path / "missing"))

    response = TestClient(fastapi_app).get("/")
    assert response.status_code == 200
    assert "start.bat" in response.text


def test_assets_cannot_escape_the_build(tmp_path, monkeypatch):
    dist = tmp_path / "dist"
    dist.mkdir()
    (tmp_path / "secret.txt").write_text("nope", encoding="utf-8")
    monkeypatch.setattr(server, "FRONTEND_DIST", str(dist))

    test_client = TestClient(fastapi_app)
    assert test_client.get("/assets/../secret.txt").status_code == 404
    assert test_client.get("/assets/missing.js").status_code == 404


# -- alert acknowledgement, filtering and snapshots ------------------------

def _insert_event(event_type, worker_id=1, **overrides):
    """Store one row in the fixture's database and return its id."""
    payload = {
        "worker_id": worker_id,
        "machine_id": "excavator_01",
        "event_type": event_type,
        "entered_at": 1700000000.0,
        "alert_at": 1700000005.0,
        "dwell_seconds": 5.0,
    }
    payload.update(overrides)
    return Database(server.DB_PATH).insert_event(payload)


def test_migration_adds_event_columns_and_is_repeatable(tmp_path):
    import sqlite3

    path = str(tmp_path / "old.db")
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE events ("
        " id INTEGER PRIMARY KEY AUTOINCREMENT,"
        " worker_id INTEGER NOT NULL,"
        " machine_id TEXT NOT NULL,"
        " event_type TEXT NOT NULL,"
        " entered_at REAL,"
        " alert_at REAL,"
        " dwell_seconds REAL NOT NULL DEFAULT 0)"
    )
    conn.execute(
        "INSERT INTO events (worker_id, machine_id, event_type, entered_at, dwell_seconds)"
        " VALUES (1, 'excavator_01', 'ZONE_ENTRY', 100.0, 0.0)"
    )
    conn.commit()
    conn.close()

    # first open migrates the old table...
    Database(path).count()
    # ...and a second open is a no-op rather than a duplicate-column error
    database = Database(path)
    assert database.count() == 1  # the pre-existing row survived

    event_id = database.insert_event(
        {
            "worker_id": 2,
            "machine_id": "excavator_01",
            "event_type": DWELL_ALERT,
            "entered_at": 200.0,
            "alert_at": 205.0,
            "dwell_seconds": 5.0,
            "zone_index": 3,
            "zone_name": "Kiln",
        }
    )
    database.set_snapshot_path(event_id, f"{event_id}.jpg")
    assert database.acknowledge_event(event_id, 210.0) is True

    row = database.get_event(event_id)
    assert row["zone_index"] == 3
    assert row["zone_name"] == "Kiln"
    assert row["snapshot_path"] == f"{event_id}.jpg"
    assert row["acknowledged"] == 1
    assert row["acknowledged_at"] == 210.0


def test_only_alerts_can_be_acknowledged(client):
    test_client, _ = client
    alert_id = _insert_event(DWELL_ALERT, worker_id=7)
    capacity_id = _insert_event(CAPACITY_ALERT, worker_id=7)
    entry_alert_id = _insert_event(ENTRY_ALERT, worker_id=7, reason="ENTRY")
    entry_id = _insert_event(ZONE_ENTRY, worker_id=7)
    safe_id = _insert_event(CAPACITY_SAFE, worker_id=7)

    first = test_client.post(f"/events/{alert_id}/ack")
    assert first.status_code == 200
    assert first.json()["event"]["acknowledged"] is True

    # a repeat ack succeeds and does not change the recorded time
    stamped = first.json()["event"]["acknowledged_at"]
    again = test_client.post(f"/events/{alert_id}/ack")
    assert again.status_code == 200
    assert again.json()["event"]["acknowledged_at"] == stamped

    assert test_client.post(f"/events/{capacity_id}/ack").status_code == 200
    # an entry alert is an alert, so it acks; the reason rides along
    entry_ack = test_client.post(f"/events/{entry_alert_id}/ack")
    assert entry_ack.status_code == 200
    assert entry_ack.json()["event"]["reason"] == "ENTRY"
    # an entry and a return-to-safe row are records of what happened, not prompts
    assert test_client.post(f"/events/{entry_id}/ack").status_code == 400
    assert test_client.post(f"/events/{safe_id}/ack").status_code == 400
    # a missing id is a 404, not a silent success
    assert test_client.post("/events/9999/ack").status_code == 404


def test_ack_all_marks_only_unacknowledged_alerts(client):
    test_client, _ = client
    from api import server

    database = Database(server.DB_PATH)
    dwell = _insert_event(DWELL_ALERT, worker_id=7)
    capacity = _insert_event(CAPACITY_ALERT, worker_id=7)
    entry_alert = _insert_event(ENTRY_ALERT, worker_id=7, reason="ENTRY")
    already = _insert_event(DWELL_ALERT, worker_id=7)
    database.acknowledge_event(already, 1700000001.0)
    entry = _insert_event(ZONE_ENTRY, worker_id=7)
    exited = _insert_event(ZONE_EXIT, worker_id=7)

    body = test_client.post("/events/ack-all").json()
    assert body == {"acknowledged": 3}

    # one shared timestamp across the three rows
    stamps = {
        database.get_event(dwell)["acknowledged_at"],
        database.get_event(capacity)["acknowledged_at"],
        database.get_event(entry_alert)["acknowledged_at"],
    }
    assert len(stamps) == 1
    # records and the already-cleared row are untouched
    assert database.get_event(entry)["acknowledged"] == 0
    assert database.get_event(exited)["acknowledged"] == 0
    assert database.get_event(already)["acknowledged_at"] == 1700000001.0
    # and a second call finds nothing left to do
    assert test_client.post("/events/ack-all").json() == {"acknowledged": 0}


def test_ack_all_honours_zone_and_type_filters(client):
    test_client, _ = client
    _insert_event(DWELL_ALERT, worker_id=1, zone_index=0, zone_name="Zone 1")
    _insert_event(DWELL_ALERT, worker_id=2, zone_index=1, zone_name="Kiln")
    _insert_event(CAPACITY_ALERT, worker_id=-1, zone_index=0, zone_name="Zone 1",
                  reason="CAPACITY", inside=2)

    assert test_client.post("/events/ack-all", params={"zone": "Kiln"}).json() == {
        "acknowledged": 1
    }
    assert test_client.post(
        "/events/ack-all", params={"event_type": "CAPACITY_ALERT"}
    ).json() == {"acknowledged": 1}
    # a non-alert kind matches nothing, it does not error
    assert test_client.post(
        "/events/ack-all", params={"event_type": "ZONE_ENTRY"}
    ).json() == {"acknowledged": 0}
    assert test_client.post("/events/ack-all").json() == {"acknowledged": 1}


def test_event_type_filter_is_bound_not_interpolated(client):
    test_client, _ = client
    _insert_event(DWELL_ALERT)
    _insert_event(ZONE_ENTRY)

    hostile = test_client.get("/events", params={"event_type": "' OR '1'='1"}).json()
    assert hostile["count"] == 0  # the value is a type that matches nothing
    assert test_client.get("/events").json()["count"] == 2  # the table is intact


def test_events_can_be_filtered_by_kind_and_acknowledged(client):
    test_client, _ = client
    alert_id = _insert_event(DWELL_ALERT)
    _insert_event(ZONE_ENTRY)
    test_client.post(f"/events/{alert_id}/ack")

    alerts = test_client.get(
        "/events", params=[("event_type", "DWELL_ALERT"), ("event_type", "CAPACITY_ALERT")]
    ).json()
    assert alerts["count"] == 1

    unacked = test_client.get("/events", params={"acknowledged": "false"}).json()
    assert unacked["count"] == 1  # the entry, not the acked alert
    acked = test_client.get("/events", params={"acknowledged": "true"}).json()
    assert acked["count"] == 1  # only the alert that was acknowledged


def test_snapshot_route_cannot_leave_its_folder(client, tmp_path, monkeypatch):
    test_client, _ = client
    snap_dir = tmp_path / "snaps"
    snap_dir.mkdir()
    monkeypatch.setattr("api.server.event_snapshot_dir", lambda: str(snap_dir))

    event_id = _insert_event(DWELL_ALERT)
    database = Database(server.DB_PATH)

    # a value that tries to walk out of the folder is refused, even when the
    # target really exists just outside it
    (tmp_path / "secret.jpg").write_bytes(b"top secret")
    database.set_snapshot_path(event_id, "../secret.jpg")
    leaked = test_client.get(f"/events/{event_id}/snapshot")
    assert leaked.status_code == 404
    assert b"secret" not in leaked.content

    # a real file inside the folder is served as a JPEG
    (tmp_path / "snaps" / f"{event_id}.jpg").write_bytes(b"\xff\xd8\xff\xe0picture")
    database.set_snapshot_path(event_id, f"{event_id}.jpg")
    served = test_client.get(f"/events/{event_id}/snapshot")
    assert served.status_code == 200
    assert served.headers["content-type"] == "image/jpeg"
    assert served.content.startswith(b"\xff\xd8")

    # an event with no picture at all is a 404, not an empty image
    other = _insert_event(DWELL_ALERT)
    assert test_client.get(f"/events/{other}/snapshot").status_code == 404
