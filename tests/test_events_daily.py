"""Tests for GET /events/daily: the 14-day alert chart counts incidents, not beeps."""

from datetime import datetime

import pytest
from fastapi.testclient import TestClient

import api.server as server
from api.server import app as fastapi_app
from database.db import Database


@pytest.fixture()
def client(tmp_path, monkeypatch):
    db_path = str(tmp_path / "daily.db")
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
    return TestClient(fastapi_app), db_path


def noon(days_ago: int = 0) -> float:
    """Local noon N days ago, far from any midnight line."""
    return (
        datetime.now().replace(hour=12, minute=0, second=0, microsecond=0).timestamp()
        - days_ago * 86400
    )


def local_day(ts: float) -> str:
    return datetime.fromtimestamp(ts).date().isoformat()


def dwell_row(worker: int, zone: int, entered: float, alert: float, name: str = "Kiln"):
    return {
        "worker_id": worker,
        "machine_id": "excavator_01",
        "event_type": "DWELL_ALERT",
        "entered_at": entered,
        "alert_at": alert,
        "dwell_seconds": 6.0,
        "zone_index": zone,
        "zone_name": name,
    }


def capacity_row(zone: int, at: float, name: str = "Kiln"):
    return {
        "worker_id": -1,
        "machine_id": "excavator_01",
        "event_type": "CAPACITY_ALERT",
        "entered_at": at,
        "alert_at": at,
        "dwell_seconds": 0.0,
        "zone_index": zone,
        "zone_name": name,
    }


def entry_alert_row(worker: int, zone: int, at: float, name: str = "Kiln"):
    return {
        "worker_id": worker,
        "machine_id": "excavator_01",
        "event_type": "ENTRY_ALERT",
        "entered_at": at,
        "alert_at": at,
        "dwell_seconds": 0.0,
        "zone_index": zone,
        "zone_name": name,
        "reason": "ENTRY",
    }


def entry_row(worker: int, zone: int, at: float):
    return {
        "worker_id": worker,
        "machine_id": "excavator_01",
        "event_type": "ZONE_ENTRY",
        "entered_at": at,
        "alert_at": None,
        "dwell_seconds": 0.0,
        "zone_index": zone,
        "zone_name": "Kiln",
    }


def day_map(body: dict) -> dict:
    return {row["date"]: (row["entry"], row["dwell"], row["capacity"]) for row in body["days"]}


def test_empty_database_returns_fourteen_zero_rows(client):
    test_client, _ = client

    body = test_client.get("/events/daily").json()

    assert body["days_requested"] == 14
    assert len(body["days"]) == 14
    assert all(
        row["entry"] == 0 and row["dwell"] == 0 and row["capacity"] == 0 for row in body["days"]
    )
    dates = [row["date"] for row in body["days"]]
    assert dates == sorted(dates)
    assert dates[-1] == datetime.now().date().isoformat()


def test_dwell_repeats_of_one_visit_count_once(client):
    test_client, db_path = client
    entered = noon()
    database = Database(db_path)
    for repeat in range(3):
        database.insert_event(dwell_row(3, 0, entered, entered + 5 + repeat * 10))

    body = test_client.get("/events/daily").json()

    assert day_map(body)[local_day(entered)] == (0, 1, 0)


def test_entry_alerts_count_as_their_own_series(client):
    test_client, db_path = client
    database = Database(db_path)
    database.insert_event(entry_alert_row(3, 0, noon()))
    database.insert_event(entry_row(7, 0, noon()))

    body = test_client.get("/events/daily").json()

    # the ENTRY_ALERT counts; the ZONE_ENTRY history row does not
    assert day_map(body)[local_day(noon())] == (1, 0, 0)


def test_two_visits_count_twice_and_capacity_counts(client):
    test_client, db_path = client
    database = Database(db_path)
    database.insert_event(dwell_row(3, 0, noon(2), noon(2) + 6))
    database.insert_event(dwell_row(3, 0, noon(1), noon(1) + 6))
    database.insert_event(capacity_row(1, noon(1)))
    database.insert_event(entry_row(3, 0, noon(1)))

    body = test_client.get("/events/daily").json()
    counts = day_map(body)

    assert counts[local_day(noon(2))] == (0, 1, 0)
    # entries are history, not alerts
    assert counts[local_day(noon(1))] == (0, 1, 1)


def test_zone_and_worker_filters(client):
    test_client, db_path = client
    database = Database(db_path)
    database.insert_event(dwell_row(3, 0, noon(), noon() + 6, name="Kiln"))
    database.insert_event(dwell_row(7, 1, noon(), noon() + 6, name="Sand"))
    today = local_day(noon())

    assert day_map(test_client.get("/events/daily", params={"zone": "0"}).json())[today] == (0, 1, 0)
    assert day_map(test_client.get("/events/daily", params={"zone": "Sand"}).json())[today] == (0, 1, 0)
    assert day_map(test_client.get("/events/daily", params={"worker_id": 7}).json())[today] == (0, 1, 0)
    assert day_map(test_client.get("/events/daily", params={"worker_id": 9}).json())[today] == (0, 0, 0)


def test_old_events_fall_outside_the_window(client):
    test_client, db_path = client
    database = Database(db_path)
    database.insert_event(capacity_row(0, noon(20)))

    body = test_client.get("/events/daily").json()
    assert all(row["capacity"] == 0 for row in body["days"])

    body = test_client.get("/events/daily", params={"days": 21}).json()
    assert len(body["days"]) == 21
    assert day_map(body)[local_day(noon(20))] == (0, 0, 1)


def test_days_is_clamped(client):
    test_client, _ = client

    assert test_client.get("/events/daily", params={"days": 0}).status_code == 422
    assert test_client.get("/events/daily", params={"days": 99}).status_code == 422
