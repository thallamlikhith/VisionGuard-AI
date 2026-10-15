"""History filters, CSV export and filter dropdowns (/events)."""

import csv
import io
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from api.server import app as fastapi_app
from api import server
from database.db import Database, IST


@pytest.fixture()
def client(tmp_path, monkeypatch):
    db_path = str(tmp_path / "history.db")
    monkeypatch.setattr("api.server.DB_PATH", db_path)
    return TestClient(fastapi_app)


def ist_epoch(day: str, hour: int = 12, minute: int = 0) -> float:
    """Noon (or the given clock time) on a plant-local day, as epoch seconds."""
    morning = datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=IST)
    return (morning + timedelta(hours=hour, minutes=minute)).timestamp()


def seed(database: Database) -> dict:
    """Rows across types, workers, zones and days; returns their ids by key."""
    ids = {}

    def add(key, **fields):
        payload = {
            "worker_id": 1,
            "machine_id": "excavator_01",
            "event_type": "ZONE_ENTRY",
            "entered_at": ist_epoch("2026-10-08"),
            "dwell_seconds": 0.0,
        }
        payload.update(fields)
        ids[key] = database.insert_event(payload)

    add("entry_w1", event_type="ZONE_ENTRY", worker_id=1, zone_index=0, zone_name="Zone 1")
    add(
        "dwell_w1",
        event_type="DWELL_ALERT",
        worker_id=1,
        zone_index=0,
        zone_name="Zone 1",
        alert_at=ist_epoch("2026-10-08") + 6,
        dwell_seconds=6.0,
        reason="DWELL",
    )
    add(
        "dwell_w2",
        event_type="DWELL_ALERT",
        worker_id=2,
        zone_index=1,
        zone_name="Kiln Platform",
        entered_at=ist_epoch("2026-10-05"),
        alert_at=ist_epoch("2026-10-05") + 9,
        dwell_seconds=9.0,
        reason="DWELL",
    )
    add(
        "capacity",
        event_type="CAPACITY_ALERT",
        worker_id=-1,
        zone_index=0,
        zone_name="Zone 1",
        alert_at=ist_epoch("2026-10-08") + 30,
        reason="CAPACITY",
        inside=2,
    )
    add(
        "exit_w1",
        event_type="ZONE_EXIT",
        worker_id=1,
        zone_index=0,
        zone_name="Zone 1",
        entered_at=ist_epoch("2026-10-08") + 60,
    )
    # 23:30 IST on Oct 7: the previous plant-local day, whatever UTC says
    add(
        "late_night",
        event_type="ZONE_ENTRY",
        worker_id=3,
        zone_index=0,
        zone_name="Zone 1",
        entered_at=ist_epoch("2026-10-07", 23, 30),
    )
    return ids


def test_filter_by_repeatable_event_type(client):
    test_client = client
    database = Database(server.DB_PATH)
    seed(database)

    body = test_client.get(
        "/events", params=[("event_type", "DWELL_ALERT"), ("event_type", "ZONE_EXIT")]
    ).json()
    assert body["total"] == 3
    assert {row["event_type"] for row in body["events"]} == {"DWELL_ALERT", "ZONE_EXIT"}


def test_filter_by_worker_and_zone(client):
    test_client = client
    database = Database(server.DB_PATH)
    seed(database)

    assert test_client.get("/events", params={"worker_id": 2}).json()["total"] == 1
    assert test_client.get("/events", params={"zone": "0"}).json()["total"] == 5
    assert (
        test_client.get("/events", params={"zone": "Kiln Platform"}).json()["total"] == 1
    )
    combined = test_client.get(
        "/events", params={"worker_id": 1, "event_type": "DWELL_ALERT"}
    ).json()
    assert combined["total"] == 1
    assert combined["events"][0]["worker_id"] == 1


def test_date_filters_are_plant_local_and_inclusive(client):
    test_client = client
    database = Database(server.DB_PATH)
    seed(database)

    # Oct 8 in IST: everything except the Oct 5 dwell and the Oct 7 night row
    day = test_client.get(
        "/events", params={"date_from": "2026-10-08", "date_to": "2026-10-08"}
    ).json()
    assert day["total"] == 4

    # a 23:30 IST row belongs to Oct 7 even though UTC still says Oct 7
    # afternoon - and a UTC-midnight row would belong to the next IST day
    database.insert_event(
        {
            "worker_id": 9,
            "machine_id": "excavator_01",
            "event_type": "ZONE_ENTRY",
            "entered_at": datetime(2026, 10, 7, 18, 0, tzinfo=timezone.utc).timestamp(),
            "dwell_seconds": 0.0,
        }
    )
    assert (
        test_client.get(
            "/events", params={"date_from": "2026-10-07", "date_to": "2026-10-07"}
        ).json()["total"]
        == 2
    )


def test_bad_dates_are_rejected(client):
    test_client = client
    assert test_client.get("/events", params={"date_from": "08-10-2026"}).status_code == 422
    assert test_client.get("/events/export.csv", params={"date_to": "soon"}).status_code == 422


def test_limit_offset_and_total(client):
    test_client = client
    database = Database(server.DB_PATH)
    seed(database)

    first = test_client.get("/events", params={"limit": 2, "offset": 0}).json()
    assert first["count"] == 2
    assert first["total"] == 6
    second = test_client.get("/events", params={"limit": 2, "offset": 2}).json()
    assert second["count"] == 2
    assert [row["id"] for row in first["events"]] != [row["id"] for row in second["events"]]


def test_filter_options_expose_workers_and_zones(client):
    test_client = client
    database = Database(server.DB_PATH)
    seed(database)

    body = test_client.get("/events/filters").json()
    # the -1 capacity sentinel is not a person and stays out of the dropdown
    assert body["workers"] == [1, 2, 3]
    assert body["zones"] == [
        {"index": 0, "name": "Zone 1"},
        {"index": 1, "name": "Kiln Platform"},
    ]


def test_csv_header_rows_and_local_time(client):
    test_client = client
    database = Database(server.DB_PATH)
    seed(database)

    response = test_client.get(
        "/events/export.csv",
        params=[("event_type", "DWELL_ALERT"), ("event_type", "CAPACITY_ALERT")],
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    today = datetime.now(tz=IST).date().isoformat()
    assert response.headers["content-disposition"] == f'attachment; filename="history_{today}.csv"'

    rows = list(csv.reader(io.StringIO(response.text)))
    assert rows[0] == [
        "Time",
        "Worker",
        "Zone",
        "Event",
        "Dwell (s)",
        "People inside",
        "Acknowledged",
        "Acknowledged at",
    ]
    by_event = {row[3]: row for row in rows[1:]}
    assert set(by_event) == {"Dwell alert", "Capacity alert"}

    # noon IST stays noon IST in the file, not shifted to server time
    assert by_event["Capacity alert"][0] == "2026-10-08 12:00:30"
    assert by_event["Capacity alert"][1] == "Zone 1 · 2 people"
    assert by_event["Capacity alert"][2] == "Zone 1"
    assert by_event["Capacity alert"][5] == "2"
    assert by_event["Dwell alert"][1] == "Worker #1"
    assert by_event["Dwell alert"][4] == "6.0"
    assert by_event["Dwell alert"][6] == "No"


def test_csv_ignores_paging_and_honours_dates(client):
    test_client = client
    database = Database(server.DB_PATH)
    seed(database)

    # limit/offset change the table, never the file: extra params are ignored
    full = list(csv.reader(io.StringIO(test_client.get("/events/export.csv").text)))
    assert len(full) == 7  # header + 6 rows
    paged = test_client.get("/events/export.csv", params={"limit": 2, "offset": 2})
    assert paged.status_code == 200
    assert len(list(csv.reader(io.StringIO(paged.text)))) == 7

    day = list(
        csv.reader(
            io.StringIO(
                test_client.get(
                    "/events/export.csv",
                    params={"date_from": "2026-10-08", "date_to": "2026-10-08"},
                ).text
            )
        )
    )
    assert len(day) == 5  # header + the 4 Oct-8 rows
