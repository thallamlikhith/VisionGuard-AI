"""Tests for the web UI endpoints: config save, start/stop, state, stream."""

import json

import pytest
from fastapi.testclient import TestClient

import api.server as server
from api.server import app as fastapi_app
from database.db import Database


@pytest.fixture()
def client(tmp_path, monkeypatch):
    db_path = str(tmp_path / "ui.db")
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


def test_a_fresh_database_has_no_zone(client):
    test_client, _ = client

    body = test_client.get("/config").json()

    # config.yaml may carry a polygon for a hand set up machine, but a machine
    # with nothing stored is not given one: there is no default danger zone
    assert body["zones"] == []
    assert body["zone_points"] == []
    # the rest of config.yaml is still the fallback
    assert body["machine"]["name"] == "Excavator 01"
    assert body["rules"]["dwell_seconds"] == 5
    assert body["allowed_people"] == 0  # default must be 0
    assert body["source"] == "0"


def test_loop_video_defaults_off_and_round_trips(client):
    """The replay box is one stored setting, off until ticked."""
    test_client, db_path = client

    assert test_client.get("/config").json()["loop_video"] is False

    test_client.post("/config", json={"loop_video": True})
    assert test_client.get("/config").json()["loop_video"] is True
    assert Database(db_path).all_settings()["loop_video"] == "1"

    test_client.post("/config", json={"loop_video": False})
    assert test_client.get("/config").json()["loop_video"] is False
    assert Database(db_path).all_settings()["loop_video"] == "0"


def test_slow_state_logs_a_warning(client, monkeypatch, caplog):
    """A /state past ~2 s names itself in the log instead of surfacing only
    as a client-side timeout toast."""
    import logging
    import time as time_module

    test_client, _ = client
    calls = {"n": 0}

    def fake_perf_counter():
        # every call steps 2.6 s: whatever interleaves, the two reads in
        # the handler bracket at least one step and the warning fires
        calls["n"] += 1
        return 100.0 + (calls["n"] - 1) * 2.6

    monkeypatch.setattr(time_module, "perf_counter", fake_perf_counter)
    with caplog.at_level(logging.WARNING, logger="api.server"):
        body = test_client.get("/state").json()
    assert body["running"] is False
    assert any("/state slow" in record.getMessage() for record in caplog.records)


def test_fast_state_logs_no_warning(client, caplog):
    import logging

    test_client, _ = client
    with caplog.at_level(logging.WARNING, logger="api.server"):
        test_client.get("/state").json()
    assert not [r for r in caplog.records if "/state slow" in r.getMessage()]


def test_ppe_operating_point_defaults(client, monkeypatch):
    """The tuned pass (conf 0.15, imgsz 960) is reported and forwarded."""
    test_client, _ = client
    ppe = test_client.get("/config").json()["ppe"]
    assert ppe["confidence"] == pytest.approx(0.15)
    assert ppe["iou"] == pytest.approx(0.45)
    assert ppe["imgsz"] == 960
    assert ppe["tta"] is True

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
    assert captured["ppe_confidence"] == pytest.approx(0.15)
    assert captured["ppe_imgsz"] == 960
    assert captured["ppe_tta"] is True


def test_start_loops_a_test_clip_only_when_the_box_is_ticked(client, monkeypatch, tmp_path):
    """Default off plays once; the stored switch replays; an explicit loop
    wins; anything that is not an uploaded test clip (or the webcam) always
    plays once."""
    test_client, _ = client
    monkeypatch.setattr(server, "VIDEO_DIR", str(tmp_path))
    clip = tmp_path / "demo.mp4"
    clip.write_bytes(b"x")
    elsewhere = tmp_path / "sub"
    elsewhere.mkdir()
    other = elsewhere / "other.mp4"
    other.write_bytes(b"x")

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

    # default off: a test clip plays once
    test_client.post("/start", json={"source": str(clip)})
    assert captured["loop"] is False

    # ticked: the same clip replays
    test_client.post("/config", json={"loop_video": True})
    test_client.post("/start", json={"source": str(clip)})
    assert captured["loop"] is True

    # an explicit loop wins over the stored switch
    test_client.post("/start", json={"source": str(clip), "loop": False})
    assert captured["loop"] is False

    # a file that is not an uploaded test clip always plays once
    test_client.post("/start", json={"source": str(other)})
    assert captured["loop"] is False

    # the webcam is unaffected either way
    test_client.post("/start", json={"source": "0"})
    assert captured["loop"] is False


def test_config_yaml_never_puts_a_zone_back_when_there_is_none_stored(client):
    test_client, db_path = client
    # the shipped config.yaml used to hand its polygon to every install, so this
    # database has the "cleared" markers but no zone
    database = Database(db_path)
    database.set_setting("zone_polygons", json.dumps([]))
    database.set_setting("zone_points", json.dumps([]))

    assert server.load_settings()["zones"] == []
    assert test_client.get("/config").json()["zones"] == []


def test_every_stored_zone_survives_a_restart(client):
    test_client, db_path = client
    first = [[100, 100], [400, 100], [400, 300], [100, 300]]
    second = [[420, 100], [620, 100], [620, 300], [420, 300]]
    Database(db_path).set_setting("zone_polygons", json.dumps([first, second]))
    Database(db_path).set_setting("zone_reference", json.dumps([640, 480]))

    with TestClient(fastapi_app):
        pass

    # a restart does not lose the second zone, and no default appears
    assert server.load_settings()["zones"] == [first, second]


def test_two_zones_are_saved_and_read_back_in_order(client):
    test_client, _ = client
    first = [[100, 100], [400, 100], [400, 300], [100, 300]]
    second = [[420, 100], [620, 100], [620, 300], [420, 300]]

    saved = test_client.post(
        "/config",
        json={"zones": [first, second], "frame_width": 640, "frame_height": 480},
    ).json()
    assert saved["saved"] is True

    reloaded = test_client.get("/config").json()
    assert reloaded["zones"] == [first, second]
    # the legacy key still carries the first polygon for an older reader
    assert reloaded["zone_points"] == first

    assert server.load_settings()["zones"] == [first, second]


def test_editing_one_zone_keeps_the_others(client):
    test_client, _ = client
    first = [[100, 100], [400, 100], [400, 300], [100, 300]]
    second = [[420, 100], [620, 100], [620, 300], [420, 300]]
    test_client.post(
        "/config",
        json={"zones": [first, second], "frame_width": 640, "frame_height": 480},
    )

    # only the first zone was moved, the whole list is sent back
    moved = [[120, 120], [420, 120], [420, 320], [120, 320]]
    test_client.post(
        "/config",
        json={"zones": [moved, second], "frame_width": 640, "frame_height": 480},
    )

    assert server.load_settings()["zones"] == [moved, second]


def test_saving_one_zone_replaces_every_stored_zone(client):
    test_client, db_path = client
    stored = [
        [[100, 100], [400, 100], [400, 300], [100, 300]],
        [[500, 100], [700, 100], [700, 300], [500, 300]],
    ]
    Database(db_path).set_setting("zone_polygons", json.dumps(stored))

    drawn = [[120, 120], [420, 120], [420, 320], [120, 320]]
    test_client.post(
        "/config", json={"zones": [drawn], "frame_width": 640, "frame_height": 480}
    )

    reloaded = server.load_settings()
    assert reloaded["zones"] == [drawn]
    assert reloaded["zone_points"] == drawn


def test_saving_nothing_empties_every_stored_zone(client):
    test_client, db_path = client
    Database(db_path).set_setting(
        "zone_polygons",
        json.dumps([[[100, 100], [400, 100], [400, 300], [100, 300]]]),
    )

    test_client.post(
        "/config", json={"zones": [], "frame_width": 640, "frame_height": 480}
    )

    reloaded = server.load_settings()
    assert reloaded["zones"] == []
    assert reloaded["zone_points"] == []


def test_a_cleared_zone_stays_cleared_after_a_restart(client):
    test_client, db_path = client
    polygon = [[100, 100], [400, 100], [450, 300], [120, 320]]
    test_client.post("/config", json={"zones": [polygon], "frame_width": 640, "frame_height": 480})
    assert test_client.get("/config").json()["zones"] == [polygon]

    test_client.post("/config", json={"zones": [], "frame_width": 640, "frame_height": 480})

    # "restart": a new process reads the same database file
    reloaded = server.load_settings()
    assert reloaded["zones"] == []
    assert reloaded["zone_points"] == []
    assert test_client.get("/config").json()["zones"] == []


def test_the_old_default_polygon_is_cleared_on_startup(client):
    test_client, db_path = client
    database = Database(db_path)
    database.set_setting(
        "zone_polygons", json.dumps([[[400, 200], [700, 200], [760, 500], [350, 500]]])
    )
    database.set_setting("zone_points", json.dumps([[400, 200], [700, 200], [760, 500], [350, 500]]))
    database.set_setting("zone_reference", json.dumps([1280, 720]))
    assert server.load_settings()["zones"], "the old default is there before the cleanup"

    with TestClient(fastapi_app):
        pass

    assert server.load_settings()["zones"] == []
    assert server.load_settings()["zone_points"] == []


def test_the_old_default_polygon_scaled_to_another_camera_is_cleared(client):
    test_client, db_path = client
    # the same polygon on a 640x360 camera, as a scaled copy would be stored
    database = Database(db_path)
    database.set_setting(
        "zone_polygons", json.dumps([[[200, 100], [350, 100], [380, 250], [175, 250]]])
    )
    database.set_setting("zone_reference", json.dumps([640, 360]))

    with TestClient(fastapi_app):
        pass

    assert server.load_settings()["zones"] == []


def test_clearing_the_old_default_keeps_a_zone_drawn_next_to_it(client):
    test_client, db_path = client
    # the leftover default from an old install, plus a zone the operator drew
    old_default = [[400, 200], [700, 200], [760, 500], [350, 500]]
    drawn = [[100, 100], [300, 100], [300, 300], [100, 300]]
    database = Database(db_path)
    database.set_setting("zone_polygons", json.dumps([drawn, old_default]))
    database.set_setting("zone_points", json.dumps(drawn))
    database.set_setting("zone_reference", json.dumps([1280, 720]))

    with TestClient(fastapi_app):
        pass

    settings = server.load_settings()
    assert settings["zones"] == [drawn]
    # the kept zone keeps the frame it was drawn on, so it is not rescaled
    assert settings["zone_reference"] == [1280, 720]


def test_two_leftover_defaults_beside_a_drawn_zone_all_go(client):
    test_client, db_path = client
    old_default = [[400, 200], [700, 200], [760, 500], [350, 500]]
    drawn = [[100, 100], [300, 100], [300, 300], [100, 300]]
    database = Database(db_path)
    database.set_setting("zone_polygons", json.dumps([old_default, drawn, old_default]))
    database.set_setting("zone_reference", json.dumps([1280, 720]))

    with TestClient(fastapi_app):
        pass

    assert server.load_settings()["zones"] == [drawn]


def test_a_zone_somebody_drew_is_never_taken_away(client):
    test_client, db_path = client
    drawn = [[100, 100], [400, 100], [450, 300], [120, 320]]
    database = Database(db_path)
    database.set_setting("zone_polygons", json.dumps([drawn]))
    database.set_setting("zone_reference", json.dumps([1280, 720]))

    with TestClient(fastapi_app):
        pass

    assert server.load_settings()["zones"] == [drawn]


def test_saving_the_polygon_and_allowed_people(client):
    test_client, db_path = client
    polygon = [[100, 100], [400, 100], [450, 300], [120, 320]]

    body = test_client.post("/config", json={"points": polygon, "allowed_people": 2}).json()
    assert body["saved"] is True

    # stored in the existing sqlite database
    settings = Database(db_path).all_settings()
    assert json.loads(settings["zone_points"]) == polygon
    assert settings["allowed_people"] == "2"

    # and read back through the API
    reloaded = test_client.get("/config").json()
    assert reloaded["zone_points"] == polygon
    assert reloaded["allowed_people"] == 2


def test_allowed_people_persists_across_requests(client):
    test_client, _ = client
    test_client.post("/config", json={"points": [[1, 1], [2, 1], [2, 2]], "allowed_people": 5})
    assert test_client.get("/config").json()["allowed_people"] == 5


def test_saving_a_malformed_point_is_rejected(client):
    test_client, _ = client
    response = test_client.post("/config", json={"points": [[1, 1], [2, 1], [2, 2, 3]]})
    assert response.status_code == 422
    # the detail is a plain string so the browser can show it as it is, and it
    # names the point that is wrong
    assert response.json()["detail"].startswith("Point 2 is invalid")


def test_a_zone_with_too_few_points_is_rejected_with_422(client):
    test_client, _ = client
    response = test_client.post("/config", json={"points": [[1, 1], [2, 1]]})
    assert response.status_code == 422
    assert "at least 3 points" in response.json()["detail"]


def test_a_zone_with_too_many_points_is_rejected_with_422(client):
    """A rounded circle is 24 corners, so the limit is 32 - not the old 6."""
    test_client, _ = client
    assert test_client.post("/config", json={"points": _wobbly_polygon(32)}).status_code == 200

    points = _wobbly_polygon(33)
    response = test_client.post("/config", json={"points": points})
    assert response.status_code == 422
    assert "at most 32 points" in response.json()["detail"]


def _wobbly_polygon(count):
    """A convex ring of `count` corners: no duplicates, no crossings, big enough."""
    import math

    cx, cy, radius = 320, 240, 200
    return [
        [
            int(round(cx + radius * math.cos(2 * math.pi * i / count))),
            int(round(cy + radius * math.sin(2 * math.pi * i / count))),
        ]
        for i in range(count)
    ]


def test_a_point_outside_the_known_frame_is_clamped_not_rejected(client):
    test_client, _ = client
    # the second corner is at x=1100, past the 640 wide frame; zone coords are
    # clamped to the frame on save rather than refused
    response = test_client.post(
        "/config",
        json={
            "points": [[10, 10], [1100, 10], [600, 400]],
            "frame_width": 640,
            "frame_height": 480,
        },
    )
    assert response.status_code == 200
    assert server.load_settings()["zones"] == [[[10, 10], [640, 10], [600, 400]]]


def test_points_inside_the_known_frame_are_accepted(client):
    test_client, _ = client
    response = test_client.post(
        "/config",
        json={
            "points": [[10, 10], [600, 10], [600, 400]],
            "frame_width": 640,
            "frame_height": 480,
        },
    )
    assert response.status_code == 200
    assert response.json()["saved"] is True


def test_bounds_are_not_checked_while_the_frame_size_is_unknown(client):
    """No frame size means no verdict - a made up 0x0 frame would reject all."""
    test_client, _ = client
    response = test_client.post("/config", json={"points": [[10, 10], [2000, 10], [2000, 2000]]})
    assert response.status_code == 200


# -- polygons that look drawn but can never enclose anybody ------------------

def test_a_repeated_closing_corner_is_merged_away(client):
    test_client, _ = client
    # the same corner written twice to close the shape is one corner, so this
    # is the same square as anywhere else and has to save
    response = test_client.post(
        "/config",
        json={
            "points": [[10, 10], [600, 10], [600, 400], [10, 10]],
            "frame_width": 640,
            "frame_height": 480,
        },
    )
    assert response.status_code == 200
    assert server.load_settings()["zones"] == [[[10, 10], [600, 10], [600, 400]]]


def test_a_zone_that_repeats_a_corner_in_the_middle_is_rejected(client):
    test_client, _ = client
    response = test_client.post(
        "/config",
        json={
            "points": [[10, 10], [600, 10], [600, 400], [10, 10], [10, 400]],
            "frame_width": 640,
            "frame_height": 480,
        },
    )
    assert response.status_code == 422
    assert "the corners must all be different" in response.json()["detail"]


def test_a_self_crossing_zone_is_rejected(client):
    test_client, _ = client
    # a bow tie: the edges meet in the middle, so it encloses nothing
    response = test_client.post(
        "/config",
        json={
            "points": [[10, 10], [600, 400], [600, 10], [10, 400]],
            "frame_width": 640,
            "frame_height": 480,
        },
    )
    assert response.status_code == 422
    assert "crosses over itself" in response.json()["detail"]


def test_a_zone_that_is_a_thin_sliver_is_rejected(client):
    test_client, _ = client
    # under 0.5% of a 640x480 frame: 400x10 is 1.3% ... 200x2 is not
    response = test_client.post(
        "/config",
        json={
            "points": [[10, 10], [210, 10], [210, 12], [10, 12]],
            "frame_width": 640,
            "frame_height": 480,
        },
    )
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert "covers only" in detail
    assert "at least 0.5%" in detail


def test_a_zone_between_half_a_percent_and_two_percent_is_accepted(client):
    test_client, _ = client
    # 1.3% of the frame: too small for the old 2% floor, big enough now
    response = test_client.post(
        "/config",
        json={
            "points": [[10, 10], [410, 10], [410, 70], [10, 70]],
            "frame_width": 640,
            "frame_height": 480,
        },
    )
    assert response.status_code == 200
    assert server.load_settings()["zones"] == [
        [[10, 10], [410, 10], [410, 70], [10, 70]]
    ]


def test_a_zone_covering_enough_of_the_frame_is_accepted(client):
    test_client, _ = client
    response = test_client.post(
        "/config",
        json={
            "points": [[10, 10], [600, 10], [600, 200], [10, 200]],
            "frame_width": 640,
            "frame_height": 480,
        },
    )
    assert response.status_code == 200


def test_clearing_the_zone_with_an_empty_list_is_still_allowed(client):
    test_client, _ = client
    response = test_client.post(
        "/config",
        json={"points": [], "frame_width": 640, "frame_height": 480},
    )
    assert response.status_code == 200


# -- which point of a person has to be inside the zone ------------------------

def test_the_anchor_defaults_to_the_body_center(client):
    test_client, _ = client

    assert test_client.get("/config").json()["anchor"] == "center"


def test_the_chosen_anchor_is_saved_and_read_back(client):
    test_client, db_path = client

    body = test_client.post(
        "/config",
        json={
            "zones": [[[100, 100], [400, 100], [400, 300], [100, 300]]],
            "frame_width": 640,
            "frame_height": 480,
            "anchor": "center",
        },
    ).json()
    assert body["saved"] is True
    assert body["anchor"] == "center"

    assert Database(db_path).all_settings()["anchor"] == "center"
    assert test_client.get("/config").json()["anchor"] == "center"


def test_saving_only_the_zones_leaves_the_anchor_alone(client):
    test_client, _ = client
    test_client.post("/config", json={"anchor": "center"})

    test_client.post(
        "/config",
        json={"zones": [[[100, 100], [400, 100], [400, 300], [100, 300]]]},
    )

    assert test_client.get("/config").json()["anchor"] == "center"


def test_an_unknown_anchor_is_rejected(client):
    test_client, _ = client

    response = test_client.post("/config", json={"anchor": "nose"})
    assert response.status_code == 422


def test_start_forwards_the_saved_anchor_to_the_pipeline(client, monkeypatch):
    test_client, _ = client
    test_client.post("/config", json={"anchor": "center"})

    captured = {}

    class FakePipeline:
        def stop(self):
            return True

        def start(self, **kwargs):
            captured.update(kwargs)

        def state(self):
            return {}

    monkeypatch.setattr(server, "_pipeline", FakePipeline())
    body = test_client.post("/start", json={}).json()

    assert captured["anchor"] == "center"
    assert body["anchor"] == "center"


def test_start_uses_the_anchor_it_was_given(client, monkeypatch):
    test_client, _ = client
    test_client.post("/config", json={"anchor": "center"})

    captured = {}

    class FakePipeline:
        def stop(self):
            return True

        def start(self, **kwargs):
            captured.update(kwargs)

        def state(self):
            return {}

    monkeypatch.setattr(server, "_pipeline", FakePipeline())
    test_client.post("/start", json={"anchor": "bottom_center"})

    assert captured["anchor"] == "bottom_center"


def test_changing_the_anchor_reaches_a_running_camera(client, monkeypatch):
    """Switching anchor must not cost the operator the running camera."""
    test_client, _ = client

    class FakePipeline:
        active = True
        source = "0"
        frame_size = (640, 480)

        def __init__(self):
            self.starts = 0
            self.stops = 0
            self.applied = {}

        def apply_settings(self, **kwargs):
            self.applied.update(kwargs)
            return [3, 4]

        def state(self):
            return {}

    fake = FakePipeline()
    monkeypatch.setattr(server, "_pipeline", fake)

    # the zone comes along because the running camera's frame size is what the
    # stored points are checked against
    zone = [[100, 100], [400, 100], [400, 300], [100, 300]]
    test_client.post("/config", json={"zones": [zone], "anchor": "center"})
    assert fake.applied["anchor"] == "center"

    # and the same without touching /config at all
    body = test_client.post("/start", json={"anchor": "bottom_center"}).json()
    assert body["anchor"] == "bottom_center"
    assert fake.applied["anchor"] == "bottom_center"
    assert (fake.starts, fake.stops) == (0, 0)


# -- several danger zones at once -------------------------------------------

def test_several_zones_are_saved_and_read_back(client):
    test_client, db_path = client
    first = [[100, 100], [400, 100], [400, 300], [100, 300]]
    second = [[500, 100], [700, 100], [700, 300], [500, 300]]

    body = test_client.post(
        "/config",
        json={"zones": [first, second], "frame_width": 800, "frame_height": 600},
    ).json()
    assert body["saved"] is True
    assert body["zones"] == [first, second]

    stored = Database(db_path).all_settings()
    assert json.loads(stored["zone_polygons"]) == [first, second]
    # the first zone is mirrored into the old key for older readers
    assert json.loads(stored["zone_points"]) == first

    assert test_client.get("/config").json()["zones"] == [first, second]


def test_the_old_single_zone_payload_still_saves_one_zone(client):
    """An older UI sends `points`; it must keep working unchanged."""
    test_client, db_path = client
    polygon = [[100, 100], [400, 100], [450, 300], [120, 320]]

    test_client.post("/config", json={"points": polygon, "allowed_people": 1})

    assert json.loads(Database(db_path).all_settings()["zone_polygons"]) == [polygon]
    assert test_client.get("/config").json()["zones"] == [polygon]
    assert test_client.get("/config").json()["zone_points"] == polygon


def test_a_zone_that_only_existed_in_the_old_key_is_still_read(client):
    """A database written before multi-zone must not lose its zone."""
    test_client, db_path = client
    polygon = [[10, 10], [60, 10], [60, 60]]
    Database(db_path).set_setting("zone_points", json.dumps(polygon))

    body = test_client.get("/config").json()

    assert body["zones"] == [polygon]
    assert body["zone_points"] == polygon


def test_saving_only_the_rules_leaves_every_zone_alone(client):
    test_client, _ = client
    first = [[100, 100], [400, 100], [400, 300], [100, 300]]
    second = [[500, 100], [700, 100], [700, 300], [500, 300]]
    test_client.post("/config", json={"zones": [first, second]})

    test_client.post("/config", json={"allowed_people": 4})

    assert test_client.get("/config").json()["zones"] == [first, second]


def test_a_bad_zone_names_which_one_is_wrong(client):
    test_client, _ = client
    good = [[100, 100], [400, 100], [400, 300], [100, 300]]
    # a bow tie in the second zone
    bow_tie = [[10, 10], [600, 400], [600, 10], [10, 400]]

    response = test_client.post(
        "/config",
        json={"zones": [good, bow_tie], "frame_width": 640, "frame_height": 480},
    )
    assert response.status_code == 422
    assert response.json()["detail"].startswith("Zone 2:")


def test_start_forwards_every_zone_to_the_pipeline(client, monkeypatch):
    test_client, _ = client
    first = [[100, 100], [400, 100], [400, 300], [100, 300]]
    second = [[500, 100], [700, 100], [700, 300], [500, 300]]
    test_client.post("/config", json={"zones": [first, second]})

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

    assert captured["zones"] == [first, second]


def test_start_updates_a_running_pipeline_without_reopening_the_camera(client, monkeypatch):
    """Saving a redrawn zone must not restart the video."""
    test_client, _ = client

    class FakePipeline:
        active = True
        source = "0"
        frame_size = (640, 480)

        def __init__(self):
            self.starts = 0
            self.stops = 0
            self.applied = {}

        def stop(self):
            self.stops += 1
            return True

        def start(self, **kwargs):
            self.starts += 1

        def apply_settings(self, **kwargs):
            self.applied.update(kwargs)
            return [3, 4]

        def state(self):
            return {}

    fake = FakePipeline()
    monkeypatch.setattr(server, "_pipeline", fake)

    zones = [[[100, 100], [400, 100], [400, 300], [100, 300]]]
    body = test_client.post("/start", json={"zones": zones}).json()

    assert body["flipped"] == [3, 4]
    assert fake.starts == 0
    assert fake.stops == 0
    assert fake.applied["zones"] == zones


def test_saving_zones_pushes_them_into_a_running_camera(client, monkeypatch):
    """/config must not leave a running camera monitoring the old zone."""
    test_client, _ = client

    class FakePipeline:
        active = True
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

    zones = [[[100, 100], [400, 100], [400, 300], [100, 300]]]
    test_client.post("/config", json={"zones": zones, "frame_width": 640, "frame_height": 480})

    assert fake.applied["zones"] == zones


def test_a_saved_zone_too_large_for_the_camera_is_cleared_on_startup(client):
    """Every polygon is checked, so one bad zone is dropped even with good ones."""
    test_client, db_path = client
    good = [[10, 10], [300, 10], [300, 300], [10, 300]]
    sliver = [[10, 10], [300, 10], [300, 11], [10, 11]]
    database = Database(db_path)
    database.set_setting("zone_polygons", json.dumps([good, sliver]))
    database.set_setting("zone_reference", json.dumps([640, 480]))

    assert "Zone 2" in server.cleanup_invalid_zone()
    assert json.loads(database.all_settings()["zone_polygons"]) == []


# -- one-time cleanup of a stored zone that cannot be used ------------------

def test_startup_clears_a_stored_zone_that_cannot_be_used(client):
    test_client, db_path = client
    database = Database(db_path)
    database.set_setting("zone_points", json.dumps([[10, 10], [20, 10], [20, 20]]))
    database.set_setting("zone_reference", json.dumps([640, 480]))

    assert server.cleanup_invalid_zone() is not None

    assert json.loads(database.all_settings()["zone_points"]) == []
    assert json.loads(database.all_settings()["zone_reference"]) == [0, 0]


def test_startup_clears_a_stored_zone_that_crosses_itself(client):
    test_client, db_path = client
    database = Database(db_path)
    database.set_setting(
        "zone_points", json.dumps([[10, 10], [600, 400], [600, 10], [10, 400]])
    )
    database.set_setting("zone_reference", json.dumps([640, 480]))

    assert "crosses over itself" in server.cleanup_invalid_zone()
    assert json.loads(database.all_settings()["zone_points"]) == []


def test_startup_keeps_a_stored_zone_that_is_fine(client):
    test_client, db_path = client
    polygon = [[100, 100], [400, 100], [450, 300], [120, 320]]
    database = Database(db_path)
    database.set_setting("zone_points", json.dumps(polygon))
    database.set_setting("zone_reference", json.dumps([640, 480]))

    assert server.cleanup_invalid_zone() is None
    assert json.loads(database.all_settings()["zone_points"]) == polygon


def test_startup_leaves_an_empty_or_missing_zone_alone(client):
    test_client, db_path = client
    assert server.cleanup_invalid_zone() is None

    Database(db_path).set_setting("zone_points", "not json at all")
    assert server.cleanup_invalid_zone() is not None
    assert json.loads(Database(db_path).all_settings()["zone_points"]) == []


def test_startup_treats_empty_zone_strings_as_empty(client, caplog):
    """A present-but-empty "[]" is no stored zone: no warning, no rewrite."""
    import logging

    test_client, db_path = client
    database = Database(db_path)
    database.set_setting("zone_polygons", json.dumps([]))
    database.set_setting("zone_points", json.dumps([]))
    database.set_setting("zone_reference", json.dumps([640, 480]))

    with caplog.at_level(logging.WARNING, logger="api.server"):
        assert server.cleanup_invalid_zone() is None

    assert not [r for r in caplog.records if "cleared on startup" in r.getMessage()]
    settings = Database(db_path).all_settings()
    assert json.loads(settings["zone_polygons"]) == []
    assert json.loads(settings["zone_points"]) == []
    assert json.loads(settings["zone_reference"]) == [640, 480]


def test_startup_cleanup_runs_when_the_app_starts(client):
    """The lifespan hook is what makes the check happen without a user action."""
    test_client, db_path = client
    database = Database(db_path)
    database.set_setting("zone_points", json.dumps([[10, 10], [20, 10], [20, 20]]))
    database.set_setting("zone_reference", json.dumps([640, 480]))

    with TestClient(fastapi_app) as fresh:
        assert fresh.get("/health").json()["status"] == "ok"

    assert json.loads(database.all_settings()["zone_points"]) == []


def test_saving_with_a_frame_size_remembers_it_as_the_zone_reference(client):
    test_client, _ = client
    test_client.post(
        "/config",
        json={
            "points": [[10, 10], [600, 10], [600, 400]],
            "frame_width": 1280,
            "frame_height": 720,
        },
    )

    assert test_client.get("/config").json()["zone_reference"] == [1280, 720]


def test_start_passes_the_zone_reference_to_the_pipeline(client, monkeypatch):
    """The pipeline needs the reference size to scale the zone later."""
    test_client, db_path = client
    test_client.post(
        "/config",
        json={
            "points": [[10, 10], [600, 10], [600, 400]],
            "frame_width": 1280,
            "frame_height": 720,
            "source": "0",
        },
    )

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

    assert tuple(captured["zone_reference"]) == (1280, 720)
    assert captured["zones"] == [[[10, 10], [600, 10], [600, 400]]]
    assert db_path  # the settings came out of the database


def test_scaling_maps_a_zone_onto_a_differently_sized_video():
    """A zone drawn on 1280x720 keeps its place on a 640x360 video."""
    from pipeline import scale_zone_points

    scaled = scale_zone_points([(0, 0), (1280, 720), (640, 360)], (1280, 720), (640, 360))

    assert scaled == [(0, 0), (640, 360), (320, 180)]


def test_scaling_leaves_the_points_alone_for_the_same_size():
    from pipeline import scale_zone_points

    points = [(10, 10), (640, 360), (320, 180)]
    assert scale_zone_points(points, (640, 360), (640, 360)) == points


def test_scaling_keeps_the_points_inside_the_new_frame():
    from pipeline import scale_zone_points

    scaled = scale_zone_points([(0, 0), (1280, 720)], (1280, 720), (640, 360))

    assert all(0 <= x <= 640 and 0 <= y <= 360 for x, y in scaled)


def test_scaling_is_a_no_op_when_a_size_is_unknown():
    from pipeline import scale_zone_points

    points = [(10, 10), (640, 360), (320, 180)]
    assert scale_zone_points(points, (0, 0), (640, 360)) == points
    assert scale_zone_points(points, (1280, 720), (0, 0)) == points


def test_mode_dwell_and_repeat_round_trip_through_config(client):
    test_client, _ = client
    response = test_client.post(
        "/config",
        json={"mode": "INSTANT", "dwell_seconds": 2.5, "repeat_alert_seconds": 30},
    )

    assert response.status_code == 200
    assert response.json()["mode"] == "INSTANT"

    rules = test_client.get("/config").json()["rules"]
    assert rules["mode"] == "INSTANT"
    assert rules["dwell_seconds"] == 2.5
    assert rules["repeat_alert_seconds"] == 30.0


def test_limit_mode_round_trips_through_config(client):
    """The third alert mode saves and loads like the other two."""
    test_client, _ = client
    response = test_client.post("/config", json={"mode": "LIMIT"})

    assert response.status_code == 200
    assert response.json()["mode"] == "LIMIT"
    assert test_client.get("/config").json()["rules"]["mode"] == "LIMIT"


def test_config_always_exposes_the_three_rule_fields(client):
    """The UI reads these on load, so they must always be present."""
    test_client, _ = client
    rules = test_client.get("/config").json()["rules"]

    assert rules["mode"] in ("TIMER", "INSTANT", "LIMIT")
    assert isinstance(rules["dwell_seconds"], float)
    assert isinstance(rules["repeat_alert_seconds"], float)


def test_a_partial_config_update_keeps_the_other_rules(client):
    test_client, _ = client
    test_client.post("/config", json={"mode": "INSTANT", "dwell_seconds": 2.5})

    # sending only the polygon must not reset the rules
    test_client.post("/config", json={"points": [[1, 1], [2, 1], [2, 2]]})

    rules = test_client.get("/config").json()["rules"]
    assert rules["mode"] == "INSTANT"
    assert rules["dwell_seconds"] == 2.5


def test_a_repeat_of_zero_saves_instead_of_being_rejected(client):
    """The settings card offers 0 for "don't repeat the alert", so the API has
    to accept it and store it as a real value."""
    test_client, _ = client
    response = test_client.post(
        "/config",
        json={"mode": "TIMER", "dwell_seconds": 2.0, "repeat_alert_seconds": 0},
    )

    assert response.status_code == 200
    assert response.json()["repeat_alert_seconds"] == 0
    assert test_client.get("/config").json()["rules"]["repeat_alert_seconds"] == 0.0


def test_a_dwell_of_zero_saves_instead_of_being_rejected(client):
    test_client, _ = client
    response = test_client.post("/config", json={"dwell_seconds": 0})

    assert response.status_code == 200
    assert response.json()["dwell_seconds"] == 0
    assert test_client.get("/config").json()["rules"]["dwell_seconds"] == 0.0


@pytest.mark.parametrize("field", ["dwell_seconds", "repeat_alert_seconds"])
def test_a_negative_wait_is_rejected(client, field):
    """A negative wait is a typo, not a setting: -1 would repeat immediately."""
    test_client, _ = client
    response = test_client.post("/config", json={field: -1})
    assert response.status_code == 422


def test_an_unknown_mode_is_rejected(client):
    test_client, _ = client
    response = test_client.post("/config", json={"mode": "SOMETIMES"})
    assert response.status_code == 422


def test_start_forwards_the_rules_to_the_pipeline(client, monkeypatch):
    test_client, _ = client
    test_client.post("/config", json={"mode": "INSTANT", "dwell_seconds": 3, "repeat_alert_seconds": 42})

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

    assert captured["mode"] == "INSTANT"
    assert captured["dwell_seconds"] == 3.0
    assert captured["repeat_alert_seconds"] == 42.0


def test_saving_an_empty_polygon_is_allowed_so_preview_works(client):
    """Clearing the zone must be possible - the video preview still runs."""
    test_client, _ = client
    test_client.post("/config", json={"points": [[1, 1], [2, 1], [2, 2]], "allowed_people": 3})
    assert test_client.post("/config", json={"points": [], "allowed_people": 0}).json()["saved"] is True
    assert test_client.get("/config").json()["zone_points"] == []


def test_negative_allowed_people_is_rejected(client):
    test_client, _ = client
    response = test_client.post("/config", json={"points": [[1, 1], [2, 1], [2, 2]], "allowed_people": -1})
    assert response.status_code == 422


def test_state_before_anything_is_started(client):
    test_client, _ = client
    body = test_client.get("/state").json()

    assert body["running"] is False
    assert body["status"] == "SAFE"
    assert body["inside"] == 0
    assert body["workers"] == []


def test_start_requires_at_least_three_points(client):
    test_client, _ = client
    response = test_client.post("/start", json={"points": [[1, 1], [2, 2]]})
    assert response.status_code == 400
    assert "3 points" in response.json()["detail"]


def test_start_reports_a_bad_source(client, tmp_path):
    test_client, _ = client
    response = test_client.post(
        "/start",
        json={"points": [[1, 1], [2, 1], [2, 2]], "source": "definitely_not_a_video.mp4"},
    )
    assert response.status_code == 400
    assert "not found" in response.json()["detail"]


def test_state_names_a_saved_video_file_that_is_gone(client, tmp_path):
    # a source saved earlier whose file no longer exists (never cloned, or
    # deleted since) is named in /state, so the stopped panel can say which
    # file is missing instead of showing a blank camera
    test_client, _ = client
    gone = str(tmp_path / "gone.mp4")
    test_client.post("/config", json={"source": gone})
    body = test_client.get("/state").json()
    assert body["running"] is False
    assert body["source_error"] == f"Video file not found at {gone}"


def test_state_names_a_missing_test_video_source(client, tmp_path, monkeypatch):
    import api.server as server_module

    test_client, _ = client
    monkeypatch.setattr(server_module, "VIDEO_DIR", str(tmp_path))
    gone = str(tmp_path / "clip.mp4")
    test_client.post("/config", json={"source": gone})
    body = test_client.get("/state").json()
    assert body["source_error"] == f"Test video not found at {gone}"


def test_state_has_no_source_error_for_a_camera_or_an_existing_file(client, tmp_path):
    test_client, _ = client
    test_client.post("/config", json={"source": "0"})
    assert test_client.get("/state").json()["source_error"] is None

    present = tmp_path / "here.mp4"
    present.write_bytes(b"fake")
    test_client.post("/config", json={"source": str(present)})
    assert test_client.get("/state").json()["source_error"] is None


def test_stop_without_start_is_harmless(client):
    test_client, _ = client
    assert test_client.post("/stop").json() == {"running": False}


def test_start_restarts_instead_of_failing_when_already_running(client, monkeypatch):
    """Applying a new zone must take effect, not raise 'already running'."""
    test_client, _ = client
    started = []

    class FakePipeline:
        def __init__(self):
            self.stops = 0

        def stop(self):
            self.stops += 1

        def start(self, **kwargs):
            started.append(kwargs)

    fake = FakePipeline()
    monkeypatch.setattr(server, "_pipeline", fake)
    points = [[1, 1], [2, 1], [2, 2]]

    first = test_client.post("/start", json={"points": points, "allowed_people": 0})
    second = test_client.post("/start", json={"points": points, "allowed_people": 4})

    assert first.status_code == 200
    assert second.status_code == 200
    assert fake.stops == 2
    assert started[0]["allowed_people"] == 0
    assert started[1]["allowed_people"] == 4


def test_explicit_empty_points_clears_the_zone_on_start(client, monkeypatch):
    """points: [] means 'zone cleared' and must not fall back to the saved zone."""
    test_client, _ = client
    test_client.post("/config", json={"points": [[10, 10], [60, 10], [60, 60]], "allowed_people": 0})

    started = {}

    class FakePipeline:
        def stop(self):
            pass

        def start(self, **kwargs):
            started.update(kwargs)

    monkeypatch.setattr(server, "_pipeline", FakePipeline())
    test_client.post("/start", json={"points": [], "allowed_people": 0})

    assert started["zones"] == []


def test_removing_every_zone_keeps_monitoring_with_no_zone(client, monkeypatch):
    """'Remove all zones' saves []. The camera must keep running and boxing
    people, it simply has no danger zone left to alert on."""
    test_client, _ = client
    saved = {
        "zones": [
            [[10, 10], [200, 10], [200, 200]],
            [[300, 10], [500, 10], [500, 200]],
        ],
        "allowed_people": 0,
        "frame_width": 640,
        "frame_height": 480,
    }
    test_client.post("/config", json=saved)

    started = {}

    class FakePipeline:
        def stop(self):
            pass

        def start(self, **kwargs):
            started.update(kwargs)

    monkeypatch.setattr(server, "_pipeline", FakePipeline())
    response = test_client.post("/start", json={**saved, "zones": []})

    assert response.status_code == 200
    assert started["zones"] == []


def test_saving_an_empty_list_forgets_the_previously_saved_zones(client):
    """The pair the UI sends is save-then-start, so /config has to record the
    emptiness, otherwise the old zones return on the next restart."""
    test_client, _ = client
    saved = {
        "zones": [
            [[10, 10], [200, 10], [200, 200]],
            [[300, 10], [500, 10], [500, 200]],
        ],
        "allowed_people": 0,
        "frame_width": 640,
        "frame_height": 480,
    }
    test_client.post("/config", json=saved)
    assert len(test_client.get("/config").json()["zones"]) == 2

    response = test_client.post("/config", json={**saved, "zones": []})

    assert response.status_code == 200
    assert test_client.get("/config").json()["zones"] == []


def test_startup_after_removing_every_zone_runs_without_a_zone(client, monkeypatch):
    """A config.yaml that has no zones at all must not be treated as a load
    failure: the app is simply monitoring with nobody in a danger zone."""
    test_client, _ = client
    test_client.post("/config", json={"zones": [], "allowed_people": 0})

    assert test_client.get("/config").json()["zones"] == []


def test_start_omitted_fields_fall_back_to_saved_settings(client, monkeypatch):
    test_client, _ = client
    test_client.post("/config", json={"points": [[10, 10], [60, 10], [60, 60]], "allowed_people": 3})

    started = {}

    class FakePipeline:
        def stop(self):
            pass

        def start(self, **kwargs):
            started.update(kwargs)

    monkeypatch.setattr(server, "_pipeline", FakePipeline())
    test_client.post("/start", json={})

    assert started["zones"] == [[[10, 10], [60, 10], [60, 60]]]
    assert started["allowed_people"] == 3


# /stream is an endless MJPEG response, so it is not unit tested here.
# It is verified against a real uvicorn server in the manual check.


# --- /snapshot: the still frame the browser draws the zone on ----------------


class StubPipeline:
    def __init__(self, jpeg, frame_width=640, frame_height=480):
        self._jpeg = jpeg
        self._state = {
            "frame_width": frame_width,
            "frame_height": frame_height,
            "frame_seq": 3,
        }

    def jpeg(self):
        return self._jpeg

    def state(self):
        return dict(self._state)


def test_snapshot_returns_the_latest_jpeg_with_the_frame_size(client, monkeypatch):
    test_client, _ = client
    monkeypatch.setattr(server, "_pipeline", StubPipeline(b"\xff\xd8jpegdata\xff\xd9", 1024, 768))

    response = test_client.get("/snapshot")

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert response.content == b"\xff\xd8jpegdata\xff\xd9"
    # the UI needs the real frame size to scale the SVG overlay correctly
    assert response.headers["x-frame-width"] == "1024"
    assert response.headers["x-frame-height"] == "768"
    assert response.headers["cache-control"] == "no-store"


def test_snapshot_reports_no_video_yet(client, monkeypatch):
    test_client, _ = client
    monkeypatch.setattr(server, "_pipeline", StubPipeline(None))

    response = test_client.get("/snapshot")

    assert response.status_code == 404
    assert response.json()["detail"] == "no video yet"


# --- #5 a partial POST /config must not reset the other fields --------------

def test_partial_config_keeps_the_saved_allowed_people(client):
    """Saving only the polygon must not silently reset allowed_people."""
    test_client, db_path = client
    polygon = [[10, 10], [60, 10], [60, 60]]
    test_client.post("/config", json={"points": polygon, "allowed_people": 4})

    test_client.post("/config", json={"points": [[5, 5], [55, 5], [55, 55]]})

    reloaded = test_client.get("/config").json()
    assert reloaded["allowed_people"] == 4
    assert reloaded["zone_points"] == [[5, 5], [55, 5], [55, 55]]
    assert Database(db_path).all_settings()["allowed_people"] == "4"


def test_partial_config_keeps_the_saved_polygon(client):
    test_client, _ = client
    polygon = [[10, 10], [60, 10], [60, 60]]
    test_client.post("/config", json={"points": polygon, "allowed_people": 1})

    test_client.post("/config", json={"allowed_people": 2})

    body = test_client.get("/config").json()
    assert body["zone_points"] == polygon
    assert body["allowed_people"] == 2


def test_saving_only_allowed_people_leaves_the_zone_alone(client):
    test_client, _ = client
    polygon = [[10, 10], [60, 10], [60, 60]]
    test_client.post("/config", json={"points": polygon, "allowed_people": 0})

    test_client.post("/config", json={"allowed_people": 3})

    assert test_client.get("/config").json()["zone_points"] == polygon


def test_empty_points_still_clears_the_zone_on_a_partial_save(client):
    """An explicit [] must clear, while an omitted points must not."""
    test_client, _ = client
    polygon = [[10, 10], [60, 10], [60, 60]]
    test_client.post("/config", json={"points": polygon, "allowed_people": 2})

    test_client.post("/config", json={"points": [], "allowed_people": 2})

    body = test_client.get("/config").json()
    assert body["zone_points"] == []
    assert body["allowed_people"] == 2


# --- #6 the machine name must be saved and read back -------------------------

def test_machine_name_is_saved_and_reloaded(client):
    test_client, db_path = client
    points = [[10, 10], [60, 10], [60, 60]]

    test_client.post("/config", json={"points": points, "machine_name": "JCB Loader 42"})

    assert Database(db_path).all_settings()["machine_name"] == "JCB Loader 42"
    assert test_client.get("/config").json()["machine"]["name"] == "JCB Loader 42"


def test_machine_name_falls_back_to_the_yaml_when_never_saved(client):
    test_client, _ = client
    assert test_client.get("/config").json()["machine"]["name"] == "Excavator 01"


def test_machine_name_is_used_by_the_running_pipeline(client, monkeypatch):
    """The name must survive into a restart, not just into the settings table."""
    test_client, _ = client
    points = [[10, 10], [60, 10], [60, 60]]
    test_client.post("/config", json={"points": points, "allowed_people": 0, "machine_name": "Site B"})

    started = {}

    class FakePipeline:
        def stop(self):
            pass

        def start(self, **kwargs):
            started.update(kwargs)

    monkeypatch.setattr(server, "_pipeline", FakePipeline())
    test_client.post("/start", json={})

    assert started["machine"]["name"] == "Site B"


def test_saved_machine_name_survives_a_process_restart(client):
    """A new server process reads the name from sqlite, not from the yaml."""
    test_client, db_path = client
    points = [[10, 10], [60, 10], [60, 60]]
    test_client.post("/config", json={"points": points, "allowed_people": 0, "machine_name": "Rig 7"})

    # a fresh client, as if the app had been restarted
    restarted = TestClient(fastapi_app)

    assert restarted.get("/config").json()["machine"]["name"] == "Rig 7"


# --- the alert flags must be part of the state the UI reads -----------------

def test_state_before_anything_is_started_includes_the_alert_flags(client):
    test_client, _ = client
    body = test_client.get("/state").json()

    assert body["capacity_alert"] is False
    assert body["worker_alert"] is False


# --- named zones and their own rules ----------------------------------------

ZONE_LEFT = [[100, 100], [400, 100], [400, 300], [100, 300]]
ZONE_RIGHT = [[500, 100], [700, 100], [700, 300], [500, 300]]
# a zone whose name is asked for in an error message
ZONE_SMALL = [[10, 10], [60, 10], [60, 60]]


def save_named_zones(client, specs, frame=True):
    test_client, _ = client
    body = {"zones": [ZONE_LEFT, ZONE_RIGHT], "zone_specs": specs}
    if frame:
        body["frame_width"] = 800
        body["frame_height"] = 600
    return test_client.post("/config", json=body)


def test_two_zones_keep_their_own_names_allowances_and_dwell_limits(client):
    test_client, db_path = client

    save_named_zones(
        client,
        [
            {"name": "Kiln Platform", "allowed_people": 2, "dwell_seconds": 3.0},
            {"name": "Loading Bay", "allowed_people": 0, "dwell_seconds": 8.5},
        ],
    )

    specs = test_client.get("/config").json()["zone_specs"]
    assert [spec["name"] for spec in specs] == ["Kiln Platform", "Loading Bay"]
    assert [spec["allowed_people"] for spec in specs] == [2, 0]
    assert [spec["dwell_seconds"] for spec in specs] == [3.0, 8.5]
    assert all(spec["enabled"] for spec in specs)

    # the shapes are still in their own key, and the rules in theirs
    stored = Database(db_path).all_settings()
    assert json.loads(stored["zone_polygons"]) == [ZONE_LEFT, ZONE_RIGHT]
    assert len(json.loads(stored["zone_specs"])) == 2


def test_the_rules_are_stored_next_to_the_shapes_and_not_inside_them(client):
    """One copy of the geometry: the polygons keep the exact shape they had."""
    test_client, db_path = client

    save_named_zones(
        client,
        [{"name": "Kiln Platform", "allowed_people": 2}, {"name": "Loading Bay"}],
    )

    stored = Database(db_path).all_settings()
    for polygon, spec in zip(json.loads(stored["zone_polygons"]), json.loads(stored["zone_specs"])):
        # no name, no allowance and no on/off flag smuggled into the geometry
        assert all(len(point) == 2 for point in polygon)
        assert set(spec) == {"name", "allowed_people", "dwell_seconds", "enabled", "using_defaults"}


def test_a_zone_saved_without_a_name_becomes_zone_1(client):
    test_client, _ = client
    test_client.post("/config", json={"zones": [ZONE_LEFT]})

    specs = test_client.get("/config").json()["zone_specs"]
    assert specs[0]["name"] == "Zone 1"


def test_a_database_from_before_named_zones_opens_as_one_working_zone(client):
    """Nothing to migrate: the old keys still produce a usable single zone."""
    test_client, db_path = client
    database = Database(db_path)
    database.set_setting("zone_polygons", json.dumps([ZONE_LEFT]))
    database.set_setting("allowed_people", "3")
    database.set_setting("dwell_seconds", "5.0")

    body = test_client.get("/config").json()

    assert body["zones"] == [ZONE_LEFT]
    assert len(body["zone_specs"]) == 1
    spec = body["zone_specs"][0]
    assert spec["name"] == "Zone 1"
    assert spec["allowed_people"] == 3, "it follows the saved global allowance"
    assert spec["dwell_seconds"] == 5.0
    assert spec["using_defaults"] is True


def test_a_zone_with_no_rule_of_its_own_keeps_following_the_defaults(client):
    test_client, _ = client
    test_client.post(
        "/config",
        json={
            "zones": [ZONE_LEFT, ZONE_RIGHT],
            "zone_specs": [
                {"name": "Own Rules", "allowed_people": 4, "dwell_seconds": 2.0},
                {"name": "Borrowed"},
            ],
            "allowed_people": 1,
            "dwell_seconds": 6.0,
        },
    )

    specs = test_client.get("/config").json()["zone_specs"]
    assert specs[0]["using_defaults"] is False
    assert specs[0]["allowed_people"] == 4
    assert specs[1]["using_defaults"] is True
    assert specs[1]["allowed_people"] == 1, "the second zone uses the global number"
    assert specs[1]["dwell_seconds"] == 6.0


def test_renaming_one_zone_leaves_the_other_zones_rules_alone(client):
    test_client, _ = client
    save_named_zones(
        client,
        [
            {"name": "Kiln Platform", "allowed_people": 2, "dwell_seconds": 3.0},
            {"name": "Loading Bay", "allowed_people": 1, "dwell_seconds": 4.0},
        ],
    )

    save_named_zones(
        client,
        [{"name": "Kiln Deck"}, {"name": "Loading Bay"}],
    )

    specs = test_client.get("/config").json()["zone_specs"]
    assert specs[0]["name"] == "Kiln Deck"
    assert specs[0]["allowed_people"] == 2, "an unsent field keeps what that zone had"
    assert specs[0]["dwell_seconds"] == 3.0
    assert specs[1]["name"] == "Loading Bay"
    assert specs[1]["allowed_people"] == 1


def test_a_zone_can_be_switched_off_and_back_on(client):
    test_client, _ = client
    save_named_zones(
        client,
        [{"name": "Kiln Platform", "enabled": False}, {"name": "Loading Bay"}],
    )
    assert test_client.get("/config").json()["zone_specs"][0]["enabled"] is False

    save_named_zones(
        client,
        [{"name": "Kiln Platform", "enabled": True}, {"name": "Loading Bay"}],
    )
    assert test_client.get("/config").json()["zone_specs"][0]["enabled"] is True


def test_a_bad_rule_names_the_zone_it_belongs_to(client):
    """With several cards on screen the user has to know which one to fix."""
    test_client, _ = client

    response = save_named_zones(
        client,
        [{"name": "Kiln Platform"}, {"name": "Loading Bay", "allowed_people": -3}],
    )

    assert response.status_code == 422
    assert "Loading Bay" in response.json()["detail"]


def test_a_negative_dwell_names_its_zone(client):
    test_client, _ = client

    response = save_named_zones(
        client, [{"name": "Kiln Platform"}, {"name": "Loading Bay", "dwell_seconds": -1}]
    )

    assert response.status_code == 422
    assert "Loading Bay" in response.json()["detail"]


def test_a_rule_that_is_not_a_number_names_its_zone(client):
    test_client, _ = client

    response = save_named_zones(
        client, [{"name": "Kiln Platform", "allowed_people": "lots"}]
    )

    assert response.status_code == 422
    assert "Kiln Platform" in response.json()["detail"]


def test_a_rule_for_a_zone_that_was_not_drawn_is_dropped(client):
    """A rule with no shape is not a zone anybody can stand in."""
    test_client, db_path = client
    test_client.post(
        "/config",
        json={"zones": [ZONE_LEFT], "zone_specs": [{"name": "Kiln Platform"}, {"name": "Ghost"}]},
    )

    specs = test_client.get("/config").json()["zone_specs"]
    assert [spec["name"] for spec in specs] == ["Kiln Platform"]
    assert len(json.loads(Database(db_path).all_settings()["zone_specs"])) == 1


def test_the_points_inside_a_rule_are_ignored(client):
    """The shapes live in zone_polygons only, so a second copy cannot drift."""
    test_client, db_path = client
    test_client.post(
        "/config",
        json={
            "zones": [ZONE_LEFT],
            "zone_specs": [{"name": "Kiln Platform", "points": ZONE_SMALL}],
            "frame_width": 800,
            "frame_height": 600,
        },
    )

    assert json.loads(Database(db_path).all_settings()["zone_polygons"]) == [ZONE_LEFT]


def test_the_old_single_zone_payload_still_produces_one_usable_zone(client):
    """The whole point of the read-time upgrade: an older UI is not broken."""
    test_client, _ = client
    test_client.post("/config", json={"points": ZONE_LEFT, "allowed_people": 2})

    body = test_client.get("/config").json()
    assert body["zone_points"] == ZONE_LEFT
    assert len(body["zone_specs"]) == 1
    assert body["zone_specs"][0]["allowed_people"] == 2


def test_state_lists_every_zone_while_nothing_is_running(client):
    """The UI has one shape to fill in, so a stopped dashboard still shows zones."""
    test_client, _ = client
    save_named_zones(
        client,
        [
            {"name": "Kiln Platform", "allowed_people": 2, "dwell_seconds": 3.0},
            {"name": "Loading Bay", "allowed_people": 0, "dwell_seconds": 8.5},
        ],
    )

    body = test_client.get("/state").json()

    reports = body["zone_reports"]
    assert [row["name"] for row in reports] == ["Kiln Platform", "Loading Bay"]
    assert [row["allowed_people"] for row in reports] == [2, 0]
    assert [row["inside"] for row in reports] == [0, 0]
    assert [row["alerting"] for row in reports] == [False, False]
    assert [row["dwell_seconds"] for row in reports] == [0.0, 0.0]
    # each row carries its own limit beside its own longest dwell, so the
    # dashboard never has to read one zone's number against another's allowance
    assert [row["dwell_limit"] for row in reports] == [3.0, 8.5]
    assert body["zone_specs"][1]["name"] == "Loading Bay"


def test_state_reports_the_workers_zone_and_the_zones_they_are_in(client, monkeypatch):
    """A worker in two zones is reported against one of them, and lists both."""
    test_client, _ = client

    from zone.danger_zone import ALERT, INSIDE, DangerZone, Presence, ZoneMonitor, ZoneRule

    # two OVERLAPPING zones, worker 1 standing in the middle of both
    monitor = ZoneMonitor(
        zone=DangerZone(
            [
                [(100, 100), (400, 100), (400, 300), (100, 300)],
                [(250, 100), (550, 100), (550, 300), (250, 300)],
            ]
        ),
        zone_rules=[
            ZoneRule(name="Kiln Platform", allowed_people=5, dwell_seconds=90.0),
            ZoneRule(name="Loading Bay", allowed_people=5, dwell_seconds=1.0),
        ],
    )
    worker = monitor.get(1)
    worker.point = (300, 200)
    worker.presences = {
        0: Presence(zone_index=0, entered_at=0.0, dwell_seconds=2.0, state=INSIDE),
        1: Presence(zone_index=1, entered_at=0.0, dwell_seconds=2.0, state=ALERT),
    }

    class StubPipeline:
        active = True

        def state(self):
            rows = monitor.summary()
            return {
                "running": True,
                "zone_reports": monitor.zone_reports(),
                "workers": rows,
            }

    monkeypatch.setattr(server, "_pipeline", StubPipeline())

    body = test_client.get("/state").json()

    row = next(r for r in body["workers"] if r["worker_id"] == 1)
    assert row["zone_index"] == 1
    assert row["zone_name"] == "Loading Bay"
    assert row["state"] == "ALERT"
    assert row["zones"] == [0, 1]


def test_start_forwards_every_zone_rule_to_the_pipeline(client, monkeypatch):
    """/start must not fall back to the global numbers for a named zone."""
    test_client, _ = client
    save_named_zones(
        client,
        [
            {"name": "Kiln Platform", "allowed_people": 2, "dwell_seconds": 3.0},
            {"name": "Loading Bay", "allowed_people": 0, "dwell_seconds": 8.5},
        ],
    )

    seen = {}

    class StubPipeline:
        active = False
        source = "0"

        def stop(self):
            return True

        def start(self, **kwargs):
            seen.update(kwargs)
            return True

    monkeypatch.setattr(server, "_pipeline", StubPipeline())
    test_client.post("/start", json={"source": "0"})

    specs = seen["zone_specs"]
    assert [spec["name"] for spec in specs] == ["Kiln Platform", "Loading Bay"]
    assert [spec["allowed_people"] for spec in specs] == [2, 0]
    assert [spec["dwell_seconds"] for spec in specs] == [3.0, 8.5]


def test_a_switched_off_zone_reaches_the_pipeline(client, monkeypatch):
    test_client, _ = client
    save_named_zones(
        client,
        [{"name": "Kiln Platform", "enabled": False}, {"name": "Loading Bay"}],
    )

    seen = {}

    class StubPipeline:
        active = False
        source = "0"

        def stop(self):
            return True

        def start(self, **kwargs):
            seen.update(kwargs)
            return True

    monkeypatch.setattr(server, "_pipeline", StubPipeline())
    test_client.post("/start", json={"source": "0"})

    assert [spec["enabled"] for spec in seen["zone_specs"]] == [False, True]


def test_changing_one_zones_rules_reaches_a_running_camera(client, monkeypatch):
    test_client, _ = client
    save_named_zones(
        client,
        [
            {"name": "Kiln Platform", "allowed_people": 2},
            {"name": "Loading Bay", "allowed_people": 1},
        ],
    )

    seen = {}

    class StubPipeline:
        active = True
        source = "0"

        def apply_settings(self, **kwargs):
            seen.update(kwargs)
            return []

    monkeypatch.setattr(server, "_pipeline", StubPipeline())
    save_named_zones(
        client,
        [
            {"name": "Kiln Platform", "allowed_people": 5},
            {"name": "Loading Bay"},
        ],
    )

    assert [spec["allowed_people"] for spec in seen["zone_specs"]] == [5, 1]


def test_start_without_zone_specs_falls_back_to_the_saved_rules(client, monkeypatch):
    test_client, _ = client
    save_named_zones(
        client,
        [
            {"name": "Kiln Platform", "allowed_people": 2, "dwell_seconds": 3.0},
            {"name": "Loading Bay", "allowed_people": 1, "dwell_seconds": 4.0},
        ],
    )

    seen = {}

    class StubPipeline:
        active = False
        source = "0"

        def stop(self):
            return True

        def start(self, **kwargs):
            seen.update(kwargs)
            return True

    monkeypatch.setattr(server, "_pipeline", StubPipeline())
    test_client.post("/start", json={"source": "0"})

    specs = seen["zone_specs"]
    assert [spec["name"] for spec in specs] == ["Kiln Platform", "Loading Bay"]
    assert [spec["allowed_people"] for spec in specs] == [2, 1]


# --- zones belong to the source they were drawn on --------------------------


def test_source_key_is_stable_and_hides_the_path():
    assert server.source_key("0") == "cam:0"
    assert server.source_key(" 3 ") == "cam:3"
    assert server.source_key("") == "cam:0"
    assert server.source_key("/data/videos/kiln.mp4") == "video:kiln.mp4"
    assert server.source_key(r"C:\data\videos\kiln.mp4") == "video:kiln.mp4"


def test_zones_are_kept_per_source_and_never_carried_over(client):
    test_client, db_path = client

    test_client.post(
        "/config",
        json={"zones": [ZONE_LEFT], "source": "0", "frame_width": 800, "frame_height": 600},
    )
    test_client.post(
        "/config",
        json={
            "zones": [ZONE_RIGHT],
            "source": "/data/videos/kiln.mp4",
            "frame_width": 1600,
            "frame_height": 1200,
        },
    )

    # the video is the active source now, so a plain read gives its zones
    assert test_client.get("/config").json()["zones"] == [ZONE_RIGHT]
    # asking for the camera gives the camera's own zones, not the video's
    assert test_client.get("/config", params={"source": "0"}).json()["zones"] == [ZONE_LEFT]

    stored = json.loads(Database(db_path).all_settings()[server.SOURCE_ZONES_KEY])
    assert stored["cam:0"]["polygons"] == [ZONE_LEFT]
    assert stored["cam:0"]["reference"] == [800, 600]
    assert stored["video:kiln.mp4"]["polygons"] == [ZONE_RIGHT]
    assert stored["video:kiln.mp4"]["reference"] == [1600, 1200]


def test_switching_source_without_zones_loads_that_sources_own_zones(client):
    test_client, _ = client
    test_client.post("/config", json={"zones": [ZONE_LEFT], "source": "0"})
    test_client.post("/config", json={"zones": [ZONE_RIGHT], "source": "clip.mp4"})

    # back to the camera, sending only the source: the video's zones must not
    # be left in the mirror
    test_client.post("/config", json={"source": "0"})

    body = test_client.get("/config").json()
    assert body["source"] == "0"
    assert body["zones"] == [ZONE_LEFT]
    # and the video's zones are still there under their own source
    assert test_client.get("/config", params={"source": "clip.mp4"}).json()["zones"] == [ZONE_RIGHT]


def test_migration_adopts_the_old_global_zones_once(client):
    test_client, db_path = client
    database = Database(db_path)
    database.set_setting("zone_polygons", json.dumps([ZONE_LEFT]))
    database.set_setting("zone_reference", json.dumps([800, 600]))
    database.set_setting("source", "0")

    server.migrate_zones_to_sources()

    stored = json.loads(database.all_settings()[server.SOURCE_ZONES_KEY])
    assert stored["cam:0"]["polygons"] == [ZONE_LEFT]
    assert stored["cam:0"]["reference"] == [800, 600]
    assert database.get_setting(server.ZONES_MIGRATED_KEY) == "1"

    # running it again must not clobber a later save
    database.set_setting("zone_polygons", json.dumps([ZONE_RIGHT]))
    server.migrate_zones_to_sources()
    stored = json.loads(database.all_settings()[server.SOURCE_ZONES_KEY])
    assert stored["cam:0"]["polygons"] == [ZONE_LEFT]


# --- a settings-only save must not be at the mercy of the live frame --------
#
# The pipeline never switches source without a restart, so while the camera
# runs at 640x480 a save that carries no shapes of its own - the debounced
# settings save and the one that only switches the Camera source field - would
# clamp a video's 1920-wide zone into the camera's picture if it were checked
# against _frame_size_for. It is checked against the reference the target
# source's own zones were filed under instead.


class RunningCameraAt640x480:
    """The picture on screen: 640x480, whatever any stored zone was drawn on."""

    active = True
    source = "0"
    frame_size = (640, 480)

    def apply_settings(self, **kwargs):
        return []

    def state(self):
        return {}


def source_entry(db_path, source):
    """The stored polygons and reference of one source, or None if never saved."""
    raw = Database(db_path).all_settings().get(server.SOURCE_ZONES_KEY)
    entries = json.loads(raw) if raw else {}
    entry = entries.get(server.source_key(source))
    if not isinstance(entry, dict):
        return None
    return {"polygons": entry.get("polygons"), "reference": entry.get("reference")}


# drawn on the video's own 1920x1080 frame: two corners are off a 640x480 one
VIDEO_ZONE = [[100, 100], [1800, 100], [1800, 900], [100, 900]]
# drawn on the camera's 1280x720 frame: also off a 640x480 one
LIVE_ZONE_BIG = [[100, 100], [900, 100], [900, 600], [100, 600]]
# drawn on a 640x480 frame, so it fits the running camera exactly
LIVE_ZONE = [[100, 100], [600, 100], [600, 400], [100, 400]]
CLIP = "/data/videos/line.mp4"


def test_switching_back_and_forth_never_rewrites_either_sources_stored_zone(
    client, monkeypatch
):
    """The round trip: video -> live -> video -> live, zone for zone.

    After every step BOTH sources' stored polygons and reference must be
    exactly what they were, because every one of those switches is a
    settings-only save validated against its own source's reference.
    """
    test_client, db_path = client
    monkeypatch.setattr(server, "_pipeline", RunningCameraAt640x480())

    # 1. the video keeps its own zone, drawn on its own 1920x1080 frame
    response = test_client.post(
        "/config",
        json={
            "zones": [VIDEO_ZONE],
            "source": CLIP,
            "frame_width": 1920,
            "frame_height": 1080,
        },
    )
    assert response.status_code == 200
    video_snapshot = source_entry(db_path, CLIP)
    assert video_snapshot == {"polygons": [VIDEO_ZONE], "reference": [1920, 1080]}

    # 2. switch to the live camera: the video's entry is untouched
    response = test_client.post("/config", json={"source": "0"})
    assert response.status_code == 200
    assert source_entry(db_path, CLIP) == video_snapshot

    # 3. the live camera gets its own zone, drawn on 640x480
    response = test_client.post(
        "/config",
        json={"zones": [LIVE_ZONE], "source": "0", "frame_width": 640, "frame_height": 480},
    )
    assert response.status_code == 200
    live_snapshot = source_entry(db_path, "0")
    assert live_snapshot == {"polygons": [LIVE_ZONE], "reference": [640, 480]}
    assert source_entry(db_path, CLIP) == video_snapshot

    # 4. back to the video: both entries stand exactly as they are
    response = test_client.post("/config", json={"source": CLIP})
    assert response.status_code == 200
    assert source_entry(db_path, CLIP) == video_snapshot
    assert source_entry(db_path, "0") == live_snapshot

    # 5. and back to the live camera again
    response = test_client.post("/config", json={"source": "0"})
    assert response.status_code == 200
    assert source_entry(db_path, CLIP) == video_snapshot
    assert source_entry(db_path, "0") == live_snapshot


def test_a_source_only_save_never_touches_a_stored_zone_or_reference(client, monkeypatch):
    """POST /config {source} while the running camera shows a different frame.

    The pipeline sits at 640x480 the whole time; the stored zones belong to
    1280x720 and 1920x1080. A source-only save must change no polygon and no
    reference of either source - and must not be refused by a clamp against
    the picture that happens to be playing.
    """
    test_client, db_path = client
    monkeypatch.setattr(server, "_pipeline", RunningCameraAt640x480())

    test_client.post(
        "/config",
        json={
            "zones": [LIVE_ZONE_BIG],
            "source": "0",
            "frame_width": 1280,
            "frame_height": 720,
        },
    )
    test_client.post(
        "/config",
        json={
            "zones": [VIDEO_ZONE],
            "source": CLIP,
            "frame_width": 1920,
            "frame_height": 1080,
        },
    )
    live_snapshot = source_entry(db_path, "0")
    video_snapshot = source_entry(db_path, CLIP)
    assert live_snapshot == {"polygons": [LIVE_ZONE_BIG], "reference": [1280, 720]}
    assert video_snapshot == {"polygons": [VIDEO_ZONE], "reference": [1920, 1080]}

    # switch to the video: 640x480 is what is on screen, 1920x1080 is what
    # the video's zone belongs to - only the second one may be used
    response = test_client.post("/config", json={"source": CLIP})
    assert response.status_code == 200
    assert source_entry(db_path, CLIP) == video_snapshot
    assert source_entry(db_path, "0") == live_snapshot

    # and back to the live camera, whose own zone belongs to 1280x720
    response = test_client.post("/config", json={"source": "0"})
    assert response.status_code == 200
    assert source_entry(db_path, CLIP) == video_snapshot
    assert source_entry(db_path, "0") == live_snapshot
