"""Tests for uploaded test videos: /videos upload, list, delete and selection.

The uploads are written with OpenCV into a tmp folder, so these tests never
touch the real data/videos/ and the metadata they assert on is read back from a
real decodable file.
"""

import json
import os

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

import api.server as server
from api.server import app as fastapi_app


def make_video(path, width=64, height=48, frames=5, fps=10):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (width, height))
    for index in range(frames):
        frame = np.full((height, width, 3), (index * 20) % 255, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return str(path)


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
    video_dir = tmp_path / "videos"
    monkeypatch.setattr(server, "DB_PATH", db_path)
    monkeypatch.setattr(server, "CONFIG_PATH", str(config_path))
    monkeypatch.setattr(server, "SNAPSHOT_PATH", str(tmp_path / "workers.json"))
    monkeypatch.setattr(server, "VIDEO_DIR", str(video_dir))
    monkeypatch.setattr(server, "_pipeline", None)
    return TestClient(fastapi_app), str(video_dir), tmp_path


def upload(test_client, data, filename="clip.avi", content_type="video/x-msvideo"):
    return test_client.post("/videos", files={"file": (filename, data, content_type)})


def read(path):
    with open(path, "rb") as handle:
        return handle.read()


# --- upload -----------------------------------------------------------------


def test_a_valid_video_is_saved_with_facts_read_from_the_file(client):
    test_client, video_dir, tmp_path = client
    source = make_video(tmp_path / "source.avi", width=64, height=48, frames=5, fps=10)

    response = upload(test_client, read(source), filename="kiln run.avi")

    assert response.status_code == 200
    body = response.json()
    # the display name is the browser's, the path is the backend's own uuid
    assert body["name"] == "kiln run.avi"
    assert body["width"] == 64
    assert body["height"] == 48
    assert body["fps"] == pytest.approx(10.0)
    assert body["duration"] == pytest.approx(0.5)
    assert body["size_mb"] > 0
    assert os.path.dirname(body["source"]) == os.path.abspath(video_dir)
    assert os.path.isfile(body["source"])
    # a sidecar sits next to it, and no database table was needed
    assert os.path.isfile(os.path.join(video_dir, f"{body['id']}.json"))


def test_a_wrong_extension_is_rejected(client):
    test_client, video_dir, _ = client
    response = upload(test_client, b"not a video", filename="notes.txt")

    assert response.status_code == 415
    assert not os.path.isdir(video_dir) or os.listdir(video_dir) == []


def test_an_oversize_upload_is_rejected(client, monkeypatch):
    test_client, video_dir, tmp_path = client
    source = make_video(tmp_path / "big.avi", frames=30)
    monkeypatch.setattr(server, "MAX_VIDEO_BYTES", 200)

    response = upload(test_client, read(source))

    assert response.status_code == 413
    # the half-written file is removed, not left behind to be listed
    assert not os.path.isdir(video_dir) or os.listdir(video_dir) == []


def test_a_mp4_name_that_is_not_a_video_is_rejected_and_removed(client):
    test_client, video_dir, _ = client
    response = upload(test_client, b"definitely not mp4 bytes", filename="broken.mp4", content_type="video/mp4")

    assert response.status_code == 422
    assert "could not be read" in response.json()["detail"]
    assert not os.path.isdir(video_dir) or os.listdir(video_dir) == []


def test_an_uploaded_name_cannot_escape_the_video_folder(client):
    test_client, video_dir, tmp_path = client
    source = make_video(tmp_path / "source.avi")

    body = upload(test_client, read(source), filename="../../escape.avi").json()

    # only the basename is kept for display, and the file itself is inside the folder
    assert body["name"] == "escape.avi"
    assert os.path.dirname(body["source"]) == os.path.abspath(video_dir)
    assert not (tmp_path / "escape.avi").exists()
    assert not (tmp_path.parent / "escape.avi").exists()


# --- list and delete --------------------------------------------------------


def test_listing_returns_the_uploaded_videos(client):
    test_client, _, tmp_path = client
    upload(test_client, read(make_video(tmp_path / "a.avi")), filename="first.avi")
    upload(test_client, read(make_video(tmp_path / "b.avi")), filename="second.avi")

    body = test_client.get("/videos").json()

    names = {item["name"] for item in body["videos"]}
    assert names == {"first.avi", "second.avi"}
    assert body["limit"] == server.MAX_VIDEOS
    assert body["max_mb"] == server.MAX_VIDEO_BYTES // (1024 * 1024)
    assert all(item["source"] for item in body["videos"])


def test_deleting_a_video_removes_both_files_and_listing_forgets_it(client):
    test_client, video_dir, tmp_path = client
    body = upload(test_client, read(make_video(tmp_path / "a.avi")), filename="first.avi").json()

    response = test_client.delete(f"/videos/{body['id']}")

    assert response.status_code == 200
    assert test_client.get("/videos").json()["videos"] == []
    assert os.listdir(video_dir) == []


def test_deleting_a_video_forgets_the_zones_drawn_on_it(client):
    test_client, _, tmp_path = client
    body = upload(test_client, read(make_video(tmp_path / "a.avi")), filename="first.avi").json()

    test_client.post(
        "/config",
        json={"zones": [[[100, 100], [400, 100], [400, 300], [100, 300]]], "source": body["source"]},
    )
    assert server.source_key(body["file"]) in json.loads(
        server.read_db().all_settings()[server.SOURCE_ZONES_KEY]
    )

    test_client.delete(f"/videos/{body['id']}")

    stored = json.loads(server.read_db().all_settings().get(server.SOURCE_ZONES_KEY, "{}"))
    assert server.source_key(body["file"]) not in stored


def test_deleting_an_unknown_video_is_a_404(client):
    test_client, _, _ = client
    assert test_client.delete("/videos/" + "0" * 32).status_code == 404
    assert test_client.delete("/videos/not-an-id").status_code == 404


def test_a_video_that_is_the_active_source_cannot_be_deleted(client, monkeypatch):
    test_client, _, tmp_path = client
    body = upload(test_client, read(make_video(tmp_path / "a.avi")), filename="first.avi").json()

    class RunningPipeline:
        active = True
        source = body["source"]

    monkeypatch.setattr(server, "_pipeline", RunningPipeline())

    response = test_client.delete(f"/videos/{body['id']}")
    assert response.status_code == 409
    assert test_client.get("/videos").json()["videos"]

    # and once it is not the active source, deleting works again
    monkeypatch.setattr(server, "_pipeline", None)
    assert test_client.delete(f"/videos/{body['id']}").status_code == 200


def test_the_video_limit_is_reported_not_worked_around(client):
    test_client, _, tmp_path = client
    for index in range(server.MAX_VIDEOS):
        response = upload(
            test_client, read(make_video(tmp_path / f"clip{index}.avi")), filename=f"clip{index}.avi"
        )
        assert response.status_code == 200

    extra = upload(test_client, read(make_video(tmp_path / "extra.avi")), filename="extra.avi")

    assert extra.status_code == 409
    assert "limit" in extra.json()["detail"]
    # nothing was silently evicted to make room
    assert len(test_client.get("/videos").json()["videos"]) == server.MAX_VIDEOS


# --- selecting a video reuses the existing source setting -------------------


def test_selecting_a_video_sets_the_source_the_camera_field_already_uses(client):
    test_client, _, tmp_path = client
    body = upload(test_client, read(make_video(tmp_path / "a.avi")), filename="first.avi").json()

    # exactly what the Camera source field does: one /config save
    response = test_client.post("/config", json={"source": body["source"]})

    assert response.status_code == 200
    assert test_client.get("/config").json()["source"] == body["source"]
    state = test_client.get("/state").json()
    assert state["source"] == body["source"]
    assert state["test_video"] is True


def test_a_stored_file_that_is_not_an_upload_is_not_a_test_video(client, tmp_path):
    test_client, _, _ = client
    other = make_video(tmp_path / "elsewhere.avi")

    test_client.post("/config", json={"source": other})

    state = test_client.get("/state").json()
    assert state["source"] == other
    assert state["test_video"] is False


# --- loop behaviour ---------------------------------------------------------


def start_and_capture_loop(test_client, monkeypatch, source, payload=None):
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
    body = {"source": source}
    if payload:
        body.update(payload)
    test_client.post("/start", json=body)
    return seen.get("loop")


def test_an_uploaded_test_video_plays_once_by_default(client, monkeypatch):
    test_client, _, tmp_path = client
    body = upload(test_client, read(make_video(tmp_path / "a.avi")), filename="first.avi").json()

    # default off: a recorded run stops rather than silently starting over
    assert start_and_capture_loop(test_client, monkeypatch, body["source"]) is False


def test_a_test_video_loops_once_the_box_is_ticked(client, monkeypatch):
    test_client, _, tmp_path = client
    body = upload(test_client, read(make_video(tmp_path / "a.avi")), filename="first.avi").json()

    test_client.post("/config", json={"loop_video": True})

    assert start_and_capture_loop(test_client, monkeypatch, body["source"]) is True


def test_any_other_file_source_plays_once_by_default(client, monkeypatch, tmp_path):
    test_client, _, _ = client
    other = make_video(tmp_path / "recorded.avi")

    assert start_and_capture_loop(test_client, monkeypatch, other) is False


def test_a_webcam_source_keeps_its_old_behaviour(client, monkeypatch):
    test_client, _, _ = client
    # a camera index never loops, and the pipeline's is_camera check ignores the
    # flag anyway; what matters is that the webcam path is untouched
    assert start_and_capture_loop(test_client, monkeypatch, "0") is False


def test_an_explicit_loop_setting_wins(client, monkeypatch, tmp_path):
    test_client, _, _ = client
    other = make_video(tmp_path / "recorded.avi")

    assert start_and_capture_loop(test_client, monkeypatch, other, {"loop": True}) is True