"""Starting the camera with no danger zone drawn yet.

Step 1 of the UI has to open the video before anybody has marked a zone, so
these tests run the real pipeline loop over a tiny generated clip with a stub
detector: people must still be boxed, and nothing may alert.
"""

import time
from dataclasses import dataclass

import cv2
import numpy as np
import pytest

from database.db import Database
from pipeline import Pipeline

WIDTH, HEIGHT = 320, 180


@dataclass
class Person:
    """The little bit of PersonTracker.track the pipeline actually uses."""

    worker_id: int
    bbox: tuple
    confidence: float


class StubTracker:
    """Reports one person in the middle of every frame."""

    def __init__(self, *_args, **_kwargs):
        self.frames = 0

    def track(self, frame, frame_index):
        self.frames += 1
        return [
            Person(
                worker_id=1,
                bbox=(WIDTH // 2 - 30, HEIGHT // 2 - 60, 60, 120),
                confidence=0.9,
            )
        ]


def write_clip(path, frames=30):
    """A short grey clip, so the pipeline has something to open."""
    video = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (WIDTH, HEIGHT))
    assert video.isOpened(), "cv2 could not write a test clip"
    for _ in range(frames):
        video.write(np.full((HEIGHT, WIDTH, 3), 40, dtype=np.uint8))
    video.release()
    return str(path)


@pytest.fixture()
def pipeline(tmp_path, monkeypatch):
    monkeypatch.setattr("pipeline.PersonTracker", StubTracker)
    pipe = Pipeline(Database(str(tmp_path / "startup.db")))
    yield pipe
    pipe.stop()


def wait_for_frames(pipe, timeout=10.0):
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        state = pipe.state()
        if state["frame_seq"] > 0 or state["error"]:
            return state
        time.sleep(0.05)
    return pipe.state()


def test_the_pipeline_runs_with_no_zone_at_all(pipeline, tmp_path):
    clip = write_clip(tmp_path / "empty-zone.mp4")

    # an empty list means "not drawn yet", not a broken request
    pipeline.start(
        source=clip,
        machine={"id": "excavator_01", "name": "Excavator 01"},
        points=[],
        allowed_people=0,
    )

    state = wait_for_frames(pipeline)

    assert state["error"] is None
    assert state["frame_seq"] > 0
    assert state["zone_points"] == []
    assert (state["frame_width"], state["frame_height"]) == (WIDTH, HEIGHT)


def test_people_are_boxed_but_nothing_alerts_without_a_zone(pipeline, tmp_path):
    clip = write_clip(tmp_path / "no-alerts.mp4")

    pipeline.start(
        source=clip,
        machine={"id": "excavator_01", "name": "Excavator 01"},
        points=[],
        allowed_people=0,
    )

    state = wait_for_frames(pipeline)

    # the detector still reports the person, so the boxes are drawn
    assert [worker["worker_id"] for worker in state["workers"]] == [1]
    assert state["inside"] == 0
    assert state["status"] == "SAFE"
    assert state["capacity_alert"] is False
    assert state["worker_alert"] is False
    assert state["message"] == ""


def test_a_saved_zone_taken_on_still_will_still_work(tmp_path, monkeypatch):
    """The point of the two tests above is not to break normal monitoring."""
    monkeypatch.setattr("pipeline.PersonTracker", StubTracker)
    pipe = Pipeline(Database(str(tmp_path / "with-zone.db")))
    clip = write_clip(tmp_path / "with-zone.mp4")

    try:
        pipe.start(
            source=clip,
            machine={"id": "excavator_01", "name": "Excavator 01"},
            # covers the middle half of the clip, where the stub person stands
            points=[[40, 20], [280, 20], [280, 160], [40, 160]],
            allowed_people=0,
            dwell_seconds=0.0,
        )

        state = wait_for_frames(pipe)

        assert state["zone_points"]
        assert state["inside"] == 1
        assert state["capacity_alert"] is True
    finally:
        pipe.stop()


# --- the saved zone name has to reach the alert text ------------------------
#
# /start and /config both hand the pipeline `zone_specs` as plain dicts, because
# that is what JSON is. Reading those dicts field-by-field is the whole job of
# _clean_zone_specs, and when it read them as objects instead, every name came
# back as "Zone N" and every per-zone allowance and dwell limit was replaced by
# the global one. The plumbing tests could not see it: they stopped at the
# stubbed `pipeline.start(**kwargs)`, where the values were still correct.

# covers the anchor point of StubTracker. Its bbox is (x1, y1, x2, y2) corner
# form, so the stub person's feet land on (95, 120).
ZONE_AROUND_THE_PERSON = [[40, 20], [280, 20], [280, 160], [40, 160]]


def named_spec(name: str, **overrides) -> dict:
    """A zone_specs entry exactly as /config hands it over: a dict, not a model."""
    spec = {
        "name": name,
        "allowed_people": 0,
        "dwell_seconds": 0.0,
        "enabled": True,
        "using_defaults": False,
    }
    spec.update(overrides)
    return spec


def wait_for_alert(pipe, timeout=10.0):
    """The state once the stub person is inside and the zone is over its limit."""
    deadline = time.perf_counter() + timeout
    state = pipe.state()
    while time.perf_counter() < deadline:
        state = pipe.state()
        if state["status"] == "ALERT" or state["error"]:
            return state
        time.sleep(0.05)
    return state


def test_a_zone_renamed_in_the_database_is_named_in_the_alert_after_a_restart(
    pipeline, tmp_path
):
    """A page refresh: /start reads the saved rules and monitoring must use them."""
    clip = write_clip(tmp_path / "renamed-on-restart.mp4")

    pipeline.start(
        source=clip,
        machine={"id": "excavator_01", "name": "Excavator 01"},
        zones=[ZONE_AROUND_THE_PERSON],
        zone_specs=[named_spec("place 1")],
        allowed_people=0,
        mode="INSTANT",
    )

    state = wait_for_alert(pipeline)

    assert state["error"] is None
    assert state["zone_reports"][0]["name"] == "place 1"
    # the operator is told which zone by name, not by its position
    assert "place 1" in state["message"]
    assert "Zone 1" not in state["message"]
    # and the per-zone rules came through with the name, not just the name alone
    assert state["zone_reports"][0]["allowed_people"] == 0
    assert state["zone_specs"][0]["name"] == "place 1"


def test_a_rename_saved_while_the_camera_runs_renames_the_alert_too(pipeline, tmp_path):
    """No restart: /config pushes the new name into the running camera."""
    clip = write_clip(tmp_path / "renamed-while-running.mp4")

    pipeline.start(
        source=clip,
        machine={"id": "excavator_01", "name": "Excavator 01"},
        zones=[ZONE_AROUND_THE_PERSON],
        zone_specs=[named_spec("Zone 2")],
        allowed_people=0,
        mode="INSTANT",
    )
    assert wait_for_alert(pipeline)["message"].endswith("Zone 2")

    # the operator renames it and saves; the camera keeps running
    pipeline.apply_settings(zone_specs=[named_spec("place 1")])

    state = wait_for_alert(pipeline)

    assert state["zone_reports"][0]["name"] == "place 1"
    assert "place 1" in state["message"]
    assert "Zone 2" not in state["message"]
    assert state["zone_specs"][0]["name"] == "place 1"


def test_a_zone_that_follows_the_defaults_still_names_itself(tmp_path, monkeypatch):
    """`using_defaults` fills in the numbers, never the name."""
    monkeypatch.setattr("pipeline.PersonTracker", StubTracker)
    pipe = Pipeline(Database(str(tmp_path / "defaults-name.db")))
    clip = write_clip(tmp_path / "defaults-name.mp4")

    try:
        pipe.start(
            source=clip,
            machine={"id": "excavator_01", "name": "Excavator 01"},
            zones=[ZONE_AROUND_THE_PERSON],
            # no numbers of its own: it borrows the globals, but the name is kept
            zone_specs=[
                {"name": "place 1", "using_defaults": True},
            ],
            allowed_people=0,
            mode="INSTANT",
        )

        state = wait_for_alert(pipe)

        assert state["zone_reports"][0]["name"] == "place 1"
        assert "place 1" in state["message"]
    finally:
        pipe.stop()


def test_a_zone_with_no_rule_of_its_own_still_gets_a_positional_name(tmp_path, monkeypatch):
    """A database from before named zones still works and is not blank."""
    monkeypatch.setattr("pipeline.PersonTracker", StubTracker)
    pipe = Pipeline(Database(str(tmp_path / "legacy-name.db")))
    clip = write_clip(tmp_path / "legacy-name.mp4")

    try:
        pipe.start(
            source=clip,
            machine={"id": "excavator_01", "name": "Excavator 01"},
            zones=[ZONE_AROUND_THE_PERSON],
            zone_specs=None,
            allowed_people=0,
            mode="INSTANT",
        )

        state = wait_for_alert(pipe)

        assert state["zone_reports"][0]["name"] == "Zone 1"
        assert state["message"].endswith("Zone 1")
    finally:
        pipe.stop()


def test_stopping_the_camera_forgets_the_frame_it_was_showing(pipeline, tmp_path):
    """A stopped pipeline reports no picture, so nothing - a saved zone on the
    next start, a state the page draws against - is scaled against the frame
    of the source that has ended."""
    clip = write_clip(tmp_path / "after-stop.mp4")

    pipeline.start(
        source=clip,
        machine={"id": "excavator_01", "name": "Excavator 01"},
        points=[],
        allowed_people=0,
    )
    running = wait_for_frames(pipeline)
    assert (running["frame_width"], running["frame_height"]) == (WIDTH, HEIGHT)

    assert pipeline.stop() is True

    stopped = pipeline.state()
    assert (stopped["frame_width"], stopped["frame_height"]) == (0, 0)
    assert pipeline.frame_size == (0, 0)


def _wait_until_done(pipe, timeout=15.0):
    deadline = time.perf_counter() + timeout
    while pipe.active and time.perf_counter() < deadline:
        time.sleep(0.05)
    assert not pipe.active, "single play did not finish"


def test_single_play_ends_cleanly_with_clip_finished(pipeline, tmp_path):
    """Loop off: the clip plays once, then the feed stops with a neutral
    "Clip finished" state - no error, and Start works again."""
    clip = write_clip(tmp_path / "once.mp4", frames=10)

    pipeline.start(
        source=clip,
        machine={"id": "excavator_01", "name": "Excavator 01"},
        points=[],
        allowed_people=0,
        realtime=False,
        loop=False,
    )
    _wait_until_done(pipeline)

    assert pipeline.error is None
    assert pipeline.finished_clip is True
    state = pipeline.state()
    assert state["running"] is False
    assert state["error"] is None
    assert state["finished_clip"] is True

    # an explicit stop is not a finished clip
    assert pipeline.stop() is True
    assert pipeline.state()["finished_clip"] is False

    # ... and Start works again straight after the finish
    pipeline.start(
        source=clip,
        machine={"id": "excavator_01", "name": "Excavator 01"},
        points=[],
        allowed_people=0,
        realtime=False,
        loop=False,
    )
    _wait_until_done(pipeline)
    assert pipeline.state()["finished_clip"] is True


def test_loop_on_replays_and_never_reports_finished(pipeline, tmp_path):
    """Loop on: the clip rewinds (frame_seq drops) and keeps publishing,
    and a user stop afterwards is a plain stop, not a finish."""
    clip = write_clip(tmp_path / "replay.mp4", frames=10)

    pipeline.start(
        source=clip,
        machine={"id": "excavator_01", "name": "Excavator 01"},
        points=[],
        allowed_people=0,
        realtime=False,
        loop=True,
    )
    try:
        deadline = time.perf_counter() + 10.0
        last = -1
        rewound = False
        while time.perf_counter() < deadline:
            seq = pipeline.state()["frame_seq"]
            if seq < last:
                rewound = True
                break
            last = max(last, seq)
            time.sleep(0.02)
        assert rewound, "looped clip never rewound"
        assert pipeline.state()["finished_clip"] is False
    finally:
        assert pipeline.stop() is True
    assert pipeline.state()["finished_clip"] is False
