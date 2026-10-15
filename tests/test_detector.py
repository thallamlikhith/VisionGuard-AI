"""Tests for the detector: model loading, OpenVINO export, frame skipping.

The real YOLO weights are never loaded here - model loading is monkeypatched
so the tests stay fast and do not need a network or a camera.
"""

import logging
import sys

import pytest

from detector import tracker as tracker_module
from detector.tracker import DetectorError, PersonTracker, configure_logging
from zone.danger_zone import ANCHORS, DangerZone, ZoneMonitor


class FakeResults:
    """Mimics the bit of ultralytics' Results that the tracker touches."""

    def __init__(self, people, ids=True):
        self.boxes = _Boxes(people, ids=ids)


class _Array(list):
    def tolist(self):
        return list(self)


class _Boxes:
    def __init__(self, people, ids=True):
        # ids=False is the "boxes found, but ByteTrack has not assigned a track
        # id yet" case: the boxes are real, boxes.id is None.
        self.id = _Array([p["id"] for p in people]) if people and ids else None
        self.xyxy = _Array([p["box"] for p in people]) if people else None
        self.conf = _Array([p["conf"] for p in people]) if people else None

    def __len__(self):
        # the number of detected boxes, which is independent of whether
        # ByteTrack has given them an id
        return len(self.xyxy) if self.xyxy is not None else 0


class Recorder:
    """Records what the fake model was asked to do, exports and inference apart."""

    def __init__(self):
        self.tracks = []
        self.exports = []

    def clear(self):
        self.tracks.clear()
        self.exports.clear()

    # the existing assertions read the inference calls straight off the recorder
    def __len__(self):
        return len(self.tracks)

    def __getitem__(self, index):
        return self.tracks[index]

    def __eq__(self, other):
        return self.tracks == other


@pytest.fixture()
def fake_yolo(monkeypatch):
    """Replace ultralytics' YOLO with a recorder that detects nothing."""
    calls = Recorder()

    class FakeModel:
        def __init__(self, path, *args, **kwargs):
            self.path = str(path)

        def track(self, frame, **kwargs):
            calls.tracks.append(kwargs)
            return [FakeResults([])]

        def export(self, **kwargs):
            calls.exports.append(kwargs)
            return "ok"

    monkeypatch.setattr(tracker_module, "YOLO", FakeModel)
    return calls


def test_tracker_defaults_to_the_configured_size(fake_yolo):
    tracker = PersonTracker("yolo11n.pt", imgsz=320, detect_every_n=1)

    assert tracker.imgsz == 320
    assert tracker.backend == "pytorch"
    fake_yolo.clear()
    tracker.track(object(), 0)
    assert fake_yolo[0]["imgsz"] == 320


def test_a_failing_model_becomes_a_detector_error(fake_yolo, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("no weights")

    monkeypatch.setattr(tracker_module, "YOLO", boom)
    with pytest.raises(DetectorError):
        PersonTracker("missing.pt")


def test_every_frame_is_detected_by_default(fake_yolo):
    tracker = PersonTracker("yolo11n.pt", detect_every_n=1)

    assert [tracker.should_detect(i) for i in range(4)] == [True, True, True, True]


def test_detect_every_n_skips_the_frames_in_between(fake_yolo):
    tracker = PersonTracker("yolo11n.pt", detect_every_n=3)

    assert [tracker.should_detect(i) for i in range(7)] == [
        True,
        False,
        False,
        True,
        False,
        False,
        True,
    ]


def test_a_skipped_frame_reuses_the_last_detections(fake_yolo):
    """The zone has to keep seeing the same people between detections."""
    people = [{"id": 3, "box": [1, 1, 10, 10], "conf": 0.9}]
    calls = []

    class FakeModel:
        def __init__(self, *args, **kwargs):
            pass

        def track(self, frame, **kwargs):
            calls.append(kwargs)
            return [FakeResults(people)]

    fake_yolo.clear()
    original = tracker_module.YOLO
    tracker_module.YOLO = FakeModel
    try:
        tracker = PersonTracker("yolo11n.pt", detect_every_n=2)
        first = tracker.track(object(), 0)
        skipped = tracker.track(object(), 1)
        second = tracker.track(object(), 2)
    finally:
        tracker_module.YOLO = original

    assert len(calls) == 3  # the construction burn plus frames 0 and 2; frame 1 never reached YOLO
    assert [p.worker_id for p in first] == [3]
    assert [p.worker_id for p in skipped] == [3]
    assert [p.worker_id for p in second] == [3]
    assert skipped[0].bbox == (1, 1, 10, 10)


class ScriptedModel:
    """A fake YOLO that plays back one result list per inference call."""

    def __init__(self, script):
        self.script = script
        self.calls = 0

    def __call__(self, *args, **kwargs):
        return self

    def track(self, frame, **kwargs):
        index = min(self.calls, len(self.script) - 1)
        self.calls += 1
        return [FakeResults(self.script[index])]


def _adaptive_tracker(monkeypatch, script, **kwargs):
    model = ScriptedModel(script)
    monkeypatch.setattr(tracker_module, "YOLO", model)
    tracker = PersonTracker("yolo11n.pt", **kwargs)
    return tracker, model


def test_adaptive_is_off_by_default(monkeypatch):
    """Without adaptive_idle_n an empty scene is still looked at every frame."""
    tracker, model = _adaptive_tracker(monkeypatch, [[]])
    for frame_index in range(4):
        assert tracker.track(object(), frame_index) == []
    assert model.calls == 5  # the construction burn plus one inference per frame


def test_adaptive_idle_scene_is_checked_every_third_frame(monkeypatch):
    """Nobody around: YOLO runs on frames 0, 3, 6, the rest reuse the last."""
    tracker, model = _adaptive_tracker(
        monkeypatch, [[]], detect_every_n=1, adaptive_idle_n=3
    )
    for frame_index in range(7):
        assert tracker.track(object(), frame_index) == []
    assert model.calls == 4  # the construction burn plus the looks on 0, 3, 6


def test_adaptive_stays_fast_while_people_are_visible(monkeypatch):
    """Somebody around: every frame is a detection frame, like detect_every_n=1."""
    person = [{"id": 3, "box": [1, 1, 10, 10], "conf": 0.9}]
    tracker, model = _adaptive_tracker(
        monkeypatch, [person], detect_every_n=1, adaptive_idle_n=3
    )
    for frame_index in range(5):
        assert [p.worker_id for p in tracker.track(object(), frame_index)] == [3]
    assert model.calls == 6  # the construction burn plus one inference per frame


def test_adaptive_wakes_up_when_somebody_reappears(monkeypatch):
    """Empty, then somebody: the return is picked up on the next idle look."""
    person = [{"id": 3, "box": [1, 1, 10, 10], "conf": 0.9}]
    tracker, model = _adaptive_tracker(
        monkeypatch,
        [[], [], person],
        detect_every_n=1,
        adaptive_idle_n=3,
    )
    seen = [tracker.track(object(), frame_index) for frame_index in range(7)]
    # the construction burn consumes the first scripted result; idle looks
    # run on frames 0 and 3, and the second look finally sees somebody
    assert model.calls == 6
    assert [p.worker_id for p in seen[6]] == [3]
    # and the fast cadence is back from then on
    assert [p.worker_id for p in tracker.track(object(), 7)] == [3]
    assert model.calls == 7


def test_adaptive_idle_n_of_one_means_off(monkeypatch):
    """adaptive_idle_n=1 is the same schedule as disabled."""
    tracker, model = _adaptive_tracker(
        monkeypatch, [[]], detect_every_n=1, adaptive_idle_n=1
    )
    for frame_index in range(4):
        tracker.track(object(), frame_index)
    assert model.calls == 5  # the construction burn plus one inference per frame


def test_a_frame_without_track_ids_does_not_start_a_second_worker(monkeypatch):
    """boxes.id is None must not drop the worker or restart its clock.

    ByteTrack can return boxes on a frame before it has assigned an id: the
    first frames of a new track, or a frame it could not match. Reporting those
    as new people (or as nobody) makes the zone logic see worker 3 leave and a
    fresh worker arrive, restart its dwell timer and write a second ZONE_ENTRY
    when the real id comes back. Reusing the previous stable tracks keeps one
    worker with one clock, which is what this asserts.
    """
    box = [10, 10, 50, 90]
    with_id = [{"id": 3, "box": box, "conf": 0.9}]
    calls = {"n": 0}

    class FakeModel:
        def __init__(self, *args, **kwargs):
            pass

        def track(self, frame, **kwargs):
            calls["n"] += 1
            # the construction burn is call 1; frame 0 and 2 carry track id
            # 3, frame 1 has boxes but no id yet
            missing = calls["n"] == 3
            return [FakeResults(with_id, ids=not missing)]

    monkeypatch.setattr(tracker_module, "YOLO", FakeModel)

    tracker = PersonTracker("yolo11n.pt", imgsz=640, detect_every_n=1)

    zone = DangerZone(zones=[[[0, 0], [100, 0], [100, 100], [0, 100]]])
    monitor = ZoneMonitor(zone=zone, mode="TIMER", dwell_seconds=5.0, allowed_people=1)
    anchor = ANCHORS["bottom_center"]

    def feed(frame_index):
        people = tracker.track(object(), frame_index)
        detections = {
            person.worker_id: (anchor(person.bbox), person.bbox, person.confidence)
            for person in people
        }
        workers, events = monitor.update_all(detections, now=float(frame_index))
        return people, workers, events

    people, workers, events = feed(0)
    assert [p.worker_id for p in people] == [3]
    assert [e.event_type for e in events] == ["ZONE_ENTRY"]

    # the missing-id frame: the worker is still reported with the same id
    people, workers, events = feed(1)
    assert [p.worker_id for p in people] == [3]
    assert [w.worker_id for w in workers] == [3]  # no second worker appeared
    assert events == []  # and no fresh ZONE_ENTRY

    # the id is back: still one worker, and the clock never restarted
    people, workers, events = feed(2)
    assert [w.worker_id for w in workers] == [3]
    assert events == []
    assert monitor.workers[3].entered_at == 0.0
    assert monitor.workers[3].dwell_seconds == 2.0


def test_openvino_is_used_when_the_export_exists(fake_yolo, monkeypatch, tmp_path):
    export_dir = tmp_path / "yolo11n_openvino_model"
    export_dir.mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(PersonTracker, "_openvino_dir", lambda self, name: str(export_dir))
    monkeypatch.setattr(PersonTracker, "_exported_imgsz", lambda self, d: 320)

    tracker = PersonTracker("yolo11n.pt", imgsz=320, openvino=True)

    assert tracker.backend == "openvino"
    assert fake_yolo.exports == []  # nothing was exported, it was already there


def test_an_export_of_the_wrong_size_is_replaced_once(fake_yolo, monkeypatch, tmp_path):
    """A 640 export cannot be used at imgsz=320, it has to be re-exported."""
    export_dir = tmp_path / "yolo11n_openvino_model"
    export_dir.mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(PersonTracker, "_openvino_dir", lambda self, name: str(export_dir))
    monkeypatch.setattr(PersonTracker, "_exported_imgsz", lambda self, d: 640)

    tracker = PersonTracker("yolo11n.pt", imgsz=320, openvino=True)

    assert tracker.backend == "openvino"
    assert fake_yolo.exports == [{"format": "openvino", "imgsz": 320, "half": True, "verbose": False}]


def test_a_matching_export_is_left_alone(fake_yolo, monkeypatch, tmp_path):
    export_dir = tmp_path / "yolo11n_openvino_model"
    export_dir.mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(PersonTracker, "_openvino_dir", lambda self, name: str(export_dir))
    monkeypatch.setattr(PersonTracker, "_exported_imgsz", lambda self, d: 320)

    PersonTracker("yolo11n.pt", imgsz=320, openvino=True)

    assert fake_yolo.exports == []


def test_a_failed_export_falls_back_to_the_pt_weights(fake_yolo, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(PersonTracker, "_exported_imgsz", lambda self, d: None)

    def boom(self, model_name, directory):
        return False

    monkeypatch.setattr(PersonTracker, "_export_openvino", boom)

    tracker = PersonTracker("yolo11n.pt", imgsz=320, openvino=True)

    assert tracker.backend == "pytorch"
    assert tracker.model.path.endswith("yolo11n.pt")


def test_an_unreadable_export_falls_back_instead_of_crashing(fake_yolo, monkeypatch, tmp_path):
    export_dir = tmp_path / "yolo11n_openvino_model"
    export_dir.mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(PersonTracker, "_openvino_dir", lambda self, name: str(export_dir))
    monkeypatch.setattr(PersonTracker, "_exported_imgsz", lambda self, d: None)

    class BrokenYOLO:
        def __init__(self, path, *args, **kwargs):
            if "openvino" in str(path):
                raise RuntimeError("corrupt export")

    monkeypatch.setattr(tracker_module, "YOLO", BrokenYOLO)

    tracker = PersonTracker("yolo11n.pt", openvino=True)

    assert tracker.backend == "pytorch"


def test_the_export_directory_sits_next_to_the_weights(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    assert PersonTracker._openvino_dir("yolo11n.pt").endswith("yolo11n_openvino_model")
    assert "openvino" in PersonTracker._openvino_dir(str(tmp_path / "yolo11n.pt"))


def test_the_export_directory_does_not_depend_on_the_working_directory(
    monkeypatch, tmp_path
):
    """The export is anchored to the weights, so any cwd finds it again."""
    weights = tmp_path / "weights" / "yolo11n.pt"
    monkeypatch.chdir(tmp_path)
    from_weights = PersonTracker._openvino_dir(str(weights))

    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    assert PersonTracker._openvino_dir(str(weights)) == from_weights


def test_an_unreadable_export_size_reports_unknown(monkeypatch, tmp_path):
    """No xml, no openvino: the answer is None so we do not re-export blindly."""
    tracker = PersonTracker.__new__(PersonTracker)
    empty = tmp_path / "empty"
    empty.mkdir()
    assert tracker._exported_imgsz(str(empty)) is None


# ---------------------------------------------------------------------------
# the machine has no openvino package: does it fall back to the .pt weights?
# ---------------------------------------------------------------------------


@pytest.fixture()
def no_openvino(monkeypatch):
    """Make `import openvino` fail, the way it does on a machine without it."""
    monkeypatch.setitem(sys.modules, "openvino", None)


class _FakeOpenvino:
    """Stands in for a working openvino install when the check needs it."""

    __version__ = "0.0-fake"


def test_the_export_size_is_unknown_without_the_openvino_package(
    no_openvino, fake_yolo, tmp_path
):
    """Reading the .xml needs openvino; without it we must not re-export."""
    tracker = PersonTracker.__new__(PersonTracker)
    export_dir = tmp_path / "yolo11n_openvino_model"
    export_dir.mkdir()
    (export_dir / "yolo11n.xml").write_text("<net/>", encoding="utf-8")

    assert tracker._exported_imgsz(str(export_dir)) is None


def test_a_missing_openvino_package_falls_back_with_a_clear_log(
    no_openvino, monkeypatch, tmp_path, caplog
):
    """The whole point of the fallback: no openvino, but detection still works.

    The fake model raises on export() when openvino is missing, which is what
    ultralytics does on a machine without the package, so the real
    _export_openvino / _load_openvino code decides the outcome.
    """

    class NoOpenvinoModel:
        def __init__(self, path, *args, **kwargs):
            self.path = str(path)

        def track(self, frame, **kwargs):
            return []

        def export(self, **kwargs):
            raise RuntimeError(
                "OpenVINO is not installed: install it with 'pip install openvino'"
            )

    export_dir = tmp_path / "yolo11n_openvino_model"  # not created: nothing to load
    monkeypatch.setattr(tracker_module, "YOLO", NoOpenvinoModel)
    monkeypatch.setattr(PersonTracker, "_openvino_dir", lambda self, name: str(export_dir))

    with caplog.at_level(logging.INFO, logger="detector.tracker"):
        tracker = PersonTracker("yolo11n.pt", imgsz=320, openvino=True)

    assert tracker.backend == "pytorch"
    assert tracker.model.path.endswith("yolo11n.pt")

    messages = [record.getMessage() for record in caplog.records]
    assert any("OpenVINO export failed" in message for message in messages)
    assert any("using yolo11n.pt" in message for message in messages)

    # the backend that ended up being used is always reported
    assert any("detector ready: backend=pytorch" in message for message in messages)


def test_an_existing_export_that_cannot_be_loaded_falls_back_with_a_clear_log(
    no_openvino, monkeypatch, tmp_path, caplog
):
    """The export is on disk but openvino cannot read it: say so, use the .pt."""
    export_dir = tmp_path / "yolo11n_openvino_model"
    export_dir.mkdir()
    monkeypatch.setattr(PersonTracker, "_openvino_dir", lambda self, name: str(export_dir))
    monkeypatch.setattr(PersonTracker, "_exported_imgsz", lambda self, d: None)

    class FakeModel:
        def __init__(self, path, *args, **kwargs):
            if "openvino" in str(path):
                raise RuntimeError("openvino is not available")
            self.path = str(path)

    monkeypatch.setattr(tracker_module, "YOLO", FakeModel)

    with caplog.at_level(logging.INFO, logger="detector.tracker"):
        tracker = PersonTracker("yolo11n.pt", imgsz=320, openvino=True)

    assert tracker.backend == "pytorch"
    messages = [record.getMessage() for record in caplog.records]
    assert any("could not load the OpenVINO model" in m and "using yolo11n.pt" in m for m in messages)


def test_the_fallback_still_detects(fake_yolo, monkeypatch, tmp_path):
    """After falling back, tracking works exactly as before."""
    monkeypatch.setattr(PersonTracker, "_export_openvino", lambda self, m, d: False)
    monkeypatch.setattr(PersonTracker, "_exported_imgsz", lambda self, d: None)

    tracker = PersonTracker("yolo11n.pt", imgsz=320, openvino=True)
    fake_yolo.clear()
    assert tracker.track(object(), 0) == []
    assert len(fake_yolo) == 1
    assert fake_yolo[0]["imgsz"] == 320


def test_an_export_that_loads_but_cannot_run_falls_back(monkeypatch, tmp_path, caplog):
    """YOLO(dir) builds its backend lazily, so it can load and still be unusable.

    This is what a machine without the openvino package looks like when an
    export directory was copied along with the project: the load succeeds, the
    first frame would fail. One blank frame at load time catches it.
    """
    export_dir = tmp_path / "yolo11n_openvino_model"
    export_dir.mkdir()
    monkeypatch.setattr(PersonTracker, "_openvino_dir", lambda self, name: str(export_dir))
    monkeypatch.setattr(PersonTracker, "_exported_imgsz", lambda self, d: None)
    monkeypatch.setitem(sys.modules, "openvino", _FakeOpenvino())

    calls = []

    class LazyBackend:
        """Loads fine, raises on inference - the real failure mode."""

        def __init__(self, path, *args, **kwargs):
            self.path = str(path)
            if "openvino" not in self.path:
                calls.append(("pt", self.path))

        def track(self, frame, **kwargs):
            if "openvino" in self.path:
                raise RuntimeError("could not create an inference request")
            return [FakeResults([])]

    monkeypatch.setattr(tracker_module, "YOLO", LazyBackend)

    with caplog.at_level(logging.INFO, logger="detector.tracker"):
        tracker = PersonTracker("yolo11n.pt", imgsz=320, openvino=True)

    assert tracker.backend == "pytorch"
    messages = [record.getMessage() for record in caplog.records]
    assert any("could not run" in message and "using yolo11n.pt" in message for message in messages)

    # and the fallback path really does run inference
    assert tracker.track(object(), 0) == []


def test_the_export_smoke_test_does_not_seed_the_tracker_state(fake_yolo, monkeypatch, tmp_path):
    """The blank frame must not create a ByteTrack session."""
    export_dir = tmp_path / "yolo11n_openvino_model"
    export_dir.mkdir()
    monkeypatch.setattr(PersonTracker, "_openvino_dir", lambda self, name: str(export_dir))
    monkeypatch.setattr(PersonTracker, "_exported_imgsz", lambda self, d: 320)

    PersonTracker("yolo11n.pt", imgsz=320, openvino=True)

    assert fake_yolo, "the smoke test should have run one inference"
    assert fake_yolo[0]["persist"] is False


def test_the_export_is_only_smoke_tested_once(fake_yolo, monkeypatch, tmp_path):
    export_dir = tmp_path / "yolo11n_openvino_model"
    export_dir.mkdir()
    monkeypatch.setattr(PersonTracker, "_openvino_dir", lambda self, name: str(export_dir))
    monkeypatch.setattr(PersonTracker, "_exported_imgsz", lambda self, d: 320)

    tracker = PersonTracker("yolo11n.pt", imgsz=320, openvino=True)
    fake_yolo.clear()

    tracker.track(object(), 0)
    tracker.track(object(), 1)

    assert len(fake_yolo) == 2  # just the two real frames, no extra probe


def test_configure_logging_sets_up_a_visible_handler():
    """Records must reach a handler, or the 'detector ready' line is lost.

    pytest attaches its own handlers to the root logger, so they are removed
    here to reproduce the CLI case: nothing configured yet.
    """
    import logging as logging_module

    root = logging_module.getLogger()
    had_handlers = list(root.handlers)
    had_level = root.level
    try:
        root.handlers.clear()
        root.setLevel(logging_module.WARNING)

        configure_logging()

        assert root.level == logging_module.INFO
        assert root.handlers, "no handler was installed, INFO records would be dropped"
        assert any(isinstance(h, logging_module.StreamHandler) for h in root.handlers)
    finally:
        root.handlers[:] = had_handlers
        root.setLevel(had_level)


def test_configure_logging_does_not_duplicate_handlers():
    """Called by both entry points, and twice by uvicorn's reload: no spam."""
    import logging as logging_module

    root = logging_module.getLogger()
    before = list(root.handlers)
    configure_logging()
    configure_logging()

    assert list(root.handlers) == before


def test_an_existing_logging_setup_is_left_alone():
    """An embedding app (or pytest) keeps its own handlers."""
    import logging as logging_module

    root = logging_module.getLogger()
    had_handlers = list(root.handlers)
    had_level = root.level
    marker = logging_module.NullHandler()
    try:
        root.handlers[:] = [marker]
        root.setLevel(logging_module.ERROR)

        configure_logging()

        assert root.handlers == [marker]
        assert root.level == logging_module.ERROR
    finally:
        root.handlers[:] = had_handlers
        root.setLevel(had_level)


# ---------------------------------------------------------------------------
# first-frame burn: every start must behave like the strict cold case, where
# ByteTrack's frame 1 is consumed by a blank frame and the first real frame
# has to confirm its boxes on a later frame before they become workers.
# Without the burn on the cache-hit path (predictor reset, no warm-up), the
# first real frame is ByteTrack frame 1 and its junk boxes instant-activate.
# ---------------------------------------------------------------------------


def _burn_spy(monkeypatch):
    """A YOLO fake recording every (frame, kwargs) track call."""
    calls = []

    class SpyModel:
        def __init__(self, *args, **kwargs):
            pass

        def track(self, frame, **kwargs):
            calls.append((frame, kwargs))
            return [FakeResults([])]

    monkeypatch.setattr(tracker_module, "YOLO", SpyModel)
    return calls


def _blank_frame(call):
    import numpy as np

    frame, _kwargs = call
    return (
        isinstance(frame, np.ndarray)
        and frame.shape == (64, 64, 3)
        and frame.dtype == np.uint8
        and not frame.any()
    )


def test_construction_burns_bytetrack_frame_one(monkeypatch):
    """A cold start runs one blank, non-persisted inference before any frame."""
    calls = _burn_spy(monkeypatch)

    PersonTracker("yolo11n.pt", detect_every_n=1)

    assert len(calls) == 1
    assert _blank_frame(calls[0]), "the burn must be a blank 64x64 frame"
    assert calls[0][1]["persist"] is False
    assert calls[0][1]["classes"] == [0]


def test_cache_hit_start_burns_frame_one_again(monkeypatch):
    """A second start reusing cached weights burns frame 1 again.

    The cache-hit path drops the predictor (fresh ByteTrack ids) but runs no
    warm-up, so without this burn its first real frame would be ByteTrack
    frame 1 and instant-activate junk. Both starts must burn.
    """
    calls = _burn_spy(monkeypatch)
    try:
        first = PersonTracker("yolo11n.pt", detect_every_n=1)
        second = PersonTracker("yolo11n.pt", detect_every_n=1)

        assert first.model is second.model  # the weights really are shared
        assert len(calls) == 2
        assert _blank_frame(calls[0])
        assert _blank_frame(calls[1])
        assert calls[0][1]["persist"] is False
        assert calls[1][1]["persist"] is False
    finally:
        tracker_module._MODEL_CACHE.clear()


def test_burn_result_never_becomes_a_worker(monkeypatch):
    """Even if the blank frame reported boxes with ids, nobody is hired."""
    seen = {"n": 0}

    class EagerModel:
        def __init__(self, *args, **kwargs):
            pass

        def track(self, frame, **kwargs):
            seen["n"] += 1
            if seen["n"] == 1:
                # the burn itself sees a (bogus) worker: it must be dropped
                return [FakeResults([{"id": 9, "box": [1, 1, 10, 10], "conf": 0.9}])]
            return [FakeResults([])]

    monkeypatch.setattr(tracker_module, "YOLO", EagerModel)

    tracker = PersonTracker("yolo11n.pt", detect_every_n=1)

    assert seen["n"] == 1
    assert tracker._last_people == []


def test_failed_burn_never_breaks_startup(monkeypatch):
    """A backend without track() (or a failing one) still constructs."""

    class Trackless:
        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setattr(tracker_module, "YOLO", Trackless)

    tracker = PersonTracker("yolo11n.pt", detect_every_n=1)

    assert tracker.backend == "pytorch"
