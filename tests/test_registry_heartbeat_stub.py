"""Local registry + heartbeat + alert-stub tests (no backend, no network)."""

import json

from services.heartbeat import CameraHeartbeatReporter, HeartbeatOutcome
from services.registry import (
    FileRegistrySource,
    RegistryPoller,
    parse_camera_entry,
)
from services.vision_alerts import VisionAlertEmitter, VisionAlertOutcome


def _entry(**over):
    base = {"id": 1, "code": "CAM-1", "detectors": ["PERSON"],
            "analysis_fps": 2.0, "video_path": "data/videos/clip.mp4"}
    base.update(over)
    return base


def test_parse_camera_entry_accepts_file_source():
    record = parse_camera_entry(_entry())
    assert record is not None and record.stream_ref() == "data/videos/clip.mp4"
    assert parse_camera_entry({"id": 1}) is None
    assert parse_camera_entry(_entry(analysis_fps=0)) is None


def test_file_registry_source_roundtrip(tmp_path):
    path = tmp_path / "cameras.json"
    path.write_text(json.dumps({"cameras": [
        _entry(id=1, code="A"),
        {"id": 2},  # bad entry skipped, rest kept
        _entry(id=3, code="C", rtsp_url="rtsp://x/y", video_path=""),
    ]}), encoding="utf-8")
    records = FileRegistrySource(path).fetch()
    assert [r.camera_id for r in records] == ["1", "3"]


def test_registry_poller_reconciles_without_backend(tmp_path):
    path = tmp_path / "cameras.json"
    path.write_text(json.dumps({"cameras": [_entry(id=1, code="A")]}),
                    encoding="utf-8")
    poller = RegistryPoller(FileRegistrySource(path), supervisor=None,
                            poll_seconds=30.0)
    first = poller.poll_once()
    assert first["added"] == 1 and first["kept"] == 0
    second = poller.poll_once()
    assert second["kept"] == 1


def test_heartbeat_reporter_maps_outcomes_and_never_raises():
    class FakeHttp:
        def __init__(self, status):
            self.status = status

        def post_json(self, path, payload):
            assert "heartbeat" in path
            assert set(payload) == {"timestamp", "timeout"}
            if isinstance(self.status, Exception):
                raise self.status
            return self.status

    assert CameraHeartbeatReporter(
        http=FakeHttp(200)).beat(7, "2026-01-01T00:00:00+00:00") is HeartbeatOutcome.SENT
    assert CameraHeartbeatReporter(
        http=FakeHttp(400)).beat(7, "2026-01-01T00:00:00+00:00") is HeartbeatOutcome.REJECTED
    assert CameraHeartbeatReporter(
        http=FakeHttp(None)).beat(7, "2026-01-01T00:00:00+00:00") is HeartbeatOutcome.FAILED
    assert CameraHeartbeatReporter(
        http=FakeHttp(RuntimeError("boom"))).beat(
        7, "2026-01-01T00:00:00+00:00") is HeartbeatOutcome.FAILED


def test_heartbeat_403_backs_off_then_reports(tmp_path):
    class Deny:
        def post_json(self, path, payload):
            return 403

    clock = [100.0]
    reporter = CameraHeartbeatReporter(http=Deny(), clock=lambda: clock[0])
    assert reporter.beat(1, "t") is HeartbeatOutcome.DENIED
    assert reporter.beat(1, "t") is HeartbeatOutcome.DENIED  # backoff, no POST
    assert reporter.auth_backoff_active is True
    clock[0] += 301.0
    assert reporter.auth_backoff_active is False


def test_vision_alert_stub_maps_outcomes(tmp_path):
    class FakeHttp:
        def __init__(self, status):
            self.status = status

        def post_json(self, path, payload):
            if isinstance(self.status, Exception):
                raise self.status
            return self.status

    assert VisionAlertEmitter(http=FakeHttp(202)).emit({"a": 1}) is VisionAlertOutcome.SENT
    assert VisionAlertEmitter(http=FakeHttp(400)).emit({"a": 1}) is VisionAlertOutcome.REJECTED
    assert VisionAlertEmitter(http=FakeHttp(None)).emit({"a": 1}) is VisionAlertOutcome.FAILED
    assert VisionAlertEmitter(http=FakeHttp(RuntimeError("x"))).emit(
        {"a": 1}) is VisionAlertOutcome.FAILED


def test_vision_alert_stub_writes_local_file(tmp_path, monkeypatch):
    target = tmp_path / "alerts.jsonl"
    monkeypatch.chdir(tmp_path)
    from services.vision_alerts import LocalAlertSink

    sink = LocalAlertSink(path=str(target))
    assert sink.post_json("/x", {"a": 1}) == 202
    assert json.loads(target.read_text(encoding="utf-8").strip()) == {"a": 1}
