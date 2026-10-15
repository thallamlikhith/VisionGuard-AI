"""Ported fire/smoke gate + payload tests (DMS test_fire_smoke, backend-free).

Covers the copied operating point exactly: conf 0.35 / fire 1.5 s /
smoke 2.5 s, class order 0 = smoke / 1 = fire, payload key set without a
dedup key. Emitter/network tests from DMS are NOT copied (they need the
Django backend); local-stub emitter tests live in
test_registry_heartbeat_stub.py.
"""

import pytest

from detector.fire_smoke import FIRE_SMOKE_CLASSES, FIRE_SMOKE_WEIGHTS_FILE
from detector.fire_smoke_gate import (
    FireSmokeGate,
    build_vision_alert_payload,
    extract_fire_smoke,
    grade_fire_smoke,
)


def raw(name, conf=0.9, box=(5, 5, 20, 20)):
    return {"class_name": name, "confidence": conf, "bbox_xyxy": list(box)}


def test_class_order_and_weights_preserved():
    assert FIRE_SMOKE_CLASSES == ("smoke", "fire")
    assert FIRE_SMOKE_WEIGHTS_FILE == "fire_smoke_v2.pt"


def test_extract_keeps_only_fire_smoke_case_insensitive_sorted():
    dets = [raw("Person", 0.99), raw("FIRE", 0.5), raw("smoke", 0.7),
            {"class_name": "Fire", "confidence": 0.95, "bbox_xyxy": [1, 1, 9, 9]},
            {"class_name": "nope"}, "garbage", raw("fire", 0.4, box=(1, 1, 2, 2))]
    kept = extract_fire_smoke(dets)
    assert [d["vision_alert_type"] for d in kept] == ["fire", "smoke", "fire", "fire"]
    assert kept[0]["confidence"] == pytest.approx(0.95)
    assert extract_fire_smoke(
        [{"class_name": "fire", "confidence": 0.9, "bbox_xyxy": [20, 20, 5, 5]}]
    ) == []


def test_grade_table():
    assert grade_fire_smoke("fire", 0.9) == ("RED", "high")
    assert grade_fire_smoke("fire", 0.5) == ("ORANGE", "high")
    assert grade_fire_smoke("smoke", 0.9) == ("ORANGE", "medium")
    assert grade_fire_smoke("smoke", 0.5) == ("YELLOW", "medium")


def test_payload_is_exactly_the_intake_keys():
    payload = build_vision_alert_payload(
        camera="CAM-1", vision_alert_type="fire", confidence=0.91,
        bbox_xyxy=[5, 5, 20, 20], track_id=None,
        reported_at="2026-01-01T00:00:00+00:00",
    )
    assert set(payload) == {"camera", "vision_alert_type", "severity", "risk_level",
                            "confidence", "bbox_xyxy", "track_id", "reported_at"}
    assert payload["severity"] == "RED" and payload["risk_level"] == "high"
    assert "dedup_key" not in payload


def test_gate_threshold_and_rearm():
    gate = FireSmokeGate(threshold=3)
    assert gate.update("c", ["fire"]) == []
    assert gate.update("c", ["fire"]) == []
    assert gate.update("c", ["fire"]) == ["fire"]
    assert gate.update("c", ["fire"]) == []
    assert gate.update("c", []) == []
    assert gate.update("c", ["fire"]) == []
    assert gate.update("c", ["fire"]) == []
    assert gate.update("c", ["fire"]) == ["fire"]
    gate.reset_camera("c")
    assert gate.update("c", ["fire"]) == []


def test_gate_per_type_independent():
    gate = FireSmokeGate(threshold=2)
    assert gate.update("c", ["fire", "smoke"]) == []
    assert sorted(gate.update("c", ["fire", "smoke"])) == ["fire", "smoke"]


def test_gate_rejects_bad_threshold():
    with pytest.raises(ValueError, match=">= 1"):
        FireSmokeGate(threshold=0)


def test_persistence_seconds_convert_to_frames_ceil_min_one():
    gate = FireSmokeGate()  # 1.5 s fire / 2.5 s smoke
    gate.set_camera_fps("cam1", 2.0)
    assert gate.required_frames("cam1", "fire") == 3
    assert gate.required_frames("cam1", "smoke") == 5
    gate.set_camera_fps("fast", 5.0)
    assert gate.required_frames("fast", "fire") == 8
    assert gate.required_frames("fast", "smoke") == 13
    gate.set_camera_fps("slow", 0.5)
    assert gate.required_frames("slow", "fire") == 1
    assert gate.required_frames("slow", "smoke") == 2
    assert gate.required_frames("unknown", "fire") == 3
    assert gate.required_frames("unknown", "smoke") == 5


def test_fire_needs_three_frames_at_cam1_rate():
    gate = FireSmokeGate()
    gate.set_camera_fps("cam1", 2.0)
    assert gate.update("cam1", ["fire"]) == []
    assert gate.update("cam1", ["fire"]) == []
    assert gate.update("cam1", ["fire"]) == ["fire"]
    assert gate.update("cam1", ["fire"]) == []


def test_smoke_needs_five_frames_at_cam1_rate():
    gate = FireSmokeGate()
    gate.set_camera_fps("cam1", 2.0)
    for _ in range(4):
        assert gate.update("cam1", ["smoke"]) == []
    assert gate.update("cam1", ["smoke"]) == ["smoke"]
    assert gate.update("cam1", ["smoke"]) == []


def test_mixed_frame_advances_both_types_independently():
    gate = FireSmokeGate()
    gate.set_camera_fps("c", 2.0)
    assert gate.update("c", ["fire", "smoke"]) == []
    assert gate.update("c", ["fire", "smoke"]) == []
    assert gate.update("c", ["fire", "smoke"]) == ["fire"]
    assert gate.update("c", ["fire", "smoke"]) == []
    assert gate.update("c", ["fire", "smoke"]) == ["smoke"]


def test_fire_only_frame_does_not_advance_smoke_counter():
    gate = FireSmokeGate()
    gate.set_camera_fps("c", 2.0)
    assert gate.update("c", ["smoke"]) == []
    assert gate.update("c", ["smoke"]) == []
    assert gate.update("c", ["fire"]) is not None
    assert gate.update("c", ["smoke"]) == []
    assert gate.update("c", ["smoke"]) == []
    assert gate.update("c", ["smoke"]) == []
    assert gate.update("c", ["smoke"]) == []
    assert gate.update("c", ["smoke"]) == ["smoke"]


def test_per_camera_persistence_override_and_reset():
    gate = FireSmokeGate()
    gate.set_camera_fps("c", 2.0)
    gate.set_camera_persistence("c", smoke_seconds=1.0)
    assert gate.required_frames("c", "smoke") == 2
    assert gate.required_frames("c", "fire") == 3
    gate.reset_camera("c")
    assert gate.required_frames("c", "smoke") == 5
    with pytest.raises(ValueError, match="> 0"):
        gate.set_camera_fps("c", 0)
    with pytest.raises(ValueError, match="> 0"):
        FireSmokeGate(fire_seconds=0)
    with pytest.raises(ValueError, match="> 0"):
        gate.set_camera_persistence("c", smoke_seconds=-1)
