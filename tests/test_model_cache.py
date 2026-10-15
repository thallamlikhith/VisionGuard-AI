"""The side detectors share one loaded YOLO object across starts.

Rebuilding them on every POST /start reloaded the weights and recompiled the
graph, so every Start press and video switch stalled. These pin the sharing and
the key (path + imgsz) without loading real weights.
"""

from __future__ import annotations

import ultralytics

from detector import model_cache


class _FakeYOLO:
    instances: list["_FakeYOLO"] = []

    def __init__(self, path):
        self.path = path
        self.names = {0: "smoke", 1: "fire"}
        _FakeYOLO.instances.append(self)


def _install(monkeypatch):
    _FakeYOLO.instances = []
    monkeypatch.setattr(ultralytics, "YOLO", _FakeYOLO)
    model_cache.clear_cache()


def test_load_yolo_shares_by_path_and_imgsz(monkeypatch):
    _install(monkeypatch)
    a, cached_a = model_cache.load_yolo("weights.pt", 640)
    b, cached_b = model_cache.load_yolo("weights.pt", 640)
    assert a is b
    assert cached_a is False and cached_b is True
    assert len(_FakeYOLO.instances) == 1

    model_cache.load_yolo("other.pt", 640)
    assert len(_FakeYOLO.instances) == 2

    model_cache.load_yolo("weights.pt", 320)
    assert len(_FakeYOLO.instances) == 3  # imgsz is part of the key
    model_cache.clear_cache()


def test_fire_smoke_detector_reuses_shared_weights(monkeypatch, tmp_path):
    _install(monkeypatch)
    from detector.fire_smoke_detector import FireSmokeDetector

    weights = tmp_path / "fire_smoke_v2.pt"
    weights.write_bytes(b"stub")

    first = FireSmokeDetector(str(weights))
    second = FireSmokeDetector(str(weights))

    assert first._model is second._model
    assert len(_FakeYOLO.instances) == 1
    model_cache.clear_cache()


def test_ppe_detector_reuses_shared_weights(monkeypatch, tmp_path):
    _install(monkeypatch)
    from detector.ppe_detector import EXPECTED_CLASSES, PPEDetector

    class _PpeFake(_FakeYOLO):
        def __init__(self, path):
            super().__init__(path)
            self.names = dict(EXPECTED_CLASSES)

    monkeypatch.setattr(ultralytics, "YOLO", _PpeFake)
    model_cache.clear_cache()

    weights = tmp_path / "ppe_v1.pt"
    weights.write_bytes(b"stub")

    first = PPEDetector(str(weights))
    second = PPEDetector(str(weights))

    assert first._model is second._model
    assert len(_PpeFake.instances) == 1
    model_cache.clear_cache()


def test_crowd_detector_reuses_shared_weights(monkeypatch, tmp_path):
    _install(monkeypatch)

    class _PersonFake(_FakeYOLO):
        def __init__(self, path):
            super().__init__(path)
            self.names = {0: "person"}

    monkeypatch.setattr(ultralytics, "YOLO", _PersonFake)
    model_cache.clear_cache()
    from detector.crowd_detector import CrowdDetector

    weights = tmp_path / "crowd_v1.pt"
    weights.write_bytes(b"stub")

    first = CrowdDetector(str(weights))
    second = CrowdDetector(str(weights))

    assert first._model is second._model
    assert len(_PersonFake.instances) == 1
    model_cache.clear_cache()
