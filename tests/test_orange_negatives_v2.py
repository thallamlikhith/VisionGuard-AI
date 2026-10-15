"""Orange false-positive check, v2 only (Prototype-local).

Adapted from DMS ``tests/test_orange_negatives.py``: the DMS version runs
v1-vs-v2 over ``tests/orange_negatives/`` at conf=0.35, imgsz=640. Here only
``fire_smoke_v2.pt`` was copied (per plan: v1 rollback weights stay in DMS),
so this test runs v2 only, resolves hits through the weights' OWN class
table (id 0 = smoke, id 1 = fire), replays each video's frames in order
through the production ``FireSmokeGate`` (defaults 1.5 s / 2.5 s unchanged)
and reports per-100 rates. Empty input dir -> skip (never fail).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

CONF_THRESHOLD = 0.35
IMG_SIZE = 640
TOP_N = 10

NEGATIVES_DIR = Path(__file__).resolve().parent / "orange_negatives"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
WEIGHTS = (Path(__file__).resolve().parent.parent
           / "detector" / "models" / "fire_smoke_v2.pt")

_VIDEO_RE = re.compile(r"^(.+)_(\d{3})$")


def _input_images() -> list[Path]:
    if not NEGATIVES_DIR.is_dir():
        return []
    return sorted(p for p in NEGATIVES_DIR.iterdir()
                  if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def _video_of(image_path: Path) -> str:
    match = _VIDEO_RE.match(image_path.stem)
    return match.group(1) if match else "<other>"


def test_orange_false_positives_v2_only():
    images = _input_images()
    if not images:
        pytest.skip("tests/orange_negatives/ is empty")
    assert WEIGHTS.is_file(), f"weights missing: {WEIGHTS}"

    import cv2
    from ultralytics import YOLO

    from detector.fire_smoke_detector import check_class_table
    from detector.fire_smoke_gate import FireSmokeGate, extract_fire_smoke

    model = YOLO(str(WEIGHTS))
    check_class_table(model.names)  # 0 = smoke, 1 = fire
    own = {int(k): str(v).strip().casefold() for k, v in model.names.items()}

    videos = sorted({_video_of(p) for p in images})
    totals = {"fire": 0, "smoke": 0}
    by_video: dict[str, dict] = {v: {"fire": 0, "smoke": 0, "images": 0}
                                 for v in videos}
    top: list[dict] = []
    frames: dict[str, object] = {}
    ordered: dict[str, list[Path]] = {v: [] for v in videos}
    for image_path in images:
        video = _video_of(image_path)
        ordered[video].append(image_path)
        frame = cv2.imread(str(image_path))
        assert frame is not None, f"unreadable image: {image_path}"
        frames[image_path.name] = frame
        result = model.predict(frame, conf=CONF_THRESHOLD,
                               imgsz=IMG_SIZE, verbose=False)[0]
        for box in result.boxes or []:
            cname = own.get(int(box.cls[0]), "")
            if cname not in ("fire", "smoke"):
                continue
            conf = float(box.conf[0])
            totals[cname] += 1
            by_video[video][cname] += 1
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
            top.append({"video": video, "frame": image_path.name,
                        "class": cname, "confidence": conf,
                        "bbox_xyxy": [x1, y1, x2, y2]})
        by_video[video]["images"] += 1

    top.sort(key=lambda h: h["confidence"], reverse=True)
    top = top[:TOP_N]

    gate = FireSmokeGate()
    gate_result: dict[str, str] = {}
    for video in videos:
        crossed_at = None
        for image_path in sorted(ordered[video]):
            result = model.predict(frames[image_path.name], conf=CONF_THRESHOLD,
                                   imgsz=IMG_SIZE, verbose=False)[0]
            as_dicts = [{"class_name": own.get(int(b.cls[0]), ""),
                         "confidence": float(b.conf[0]),
                         "bbox_xyxy": [float(v) for v in b.xyxy[0].tolist()],
                         "track_id": None} for b in (result.boxes or [])]
            kept = extract_fire_smoke(as_dicts)
            by_type: dict[str, list] = {}
            for det in kept:
                by_type.setdefault(det["vision_alert_type"], []).append(det)
            crossed = gate.update(video, list(by_type))
            if crossed and crossed_at is None:
                crossed_at = f"{image_path.name} (types={','.join(sorted(crossed))})"
        gate_result[video] = (f"ALERT would fire at {crossed_at}"
                              if crossed_at else "no alert")

    n = len(images)

    def per100(hits: int, total: int) -> float:
        return round(hits / total * 100, 2) if total else 0.0

    print(f"\n=== orange FP v2-only ({n} images, conf={CONF_THRESHOLD}) ===")
    print(f"  v2: fire {totals['fire']} ({per100(totals['fire'], n)}/100), "
          f"smoke {totals['smoke']} ({per100(totals['smoke'], n)}/100)")
    for video in videos:
        v = by_video[video]
        print(f"    {video}: {v['images']} imgs, fire {v['fire']}, smoke {v['smoke']}")
    print("  gate replay (production thresholds, unchanged):")
    for video in videos:
        print(f"    {video}: {gate_result[video]}")
