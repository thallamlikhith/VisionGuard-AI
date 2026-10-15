"""PPE golden regression (lean port of DMS test_ppe_regression).

Same fixtures, same comparison rules (exact count + ordered class pairs,
confidence tol 1e-3, bbox tol 1.0 px), but through the lean
``detector.ppe_detector.PPEDetector`` instead of the DMS VisionService
(which is deliberately not copied). The 14-class vocabulary round-trip is
checked first.
"""

import json
from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures" / "ppe_golden"
GOLDEN_IMAGES = sorted(FIXTURES_DIR.glob("*.jpg"))

CONF_TOL = 1e-3
BBOX_TOL = 1.0

EXPECTED_CLASSES = {
    0: "Fall-Detected",
    1: "Gloves",
    2: "Goggles",
    3: "Hardhat",
    4: "Ladder",
    5: "Mask",
    6: "NO-Gloves",
    7: "NO-Goggles",
    8: "NO-Hardhat",
    9: "NO-Mask",
    10: "NO-Safety Vest",
    11: "Person",
    12: "Safety Cone",
    13: "Safety Vest",
}


def test_golden_images_present():
    assert len(GOLDEN_IMAGES) >= 3, (
        f"Expected at least 3 golden images in {FIXTURES_DIR}, found {len(GOLDEN_IMAGES)}"
    )


def test_ppe_class_vocabulary_roundtrip():
    from detector.ppe_detector import EXPECTED_CLASSES as ACTUAL
    from detector.ppe_detector import check_class_table

    assert ACTUAL == EXPECTED_CLASSES
    check_class_table({i: name for i, name in EXPECTED_CLASSES.items()})
    assert len(ACTUAL) == 14


@pytest.mark.parametrize("image_path", GOLDEN_IMAGES, ids=[p.stem for p in GOLDEN_IMAGES])
def test_ppe_golden_regression(image_path):
    import cv2

    from detector.ppe_detector import PPEDetector

    detector = PPEDetector()
    frame = cv2.imread(str(image_path))
    assert frame is not None, f"unreadable image: {image_path}"
    actual = []
    for d in detector.predict(frame):
        actual.append({
            "class_id": d["class_id"],
            "class_name": d["class_name"],
            "confidence": round(float(d["confidence"]), 6),
            "bounding_box": [round(float(v), 2) for v in d["bbox_xyxy"]],
        })

    expected_file = image_path.with_suffix(".expected.json")
    assert expected_file.exists(), f"Missing baseline {expected_file.name}"
    expected = json.loads(expected_file.read_text(encoding="utf-8"))["detections"]

    assert len(actual) == len(expected), (
        f"{image_path.name}: detection count changed "
        f"{len(expected)} -> {len(actual)}\nexpected: {expected}\nactual: {actual}"
    )
    for idx, (exp, act) in enumerate(zip(expected, actual)):
        assert act["class_id"] == exp["class_id"], (
            f"{image_path.name} det#{idx}: class_id {exp['class_id']} -> {act['class_id']}"
        )
        assert act["class_name"] == exp["class_name"], (
            f"{image_path.name} det#{idx}: class_name '{exp['class_name']}' -> '{act['class_name']}'"
        )
        assert abs(act["confidence"] - exp["confidence"]) <= CONF_TOL, (
            f"{image_path.name} det#{idx} ({exp['class_name']}): confidence "
            f"{exp['confidence']} -> {act['confidence']} (tol {CONF_TOL})"
        )
        for ci, (e, a) in enumerate(zip(exp["bounding_box"], act["bounding_box"])):
            assert abs(a - e) <= BBOX_TOL, (
                f"{image_path.name} det#{idx} ({exp['class_name']}) bbox[{ci}]: "
                f"{e} -> {a} (tol {BBOX_TOL}px)"
            )
