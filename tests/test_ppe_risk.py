"""PPE risk grader tests (ported from DMS test_ppe_risk, backend-free).

Covers vocabulary, grade boundaries, input shapes, thresholds, and the
emission-deferral marker. No model, no network.
"""

import pytest

from detector.ppe_risk import (
    MODEL_CLASS_NAMES,
    PPE_EMISSION_DEFERRED,
    VIOLATION_CLASSES,
    PpeRiskGrade,
    grade_frame,
    normalize_detections,
)


def person(conf=0.9):
    return {"class_name": "Person", "confidence": conf, "bounding_box": (0, 0, 10, 20)}


def violation(name="NO-Hardhat", conf=0.8):
    return {"class_name": name, "confidence": conf, "bounding_box": (0, 0, 10, 10)}


def test_emission_is_deferred():
    assert PPE_EMISSION_DEFERRED is True


def test_empty_frame_is_none():
    result = grade_frame([])
    assert result.grade == PpeRiskGrade.NONE
    assert result.violation_ratio == 0.0
    assert (result.violation_count, result.person_count) == (0, 0)


def test_compliant_crew_is_none():
    detections = [person(), person(),
                  {"class_name": "Hardhat", "confidence": 0.9, "bounding_box": (0, 0, 5, 5)},
                  {"class_name": "Safety Vest", "confidence": 0.9, "bounding_box": (0, 0, 5, 5)}]
    result = grade_frame(detections)
    assert result.grade == PpeRiskGrade.NONE
    assert result.other_count == 2


def test_boundary_ratios_take_the_higher_grade():
    assert grade_frame([person() for _ in range(5)] + [violation()]).grade == PpeRiskGrade.LOW  # 0.2
    assert grade_frame([person() for _ in range(4)] + [violation()]).grade == PpeRiskGrade.MEDIUM  # 0.25
    assert grade_frame([person(), person(), violation()]).grade == PpeRiskGrade.HIGH  # 0.5
    assert grade_frame([person(), violation()]).grade == PpeRiskGrade.CRITICAL  # 1.0
    assert grade_frame([violation()]).grade == PpeRiskGrade.NONE  # no persons: no data, no grade


def test_all_five_violation_classes_recognised():
    detections = [person() for _ in range(10)]
    detections += [violation(name) for name in VIOLATION_CLASSES]
    result = grade_frame(detections)
    assert result.violation_count == 5
    assert result.violation_ratio == 0.5
    assert result.grade == PpeRiskGrade.HIGH
    assert len(result.violations_by_class) == 5


def test_falls_reported_never_graded():
    detections = [{"class_name": "Fall-Detected", "confidence": 0.9,
                   "bounding_box": (0, 0, 10, 10)} for _ in range(3)]
    result = grade_frame(detections)
    assert result.fall_count == 3
    assert result.grade == PpeRiskGrade.NONE


def test_model_table_has_fourteen_classes():
    assert len(MODEL_CLASS_NAMES) == 14
    assert MODEL_CLASS_NAMES[11] == "Person"
    assert MODEL_CLASS_NAMES[0] == "Fall-Detected"


def test_bare_class_ids_resolve_through_model_table():
    detections = [
        {"class_id": 11, "confidence": 0.9},
        {"class_id": 8, "confidence": 0.8},
        {"class_id": 0, "confidence": 0.7},
    ]
    result = grade_frame(detections)
    assert (result.person_count, result.violation_count, result.fall_count) == (1, 1, 1)


def test_unknowable_detections_skipped_not_crash():
    detections = [{}, {"class_name": "  "}, {"class_id": 99}, None, "junk",
                  person()]
    result = grade_frame(detections)
    assert result.person_count == 1
    assert result.violation_count == 0


def test_misordered_thresholds_rejected():
    with pytest.raises(ValueError, match="medium < high"):
        grade_frame([person()], medium_ratio=0.9, high_ratio=0.5)
    with pytest.raises(ValueError, match="medium < high"):
        grade_frame([person()], critical_ratio=0.1)


def test_normalize_preserves_classes():
    normalized = normalize_detections([person(), violation("NO-Mask")])
    assert [d.class_name for d in normalized] == ["Person", "NO-Mask"]


def test_violations_without_persons_grade_none_not_critical():
    """Bare violations with nobody in view are no data, not a critical site.

    The counts stay on the result for the readout, but the grade must not
    reflect them: the old ratio-over-max(persons, 1) math turned two stray
    boxes into a phantom critical (seen live as a t=0.0 critical on an
    empty PPE.mp4 frame).
    """
    result = grade_frame([violation("NO-Hardhat"), violation("NO-Safety Vest")])
    assert result.grade == PpeRiskGrade.NONE
    assert result.person_count == 0
    assert result.violation_count == 2
    assert result.violation_ratio == 0.0
    assert result.violations_by_class == (("NO-Hardhat", 1), ("NO-Safety Vest", 1))


def test_falls_without_persons_grade_none():
    result = grade_frame(
        [{"class_name": "Fall-Detected", "confidence": 0.9,
          "bounding_box": (0, 0, 10, 10)},
         violation("NO-Hardhat")]
    )
    assert result.grade == PpeRiskGrade.NONE
    assert result.fall_count == 1
    assert result.violation_count == 1


def test_small_crew_keeps_ratio_grades():
    """With persons present the ratio semantics are unchanged: one person
    with one violation is a 1.0 ratio (critical), two people with one is
    0.5 (high, inclusive lower bound)."""
    assert grade_frame([person(), violation()]).grade == PpeRiskGrade.CRITICAL
    assert grade_frame([person(), person(), violation()]).grade == PpeRiskGrade.HIGH
