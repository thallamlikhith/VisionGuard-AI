"""PPE risk grader (copied from DMS AI_services).

Source: DMS ``AI_services/app/vision/streaming/ppe_risk.py`` (read-only).
Adapted: the DMS ``app.shared.core.config.settings`` import is replaced by
explicit default arguments (same plan values: medium 0.25, high 0.5,
critical 1.0). No DMS dependency; pure functions, no I/O.

Preserved exactly:
- vocabulary: persons ``Person``; violations ``NO-Gloves`` (6),
  ``NO-Goggles`` (7), ``NO-Hardhat`` (8), ``NO-Mask`` (9),
  ``NO-Safety Vest`` (10); ``Fall-Detected`` reported, never graded;
  everything else ``other``
- full 14-class table in model order for bare ``class_id`` detections
- grade math: violation_ratio = violations / max(persons, 1); boundaries
  are inclusive lower bounds of the upper grade
- ``PPE_EMISSION_DEFERRED = True``: grade into stats only, emit nowhere
  (product decision, mirrors DMS plan §9.1 / contract item 5)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional

__all__ = [
    "PERSON_CLASS",
    "FALL_CLASS",
    "VIOLATION_CLASSES",
    "MODEL_CLASS_NAMES",
    "PPE_EMISSION_DEFERRED",
    "PpeRiskGrade",
    "PpeDetection",
    "PpeRiskResult",
    "normalize_detections",
    "grade_frame",
]

PERSON_CLASS = "Person"
FALL_CLASS = "Fall-Detected"
VIOLATION_CLASSES = (
    "NO-Gloves",
    "NO-Goggles",
    "NO-Hardhat",
    "NO-Mask",
    "NO-Safety Vest",
)
MODEL_CLASS_NAMES = (
    "Fall-Detected",  # 0
    "Gloves",  # 1
    "Goggles",  # 2
    "Hardhat",  # 3
    "Ladder",  # 4
    "Mask",  # 5
    "NO-Gloves",  # 6
    "NO-Goggles",  # 7
    "NO-Hardhat",  # 8
    "NO-Mask",  # 9
    "NO-Safety Vest",  # 10
    "Person",  # 11
    "Safety Cone",  # 12
    "Safety Vest",  # 13
)

#: PPE-live emission is deferred, not unbuilt: grade into stats, emit
#: nowhere, until a routing decision exists. Flipping this flag alone
#: changes nothing (there is no emit hook wired to it).
PPE_EMISSION_DEFERRED = True

#: Plan defaults (DMS PPE_RISK_*_RATIO): strictly increasing, positive.
DEFAULT_MEDIUM_RATIO = 0.25
DEFAULT_HIGH_RATIO = 0.5
DEFAULT_CRITICAL_RATIO = 1.0


class PpeRiskGrade(str):
    """Site-level grade for one frame. ``NONE`` = no violations seen."""

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass(frozen=True)
class PpeDetection:
    class_name: str
    confidence: float = 1.0


@dataclass(frozen=True)
class PpeRiskResult:
    grade: str
    violation_ratio: float
    violation_count: int
    person_count: int
    fall_count: int
    other_count: int
    violations_by_class: tuple = ()


def _coerce_name(detection: Any) -> Optional[str]:
    if isinstance(detection, dict):
        name = detection.get("class_name")
        if isinstance(name, str) and name.strip():
            return name.strip()
        class_id = detection.get("class_id")
        if isinstance(class_id, bool):
            return None
        if isinstance(class_id, int) and 0 <= class_id < len(MODEL_CLASS_NAMES):
            return MODEL_CLASS_NAMES[class_id]
        return None
    name = getattr(detection, "class_name", None)
    if isinstance(name, str) and name.strip():
        return name.strip()
    class_id = getattr(detection, "class_id", None)
    if isinstance(class_id, bool):
        return None
    if isinstance(class_id, int) and 0 <= class_id < len(MODEL_CLASS_NAMES):
        return MODEL_CLASS_NAMES[class_id]
    return None


def _coerce_confidence(detection: Any) -> float:
    if isinstance(detection, dict):
        confidence = detection.get("confidence", 1.0)
    else:
        confidence = getattr(detection, "confidence", 1.0)
    try:
        value = float(confidence)
    except (TypeError, ValueError):
        return 0.0
    if value != value:  # NaN
        return 0.0
    return min(max(value, 0.0), 1.0)


def normalize_detections(detections: Iterable[Any]) -> list:
    """Accept dict / RawDetection-like / attribute shapes; skip unknowable."""
    normalized = []
    for detection in detections or ():
        name = _coerce_name(detection)
        if name is None:
            continue
        normalized.append(PpeDetection(
            class_name=name, confidence=_coerce_confidence(detection)
        ))
    return normalized


def _canonical_violation(name: str) -> Optional[str]:
    folded = " ".join(name.strip().split()).casefold()
    for canonical in VIOLATION_CLASSES:
        if folded == canonical.casefold():
            return canonical
    return None


def grade_frame(
    detections: Iterable[Any],
    *,
    medium_ratio: float = DEFAULT_MEDIUM_RATIO,
    high_ratio: float = DEFAULT_HIGH_RATIO,
    critical_ratio: float = DEFAULT_CRITICAL_RATIO,
) -> PpeRiskResult:
    """Grade one frame's PPE detections. Pure: no I/O, no clock, no globals."""
    medium = float(medium_ratio)
    high = float(high_ratio)
    critical = float(critical_ratio)
    if not (0.0 < medium < high < critical):
        raise ValueError(
            "PPE risk boundaries must satisfy 0 < medium < high < critical, "
            f"got medium={medium}, high={high}, critical={critical}"
        )

    persons = 0
    falls = 0
    other = 0
    violations: dict = {}
    for detection in normalize_detections(detections):
        folded = " ".join(detection.class_name.split()).casefold()
        if folded == PERSON_CLASS.casefold():
            persons += 1
        elif folded == FALL_CLASS.casefold():
            falls += 1
        elif _canonical_violation(detection.class_name) is not None:
            canonical = _canonical_violation(detection.class_name)
            assert canonical is not None
            violations[canonical] = violations.get(canonical, 0) + 1
        else:
            other += 1

    violation_count = sum(violations.values())
    # No person in the frame means no grade: a ratio over zero persons is
    # not a measurement (the max(persons, 1) guard below would turn bare
    # violations into a phantom critical). The counts stay in the result
    # for the readout, but they must not set the grade.
    if persons == 0:
        return PpeRiskResult(
            grade=PpeRiskGrade.NONE,
            violation_ratio=0.0,
            violation_count=violation_count,
            person_count=0,
            fall_count=falls,
            other_count=other,
            violations_by_class=tuple(sorted(violations.items())),
        )
    ratio = (violation_count / max(persons, 1)) if violation_count else 0.0
    if violation_count == 0:
        grade = PpeRiskGrade.NONE
    elif ratio < medium:
        grade = PpeRiskGrade.LOW
    elif ratio < high:
        grade = PpeRiskGrade.MEDIUM
    elif ratio < critical:
        grade = PpeRiskGrade.HIGH
    else:
        grade = PpeRiskGrade.CRITICAL
    return PpeRiskResult(
        grade=grade,
        violation_ratio=round(ratio, 4),
        violation_count=violation_count,
        person_count=persons,
        fall_count=falls,
        other_count=other,
        violations_by_class=tuple(sorted(violations.items())),
    )
