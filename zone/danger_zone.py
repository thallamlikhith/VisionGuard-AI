"""Danger zone polygon + per-worker state machine (OUTSIDE / INSIDE / ALERT).

This module has no OpenCV / YOLO dependency on purpose: the whole safety
logic is plain Python and is unit tested in tests/test_zone.py.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Iterable, Optional

OUTSIDE = "OUTSIDE"
INSIDE = "INSIDE"
ALERT = "ALERT"

SAFE = "SAFE"

ZONE_ENTRY = "ZONE_ENTRY"
ENTRY_ALERT = "ENTRY_ALERT"
DWELL_ALERT = "DWELL_ALERT"
ZONE_EXIT = "ZONE_EXIT"
CAPACITY_ALERT = "CAPACITY_ALERT"
CAPACITY_SAFE = "CAPACITY_SAFE"

# Why an event was written, stored on the row next to the type. ENTRY covers
# the ZONE_ENTRY history row and, in INSTANT mode, the ENTRY_ALERT itself;
# DWELL the dwell alerts; CAPACITY the capacity rows. ZONE_EXIT carries none:
# it ends a visit rather than starting, breaching or repeating one.
REASON_ENTRY = "ENTRY"
REASON_DWELL = "DWELL"
REASON_CAPACITY = "CAPACITY"

# How long a worker has to be continuously outside a zone before it counts as
# having left. A worker standing on the boundary crosses it back and forth every
# few frames; without this the dwell clock restarted on every crossing, so the
# worker could never reach the dwell limit, and each crossing wrote a
# ZONE_EXIT/ZONE_ENTRY pair. Entering stays immediate: only the exit is held
# back, and the dwell clock keeps running through the grace.
EXIT_GRACE = 0.6

# A point counts as inside only when it is more than this many pixels inside the
# boundary. A worker standing on the line has their anchor flicker across it as
# the box jitters, which showed up as the count changing by one for a frame or
# two. The margin is a setting (config.yaml danger_zone.zone_margin_px); 0 means
# the plain point-in-polygon test.
ZONE_MARGIN_PX = 5.0


class ZoneError(Exception):
    """Raised when a danger zone polygon cannot be built."""


def point_in_polygon(point: tuple[float, float], polygon: list[tuple[int, int]]) -> bool:
    """Ray casting point-in-polygon test (horizontal ray, no imports needed)."""
    x, y = point
    inside = False
    count = len(polygon)
    for i in range(count):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % count]
        # does the edge (x1,y1)-(x2,y2) straddle the horizontal line of our point?
        if (y1 > y) != (y2 > y):
            # at what x does that edge cross the line y = our y?
            x_cross = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < x_cross:
                inside = not inside
    return inside


def _distance_to_edge(
    point: tuple[float, float], a: tuple[int, int], b: tuple[int, int]
) -> float:
    """Shortest distance from a point to the segment a-b, in pixels."""
    px, py = point
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    if dx == 0 and dy == 0:
        return ((px - ax) ** 2 + (py - ay) ** 2) ** 0.5
    t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    return ((px - (ax + t * dx)) ** 2 + (py - (ay + t * dy)) ** 2) ** 0.5


def point_in_polygon_margin(
    point: tuple[float, float], polygon: list[tuple[int, int]], margin: float
) -> bool:
    """True when the point is inside the polygon by more than `margin` pixels.

    A margin of 0 is the plain ray-cast test. Otherwise the point also has to be
    further than `margin` from every edge, so a point sitting on (or just across)
    the boundary does not flip inside and outside as the box jitters.
    """
    if not point_in_polygon(point, polygon):
        return False
    if margin <= 0:
        return True
    count = len(polygon)
    nearest = min(
        _distance_to_edge(point, polygon[i], polygon[(i + 1) % count])
        for i in range(count)
    )
    return nearest > margin


def bottom_center(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    """Feet position of a box: middle of the bottom edge."""
    x1, _, x2, y2 = bbox
    return ((x1 + x2) // 2, y2)


def center(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    """Middle of a box."""
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) // 2, (y1 + y2) // 2)


def _bbox_area(bbox: tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1) * max(0, y2 - y1)


def _bbox_overlaps_zone(bbox: tuple[int, int, int, int], polygon: list[tuple[int, int]], threshold: float) -> bool:
    if not polygon:
        return False
    try:
        import cv2
        import numpy as np

        x1, y1, x2, y2 = bbox
        box_poly = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
        zone_poly = np.array([[int(p[0]), int(p[1])] for p in polygon], dtype=np.float32)
        if zone_poly.shape[0] < 3:
            return False
        inter = cv2.intersectConvexConvex(box_poly, zone_poly)[0]
        if inter <= 0:
            return False
        box_area = _bbox_area(bbox)
        if box_area <= 0:
            return False
        return (inter / box_area) > threshold
    except Exception:
        return False


ANCHORS = {"bottom_center": bottom_center, "center": center}

# --- what makes a danger zone usable ---------------------------------------
# A zone that is technically a polygon but covers nothing (all corners on one
# line, a bow tie, two identical corners) is worse than no zone at all: it
# looks drawn on screen yet can never contain anybody. These are the rules the
# API rejects and the UI checks with, so both agree on what "valid" means.
MIN_ZONE_POINTS = 3
# Paint-style drawing produces rounded and freehand shapes, which need far
# more corners than the 6 a hand-placed polygon did (a circle is 24).
MAX_ZONE_POINTS = 32
# a sliver of the frame is never a real danger area
MIN_ZONE_AREA_RATIO = 0.005


def polygon_area(points: list[tuple[int, int]]) -> float:
    """Shoelace area in square pixels."""
    total = 0.0
    count = len(points)
    for index in range(count):
        x1, y1 = points[index]
        x2, y2 = points[(index + 1) % count]
        total += x1 * y2 - x2 * y1
    return abs(total) / 2.0


def _orientation(a, b, c) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _segments_cross(a, b, c, d) -> bool:
    """True only for a proper crossing, so shared corners are not a crossing."""
    d1 = _orientation(c, d, a)
    d2 = _orientation(c, d, b)
    d3 = _orientation(a, b, c)
    d4 = _orientation(a, b, d)
    return d1 * d2 < 0 and d3 * d4 < 0


def polygon_is_simple(points: list[tuple[int, int]]) -> bool:
    """False when two edges that do not share a corner cross over each other."""
    count = len(points)
    for i in range(count):
        a, b = points[i], points[(i + 1) % count]
        for j in range(i + 1, count):
            if j == i or (j + 1) % count == i or (i + 1) % count == j:
                continue  # neighbours share a corner by definition
            if _segments_cross(a, b, points[j], points[(j + 1) % count]):
                return False
    return True


def drop_repeats(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge corners that sit on the same pixel as the one before them.

    A shape clamped against the edge of the frame lands several corners on the
    same point, and a zero length edge is not a polygon. Only neighbours are
    merged; a shape that folds back onto an earlier corner is a different
    mistake and is left alone so it can be reported.
    """
    kept: list[tuple[int, int]] = []
    for point in points:
        if kept and kept[-1] == point:
            continue
        kept.append(point)
    # the shape closes from the last corner back to the first
    if len(kept) > 1 and kept[0] == kept[-1]:
        kept.pop()
    return kept


def zone_problem(
    points: list[tuple[int, int]],
    width: int = 0,
    height: int = 0,
) -> Optional[str]:
    """Why this polygon cannot be used, or None when it is fine.

    An empty list is not a problem: it is how the UI says "no zone yet".
    width/height of 0 means the frame size is unknown, so the checks that need
    a frame are skipped instead of failing against a made-up 0x0 picture.
    """
    if not points:
        return None

    # corners doubled up by clamping are not a real problem, so they go before
    # anything is measured: a circle dragged over the frame edge is still a
    # circle once its repeated border points are dropped
    points = drop_repeats([(int(point[0]), int(point[1])) for point in points])
    if not points:
        return None

    count = len(points)
    if count < MIN_ZONE_POINTS:
        return f"A danger zone needs at least {MIN_ZONE_POINTS} points, got {count}."
    if count > MAX_ZONE_POINTS:
        return f"A danger zone can have at most {MAX_ZONE_POINTS} points, got {count}."

    for index, point in enumerate(points):
        x, y = point
        if x < 0 or y < 0:
            return f"Point {index + 1} ({x}, {y}) has a negative coordinate."
        if width > 0 and height > 0 and (x > width or y > height):
            return f"Point {index + 1} ({x}, {y}) is outside the {width}x{height} video."

    seen: dict[tuple[int, int], int] = {}
    for index, point in enumerate(points):
        if point in seen:
            return (
                f"Point {index + 1} and point {seen[point] + 1} are at the same "
                f"position - the corners must all be different."
            )
        seen[point] = index

    if not polygon_is_simple(points):
        return "The zone crosses over itself - the corners have to go around in order."

    if width > 0 and height > 0:
        area = polygon_area(points)
        ratio = area / float(width * height)
        if ratio < MIN_ZONE_AREA_RATIO:
            return (
                f"The zone covers only {ratio * 100:.1f}% of the frame - it has to "
                f"cover at least {MIN_ZONE_AREA_RATIO * 100:.1f}%."
            )

    return None


def _is_point(value) -> bool:
    """True for a single [x, y] corner, False for a whole polygon."""
    try:
        return len(value) == 2 and all(isinstance(item, (int, float)) for item in value)
    except TypeError:
        return False


def _normalise_polygons(value) -> list[list[tuple[int, int]]]:
    """Turn a flat point list or a list of polygons into polygons of ints.

    Both shapes are accepted so the new multi-zone UI and the old single-zone
    config can share one code path:
        [[10, 10], [60, 10]]             -> one zone
        [[[10, 10], [60, 10]], [[..]]]   -> two zones
    """
    if not value:
        return []
    items = list(value)
    if all(_is_point(item) for item in items):
        return [[(int(x), int(y)) for x, y in items]]
    polygons: list[list[tuple[int, int]]] = []
    for item in items:
        if _is_point(item):
            # a mix of corners and polygons is not something we can guess at
            raise ZoneError("a danger zone is either a list of points or a list of zones")
        polygon = [(int(x), int(y)) for x, y in item]
        if polygon:
            polygons.append(polygon)
    return polygons


@dataclass
class ZoneRule:
    """One named danger area and the rules that belong to it alone.

    `mode`, `repeat_alert_seconds` and the anchor point stay global: they are
    about the machine and how it is watched, not about one area. What is per
    zone is the name, how many people are allowed, how long they may stand
    there, and whether the zone is watched at all.

    The shape lives on DangerZone.polygons and only there. `points` here is a
    copy kept so a rule can be read on its own, and it defaults to empty so a
    caller that already holds the geometry elsewhere (the pipeline, the API) can
    pass no points and let the monitor line the rules up with what it has.
    """

    name: str
    points: list[tuple[int, int]] = field(default_factory=list)
    allowed_people: int = 0
    dwell_seconds: float = 5.0
    enabled: bool = True
    # True while this zone has never had its own allowance or dwell limit typed
    # in, so it still tracks the global defaults. Kept as a flag rather than
    # worked out by comparing numbers, because a zone whose rule happens to
    # equal the default would otherwise claim it is following it.
    using_defaults: bool = True
    count_by: str = "foot_point"
    overlap_threshold: float = 0.10

    @property
    def usable(self) -> bool:
        """A zone is only monitored when it is on and can enclose somebody."""
        return self.enabled and len(self.points) >= 3


@dataclass
class Presence:
    """One worker inside one zone: a clock that belongs to that pair alone.

    A person standing in two overlapping zones has one Presence per zone, each
    with its own entry time, its own state and its own dwell. That is what lets
    the two zones alert independently and what stops one zone's clock from
    being reset by a step into the other.
    """

    zone_index: int
    entered_at: Optional[float] = None  # timeline seconds (video clock)
    state: str = OUTSIDE
    dwell_seconds: float = 0.0
    last_alert_at: Optional[float] = None
    entered_wall: Optional[float] = None
    alerted_wall: Optional[float] = None
    # set on the first frame the anchor is seen outside this zone while the exit
    # grace runs; cleared again the moment the worker is back inside
    outside_since: Optional[float] = None  # timeline seconds
    outside_wall: Optional[float] = None  # unix timestamp of that first frame


@dataclass
class Worker:
    """Live state of one tracked worker.

    `presences` is the truth: one entry per zone the worker is standing in.
    The single set of fields below is a copy of the zone the worker is
    ATTRIBUTED to (see ZoneMonitor.attributed_zone), refreshed every frame.
    They exist because the box drawing, the database-free CLI output and a good
    deal of the test suite read them, and because one worker still has one
    headline state for one row on screen. They are never the source for /state.
    """

    worker_id: int
    state: str = OUTSIDE
    entered_at: Optional[float] = None  # timeline seconds (video clock)
    last_alert_at: Optional[float] = None  # timeline seconds
    entered_wall: Optional[float] = None  # unix timestamp of the entry, for the database
    alerted_wall: Optional[float] = None  # unix timestamp of the last alert
    point: tuple[int, int] = (0, 0)
    bbox: Optional[tuple[int, int, int, int]] = None
    confidence: float = 0.0
    dwell_seconds: float = 0.0
    missing_frames: int = 0  # how many frames in a row the track was not seen
    # the zone this worker's fields above were copied from
    zone_index: int = 0
    presences: dict[int, Presence] = field(default_factory=dict)

    @property
    def label(self) -> str:
        return f"Worker #{self.worker_id}"

    @property
    def inside(self) -> bool:
        return any(p.state in (INSIDE, ALERT) for p in self.presences.values())


@dataclass
class Event:
    """Something worth storing in SQLite.

    `zone_index` / `zone_name` say which danger area the event is about. A
    worker in two overlapping zones produces one event per zone, and without
    these two fields those rows could not be told apart in the database.
    """

    worker_id: int
    machine_id: str
    event_type: str
    entered_at: Optional[float] = None
    alert_at: Optional[float] = None
    dwell_seconds: float = 0.0
    zone_index: Optional[int] = None
    zone_name: str = ""
    reason: Optional[str] = None
    # how many people were inside the zone when the row was written. Only
    # capacity rows carry one: a capacity alert is about a crowd, not a
    # person, so "Worker #-1" is never a useful thing to show for it.
    inside: Optional[int] = None

    def to_dict(self) -> dict:
        # zone_index / zone_name are stored alongside the event: without them a
        # worker in two overlapping zones would produce rows that cannot be told
        # apart once they leave the state machine.
        return {
            "worker_id": self.worker_id,
            "machine_id": self.machine_id,
            "event_type": self.event_type,
            "entered_at": self.entered_at,
            "alert_at": self.alert_at,
            "dwell_seconds": self.dwell_seconds,
            "zone_index": self.zone_index,
            "zone_name": self.zone_name,
            "reason": self.reason,
            "inside": self.inside,
        }


@dataclass(init=False)
class DangerZone:
    """One or more polygons in pixel coordinates.

    A machine can have several danger areas (a conveyor and a loading bay, for
    example), so this holds a LIST of polygons. This class is geometry only: it
    does not know a zone's name, its allowance or its dwell limit. Those live on
    ZoneMonitor's ZoneRule list, index aligned with `polygons`.

    For backward compatibility it still accepts one flat point list - the shape
    the older config.yaml, SQLite rows and CLI use:

        DangerZone([(0, 0), (10, 0), (10, 10)])       # one zone
        DangerZone([[[0, 0], ...], [[50, 50], ...]])  # two zones
        DangerZone(zones=[[[0, 0], ...]])             # explicit

    An empty list means "zone not configured yet": contains() is then always
    False so the video can still be previewed before anything is drawn.
    """

    polygons: list[list[tuple[int, int]]] = field(default_factory=list)

    def __init__(self, points=None, zones=None) -> None:
        source = zones if zones is not None else points
        # corners doubled up on the frame edge are merged away here, so what is
        # monitored is the shape the user meant rather than its flat border
        self.polygons = [
            drop_repeats(polygon) for polygon in _normalise_polygons(source)
        ]
        self.polygons = [polygon for polygon in self.polygons if polygon]
        for index, polygon in enumerate(self.polygons):
            if len(polygon) < 3:
                raise ZoneError(
                    f"danger zone {index + 1} needs at least 3 points, got {len(polygon)}"
                )

    @property
    def points(self) -> list[tuple[int, int]]:
        """The first polygon, for the many places that only ever had one."""
        return self.polygons[0] if self.polygons else []

    def __len__(self) -> int:
        return len(self.polygons)

    @property
    def active(self) -> bool:
        return bool(self.polygons)

    def contains(self, point: tuple[float, float]) -> bool:
        """True when the point falls inside ANY of the polygons."""
        return any(
            len(polygon) >= 3 and point_in_polygon(point, polygon) for polygon in self.polygons
        )


class ZoneMonitor:
    """State machine for every tracked worker, per danger zone.

    update_all() is called once per frame with all workers currently seen.
    A worker that disappears from the frame counts as having left the zone,
    which resets its dwell timer.

    Every zone is evaluated on its own people: one allowance and one dwell
    limit per zone, so a worker counted in zone 1 never uses up zone 2's
    allowance. A worker standing in two overlapping zones is in both, with one
    clock per zone, and each event says which zone it belongs to.
    """

    def __init__(
        self,
        zone: Optional[DangerZone] = None,
        machine_id: str = "machine",
        mode: str = "TIMER",
        dwell_seconds: float = 5.0,
        repeat_alert_seconds: float = 10.0,
        lost_timeout_frames: int = 15,
        allowed_people: int = 0,
        zone_rules: Optional[list[ZoneRule]] = None,
        exit_grace_seconds: float = EXIT_GRACE,
        zone_margin_px: float = ZONE_MARGIN_PX,
    ) -> None:
        mode = str(mode).upper()
        # LIMIT ("Only when over the limit") needs no special-casing below:
        # entries stay INSIDE like TIMER (never ENTRY_ALERT) and _advance()
        # only raises dwell alerts in TIMER, so LIMIT gets capacity alerts
        # alone.
        if mode not in ("INSTANT", "TIMER", "LIMIT"):
            raise ZoneError(f"unsupported alert mode: {mode}")

        self.machine_id = machine_id
        self.mode = mode
        self.repeat_alert_seconds = float(repeat_alert_seconds)
        self.lost_timeout_frames = int(lost_timeout_frames)
        # a worker must be outside this long before a ZONE_EXIT is written. 0
        # means "exit on the first outside frame", the behaviour before the
        # grace existed.
        self.exit_grace_seconds = max(0.0, float(exit_grace_seconds))
        # a point must be this far inside the boundary to count as in the zone
        self.zone_margin_px = max(0.0, float(zone_margin_px))

        # the global values stay the DEFAULTS for a zone that has not been given
        # its own. `dwell_seconds` / `allowed_people` are still readable here
        # because they are what a zone with no rules of its own uses.
        self.dwell_seconds = float(dwell_seconds)
        self.allowed_people = max(0, int(allowed_people))

        # one rule per polygon, index aligned with DangerZone.polygons
        self.zones: list[ZoneRule] = []
        self._zone = DangerZone(zones=[])
        self.workers: dict[int, Worker] = {}

        # capacity is a property of one zone, not of the machine: every zone
        # counts its own people against its own allowance and keeps its own
        # alert, so zone 1 going quiet can never clear zone 2's alarm.
        self.inside_by_zone: dict[int, int] = {}
        self.capacity_by_zone: dict[int, bool] = {}
        self.message_by_zone: dict[int, str] = {}

        # aggregates over all the zones, kept for the callers that only ever had
        # one zone: the drawing code, the CLI and a good deal of the tests.
        self.inside_count = 0
        self.capacity_alert = False
        self.alert_message = ""

        if zone is not None:
            self._zone = zone
        if zone_rules:
            self.set_zone_rules(zone_rules)
        self._sync_zones()

    # -- zone set ----------------------------------------------------------

    @property
    def zone(self) -> DangerZone:
        """The polygons as one shape, for the callers that predate named zones."""
        return self._zone

    @zone.setter
    def zone(self, value: DangerZone) -> None:
        self._zone = value

    def zone_name(self, index: int) -> str:
        if 0 <= index < len(self.zones):
            return self.zones[index].name or f"Zone {index + 1}"
        return f"Zone {index + 1}"

    def _sync_zones(self) -> None:
        """Line the rules up with the polygons that are actually there.

        The geometry is the source of truth, so it decides how many zones there
        are. What each index already knows - the name, the allowance, the dwell
        limit, whether it is on - is kept; a zone beyond the end of the list
        starts from the global defaults. Called from reset_timing() because that
        is the one moment the geometry is swapped under a running camera.
        """
        polygons = self._zone.polygons
        known = list(self.zones)
        synced: list[ZoneRule] = []
        for index, polygon in enumerate(polygons):
            if index < len(known):
                rule = known[index]
                synced.append(
                    ZoneRule(
                        name=rule.name or f"Zone {index + 1}",
                        points=polygon,
                        allowed_people=rule.allowed_people,
                        dwell_seconds=rule.dwell_seconds,
                        enabled=rule.enabled,
                        using_defaults=rule.using_defaults,
                        count_by=getattr(rule, "count_by", "foot_point"),
                        overlap_threshold=getattr(rule, "overlap_threshold", 0.10),
                    )
                )
            else:
                synced.append(
                    ZoneRule(
                        name=f"Zone {index + 1}",
                        points=polygon,
                        allowed_people=self.allowed_people,
                        dwell_seconds=self.dwell_seconds,
                        using_defaults=True,
                        count_by="foot_point",
                        overlap_threshold=0.10,
                    )
                )
        self.zones = synced

    def set_zone_rules(self, rules: list[ZoneRule]) -> None:
        """Apply names, allowances, dwell limits and on/off per zone index.

        Called while the camera runs, so a zone that is switched off stops
        counting anybody straight away. The geometry is not touched here.
        """
        self.zones = [
            ZoneRule(
                name=(rule.name or f"Zone {index + 1}").strip(),
                points=list(self.zones[index].points) if index < len(self.zones) else [],
                allowed_people=max(0, int(rule.allowed_people)),
                dwell_seconds=max(0.0, float(rule.dwell_seconds)),
                enabled=bool(rule.enabled),
                using_defaults=bool(rule.using_defaults),
                count_by=str(getattr(rule, "count_by", "foot_point")) if hasattr(rule, "count_by") else (rule.get("count_by", "foot_point") if isinstance(rule, dict) else "foot_point"),
                overlap_threshold=float(getattr(rule, "overlap_threshold", 0.10)) if hasattr(rule, "overlap_threshold") else (rule.get("overlap_threshold", 0.10) if isinstance(rule, dict) else 0.10),
            )
            for index, rule in enumerate(rules)
        ]

    def apply_defaults(
        self,
        allowed_people: Optional[int] = None,
        dwell_seconds: Optional[float] = None,
        zone_rules: Optional[list[ZoneRule]] = None,
    ) -> None:
        """Change the global defaults, and the per-zone rules when sent.

        A zone that is still flagged `using_defaults` follows the new value; one
        that has been given a rule of its own keeps it, because that is the
        whole point of having typed it in.
        """
        if allowed_people is not None:
            self.allowed_people = max(0, int(allowed_people))
        if dwell_seconds is not None:
            self.dwell_seconds = float(dwell_seconds)

        if zone_rules is not None:
            self.set_zone_rules(zone_rules)
            return

        for rule in self.zones:
            if rule.using_defaults:
                rule.allowed_people = self.allowed_people
                rule.dwell_seconds = self.dwell_seconds

    def zones_containing(self, point: tuple[float, float], bbox: Optional[tuple[int, int, int, int]] = None) -> list[int]:
        """Every zone the point is inside, lowest index first.

        A zone that is switched off or has too few corners to enclose anybody is
        skipped: it is not watched, so nothing inside it may be counted.
        """
        contained = []
        for index, rule in enumerate(self.zones):
            if not rule.usable:
                continue
            inside = False
            if getattr(rule, "count_by", "foot_point") == "overlap":
                if bbox is not None:
                    inside = self._bbox_overlaps_zone(bbox, rule.points, getattr(rule, "overlap_threshold", 0.10))
            else:
                inside = point_in_polygon_margin(point, rule.points, self.zone_margin_px)
            if inside:
                contained.append(index)
        return contained

    def contains(self, point: tuple[float, float]) -> bool:
        """True when the point falls inside ANY zone. Kept for old callers."""
        return bool(self.zones_containing(point))

    # -- attribution -------------------------------------------------------

    def attributed_zone(self, worker: Worker) -> int:
        """The one zone this worker's reported state belongs to.

        An alarming zone wins, because that is the thing the operator has to
        act on; between two alarming zones the longer-standing one is the worse
        news. Otherwise the lowest zone containing the worker, which keeps the
        choice stable from frame to frame. A worker outside everything is
        attributed to zone 0 and reads OUTSIDE there.
        """
        alerting = [
            presence
            for presence in worker.presences.values()
            if presence.state == ALERT
        ]
        if alerting:
            return max(alerting, key=lambda p: (p.dwell_seconds, -p.zone_index)).zone_index
        inside = sorted(
            presence.zone_index
            for presence in worker.presences.values()
            if presence.state in (INSIDE, ALERT)
        )
        return inside[0] if inside else 0

    def _sync_worker(self, worker: Worker) -> None:
        """Copy the attributed zone's presence onto the single headline fields."""
        index = self.attributed_zone(worker)
        worker.zone_index = index
        presence = worker.presences.get(index)
        if presence is None:
            worker.state = OUTSIDE
            worker.entered_at = None
            worker.last_alert_at = None
            worker.entered_wall = None
            worker.alerted_wall = None
            worker.dwell_seconds = 0.0
            return
        worker.state = presence.state
        worker.entered_at = presence.entered_at
        worker.last_alert_at = presence.last_alert_at
        worker.entered_wall = presence.entered_wall
        worker.alerted_wall = presence.alerted_wall
        worker.dwell_seconds = presence.dwell_seconds

    # -- helpers ---------------------------------------------------------

    def get(self, worker_id: int) -> Worker:
        worker = self.workers.get(worker_id)
        if worker is None:
            worker = Worker(worker_id=worker_id)
            self.workers[worker_id] = worker
        return worker

    def _zone_event(
        self, worker: Worker, presence: Presence, event_type: str, wall: float,
        reason: Optional[str] = None,
    ) -> Event:
        """An event about one zone, built from that zone's own clock."""
        return Event(
            worker_id=worker.worker_id,
            machine_id=self.machine_id,
            event_type=event_type,
            entered_at=presence.entered_wall,
            alert_at=presence.alerted_wall,
            dwell_seconds=round(presence.dwell_seconds, 2),
            zone_index=presence.zone_index,
            zone_name=self.zone_name(presence.zone_index),
            reason=reason,
        )

    def _event(self, worker: Worker, event_type: str, wall: float) -> Event:
        """An event about the zone this worker is currently attributed to."""
        presence = worker.presences.get(self.attributed_zone(worker))
        if presence is None:
            return Event(
                worker_id=worker.worker_id,
                machine_id=self.machine_id,
                event_type=event_type,
                zone_index=worker.zone_index,
                zone_name=self.zone_name(worker.zone_index),
            )
        return self._zone_event(worker, presence, event_type, wall)

    def _enter(
        self, worker: Worker, presence: Presence, now: float, wall: float
    ) -> list[Event]:
        presence.state = INSIDE
        presence.entered_at = now
        presence.entered_wall = wall
        presence.alerted_wall = None
        presence.last_alert_at = None
        presence.dwell_seconds = 0.0
        presence.outside_since = None
        presence.outside_wall = None
        events = [self._zone_event(worker, presence, ZONE_ENTRY, wall, REASON_ENTRY)]

        if self.mode == "INSTANT":
            # "As soon as they enter": one ENTRY alert per entry, never a
            # dwell alert and never a repeat. The state still flips to ALERT,
            # so the banner, the boxes and the zone reports treat it as one.
            presence.state = ALERT
            presence.last_alert_at = now
            presence.alerted_wall = wall
            events.append(self._zone_event(worker, presence, ENTRY_ALERT, wall, REASON_ENTRY))
        return events

    def _exit(self, worker: Worker, presence: Presence, now: float, wall: float) -> list[Event]:
        return [self._zone_event(worker, presence, ZONE_EXIT, wall)]

    def reset_timing(self) -> list[int]:
        """Re-check every known worker against the current set of zones.

        Called after the zones were swapped in place, and for the same reason
        when a zone is switched off. A worker the new outlines now exclude must
        not keep an alert it earned under the old ones, and one they now include
        must not inherit the dwell time it accumulated outside them, so their
        timing is dropped here.

        No events are produced: moving an outline or switching a zone off is not
        a worker walking in or out. The next update_all() sees the new status and
        starts a fresh dwell clock for anybody now inside, which is where the
        entry event belongs. Capacity is recalculated the same silent way, so a
        zone that is switched off stops counting without writing anything.

        Returns the ids whose inside/outside status changed.
        """
        self._sync_zones()

        flipped: list[int] = []
        for worker in self.workers.values():
            inside_zones = set(self.zones_containing(worker.point))
            if inside_zones == set(worker.presences):
                continue
            flipped.append(worker.worker_id)
            # dropped, not set to OUTSIDE: the next frame sees a worker who is
            # in a zone with no clock running, and _enter() starts it. That is
            # what puts the entry event on the right frame.
            worker.presences = {}
            for index in sorted(inside_zones):
                worker.presences[index] = Presence(zone_index=index, state=INSIDE)
            self._sync_worker(worker)

        # an alert the new zones no longer justify must not linger for a frame,
        # and each zone is judged on its own people alone
        for index in range(len(self.zones)):
            inside = self._zone_occupants(index)
            self.inside_by_zone[index] = len(inside)
            rule = self._zone_rule(index)
            if len(inside) <= rule.allowed_people:
                self.capacity_by_zone[index] = False
                self.message_by_zone.pop(index, None)
            elif self.capacity_by_zone.get(index):
                # Still over the limit, but the rule it was judged against has
                # just changed - a rename, or a new allowance. The stored message
                # quotes both, so it is rebuilt here; leaving the old one behind
                # is how "Worker #6 is inside Zone 2" survived a rename to
                # "place 1". The flag is deliberately not touched: the next
                # update_all() then still sees no change of state and writes no
                # second CAPACITY_ALERT for a breach that never ended.
                self.message_by_zone[index] = self._build_alert_message(
                    rule, inside, None
                )
        self._sync_aggregates()
        return flipped

    def _zone_rule(self, index: int) -> ZoneRule:
        """The rule for one zone, or an inert one when the index is gone."""
        if 0 <= index < len(self.zones):
            return self.zones[index]
        return ZoneRule(name=f"Zone {index + 1}", points=[], enabled=False)

    def _zone_occupants(self, index: int) -> list[Worker]:
        """Everybody standing in one zone, whoever else they are also in."""
        return [
            worker
            for worker in self.workers.values()
            if index in worker.presences and worker.presences[index].state in (INSIDE, ALERT)
        ]

    # -- main entry point ------------------------------------------------

    def update_all(
        self,
        detections: dict[int, tuple[tuple[int, int], tuple[int, int, int, int], float]],
        now: float,
        wall: Optional[float] = None,
    ) -> tuple[list[Worker], list[Event]]:
        """Apply one frame of detections.

        detections maps worker_id -> (anchor_point, bbox, confidence).
        `now` is the timeline used for dwell timing (video seconds), `wall`
        is the real unix timestamp stored with the events.
        """
        wall = time.time() if wall is None else wall
        events: list[Event] = []
        seen: set[int] = set()
        # which worker stepped into which zone this frame, for the capacity
        # message: zone -> worker id
        entered_by_zone: dict[int, int] = {}

        for worker_id, (point, bbox, confidence) in detections.items():
            seen.add(worker_id)
            worker = self.get(worker_id)
            worker.point = point
            worker.bbox = bbox
            worker.confidence = confidence
            worker.missing_frames = 0

            # A person in two overlapping zones is in BOTH, so each zone gets
            # its own presence and its own clock. Walking out of one is a
            # ZONE_EXIT for that zone alone and leaves the other running.
            inside_zones = self.zones_containing(point, bbox)
            inside_set = set(inside_zones)
            was_in = set(worker.presences)

            for index in inside_zones:
                presence = worker.presences.get(index)
                if presence is None:
                    presence = Presence(zone_index=index)
                    worker.presences[index] = presence
                    events.extend(self._enter(worker, presence, now, wall))
                    entered_by_zone[index] = worker_id
                    continue
                # back inside before the grace ran out (or never outside at all):
                # cancel the pending exit. Nothing is recorded, and the clock is
                # not restarted - stepping over the edge is not a fresh entry.
                presence.outside_since = None
                presence.outside_wall = None
                events.extend(self._advance(worker, presence, now, wall))

            for index in sorted(was_in - inside_set):
                presence = worker.presences[index]
                if presence.outside_since is None:
                    presence.outside_since = now
                    presence.outside_wall = wall
                if now - presence.outside_since < self.exit_grace_seconds:
                    # still inside the grace: keep the clock running and keep
                    # counting them (capacity follows the same grace). Write
                    # nothing, so a step back in is not a fresh entry either.
                    if presence.entered_at is not None:
                        presence.dwell_seconds = max(0.0, now - presence.entered_at)
                    continue
                # a real exit: record the dwell at the moment they first left,
                # not at the end of the grace, so the debounce cannot inflate it
                if presence.entered_at is not None:
                    presence.dwell_seconds = max(
                        0.0, presence.outside_since - presence.entered_at
                    )
                worker.presences.pop(index)
                events.extend(self._exit(worker, presence, now, wall))

            self._sync_worker(worker)

        # Workers that vanished from the frame are kept until they have been
        # missing for lost_timeout_frames in a row. A single dropped detection
        # (motion blur, occlusion, detector jitter) must not raise ZONE_EXIT
        # and clear an active alert.
        for worker_id in list(self.workers):
            if worker_id in seen:
                continue
            worker = self.workers[worker_id]
            worker.missing_frames += 1

            if worker.missing_frames >= self.lost_timeout_frames:
                # forgotten: one exit per zone it was standing in
                for index in sorted(worker.presences):
                    events.extend(self._exit(worker, worker.presences[index], now, wall))
                worker.presences.clear()
                del self.workers[worker_id]
            elif worker.presences:
                # still counted as inside: every clock keeps running
                for presence in worker.presences.values():
                    if presence.entered_at is not None:
                        presence.dwell_seconds = max(0.0, now - presence.entered_at)
                self._sync_worker(worker)

        events.extend(self._update_capacity(entered_by_zone, wall))

        return list(self.workers.values()), events

    def _advance(
        self, worker: Worker, presence: Presence, now: float, wall: float
    ) -> list[Event]:
        """One frame of a worker who stays inside the same zone.

        The clock belongs to this zone, so the dwell limit compared against here
        is that zone's own and the repeat interval is the global one.
        """
        rule = self._zone_rule(presence.zone_index)

        if presence.entered_at is None:
            # reset_timing() parks a worker that a new outline now covers with a
            # cleared clock. It has to start here: while entered_at stays None
            # the dwell is recomputed as 0 on every frame, so the timer rule
            # could never fire for somebody already being tracked when the
            # outline was saved. The entry event belongs here too.
            return self._enter(worker, presence, now, wall)

        presence.dwell_seconds = max(0.0, now - presence.entered_at)

        if self.mode != "TIMER":
            return []

        if presence.state == INSIDE and presence.dwell_seconds >= rule.dwell_seconds:
            presence.state = ALERT
            presence.last_alert_at = now
            presence.alerted_wall = wall
            return [self._zone_event(worker, presence, DWELL_ALERT, wall, REASON_DWELL)]

        if presence.state == ALERT and self.repeat_alert_seconds > 0:
            # TIMER mode re-alerts a worker that stays too long; INSTANT mode is
            # excluded because it must alert exactly once per entry.
            if (
                presence.last_alert_at is None
                or now - presence.last_alert_at >= self.repeat_alert_seconds
            ):
                presence.last_alert_at = now
                presence.alerted_wall = wall
                return [self._zone_event(worker, presence, DWELL_ALERT, wall, REASON_DWELL)]

        return []

    def _capacity_event(self, event_type: str, zone_index: int, dwell: float, wall: float,
                          inside: Optional[int] = None) -> Event:
        return Event(
            worker_id=-1,
            machine_id=self.machine_id,
            event_type=event_type,
            entered_at=wall,
            alert_at=wall if event_type == CAPACITY_ALERT else None,
            dwell_seconds=round(dwell, 2),
            zone_index=zone_index,
            zone_name=self.zone_name(zone_index),
            reason=REASON_CAPACITY,
            inside=inside,
        )

    def _update_capacity(self, entered_by_zone: dict[int, int], wall: float) -> list[Event]:
        """Compare each zone's own people against its own allowance.

        Per zone, and only on a change of that zone's state, so an alert is not
        repeated every frame while people stay inside and one zone coming back
        to normal never clears another zone's alarm.
        """
        events: list[Event] = []

        for index in range(len(self.zones)):
            rule = self.zones[index]
            if not rule.usable:
                # a switched-off or unusable zone holds nobody: drop its count
                # and its alarm without writing anything, the same way a
                # redrawn outline does
                self.inside_by_zone[index] = 0
                if self.capacity_by_zone.get(index):
                    self.capacity_by_zone[index] = False
                    self.message_by_zone.pop(index, None)
                continue

            inside = self._zone_occupants(index)
            self.inside_by_zone[index] = len(inside)
            exceeded = len(inside) > rule.allowed_people
            was = self.capacity_by_zone.get(index, False)

            if exceeded == was:
                if exceeded:
                    self.message_by_zone[index] = self._build_alert_message(
                        rule, inside, entered_by_zone.get(index)
                    )
                continue

            self.capacity_by_zone[index] = exceeded
            if not exceeded:
                self.message_by_zone.pop(index, None)
                events.append(self._capacity_event(CAPACITY_SAFE, index, 0.0, wall,
                                                   inside=len(inside)))
                continue

            self.message_by_zone[index] = self._build_alert_message(
                rule, inside, entered_by_zone.get(index)
            )
            events.append(self._capacity_event(CAPACITY_ALERT, index, 0.0, wall,
                                               inside=len(inside)))

        self._sync_aggregates()
        return events

    def _build_alert_message(
        self, rule: ZoneRule, inside: list[Worker], entered_worker_id: Optional[int]
    ) -> str:
        """The capacity message for one zone, naming that zone."""
        if entered_worker_id is not None:
            return f"Worker #{entered_worker_id} entered {rule.name}"
        if len(inside) == 1:
            return f"Worker #{inside[0].worker_id} is inside {rule.name}"
        return (
            f"{len(inside)} people inside {rule.name}, "
            f"{rule.allowed_people} allowed"
        )

    def _dwell_alert_message(self) -> str:
        """The dwell message for the worst zone, naming that zone.

        The zone is named because with several zones an operator has to be told
        WHICH one the seconds were spent in.
        """
        alerting: list[tuple[float, int, int]] = []
        for worker in self.workers.values():
            for presence in worker.presences.values():
                if presence.state == ALERT:
                    alerting.append((presence.dwell_seconds, presence.zone_index, worker.worker_id))
        if not alerting:
            return ""
        longest, index, worker_id = max(alerting, key=lambda item: (item[0], -item[1]))
        name = self.zone_name(index)
        if len(alerting) == 1:
            return f"Worker #{worker_id} in {name} ({longest:.1f}s)"
        return f"{len(alerting)} workers in danger zones (dwell limit reached)"

    def _sync_aggregates(self) -> None:
        """Fold the per-zone numbers into the single-zone fields.

        `inside_count` counts PEOPLE, not presences: one worker in two zones is
        one person, which is what the old global counter reported and what the
        summary tiles show. `capacity_alert` is the OR of the zones, so any zone
        over its own allowance puts the banner in ALERT.
        """
        self.inside_count = sum(1 for worker in self.workers.values() if worker.inside)
        self.capacity_alert = any(self.capacity_by_zone.values())
        self.alert_message = (
            self._worst_capacity_message()
            if self.capacity_alert
            else self._dwell_alert_message()
        )

    def _worst_capacity_message(self) -> str:
        """The capacity message of the zone that is furthest over its allowance."""
        alerting = [index for index, on in self.capacity_by_zone.items() if on]
        if not alerting:
            return ""
        # most people over the limit wins; a tie goes to the lowest index
        worst = min(
            alerting,
            key=lambda index: (-(self.inside_by_zone.get(index, 0)
                                - self._zone_rule(index).allowed_people), index),
        )
        return self.message_by_zone.get(worst, "")

    def zone_reports(self) -> list[dict]:
        """One row per zone for /state: what this zone is doing right now.

        `dwell_seconds` is the longest anyone has been in THAT zone and
        `dwell_limit` is that zone's own limit, so the pair can be shown without
        the frontend having to guess which global applies. `alerting` is true if
        the zone is over its allowance or anyone in it has passed that zone's
        dwell limit.
        """
        rows = []
        for index, rule in enumerate(self.zones):
            presences = [
                worker.presences[index]
                for worker in self.workers.values()
                if index in worker.presences
            ]
            alerting = self.capacity_by_zone.get(index, False) or any(
                presence.state == ALERT for presence in presences
            )
            longest = max(
                (presence.dwell_seconds for presence in presences), default=0.0
            )
            rows.append(
                {
                    "index": index,
                    "name": rule.name or f"Zone {index + 1}",
                    "inside": len(presences),
                    "allowed_people": rule.allowed_people,
                    "dwell_seconds": round(longest, 2),
                    "dwell_limit": round(float(rule.dwell_seconds), 2),
                    "alerting": bool(alerting),
                    "enabled": rule.enabled,
                }
            )
        return rows

    def status(self) -> dict:
        """Compact status for the UI and the API.

        `capacity_alert` is any zone over its own allowance, `worker_alert` is
        any worker past a dwell limit. The overall `status` is ALERT when either
        one is active, so the UI never shows SAFE while somebody is alerting.
        """
        worker_alert = any(
            presence.state == ALERT
            for worker in self.workers.values()
            for presence in worker.presences.values()
        )

        if self.capacity_alert:
            message = self.alert_message
        elif worker_alert:
            message = self._dwell_alert_message()
        else:
            message = ""

        return {
            "status": ALERT if (self.capacity_alert or worker_alert) else SAFE,
            "capacity_alert": self.capacity_alert,
            "worker_alert": worker_alert,
            "allowed_people": self.allowed_people,
            "inside": self.inside_count,
            "message": message,
        }

    # -- read helpers for the dashboard ----------------------------------

    def summary(self) -> list[dict]:
        """One row per worker for /state, from the zone they are attributed to.

        `state`, `dwell_seconds` and `zone_index` all come from the same
        presence, so the row cannot say "ALERT" and then quote the seconds of a
        different zone. `zones` lists every zone they are actually standing in,
        because that is not the same as the one this row reports.
        """
        rows = []
        for worker in self.workers.values():
            index = self.attributed_zone(worker)
            presence = worker.presences.get(index)
            rows.append(
                {
                    "worker_id": worker.worker_id,
                    "state": presence.state if presence else OUTSIDE,
                    "dwell_seconds": round(presence.dwell_seconds, 2) if presence else 0.0,
                    "zone_index": index if presence else -1,
                    "zone_name": self.zone_name(index) if presence else "",
                    "zones": sorted(worker.presences),
                    # the point the inside/outside test actually used, in frame
                    # pixels. The dashboard needs it to say "this person's feet
                    # are below the frame, so they cannot be counted as inside",
                    # which is otherwise invisible: the state still reads OUTSIDE
                    # and nothing looks wrong.
                    "point": [int(worker.point[0]), int(worker.point[1])],
                }
            )
        rows.sort(key=lambda row: row["worker_id"])
        return rows

    def states(self) -> Iterable[str]:
        return (worker.state for worker in self.workers.values())
