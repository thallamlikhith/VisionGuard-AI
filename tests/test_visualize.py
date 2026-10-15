"""Tests for the visualization fixes: the capacity alert must appear on the
video itself (#1) and the two warning strips must not overlap (#8).

These draw onto a real numpy frame and inspect the pixels, so a banner that
silently stops drawing fails the test.
"""

import cv2
import numpy as np

from visualize import (
    ALERT_BANNER_HEIGHT,
    BANNERS_BOTTOM,
    CAPACITY_BANNER_HEIGHT,
    COLORS,
    WHITE,
    ZONE_COLOR,
    draw_alert_banner,
    draw_capacity_banner,
    draw_person_box,
    draw_ppe_box,
    draw_zone,
)
from zone.danger_zone import (
    ALERT,
    INSIDE,
    OUTSIDE,
    DangerZone,
    Worker,
    ZoneMonitor,
    bottom_center,
    center,
)

SQUARE = [(100, 100), (300, 100), (300, 300), (100, 300)]
BBOX_INSIDE = (150, 150, 250, 250)

WIDTH, HEIGHT = 640, 480
BACKGROUND = (10, 10, 10)


def blank() -> np.ndarray:
    return np.full((HEIGHT, WIDTH, 3), BACKGROUND, dtype=np.uint8)


def changed_mask(frame: np.ndarray) -> np.ndarray:
    """True where a pixel is not the background colour."""
    background = np.array(BACKGROUND, dtype=np.uint8)
    return np.any(frame != background, axis=-1)


def strip_is_painted(frame: np.ndarray, top: int, height: int) -> bool:
    return bool(changed_mask(frame[top : top + height]).any())


def zone_labels(polygons, zone_names, machine_name: str = "Kiln 01") -> list[str]:
    """The labels draw_zone paints, in order.

    OpenCV cannot read text back out of a picture, so the calls are recorded
    instead. That checks the real wording rather than a guess at it, and it does
    not depend on where the label happens to land on the frame.
    """
    written: list[str] = []
    original = cv2.putText

    def spy(image, text, *args, **kwargs):
        written.append(str(text))
        return original(image, text, *args, **kwargs)

    cv2.putText = spy
    try:
        draw_zone(blank(), DangerZone(polygons), machine_name, zone_names=zone_names)
    finally:
        cv2.putText = original
    return written


def alerting_worker() -> Worker:
    return Worker(
        worker_id=7,
        state=ALERT,
        dwell_seconds=6.0,
        bbox=(150, 150, 250, 250),
        point=(200, 250),
    )


def monitor_with(allowed=0, **kwargs) -> ZoneMonitor:
    return ZoneMonitor(
        zone=DangerZone(SQUARE),
        machine_id="excavator_01",
        allowed_people=allowed,
        **kwargs,
    )


def inside_worker(worker_id: int) -> dict:
    return {worker_id: (bottom_center(BBOX_INSIDE), BBOX_INSIDE, 0.9)}


# --- #1 the capacity alert must be drawn on the video ------------------------

def test_capacity_banner_draws_when_over_capacity():
    monitor = monitor_with()
    monitor.update_all(inside_worker(3), now=0.0, wall=100.0)
    assert monitor.capacity_alert is True

    frame = blank()
    draw_capacity_banner(frame, monitor.status())

    assert strip_is_painted(frame, 0, CAPACITY_BANNER_HEIGHT), (
        "capacity alert was NOT drawn on the video"
    )


def test_capacity_banner_stays_hidden_when_within_the_limit():
    monitor = monitor_with(allowed=2)
    monitor.update_all(
        {**inside_worker(1), **inside_worker(2)},
        now=0.0,
        wall=100.0,
    )
    assert monitor.capacity_alert is False

    frame = blank()
    draw_capacity_banner(frame, monitor.status())

    assert not changed_mask(frame).any(), "capacity banner drawn while within the limit"


def test_pipeline_style_call_draws_the_over_capacity_alert():
    """Exactly the sequence the pipeline runs every frame."""
    monitor = monitor_with(mode="TIMER", dwell_seconds=30.0)
    monitor.update_all(inside_worker(7), now=0.0, wall=100.0)

    frame = blank()
    status = monitor.status()
    draw_capacity_banner(frame, status, top=0)

    assert status["capacity_alert"] is True
    assert status["status"] == "ALERT"
    assert strip_is_painted(frame, 0, CAPACITY_BANNER_HEIGHT), (
        "alert missing from the MJPEG frame"
    )


def test_capacity_banner_text_is_bright_and_readable():
    monitor = monitor_with()
    monitor.update_all(inside_worker(3), now=0.0, wall=100.0)

    frame = blank()
    draw_capacity_banner(frame, monitor.status(), top=0)

    strip = frame[:CAPACITY_BANNER_HEIGHT]
    grey = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
    assert grey.max() > 200, "no bright text pixels in the capacity banner"


# --- #8 the two banners must not overlap ------------------------------------

def test_capacity_and_dwell_banners_occupy_separate_strips():
    """The capacity alert owns the top strip, the dwell alert the one below."""
    monitor = monitor_with(mode="INSTANT")
    monitor.update_all(inside_worker(7), now=0.0, wall=100.0)

    frame = blank()
    draw_capacity_banner(frame, monitor.status(), top=0)
    draw_alert_banner(frame, [alerting_worker()], top=BANNERS_BOTTOM - ALERT_BANNER_HEIGHT)

    assert strip_is_painted(frame, 0, CAPACITY_BANNER_HEIGHT), "capacity strip empty"
    assert strip_is_painted(
        frame, CAPACITY_BANNER_HEIGHT, ALERT_BANNER_HEIGHT
    ), "dwell strip empty"
    assert not changed_mask(frame[BANNERS_BOTTOM:]).any(), (
        "something was drawn below the two reserved strips"
    )


def test_capacity_banner_does_not_paint_below_its_own_height():
    frame = blank()
    draw_capacity_banner(
        frame,
        {"capacity_alert": True, "inside": 1, "allowed_people": 0},
        top=0,
    )

    assert strip_is_painted(frame, 0, CAPACITY_BANNER_HEIGHT)
    assert not changed_mask(frame[CAPACITY_BANNER_HEIGHT:]).any()


def test_dwell_banner_is_positioned_below_the_capacity_strip():
    """The lower strip alone must fill when there is no capacity alert."""
    monitor = monitor_with(allowed=5)  # never over capacity
    frame = blank()

    draw_capacity_banner(frame, monitor.status(), top=0)
    draw_alert_banner(frame, [alerting_worker()], top=BANNERS_BOTTOM - ALERT_BANNER_HEIGHT)

    assert not strip_is_painted(frame, 0, CAPACITY_BANNER_HEIGHT), (
        "capacity strip painted although capacity_alert is False"
    )
    assert strip_is_painted(frame, CAPACITY_BANNER_HEIGHT, ALERT_BANNER_HEIGHT), (
        "dwell strip empty although a worker is ALERT"
    )


def test_lower_strip_stays_empty_when_no_worker_is_alerting():
    monitor = monitor_with()
    monitor.update_all(inside_worker(3), now=0.0, wall=100.0)

    frame = blank()
    draw_capacity_banner(frame, monitor.status(), top=0)
    draw_alert_banner(
        frame,
        [Worker(worker_id=9, state=INSIDE, bbox=(150, 150, 250, 250), point=(200, 250))],
        top=BANNERS_BOTTOM - ALERT_BANNER_HEIGHT,
    )

    assert strip_is_painted(frame, 0, CAPACITY_BANNER_HEIGHT)
    assert not strip_is_painted(frame, CAPACITY_BANNER_HEIGHT, ALERT_BANNER_HEIGHT), (
        "dwell strip painted although no worker is in the ALERT state"
    )


def test_both_banners_can_be_shown_at_once_on_a_small_frame():
    """Sanity check that the layout still fits a small video."""
    frame = np.full((120, 160, 3), BACKGROUND, dtype=np.uint8)
    draw_capacity_banner(
        frame,
        {"capacity_alert": True, "inside": 3, "allowed_people": 1},
        top=0,
    )
    draw_alert_banner(frame, [alerting_worker()], top=BANNERS_BOTTOM - ALERT_BANNER_HEIGHT)

    assert BANNERS_BOTTOM < frame.shape[0]
    assert strip_is_painted(frame, 0, CAPACITY_BANNER_HEIGHT)
    assert strip_is_painted(frame, CAPACITY_BANNER_HEIGHT, ALERT_BANNER_HEIGHT)


def test_banner_strip_heights_are_consistent():
    assert CAPACITY_BANNER_HEIGHT == 40
    assert ALERT_BANNER_HEIGHT == 36
    assert BANNERS_BOTTOM == CAPACITY_BANNER_HEIGHT + ALERT_BANNER_HEIGHT
    assert BANNERS_BOTTOM < HEIGHT


# --- the web frame stays clean ---------------------------------------------


def test_person_box_draws_only_around_the_worker():
    frame = blank()
    draw_person_box(frame, alerting_worker())

    painted = changed_mask(frame)
    # the box edges plus the label above it, all near the worker
    assert painted[150:250, 150:250].any()
    # the capacity / dwell strips at the top of the frame must stay empty
    assert not painted[0:BANNERS_BOTTOM].any()


def test_person_box_label_sits_above_the_head():
    frame = blank()
    draw_person_box(frame, alerting_worker())

    assert changed_mask(frame[:150, 140:280]).any(), "expected a label above the box"


def test_person_box_label_moves_below_when_there_is_no_room():
    frame = blank()
    worker = Worker(
        worker_id=3,
        state=INSIDE,
        dwell_seconds=1.5,
        bbox=(150, 0, 250, 100),
        point=(200, 100),
    )
    draw_person_box(frame, worker)

    assert changed_mask(frame[100:160, 140:280]).any(), "expected the label under the box"


def _label_overlaps(first, second) -> bool:
    """True when two (left, top, right, bottom) rects share interior."""
    return (max(first[0], second[0]) < min(first[2], second[2])
            and max(first[1], second[1]) < min(first[3], second[3]))


def test_close_ppe_labels_cascade_instead_of_overlapping():
    """Three violations stacked on each other: every label rect is disjoint."""
    frame = blank()
    placed: list = []
    draw_ppe_box(frame, (100, 200, 170, 270), "NO HARDHAT 0.71", placed=placed)
    draw_ppe_box(frame, (105, 210, 175, 280), "NO SAFETY VEST 0.66", placed=placed)
    draw_ppe_box(frame, (110, 220, 180, 290), "NO MASK 0.60", placed=placed)

    assert len(placed) == 3
    assert len({tuple(rect) for rect in placed}) == 3  # nothing shared a slot
    for first in placed:
        for second in placed:
            if first is not second:
                assert not _label_overlaps(first, second)


def test_ppe_labels_avoid_person_labels():
    """A violation box on a worker's chest must not paint over the #id label."""
    frame = blank()
    worker = Worker(
        worker_id=3,
        dwell_seconds=4.2,
        bbox=(100, 200, 170, 320),
        point=(135, 320),
    )
    placed: list = []
    draw_person_box(frame, worker, placed=placed)
    draw_ppe_box(frame, (105, 210, 175, 280), "NO HARDHAT 0.71", placed=placed)

    assert len(placed) == 2
    assert not _label_overlaps(placed[0], placed[1])


def test_labels_keep_historical_slots_without_a_registry():
    """placed=None draws exactly where it always did (no cascade)."""
    frame = blank()
    draw_ppe_box(frame, (100, 200, 170, 270), "NO HARDHAT 0.71")
    # label block above the box, same rows as before the registry existed
    assert changed_mask(frame[170:200, 90:260]).any()


def test_person_box_without_a_bbox_draws_nothing():
    frame = blank()
    draw_person_box(frame, Worker(worker_id=1, state=INSIDE, bbox=None))

    assert not changed_mask(frame).any()


# --- the anchor dot on the web frame ----------------------------------------


def test_person_box_marks_the_anchor_point():
    """The dot is the pixel the zone test uses, so the picture has to show it.

    The browser is the only place a real alarm comes from and it gets this
    function; draw_worker() has always painted the dot, so without this the
    command line window explained the rule and the dashboard did not.
    """
    frame = blank()
    worker = alerting_worker()
    draw_person_box(frame, worker)

    painted = changed_mask(frame)
    x, y = worker.point
    # a white filled centre, so the point is legible over any state colour
    assert tuple(int(channel) for channel in frame[y, x]) == WHITE
    # ringed in the worker's own colour at radius 5, like draw_worker does
    assert tuple(int(channel) for channel in frame[y - 5, x]) == COLORS[ALERT]
    # and nothing painted on the far side, i.e. it is a dot and not a blob
    assert not painted[y - 6, x]


def test_person_box_anchor_dot_follows_the_anchor():
    """Feet and Body center are different pixels, and the dot has to be on the
    one that was actually tested - not a fixed spot near the bottom."""
    frame = blank()
    box = (150, 150, 250, 250)
    feet = Worker(worker_id=1, state=OUTSIDE, bbox=box, point=bottom_center(box))
    middle = Worker(worker_id=1, state=OUTSIDE, bbox=box, point=center(box))
    draw_person_box(frame, feet)
    feet_frame = frame.copy()
    frame = blank()
    draw_person_box(frame, middle)

    assert changed_mask(feet_frame)[250, 200], "feet anchor is the bottom edge"
    assert changed_mask(frame)[200, 200], "body centre anchor is the middle"


SECOND_SQUARE = [(400, 100), (600, 100), (600, 300), (400, 300)]


def test_a_single_zone_keeps_the_machine_name_it_always_had():
    """One zone has no ambiguity, so the label must not become 'Zone 1'."""
    labels = zone_labels([SQUARE], ["Kiln Platform"])

    assert labels == ["DANGER ZONE - Kiln 01"]


def test_each_zone_is_labelled_with_the_name_its_rules_use():
    """With several zones the label has to say WHICH zone an alert is about."""
    labels = zone_labels([SQUARE, SECOND_SQUARE], ["Kiln Platform", "Loading Bay"])

    assert labels == ["DANGER ZONE - Kiln Platform", "DANGER ZONE - Loading Bay"]


def test_a_zone_label_falls_back_to_the_machine_name_when_no_names_are_given():
    labels = zone_labels([SQUARE], None)

    assert labels == ["DANGER ZONE - Kiln 01"]


def test_a_partly_named_set_of_zones_still_labels_every_zone():
    """A missing name must not leave one polygon without a label."""
    labels = zone_labels([SQUARE, SECOND_SQUARE], ["Kiln Platform"])

    assert labels == ["DANGER ZONE - Kiln Platform", "DANGER ZONE - Kiln 01"]


# --- the three occupancy states: red is reserved for an active alert --------

EDGE_PIXEL = (100, 200)  # on the top edge of SQUARE


def test_zone_outline_goes_solid_red_while_alerting():
    frame = blank()
    draw_zone(frame, DangerZone([SQUARE]), "Kiln 01", zone_tones=["alert"])

    assert tuple(int(channel) for channel in frame[EDGE_PIXEL]) == COLORS[ALERT]


def test_occupied_zone_keeps_the_orange_outline():
    frame = blank()
    draw_zone(frame, DangerZone([SQUARE]), "Kiln 01", zone_tones=["occupy"])

    assert tuple(int(channel) for channel in frame[EDGE_PIXEL]) == ZONE_COLOR


def test_empty_zone_keeps_the_default_outline():
    frame = blank()
    draw_zone(frame, DangerZone([SQUARE]), "Kiln 01")

    assert tuple(int(channel) for channel in frame[EDGE_PIXEL]) == ZONE_COLOR


def test_person_box_goes_red_in_a_capacity_breach_without_a_dwell_alert():
    """A crowd over the allowance alerts the zone while its people are still
    INSIDE, so the box is forced red instead of staying amber."""
    frame = blank()
    worker = Worker(
        worker_id=3, state=INSIDE, dwell_seconds=1.0, bbox=BBOX_INSIDE, point=(200, 250)
    )
    draw_person_box(frame, worker, alert=True)

    x, y = worker.point
    assert tuple(int(channel) for channel in frame[y - 5, x]) == COLORS[ALERT]


def test_person_box_stays_amber_while_within_the_limit():
    frame = blank()
    worker = Worker(
        worker_id=3, state=INSIDE, dwell_seconds=1.0, bbox=BBOX_INSIDE, point=(200, 250)
    )
    draw_person_box(frame, worker)

    x, y = worker.point
    assert tuple(int(channel) for channel in frame[y - 5, x]) == COLORS[INSIDE]