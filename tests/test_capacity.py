"""Tests for the allowed-people capacity rule (the web UI alert logic)."""

import pytest

from zone.danger_zone import (
    CAPACITY_ALERT,
    CAPACITY_SAFE,
    INSIDE,
    SAFE,
    DangerZone,
    ZoneMonitor,
    bottom_center,
)

SQUARE = [(100, 100), (300, 100), (300, 300), (100, 300)]

BBOX_INSIDE = (150, 150, 250, 250)  # bottom-centre (200, 250) -> inside
BBOX_OUTSIDE = (0, 0, 40, 40)  # bottom-centre (20, 40)  -> outside


def make_monitor(allowed=0, mode="TIMER", dwell=5.0):
    return ZoneMonitor(
        zone=DangerZone(SQUARE),
        machine_id="excavator_01",
        mode=mode,
        dwell_seconds=dwell,
        repeat_alert_seconds=0.0,
        allowed_people=allowed,
    )


def detect(*worker_ids, bbox=BBOX_INSIDE):
    return {wid: (bottom_center(bbox), bbox, 0.9) for wid in worker_ids}


def types(events):
    return [e.event_type for e in events]


# --- allowed = 0 -----------------------------------------------------------

def test_allowed_zero_is_the_default():
    monitor = ZoneMonitor(zone=DangerZone(SQUARE), machine_id="m")
    assert monitor.allowed_people == 0


def test_allowed_zero_one_person_inside_alerts_immediately():
    monitor = make_monitor(allowed=0)
    workers, events = monitor.update_all(detect(3), now=0.0, wall=100.0)

    assert CAPACITY_ALERT in types(events)
    assert monitor.capacity_alert is True
    assert monitor.inside_count == 1
    assert monitor.status()["status"] == "ALERT"
    assert "Worker #3" in monitor.alert_message
    # the row carries the crowd size, not a worker: the UI reads this, never -1
    alert = next(e for e in events if e.event_type == CAPACITY_ALERT)
    assert alert.inside == 1
    assert alert.to_dict()["inside"] == 1


def test_allowed_zero_alerts_without_waiting_for_the_dwell_timer():
    """dwell_seconds is 30s, but the alert must come on the first frame."""
    monitor = make_monitor(allowed=0, mode="TIMER", dwell=30.0)
    monitor.update_all(detect(3), now=0.0, wall=100.0)

    assert monitor.capacity_alert is True
    assert monitor.get(3).state == INSIDE  # per worker state is still just INSIDE
    assert monitor.get(3).dwell_seconds == 0.0


def test_no_duplicates_while_the_person_stays_inside():
    monitor = make_monitor(allowed=0)
    _, first = monitor.update_all(detect(3), now=0.0, wall=100.0)
    assert types(first).count(CAPACITY_ALERT) == 1

    for step in range(1, 40):  # 39 more frames with the worker inside
        _, events = monitor.update_all(detect(3), now=float(step), wall=100.0 + step)
        assert CAPACITY_ALERT not in types(events), f"duplicate alert on frame {step}"

    assert monitor.capacity_alert is True


def test_alert_survives_a_single_missed_detection():
    """A dropped frame must not clear an active capacity alert."""
    monitor = make_monitor(allowed=0)
    monitor.update_all(detect(3), now=0.0, wall=100.0)

    _, events = monitor.update_all({}, now=1.0, wall=101.0)  # 1 miss

    assert CAPACITY_SAFE not in types(events)
    assert monitor.capacity_alert is True
    assert monitor.status()["status"] == "ALERT"
    assert monitor.inside_count == 1


def test_alert_clears_when_the_zone_becomes_empty():
    monitor = make_monitor(allowed=0)
    monitor.update_all(detect(3), now=0.0, wall=100.0)

    # the worker has to be gone for the whole timeout before the zone is empty
    workers, events = monitor.update_all({}, now=1.0, wall=101.0)
    assert monitor.inside_count == 1

    all_events = []
    for frame in range(2, monitor.lost_timeout_frames + 2):
        _, events = monitor.update_all({}, now=float(frame), wall=100.0 + frame)
        all_events += events

    assert CAPACITY_SAFE in types(all_events)
    assert monitor.capacity_alert is False
    assert monitor.status()["status"] == SAFE
    assert monitor.inside_count == 0


def test_alert_returns_after_a_second_visit():
    monitor = make_monitor(allowed=0)
    monitor.update_all(detect(3), now=0.0, wall=100.0)

    # long enough for the first visit to be forgotten
    for frame in range(1, monitor.lost_timeout_frames + 2):
        monitor.update_all({}, now=float(frame), wall=100.0 + frame)
    assert monitor.capacity_alert is False

    _, events = monitor.update_all(detect(3), now=20.0, wall=120.0)

    assert types(events).count(CAPACITY_ALERT) == 1


# --- allowed = 2 -----------------------------------------------------------

def test_allowed_two_two_inside_is_safe():
    monitor = make_monitor(allowed=2)
    _, events = monitor.update_all(detect(1, 2), now=0.0, wall=100.0)

    assert CAPACITY_ALERT not in types(events)
    assert monitor.capacity_alert is False
    assert monitor.inside_count == 2
    assert monitor.status() == {
        "status": SAFE,
        "capacity_alert": False,
        "worker_alert": False,
        "allowed_people": 2,
        "inside": 2,
        "message": "",
    }


def test_allowed_two_three_inside_alerts():
    monitor = make_monitor(allowed=2)
    monitor.update_all(detect(1, 2), now=0.0, wall=100.0)
    workers, events = monitor.update_all(detect(1, 2, 3), now=1.0, wall=101.0)

    assert CAPACITY_ALERT in types(events)
    assert monitor.capacity_alert is True
    assert monitor.inside_count == 3
    assert "Worker #3" in monitor.alert_message


def test_allowed_two_drops_back_to_safe_when_one_leaves():
    monitor = make_monitor(allowed=2)
    monitor.update_all(detect(1, 2, 3), now=0.0, wall=100.0)
    assert monitor.capacity_alert is True

    # worker 3 is still seen, but has walked outside the polygon. the exit
    # only lands once the grace elapses, so hold them outside past it
    monitor.update_all({**detect(1, 2), 3: (bottom_center(BBOX_OUTSIDE), BBOX_OUTSIDE, 0.9)}, now=1.0, wall=101.0)
    monitor.update_all({**detect(1, 2), 3: (bottom_center(BBOX_OUTSIDE), BBOX_OUTSIDE, 0.9)}, now=1.7, wall=101.7)
    assert monitor.capacity_alert is False
    assert monitor.inside_count == 2


def test_allowed_two_stays_alerted_while_the_third_worker_is_briefly_lost():
    """Dropping a track for a frame must not drop the count below the limit."""
    monitor = make_monitor(allowed=2)
    monitor.update_all(detect(1, 2, 3), now=0.0, wall=100.0)
    assert monitor.capacity_alert is True

    monitor.update_all(detect(1, 2), now=1.0, wall=101.0)  # 3 is lost for one frame

    assert monitor.capacity_alert is True
    assert monitor.inside_count == 3


def test_only_people_inside_the_polygon_are_counted():
    monitor = make_monitor(allowed=0)
    monitor.update_all({**detect(1), 2: (bottom_center(BBOX_OUTSIDE), BBOX_OUTSIDE, 0.9)}, now=0.0, wall=100.0)

    assert monitor.inside_count == 1
    assert monitor.capacity_alert is True


# --- allowed = 1: exactly at the limit is not over it ------------------------

def test_allowed_one_single_person_never_raises_capacity():
    """1 person with 1 allowed is exactly at the limit, not over it: no
    capacity alert in any alert mode."""
    for mode in ("TIMER", "INSTANT", "LIMIT"):
        monitor = make_monitor(allowed=1, mode=mode)
        _, events = monitor.update_all(detect(3), now=0.0, wall=100.0)

        assert CAPACITY_ALERT not in types(events)
        assert monitor.capacity_alert is False


def test_allowed_one_second_person_raises_capacity_in_every_mode():
    for mode in ("TIMER", "INSTANT", "LIMIT"):
        monitor = make_monitor(allowed=1, mode=mode)
        monitor.update_all(detect(3), now=0.0, wall=100.0)
        _, events = monitor.update_all(detect(3, 4), now=1.0, wall=101.0)

        assert CAPACITY_ALERT in types(events)
        assert monitor.capacity_alert is True


# --- configuration ---------------------------------------------------------

@pytest.mark.parametrize("allowed,inside,expected", [(0, 0, SAFE), (0, 1, "ALERT"), (2, 2, SAFE), (2, 3, "ALERT")])
def test_the_four_required_cases(allowed, inside, expected):
    """allowed=0/inside=0 SAFE, 0/1 ALERT, 2/2 SAFE, 2/3 ALERT."""
    monitor = make_monitor(allowed=allowed)
    ids = list(range(1, inside + 1))
    monitor.update_all(detect(*ids) if ids else {}, now=0.0, wall=100.0)
    assert monitor.status()["status"] == expected


def test_negative_allowed_is_clamped_to_zero():
    monitor = make_monitor(allowed=-5)
    assert monitor.allowed_people == 0


def test_capacity_event_is_logged_with_the_machine_id():
    monitor = make_monitor(allowed=0)
    _, events = monitor.update_all(detect(3), now=0.0, wall=100.0)
    alert = [e for e in events if e.event_type == CAPACITY_ALERT][0]

    assert alert.machine_id == "excavator_01"
    assert alert.alert_at == 100.0


# --- status() must describe both alert kinds separately ----------------------

def test_status_exposes_both_alert_flags():
    monitor = make_monitor(allowed=0, mode="TIMER", dwell=30.0)
    monitor.update_all(detect(3), now=0.0, wall=100.0)
    status = monitor.status()

    assert status["capacity_alert"] is True
    assert status["worker_alert"] is False  # dwell has not elapsed yet
    assert status["status"] == "ALERT"  # capacity alone is enough to alert
    assert status["allowed_people"] == 0
    assert status["inside"] == 1


def test_dwell_alert_raises_worker_alert_before_any_exit():
    """#2: a worker past the dwell time must not still be reported as SAFE."""
    monitor = make_monitor(allowed=5, mode="TIMER", dwell=5.0)
    monitor.update_all(detect(3), now=0.0, wall=100.0)

    assert monitor.status()["status"] == SAFE
    assert monitor.status()["worker_alert"] is False

    monitor.update_all(detect(3), now=6.0, wall=106.0)  # dwell elapsed

    status = monitor.status()
    assert status["worker_alert"] is True
    assert status["status"] == "ALERT"
    assert "Worker #3" in status["message"]


def test_capacity_alert_and_dwell_alert_can_both_be_set():
    monitor = make_monitor(allowed=0, mode="TIMER", dwell=5.0)
    monitor.update_all(detect(3), now=0.0, wall=100.0)
    monitor.update_all(detect(3), now=6.0, wall=106.0)

    status = monitor.status()
    assert status["capacity_alert"] is True
    assert status["worker_alert"] is True
    assert status["status"] == "ALERT"


def test_status_message_prefers_the_dwell_detail():
    monitor = make_monitor(allowed=0, mode="TIMER", dwell=5.0)
    monitor.update_all(detect(3), now=0.0, wall=100.0)
    monitor.update_all(detect(3), now=6.0, wall=106.0)

    assert "Worker #3" in monitor.status()["message"]


def test_status_flags_return_to_false_after_the_zone_empties():
    monitor = make_monitor(allowed=0, mode="TIMER", dwell=5.0)
    monitor.update_all(detect(3), now=0.0, wall=100.0)
    monitor.update_all(detect(3), now=6.0, wall=106.0)

    # walk out for real this time: hold them outside past the grace so the
    # exit (and the capacity-safe it clears) actually lands
    monitor.update_all({3: (bottom_center(BBOX_OUTSIDE), BBOX_OUTSIDE, 0.9)}, now=7.0, wall=107.0)
    monitor.update_all({3: (bottom_center(BBOX_OUTSIDE), BBOX_OUTSIDE, 0.9)}, now=7.7, wall=107.7)

    status = monitor.status()
    assert status["capacity_alert"] is False
    assert status["worker_alert"] is False
    assert status["status"] == SAFE
    assert status["message"] == ""


def test_status_is_safe_with_nobody_inside():
    monitor = make_monitor(allowed=0)
    status = monitor.status()

    assert status == {
        "status": SAFE,
        "capacity_alert": False,
        "worker_alert": False,
        "allowed_people": 0,
        "inside": 0,
        "message": "",
    }


# --- the single-zone behaviour is unchanged ----------------------------------
#
# Zones now carry their own allowance, so the obvious worry is that the common
# one-zone case drifted. These are the proof: with one polygon the zone uses the
# global numbers and every field the old UI reads still reads the same.

def test_one_zone_uses_the_global_allowed_people():
    monitor = make_monitor(allowed=2)

    monitor.update_all(detect(1, 2), now=0.0, wall=100.0)
    monitor.update_all(detect(1, 2, 3), now=0.1, wall=100.1)

    assert monitor.status()["capacity_alert"] is True
    assert monitor.status()["allowed_people"] == 2


def test_one_zone_reports_the_same_row_the_way_it_always_did():
    """The row the dashboard draws still has one state, one dwell, no zones."""
    monitor = make_monitor(allowed=5, dwell=5.0)
    monitor.update_all(detect(7), now=0.0, wall=100.0)

    row = monitor.summary()[0]

    assert row["worker_id"] == 7
    assert row["state"] == INSIDE
    assert row["dwell_seconds"] == 0.0
    assert row["zone_index"] == 0
    assert row["zones"] == [0]


def test_one_zone_still_reports_a_single_inside_count():
    monitor = make_monitor(allowed=5)
    monitor.update_all(detect(1, 2, 3), now=0.0, wall=100.0)

    assert monitor.inside_count == 3
    assert monitor.status()["inside"] == 3


def test_a_worker_walking_out_still_reads_outside_on_one_zone():
    monitor = make_monitor(allowed=5)
    monitor.update_all(detect(1), now=0.0, wall=100.0)
    # one frame outside is inside the grace; stay out past it for a real exit
    monitor.update_all({1: (bottom_center(BBOX_OUTSIDE), BBOX_OUTSIDE, 0.9)}, now=1.0, wall=101.0)
    monitor.update_all({1: (bottom_center(BBOX_OUTSIDE), BBOX_OUTSIDE, 0.9)}, now=1.7, wall=101.7)

    row = monitor.summary()[0]

    assert row["state"] == "OUTSIDE"
    assert row["dwell_seconds"] == 0.0
    assert row["zones"] == []
    assert monitor.inside_count == 0


def test_one_zone_keeps_the_dwell_alert_and_the_zone_named_zone_1():
    monitor = make_monitor(allowed=5, dwell=2.0)
    monitor.update_all(detect(4), now=0.0, wall=100.0)
    _, events = monitor.update_all(detect(4), now=3.0, wall=103.0)

    assert [e.event_type for e in events if e.event_type == "DWELL_ALERT"]
    assert "Zone 1" in monitor.status()["message"]
