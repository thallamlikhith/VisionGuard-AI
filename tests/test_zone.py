"""Core zone logic tests - no video, no YOLO, no camera needed."""

import pytest

from zone.danger_zone import (
    ALERT,
    CAPACITY_ALERT,
    CAPACITY_SAFE,
    DWELL_ALERT,
    ENTRY_ALERT,
    INSIDE,
    OUTSIDE,
    REASON_CAPACITY,
    REASON_DWELL,
    REASON_ENTRY,
    SAFE,
    ZONE_ENTRY,
    ZONE_EXIT,
    ZoneRule,
    DangerZone,
    ZoneError,
    ZoneMonitor,
    bottom_center,
    center,
    point_in_polygon,
)

# a simple square: x 100..300, y 100..300
SQUARE = [(100, 100), (300, 100), (300, 300), (100, 300)]

INSIDE_POINT = (200, 200)
OUTSIDE_POINT = (50, 50)

BBOX_INSIDE = (150, 150, 250, 250)      # bottom-center (200, 250) -> inside
BBOX_OUTSIDE = (0, 0, 40, 40)           # bottom-center (20, 40)  -> outside


def make_monitor(
    mode="TIMER", dwell=5.0, repeat=0.0, allowed=0, lost_timeout_frames=15, exit_grace=0.6
):
    return ZoneMonitor(
        zone=DangerZone(SQUARE),
        machine_id="excavator_01",
        mode=mode,
        dwell_seconds=dwell,
        repeat_alert_seconds=repeat,
        allowed_people=allowed,
        lost_timeout_frames=lost_timeout_frames,
        exit_grace_seconds=exit_grace,
    )


def detect(worker_id, bbox, confidence=0.9):
    return {worker_id: (bottom_center(bbox), bbox, confidence)}


def worker_events(events):
    """Per-worker events only, without the capacity rule events."""
    return [e for e in events if e.event_type not in (CAPACITY_ALERT, CAPACITY_SAFE)]


# -- polygon ---------------------------------------------------------------

def test_point_in_polygon_inside():
    assert point_in_polygon(INSIDE_POINT, SQUARE) is True


def test_point_in_polygon_outside():
    assert point_in_polygon(OUTSIDE_POINT, SQUARE) is False
    assert point_in_polygon((400, 200), SQUARE) is False   # right of the square
    assert point_in_polygon((200, 90), SQUARE) is False    # just above the top edge


def test_zone_needs_three_points():
    from zone.danger_zone import ZoneError

    with pytest.raises(ZoneError):
        DangerZone([(0, 0), (1, 1)])


def test_empty_zone_is_inactive_so_the_video_can_be_previewed():
    zone = DangerZone([])

    assert zone.active is False
    assert zone.contains((100, 100)) is False

    monitor = ZoneMonitor(zone=zone, machine_id="m", allowed_people=0)
    workers, events = monitor.update_all(detect(1, BBOX_INSIDE), now=0.0, wall=100.0)

    assert workers[0].state == OUTSIDE
    assert events == []
    assert monitor.capacity_alert is False


def test_anchor_points():
    assert bottom_center(BBOX_INSIDE) == (200, 250)
    assert center(BBOX_INSIDE) == (200, 200)


# -- no zone at all ---------------------------------------------------------

@pytest.mark.parametrize("mode", ["TIMER", "INSTANT", "LIMIT"])
def test_monitoring_with_no_zone_never_alerts(mode):
    """A machine with no zone drawn boxes people and stays quiet forever."""
    monitor = ZoneMonitor(
        zone=DangerZone([]),
        machine_id="excavator_01",
        mode=mode,
        dwell_seconds=0.0,
        allowed_people=0,
    )

    events = []
    # a person standing in the middle of the frame for a long time
    for frame in range(600):
        _, frame_events = monitor.update_all(detect(1, BBOX_INSIDE), now=frame / 10.0, wall=100.0)
        events.extend(frame_events)

    assert worker_events(events) == []
    assert monitor.capacity_alert is False
    assert [worker.state for worker in monitor.workers.values()] == [OUTSIDE]
    assert all(worker.dwell_seconds == 0.0 for worker in monitor.workers.values())


# -- states ----------------------------------------------------------------

def test_worker_outside_stays_outside():
    monitor = make_monitor()
    workers, events = monitor.update_all(detect(1, BBOX_OUTSIDE), now=0.0, wall=100.0)

    assert workers[0].state == OUTSIDE
    assert events == []


def test_worker_enters_zone():
    monitor = make_monitor()
    _, events = monitor.update_all(detect(1, BBOX_INSIDE), now=0.0, wall=100.0)

    assert [e.event_type for e in worker_events(events)] == [ZONE_ENTRY]
    assert monitor.get(1).state == INSIDE


def test_timer_dwell_and_alert():
    monitor = make_monitor(mode="TIMER", dwell=5.0)

    monitor.update_all(detect(1, BBOX_INSIDE), now=0.0, wall=100.0)
    workers, events = monitor.update_all(detect(1, BBOX_INSIDE), now=2.0, wall=102.0)
    assert workers[0].state == INSIDE
    assert workers[0].dwell_seconds == pytest.approx(2.0)
    assert events == []

    workers, events = monitor.update_all(detect(1, BBOX_INSIDE), now=5.0, wall=105.0)
    assert workers[0].state == ALERT
    assert workers[0].dwell_seconds == pytest.approx(5.0)
    assert [e.event_type for e in worker_events(events)] == [DWELL_ALERT]
    assert events[0].dwell_seconds == pytest.approx(5.0)
    assert events[0].alert_at == 105.0


def test_timer_does_not_alert_early():
    monitor = make_monitor(mode="TIMER", dwell=5.0)
    monitor.update_all(detect(1, BBOX_INSIDE), now=0.0, wall=100.0)
    workers, events = monitor.update_all(detect(1, BBOX_INSIDE), now=4.99, wall=104.99)

    assert workers[0].state == INSIDE
    assert events == []


def test_instant_alerts_on_entry():
    monitor = make_monitor(mode="INSTANT")
    workers, events = monitor.update_all(detect(1, BBOX_INSIDE), now=0.0, wall=100.0)

    assert workers[0].state == ALERT
    assert [e.event_type for e in worker_events(events)] == [ZONE_ENTRY, ENTRY_ALERT]


def test_instant_alert_is_not_repeated_every_frame():
    monitor = make_monitor(mode="INSTANT", repeat=0.0)
    monitor.update_all(detect(1, BBOX_INSIDE), now=0.0, wall=100.0)
    _, events = monitor.update_all(detect(1, BBOX_INSIDE), now=1.0, wall=101.0)
    assert events == []


def test_repeat_of_zero_alerts_exactly_once():
    """`repeat_alert_seconds = 0` is the "do not repeat" the UI offers.

    It has to survive the whole trip: the API accepts it, the zone is built with
    it, and a person who stands inside for a long time still gets one alert.
    """
    monitor = make_monitor(mode="TIMER", dwell=1.0, repeat=0.0)
    monitor.update_all(detect(1, BBOX_INSIDE), now=0.0, wall=100.0)
    _, events = monitor.update_all(detect(1, BBOX_INSIDE), now=1.0, wall=101.0)
    assert [e.event_type for e in worker_events(events)] == [DWELL_ALERT]

    # two minutes later, and then some: still nothing new
    for now in (30.0, 61.0, 120.0, 3600.0):
        _, events = monitor.update_all(detect(1, BBOX_INSIDE), now=now, wall=100.0 + now)
        assert events == []


def test_dwell_of_zero_alerts_with_no_time_waited():
    """`dwell_seconds = 0` means "alarm straight away", not "wait forever".

    The entry frame raises ZONE_ENTRY and the dwell rule is checked on the
    following frame, so with a wait of 0 the alert lands on that next frame
    with `now` unchanged: no time passes between entry and alarm.
    """
    monitor = make_monitor(mode="TIMER", dwell=0.0, repeat=0.0)

    workers, events = monitor.update_all(detect(1, BBOX_INSIDE), now=0.0, wall=100.0)
    assert [e.event_type for e in worker_events(events)] == [ZONE_ENTRY]
    assert workers[0].state == INSIDE

    workers, events = monitor.update_all(detect(1, BBOX_INSIDE), now=0.0, wall=100.0)
    assert [e.event_type for e in worker_events(events)] == [DWELL_ALERT]
    assert workers[0].state == ALERT

    # and, like every other alert, it is not raised again
    _, events = monitor.update_all(detect(1, BBOX_INSIDE), now=10.0, wall=110.0)
    assert events == []


def test_repeat_alert_after_repeat_seconds():
    monitor = make_monitor(mode="TIMER", dwell=1.0, repeat=10.0)
    monitor.update_all(detect(1, BBOX_INSIDE), now=0.0, wall=100.0)
    monitor.update_all(detect(1, BBOX_INSIDE), now=1.0, wall=101.0)  # first alert

    _, events = monitor.update_all(detect(1, BBOX_INSIDE), now=5.0, wall=105.0)
    assert events == []

    _, events = monitor.update_all(detect(1, BBOX_INSIDE), now=11.0, wall=111.0)
    assert [e.event_type for e in worker_events(events)] == [DWELL_ALERT]


# -- exit / reset ----------------------------------------------------------

def test_zone_exit_resets_timer():
    monitor = make_monitor(mode="TIMER", dwell=5.0)
    monitor.update_all(detect(1, BBOX_INSIDE), now=0.0, wall=100.0)
    monitor.update_all(detect(1, BBOX_INSIDE), now=3.0, wall=103.0)

    # the first outside frame starts the grace; the exit only lands once it
    # elapses, and its dwell is frozen at the moment they first left
    monitor.update_all(detect(1, BBOX_OUTSIDE), now=4.0, wall=104.0)
    workers, events = monitor.update_all(detect(1, BBOX_OUTSIDE), now=4.7, wall=104.7)
    assert [e.event_type for e in worker_events(events)] == [ZONE_EXIT]
    assert events[0].dwell_seconds == pytest.approx(4.0)
    assert workers[0].state == OUTSIDE
    assert workers[0].entered_at is None
    assert workers[0].dwell_seconds == 0.0


def test_re_entry_starts_a_fresh_timer():
    monitor = make_monitor(mode="TIMER", dwell=5.0)
    monitor.update_all(detect(1, BBOX_INSIDE), now=0.0, wall=100.0)
    monitor.update_all(detect(1, BBOX_INSIDE), now=4.0, wall=104.0)
    # one frame over the edge is inside the grace and no longer an exit, so
    # stay outside until the grace elapses to make the exit real
    monitor.update_all(detect(1, BBOX_OUTSIDE), now=4.5, wall=104.5)
    monitor.update_all(detect(1, BBOX_OUTSIDE), now=5.2, wall=105.2)

    _, events = monitor.update_all(detect(1, BBOX_INSIDE), now=6.0, wall=106.0)
    assert [e.event_type for e in worker_events(events)] == [ZONE_ENTRY]
    assert monitor.get(1).dwell_seconds == 0.0

    _, events = monitor.update_all(detect(1, BBOX_INSIDE), now=10.9, wall=110.9)
    assert events == []
    _, events = monitor.update_all(detect(1, BBOX_INSIDE), now=11.0, wall=111.0)
    assert [e.event_type for e in worker_events(events)] == [DWELL_ALERT]


def test_lost_worker_stays_in_the_zone_until_the_timeout():
    """A single dropped detection must NOT raise ZONE_EXIT.

    The worker keeps its zone state until lost_timeout_frames misses in a
    row, so a momentary detector miss cannot clear an active alert.
    """
    monitor = make_monitor()
    monitor.update_all(detect(7, BBOX_INSIDE), now=0.0, wall=100.0)

    workers, events = monitor.update_all({}, now=1.0, wall=101.0)  # 1 miss

    assert worker_events(events) == []
    assert monitor.get(7).state == INSIDE  # still counted as inside
    assert monitor.inside_count == 1

    # the exit is only raised once the timeout is actually reached
    while monitor.get(7).missing_frames < monitor.lost_timeout_frames - 1:
        _, events = monitor.update_all({}, now=1.0, wall=101.0)
        assert worker_events(events) == []

    workers, events = monitor.update_all({}, now=1.0, wall=101.0)
    assert [e.event_type for e in worker_events(events)] == [ZONE_EXIT]
    assert monitor.get(7).state == OUTSIDE


def test_each_worker_has_its_own_timer():
    monitor = make_monitor(mode="TIMER", dwell=5.0)
    monitor.update_all({**detect(1, BBOX_INSIDE), **detect(2, BBOX_OUTSIDE)}, now=0.0, wall=100.0)

    _, events = monitor.update_all({**detect(1, BBOX_INSIDE), **detect(2, BBOX_INSIDE)}, now=3.0, wall=103.0)
    assert [e.event_type for e in worker_events(events)] == [ZONE_ENTRY]
    assert monitor.get(1).dwell_seconds == pytest.approx(3.0)
    assert monitor.get(2).dwell_seconds == 0.0
    assert monitor.get(1).state == INSIDE


def test_empty_detection_frame_is_safe():
    monitor = make_monitor()
    workers, events = monitor.update_all({}, now=0.0, wall=100.0)
    assert workers == []
    assert events == []


def test_event_timestamps_point_to_entry_and_alert():
    monitor = make_monitor(mode="TIMER", dwell=5.0)
    monitor.update_all(detect(1, BBOX_INSIDE), now=0.0, wall=1000.0)
    _, events = monitor.update_all(detect(1, BBOX_INSIDE), now=5.0, wall=1005.0)
    alert = [e for e in events if e.event_type == DWELL_ALERT][0]

    # entered_at must be the real entry moment, not the alert moment
    assert alert.entered_at == 1000.0
    assert alert.alert_at == 1005.0
    assert alert.dwell_seconds == pytest.approx(5.0)


def test_exit_event_keeps_the_entry_timestamp():
    monitor = make_monitor(mode="TIMER", dwell=5.0)
    monitor.update_all(detect(1, BBOX_INSIDE), now=0.0, wall=1000.0)
    _, events = monitor.update_all(detect(1, BBOX_INSIDE), now=6.0, wall=1006.0)
    monitor.update_all(detect(1, BBOX_OUTSIDE), now=7.0, wall=1007.0)
    _, events = monitor.update_all(detect(1, BBOX_OUTSIDE), now=7.7, wall=1007.7)
    exit_event = events[0]

    assert exit_event.event_type == ZONE_EXIT
    assert exit_event.entered_at == 1000.0
    # dwell is frozen at the frame they first left, not the end of the grace
    assert exit_event.dwell_seconds == pytest.approx(7.0)


def test_re_entry_does_not_reuse_the_old_entry_timestamp():
    monitor = make_monitor(mode="INSTANT")
    monitor.update_all(detect(1, BBOX_INSIDE), now=0.0, wall=2000.0)
    monitor.update_all(detect(1, BBOX_OUTSIDE), now=1.0, wall=2001.0)
    monitor.update_all(detect(1, BBOX_OUTSIDE), now=1.7, wall=2001.7)  # grace elapses
    _, events = monitor.update_all(detect(1, BBOX_INSIDE), now=2.0, wall=2030.0)

    entry, alert = worker_events(events)
    assert entry.entered_at == 2030.0
    assert alert.entered_at == 2030.0
    assert alert.alert_at == 2030.0


# -- exit grace: stepping over the edge must not churn events ---------------

def test_a_worker_flapping_on_the_edge_writes_no_exit_and_keeps_the_clock():
    monitor = make_monitor(mode="TIMER", dwell=5.0, exit_grace=0.6)
    monitor.update_all(detect(1, BBOX_INSIDE), now=0.0, wall=100.0)
    monitor.update_all(detect(1, BBOX_INSIDE), now=2.0, wall=102.0)

    # quick steps over the edge and back, each shorter than the grace
    _, events = monitor.update_all(detect(1, BBOX_OUTSIDE), now=2.2, wall=102.2)
    assert worker_events(events) == []
    _, events = monitor.update_all(detect(1, BBOX_INSIDE), now=2.5, wall=102.5)
    assert worker_events(events) == []
    _, events = monitor.update_all(detect(1, BBOX_OUTSIDE), now=2.7, wall=102.7)
    assert worker_events(events) == []
    _, events = monitor.update_all(detect(1, BBOX_INSIDE), now=3.0, wall=103.0)

    # no exit and no fresh entry: the original clock never restarted
    assert worker_events(events) == []
    assert monitor.get(1).entered_at == 0.0
    assert monitor.get(1).dwell_seconds == pytest.approx(3.0)
    assert monitor.inside_count == 1


def test_a_real_exit_after_the_grace_writes_exactly_one_exit():
    monitor = make_monitor(mode="TIMER", dwell=5.0, exit_grace=0.6)
    monitor.update_all(detect(1, BBOX_INSIDE), now=0.0, wall=100.0)

    # one frame outside is not enough; staying out past the grace is
    monitor.update_all(detect(1, BBOX_OUTSIDE), now=1.0, wall=101.0)
    _, events = monitor.update_all(detect(1, BBOX_OUTSIDE), now=1.7, wall=101.7)
    assert [e.event_type for e in worker_events(events)] == [ZONE_EXIT]

    # and it never fires a second time
    _, events = monitor.update_all(detect(1, BBOX_OUTSIDE), now=3.0, wall=103.0)
    assert worker_events(events) == []
    assert monitor.inside_count == 0
    assert monitor.get(1).state == OUTSIDE


def test_lost_worker_is_forgotten_after_the_timeout():
    monitor = make_monitor()
    for frame in range(14):
        monitor.update_all(detect(9, BBOX_INSIDE), now=float(frame), wall=100.0 + frame)
    assert 9 in monitor.workers

    for frame in range(14, 28):  # 14 frames in a row without the track
        monitor.update_all({}, now=float(frame), wall=100.0 + frame)
    assert 9 in monitor.workers  # 14 misses is not enough yet

    monitor.update_all({}, now=28.0, wall=128.0)  # 15th miss
    assert 9 not in monitor.workers
    assert monitor.summary() == []


def test_brief_detection_gaps_do_not_forget_the_worker():
    monitor = make_monitor()
    for frame in range(14):
        monitor.update_all(detect(9, BBOX_INSIDE), now=float(frame), wall=100.0 + frame)
    monitor.update_all({}, now=15.0, wall=115.0)  # 1 frame gap
    assert 9 in monitor.workers

    monitor.update_all(detect(9, BBOX_INSIDE), now=16.0, wall=116.0)
    assert monitor.get(9).missing_frames == 0


def test_summary_and_stable_ids():
    monitor = make_monitor()
    monitor.update_all({**detect(1, BBOX_OUTSIDE), **detect(3, BBOX_INSIDE)}, now=0.0, wall=100.0)
    summary = monitor.summary()

    assert [row["worker_id"] for row in summary] == [1, 3]
    assert summary[1]["state"] == INSIDE
    assert monitor.get(3).label == "Worker #3"


def test_summary_publishes_the_anchor_point():
    """The dashboard needs the point the zone test actually used.

    This is what lets it say "this person's anchor point is off screen" instead
    of leaving the operator to wonder why somebody reads OUTSIDE.
    """
    monitor = make_monitor()
    monitor.update_all({**detect(1, BBOX_OUTSIDE), **detect(3, BBOX_INSIDE)}, now=0.0, wall=100.0)

    points = {row["worker_id"]: row["point"] for row in monitor.summary()}

    # bottom_center of each box, matching what the monitor stored
    assert points == {1: [20, 40], 3: [200, 250]}
    # a plain list of ints, so it survives JSON unchanged on the way out
    assert isinstance(points[3], list)
    assert all(isinstance(value, int) for value in points[3])


def test_summary_point_tracks_the_anchor_that_was_used():
    """Feet and Body center test different pixels, and the published point has
    to be the one that was actually used - not a fixed spot.

    The anchor is chosen by the caller, which hands update_all the point to
    test, so the two anchors are passed in explicitly here.
    """
    for anchor_point, expected in ((bottom_center(BBOX_INSIDE), [200, 250]),
                                   (center(BBOX_INSIDE), [200, 200])):
        monitor = make_monitor()
        monitor.update_all({3: (anchor_point, BBOX_INSIDE, 0.9)}, now=0.0, wall=100.0)

        row = monitor.summary()[0]
        assert row["point"] == expected
        assert monitor.get(3).state == INSIDE, "both anchors sit inside the square"


# --- #4 a momentary miss must not clear the alert ---------------------------

def test_lost_worker_keeps_counting_against_the_allowed_limit():
    monitor = make_monitor(allowed=1, lost_timeout_frames=5)
    monitor.update_all(
        {**detect(1, BBOX_INSIDE), **detect(2, BBOX_INSIDE)},
        now=0.0,
        wall=100.0,
    )
    assert monitor.capacity_alert is True

    for frame in range(1, 5):  # 4 misses, one short of the timeout
        monitor.update_all({}, now=float(frame), wall=100.0 + frame)
        assert monitor.capacity_alert is True, f"alert cleared on miss {frame}"
        assert monitor.inside_count == 2

    monitor.update_all({}, now=5.0, wall=105.0)  # 5th miss -> timeout
    assert monitor.capacity_alert is False


def test_dwell_timer_keeps_running_while_the_track_is_missing():
    """A miss must not freeze the dwell countdown."""
    monitor = make_monitor(mode="TIMER", dwell=5.0, lost_timeout_frames=10)
    monitor.update_all(detect(1, BBOX_INSIDE), now=0.0, wall=100.0)

    monitor.update_all({}, now=3.0, wall=103.0)  # missed, but still inside

    assert monitor.get(1).state == INSIDE
    assert monitor.get(1).dwell_seconds == pytest.approx(3.0)


def test_a_worker_seen_again_resets_the_missing_counter():
    monitor = make_monitor(lost_timeout_frames=5)
    monitor.update_all(detect(1, BBOX_INSIDE), now=0.0, wall=100.0)

    for frame in range(1, 4):
        monitor.update_all({}, now=float(frame), wall=100.0 + frame)
    assert monitor.get(1).missing_frames == 3

    monitor.update_all(detect(1, BBOX_INSIDE), now=4.0, wall=104.0)

    assert monitor.get(1).missing_frames == 0
    assert monitor.get(1).state == INSIDE


def test_the_timeout_is_honoured_not_a_single_miss():
    monitor = make_monitor(lost_timeout_frames=3)
    monitor.update_all(detect(1, BBOX_INSIDE), now=0.0, wall=100.0)

    monitor.update_all({}, now=1.0, wall=101.0)
    assert 1 in monitor.workers
    monitor.update_all({}, now=2.0, wall=102.0)
    assert 1 in monitor.workers

    monitor.update_all({}, now=3.0, wall=103.0)
    assert 1 not in monitor.workers


def test_a_worker_seen_outside_exits_after_the_grace_not_the_lost_timeout():
    """A real exit waits out the grace, not the 15-frame lost-track timeout."""
    monitor = make_monitor(lost_timeout_frames=15)
    monitor.update_all(detect(1, BBOX_INSIDE), now=0.0, wall=100.0)

    monitor.update_all(detect(1, BBOX_OUTSIDE), now=1.0, wall=101.0)
    _, events = monitor.update_all(detect(1, BBOX_OUTSIDE), now=1.7, wall=101.7)

    assert [e.event_type for e in worker_events(events)] == [ZONE_EXIT]
    assert monitor.inside_count == 0


# --- #7 INSTANT mode alerts ENTRY once per entry, never a dwell alert -------

def test_instant_mode_alerts_once_per_entry():
    monitor = make_monitor(mode="INSTANT", repeat=1.0)
    monitor.update_all(detect(1, BBOX_INSIDE), now=0.0, wall=100.0)

    for frame in range(1, 30):  # 29 more frames with the worker still inside
        _, events = monitor.update_all(detect(1, BBOX_INSIDE), now=float(frame), wall=100.0 + frame)
        assert [e.event_type for e in worker_events(events)] == [], (
            f"repeated alert on frame {frame}"
        )


def test_instant_mode_alerts_again_after_a_new_entry():
    monitor = make_monitor(mode="INSTANT", repeat=1.0)
    monitor.update_all(detect(1, BBOX_INSIDE), now=0.0, wall=100.0)
    monitor.update_all(detect(1, BBOX_OUTSIDE), now=1.0, wall=101.0)
    monitor.update_all(detect(1, BBOX_OUTSIDE), now=1.7, wall=101.7)  # grace elapses

    _, events = monitor.update_all(detect(1, BBOX_INSIDE), now=2.0, wall=102.0)

    assert types_of(worker_events(events)) == [ZONE_ENTRY, ENTRY_ALERT]


def test_timer_mode_still_repeats_the_alert():
    """The repeat feature must survive for the TIMER mode it was built for."""
    monitor = make_monitor(mode="TIMER", dwell=5.0, repeat=2.0)
    monitor.update_all(detect(1, BBOX_INSIDE), now=0.0, wall=100.0)

    _, events = monitor.update_all(detect(1, BBOX_INSIDE), now=5.0, wall=105.0)
    assert types_of(worker_events(events)) == [DWELL_ALERT]

    # 2s repeat window has elapsed while still inside
    _, events = monitor.update_all(detect(1, BBOX_INSIDE), now=7.5, wall=107.5)
    assert types_of(worker_events(events)) == [DWELL_ALERT]


def types_of(events):
    return [e.event_type for e in events]


# --- LIMIT mode alerts only when over the limit ------------------------------

def test_limit_mode_writes_no_entry_or_dwell_alerts():
    """Entries are history rows and dwell time is only tracked, never an
    alert - even after staying far past the dwell limit."""
    monitor = make_monitor(mode="LIMIT", dwell=5.0, allowed=1)

    seen = []
    for frame in range(30):
        _, events = monitor.update_all(detect(1, BBOX_INSIDE), now=float(frame), wall=100.0 + frame)
        seen.extend(types_of(events))

    assert ENTRY_ALERT not in seen
    assert DWELL_ALERT not in seen
    assert ZONE_ENTRY in seen
    assert monitor.get(1).state == INSIDE
    assert monitor.status()["status"] == "SAFE"


def test_limit_mode_one_person_within_allowance_stays_inside():
    """1 person with 1 allowed: orange, not red - INSIDE, no capacity alert."""
    monitor = make_monitor(mode="LIMIT", dwell=5.0, allowed=1)
    _, events = monitor.update_all(detect(3, BBOX_INSIDE), now=0.0, wall=100.0)

    assert CAPACITY_ALERT not in types_of(events)
    assert monitor.capacity_alert is False
    assert monitor.get(3).state == INSIDE
    assert monitor.status()["status"] == "SAFE"


def test_limit_mode_two_people_raise_a_capacity_alert():
    """2 people with 1 allowed: red - a capacity alert carrying the crowd."""
    monitor = make_monitor(mode="LIMIT", dwell=5.0, allowed=1)
    monitor.update_all(detect(3, BBOX_INSIDE), now=0.0, wall=100.0)
    all_events = []
    for frame in range(1, 4):
        _, events = monitor.update_all(
            {3: (bottom_center(BBOX_INSIDE), BBOX_INSIDE, 0.9),
             4: (bottom_center(BBOX_INSIDE), BBOX_INSIDE, 0.9)},
            now=float(frame),
            wall=100.0 + frame,
        )
        all_events.extend(events)

    alerts = [e for e in all_events if e.event_type == CAPACITY_ALERT]
    assert len(alerts) == 1
    assert alerts[0].inside == 2
    assert monitor.status()["status"] == "ALERT"
    # nobody's own clock flips: the zone is over, not the people
    assert monitor.get(3).state == INSIDE
    assert monitor.get(4).state == INSIDE


def test_limit_mode_accepts_a_new_entry_while_over():
    """A third person joining a breach writes no entry alert either."""
    monitor = make_monitor(mode="LIMIT", dwell=5.0, allowed=1)
    monitor.update_all(detect(3, BBOX_INSIDE), now=0.0, wall=100.0)
    monitor.update_all(
        {3: (bottom_center(BBOX_INSIDE), BBOX_INSIDE, 0.9),
         4: (bottom_center(BBOX_INSIDE), BBOX_INSIDE, 0.9)},
        now=1.0,
        wall=101.0,
    )
    _, events = monitor.update_all(
        {3: (bottom_center(BBOX_INSIDE), BBOX_INSIDE, 0.9),
         4: (bottom_center(BBOX_INSIDE), BBOX_INSIDE, 0.9),
         5: (bottom_center(BBOX_INSIDE), BBOX_INSIDE, 0.9)},
        now=2.0,
        wall=102.0,
    )

    assert ENTRY_ALERT not in types_of(events)
    assert DWELL_ALERT not in types_of(events)


# -- several danger zones per machine ---------------------------------------

# two squares side by side: x 100..300 and x 500..700, both y 100..300
ZONE_A = [(100, 100), (300, 100), (300, 300), (100, 300)]
ZONE_B = [(500, 100), (700, 100), (700, 300), (500, 300)]


def multi_monitor(*polygons, **kwargs):
    return ZoneMonitor(
        zone=DangerZone([list(polygon) for polygon in polygons]),
        machine_id="excavator_01",
        **kwargs,
    )


def test_a_flat_point_list_still_means_one_zone():
    """The old single-zone shape has to keep working unchanged."""
    zone = DangerZone(SQUARE)

    assert len(zone) == 1
    assert zone.points == SQUARE
    assert zone.contains(INSIDE_POINT) is True


def test_a_point_inside_any_one_of_the_zones_counts_as_inside():
    zone = DangerZone([ZONE_A, ZONE_B])

    assert zone.contains((200, 200)) is True    # zone A
    assert zone.contains((600, 200)) is True    # zone B
    assert zone.contains((400, 200)) is False   # the gap between them
    assert zone.contains((50, 50)) is False


def test_points_is_still_the_first_zone_for_old_callers():
    zone = DangerZone([ZONE_A, ZONE_B])

    assert zone.points == [tuple(point) for point in ZONE_A]
    assert len(zone.polygons) == 2


def test_a_zone_inside_the_second_zone_is_counted_by_the_monitor():
    monitor = multi_monitor(ZONE_A, ZONE_B, allowed_people=0)
    bbox_in_second_zone = (550, 150, 650, 250)

    workers, events = monitor.update_all(detect(1, bbox_in_second_zone), now=0.0, wall=100.0)

    assert types_of(worker_events(events)) == [ZONE_ENTRY]
    assert workers[0].state == INSIDE
    assert monitor.inside_count == 1


def test_people_in_different_zones_are_all_counted():
    monitor = multi_monitor(ZONE_A, ZONE_B, allowed_people=0)

    monitor.update_all(detect(1, (150, 150, 250, 250)), now=0.0, wall=100.0)
    workers, _ = monitor.update_all(detect(2, (550, 150, 650, 250)), now=0.1, wall=100.1)

    assert monitor.inside_count == 2
    assert all(worker.state == INSIDE for worker in workers)


def test_two_zones_can_alert_at_the_same_time():
    """One worker standing in each zone: both are inside and both alert."""
    monitor = multi_monitor(ZONE_A, ZONE_B, allowed_people=0)

    # one frame with a person in each zone: two zones alert in the same frame
    seen = {**detect(1, (150, 150, 250, 250)), **detect(2, (550, 150, 650, 250))}
    workers, events = monitor.update_all(seen, now=0.0, wall=100.0)

    assert types_of(worker_events(events)) == [ZONE_ENTRY, ZONE_ENTRY]
    assert monitor.inside_count == 2
    assert all(worker.state == INSIDE for worker in workers)


def test_the_gap_between_two_zones_is_outside():
    monitor = multi_monitor(ZONE_A, ZONE_B, allowed_people=0)
    bbox_in_the_gap = (350, 150, 450, 250)

    workers, events = monitor.update_all(detect(1, bbox_in_the_gap), now=0.0, wall=100.0)

    assert worker_events(events) == []
    assert workers[0].state == OUTSIDE
    assert monitor.inside_count == 0


def test_capacity_is_counted_per_zone_not_across_all_of_them():
    """Two zones, one person in each: neither zone is over ITS OWN limit.

    This used to be called test_capacity_is_counted_across_all_the_zones and
    asserted the opposite. The polygons were merged into one union and everybody
    counted against one allowance, so a worker standing in zone 1 used up zone
    2's room. That hid a real breach: two people in a one-person loading bay
    looked fine as long as somebody was standing somewhere else. The allowance
    belongs to the zone, so it is counted there.
    """
    monitor = multi_monitor(ZONE_A, ZONE_B, mode="INSTANT", allowed_people=1)

    monitor.update_all(detect(1, (150, 150, 250, 250)), now=0.0, wall=100.0)
    _, events = monitor.update_all(detect(2, (550, 150, 650, 250)), now=0.1, wall=100.1)

    assert monitor.inside_count == 2
    assert not any(event.event_type == CAPACITY_ALERT for event in events)


def test_two_people_in_one_zone_still_alert_on_that_zone_alone():
    """The other half of the change: a breach inside a single zone is a breach."""
    monitor = multi_monitor(ZONE_A, ZONE_B, mode="INSTANT", allowed_people=1)

    monitor.update_all(detect(1, (150, 150, 250, 250)), now=0.0, wall=100.0)
    _, events = monitor.update_all(detect(1, (150, 150, 250, 250)), now=0.1, wall=100.1)
    events.extend(
        monitor.update_all(
            {**detect(2, (160, 160, 260, 260)), **detect(1, (150, 150, 250, 250))},
            now=0.2,
            wall=100.2,
        )[1]
    )

    assert [event.event_type for event in events if event.event_type == CAPACITY_ALERT]
    breaches = [e for e in events if e.event_type == CAPACITY_ALERT]
    assert [event.zone_index for event in breaches] == [0]


def test_swapping_the_zone_set_moves_people_between_inside_and_outside():
    monitor = multi_monitor(ZONE_A, mode="INSTANT", allowed_people=0)
    monitor.update_all(detect(1, (150, 150, 250, 250)), now=0.0, wall=100.0)
    assert monitor.inside_count == 1

    # the zone moves to the other half of the frame, so the worker is now out
    monitor.zone = DangerZone([ZONE_B])
    flipped = monitor.reset_timing()

    assert flipped == [1]
    assert monitor.inside_count == 0


def test_the_dwell_clock_starts_for_a_worker_the_new_zone_covers():
    """Saving an outline over somebody who was outside it a moment earlier.

    reset_timing() parks them inside with the clock cleared, so the dwell has to
    start on the next frame. Without that the timer stayed at zero for ever and
    the dwell rule never fired for them.
    """
    monitor = multi_monitor(ZONE_A, mode="TIMER", dwell_seconds=1.0, allowed_people=5)
    # the worker is being tracked, but the only zone does not reach them yet
    monitor.update_all(detect(1, (550, 150, 650, 250)), now=0.0, wall=100.0)
    assert monitor.get(1).state == OUTSIDE

    # the outline is saved over them: inside now, clock cleared
    monitor.zone = DangerZone([ZONE_B])
    assert monitor.reset_timing() == [1]
    assert monitor.get(1).entered_at is None
    assert monitor.get(1).state == INSIDE

    # first frame back inside: the fresh clock starts and the entry is recorded
    _, events = monitor.update_all(detect(1, (550, 150, 650, 250)), now=8.0, wall=108.0)
    assert monitor.get(1).entered_at == 8.0
    assert monitor.get(1).dwell_seconds == 0.0
    assert [e.event_type for e in worker_events(events)] == [ZONE_ENTRY]

    # still short of the dwell, so no alarm
    _, events = monitor.update_all(detect(1, (550, 150, 650, 250)), now=8.5, wall=108.5)
    assert monitor.get(1).state == INSIDE
    assert worker_events(events) == []

    # the dwell is reached, and the timer rule fires
    _, events = monitor.update_all(detect(1, (550, 150, 650, 250)), now=9.0, wall=109.0)
    assert monitor.get(1).state == ALERT
    assert [e.event_type for e in worker_events(events)] == [DWELL_ALERT]


def test_a_worker_parked_inside_by_reset_timing_alerts_at_once_in_instant_mode():
    monitor = multi_monitor(ZONE_A, mode="INSTANT", allowed_people=5)
    monitor.update_all(detect(1, (550, 150, 650, 250)), now=0.0, wall=100.0)
    monitor.zone = DangerZone([ZONE_B])
    monitor.reset_timing()

    _, events = monitor.update_all(detect(1, (550, 150, 650, 250)), now=8.0, wall=108.0)

    assert monitor.get(1).state == ALERT
    assert [e.event_type for e in worker_events(events)] == [ZONE_ENTRY, ENTRY_ALERT]


def test_a_zone_with_only_two_points_is_rejected_by_number():
    from zone.danger_zone import ZoneError

    with pytest.raises(ZoneError):
        DangerZone([SQUARE, [(0, 0), (1, 1)]])

def test_a_zone_with_only_two_points_is_rejected_by_number():
    from zone.danger_zone import ZoneError

    with pytest.raises(ZoneError):
        DangerZone([SQUARE, [(0, 0), (1, 1)]])


# -- named zones, one set of rules each --------------------------------------

# two squares that OVERLAP: x 100..400 and x 250..550, both y 100..300.
# A point in the middle (x 250..400) is inside both.
OVER_A = [(100, 100), (400, 100), (400, 300), (100, 300)]
OVER_B = [(250, 100), (550, 100), (550, 300), (250, 300)]
BBOX_OVERLAP = (280, 150, 380, 250)   # bottom-centre (330, 250) -> inside both
BBOX_ONLY_A = (140, 150, 240, 250)     # (190, 250) -> inside A only
BBOX_ONLY_B = (460, 150, 540, 250)     # (500, 250) -> inside B only
BBOX_NOWHERE = (0, 0, 40, 40)          # (20, 40) -> outside both


def named_monitor(polygons, names=None, allowed=0, dwell=5.0, mode="TIMER", repeat=0.0):
    """A monitor whose zones carry names and rules of their own."""
    rules = [
        ZoneRule(
            name=(names[index] if names else f"Zone {index + 1}"),
            points=polygon,
            allowed_people=allowed,
            dwell_seconds=dwell,
            enabled=True,
            using_defaults=False,
        )
        for index, polygon in enumerate(polygons)
    ]
    return ZoneMonitor(
        zone=DangerZone([list(polygon) for polygon in polygons]),
        machine_id="kiln_line",
        mode=mode,
        dwell_seconds=dwell,
        repeat_alert_seconds=repeat,
        allowed_people=allowed,
        zone_rules=rules,
    )


def test_two_zones_can_have_different_allowances():
    monitor = named_monitor([OVER_A, OVER_B], allowed=0)
    monitor.set_zone_rules(
        [
            ZoneRule(name="Kiln Platform", points=[], allowed_people=2, dwell_seconds=5.0, using_defaults=False),
            ZoneRule(name="Loading Bay", points=[], allowed_people=0, dwell_seconds=5.0, using_defaults=False),
        ]
    )

    monitor.update_all(detect(1, BBOX_ONLY_A), now=0.0, wall=100.0)
    monitor.update_all(detect(2, BBOX_ONLY_A), now=0.1, wall=100.1)
    _, events = monitor.update_all(
        {**detect(1, BBOX_ONLY_A), **detect(2, BBOX_ONLY_A), **detect(3, BBOX_ONLY_B)},
        now=0.2,
        wall=100.2,
    )

    breaches = [event.zone_index for event in events if event.event_type == CAPACITY_ALERT]
    assert breaches == [1], "only the zero-allowance zone is over its limit"
    reports = {row["name"]: row for row in monitor.zone_reports()}
    assert reports["Kiln Platform"]["inside"] == 2
    assert reports["Loading Bay"]["inside"] == 1


def test_a_worker_in_zone_1_does_not_use_up_zone_2_s_allowance():
    """Zone 1 full, zone 2 empty: zone 2 must still be quiet."""
    monitor = named_monitor([OVER_A, OVER_B], allowed=1)
    monitor.set_zone_rules(
        [
            ZoneRule(name="Kiln Platform", points=[], allowed_people=1, using_defaults=False),
            ZoneRule(name="Loading Bay", points=[], allowed_people=1, using_defaults=False),
        ]
    )

    monitor.update_all(detect(1, BBOX_ONLY_A), now=0.0, wall=100.0)
    _, events = monitor.update_all(
        {**detect(1, BBOX_ONLY_A), **detect(2, BBOX_ONLY_B)}, now=0.1, wall=100.1
    )

    assert monitor.inside_by_zone[0] == 1
    assert monitor.inside_by_zone[1] == 1
    assert not any(event.event_type == CAPACITY_ALERT for event in events)


def test_a_worker_in_two_zones_is_inside_both_with_one_clock_each():
    monitor = named_monitor([OVER_A, OVER_B], allowed=5, dwell=5.0, mode="TIMER")

    _, events = monitor.update_all(detect(1, BBOX_OVERLAP), now=0.0, wall=100.0)

    assert sorted(event.zone_index for event in worker_events(events)) == [0, 1]
    assert sorted(event.zone_name for event in worker_events(events)) == [
        "Zone 1",
        "Zone 2",
    ]


def test_each_overlapping_zone_alerts_on_its_own_dwell_limit():
    """Two zones, two different limits: the short one fires, the long one waits."""
    monitor = named_monitor([OVER_A, OVER_B], allowed=5, dwell=5.0, mode="TIMER")
    monitor.set_zone_rules(
        [
            ZoneRule(name="Quick", points=[], allowed_people=5, dwell_seconds=2.0, using_defaults=False),
            ZoneRule(name="Slow", points=[], allowed_people=5, dwell_seconds=8.0, using_defaults=False),
        ]
    )

    monitor.update_all(detect(1, BBOX_OVERLAP), now=0.0, wall=100.0)
    _, events = monitor.update_all(detect(1, BBOX_OVERLAP), now=3.0, wall=103.0)

    fired = [e for e in worker_events(events) if e.event_type == DWELL_ALERT]
    assert [event.zone_name for event in fired] == ["Quick"]

    _, events = monitor.update_all(detect(1, BBOX_OVERLAP), now=9.0, wall=109.0)
    fired = [e for e in worker_events(events) if e.event_type == DWELL_ALERT]
    assert [event.zone_name for event in fired] == ["Slow"]


def test_leaving_one_of_two_overlapping_zones_exits_that_zone_only():
    """The other zone's clock keeps running - this is the reason for the change."""
    monitor = named_monitor([OVER_A, OVER_B], allowed=5, dwell=5.0, mode="TIMER")

    monitor.update_all(detect(1, BBOX_OVERLAP), now=0.0, wall=100.0)
    # step west: still in zone 1, out of zone 2. the exit lands only once the
    # grace elapses, so keep zone 2's stamp from the first outside frame
    monitor.update_all(detect(1, BBOX_ONLY_A), now=3.0, wall=103.0)
    _, events = monitor.update_all(detect(1, BBOX_ONLY_A), now=3.6, wall=103.6)

    exits = [e for e in worker_events(events) if e.event_type == ZONE_EXIT]
    assert [event.zone_index for event in exits] == [1]
    assert [event.zone_name for event in exits] == ["Zone 2"]

    # zone 1 kept its original entry time, so its dwell did not restart
    worker = monitor.get(1)
    assert sorted(worker.presences) == [0]
    assert worker.presences[0].entered_at == 0.0
    assert worker.presences[0].dwell_seconds == pytest.approx(3.6)


def test_entering_the_second_zone_does_not_reset_the_first_clocks_dwell():
    """Entering a new zone is not an exit, so zone 1 must not be re-entered."""
    monitor = named_monitor([OVER_A, OVER_B], allowed=5, dwell=5.0, mode="TIMER")

    monitor.update_all(detect(1, BBOX_ONLY_A), now=0.0, wall=100.0)
    _, events = monitor.update_all(detect(1, BBOX_OVERLAP), now=2.0, wall=102.0)

    assert [e.event_type for e in worker_events(events)] == [ZONE_ENTRY]  # zone 2 only
    assert events[0].zone_index == 1
    assert monitor.get(1).presences[0].entered_at == 0.0
    assert monitor.get(1).presences[0].dwell_seconds == pytest.approx(2.0)


def test_one_zone_returning_to_safe_does_not_clear_another_zone_s_alert():
    """Capacity state is per zone: zone 2 going quiet leaves zone 1 alarming."""
    monitor = named_monitor([OVER_A, OVER_B], allowed=0)

    # one person in zone 1 (over its 0 allowance) and one in zone 2 (over its 0)
    _, first = monitor.update_all(detect(1, BBOX_ONLY_A), now=0.0, wall=100.0)
    _, events = monitor.update_all(
        {**detect(1, BBOX_ONLY_A), **detect(2, BBOX_ONLY_B)}, now=0.1, wall=100.1
    )
    alerting = [e.zone_index for e in first + events if e.event_type == CAPACITY_ALERT]
    assert alerting == [0, 1]

    # zone 2 empties: its CAPACITY_SAFE lands only after the exit grace, and
    # it must not touch zone 1
    monitor.update_all(
        {**detect(1, BBOX_ONLY_A), **detect(2, BBOX_ONLY_A)}, now=0.2, wall=100.2
    )
    _, events = monitor.update_all(
        {**detect(1, BBOX_ONLY_A), **detect(2, BBOX_ONLY_A)}, now=0.8, wall=100.8
    )
    safe = [e.zone_index for e in events if e.event_type == CAPACITY_SAFE]
    assert safe == [1]

    assert monitor.capacity_by_zone[0] is True
    assert monitor.capacity_by_zone[1] is False
    assert monitor.status()["status"] == ALERT
    assert monitor.status()["capacity_alert"] is True


def test_a_disabled_zone_never_alerts_and_writes_no_exit_event():
    """Switching a zone off is like redrawing it: silent, and everybody is OUTSIDE."""
    monitor = named_monitor([OVER_A, OVER_B], allowed=0, mode="INSTANT")
    monitor.update_all(detect(1, BBOX_ONLY_A), now=0.0, wall=100.0)
    assert monitor.capacity_by_zone[0] is True

    monitor.set_zone_rules(
        [
            ZoneRule(name="Zone 1", points=[], allowed_people=0, enabled=False, using_defaults=False),
            ZoneRule(name="Zone 2", points=[], allowed_people=0, enabled=True, using_defaults=False),
        ]
    )
    monitor.reset_timing()

    assert monitor.capacity_by_zone[0] is False
    assert monitor.get(1).state == OUTSIDE

    # the next frames must stay quiet in the switched-off zone
    _, events = monitor.update_all(detect(1, BBOX_ONLY_A), now=1.0, wall=101.0)
    assert events == []
    assert monitor.inside_by_zone[0] == 0
    assert monitor.get(1).state == OUTSIDE


def test_the_alert_message_names_the_zone_it_is_about():
    monitor = named_monitor([OVER_A, OVER_B], allowed=5, dwell=2.0, mode="TIMER")
    monitor.set_zone_rules(
        [
            ZoneRule(name="Kiln Platform", points=[], allowed_people=5, dwell_seconds=2.0, using_defaults=False),
            ZoneRule(name="Loading Bay", points=[], allowed_people=5, dwell_seconds=9.0, using_defaults=False),
        ]
    )

    monitor.update_all(detect(3, BBOX_ONLY_A), now=0.0, wall=100.0)
    monitor.update_all(detect(3, BBOX_ONLY_A), now=3.0, wall=103.0)

    assert monitor.status()["message"] == "Worker #3 in Kiln Platform (3.0s)"


def test_the_capacity_message_names_its_zone():
    monitor = named_monitor([OVER_A, OVER_B], allowed=0)
    monitor.set_zone_rules(
        [
            ZoneRule(name="Kiln Platform", points=[], allowed_people=5, using_defaults=False),
            ZoneRule(name="Loading Bay", points=[], allowed_people=0, using_defaults=False),
        ]
    )

    monitor.update_all(detect(4, BBOX_ONLY_B), now=0.0, wall=100.0)

    assert "Loading Bay" in monitor.status()["message"]


def test_renaming_an_alerting_zone_rewrites_the_message():
    """The saved name has to reach the banner, not just the card.

    The message quotes the zone it is about, so a rename has to reach the text
    that is already on screen. It used to survive the rename: the zone's rules
    were replaced, but the message built under the old name was kept, and the
    operator read "Worker #6 is inside Zone 2" about a zone called "place 1".
    """
    monitor = named_monitor([OVER_A, OVER_B], names=["Zone 1", "Zone 2"], allowed=0, mode="INSTANT")
    monitor.update_all(detect(6, BBOX_ONLY_B), now=0.0, wall=100.0)

    assert monitor.status()["message"].endswith("Zone 2")

    # the operator renames zone 2 and saves; the breach never ended
    monitor.set_zone_rules(
        [
            ZoneRule(name="Zone 1", points=[], allowed_people=0, using_defaults=False),
            ZoneRule(name="place 1", points=[], allowed_people=0, using_defaults=False),
        ]
    )
    monitor.reset_timing()

    message = monitor.status()["message"]
    assert message.endswith("place 1")
    assert "Zone 2" not in message
    # still the same breach: the flag never went false, so no second alert
    assert monitor.capacity_by_zone[1] is True
    assert monitor.status()["capacity_alert"] is True


def test_lowering_a_zone_allowance_rewrites_the_message_it_quotes():
    """The message also quotes the allowance, so that number is not frozen either."""
    monitor = named_monitor(
        [OVER_A, OVER_B], names=["Zone 1", "Zone 2"], allowed=9, dwell=90.0, mode="TIMER"
    )
    monitor.update_all(detect(6, BBOX_ONLY_B), now=0.0, wall=100.0)

    assert monitor.status()["status"] == "SAFE"

    # the allowance drops to 0, so the same standing worker is now a breach
    monitor.set_zone_rules(
        [
            ZoneRule(name="Zone 1", points=[], allowed_people=9, dwell_seconds=90.0, using_defaults=False),
            ZoneRule(name="Zone 2", points=[], allowed_people=0, dwell_seconds=90.0, using_defaults=False),
        ]
    )
    monitor.reset_timing()
    monitor.update_all(detect(6, BBOX_ONLY_B), now=0.1, wall=100.1)

    assert monitor.status()["capacity_alert"] is True
    assert "Zone 2" in monitor.status()["message"]


def test_a_worker_is_attributed_to_the_alerting_zone_and_not_the_lowest():
    """Two zones, both containing them: the row reports the one that is alarming."""
    monitor = named_monitor([OVER_A, OVER_B], allowed=5, dwell=5.0, mode="TIMER")
    monitor.set_zone_rules(
        [
            ZoneRule(name="Quiet Zone", points=[], allowed_people=5, dwell_seconds=90.0, using_defaults=False),
            ZoneRule(name="Noisy Zone", points=[], allowed_people=5, dwell_seconds=2.0, using_defaults=False),
        ]
    )

    monitor.update_all(detect(1, BBOX_OVERLAP), now=0.0, wall=100.0)
    monitor.update_all(detect(1, BBOX_OVERLAP), now=3.0, wall=103.0)

    row = next(row for row in monitor.summary() if row["worker_id"] == 1)
    assert row["zone_index"] == 1
    assert row["zone_name"] == "Noisy Zone"
    assert row["state"] == ALERT
    assert row["dwell_seconds"] == pytest.approx(3.0)
    # the mirror fields agree with the row, so the drawing code shows the same
    assert monitor.get(1).zone_index == 1
    assert monitor.get(1).state == ALERT


def test_a_quiet_worker_in_two_zones_is_attributed_to_the_lowest_index():
    monitor = named_monitor([OVER_A, OVER_B], allowed=5, dwell=5.0, mode="TIMER")

    monitor.update_all(detect(1, BBOX_OVERLAP), now=0.0, wall=100.0)

    row = next(row for row in monitor.summary() if row["worker_id"] == 1)
    assert row["zone_index"] == 0
    assert row["zones"] == [0, 1]
    assert row["state"] == INSIDE


def test_a_worker_outside_every_zone_reads_outside_with_no_zone():
    monitor = named_monitor([OVER_A, OVER_B], allowed=5)

    monitor.update_all(detect(1, BBOX_NOWHERE), now=0.0, wall=100.0)

    row = next(row for row in monitor.summary() if row["worker_id"] == 1)
    assert row["state"] == OUTSIDE
    assert row["zone_index"] == -1
    assert row["zones"] == []
    assert row["dwell_seconds"] == 0.0


def test_zone_reports_cover_every_zone_with_its_own_numbers():
    monitor = named_monitor([OVER_A, OVER_B], allowed=1, dwell=5.0, mode="TIMER")
    monitor.set_zone_rules(
        [
            ZoneRule(name="Kiln Platform", points=[], allowed_people=1, dwell_seconds=5.0, using_defaults=False),
            ZoneRule(name="Loading Bay", points=[], allowed_people=0, dwell_seconds=5.0, using_defaults=False),
        ]
    )
    monitor.update_all(detect(1, BBOX_ONLY_A), now=0.0, wall=100.0)
    monitor.update_all(detect(2, BBOX_OVERLAP), now=1.0, wall=101.0)

    reports = monitor.zone_reports()
    assert [row["index"] for row in reports] == [0, 1]
    assert [row["name"] for row in reports] == ["Kiln Platform", "Loading Bay"]
    assert reports[0]["inside"] == 2
    assert reports[0]["allowed_people"] == 1
    assert reports[0]["alerting"] is True
    assert reports[1]["inside"] == 1
    assert reports[1]["allowed_people"] == 0
    assert reports[1]["alerting"] is True
    assert all(row["enabled"] for row in reports)
    assert reports[0]["dwell_seconds"] == pytest.approx(1.0)
    # and its own limit, which is what the first row's dwell is measured against
    assert reports[0]["dwell_limit"] == pytest.approx(5.0)
    assert reports[1]["dwell_limit"] == pytest.approx(5.0)


def test_a_disabled_zone_is_reported_as_disabled_and_empty():
    monitor = named_monitor([OVER_A, OVER_B], allowed=5)
    monitor.set_zone_rules(
        [
            ZoneRule(name="Kiln Platform", points=[], enabled=False, using_defaults=False),
            ZoneRule(name="Loading Bay", points=[], enabled=True, using_defaults=False),
        ]
    )

    reports = monitor.zone_reports()

    assert reports[0]["enabled"] is False
    assert reports[0]["inside"] == 0
    assert reports[1]["enabled"] is True


def test_a_zone_using_the_defaults_follows_a_new_default():
    monitor = named_monitor([OVER_A, OVER_B], allowed=1)
    monitor.set_zone_rules(
        [
            ZoneRule(name="Kiln Platform", points=[], allowed_people=1, using_defaults=True),
            ZoneRule(name="Loading Bay", points=[], allowed_people=1, using_defaults=False),
        ]
    )

    monitor.apply_defaults(allowed_people=4)

    assert monitor.zones[0].allowed_people == 4, "a default-following zone follows it"
    assert monitor.zones[1].allowed_people == 1, "a zone with its own rule keeps it"


def test_a_new_zone_starts_from_the_global_defaults():
    monitor = make_monitor(allowed=7, dwell=3.5)
    monitor.zone = DangerZone([ZONE_A, ZONE_B])
    monitor.reset_timing()

    assert [rule.name for rule in monitor.zones] == ["Zone 1", "Zone 2"]
    assert all(rule.allowed_people == 7 for rule in monitor.zones)
    assert all(rule.dwell_seconds == 3.5 for rule in monitor.zones)
    assert all(rule.using_defaults for rule in monitor.zones)


def test_events_carry_their_reason():
    monitor = make_monitor(mode="TIMER", dwell=5.0)
    _, events = monitor.update_all(detect(1, BBOX_INSIDE), now=0.0, wall=100.0)
    assert [(e.event_type, e.reason) for e in worker_events(events)] == [
        (ZONE_ENTRY, REASON_ENTRY)
    ]
    _, events = monitor.update_all(detect(1, BBOX_INSIDE), now=5.0, wall=105.0)
    assert [(e.event_type, e.reason) for e in worker_events(events)] == [
        (DWELL_ALERT, REASON_DWELL)
    ]


def test_instant_entry_alert_carries_the_entry_reason():
    monitor = make_monitor(mode="INSTANT")
    _, events = monitor.update_all(detect(1, BBOX_INSIDE), now=0.0, wall=100.0)
    assert [(e.event_type, e.reason) for e in worker_events(events)] == [
        (ZONE_ENTRY, REASON_ENTRY),
        (ENTRY_ALERT, REASON_ENTRY),
    ]


def test_capacity_alert_carries_the_capacity_reason():
    monitor = make_monitor(mode="TIMER", allowed=0)
    _, events = monitor.update_all(detect(1, BBOX_INSIDE), now=0.0, wall=100.0)
    caps = [e for e in events if e.event_type == CAPACITY_ALERT]
    assert len(caps) == 1
    assert caps[0].reason == REASON_CAPACITY


def test_events_carry_the_zone_they_are_about():
    monitor = named_monitor([OVER_A, OVER_B], allowed=5, mode="INSTANT")

    _, events = monitor.update_all(detect(1, BBOX_ONLY_B), now=0.0, wall=100.0)

    assert [(e.event_type, e.zone_index, e.zone_name) for e in worker_events(events)] == [
        (ZONE_ENTRY, 1, "Zone 2"),
        (ENTRY_ALERT, 1, "Zone 2"),
    ]


def test_a_forgotten_worker_exits_every_zone_it_was_in():
    monitor = named_monitor(
        [OVER_A, OVER_B], allowed=5, mode="TIMER", repeat=0.0
    )
    monitor.lost_timeout_frames = 2
    monitor.update_all(detect(1, BBOX_OVERLAP), now=0.0, wall=100.0)

    monitor.update_all({}, now=1.0, wall=101.0)
    _, events = monitor.update_all({}, now=2.0, wall=102.0)

    exits = sorted(e.zone_index for e in worker_events(events) if e.event_type == ZONE_EXIT)
    assert exits == [0, 1]

