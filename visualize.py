"""All frame drawing. Kept in one place so app.py stays readable."""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from zone.danger_zone import ALERT, INSIDE, OUTSIDE, DangerZone, Worker

FONT = cv2.FONT_HERSHEY_SIMPLEX

# The three person/zone states, shared by the stream drawing and (as hex) the
# web UI. Red is reserved for an active alert: green = empty/outside, orange =
# inside but within the limit, red = alerting.
COLORS = {
    OUTSIDE: (0, 200, 0),  # green
    INSIDE: (0, 200, 255),  # amber-orange: inside, within the limit
    ALERT: (0, 0, 255),  # red
}
ZONE_COLOR = (0, 165, 255)  # orange: the default danger-area outline
ZONE_FILL = (0, 140, 255)
# an occupied zone keeps the orange outline but fills stronger than an empty
# one, so the two states read apart on the picture
OCCUPY_FILL_ALPHA = 0.35
IDLE_FILL_ALPHA = 0.25
ALERT_ZONE_COLOR = (0, 0, 255)  # red: solid red zone while alerting
ALERT_ZONE_FILL = (0, 0, 255)
# Fire/smoke alert boxes. Both stay clear of the person/zone palette above:
# scarlet (deep orange-red, G=69 vs amber G=200 and alert red G=0) for fire,
# steel blue (no green/amber/red in it) for smoke.
FIRE_BOX_COLOR = (0, 69, 255)
SMOKE_BOX_COLOR = (180, 130, 70)
# PPE violation boxes. Magenta sits clear of every verdict and alert colour
# above (green/amber/red/orange/scarlet/steel-blue): PPE is grading-only, so
# its boxes must never read as an active alert.
PPE_BOX_COLOR = (255, 0, 255)
MACHINE_COLOR = (255, 210, 0)
ALERT_FILL = (0, 0, 190)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# The two warning strips are stacked so they never paint over each other:
# the capacity alert (immediate) sits on top, the dwell alert underneath.
CAPACITY_BANNER_HEIGHT = 40
ALERT_BANNER_HEIGHT = 36
BANNERS_BOTTOM = CAPACITY_BANNER_HEIGHT + ALERT_BANNER_HEIGHT  # 76
INFO_TOP = BANNERS_BOTTOM + 20  # 96


def draw_zone(
    frame,
    zone: DangerZone,
    machine_name: str,
    show_fill: bool = True,
    zone_names: Optional[list] = None,
    zone_tones: Optional[list] = None,
) -> None:
    """Outline + label every polygon, so several danger areas all show up.

    `zone_names` labels each polygon with the name its rules are filed under, so
    a person watching the video can tell which zone an alert is about. Without
    it the label stays what it always was, and with a single zone that keeps the
    familiar "DANGER ZONE - machine" wording rather than inventing a new one.

    `zone_tones` is one "idle" | "occupy" | "alert" per polygon (anything else
    reads as idle). An alerting zone goes solid red; an occupied one keeps the
    orange outline with a stronger fill; an empty one is the default outline.
    """
    tones = list(zone_tones) if zone_tones else []
    for index, polygon in enumerate(zone.polygons):
        if len(polygon) < 3:
            continue
        points = np.array(polygon, dtype=np.int32)
        tone = tones[index] if index < len(tones) else "idle"

        if tone == "alert":
            outline, fill, alpha = ALERT_ZONE_COLOR, ALERT_ZONE_FILL, OCCUPY_FILL_ALPHA
        elif tone == "occupy":
            outline, fill, alpha = ZONE_COLOR, ZONE_FILL, OCCUPY_FILL_ALPHA
        else:
            outline, fill, alpha = ZONE_COLOR, ZONE_FILL, IDLE_FILL_ALPHA

        if show_fill:
            overlay = frame.copy()
            cv2.fillPoly(overlay, [points], fill)
            cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)

        cv2.polylines(frame, [points], True, outline, 2)

        # label above the highest corner of the polygon
        highest = points[int(np.argmin(points[:, 1]))]
        name = zone_names[index] if zone_names and index < len(zone_names) else ""
        text = (
            f"DANGER ZONE - {name}"
            if name and len(zone.polygons) > 1
            else f"DANGER ZONE - {machine_name}"
        )
        cv2.putText(
            frame,
            text,
            (int(highest[0]), max(24, int(highest[1]) - 14)),
            FONT,
            0.6,
            MACHINE_COLOR,
            2,
            cv2.LINE_AA,
        )


def draw_person_box(frame, worker: Worker, alert: bool = False,
                    placed: Optional[list] = None) -> None:
    """The plain version the browser overlay gets: box plus one small label.

    The web UI already shows capacity, alert and zone information as text
    around the video, so the JPEG itself stays almost untouched.

    `alert` forces the red box even when the worker's own state has not
    flipped (yet): a capacity breach alerts the zone while the people inside
    are still INSIDE, and red is reserved for exactly that. Without it an
    over-capacity crowd would stay amber on the picture.

    `placed` is the frame's label registry (see _draw_alert_box): person
    labels claim space first so later PPE/fire/smoke labels avoid them.
    """
    if worker.bbox is None:
        return

    color = COLORS[ALERT] if (alert or worker.state == ALERT) else COLORS.get(worker.state, COLORS[OUTSIDE])
    x1, y1, x2, y2 = worker.bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # The exact point the inside/outside test uses, so the picture shows what is
    # being measured. Without it the dot only appears in the command line window
    # (draw_worker paints it too) and the browser - the only place a real alarm
    # is raised from - cannot explain why somebody counts as inside or outside.
    cv2.circle(frame, worker.point, 5, WHITE, -1)
    cv2.circle(frame, worker.point, 5, color, 1)

    label = f"#{worker.worker_id}  {worker.dwell_seconds:.1f}s"
    scale = 0.45
    thickness = 1
    (text_w, text_h), baseline = cv2.getTextSize(label, FONT, scale, thickness)

    pad = 4
    box_h = text_h + baseline + pad
    if placed is None:
        left = min(max(0, x1), frame.shape[1] - text_w - 2 * pad)
        top = y1 - box_h
        if top < 0:  # no room above the head -> hang the label below the box
            top = min(y1 + 2, frame.shape[0] - box_h)
    else:
        label, (left, top, _right, _bottom) = _claim_label(
            frame, x1, y1, label, scale, thickness, pad, placed)
        # the claim may have shortened the label: remeasure so the painted
        # block matches the registered rect exactly
        (text_w, text_h), baseline = cv2.getTextSize(label, FONT, scale, thickness)
        box_h = text_h + baseline + pad

    cv2.rectangle(frame, (left, top), (left + text_w + 2 * pad, top + box_h), color, -1)
    cv2.putText(
        frame,
        label,
        (left + pad, top + pad + text_h),
        FONT,
        scale,
        BLACK,
        thickness,
        cv2.LINE_AA,
    )


def draw_fire_box(frame, bbox, label: str = "FIRE", placed: Optional[list] = None) -> None:
    """Scarlet box + label for a fire detection (alert palette, not amber)."""
    _draw_alert_box(frame, bbox, label, FIRE_BOX_COLOR, placed)


def draw_smoke_box(frame, bbox, label: str = "SMOKE", placed: Optional[list] = None) -> None:
    """Steel-blue box + label for a smoke detection (no alert colour)."""
    _draw_alert_box(frame, bbox, label, SMOKE_BOX_COLOR, placed)


def draw_ppe_box(frame, bbox, label: str = "PPE", placed: Optional[list] = None) -> None:
    """Magenta box + label for a PPE violation (grading-only, never an alert)."""
    _draw_alert_box(frame, bbox, label, PPE_BOX_COLOR, placed)


def _draw_alert_box(frame, bbox, label: str, color, placed: Optional[list] = None) -> None:
    """One labelled box in the given color. No-ops on a bad/empty box.

    `placed` is the frame's label registry (a list this call appends its
    label rect to): crowded frames cascade instead of painting over each
    other. None keeps the historical above-else-below placement untouched.
    """
    try:
        x1, y1, x2, y2 = (int(v) for v in bbox)
    except (TypeError, ValueError):
        return
    if x2 <= x1 or y2 <= y1:
        return
    height, width = frame.shape[:2]
    x1 = min(max(0, x1), width - 1)
    y1 = min(max(0, y1), height - 1)
    x2 = min(max(0, x2), width - 1)
    y2 = min(max(0, y2), height - 1)
    if x2 <= x1 or y2 <= y1:
        return
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    scale, thickness = 0.5, 1
    pad = 3
    if placed is None:
        top = y1 - _label_height(label, scale, thickness, pad)
        if top < 0:
            top = min(y1 + 2, height - _label_height(label, scale, thickness, pad))
        _paint_label(frame, label, x1, top, scale, thickness, pad, color)
        return
    text, (left, top, _right, _bottom) = _claim_label(
        frame, x1, y1, label, scale, thickness, pad, placed)
    _paint_label(frame, text, left, top, scale, thickness, pad, color)


def _label_height(label: str, scale: float, thickness: int, pad: int) -> int:
    """Full label block height (text + baseline + pad), for slot math."""
    (_text_w, text_h), baseline = cv2.getTextSize(label, FONT, scale, thickness)
    return text_h + baseline + pad


def _paint_label(frame, label: str, left: int, top: int, scale: float,
                 thickness: int, pad: int, color) -> None:
    """Background block + text at an already-chosen slot."""
    (text_w, text_h), baseline = cv2.getTextSize(label, FONT, scale, thickness)
    cv2.rectangle(frame, (left, top), (left + text_w + 2 * pad, top + text_h + baseline + pad),
                  color, -1)
    cv2.putText(frame, label, (left + pad, top + pad + text_h), FONT, scale,
                BLACK, thickness, cv2.LINE_AA)


def _rects_overlap(first: tuple, second: tuple) -> bool:
    """True when two (left, top, right, bottom) rects share interior."""
    return (max(first[0], second[0]) < min(first[2], second[2])
            and max(first[1], second[1]) < min(first[3], second[3]))


def _shorten_label(label: str) -> str:
    """Drop a trailing confidence ("NO HARDHAT 0.71" -> "NO HARDHAT")."""
    head, _, _tail = label.rpartition(" ")
    return head or label


def _claim_label(frame, x1: int, y1: int, label: str, scale: float,
                 thickness: int, pad: int, placed: list) -> tuple:
    """First free slot for a label, registered into `placed`.

    Tries the full label above the box, below it, then cascading further
    down; then the same with the shortened label. Falls back to the clamped
    above slot (drawn anyway, never crashes) when everything is taken.
    Returns (text actually drawn, rect).
    """
    height, width = frame.shape[:2]
    for text in (label, _shorten_label(label)):
        (text_w, text_h), baseline = cv2.getTextSize(text, FONT, scale, thickness)
        box_h = text_h + baseline + pad
        candidates = [y1 - box_h - pad, y1 + 2]
        candidates.extend(y1 + 2 + step * (box_h + 2) for step in (1, 2, 3))
        for top in candidates:
            left = max(0, min(x1, width - text_w - 2 * pad))
            top = max(0, min(top, height - box_h))
            rect = (left, top, left + text_w + 2 * pad, top + box_h)
            if not any(_rects_overlap(rect, other) for other in placed):
                placed.append(rect)
                return text, rect
    text = _shorten_label(label)
    (text_w, text_h), baseline = cv2.getTextSize(text, FONT, scale, thickness)
    box_h = text_h + baseline + pad
    top = max(0, min(y1 - box_h - pad, height - box_h))
    left = max(0, min(x1, width - text_w - 2 * pad))
    rect = (left, top, left + text_w + 2 * pad, top + box_h)
    placed.append(rect)
    return text, rect


def draw_worker(frame, worker: Worker, show_confidence: bool = True, alert: bool = False) -> None:
    """The debug-window version: box plus the full state block.

    `alert` is the same capacity override as draw_person_box: the CLI window
    must agree with the stream about who is red.
    """
    color = COLORS[ALERT] if (alert or worker.state == ALERT) else COLORS.get(worker.state, COLORS[OUTSIDE])
    if worker.bbox is None:
        return

    x1, y1, x2, y2 = worker.bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    lines = [worker.label, worker.state, f"{worker.dwell_seconds:.1f} sec"]
    if show_confidence:
        lines.append(f"conf {worker.confidence:.2f}")

    line_height = 18
    box_width = 135
    top = y1 - line_height * len(lines) - 10
    if top < 0:  # not enough room above the box -> draw below it
        top = min(y1 + 2, frame.shape[0] - line_height * len(lines) - 10)

    cv2.rectangle(frame, (x1, top), (x1 + box_width, top + line_height * len(lines) + 6), color, -1)
    for index, text in enumerate(lines):
        cv2.putText(
            frame,
            text,
            (x1 + 5, top + line_height * (index + 1) - 2),
            FONT,
            0.45,
            BLACK,
            1,
            cv2.LINE_AA,
        )

    # the exact point used for the inside/outside test
    cv2.circle(frame, worker.point, 5, WHITE, -1)
    cv2.circle(frame, worker.point, 5, color, 1)


def draw_alert_banner(frame, workers: list[Worker], top: int = 0) -> None:
    """Dwell alert strip. `top` shifts it down so it cannot overlap the
    capacity banner, which the pipeline draws at top=0."""
    alerting = [worker for worker in workers if worker.state == ALERT]
    if not alerting:
        return

    # -1 on the fill covers both corner rows, so the bottom edge is one row short
    cv2.rectangle(frame, (0, top), (frame.shape[1], top + ALERT_BANNER_HEIGHT - 1), ALERT_FILL, -1)
    text = "ALERT: " + ", ".join(
        f"Worker #{worker.worker_id} in danger zone ({worker.dwell_seconds:.1f} sec)" for worker in alerting
    )
    cv2.putText(frame, text, (10, top + ALERT_BANNER_HEIGHT - 11), FONT, 0.6, WHITE, 2, cv2.LINE_AA)


def draw_info(frame, lines: list[str], top: int = INFO_TOP) -> None:
    y = top
    for line in lines:
        cv2.putText(frame, line, (10, y), FONT, 0.5, WHITE, 1, cv2.LINE_AA)
        y += 18


def draw_capacity_banner(frame, status: dict, top: int = 0) -> None:
    """Big warning strip when more people are inside than are allowed.

    Reads `capacity_alert` from ZoneMonitor.status(). It is drawn on the video
    itself so an over-capacity zone is visible immediately, not only in the
    browser overlay.
    """
    if not status.get("capacity_alert"):
        return

    height = CAPACITY_BANNER_HEIGHT
    # -1 on the fill covers both corner rows, so the bottom edge is one row short
    cv2.rectangle(frame, (0, top), (frame.shape[1], top + height - 1), ALERT_FILL, -1)
    text = f"DANGER ZONE ALERT - inside {status['inside']} / allowed {status['allowed_people']}"
    cv2.putText(frame, text, (10, top + height - 13), FONT, 0.65, WHITE, 2, cv2.LINE_AA)
