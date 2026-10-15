"""Trinetra AI - main video processing loop.

    python app.py --source video.mp4
    python app.py --source 0
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

import cv2

from config import ConfigError, load_config
from database.db import Database, DatabaseError
from detector.tracker import LOG_FORMAT, DetectorError, PersonTracker
from visualize import draw_alert_banner, draw_info, draw_worker, draw_zone
from zone.danger_zone import ALERT, ANCHORS, DangerZone, ZoneError, ZoneMonitor

DEFAULT_DB = os.path.join("data", "trinetra.db")
SNAPSHOT_PATH = os.path.join("data", "workers.json")
SNAPSHOT_EVERY = 5  # frames

# so the line saying which detector backend is in use shows up in the console
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect workers and warn when they enter a machine danger zone.",
    )
    parser.add_argument(
        "--source",
        default="0",
        help="video file path (video.mp4) or webcam index (0, 1, ...). Default: 0",
    )
    parser.add_argument("--config", default="config.yaml", help="path to config.yaml")
    parser.add_argument("--db", default=DEFAULT_DB, help="SQLite database file")
    parser.add_argument("--model", default=None, help="override the YOLO model from config.yaml")
    parser.add_argument("--no-show", action="store_true", help="do not open a window (headless)")
    parser.add_argument("--save", default=None, help="write the annotated video to this file (mp4)")
    parser.add_argument("--max-frames", type=int, default=0, help="stop after N frames (0 = no limit)")
    parser.add_argument(
        "--snapshot",
        default=SNAPSHOT_PATH,
        help="JSON file with the live worker states, read by the dashboard",
    )
    return parser.parse_args(argv)


def open_source(source: str):
    """Return (capture, is_camera). Raises ValueError with a friendly message."""
    is_camera = source.isdigit()
    if is_camera:
        index = int(source)
        if index < 0:
            raise ValueError(f"invalid camera index: {source}")
        path = index
    else:
        if not os.path.isfile(source):
            raise ValueError(f"video file not found: {source}")
        path = source

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        kind = f"camera {source}" if is_camera else f"video '{source}'"
        raise ValueError(f"could not open {kind}")
    return cap, is_camera


class Timeline:
    """Dwell time should follow the video's own clock, not wall clock."""

    def __init__(self, cap, is_camera: bool) -> None:
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.is_camera = is_camera
        self._start = time.monotonic()

    def update(self, cap, frame_index: int) -> float:
        if self.is_camera:
            return time.monotonic() - self._start
        position_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        if position_ms and position_ms > 0:
            self._position_ms = position_ms / 1000.0
            return self._position_ms
        return frame_index / self.fps


def write_snapshot(path: str, machine: dict, rules: dict, workers: list) -> None:
    payload = {
        "machine": machine,
        "rules": rules,
        "updated_at": time.time(),
        "workers": [
            {
                "worker_id": worker.worker_id,
                "state": worker.state,
                "dwell_seconds": round(worker.dwell_seconds, 2),
            }
            for worker in sorted(workers, key=lambda w: w.worker_id)
        ],
    }
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        temp = path + ".tmp"
        with open(temp, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        os.replace(temp, path)
    except OSError as exc:
        print(f"[warn] could not write snapshot {path}: {exc}")


def check_zone_size(zone: DangerZone, width: int, height: int) -> None:
    """Warn about every polygon that reaches past the video."""
    for polygon in zone.polygons:
        max_x = max(point[0] for point in polygon)
        max_y = max(point[1] for point in polygon)
        if max_x > width or max_y > height:
            print(
                f"[warn] danger zone reaches ({max_x}, {max_y}) but the video is only "
                f"{width}x{height} - update danger_zone.points in config.yaml"
            )


def main(argv=None) -> int:
    args = parse_args(argv)

    # -- configuration --------------------------------------------------
    try:
        settings = load_config(args.config)
    except ConfigError as exc:
        print(f"[error] {exc}")
        return 1

    machine = settings["machine"]
    rules = settings["rules"]
    detector_settings = settings["detector"]
    if args.model:
        detector_settings["model"] = args.model

    try:
        zone = DangerZone(settings["danger_zone"]["points"])
        monitor = ZoneMonitor(
            zone=zone,
            machine_id=machine["id"],
            mode=rules["mode"],
            dwell_seconds=rules["dwell_seconds"],
            repeat_alert_seconds=settings["runtime"]["repeat_alert_seconds"],
        )
    except ZoneError as exc:
        print(f"[error] {exc}")
        return 1

    anchor = ANCHORS[settings["danger_zone"]["anchor"]]

    # -- database -------------------------------------------------------
    database = None
    try:
        database = Database(args.db)
        print(f"[db] {args.db}")
    except DatabaseError as exc:
        print(f"[error] {exc}")
        return 1

    # -- video ----------------------------------------------------------
    try:
        cap, is_camera = open_source(args.source)
    except ValueError as exc:
        print(f"[error] {exc}")
        return 2

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    print(f"[video] {'camera ' + args.source if is_camera else args.source} - {width}x{height}")
    print(f"[zone] machine={machine['name']} mode={rules['mode']} dwell={rules['dwell_seconds']}s")
    if zone.is_empty():
        # No zone is a normal state: people are still boxed on the picture,
        # but nothing can ever be inside a polygon that is not there.
        print("No danger zone set")
    else:
        check_zone_size(zone, width, height)
    print(f"[model] loading {detector_settings['model']} at imgsz={detector_settings['imgsz']} "
          f"(detect every {detector_settings['detect_every_n']} frames) ...")

    try:
        tracker = PersonTracker(
            model_name=detector_settings["model"],
            confidence=detector_settings["confidence"],
            person_class_id=detector_settings["person_class_id"],
            imgsz=detector_settings["imgsz"],
            detect_every_n=detector_settings["detect_every_n"],
            openvino=detector_settings.get("openvino", False),
            adaptive_idle_n=detector_settings.get("adaptive_idle_n", 0),
            person_confirm_conf=float(detector_settings.get("person_confirm_conf", 0.4)),
            person_confirm_frames=int(detector_settings.get("person_confirm_frames", 3)),
            person_instant_conf=float(detector_settings.get("person_instant_conf", 0.6)),
            person_confirm_move_ratio=float(detector_settings.get("person_confirm_move_ratio", 0.05)),
        )
    except DetectorError as exc:
        print(f"[error] {exc}")
        cap.release()
        return 1

    print("[run] press 'q' to quit" if not args.no_show else "[run] headless mode")

    writer = None
    if args.save:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            print(f"[warn] could not open writer for {args.save}")
            writer = None

    timeline = Timeline(cap, is_camera)
    frame_index = 0
    event_count = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            now = timeline.update(cap, frame_index)

            try:
                people = tracker.track(frame, frame_index)
            except DetectorError as exc:
                print(f"[error] {exc}")
                break

            detections = {
                person.worker_id: (anchor(person.bbox), person.bbox, person.confidence)
                for person in people
            }
            workers, events = monitor.update_all(detections, now)

            for event in events:
                try:
                    database.insert_event(event)
                    event_count += 1
                    print(f"[event] {event.event_type} Worker #{event.worker_id} dwell={event.dwell_seconds:.1f}s")
                except DatabaseError as exc:
                    print(f"[error] {exc}")

            if frame_index % SNAPSHOT_EVERY == 0:
                write_snapshot(args.snapshot, machine, rules, workers)

            # Red is reserved for an active alert, orange for occupied: the CLI
            # window follows the same three states as the web UI.
            try:
                rows = monitor.zone_reports()
            except (AttributeError, IndexError):
                rows = []
            tones = []
            for index in range(len(zone.polygons)):
                row = rows[index] if index < len(rows) else None
                if row and row.get("alerting"):
                    tones.append("alert")
                elif row and row.get("inside", 0) > 0:
                    tones.append("occupy")
                else:
                    tones.append("idle")
            draw_zone(
                frame,
                zone,
                machine["name"],
                show_fill=settings["runtime"]["show_zone_fill"],
                zone_names=[rule.name for rule in monitor.zones],
                zone_tones=tones,
            )
            alert_zones = {index for index, tone in enumerate(tones) if tone == "alert"}
            for worker in workers:
                in_alert_zone = worker.state == ALERT or any(
                    index in alert_zones for index in worker.presences
                )
                draw_worker(frame, worker, alert=in_alert_zone)
            draw_alert_banner(frame, workers)
            draw_info(
                frame,
                [
                    f"{machine['name']} | mode={rules['mode']} limit={rules['dwell_seconds']}s",
                    f"workers={len(workers)} events={event_count}",
                ],
            )

            if writer is not None:
                writer.write(frame)

            if not args.no_show:
                cv2.imshow("Trinetra AI", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    print("\n[run] stopped by user")
                    break

            frame_index += 1
            if args.max_frames and frame_index >= args.max_frames:
                print(f"\n[run] reached --max-frames {args.max_frames}")
                break
    except KeyboardInterrupt:
        print("\n[run] interrupted")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if not args.no_show:
            cv2.destroyAllWindows()
        write_snapshot(args.snapshot, machine, rules, monitor.workers.values())

    print(f"[done] frames={frame_index} events={event_count}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
