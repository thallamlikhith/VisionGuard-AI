"""Counts regression harness: persons-per-frame histograms on the 3x clip.

How to run (from anywhere; dependencies from requirements.txt must be
installed, and yolo11n.pt present or downloadable):

    python tools/count_harness.py

It runs the project's own PersonTracker with the reference settings
(confidence=0.25, imgsz=640, detect_every_n=1, openvino=True) over
data/videos/8dc7a9cb1d2f4061a605203063c8a534_x3.mp4 (576 frames) and prints
two histograms. Compare them with tools/BASELINE.md - a difference means the
detection or tracking behaviour changed.
"""

from __future__ import annotations

import os
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import cv2  # noqa: E402

from detector.tracker import PersonTracker  # noqa: E402

CLIP = os.path.join(ROOT, "data", "videos", "8dc7a9cb1d2f4061a605203063c8a534_x3.mp4")


def main() -> int:
    tracker = PersonTracker(
        model_name="yolo11n.pt",
        confidence=0.25,
        imgsz=640,
        detect_every_n=1,
        openvino=True,
    )
    capture = cv2.VideoCapture(CLIP)
    if not capture.isOpened():
        print(f"could not open {CLIP}")
        return 1

    persons_per_frame: Counter = Counter()
    frames_seen_per_id: Counter = Counter()
    frame_index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        people = tracker.track(frame, frame_index)
        persons_per_frame[len(people)] += 1
        for person in people:
            frames_seen_per_id[person.worker_id] += 1
        frame_index += 1
    capture.release()

    print("frames:", frame_index)
    print("persons_per_frame:", dict(sorted(persons_per_frame.items())))
    print("frames_seen_per_id:", dict(sorted(frames_seen_per_id.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
