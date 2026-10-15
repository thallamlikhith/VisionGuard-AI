# Regression baselines

Reference values captured 07-10-2026 (Task 0A/0B). Re-run every check after
any change to detection, tracking, zone logic, or further cleanup.

## 1. Counts harness

```
python tools/count_harness.py
```

Method: the project's `PersonTracker` with `confidence=0.25`, `imgsz=640`,
`detect_every_n=1`, `openvino=True`, over
`data/videos/8dc7a9cb1d2f4061a605203063c8a534_x3.mp4` (576 frames, 1920x1080).

Expected output:

```
frames: 576
persons_per_frame: {0: 51, 1: 171, 2: 354}
frames_seen_per_id: {1: 132, 2: 161, 3: 132, 4: 161, 5: 132, 6: 161}
```

Note: the older expected counts `{0: 258, 1: 120, 2: 198}` came from a
diagnosis harness (`diagnose_counts.py`) that no longer exists anywhere -
working tree, `Prototype.zip`, or the full git history. These values replace
it as the reproducible reference.

## 2. Test suites

| Check | Command | Expected |
| --- | --- | --- |
| Backend | `python -m pytest tests/ -q` | **309 passed** |
| Frontend | `cd frontend && npm test` | **180 passed** |
| Types | `npx tsc -b` | clean, no output |
| Build | `npm run build` | ok |

## 3. File manifest

Before-manifest of the cleanup:
`%TEMP%\opencode\prototype-filelist-before.txt`
(node_modules excluded). Compare with the after-manifest after any deletion.
