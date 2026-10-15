"""Suite-wide isolation for snapshot writers.

Fire/zone alert snapshots otherwise land in the repo's data/ folder whenever
a test drives a crossing with fake detectors. Pointing both directories at a
per-test tmp folder keeps the working tree clean; production paths (env unset
or set explicitly) are unaffected.
"""

import pytest


@pytest.fixture(autouse=True)
def _snapshot_dirs_in_tmp(tmp_path, monkeypatch):
    monkeypatch.setenv("TRINETRA_SNAPSHOT_DIR", str(tmp_path / "snaps"))
    monkeypatch.setenv("TRINETRA_FIRE_SNAPSHOT_DIR", str(tmp_path / "fire-snaps"))
