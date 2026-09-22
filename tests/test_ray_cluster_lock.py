"""Two ``ray_cluster`` tests on two xdist workers never overlap. A child pytest runs a file
of two marked tests, each recording its start and end to one shared file around a short sleep,
under ``-n 2`` with the lock loaded the way ``tests/conftest.py`` loads it; the intervals must
be disjoint. Without the lock, two workers run them at once.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

TWO_MARKED_TESTS = textwrap.dedent(
    """
    import json
    import os
    import time

    import pytest


    def _record(name):
        started = time.monotonic()
        # Long enough that two workers starting within the same second would overlap.
        time.sleep(1.0)
        with open(os.path.join(os.environ["INTERVALS_DIR"], name + ".json"), "w") as fh:
            json.dump([name, started, time.monotonic()], fh)


    @pytest.mark.ray_cluster
    def test_first():
        _record("first")


    @pytest.mark.ray_cluster
    def test_second():
        _record("second")
    """
)


def test_two_marked_tests_on_two_workers_run_one_after_the_other(tmp_path):
    test_file = tmp_path / "test_two_clusters.py"
    test_file.write_text(TWO_MARKED_TESTS, encoding="utf-8")
    intervals_dir = tmp_path / "intervals"
    intervals_dir.mkdir()
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", str(test_file), "-n", "2", "-p", "no:cacheprovider",
         "-p", "tests.ray_cluster_lock", "-q"],
        cwd=REPO, capture_output=True, text=True, timeout=120,
        env={**os.environ, "INTERVALS_DIR": str(intervals_dir)},
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    rows = [json.loads(path.read_text()) for path in sorted(intervals_dir.glob("*.json"))]
    assert len(rows) == 2, rows
    (_, first_start, first_end), (_, second_start, second_end) = sorted(rows, key=lambda r: r[1])
    assert first_end <= second_start, rows
