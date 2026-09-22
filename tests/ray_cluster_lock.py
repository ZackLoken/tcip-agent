"""One Ray cluster at a time across xdist workers.

A test marked ``ray_cluster`` starts a Ray cluster of its own. Left unserialized, two workers
can start two within seconds of each other, and beside the suite's own workers that has
exhausted this machine's memory and stalled a leg. The mark takes a file lock, shared by every
worker of one xdist session through the run id xdist exports, for the whole of the test's
protocol, so at most one cluster runs at a time; without xdist there is one process and
nothing to serialize. The wrapper runs outside pytest-timeout's, so time spent waiting for the
lock never counts against the test's own timeout. xdist's ``--dist loadgroup`` would serialize
the same tests, but that scheduler hands every worker one test at a time with a controller
round trip between, which halved the suite's throughput when measured here.
``tests/conftest.py`` imports the hook, and a run outside the suite loads it with
``-p tests.ray_cluster_lock``.
"""

from __future__ import annotations

from collections.abc import Generator
import os
from pathlib import Path
import tempfile

import pytest

MARK = "ray_cluster"


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", f"{MARK}: starts a Ray cluster; one such test runs at a time across xdist workers",
    )


@pytest.hookimpl(wrapper=True, tryfirst=True)
def pytest_runtest_protocol(item: pytest.Item) -> Generator[None, object, object]:
    run_id = os.environ.get("PYTEST_XDIST_TESTRUNUID")
    if item.get_closest_marker(MARK) is None or run_id is None:
        return (yield)
    from filelock import FileLock

    with FileLock(Path(tempfile.gettempdir()) / f"tcip-ray-cluster-{run_id}.lock"):
        return (yield)
