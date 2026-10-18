"""One ``ray_cluster``-marked test at a time across xdist workers.

Under xdist, a test marked ``ray_cluster`` runs its whole protocol holding a file lock named by
the run id xdist exports, which every worker of the session shares; without xdist or the mark no
lock is taken. The protocol hook is a ``tryfirst`` wrapper. A run outside the suite loads it
with ``-p tests.ray_cluster_lock``.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest

MARK = "ray_cluster"


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        f"{MARK}: starts a Ray cluster; one such test runs at a time across xdist workers",
    )


@pytest.hookimpl(wrapper=True, tryfirst=True)
def pytest_runtest_protocol(
    item: pytest.Item,
) -> Generator[None, object, object]:
    run_id = os.environ.get("PYTEST_XDIST_TESTRUNUID")
    if item.get_closest_marker(MARK) is None or run_id is None:
        return (yield)
    from filelock import FileLock

    with FileLock(
        Path(tempfile.gettempdir()) / f"tcip-ray-cluster-{run_id}.lock"
    ):
        return (yield)
