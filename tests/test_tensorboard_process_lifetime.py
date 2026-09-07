"""A TensorBoard child's life is tied to the process that launched it.

Before this fix, nothing stopped the child once its launcher was gone: a hard-killed parent
(a capture harness, or a task manager ending a backend) left it serving forever. Each test here
spawns a real parent process (``_tensorboard_parent.py``) that launches a real child through the
manager, with the child's own command swapped for a sleep loop so no TensorBoard install is
required, then watches what becomes of the child once the parent is gone.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import psutil
import pytest

_PARENT_SCRIPT = Path(__file__).parent / "_tensorboard_parent.py"
_DEATH_TIMEOUT = 10.0


def _child_alive(pid: int) -> bool:
    try:
        return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def _wait_until_dead(pid: int, timeout: float = _DEATH_TIMEOUT) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _child_alive(pid):
            return True
        time.sleep(0.2)
    return not _child_alive(pid)


def _force_kill(pid: int) -> None:
    try:
        psutil.Process(pid).kill()
    except psutil.NoSuchProcess:
        pass


def _spawn_parent(tmp_path: Path, mode: str) -> tuple[subprocess.Popen, int]:
    logdir = tmp_path / "tb"
    logdir.mkdir()
    parent = subprocess.Popen(
        [sys.executable, str(_PARENT_SCRIPT), str(logdir), mode],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    pid_line = parent.stdout.readline()
    return parent, int(pid_line.strip())


@pytest.mark.skipif(
    sys.platform == "darwin",
    reason="macOS has neither a job object nor a PDEATHSIG equivalent; a killed parent never runs the atexit hook",
)
def test_child_is_gone_within_ten_seconds_of_a_hard_kill(tmp_path):
    parent, child_pid = _spawn_parent(tmp_path, "sleep")
    try:
        parent.kill()
        parent.wait(timeout=_DEATH_TIMEOUT)
        assert _wait_until_dead(child_pid)
    finally:
        _force_kill(child_pid)


def test_child_is_gone_within_ten_seconds_of_a_normal_exit(tmp_path):
    parent, child_pid = _spawn_parent(tmp_path, "exit")
    try:
        parent.wait(timeout=_DEATH_TIMEOUT)
        assert _wait_until_dead(child_pid)
    finally:
        _force_kill(child_pid)
