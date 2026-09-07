"""A TensorBoard child's life is tied to the process that launched it.

Each test here spawns a real parent process (``_tensorboard_parent.py``) that launches a real
child through the manager, with the child's own command swapped for a sleep loop so no
TensorBoard install is required, then watches what becomes of the child once the parent is gone.
The normal-exit test disables the platform tie (``--no-tie``) so it proves the ``atexit`` hook
alone, since the job object and the guardian each end the child on a normal exit too; the
hard-kill tests keep the tie and prove it directly, one launching from the main thread and one
from a background thread that has already exited by the time the parent is killed.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import psutil

_PARENT_SCRIPT = Path(__file__).parent / "_tensorboard_parent.py"
_DEATH_TIMEOUT = 10.0


def _same_process(pid: int, create_time: float) -> psutil.Process | None:
    """The live process at ``pid``, or ``None`` if it is gone or the pid has been reused."""
    try:
        proc = psutil.Process(pid)
        return proc if proc.create_time() == create_time else None
    except psutil.NoSuchProcess:
        return None


def _child_alive(pid: int, create_time: float) -> bool:
    proc = _same_process(pid, create_time)
    if proc is None:
        return False
    try:
        return proc.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def _wait_until_dead(pid: int, create_time: float, timeout: float = _DEATH_TIMEOUT) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _child_alive(pid, create_time):
            return True
        time.sleep(0.2)
    return not _child_alive(pid, create_time)


def _force_kill(pid: int, create_time: float) -> None:
    proc = _same_process(pid, create_time)
    if proc is not None:
        try:
            proc.kill()
        except psutil.NoSuchProcess:
            pass


def _spawn_parent(tmp_path: Path, mode: str, *flags: str) -> subprocess.Popen:
    logdir = tmp_path / "tb"
    logdir.mkdir()
    return subprocess.Popen(
        [sys.executable, str(_PARENT_SCRIPT), str(logdir), mode, *flags],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _read_child_pid(parent: subprocess.Popen) -> tuple[int, float]:
    """The child pid the parent printed, and its creation time for later identity checks."""
    pid_line = parent.stdout.readline()
    child_pid = int(pid_line.strip())
    return child_pid, psutil.Process(child_pid).create_time()


def test_child_is_gone_within_ten_seconds_of_a_hard_kill(tmp_path):
    parent = _spawn_parent(tmp_path, "sleep")
    child_pid = create_time = None
    try:
        child_pid, create_time = _read_child_pid(parent)
        parent.kill()
        parent.wait(timeout=_DEATH_TIMEOUT)
        assert _wait_until_dead(child_pid, create_time)
    finally:
        if child_pid is not None:
            _force_kill(child_pid, create_time)
        parent.stdout.close()
        parent.stderr.close()


def test_child_survives_its_launching_thread_and_dies_within_ten_seconds_of_a_hard_kill(tmp_path):
    parent = _spawn_parent(tmp_path, "sleep", "--thread")
    child_pid = create_time = None
    try:
        child_pid, create_time = _read_child_pid(parent)
        # the launching thread has already joined by the time the pid line was printed.
        assert _child_alive(child_pid, create_time)
        assert parent.poll() is None
        parent.kill()
        parent.wait(timeout=_DEATH_TIMEOUT)
        assert _wait_until_dead(child_pid, create_time)
    finally:
        if child_pid is not None:
            _force_kill(child_pid, create_time)
        parent.stdout.close()
        parent.stderr.close()


def test_child_is_gone_within_ten_seconds_of_a_normal_exit(tmp_path):
    parent = _spawn_parent(tmp_path, "exit", "--no-tie")
    child_pid = create_time = None
    try:
        child_pid, create_time = _read_child_pid(parent)
        parent.wait(timeout=_DEATH_TIMEOUT)
        assert _wait_until_dead(child_pid, create_time)
    finally:
        if child_pid is not None:
            _force_kill(child_pid, create_time)
        parent.stdout.close()
        parent.stderr.close()
