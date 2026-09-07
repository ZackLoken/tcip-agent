"""A TensorBoard child's life is tied to the process that launched it.

Each test here spawns a real parent process (``_tensorboard_parent.py``) that launches a real
child through the manager, with the child's own command swapped for a sleep loop so no
TensorBoard install is required, then watches what becomes of the child once the parent is gone.
The parent prints the pid ``launch_tensorboard`` returned and the TensorBoard stand-in's own pid
on one line; on POSIX these differ (a guardian process sits in front of the stand-in) and both
deaths are asserted, on Windows they are the same pid. The normal-exit test disables the platform
tie (``--no-tie``) so it proves the ``atexit`` hook alone, since the job object and the guardian
each end the child on a normal exit too; the hard-kill tests keep the tie and prove it directly,
one launching from the main thread and one from a background thread that has already exited by
the time the parent is killed.
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


def _force_kill_and_wait(pid: int, create_time: float, timeout: float = _DEATH_TIMEOUT) -> None:
    proc = _same_process(pid, create_time)
    if proc is None:
        return
    try:
        proc.kill()
        proc.wait(timeout=timeout)
    except (psutil.NoSuchProcess, psutil.TimeoutExpired):
        pass


def _kill_and_wait_parent(parent: subprocess.Popen, timeout: float = _DEATH_TIMEOUT) -> None:
    if parent.poll() is None:
        parent.kill()
    try:
        parent.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
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


def _read_pids(parent: subprocess.Popen) -> tuple[tuple[int, float], tuple[int, float]]:
    """The pid ``launch_tensorboard`` returned and the TensorBoard stand-in's own pid, each read
    and paired with its creation time immediately as it arrives, since a pid's identity is bound
    to the process alive at the moment its creation time was captured, not to the number itself.
    """
    line = parent.stdout.readline()
    returned_pid, standin_pid = (int(p) for p in line.split())
    returned = (returned_pid, psutil.Process(returned_pid).create_time())
    standin = (standin_pid, psutil.Process(standin_pid).create_time())
    return returned, standin


def test_child_is_gone_within_ten_seconds_of_a_hard_kill(tmp_path):
    parent = _spawn_parent(tmp_path, "sleep")
    returned = standin = None
    try:
        returned, standin = _read_pids(parent)
        parent.kill()
        parent.wait(timeout=_DEATH_TIMEOUT)
        assert _wait_until_dead(*standin)
        if sys.platform != "win32":
            assert _wait_until_dead(*returned)
    finally:
        if standin is not None:
            _force_kill_and_wait(*standin)
        if returned is not None:
            _force_kill_and_wait(*returned)
        _kill_and_wait_parent(parent)
        parent.stdout.close()
        parent.stderr.close()


def test_child_survives_its_launching_thread_and_dies_within_ten_seconds_of_a_hard_kill(tmp_path):
    parent = _spawn_parent(tmp_path, "sleep", "--thread")
    returned = standin = None
    try:
        returned, standin = _read_pids(parent)
        # the launching thread has already joined by the time the pid line was printed.
        assert _child_alive(*standin)
        assert parent.poll() is None
        parent.kill()
        parent.wait(timeout=_DEATH_TIMEOUT)
        assert _wait_until_dead(*standin)
        if sys.platform != "win32":
            assert _wait_until_dead(*returned)
    finally:
        if standin is not None:
            _force_kill_and_wait(*standin)
        if returned is not None:
            _force_kill_and_wait(*returned)
        _kill_and_wait_parent(parent)
        parent.stdout.close()
        parent.stderr.close()


def test_child_is_gone_within_ten_seconds_of_a_normal_exit(tmp_path):
    parent = _spawn_parent(tmp_path, "exit", "--no-tie")
    returned = standin = None
    try:
        returned, standin = _read_pids(parent)
        parent.wait(timeout=_DEATH_TIMEOUT)
        assert _wait_until_dead(*standin)
        if sys.platform != "win32":
            assert _wait_until_dead(*returned)
    finally:
        if standin is not None:
            _force_kill_and_wait(*standin)
        if returned is not None:
            _force_kill_and_wait(*returned)
        _kill_and_wait_parent(parent)
        parent.stdout.close()
        parent.stderr.close()
