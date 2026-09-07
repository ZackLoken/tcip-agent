"""A TensorBoard child's life is tied to the process that launched it.

Each test here spawns a real parent process (``_tensorboard_parent.py``) that launches a real
child through the manager, with the child's own command swapped for a sleep loop so no
TensorBoard install is required, then watches what becomes of the child once the parent is gone.
The parent prints the pid ``launch_tensorboard`` returned and the TensorBoard stand-in's own pid
on one line; on POSIX these differ only when the tie is in place, since a guardian process then
sits in front of the stand-in, and both deaths are asserted. On Windows, and under ``--no-tie``
where no guardian exists, the two pids are the same, so the ``--no-tie`` test asserts one process.
The normal-exit test disables the platform tie (``--no-tie``) so it proves the ``atexit`` hook
alone, since the job object and the guardian each end the child on a normal exit too; the
hard-kill tests keep the tie and prove it directly, one launching from the main thread and one
from a background thread that has already exited by the time the parent is killed.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import psutil
import pytest

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

    Raises naming the parent's stderr if the pid line never arrives, so a parent that died before
    printing is diagnosed rather than failing on an empty line's unpacking.
    """
    line = parent.stdout.readline()
    if not line:
        raise RuntimeError(
            f"parent process printed no pid line; stderr:\n{parent.stderr.read()}"
        )
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
        # --no-tie means no guardian sits in front of the stand-in: returned and standin name
        # the same process here, so one death check covers both.
        returned, standin = _read_pids(parent)
        parent.wait(timeout=_DEATH_TIMEOUT)
        assert _wait_until_dead(*standin)
    finally:
        if standin is not None:
            _force_kill_and_wait(*standin)
        if returned is not None:
            _force_kill_and_wait(*returned)
        _kill_and_wait_parent(parent)
        parent.stdout.close()
        parent.stderr.close()


def _wait_for_guardian_child(guardian_pid: int, timeout: float = 5.0) -> psutil.Process:
    deadline = time.monotonic() + timeout
    children: list = []
    while time.monotonic() < deadline:
        children = psutil.Process(guardian_pid).children()
        if len(children) == 1:
            return children[0]
        time.sleep(0.1)
    raise AssertionError(
        f"expected exactly one guardian child of pid {guardian_pid} within {timeout} seconds, "
        f"found {len(children)}"
    )


@pytest.mark.skipif(sys.platform == "win32", reason="no guardian there")
def test_stop_ends_a_child_that_ignores_sigterm_before_the_call_returns(monkeypatch, tmp_path):
    """A regression guard for the guardian's grace staying under the manager's wait: without
    it, a stand-in that ignores SIGTERM can be orphaned by a race rather than a deterministic
    failure.
    """
    from tcip_mcp.pipelines.training import tensorboard_manager as tb

    monkeypatch.setattr(
        tb, "_tensorboard_argv",
        lambda logdir, port: [
            sys.executable, "-c",
            "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(120)",
        ],
    )

    standin_pid = standin_create_time = None
    try:
        info = tb.launch_tensorboard(str(tmp_path), key="stubborn-run")
        standin = _wait_for_guardian_child(info["pid"])
        standin_pid, standin_create_time = standin.pid, standin.create_time()
        result = tb.stop_tensorboard(key="stubborn-run")
        assert result["status"] == "stopped"
        assert not _child_alive(standin_pid, standin_create_time)
    finally:
        if standin_pid is not None:
            _force_kill_and_wait(standin_pid, standin_create_time)


def test_guardian_usage_error_for_a_missing_term_grace():
    """The guardian's own argument parsing refuses a missing --term-grace with the usage
    message rather than a traceback, run as the real subprocess entry point."""
    from tcip_mcp.pipelines.training import tensorboard_guardian as guardian

    result = subprocess.run(
        [sys.executable, "-m", "tcip_mcp.pipelines.training.tensorboard_guardian",
         "--parent", str(os.getpid()), "--", sys.executable, "-c", "pass"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 1
    assert guardian._USAGE in result.stderr
