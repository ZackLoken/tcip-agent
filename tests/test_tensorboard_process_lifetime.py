"""A TensorBoard child's life is tied to the process that launched it.

A real parent process (``_tensorboard_parent.py``) launches a real child through the manager,
with the child's own command swapped for a sleep loop, and whether the child lives is checked
after the parent is gone. The parent prints the pid of the child the manager tracks and the
TensorBoard stand-in's own pid on one line; on POSIX
these differ when the tie is in place, since a guardian process sits in front of the
stand-in, and on Windows or under ``--no-tie`` they name one process. Under ``--no-tie`` the
``atexit`` hook alone ends the child on a normal exit; with the tie kept, a hard kill of the
parent ends it, whether the launch ran on the main thread or on a background thread that has
exited. ``stop_tensorboard`` is driven in-process, and the guardian's argument refusals run
through its own ``python -m`` entry point with no parent process.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import psutil
import pytest

from tests._tensorboard_parent import _standin_pid

_PARENT_SCRIPT = Path(__file__).parent / "_tensorboard_parent.py"
_DEATH_TIMEOUT = 10.0


def _same_process(pid: int, create_time: float) -> psutil.Process | None:
    """Return the live process at ``pid``, or ``None`` if it is gone or the pid was reused."""
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


def _wait_until_dead(
    pid: int, create_time: float, timeout: float = _DEATH_TIMEOUT
) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _child_alive(pid, create_time):
            return True
        time.sleep(0.2)
    return not _child_alive(pid, create_time)


def _force_kill_and_wait(
    pid: int, create_time: float, timeout: float = _DEATH_TIMEOUT
) -> None:
    proc = _same_process(pid, create_time)
    if proc is None:
        return
    try:
        proc.kill()
        proc.wait(timeout=timeout)
    except (psutil.NoSuchProcess, psutil.TimeoutExpired):
        pass


def _kill_and_wait_parent(
    parent: subprocess.Popen, timeout: float = _DEATH_TIMEOUT
) -> None:
    if parent.poll() is None:
        parent.kill()
    try:
        parent.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        pass


def _spawn_parent(tmp_path: Path, mode: str, *flags: str) -> subprocess.Popen:
    directory = tmp_path / "run"
    directory.mkdir()
    return subprocess.Popen(
        [sys.executable, str(_PARENT_SCRIPT), str(directory), mode, *flags],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _read_pids(
    parent: subprocess.Popen,
) -> tuple[tuple[int, float], tuple[int, float]]:
    """Return the pid of the child the manager tracks and the stand-in's own, each timed.

    Each pid is paired with its creation time as soon as it is read.

    Raises RuntimeError naming the parent's stderr when the parent prints no pid line.
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


def test_child_is_gone_within_ten_seconds_of_a_hard_kill(
    tmp_path: Path,
) -> None:
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


def test_child_survives_its_launching_thread_and_dies_within_ten_seconds_of_a_hard_kill(
    tmp_path: Path,
) -> None:
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


def test_child_is_gone_within_ten_seconds_of_a_normal_exit(
    tmp_path: Path,
) -> None:
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


@pytest.mark.skipif(sys.platform == "win32", reason="no guardian there")
def test_stop_ends_a_child_that_ignores_sigterm_before_the_call_returns(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A stand-in that ignores SIGTERM is ended before ``stop_tensorboard`` returns."""
    from tcip_mcp.pipelines.training import tensorboard_manager as tb
    from tests._tensorboard_parent import listening_argv

    monkeypatch.setattr(
        tb,
        "_tensorboard_argv",
        lambda logdir, port, prefix: listening_argv(port, ignore_term=True),
    )

    guardian_identity: tuple[int, float] | None = None
    standin_identity: tuple[int, float] | None = None
    key = tb._board_key(tmp_path)
    try:
        info = tb.launch_tensorboard(tmp_path)
        if not isinstance(info, tb.TensorboardServing):
            raise RuntimeError(f"launch_tensorboard did not serve: {info}")
        guardian_pid = tb._TB_PROCESSES[key].proc.pid
        guardian_identity = (
            guardian_pid,
            psutil.Process(guardian_pid).create_time(),
        )
        standin_num = _standin_pid(guardian_pid, guardian_expected=True)
        standin = psutil.Process(standin_num)
        standin_identity = (standin.pid, standin.create_time())
        result = tb.stop_tensorboard(key)
        assert result["status"] == "stopped"
        assert not _child_alive(*standin_identity)
    finally:
        # captured before the cleanup below might end the guardian, and only from the guardian
        # launched here, never from a stranger that took its pid.
        descendants: list[tuple[int, float]] = []
        guardian = (
            _same_process(*guardian_identity)
            if guardian_identity is not None
            else None
        )
        if guardian is not None:
            try:
                descendants = [
                    (proc.pid, proc.create_time())
                    for proc in guardian.children(recursive=True)
                ]
            except psutil.NoSuchProcess:
                descendants = []
        tb.stop_tensorboard(key)
        live: list[psutil.Process] = []
        for pid, create_time in descendants:
            proc = _same_process(pid, create_time)
            if proc is None:
                continue
            try:
                proc.kill()
                live.append(proc)
            except psutil.NoSuchProcess:
                pass
        psutil.wait_procs(live, timeout=_DEATH_TIMEOUT)
        if standin_identity is not None:
            _force_kill_and_wait(*standin_identity)


def _run_guardian(*args: str) -> subprocess.CompletedProcess:
    """Run the guardian's real ``python -m`` entry point with ``args``; return the result."""
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "tcip_mcp.pipelines.training.tensorboard_guardian",
            *args,
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_guardian_usage_error_for_a_missing_term_grace() -> None:
    """The guardian refuses a missing --term-grace with its usage message.

    Run through its real subprocess entry point, it exits 1 with the usage message on stderr.
    """
    from tcip_mcp.pipelines.training import tensorboard_guardian as guardian

    result = _run_guardian(
        "--parent", str(os.getpid()), "--", sys.executable, "-c", "pass"
    )
    assert result.returncode == 1
    assert guardian._USAGE in result.stderr


def test_guardian_usage_error_for_a_non_integer_parent() -> None:
    """A --parent that does not parse as an integer exits 1 with the usage message."""
    from tcip_mcp.pipelines.training import tensorboard_guardian as guardian

    result = _run_guardian(
        "--parent",
        "x",
        "--term-grace",
        "1",
        "--",
        sys.executable,
        "-c",
        "pass",
    )
    assert result.returncode == 1
    assert guardian._USAGE in result.stderr


def test_guardian_usage_error_for_a_non_numeric_term_grace() -> None:
    """A --term-grace that does not parse as a float exits 1 with the usage message."""
    from tcip_mcp.pipelines.training import tensorboard_guardian as guardian

    result = _run_guardian(
        "--parent",
        str(os.getpid()),
        "--term-grace",
        "x",
        "--",
        sys.executable,
        "-c",
        "pass",
    )
    assert result.returncode == 1
    assert guardian._USAGE in result.stderr


def test_guardian_usage_error_for_a_negative_term_grace() -> None:
    """A negative --term-grace is a usage error, since the guardian has no wait to use it for."""
    from tcip_mcp.pipelines.training import tensorboard_guardian as guardian

    result = _run_guardian(
        "--parent",
        str(os.getpid()),
        "--term-grace",
        "-1",
        "--",
        sys.executable,
        "-c",
        "pass",
    )
    assert result.returncode == 1
    assert guardian._USAGE in result.stderr


@pytest.mark.parametrize("term_grace", ["nan", "inf"])
def test_guardian_usage_error_for_a_non_finite_term_grace(
    term_grace: str,
) -> None:
    """The guardian refuses a non-finite --term-grace with its usage message.

    Nan and inf both parse as floats; each exits 1 with the usage message.
    """
    from tcip_mcp.pipelines.training import tensorboard_guardian as guardian

    result = _run_guardian(
        "--parent",
        str(os.getpid()),
        "--term-grace",
        term_grace,
        "--",
        sys.executable,
        "-c",
        "pass",
    )
    assert result.returncode == 1
    assert guardian._USAGE in result.stderr


@pytest.mark.parametrize("parent_pid", ["0", "-1"])
def test_guardian_usage_error_for_a_non_positive_parent(
    parent_pid: str,
) -> None:
    """The guardian refuses a --parent of zero or less with its usage message.

    Such a value parses as an integer; it exits 1 with the usage message.
    """
    from tcip_mcp.pipelines.training import tensorboard_guardian as guardian

    result = _run_guardian(
        "--parent",
        parent_pid,
        "--term-grace",
        "1",
        "--",
        sys.executable,
        "-c",
        "pass",
    )
    assert result.returncode == 1
    assert guardian._USAGE in result.stderr
