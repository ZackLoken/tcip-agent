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
from a background thread that has already exited by the time the parent is killed. The
stop-path test launches in-process instead, to prove ``stop_tensorboard`` itself ends a child
that ignores SIGTERM before the call returns. The guardian usage-error tests run the guardian's
own ``python -m`` entry point directly, with no parent process spawned at all, since they are
proving the argument parser's refusals rather than any process lifetime.
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


@pytest.mark.skipif(sys.platform == "win32", reason="no guardian there")
def test_stop_ends_a_child_that_ignores_sigterm_before_the_call_returns(monkeypatch, tmp_path):
    """A regression guard for the guardian's grace staying under the manager's wait: without
    it, a stand-in that ignores SIGTERM can be orphaned by a race rather than a deterministic
    failure. The cleanup covers two ways this test itself can fail: while the guardian is
    still alive, its descendants are enumerated by pid, the guardian itself first re-checked
    against the create time captured at launch, and each is re-checked by create time (a pid
    inside the stop's ten-second window can be reused) before any is signalled; once the
    stand-in's own pid is known, captured the moment ``_standin_pid`` answers, it is
    force-killed directly, covering a failure after ``stop_tensorboard`` has already reaped the
    guardian and left the stand-in reparented with nothing watching it. A launch that answers
    with no ``pid`` is beyond this cleanup: the guardian is dead by then, so a stand-in it
    spawned inside the startup grace is already reparented away from every tree this test can
    enumerate.
    """
    from tcip_mcp.pipelines.training import tensorboard_manager as tb

    monkeypatch.setattr(
        tb, "_tensorboard_argv",
        lambda logdir, port: [
            sys.executable, "-c",
            "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(120)",
        ],
    )

    guardian_identity: tuple[int, float] | None = None
    standin_identity: tuple[int, float] | None = None
    try:
        info = tb.launch_tensorboard(str(tmp_path), key="stubborn-run")
        guardian_pid = info.get("pid")
        if guardian_pid is None:
            raise RuntimeError(f"launch_tensorboard did not return a pid: {info}")
        guardian_identity = (guardian_pid, psutil.Process(guardian_pid).create_time())
        standin_num = _standin_pid(guardian_pid, guardian_expected=True)
        standin = psutil.Process(standin_num)
        standin_identity = (standin.pid, standin.create_time())
        result = tb.stop_tensorboard(key="stubborn-run")
        assert result["status"] == "stopped"
        assert not _child_alive(*standin_identity)
    finally:
        # captured before the cleanup below might end the guardian, and only from the guardian
        # launched here, never from a stranger that took its pid.
        descendants: list[tuple[int, float]] = []
        guardian = _same_process(*guardian_identity) if guardian_identity is not None else None
        if guardian is not None:
            try:
                descendants = [
                    (proc.pid, proc.create_time()) for proc in guardian.children(recursive=True)
                ]
            except psutil.NoSuchProcess:
                descendants = []
        tb.stop_tensorboard(key="stubborn-run")
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
    """Run the guardian's real ``python -m`` entry point with ``args``, so a usage-error
    assertion exercises the actual subprocess boundary rather than ``_parse_args`` alone."""
    return subprocess.run(
        [sys.executable, "-m", "tcip_mcp.pipelines.training.tensorboard_guardian", *args],
        capture_output=True, text=True, timeout=30,
    )


def test_guardian_usage_error_for_a_missing_term_grace():
    """The guardian's own argument parsing refuses a missing --term-grace with the usage
    message rather than a traceback, run as the real subprocess entry point."""
    from tcip_mcp.pipelines.training import tensorboard_guardian as guardian

    result = _run_guardian("--parent", str(os.getpid()), "--", sys.executable, "-c", "pass")
    assert result.returncode == 1
    assert guardian._USAGE in result.stderr


def test_guardian_usage_error_for_a_non_integer_parent():
    """A --parent that does not parse as an integer is a usage error, not a traceback."""
    from tcip_mcp.pipelines.training import tensorboard_guardian as guardian

    result = _run_guardian(
        "--parent", "x", "--term-grace", "1", "--", sys.executable, "-c", "pass"
    )
    assert result.returncode == 1
    assert guardian._USAGE in result.stderr


def test_guardian_usage_error_for_a_non_numeric_term_grace():
    """A --term-grace that does not parse as a float is a usage error, not a traceback."""
    from tcip_mcp.pipelines.training import tensorboard_guardian as guardian

    result = _run_guardian(
        "--parent", str(os.getpid()), "--term-grace", "x", "--", sys.executable, "-c", "pass"
    )
    assert result.returncode == 1
    assert guardian._USAGE in result.stderr


def test_guardian_usage_error_for_a_negative_term_grace():
    """A negative --term-grace is a usage error, since the guardian has no wait to use it for."""
    from tcip_mcp.pipelines.training import tensorboard_guardian as guardian

    result = _run_guardian(
        "--parent", str(os.getpid()), "--term-grace", "-1", "--", sys.executable, "-c", "pass"
    )
    assert result.returncode == 1
    assert guardian._USAGE in result.stderr


@pytest.mark.parametrize("term_grace", ["nan", "inf"])
def test_guardian_usage_error_for_a_non_finite_term_grace(term_grace):
    """nan and inf both parse as floats but name no usable escalation wait, so both are usage
    errors rather than a grace the guardian would wait on forever or not at all."""
    from tcip_mcp.pipelines.training import tensorboard_guardian as guardian

    result = _run_guardian(
        "--parent", str(os.getpid()), "--term-grace", term_grace,
        "--", sys.executable, "-c", "pass",
    )
    assert result.returncode == 1
    assert guardian._USAGE in result.stderr


@pytest.mark.parametrize("parent_pid", ["0", "-1"])
def test_guardian_usage_error_for_a_non_positive_parent(parent_pid):
    """A --parent of zero or less parses as an integer but names no real process, so it is a
    usage error rather than a pid the guardian would poll forever."""
    from tcip_mcp.pipelines.training import tensorboard_guardian as guardian

    result = _run_guardian(
        "--parent", parent_pid, "--term-grace", "1", "--", sys.executable, "-c", "pass"
    )
    assert result.returncode == 1
    assert guardian._USAGE in result.stderr
