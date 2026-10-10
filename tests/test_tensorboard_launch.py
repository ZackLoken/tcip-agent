"""launch_tensorboard reports a URL only once the child has proved it survived startup.

The command itself is substituted (a dying one, then a living one) so the process
lifecycle is exercised for real without depending on a TensorBoard install. Three tests use a
``Popen`` stand-in whose own ``wait`` never confirms a kill: one drives ``launch_tensorboard``'s
own failure path, proving the post-kill wait there is caught the same way ``stop_tensorboard``'s
is; one drives ``stop_tensorboard`` itself, proving a kill it cannot confirm is logged there; one
drives ``_stop_all_tracked`` (the exit sweep), proving its own separate log line fires too. The
last test proves the module-level grace-under-wait check by importing a copy of the module with
the constant pushed past its margin.
"""

from __future__ import annotations

import logging
import subprocess
import sys


def test_launch_reports_the_failure_when_the_process_exits_immediately(monkeypatch, tmp_path):
    from tcip_mcp.pipelines.training import tensorboard_manager as tb

    real_popen = subprocess.Popen

    def dying_popen(cmd, **kwargs):
        proc = real_popen(
            [sys.executable, "-c",
             "import sys; sys.stderr.write('tensorboard is not installed\\n'); sys.exit(3)"],
            **kwargs,
        )
        proc.wait(timeout=30)
        return proc

    monkeypatch.setattr(tb.subprocess, "Popen", dying_popen)

    info = tb.launch_tensorboard(str(tmp_path))

    assert "url" not in info
    assert "exited during startup" in info["error"]
    assert "tensorboard is not installed" in info["output"]
    assert tb.running_url(str(tmp_path)) is None


def _never_confirms_kill_popen_class(spawned: list):
    """A ``Popen`` stand-in class whose own ``wait`` always raises ``TimeoutExpired``: it starts
    a real sleeping process so ``kill()`` has something to act on, appends that real process to
    ``spawned`` for the caller to clean up, but never itself reports the kill confirmed."""
    real_popen = subprocess.Popen

    class _NeverConfirmsKillPopen:
        def __init__(self, *args, **kwargs):
            self._real = real_popen([sys.executable, "-c", "import time; time.sleep(120)"])
            spawned.append(self._real)
            self.pid = self._real.pid

        def poll(self):
            return self._real.poll()

        def terminate(self):
            pass  # never lets the graceful path succeed, so the caller always reaches kill().

        def kill(self):
            self._real.kill()

        def wait(self, timeout=None):
            raise subprocess.TimeoutExpired(cmd="standin", timeout=timeout or 0)

    return _NeverConfirmsKillPopen


def test_launch_folds_an_unconfirmed_kill_into_the_error_after_a_raising_tie_assignment(
    monkeypatch, tmp_path
):
    """The launch-failure path's own post-kill wait is caught the same way the stop path's is:
    a tie-assignment failure whose child will not confirm its kill still returns the existing
    error shape, naming the pid, instead of letting TimeoutExpired escape."""
    from tcip_mcp.pipelines.training import tensorboard_manager as tb

    spawned: list[subprocess.Popen] = []
    monkeypatch.setattr(tb.subprocess, "Popen", _never_confirms_kill_popen_class(spawned))
    # _assign_to_win_job exists only on a Windows import of the manager; raising=False installs it
    # on every platform so the forced win32 branch below reaches it there too.
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(
        tb, "_assign_to_win_job",
        lambda proc: (_ for _ in ()).throw(RuntimeError("tie assignment failed")),
        raising=False,
    )

    try:
        info = tb.launch_tensorboard(str(tmp_path))
        assert "error" in info
        assert "tie assignment failed" in info["error"]
        assert "kill unconfirmed" in info["error"]
        assert str(spawned[0].pid) in info["error"]
    finally:
        for proc in spawned:
            if proc.poll() is None:
                proc.kill()
            proc.wait(timeout=30)


def test_stop_logs_a_kill_that_never_confirms(monkeypatch, tmp_path, caplog):
    """``kill_unconfirmed`` is logged (the run id and pid), not only returned, since the exit
    sweep discards the return value and a still-running TensorBoard needs to be visible."""
    from tcip_mcp.pipelines.training import tensorboard_manager as tb

    spawned: list[subprocess.Popen] = []
    monkeypatch.setattr(tb.subprocess, "Popen", _never_confirms_kill_popen_class(spawned))
    # the stand-in carries none of the platform tie's own machinery (no _handle on Windows); the
    # tie itself is not what this test is about.
    monkeypatch.setattr(tb, "_assign_to_win_job", lambda proc: None, raising=False)

    try:
        info = tb.launch_tensorboard(str(tmp_path / "unconfirmed-run"))
        with caplog.at_level(logging.WARNING, logger=tb.logger.name):
            result = tb.stop_tensorboard(str(tmp_path / "unconfirmed-run"))
        assert result["status"] == "kill_unconfirmed"
        assert any(
            "unconfirmed-run" in r.getMessage() and str(info["pid"]) in r.getMessage()
            for r in caplog.records
        )
    finally:
        for proc in spawned:
            if proc.poll() is None:
                proc.kill()
            proc.wait(timeout=30)


def test_exit_sweep_logs_a_kill_that_never_confirms(monkeypatch, tmp_path, caplog):
    """``_stop_all_tracked``'s own log line fires on the run its sweep leaves running, not
    only ``stop_tensorboard``'s own line for a caller that reads its return value."""
    from tcip_mcp.pipelines.training import tensorboard_manager as tb

    spawned: list[subprocess.Popen] = []
    monkeypatch.setattr(tb.subprocess, "Popen", _never_confirms_kill_popen_class(spawned))
    monkeypatch.setattr(tb, "_assign_to_win_job", lambda proc: None, raising=False)

    try:
        info = tb.launch_tensorboard(str(tmp_path / "sweep-unconfirmed-run"))
        with caplog.at_level(logging.WARNING, logger=tb.logger.name):
            tb._stop_all_tracked()
        assert any(
            "Exit sweep" in r.getMessage()
            and "sweep-unconfirmed-run" in r.getMessage()
            and str(info["pid"]) in r.getMessage()
            for r in caplog.records
        )
    finally:
        for proc in spawned:
            if proc.poll() is None:
                proc.kill()
            proc.wait(timeout=30)


def test_launch_returns_a_url_for_a_process_that_stays_up(monkeypatch, tmp_path):
    from tcip_mcp.pipelines.training import tensorboard_manager as tb

    real_popen = subprocess.Popen

    def living_popen(cmd, **kwargs):
        return real_popen([sys.executable, "-c", "import time; time.sleep(30)"], **kwargs)

    monkeypatch.setattr(tb.subprocess, "Popen", living_popen)

    from tcip_mcp.web_client import LOOPBACK_HOST

    info = tb.launch_tensorboard(str(tmp_path))
    try:
        assert info["url"] == f"http://{LOOPBACK_HOST}:{info['port']}"
        assert tb.running_url(str(tmp_path)) == info["url"]
    finally:
        tb.stop_tensorboard(str(tmp_path))
    assert tb.running_url(str(tmp_path)) is None


def test_launch_reports_the_platform_lifetime_tie(monkeypatch, tmp_path):
    """lifetime_tie names the mechanism a normal launch actually got, through the argv seam
    with a sleeping stand-in rather than a real TensorBoard install."""
    from tcip_mcp.pipelines.training import tensorboard_manager as tb

    monkeypatch.setattr(
        tb, "_tensorboard_argv",
        lambda logdir, port: [sys.executable, "-c", "import time; time.sleep(30)"],
    )

    info = tb.launch_tensorboard(str(tmp_path))
    try:
        assert info["lifetime_tie"] == ("job" if sys.platform == "win32" else "guardian")
    finally:
        tb.stop_tensorboard(str(tmp_path))


def test_a_board_is_tracked_under_the_one_spelling_the_store_keys_its_directory_by(
    monkeypatch, tmp_path
):
    """A board launched under one spelling of a log directory that does not exist yet is found
    and stopped under any spelling ``canonical_path`` equates with it, and only those: a case
    variant of the missing tail on a platform whose case rule folds case, a second directory on
    one that does not."""
    from tcip_mcp.pipelines.training import tensorboard_manager as tb
    from tcip_store import canonical_path

    monkeypatch.setattr(
        tb, "_tensorboard_argv",
        lambda logdir, port: [sys.executable, "-c", "import time; time.sleep(30)"],
    )
    launched, other = str(tmp_path / "Board"), str(tmp_path / "BOARD")
    one_directory = canonical_path(launched) == canonical_path(other)

    info = tb.launch_tensorboard(launched)
    try:
        assert (tb.running_url(other) == info["url"]) is one_directory
        assert (tb.stop_tensorboard(other)["status"] == "stopped") is one_directory
        assert (tb.running_url(launched) is None) is one_directory
    finally:
        tb.stop_tensorboard(launched)
        tb.stop_tensorboard(other)


def test_the_guardians_grace_leaves_a_second_under_the_stop_wait():
    """The guardian's kill lands before ``stop_tensorboard`` gives up waiting on it."""
    from tcip_mcp.pipelines.training import tensorboard_manager as tb

    assert tb._GUARDIAN_TERM_GRACE_SECONDS + 1.0 < tb._STOP_WAIT_SECONDS
