"""launch_tensorboard answers an error for a child dying in startup and a URL for a living one.

The command is substituted by real child processes, a dying one and a living one. A ``Popen``
stand-in whose ``wait`` always raises ``TimeoutExpired`` drives ``launch_tensorboard``'s failure
path, ``stop_tensorboard`` and the exit sweep (``_stop_all_tracked``), and each reports or logs
the unconfirmed kill.
"""

from __future__ import annotations

import logging
import subprocess
import sys


class DyingChild(subprocess.Popen):
    """A real child that writes a missing-install message and exits with status 3.

    It runs in place of the command it was handed, and construction waits for its exit.
    """

    def __init__(self, cmd, **kwargs) -> None:
        super().__init__(
            [
                sys.executable,
                "-c",
                "import sys; sys.stderr.write('tensorboard is not installed\\n'); sys.exit(3)",
            ],
            **kwargs,
        )
        self.wait(timeout=30)


class LivingChild(subprocess.Popen):
    """A real child that sleeps for thirty seconds, in place of the command it was handed."""

    def __init__(self, cmd, **kwargs) -> None:
        super().__init__(
            [sys.executable, "-c", "import time; time.sleep(30)"], **kwargs
        )


def test_launch_reports_the_failure_when_the_process_exits_immediately(
    monkeypatch, tmp_path
):
    from tcip_mcp.pipelines.training import tensorboard_manager as tb

    monkeypatch.setattr(tb.subprocess, "Popen", DyingChild)

    info = tb.launch_tensorboard(str(tmp_path))

    assert "url" not in info
    assert "exited during startup" in info["error"]
    assert "tensorboard is not installed" in info["output"]
    assert tb.running_url(str(tmp_path)) is None


def _never_confirms_kill_popen_class(spawned: list):
    """Return a ``Popen`` stand-in class whose own ``wait`` always raises ``TimeoutExpired``.

    It starts a real sleeping process, appended to ``spawned``, which ``kill()`` kills;
    ``terminate`` does nothing.
    """
    real_popen = subprocess.Popen

    class _NeverConfirmsKillPopen:
        def __init__(self, *args, **kwargs):
            self._real = real_popen(
                [sys.executable, "-c", "import time; time.sleep(120)"]
            )
            spawned.append(self._real)
            self.pid = self._real.pid

        def poll(self):
            return self._real.poll()

        def terminate(self):
            pass  # never lets the graceful path succeed, so the caller always reaches kill().

        def kill(self):
            self._real.kill()

        def wait(self, timeout=None):
            raise subprocess.TimeoutExpired(
                cmd="standin", timeout=timeout or 0
            )

    return _NeverConfirmsKillPopen


def test_launch_folds_an_unconfirmed_kill_into_the_error_after_a_raising_tie_assignment(
    monkeypatch, tmp_path
):
    """A failed tie assignment whose child's kill goes unconfirmed returns an error naming both.

    The error names the assignment failure, the unconfirmed kill and the pid.
    """
    from tcip_mcp.pipelines.training import tensorboard_manager as tb

    spawned: list[subprocess.Popen] = []
    monkeypatch.setattr(
        tb.subprocess, "Popen", _never_confirms_kill_popen_class(spawned)
    )
    # _assign_to_win_job exists only on a Windows import of the manager; raising=False installs it
    # on every platform so the forced win32 branch below reaches it there too.
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(
        tb,
        "_assign_to_win_job",
        lambda proc: (_ for _ in ()).throw(
            RuntimeError("tie assignment failed")
        ),
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
    """A stop whose kill goes unconfirmed returns ``kill_unconfirmed``.

    A warning names the run and the pid.
    """
    from tcip_mcp.pipelines.training import tensorboard_manager as tb

    spawned: list[subprocess.Popen] = []
    monkeypatch.setattr(
        tb.subprocess, "Popen", _never_confirms_kill_popen_class(spawned)
    )
    # the stand-in carries none of the platform tie's own machinery (no _handle on Windows); the
    # tie itself is not what this test is about.
    monkeypatch.setattr(
        tb, "_assign_to_win_job", lambda proc: None, raising=False
    )

    try:
        info = tb.launch_tensorboard(str(tmp_path / "unconfirmed-run"))
        with caplog.at_level(logging.WARNING, logger=tb.logger.name):
            result = tb.stop_tensorboard(str(tmp_path / "unconfirmed-run"))
        assert result["status"] == "kill_unconfirmed"
        assert any(
            "unconfirmed-run" in r.getMessage()
            and str(info["pid"]) in r.getMessage()
            for r in caplog.records
        )
    finally:
        for proc in spawned:
            if proc.poll() is None:
                proc.kill()
            proc.wait(timeout=30)


def test_exit_sweep_logs_a_kill_that_never_confirms(
    monkeypatch, tmp_path, caplog
):
    """The exit sweep logs a run whose kill goes unconfirmed.

    Its ``Exit sweep`` warning names the run and its pid.
    """
    from tcip_mcp.pipelines.training import tensorboard_manager as tb

    spawned: list[subprocess.Popen] = []
    monkeypatch.setattr(
        tb.subprocess, "Popen", _never_confirms_kill_popen_class(spawned)
    )
    monkeypatch.setattr(
        tb, "_assign_to_win_job", lambda proc: None, raising=False
    )

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


def test_launch_returns_a_url_for_a_process_that_stays_up(
    monkeypatch, tmp_path
):
    from tcip_mcp.pipelines.training import tensorboard_manager as tb

    monkeypatch.setattr(tb.subprocess, "Popen", LivingChild)

    from tcip_mcp.web_client import LOOPBACK_HOST

    info = tb.launch_tensorboard(str(tmp_path))
    try:
        assert info["url"] == f"http://{LOOPBACK_HOST}:{info['port']}"
        assert tb.running_url(str(tmp_path)) == info["url"]
    finally:
        tb.stop_tensorboard(str(tmp_path))
    assert tb.running_url(str(tmp_path)) is None


def test_launch_reports_the_platform_lifetime_tie(monkeypatch, tmp_path):
    """A launch's lifetime_tie is ``job`` on Windows and ``guardian`` elsewhere.

    The board's argv is replaced by a sleeping Python process.
    """
    from tcip_mcp.pipelines.training import tensorboard_manager as tb

    monkeypatch.setattr(
        tb,
        "_tensorboard_argv",
        lambda logdir, port: [
            sys.executable,
            "-c",
            "import time; time.sleep(30)",
        ],
    )

    info = tb.launch_tensorboard(str(tmp_path))
    try:
        assert info["lifetime_tie"] == (
            "job" if sys.platform == "win32" else "guardian"
        )
    finally:
        tb.stop_tensorboard(str(tmp_path))


def test_a_board_is_tracked_under_the_one_spelling_the_store_keys_its_directory_by(
    monkeypatch, tmp_path
):
    """A board is tracked under the one spelling the store keys its directory by.

    A board launched at the not-yet-existing ``Board`` is found and stopped under ``BOARD``
    exactly when ``canonical_path`` equates the two spellings; when it does not, the board is
    neither found nor stopped there.
    """
    from tcip_mcp.pipelines.training import tensorboard_manager as tb
    from tcip_store import canonical_path

    monkeypatch.setattr(
        tb,
        "_tensorboard_argv",
        lambda logdir, port: [
            sys.executable,
            "-c",
            "import time; time.sleep(30)",
        ],
    )
    launched, other = str(tmp_path / "Board"), str(tmp_path / "BOARD")
    one_directory = canonical_path(launched) == canonical_path(other)

    info = tb.launch_tensorboard(launched)
    try:
        assert (tb.running_url(other) == info["url"]) is one_directory
        assert (
            tb.stop_tensorboard(other)["status"] == "stopped"
        ) is one_directory
        assert (tb.running_url(launched) is None) is one_directory
    finally:
        tb.stop_tensorboard(launched)
        tb.stop_tensorboard(other)


def test_the_guardians_grace_leaves_a_second_under_the_stop_wait():
    """The guardian's term grace sits more than a second under the stop wait."""
    from tcip_mcp.pipelines.training import tensorboard_manager as tb

    assert tb._GUARDIAN_TERM_GRACE_SECONDS + 1.0 < tb._STOP_WAIT_SECONDS
