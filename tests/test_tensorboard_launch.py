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

import importlib.util
import logging
import subprocess
import sys
from pathlib import Path

import pytest


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

    info = tb.launch_tensorboard(str(tmp_path), key="dead-run")

    assert "url" not in info
    assert "exited during startup" in info["error"]
    assert "tensorboard is not installed" in info["output"]
    assert all(entry["key"] != "dead-run" for entry in tb.list_tensorboard())


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
        info = tb.launch_tensorboard(str(tmp_path), key="raising-tie-run")
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
    monkeypatch.setattr(tb, "_DISABLE_LIFETIME_TIE", True)

    try:
        info = tb.launch_tensorboard(str(tmp_path), key="unconfirmed-run")
        with caplog.at_level(logging.WARNING, logger=tb.logger.name):
            result = tb.stop_tensorboard(key="unconfirmed-run")
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
    monkeypatch.setattr(tb, "_DISABLE_LIFETIME_TIE", True)

    try:
        info = tb.launch_tensorboard(str(tmp_path), key="sweep-unconfirmed-run")
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

    info = tb.launch_tensorboard(str(tmp_path), key="live-run")
    try:
        assert info["url"] == f"http://localhost:{info['port']}"
        assert any(entry["key"] == "live-run" for entry in tb.list_tensorboard())
    finally:
        tb.stop_tensorboard(key="live-run")
    assert all(entry["key"] != "live-run" for entry in tb.list_tensorboard())


def test_launch_reports_the_platform_lifetime_tie(monkeypatch, tmp_path):
    """lifetime_tie names the mechanism a normal launch actually got, through the argv seam
    with a sleeping stand-in rather than a real TensorBoard install."""
    from tcip_mcp.pipelines.training import tensorboard_manager as tb

    monkeypatch.setattr(
        tb, "_tensorboard_argv",
        lambda logdir, port: [sys.executable, "-c", "import time; time.sleep(30)"],
    )

    info = tb.launch_tensorboard(str(tmp_path), key="tie-run")
    try:
        assert info["lifetime_tie"] == ("job" if sys.platform == "win32" else "guardian")
    finally:
        tb.stop_tensorboard(key="tie-run")


def test_launch_reports_the_tie_disabled_under_the_test_seam(monkeypatch, tmp_path):
    from tcip_mcp.pipelines.training import tensorboard_manager as tb

    monkeypatch.setattr(
        tb, "_tensorboard_argv",
        lambda logdir, port: [sys.executable, "-c", "import time; time.sleep(30)"],
    )
    monkeypatch.setattr(tb, "_DISABLE_LIFETIME_TIE", True)

    info = tb.launch_tensorboard(str(tmp_path), key="tie-disabled-run")
    try:
        assert info["lifetime_tie"] == "none: disabled for test"
    finally:
        tb.stop_tensorboard(key="tie-disabled-run")


def test_a_record_id_spelled_like_a_sweeps_key_runs_its_own_board_beside_it(monkeypatch, tmp_path):
    """A run passes launch_tensorboard no key at all, so it is keyed by its own resolved log
    directory, always path-shaped; a sweep's or a trial's key is a bare identifier with no path
    separator (routes/tuning.py's f"sweep_{sweep_id}"). The two keyspaces cannot intersect by
    construction, so a record id spelled exactly like a sweep's own key still runs its own board
    beside it: coverage of that construction, not a race."""
    from tcip_mcp.pipelines.training import tensorboard_manager as tb

    real_popen = subprocess.Popen

    def living_popen(cmd, **kwargs):
        return real_popen([sys.executable, "-c", "import time; time.sleep(30)"], **kwargs)

    monkeypatch.setattr(tb.subprocess, "Popen", living_popen)

    sweep_key = "sweep_abc123"
    sweep_info = tb.launch_tensorboard(str(tmp_path / "sweep"), key=sweep_key)
    run_logdir = tmp_path / sweep_key / "tensorboard"
    run_info = tb.launch_tensorboard(str(run_logdir))
    try:
        assert "url" in sweep_info and "url" in run_info
        assert run_info["logdir"] != sweep_key
        keys = {entry["key"] for entry in tb.list_tensorboard()}
        assert keys == {sweep_key, str(run_logdir.resolve())}
    finally:
        tb.stop_tensorboard(key=sweep_key)
        tb.stop_tensorboard(key=str(run_logdir.resolve()))


def test_manager_refuses_to_import_when_the_grace_leaves_no_margin(tmp_path):
    """The module-level check holds the grace-under-wait constraint in code, not only in
    prose: a copy with the grace pushed past the margin fails at import."""
    from tcip_mcp.pipelines.training import tensorboard_manager as tb

    source = Path(tb.__file__).read_text(encoding="utf-8")
    broken = source.replace(
        "_GUARDIAN_TERM_GRACE_SECONDS = 2.0", "_GUARDIAN_TERM_GRACE_SECONDS = 4.5"
    )
    assert broken != source
    module_path = tmp_path / "broken_tensorboard_manager.py"
    module_path.write_text(broken, encoding="utf-8")

    spec = importlib.util.spec_from_file_location("broken_tensorboard_manager", module_path)
    module = importlib.util.module_from_spec(spec)
    # the dataclass below needs its own module registered to resolve string annotations.
    sys.modules[spec.name] = module
    try:
        with pytest.raises(RuntimeError, match="_GUARDIAN_TERM_GRACE_SECONDS"):
            spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
