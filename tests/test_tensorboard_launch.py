"""launch_tensorboard answers an error for a child dying in startup and a URL for a living one.

The command is substituted by real child processes, a dying one and living ones accepting on the
port the launch hands them. A ``Popen``
stand-in whose ``wait`` always raises ``TimeoutExpired`` drives ``launch_tensorboard``'s failure
path, ``stop_tensorboard`` and the exit sweep (``_stop_all_tracked``), and each reports or logs
the unconfirmed kill.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from typing import TYPE_CHECKING

from tests._tensorboard_parent import listening_argv

if TYPE_CHECKING:
    from pathlib import Path
    from subprocess import _FILE
    from typing import NoReturn, TypedDict, Unpack

    import pytest

    class _PopenStreams(TypedDict, total=False):
        """The stream keywords ``launch_tensorboard`` hands ``Popen``."""

        stdout: _FILE
        stderr: _FILE


class DyingChild(subprocess.Popen):
    """A real child that writes a missing-install message and exits with status 3.

    It runs in place of the command it was handed, and construction waits for its exit.
    """

    def __init__(
        self, cmd: list[str], **kwargs: Unpack[_PopenStreams]
    ) -> None:
        super().__init__(
            [
                sys.executable,
                "-c",
                "import sys; sys.stderr.write('tensorboard is not installed\\n'); sys.exit(3)",
            ],
            **kwargs,
        )
        self.wait(timeout=30)


def _port_of(cmd: list[str]) -> int:
    """Return the ``--port`` a TensorBoard command line names."""
    return int(cmd[cmd.index("--port") + 1])


class LivingChild(subprocess.Popen):
    """A real child accepting on the command's port for thirty seconds, in place of the command."""

    def __init__(
        self, cmd: list[str], **kwargs: Unpack[_PopenStreams]
    ) -> None:
        super().__init__(listening_argv(_port_of(cmd), seconds=30), **kwargs)


def test_launch_reports_the_failure_when_the_process_exits_immediately(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from tcip_mcp.pipelines.training import tensorboard_manager as tb

    monkeypatch.setattr(tb.subprocess, "Popen", DyingChild)

    info = tb.launch_tensorboard(tmp_path)

    assert isinstance(info, tb.TensorboardFailure)
    assert "exited during startup" in info.error
    assert info.output is not None
    assert "tensorboard is not installed" in info.output
    assert tb.running(tmp_path) is None


def test_a_launch_that_takes_no_port_answers_its_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A launch that cannot take a port answers a failure naming why and tracks nothing."""
    from tcip_mcp import web_client
    from tcip_mcp.pipelines.training import tensorboard_manager as tb

    def no_port(requested: int) -> int:
        raise OSError("no port could be bound")

    monkeypatch.setattr(web_client, "free_port", no_port)

    info = tb.launch_tensorboard(tmp_path)

    assert isinstance(info, tb.TensorboardFailure)
    assert "no port could be bound" in info.error
    assert tb.running(tmp_path) is None


def _never_confirms_kill_popen_class(spawned: list[subprocess.Popen]) -> type:
    """Return a ``Popen`` stand-in class whose own ``wait`` always raises ``TimeoutExpired``.

    It starts a real sleeping process, appended to ``spawned``, which ``kill()`` kills;
    ``terminate`` does nothing.
    """
    real_popen = subprocess.Popen

    class _NeverConfirmsKillPopen:
        def __init__(self, cmd: list[str], **kwargs: object) -> None:
            self._real = real_popen(listening_argv(_port_of(cmd)))
            spawned.append(self._real)
            self.pid = self._real.pid

        def poll(self) -> int | None:
            return self._real.poll()

        def terminate(self) -> None:
            pass  # never lets the graceful path succeed, so the caller always reaches kill().

        def kill(self) -> None:
            self._real.kill()

        def wait(self, timeout: float | None = None) -> NoReturn:
            raise subprocess.TimeoutExpired(
                cmd="standin", timeout=timeout or 0
            )

    return _NeverConfirmsKillPopen


def test_launch_folds_an_unconfirmed_kill_into_the_error_after_a_raising_tie_assignment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
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
        info = tb.launch_tensorboard(tmp_path)
        assert isinstance(info, tb.TensorboardFailure)
        assert "tie assignment failed" in info.error
        assert "kill unconfirmed" in info.error
        assert str(spawned[0].pid) in info.error
    finally:
        for proc in spawned:
            if proc.poll() is None:
                proc.kill()
            proc.wait(timeout=30)


def test_stop_logs_a_kill_that_never_confirms(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
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
        run = tmp_path / "unconfirmed-run"
        info = tb.launch_tensorboard(run)
        assert isinstance(info, tb.TensorboardServing)
        with caplog.at_level(logging.WARNING, logger=tb.logger.name):
            result = tb.stop_tensorboard(tb._board_key(run))
        assert result["status"] == "kill_unconfirmed"
        assert any(
            "unconfirmed-run" in r.getMessage()
            and str(spawned[0].pid) in r.getMessage()
            for r in caplog.records
        )
    finally:
        for proc in spawned:
            if proc.poll() is None:
                proc.kill()
            proc.wait(timeout=30)


def test_exit_sweep_logs_a_kill_that_never_confirms(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
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
        info = tb.launch_tensorboard(tmp_path / "sweep-unconfirmed-run")
        assert isinstance(info, tb.TensorboardServing)
        with caplog.at_level(logging.WARNING, logger=tb.logger.name):
            tb._stop_all_tracked()
        assert any(
            "Exit sweep" in r.getMessage()
            and "sweep-unconfirmed-run" in r.getMessage()
            and str(spawned[0].pid) in r.getMessage()
            for r in caplog.records
        )
    finally:
        for proc in spawned:
            if proc.poll() is None:
                proc.kill()
            proc.wait(timeout=30)


def test_launch_serves_the_board_path_for_a_process_that_stays_up(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A living child serves under its run's board path, which its URL names, until stopped."""
    from tcip_mcp.pipelines.training import tensorboard_manager as tb

    monkeypatch.setattr(tb.subprocess, "Popen", LivingChild)

    info = tb.launch_tensorboard(tmp_path)
    try:
        assert isinstance(info, tb.TensorboardServing)
        assert info.url == f"/api/training/runs/{tmp_path.name}/tensorboard/"
        assert tb.running(tmp_path) == info
    finally:
        tb.stop_tensorboard(tb._board_key(tmp_path))
    assert tb.running(tmp_path) is None


def test_launch_answers_its_url_only_once_the_board_accepts_a_connection(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A board that takes two seconds to listen is reachable at the URL the launch answers."""
    import socket

    from tcip_mcp.pipelines.training import tensorboard_manager as tb
    from tcip_mcp.web_client import LOOPBACK_HOST

    monkeypatch.setattr(
        tb,
        "_tensorboard_argv",
        lambda logdir, port, prefix: listening_argv(
            port, delay=2.0, seconds=30
        ),
    )

    info = tb.launch_tensorboard(tmp_path)
    try:
        assert isinstance(info, tb.TensorboardServing)
        socket.create_connection((LOOPBACK_HOST, info.port), timeout=1).close()
    finally:
        tb.stop_tensorboard(tb._board_key(tmp_path))


def test_a_board_is_tracked_under_the_one_spelling_the_store_keys_its_directory_by(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A board is tracked under the one spelling the store keys its directory by.

    A board launched for the not-yet-existing run ``Board`` is found and stopped under ``BOARD``
    exactly when ``canonical_path`` equates the two spellings; when it does not, the board is
    neither found nor stopped there.
    """
    from tcip_mcp.experiments import board_of
    from tcip_mcp.pipelines.training import tensorboard_manager as tb
    from tcip_store import canonical_path

    monkeypatch.setattr(
        tb,
        "_tensorboard_argv",
        lambda logdir, port, prefix: listening_argv(port, seconds=30),
    )
    launched, other = tmp_path / "Board", tmp_path / "BOARD"
    one_directory = canonical_path(launched) == canonical_path(other)

    info = tb.launch_tensorboard(launched)
    try:
        assert (tb.running(other) == info) is one_directory
        assert (
            tb.stop_tensorboard(str(board_of(other)))["status"] == "stopped"
        ) is one_directory
        assert (tb.running(launched) is None) is one_directory
    finally:
        tb.stop_tensorboard(str(board_of(launched)))
        tb.stop_tensorboard(str(board_of(other)))


def test_the_installed_tensorboard_declares_the_reload_selector() -> None:
    """The reload selector names an element of the installed reload component's own template.

    The installed bundle (``webfiles.zip``'s ``index.js``) compiles the component its first part
    names into a definition whose template starts elements by tag and attribute-constant index;
    one of them must be the selector's element, carrying its class among the classes its
    constants declare (after Angular's class marker, ``1``). A TensorBoard renaming the
    component, the element or the class fails here.
    """
    import json
    import re
    import zipfile
    from importlib.resources import files
    from itertools import takewhile

    from tcip_mcp.pipelines.training.tensorboard_manager import RELOAD_SELECTOR

    component, element = RELOAD_SELECTOR.split(" ")
    tag, class_name = element.split(".")
    with (
        files("tensorboard").joinpath("webfiles.zip").open("rb") as bundle,
        zipfile.ZipFile(bundle) as webfiles,
    ):
        script = webfiles.read("index.js").decode("utf-8")

    definition = script[script.index(f'selectors:[["{component}"]]') :]
    consts_at = definition.index("consts:") + len("consts:")
    consts, _ = json.JSONDecoder().raw_decode(definition[consts_at:])
    template_at = definition.index("template:function")
    template = definition[
        template_at : definition.index("dependencies:", template_at)
    ]
    started = [
        (started_tag, consts[int(index)])
        for started_tag, index in re.findall(
            r'\w+\(\d+,"([\w-]+)",(\d+)\)', template
        )
    ]

    def classes(attributes: list) -> list[str]:
        if 1 not in attributes:
            return []
        after = attributes[attributes.index(1) + 1 :]
        return list(takewhile(lambda value: isinstance(value, str), after))

    assert any(
        started_tag == tag and class_name in classes(attributes)
        for started_tag, attributes in started
    ), started


def test_the_guardians_grace_leaves_a_second_under_the_stop_wait() -> None:
    """The guardian's term grace sits more than a second under the stop wait."""
    from tcip_mcp.pipelines.training import tensorboard_manager as tb

    assert tb._GUARDIAN_TERM_GRACE_SECONDS + 1.0 < tb._STOP_WAIT_SECONDS
