"""The fake agent program the agent terminal's tests launch, an executable that runs it, and a PTY.

The PTY (:class:`StubPty`) stands in for a spawned process whose termination survives or stops
as a test says, installed as every spawn's by :func:`wire_stub_spawn`.
"""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path
from typing import TYPE_CHECKING

from tcip_web import terminal as pty_host

if TYPE_CHECKING:
    import pytest

FAKE = Path(__file__).parent / "fake_terminal_app.py"
"""The fake agent program (``tests/fake_terminal_app.py``)."""

ABSENT = "definitely-not-a-real-cli-xyz"
"""An executable name no machine has on its PATH."""


class StubPty(pty_host._WinPty):
    """A process that, on ``terminate``, survives or stops as ``survives`` says, counting calls.

    Its ``write`` takes the whole of what it is given and appends it, decoded, to ``writes``.
    """

    def __init__(self, *, survives: bool) -> None:
        self._alive = True
        self._survives = survives
        self.terminate_calls = 0
        self.writes: list[str] = []
        self._written = threading.Condition()

    def isalive(self) -> bool:
        return self._alive

    def write(self, data: bytes) -> int:
        with self._written:
            self.writes.append(data.decode("utf-8"))
            self._written.notify_all()
        return len(data)

    def written(self, count: int) -> list[str]:
        """Return ``writes`` once it holds ``count`` entries, waiting up to thirty seconds."""
        with self._written:
            assert self._written.wait_for(
                lambda: len(self.writes) >= count, timeout=30
            ), self.writes
            return list(self.writes)

    def terminate(self) -> None:
        self.terminate_calls += 1
        if not self._survives:
            self._alive = False

    def exit(self) -> None:
        """End the process on its own, as one that survived its termination may later."""
        self._alive = False


def wire_stub_spawn(monkeypatch: pytest.MonkeyPatch, stub: StubPty) -> None:
    """Make ``stub`` the process every spawn starts, its output read by no reader."""
    monkeypatch.setattr(pty_host, "spawn_pty", lambda *a, **kw: stub)
    monkeypatch.setattr(pty_host, "start_reader", lambda *a, **kw: None)


def fake_executable(directory: Path) -> Path:
    """Write an executable that runs the fake program with the arguments it is given.

    It is ``fake_agent`` (``fake_agent.cmd`` on Windows) in the newly made ``directory``; the
    return is its path.
    """
    directory.mkdir()
    if os.name == "nt":
        wrapper = directory / "fake_agent.cmd"
        wrapper.write_text(
            f'@"{sys.executable}" -u "{FAKE}" %*\r\n', encoding="utf-8"
        )
    else:
        wrapper = directory / "fake_agent"
        wrapper.write_text(
            f'#!/bin/sh\nexec "{sys.executable}" -u "{FAKE}" "$@"\n',
            encoding="utf-8",
        )
        wrapper.chmod(0o755)
    return wrapper
