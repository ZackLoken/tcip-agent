"""The fake agent program the agent terminal's tests launch, and an executable that runs it."""

from __future__ import annotations

import os
import sys
from pathlib import Path

FAKE = Path(__file__).parent / "fake_terminal_app.py"
"""The fake agent program (``tests/fake_terminal_app.py``)."""


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
