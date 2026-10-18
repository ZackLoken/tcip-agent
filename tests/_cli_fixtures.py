"""The ``tcip`` console commands, each run in a process of its own."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def run_tcip(
    command: str,
    args: list[str],
    *,
    cwd: Path | None = None,
    timeout: float = 120,
) -> subprocess.CompletedProcess:
    """Run ``tcip <command> <args>`` through the console dispatcher in a child process.

    The child starts in ``cwd``. Returns the finished process with its output as text.
    """
    return subprocess.run(
        [sys.executable, "-m", "tcip_web.cli", command, *args],
        cwd=None if cwd is None else str(cwd),
        capture_output=True,
        text=True,
        timeout=timeout,
    )
