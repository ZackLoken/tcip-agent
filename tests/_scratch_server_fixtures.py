"""Helpers for tests that start the web backend as its own process over a scratch harness root.

They run ``tools/serve_capture_app.py``'s command line as its users do, find the backends still
running over a harness root, and ask a running backend which workspace it serves.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

import psutil
import pytest

from tcip_mcp.web_client import LOOPBACK_HOST
from tools import serve_capture_app

ON_WINDOWS = pytest.mark.skipif(
    sys.platform != "win32",
    reason="the serving tool ends its server with taskkill /T, which only Windows has",
)


def serve(
    *args: str, cwd: Path | None = None
) -> subprocess.CompletedProcess[str]:
    """Run ``python tools/serve_capture_app.py`` with ``args`` and return what it did."""
    script = serve_capture_app.REPO_ROOT / "tools" / "serve_capture_app.py"
    return subprocess.run(
        [sys.executable, str(script), *args],
        capture_output=True,
        text=True,
        cwd=cwd,
        check=False,
        timeout=300,
    )


def started(*args: str, cwd: Path | None = None) -> dict:
    """Run a ``start`` that must succeed and return the ``server_info.json`` it wrote."""
    done = serve("start", *args, cwd=cwd)
    assert done.returncode == 0, done.stderr
    return dict(json.loads(done.stdout))


def stopped(root: str, cwd: Path | None = None) -> None:
    """Run a ``stop`` of ``root`` that must succeed."""
    done = serve("stop", root, cwd=cwd)
    assert done.returncode == 0, done.stderr


def served_processes(root: Path) -> list[psutil.Process]:
    """Return the running web backends (``-m tcip_web``) whose working directory is ``root``."""
    found = []
    for process in psutil.process_iter():
        try:
            if "tcip_web" in process.cmdline() and os.path.samefile(
                process.cwd(), root
            ):
                found.append(process)
        except (psutil.Error, OSError):
            continue
    return found


def end_served_processes(root: Path) -> None:
    """Kill every backend :func:`served_processes` finds over ``root``, so a failure leaves none."""
    for process in served_processes(root):
        try:
            process.kill()
        except psutil.Error:
            continue


def answered_workspace(port: int) -> str:
    """Return the workspace the backend answering on ``port`` names on its projects route."""
    url = f"http://{LOOPBACK_HOST}:{port}/api/projects"
    with urllib.request.urlopen(url, timeout=10) as resp:
        return str(json.load(resp)["workspace"])
