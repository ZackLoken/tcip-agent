"""Paths under a project the caller names."""

from __future__ import annotations

import time
from pathlib import Path


def project_state_dir(project: str | Path) -> Path:
    """Return a project's ``<project>/.tcip/state``."""
    return Path(project) / ".tcip" / "state"


def viz_output_path(project: Path, name: str, suffix: str = ".png") -> str:
    """Return a fresh absolute path under ``<project>/.tcip/artifacts/viz/``.

    The directory is created; ``name`` becomes part of the file name.
    """
    viz_dir = (Path(project) / ".tcip" / "artifacts" / "viz").resolve()
    viz_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return str(
        viz_dir / f"{stamp}_{time.time_ns() % 1_000_000:06d}_{name}{suffix}"
    )
