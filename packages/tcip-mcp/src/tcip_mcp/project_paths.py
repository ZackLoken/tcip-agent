"""Paths under a project the caller names, and the repository root this package sits in."""

from __future__ import annotations

import time
from pathlib import Path


def project_state_dir(project: str | Path) -> Path:
    """A project's ``<project>/.tcip/state``."""
    return Path(project) / ".tcip" / "state"


def viz_output_path(project: Path, name: str, suffix: str = ".png") -> str:
    """A fresh absolute path under ``<project>/.tcip/artifacts/viz/`` for one rendered artifact,
    the directory created; ``name`` becomes part of the file name."""
    viz_dir = (Path(project) / ".tcip" / "artifacts" / "viz").resolve()
    viz_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return str(viz_dir / f"{stamp}_{time.time_ns() % 1_000_000:06d}_{name}{suffix}")


def repo_root_from_here() -> Path:
    """The repo root inferred from this file's location: the nearest ancestor holding
    ``.mcp.json``, checked across every ancestor before falling back to the nearest one holding
    ``CLAUDE.md``. Stable regardless of cwd.
    """
    here = Path(__file__).resolve()
    parents = list(here.parents)
    for parent in parents:
        if (parent / ".mcp.json").is_file():
            return parent
    for parent in parents:
        if (parent / "CLAUDE.md").is_file():
            return parent
    return Path.cwd()
