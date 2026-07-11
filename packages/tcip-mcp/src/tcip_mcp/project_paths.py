"""Stable resolution of the platform state root, independent of a process's cwd.

Durable *platform* state — the ``@audited`` log and the experiment store — historically
anchored to ``Path(".tcip/...")``, i.e. the process's *current working directory*. Two
processes launched from different directories (an agent that ``cd``-ed, a backend started
from a subdir) then silently write to different ``.tcip/`` trees and fragment the record
(the audit found a stray ``frontend/.tcip/`` demonstrating exactly this).

Resolution order:
  1. ``$TCIP_PROJECT_ROOT`` if set — the MCP server and the web backend set it to the repo
     root at startup (``pin_project_root``), so every process agrees regardless of cwd.
  2. otherwise the current working directory — the historical default, so nothing changes
     for tests or an un-pinned run.

Project-scoped state (``<project>/.tcip/...`` — model registry, reports, retrospectives,
gui.json) already takes an explicit ``project_path`` and is unaffected; this only concerns
the platform-level defaults.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

ENV_VAR = "TCIP_PROJECT_ROOT"


def project_root() -> Path:
    """The platform state root: ``$TCIP_PROJECT_ROOT`` if set, else the current directory."""
    override = os.environ.get(ENV_VAR)
    return Path(override) if override else Path.cwd()


def resolve_state(path: Path) -> Path:
    """Resolve a platform-state path against the pinned root, at USE time.

    - An already-absolute ``path`` is returned unchanged (e.g. a test that rebinds
      ``AUDIT_PATH``/``EXPERIMENTS_DIR`` to a tmp dir).
    - A relative ``path`` is prefixed with ``$TCIP_PROJECT_ROOT`` when pinned, so a process
      launched from a subdir still writes to the one platform ``.tcip/``.
    - When unpinned, a relative ``path`` is returned as-is → resolved against the current
      directory at use, preserving the historical default (and per-test cwd isolation).
    """
    if path.is_absolute():
        return path
    override = os.environ.get(ENV_VAR)
    return Path(override) / path if override else path


def repo_root_from_here() -> Path:
    """The repo root inferred from this file's location — the nearest ancestor holding
    ``.mcp.json`` or ``CLAUDE.md``. Stable regardless of cwd; used to *pin* the env var."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / ".mcp.json").is_file() or (parent / "CLAUDE.md").is_file():
            return parent
    return Path.cwd()


def pin_project_root() -> Path:
    """Pin ``$TCIP_PROJECT_ROOT`` to the repo root (if not already set) and log it once.

    Call at the top of a long-running entry point (MCP server, web backend) *before* the
    audit/experiment modules resolve their paths, so all of them — and any child process —
    agree on one ``.tcip/`` even if the launch cwd differs.
    """
    root = os.environ.setdefault(ENV_VAR, str(repo_root_from_here()))
    logger.info("TCIP platform state root: %s", root)
    return Path(root)
