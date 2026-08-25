"""Shared path resolution for the one-off analysis scripts: no machine-specific hardcoding.

These scripts were originally written on a GPU workstation with the dataset under the repo's
``data/`` dir and an absolute ``c:/Users/<name>/...`` root baked in. That doesn't port. Resolve
locations here instead:

- ``vf_root()``: the Valley_Farm sample project dir, from ``$TCIP_VF_ROOT``. Never a project
  name or an absolute machine path in source, and never guessed from the workspace's
  active-project marker; refuses when the variable is unset, naming it.
- ``repo_root()``: the repository root, derived from this file's location.

Dates use the canonical ``YYYY-MM-DD`` folder convention (matching ``dataset_layout``), e.g.
``2026-02-11``, not the old ``2-11-26`` format the original scripts hardcoded.
"""
from __future__ import annotations

import os
from pathlib import Path

CATKIN_DATE = "2026-02-11"  # Feb-11 catkin acquisition (the labeled baseline set)
BUSH_DATE = "2026-03-02"    # Mar-02 bush acquisition


def repo_root() -> Path:
    """Repository root (scripts/ lives one level under it)."""
    return Path(__file__).resolve().parents[1]


def vf_root() -> Path:
    """Valley_Farm sample project dir, named by ``$TCIP_VF_ROOT``.

    Refuses when the variable is unset, naming it, rather than guessing at a location or
    reading the workspace's active-project marker.
    """
    env = os.environ.get("TCIP_VF_ROOT")
    if not env:
        raise SystemExit("No Valley_Farm project to analyse: set TCIP_VF_ROOT to its path.")
    return Path(env).expanduser()
