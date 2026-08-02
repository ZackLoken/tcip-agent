"""Shared path resolution for the one-off analysis scripts: no machine-specific hardcoding.

These scripts were originally written on a GPU workstation with the dataset under the repo's
``data/`` dir and an absolute ``c:/Users/<name>/...`` root baked in. That doesn't port. Resolve
locations here instead:

- ``vf_root()``: the Valley_Farm catkin project dir, from ``$TCIP_VF_ROOT`` (override), else the
  standard sample-project location under the user's home. Never an absolute machine path in source.
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
    """Valley_Farm catkin project dir. ``$TCIP_VF_ROOT`` wins; else the standard sample location."""
    env = os.environ.get("TCIP_VF_ROOT")
    if env:
        return Path(env).expanduser()
    default = Path.home() / "tcip-projects" / "hazelnut_catkin-05-50-95-per-date_valley-farm"
    if not default.exists():
        raise SystemExit(
            "Valley_Farm project not found. Set TCIP_VF_ROOT to its path, e.g.\n"
            "  export TCIP_VF_ROOT=/path/to/hazelnut_catkin-05-50-95-per-date_valley-farm"
        )
    return default
