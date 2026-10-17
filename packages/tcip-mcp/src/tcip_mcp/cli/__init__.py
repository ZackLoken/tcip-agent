"""Operator command sub-package, one module per ``tcip`` subcommand.

Each module exposes ``main(argv)``, which returns the exit code.
"""

from __future__ import annotations

from pathlib import Path


def bound_project(value: str) -> Path:
    """Bind this process's store and return the project at ``value``, resolved.

    Exits naming the path when it holds no readable project record.
    """
    from tcip_mcp.project_record import existing_project
    from tcip_store import bind

    bind()
    try:
        return existing_project(value)[0]
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
