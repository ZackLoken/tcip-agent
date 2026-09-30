"""Operator command sub-package: each module is one ``tcip`` subcommand's implementation, exposing
``main(argv)`` and returning the exit code.
"""

from __future__ import annotations

from pathlib import Path


def bound_project(value: str) -> Path:
    """Bind this process's store and return the project at ``value``, resolved; exits naming the
    path when it holds no readable project record."""
    from tcip_store.binding import bind_default

    from tcip_mcp.project_record import existing_project

    bind_default()
    try:
        return existing_project(value)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
