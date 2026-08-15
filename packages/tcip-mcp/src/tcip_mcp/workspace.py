"""Workspace resolver: where TCIP projects live on disk.

A single configured workspace root (``TCIP_WORKSPACE``, default ``~/tcip-projects/``)
holds one directory per project (``<workspace>/<name>/``). The agent creates
projects here (``ingest_images``); the GUI lists and opens them (``/api/projects``).
Both ``tcip-mcp`` tools and the ``tcip-web`` backend resolve through this module so
there is one source of truth for the workspace location, the same spirit as
:mod:`tcip_mcp.dataset_layout` for the in-project layout.

The active-project marker (``<workspace>/.active``) records which workspace project
the GUI should open. The agent sets it after ingesting a project so the breeder flow
("I structured your images, opening ``<crop>_<trait>_valley-farm``") closes the loop.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

from tcip_store import Key, StoreDescriptor, register_store, text_codec
from tcip_store.file_backend import RootedFileLocator

logger = logging.getLogger(__name__)

DEFAULT_WORKSPACE = Path.home() / "tcip-projects"
ACTIVE_MARKER = ".active"

_logged_root: Optional[str] = None


def workspace_root() -> Path:
    """Resolve (and create) the workspace root; log its absolute path once.

    Reads ``TCIP_WORKSPACE`` (``~`` expanded), defaulting to ``~/tcip-projects/``.
    Creates it on first use so callers never race a missing directory.
    """
    global _logged_root
    raw = os.environ.get("TCIP_WORKSPACE", "").strip()
    root = Path(raw).expanduser() if raw else DEFAULT_WORKSPACE
    root.mkdir(parents=True, exist_ok=True)
    root = root.resolve()
    key = str(root)
    if _logged_root != key:
        logger.info("TCIP workspace: %s", root)
        _logged_root = key
    return root


def is_valid_name(name: str) -> bool:
    """True if ``name`` is a single, safe path segment usable as a project folder."""
    seg = (name or "").strip()
    return bool(seg) and not any(c in seg for c in ("/", "\\", ":")) and seg not in (".", "..")


def project_path(name: str) -> Path:
    """Absolute path for a named project under the workspace.

    ``name`` is a single project slug (``{crop}_{trait}_{site}``); path separators
    and traversal are rejected so a name can never escape the workspace.
    """
    if not is_valid_name(name):
        raise ValueError(f"invalid project name: {name!r}")
    return workspace_root() / name.strip()


def active_marker_path() -> Path:
    """Path of the active-project marker file under the workspace."""
    root = workspace_root()
    return Path(root, *_MARKER_DOC.relative_path(str(root), _MARKER_PARTS).parts)


# ── the active-project marker store ──────────────────────────────────────────

_MARKER_DOC = RootedFileLocator()
"""The marker file, at the workspace root itself."""

ACTIVE_PROJECT_STORE = "workspace_active_project"
_MARKER_PARTS = (ACTIVE_MARKER,)
register_store(
    StoreDescriptor(
        name=ACTIVE_PROJECT_STORE,
        kind="record",
        key_fields=("document",),
        codec=text_codec(trailing_newline=True),
        concurrency="last_writer_wins",
        locator=_MARKER_DOC,
    )
)


def active_project_key() -> Key:
    """The workspace's active-project marker.

    ``last_writer_wins``: ``set_active_project`` writes the name it was given and reads
    nothing first, so adopting a project is a whole replacement rather than an edit.
    """
    return Key(ACTIVE_PROJECT_STORE, str(workspace_root()), _MARKER_PARTS)


def read_active_project() -> Optional[str]:
    """Return the active project's name, or ``None`` if the marker is absent/empty."""
    p = active_marker_path()
    if not p.is_file():
        return None
    try:
        val = p.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError):
        # A marker written with the wrong encoding (e.g. UTF-16 from PowerShell) must not
        # 500 the whole front door: treat it as unset.
        return None
    return val or None


def resolve_project_path(given: str) -> str:
    """A given path wins; empty falls back to the active project's root (the live GUI session)."""
    if given:
        return given
    name = read_active_project()
    return str(project_path(name)) if name else given


def set_active_project(name: str) -> Path:
    """Adopt a workspace project: write the marker atomically and repin platform state to it.

    ``name`` must be a valid workspace project slug. The write is atomic (temp file +
    ``os.replace``) so two concurrent writers can't tear the file.

    Adopting also repins this process's ``TCIP_PROJECT_ROOT`` to the project, so the
    ``@audited`` log, the experiment store, and the model registry all resolve under
    ``<project>/.tcip/``, one self-contained ``.tcip`` per project. The repin is an
    explicit action (not a passive marker read), so an in-flight training run keeps writing
    to the project it started under until the agent deliberately adopts a different one.
    """
    if not is_valid_name(name):
        raise ValueError(f"invalid project name: {name!r}")
    from tcip_mcp.project_paths import ENV_VAR
    from tcip_mcp.utils.atomic_io import atomic_write_text

    root = project_path(name)
    p = active_marker_path()
    atomic_write_text(p, name.strip() + "\n")
    os.environ[ENV_VAR] = str(root)
    return p
