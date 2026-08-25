"""Workspace resolver: where TCIP projects live on disk.

A single configured workspace root (``TCIP_WORKSPACE``, default ``~/tcip-projects/``)
holds one directory per project (``<workspace>/<name>/``). The agent creates
projects here (``ingest_images``); the GUI lists and opens them (``/api/projects``).
Both ``tcip-mcp`` tools and the ``tcip-web`` backend resolve through this module so
there is one source of truth for the workspace location, the same spirit as
:mod:`tcip_mcp.dataset_layout` for the in-project layout.

The active-project marker (``<workspace>/.active``) records which workspace project
the GUI should open. The agent sets it after ingesting a project so the breeder flow
("I structured your images, opening ``<crop>_<subject>_<phenotype>``") closes the loop.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Optional

from tcip_store import DecodeError, Key, StoreDescriptor, read, register_store, replace, text_codec
from tcip_store.file_backend import RootedFileLocator

logger = logging.getLogger(__name__)

DEFAULT_WORKSPACE = Path.home() / "tcip-projects"
ACTIVE_MARKER = ".active"

_logged_root: Optional[str] = None


def workspace_root(*, create: bool = True) -> Path:
    """Resolve the workspace root; log its absolute path once.

    Reads ``TCIP_WORKSPACE`` (``~`` expanded), defaulting to ``~/tcip-projects/``.
    ``create`` (default ``True``) creates it on first use so callers never race a missing
    directory; a caller that only reads (the session-start hook) passes ``create=False`` so
    the read cannot bring a workspace directory into existence on its own.
    """
    global _logged_root
    raw = os.environ.get("TCIP_WORKSPACE", "").strip()
    root = Path(raw).expanduser() if raw else DEFAULT_WORKSPACE
    if create:
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


_SEGMENT = r"[a-z0-9]+(?:-[a-z0-9]+)*"
_SEGMENT_RE = re.compile(rf"^{_SEGMENT}$")


def format_project_name(crop: str, subject: str, phenotype: str) -> str:
    """Join a workspace project's three segments into its directory-name shape.

    Provisional (owner's naming ruling): a project name encodes crop, subject and
    phenotype, three lowercase segments joined by underscores with hyphens allowed inside
    a segment. No segment is checked against a vocabulary: the ruling forbids validating
    the phenotype against ``crops.yml``'s trait names, a subject is governed by the
    project's own class registry rather than by ``crops.yml``, and the platform holds no
    crop-name vocabulary to check the first segment against. Raises, naming the segment,
    when one does not fit that shape.
    """
    for label, seg in (("crop", crop), ("subject", subject), ("phenotype", phenotype)):
        if not _SEGMENT_RE.match(seg):
            raise ValueError(f"{label} segment {seg!r} is not lowercase alphanumeric-hyphen")
    return f"{crop}_{subject}_{phenotype}"


def parse_project_name(name: str) -> tuple[str, str, str]:
    """Split a workspace project's directory name into ``(crop, subject, phenotype)``.

    Provisional: the inverse of :func:`format_project_name`. Raises, naming which segment
    is missing or malformed, when ``name`` does not fit the three-segment shape.
    """
    parts = name.split("_")
    if len(parts) != 3:
        raise ValueError(
            f"{name!r} does not fit crop_subject_phenotype: expected three "
            f"underscore-separated segments, found {len(parts)}"
        )
    crop, subject, phenotype = parts
    for label, seg in (("crop", crop), ("subject", subject), ("phenotype", phenotype)):
        if not _SEGMENT_RE.match(seg):
            raise ValueError(f"{name!r} has an invalid {label} segment: {seg!r}")
    return crop, subject, phenotype


def project_path(name: str) -> Path:
    """Absolute path for a named project under the workspace.

    ``name`` is a single path segment; path separators and traversal are rejected so a
    name can never escape the workspace. This does not check the ``crop_subject_phenotype``
    shape (:func:`format_project_name`/:func:`parse_project_name`): every door that opens,
    lists or adopts an existing project accepts any safely-named one, since a directory the
    platform did not create is opened by the name it already has. Only the doors that
    create a workspace directory (``ingest_images``, ``init_project``, ``import_project``)
    hold a new name to the shape.
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


def active_project_key(*, create: bool = True) -> Key:
    """The workspace's active-project marker.

    ``last_writer_wins``: ``set_active_project`` writes the name it was given and reads
    nothing first, so adopting a project is a whole replacement rather than an edit.
    ``create`` threads through to :func:`workspace_root`.
    """
    return Key(ACTIVE_PROJECT_STORE, str(workspace_root(create=create)), _MARKER_PARTS)


def read_active_project(*, create: bool = True) -> Optional[str]:
    """Return the active project's name, or ``None`` if the marker is absent/empty.

    ``create`` threads through to :func:`workspace_root`; a caller that must not bring a
    workspace directory into existence on a bare read passes ``create=False``.
    """
    try:
        val = read(active_project_key(create=create), default=None)
    except (OSError, DecodeError):
        # A marker written with the wrong encoding (e.g. UTF-16 from PowerShell) must not
        # 500 the whole front door: treat it as unset.
        return None
    return (val.strip() or None) if val is not None else None


def active_project_if_present(*, create: bool = True) -> Optional[tuple[str, Path]]:
    """The active marker's project, only when its ``.tcip`` still exists on disk; else ``None``.

    Collapses "no marker" and "marker names a project that is gone" into the same ``None``:
    a caller that must tell those two apart (the session-start hook's directive) reads
    :func:`read_active_project` itself first. One check shared by every reader that must not
    report a name whose project has vanished: the workspace projects list route's
    ``active``/``active_path`` fields and the session-start hook's directive both read this.
    """
    root = workspace_root(create=create)
    name = read_active_project(create=create)
    if not name:
        return None
    path = root / name
    if not (path / ".tcip").is_dir():
        return None
    return name, path


def resolve_project_path(given: str) -> str:
    """A given path wins; empty falls back to the active project's root (the live GUI session)."""
    if given:
        return given
    name = read_active_project()
    return str(project_path(name)) if name else given


def set_active_project(name: str) -> Path:
    """Adopt a workspace project: write the marker atomically and repin platform state to it.

    ``name`` must name an existing workspace project (its ``.tcip`` must already be a
    directory); adoption opens what is there, it does not create anything. Any safely-named
    directory is adoptable, conforming to ``crop_subject_phenotype`` or not: only the doors
    that create a workspace directory hold a new name to that shape. The marker is replaced
    whole under its own lock, so two concurrent writers can't tear the file.

    Adopting also repins this process's ``TCIP_PROJECT_ROOT`` to the project, so the
    ``@audited`` log, the experiment store, and the model registry all resolve under
    ``<project>/.tcip/``, one self-contained ``.tcip`` per project. The repin is an
    explicit action (not a passive marker read), so an in-flight training run keeps writing
    to the project it started under until the agent deliberately adopts a different one.
    """
    if not is_valid_name(name):
        raise ValueError(f"invalid project name: {name!r}")
    root = project_path(name)
    if not (root / ".tcip").is_dir():
        raise ValueError(f"no such workspace project (missing .tcip): {name!r}")
    from tcip_mcp.project_paths import ENV_VAR

    replace(active_project_key(), name.strip())
    os.environ[ENV_VAR] = str(root)
    return active_marker_path()
