"""Workspace resolver: where TCIP projects live on disk.

A single configured workspace root (``TCIP_WORKSPACE``, default ``~/tcip-projects/``)
holds one directory per project (``<workspace>/<name>/``). The agent creates
projects here (``ingest_images``); the GUI lists and opens them (``/api/projects``).
Both ``tcip-mcp`` tools and the ``tcip-web`` backend resolve through this module so
there is one source of truth for the workspace location, the same spirit as
:mod:`tcip_mcp.dataset_layout` for the in-project layout.

The active-project marker (``<workspace>/.active``) records which workspace project
is the startup root of a process that opts in (``tcip_mcp.project_paths.pin_project_root``,
``from_marker=True``) and which one the GUI should open. The agent sets it after ingesting
a project so the breeder flow ("I structured your images, opening
``<crop>_<subject>_<phenotype>``") closes the loop; adopting it also repins the adopting
process's own platform-state root at once, the web backend on the agent's adopt signal, and
the MCP server on its next start inside the platform's own agent terminal.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

from tcip_store import DecodeError, Key, StoreDescriptor, read, register_store, replace, text_codec
from tcip_store.file_backend import RootedFileLocator
from tcip_store.layout_claims import NAME_SEGMENT as _SEGMENT_RE

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


def project_path(name: str, *, create: bool = True) -> Path:
    """Absolute path for a named project under the workspace.

    ``name`` is a single path segment; path separators and traversal are rejected so a
    name can never escape the workspace. This does not check the ``crop_subject_phenotype``
    shape (:func:`format_project_name`/:func:`parse_project_name`): every door that opens,
    lists or adopts an existing project accepts any safely-named one, since a directory the
    platform did not create is opened by the name it already has. Only the doors that
    create a workspace directory (``ingest_images``, ``init_project``, ``import_project``)
    hold a new name to the shape; the first two also take an authored site, recorded on the
    project's own record (:mod:`tcip_mcp.project_record`).

    ``create`` threads through to :func:`workspace_root`; a caller that must not bring the
    workspace root into existence on a bare resolve passes ``create=False``.
    """
    if not is_valid_name(name):
        raise ValueError(f"invalid project name: {name!r}")
    return workspace_root(create=create) / name.strip()


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
        frozen=True,
        cannot_carry_field="a single project-name string",
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


def adoptable_project_root(name: str) -> Path:
    """The path a workspace project's name resolves to, when it is safe to open.

    Raises ``ValueError``, naming which check failed: an unsafe name (path separators, ``..``,
    empty) or a safely-named path whose ``.tcip`` is not a directory (nothing there to open).
    The one predicate every reader that must tell "no marker" apart from "the marker names a
    project that is not adoptable" calls: :func:`set_active_project`, :func:`
    active_project_if_present` (folding the raise to ``None``), and
    ``tcip_mcp.project_paths.pin_project_root`` when binding from the marker.
    """
    if not is_valid_name(name):
        raise ValueError(f"invalid project name: {name!r}")
    root = project_path(name, create=False)
    if not (root / ".tcip").is_dir():
        raise ValueError(f"no such workspace project (missing .tcip): {name!r}")
    return root


def active_project_if_present(*, create: bool = True) -> Optional[tuple[str, Path]]:
    """The active marker's project, only when its ``.tcip`` still exists on disk; else ``None``.

    Collapses "no marker" and "marker names a project that is gone" into the same ``None``:
    a caller that must tell those two apart (the session-start hook's directive,
    ``project_paths.pin_project_root``) reads :func:`read_active_project` and
    :func:`adoptable_project_root` itself. One check shared by every reader that must not
    report a name whose project has vanished: the workspace projects list route's
    ``active``/``active_path`` fields and the session-start hook's directive both read this.

    Resolves through :func:`adoptable_project_root`, so a marker holding a traversal name
    (``../escapee``) names no adoptable project rather than a path outside the workspace.
    """
    name = read_active_project(create=create)
    if not name:
        return None
    try:
        path = adoptable_project_root(name)
    except ValueError:
        return None
    return name, path


def marker_problem(*, create: bool = False) -> Optional[str]:
    """Why :func:`active_project_if_present` answered ``None``, or ``None`` when there was
    simply no marker to have a problem with.

    Call only after :func:`active_project_if_present` has already answered ``None``: it folds
    "no marker" and "the marker names a project that is not adoptable" together, and this is
    the one place that tells them apart, for every reader that needs to
    (``project_paths.pin_project_root``, the workspace projects' divergence report, the web
    backend's own re-read on the agent's adopt signal). A store refusal or a lock timeout
    reading the marker is caught and returned as the problem text, the same as an unadoptable
    name, since either way the process must carry on rather than raise.
    """
    try:
        name = read_active_project(create=create)
    except Exception as exc:  # noqa: BLE001 - a store refusal or lock timeout, returned as text
        return str(exc)
    if not name:
        return None
    try:
        adoptable_project_root(name)
    except ValueError as exc:
        return str(exc)
    return None


def resolve_project_path(given: str) -> str:
    """A given path wins; empty falls back to the active project's root (the live GUI session)."""
    if given:
        return given
    name = read_active_project()
    return str(project_path(name)) if name else given


def set_active_project(name: str) -> Path:
    """Adopt a workspace project: write the marker atomically and repin platform state to it.

    ``name`` must name an existing workspace project (its ``.tcip`` must already be a
    directory, :func:`adoptable_project_root`); adoption opens what is there, it does not
    create anything. Any safely-named directory is adoptable, conforming to
    ``crop_subject_phenotype`` or not: only the doors that create a workspace directory hold
    a new name to that shape. The marker is replaced whole under its own lock, so two
    concurrent writers can't tear the file.

    Adopting also repins this process's platform-state root to the project, so the
    ``@audited`` log, the experiment store, and the model registry all resolve under
    ``<project>/.tcip/``, one self-contained ``.tcip`` per project. The repin is an explicit
    action (not a passive marker read) and reaches only this process: the web backend repins
    on the agent's adopt signal, and the MCP server the next time it starts inside the
    platform's own agent terminal, so a training run in flight keeps writing to the project
    it started under until it is deliberately adopted.
    """
    root = adoptable_project_root(name)
    from tcip_mcp.project_paths import repin_platform_root

    replace(active_project_key(), name.strip())
    repin_platform_root(root)
    return active_marker_path()
