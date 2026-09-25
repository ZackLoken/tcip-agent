"""Workspace resolver: where TCIP projects live on disk.

A single configured workspace root (``TCIP_WORKSPACE``, default ``~/tcip-projects/``) holds one
directory per project (``<workspace>/<name>/``).

The active-project marker (``<workspace>/.active``) records which workspace project is the startup
root of a process that opts in and which one the GUI should open. Adopting a project also repins
the adopting process's own platform-state root at once.

A second, per-project marker (``<project>/.tcip/pending_removal.json``) records that a project's
own directory has been archived and is waiting to move into the workspace's holding directory: a
project carrying one is not adoptable (:func:`adoptable_project_root`), still names itself
(:func:`workspace_project_root`, :func:`workspace_project_name`), and moves at the next backend
start or through ``tcip complete-removals`` (:mod:`tcip_mcp.project_removal`).

A third, sibling marker (``<project>/.tcip/pending_rename.json``) records that a project's
directory is waiting to be renamed onto a new name within the same workspace: a project carrying
one is not adoptable either, still names itself the same way, and renames at the next backend start
or through ``tcip complete-renames`` (:mod:`tcip_mcp.project_rename`).
:func:`pending_marker_or_none` reads both markers, naming which kind it found.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import NamedTuple, Optional

from tcip_store import (
    RECORD_JSON,
    DecodeError,
    Key,
    SchemaVersionRefused,
    StoreDescriptor,
    StoreError,
    read,
    register_store,
    replace,
    text_codec,
)
from tcip_store.file_backend import RootedFileLocator
from tcip_store.layout_claims import NAME_SEGMENT as _SEGMENT_RE

logger = logging.getLogger(__name__)

DEFAULT_WORKSPACE = Path.home() / "tcip-projects"
ACTIVE_MARKER = ".active"

_logged_root: Optional[str] = None


def configured_workspace() -> Optional[Path]:
    """The ``~``-expanded ``TCIP_WORKSPACE`` path, or ``None`` when it is unset or blank.

    Neither creates the directory nor resolves it against the filesystem; :func:`workspace_root` is
    the resolving, directory-creating form.
    """
    raw = os.environ.get("TCIP_WORKSPACE", "").strip()
    return Path(raw).expanduser() if raw else None


def workspace_root(*, create: bool = True) -> Path:
    """Resolve the workspace root; log its absolute path once.

    Reads ``TCIP_WORKSPACE`` (``~`` expanded, via :func:`configured_workspace`), defaulting to
    ``~/tcip-projects/``. ``create`` (default ``True``) creates it on first use; ``create=False``
    never brings a workspace directory into existence.
    """
    global _logged_root
    configured = configured_workspace()
    root = configured if configured is not None else DEFAULT_WORKSPACE
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

    Tentative: a project name encodes crop, subject and phenotype, three lowercase segments joined
        by underscores with hyphens allowed inside a segment. No segment is checked against a
        vocabulary. Raises, naming the segment, when one does not fit that shape.
    """
    for label, seg in (("crop", crop), ("subject", subject), ("phenotype", phenotype)):
        if not _SEGMENT_RE.match(seg):
            raise ValueError(f"{label} segment {seg!r} is not lowercase alphanumeric-hyphen")
    return f"{crop}_{subject}_{phenotype}"


def parse_project_name(name: str) -> tuple[str, str, str]:
    """Split a workspace project's directory name into ``(crop, subject, phenotype)``.

    Tentative: the inverse of :func:`format_project_name`. Raises, naming which segment
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

    ``name`` is a single path segment; path separators and traversal are rejected so a name can
    never escape the workspace. This does not check the ``crop_subject_phenotype`` shape
    (:func:`format_project_name`/:func:`parse_project_name`).

    ``create`` threads through to :func:`workspace_root`.
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

    ``last_writer_wins``: adopting a project is a whole replacement rather than an edit. A release
        deletes it with ``expect=`` the version it read, so a concurrent adoption landing between
        that read and the delete raises ``VersionConflict``. ``create`` threads through to
        :func:`workspace_root`.
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


# ── the pending-removal marker store ─────────────────────────────────────────

_PENDING_REMOVAL_DOC = RootedFileLocator(prefix=(".tcip",), suffix=".json")
"""A project's own top-level ``.tcip`` document, the shape ``dataset_registry`` uses."""

PENDING_REMOVAL_STORE = "pending_removal"
_PENDING_REMOVAL_PARTS = ("pending_removal",)
register_store(
    StoreDescriptor(
        name=PENDING_REMOVAL_STORE,
        kind="record",
        key_fields=("document",),
        frozen=True,
        codec=RECORD_JSON,
        concurrency="cas",
        locator=_PENDING_REMOVAL_DOC,
    )
)


def pending_removal_key(project_root: str | Path) -> Key:
    """The workspace project's own pending-removal marker, one document per project, the same
    per-project, ``.tcip``-rooted shape as :func:`pending_rename_key`.

    ``cas``: written once with ``expect=Version.ABSENT`` (a second request over an existing marker
        is a conflict, not an overwrite) and deleted at the version the completing walk read it at.
    """
    return Key(PENDING_REMOVAL_STORE, str(Path(project_root).absolute()), _PENDING_REMOVAL_PARTS)


def pending_removal_record(project_root: str | Path) -> Optional[dict]:
    """The project's pending-removal marker, or ``None`` when it carries none.

    Reads through the seam and lets the store's own refusals (``StoreError``, ``DecodeError``,
    ``SchemaVersionRefused``) propagate rather than folding them to ``None``.
    """
    return read(pending_removal_key(project_root), default=None)


def pending_removal_or_none(project_root: str | Path) -> Optional[dict]:
    """:func:`pending_removal_record`, with a marker the store refuses to read folded to ``None``."""
    try:
        return pending_removal_record(project_root)
    except (StoreError, DecodeError, SchemaVersionRefused):
        return None


class ProjectPendingRemoval(ValueError):
    """A workspace project's own ``pending_removal`` marker names it, naming when the request was
    made and that the move to the workspace's holding directory completes at the next backend
    start, or through ``tcip complete-removals``. A ``ValueError`` subclass.
    """


# ── the pending-rename marker store ──────────────────────────────────────────

_PENDING_RENAME_DOC = RootedFileLocator(prefix=(".tcip",), suffix=".json")
"""A project's own top-level ``.tcip`` document, the shape :data:`_PENDING_REMOVAL_DOC` uses."""

PENDING_RENAME_STORE = "pending_rename"
_PENDING_RENAME_PARTS = ("pending_rename",)
register_store(
    StoreDescriptor(
        name=PENDING_RENAME_STORE,
        kind="record",
        key_fields=("document",),
        frozen=True,
        codec=RECORD_JSON,
        concurrency="cas",
        locator=_PENDING_RENAME_DOC,
    )
)


def pending_rename_key(project_root: str | Path) -> Key:
    """The workspace project's own pending-rename marker, one document per project.

    ``cas``: written once with ``expect=Version.ABSENT`` (a second request over an existing marker
        is a conflict, not an overwrite) and deleted at the version the completing walk or a
        withdrawal read it at.
    """
    return Key(PENDING_RENAME_STORE, str(Path(project_root).absolute()), _PENDING_RENAME_PARTS)


def pending_rename_record(project_root: str | Path) -> Optional[dict]:
    """The project's pending-rename marker, or ``None`` when it carries none.

    Reads through the seam and lets the store's own refusals (``StoreError``, ``DecodeError``,
    ``SchemaVersionRefused``) propagate.
    """
    return read(pending_rename_key(project_root), default=None)


def pending_rename_or_none(project_root: str | Path) -> Optional[dict]:
    """:func:`pending_rename_record`, with a marker the store refuses to read folded to ``None``."""
    try:
        return pending_rename_record(project_root)
    except (StoreError, DecodeError, SchemaVersionRefused):
        return None


class ProjectPendingRename(ValueError):
    """A workspace project's own ``pending_rename`` marker names it, naming the new name and that
    the rename completes at the next backend start, or through ``tcip complete-renames``. A
    ``ValueError`` subclass.
    """


class PendingMarker(NamedTuple):
    """Which of the two per-project markers a project carries, and its record."""
    kind: str
    """``"removal"`` or ``"rename"``."""
    record: dict


def pending_marker_or_none(project_root: str | Path) -> Optional[PendingMarker]:
    """Whichever of the two per-project markers ``project_root`` carries, or ``None`` when it
    carries neither: the removal marker is read first, then the rename marker, through
    :func:`pending_removal_or_none` and :func:`pending_rename_or_none`.
    """
    removal = pending_removal_or_none(project_root)
    if removal is not None:
        return PendingMarker("removal", removal)
    rename = pending_rename_or_none(project_root)
    if rename is not None:
        return PendingMarker("rename", rename)
    return None


def workspace_project_root(name: str) -> Path:
    """The path a workspace project's name resolves to, when its ``.tcip`` exists on disk.

    Raises ``ValueError``, naming which check failed: an unsafe name (path separators, ``..``,
    empty) or a safely-named path whose ``.tcip`` is not a directory. Says nothing about whether
    the project is pending removal or pending rename.
    """
    if not is_valid_name(name):
        raise ValueError(f"invalid project name: {name!r}")
    root = project_path(name, create=False)
    if not (root / ".tcip").is_dir():
        raise ValueError(f"no such workspace project (missing .tcip): {name!r}")
    return root


def adoptable_project_root(name: str) -> Path:
    """The path a workspace project's name resolves to, when it is safe to open.

    Raises ``ValueError``, naming which check failed: everything :func:`workspace_project_root`
    already checks, plus a pending project (:class:`ProjectPendingRemoval` or
    :class:`ProjectPendingRename`, naming when the request was made and what completes at the next
    backend start). A marker either store refuses to read (``StoreError``, ``DecodeError``,
    ``SchemaVersionRefused``) is not treated as one: the project stays adoptable exactly as it
    would with no marker at all.
    """
    root = workspace_project_root(name)
    pending = pending_marker_or_none(root)
    if pending is None:
        return root
    if pending.kind == "removal":
        raise ProjectPendingRemoval(
            f"{name!r} is pending removal (requested {pending.record['requested_at']}); it "
            "moves to the workspace's holding directory at the next backend start, or through "
            "tcip complete-removals"
        )
    raise ProjectPendingRename(
        f"{name!r} is pending rename to {pending.record['new_name']!r} (requested "
        f"{pending.record['requested_at']}); it renames at the next backend start, or through "
        "tcip complete-renames"
    )


def active_project_if_present(*, create: bool = True) -> Optional[tuple[str, Path]]:
    """The active marker's project, only when its ``.tcip`` still exists on disk; else ``None``.

    Collapses "no marker" and "marker names a project that is gone" into the same ``None``;
    :func:`read_active_project` and :func:`adoptable_project_root` tell them apart.

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
    """Why :func:`active_project_if_present` answered ``None``, or ``None`` when there was simply
    no marker to have a problem with.

    Call only after :func:`active_project_if_present` has already answered ``None``. A store
    refusal or a lock timeout reading the marker is caught and returned as the problem text, the
    same as an unadoptable name.
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
    """A given path wins; empty falls back to the active project's root (the live GUI session).
    Reads the marker's name with no pending-marker check.
    """
    if given:
        return given
    name = read_active_project()
    return str(project_path(name)) if name else given


def workspace_project_name(root: Path) -> Optional[str]:
    """``root``'s workspace project name, when it is exactly one project's own directory; else
    ``None``.

    Compared by filesystem identity (``os.path.samefile`` against the workspace root, not string
    equality), so a case variant or a substituted drive of the same directory still resolves. A
    root nested inside a project, the workspace root itself, or a directory that is not one the
    workspace holds (:func:`workspace_project_root`) names no project. A project pending removal or
    pending rename still names itself here.
    """
    ws = workspace_root(create=False)
    resolved = Path(root)
    if not resolved.is_dir():
        return None
    try:
        if not os.path.samefile(resolved.parent, ws):
            return None
    except OSError:
        return None
    name = resolved.name
    if not is_valid_name(name):
        return None
    try:
        workspace_project_root(name)
    except ValueError:
        return None
    return name


def activate_project(name: str) -> Path:
    """Adopt a workspace project: write the marker atomically and repin platform state to it.

    ``name`` must name an existing workspace project (its ``.tcip`` must already be a directory,
    :func:`adoptable_project_root`); adoption opens what is there, it does not create anything. Any
    safely-named directory is adoptable, conforming to ``crop_subject_phenotype`` or not. The
    marker is replaced whole under its own lock, so two concurrent writers can't tear the file.

    Adopting also repins this process's platform-state root to the project, so the ``@audited``
    log, the experiment store, and the model registry all resolve under ``<project>/.tcip/``. The
    repin reaches only this process.
    """
    root = adoptable_project_root(name)
    from tcip_mcp.project_paths import repin_platform_root

    replace(active_project_key(), name.strip())
    repin_platform_root(root)
    return active_marker_path()
