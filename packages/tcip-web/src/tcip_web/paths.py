r"""Path confinement for client-supplied paths.

:func:`assert_path_allowed` resolves a client-supplied path and returns it. The allow-set is always
non-empty: the workspace root, every workspace project's registered dataset roots, and the additive
``TCIP_IMAGE_ROOTS`` list. Containment is decided by filesystem identity (the same device and file
as an allowed root, walking the candidate's resolved ancestors), never by comparing spellings, so a
case variant, a substituted or mapped drive, a junction, or an extended ``\\\\?\\`` prefix neither
admits an outside path nor refuses an inside one. Any error while resolving or comparing refuses.

Two directory names are excluded regardless of containment, by name: ``.imports`` (the import
door's private staging tree) and ``.removed`` (the workspace's holding directory a removed project
moves into); a registered dataset or a ``TCIP_IMAGE_ROOTS`` entry under a directory of either name
is refused too. A workspace project pending removal or pending rename is excluded by identity
(:func:`allowed_roots`).
"""

from __future__ import annotations

import os
from pathlib import Path, PurePosixPath

from tcip_web.trust_boundary import exposed_arrival

__all__ = [
    "allowed_image_roots", "allowed_roots", "assert_path_allowed", "assert_project_root_allowed",
    "exposed_arrival", "safe_join", "within",
]


def allowed_image_roots() -> list[Path]:
    """The additive ``TCIP_IMAGE_ROOTS`` entries (os.pathsep list), resolved; empty when unset.

    These are added on top of the derived allow-set (:func:`allowed_roots`), the recovery path for
    a legitimate root the platform does not manage. Setting it never narrows anything.
    """
    raw = os.environ.get("TCIP_IMAGE_ROOTS", "").strip()
    if not raw:
        return []
    return [Path(r).resolve() for r in raw.split(os.pathsep) if r.strip()]


def _workspace_projects(workspace: Path) -> list[Path]:
    try:
        return sorted(p for p in workspace.iterdir() if (p / ".tcip").is_dir())
    except OSError:
        return []


def allowed_roots() -> tuple[list[Path], list[Path]]:
    """Every root a client-supplied path may resolve under, and every root refused by identity
    despite sitting inside one, derived at call time.

    The roots: the workspace root, then each workspace project's registered dataset roots, then
    ``TCIP_IMAGE_ROOTS``. The platform-state root (``TCIP_STATE_ROOT``) is not a member on its own.

    The excluded roots: the resolved root of every workspace project carrying a pending-removal or
    pending-rename marker (``tcip_mcp.workspace.pending_marker_or_none``). A pending project's own
    root, and its registered datasets, still belong to the roots list above; only opening a path
    under it is refused.

    A project whose dataset registry will not decode raises.
    """
    from tcip_mcp import workspace as _workspace
    from tcip_mcp.tools.project_tools import dataset_entry_path, read_datasets
    from tcip_mcp.workspace import workspace_root
    from tcip_store import DecodeError

    workspace = workspace_root()
    roots: list[Path] = [workspace]
    excluded: list[Path] = []
    for project in _workspace_projects(workspace):
        # A project reached through a junction or symlink resolves outside the workspace and is
        # admitted as itself, not only through the workspace it is listed from.
        resolved_project = project.resolve()
        roots.append(resolved_project)
        if _workspace.pending_marker_or_none(resolved_project) is not None:
            excluded.append(resolved_project)
        try:
            entries = read_datasets(project)
        except DecodeError as exc:
            raise RuntimeError(
                f"the dataset registry of project {project} will not decode, so its registered "
                f"roots cannot be admitted: {exc}"
            ) from exc
        roots.extend(dataset_entry_path(project, e) for e in entries)
    roots.extend(allowed_image_roots())
    seen: set[str] = set()
    unique: list[Path] = []
    for root in roots:
        key = str(root)
        if key not in seen:
            seen.add(key)
            unique.append(root)
    return unique, excluded


def _existing_anchor(resolved: Path) -> Path | None:
    """The candidate itself when it exists, else its nearest existing ancestor.

    Only a missing segment is climbed past; any other error while examining a candidate propagates.
    """
    for candidate in (resolved, *resolved.parents):
        try:
            candidate.stat()
        except FileNotFoundError:
            continue
        except NotADirectoryError:
            continue
        return candidate
    return None


def _contained(anchor: Path, root: Path) -> bool:
    """Whether ``anchor`` is the same directory as ``root`` or sits below it, by identity."""
    for ancestor in (anchor, *anchor.parents):
        if os.path.samefile(ancestor, root):
            return True
    return False


def within(resolved: Path, root: Path) -> bool:
    """Whether an already-resolved path sits at or under ``root``, by filesystem identity.

    A path that does not exist yet is judged by its nearest existing ancestor. Any error while
    examining or comparing answers False: a path or root that cannot be examined admits nothing.
    """
    try:
        anchor = _existing_anchor(resolved)
        return anchor is not None and root.exists() and _contained(anchor, root)
    except OSError:
        return False


def _excluded_by_name(resolved: Path) -> bool:
    """Whether ``resolved``'s ancestry passes through ``.imports`` or ``.removed``."""
    return ".imports" in resolved.parts or ".removed" in resolved.parts


def assert_path_allowed(path: str | Path) -> Path:
    """Resolve ``path`` and ensure it sits under an allowed root; return the resolved path.

    A path that does not exist yet (a file about to be written) is judged by its nearest existing
    ancestor. An ``.imports`` or ``.removed`` staging tree is never admitted, by name
    (:func:`_excluded_by_name`); a path under a workspace project pending removal or pending rename
    is refused by identity (:func:`allowed_roots`' excluded roots), ahead of the ordinary
    containment check. Raises :class:`ValueError` naming the roots checked on refusal, and on any
    resolution or comparison error.
    """
    try:
        resolved = Path(path).resolve()
        anchor = _existing_anchor(resolved)
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"path {path!s} cannot be examined: {exc}") from exc
    roots, excluded = allowed_roots()
    if anchor is not None and not _excluded_by_name(resolved):
        for root in excluded:
            try:
                if root.exists() and _contained(anchor, root):
                    raise ValueError(
                        f"path {resolved} is under {root}, a workspace project pending "
                        "removal or rename"
                    )
            except OSError as exc:
                raise ValueError(
                    f"path {resolved} could not be compared against {root}: {exc}"
                ) from exc
        for root in roots:
            try:
                if root.exists() and _contained(anchor, root):
                    return resolved
            except OSError as exc:
                raise ValueError(
                    f"path {resolved} could not be compared against {root}: {exc}"
                ) from exc
    raise ValueError(
        f"path {resolved} is outside the allowed roots "
        f"({', '.join(str(r) for r in roots)}); register the dataset to a workspace project "
        "or add its root to TCIP_IMAGE_ROOTS"
    )


def assert_project_root_allowed(project_root: str | Path) -> Path:
    """Confine a client-supplied ``project_root`` through :func:`assert_path_allowed`; raises
    :class:`ValueError` on refusal.
    """
    return assert_path_allowed(project_root)


def safe_join(root: Path | str, *parts: str) -> Path:
    """Join ``parts`` under ``root``, rejecting traversal and absolute paths.

    Raises :class:`ValueError` if the resolved path escapes ``root``.
    """
    base = Path(root).resolve()
    # Accept forward slashes on Windows by normalizing via PurePosixPath first
    rel_parts: list[str] = []
    for part in parts:
        if not part:
            continue
        posix = PurePosixPath(part.replace("\\", "/"))
        if posix.is_absolute():
            raise ValueError(f"absolute path not allowed: {part!r}")
        for seg in posix.parts:
            if seg in ("..",):
                raise ValueError(f"path traversal not allowed: {part!r}")
            rel_parts.append(seg)
    candidate = base.joinpath(*rel_parts).resolve()
    try:
        candidate.relative_to(base)
    except ValueError as exc:
        raise ValueError(f"resolved path {candidate} is outside {base}") from exc
    return candidate
