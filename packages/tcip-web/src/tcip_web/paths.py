"""Path confinement for client-supplied paths: :func:`assert_path_allowed` and the route forms
built on it."""

from __future__ import annotations

import os
from pathlib import Path, PurePosixPath
from typing import Any

from tcip_mcp.registry_paths import located, nearest_containing_ancestor

__all__ = [
    "allowed_optional", "allowed_path", "allowed_roots", "assert_path_allowed",
    "image_roots_from_environment", "resolved_path", "safe_join", "within",
]


def image_roots_from_environment() -> tuple[Path, ...]:
    """The additive ``TCIP_IMAGE_ROOTS`` entries (os.pathsep list), resolved; empty when unset.
    Read once, at the backend's start.

    These are added on top of the derived allow-set (:func:`allowed_roots`), the recovery path for
    a legitimate root the platform does not manage. Setting it never narrows anything.
    """
    raw = os.environ.get("TCIP_IMAGE_ROOTS", "").strip()
    return tuple(Path(r).resolve() for r in raw.split(os.pathsep) if r.strip())


def allowed_roots() -> list[Path]:
    """Every root a client-supplied path may resolve under: the backend's workspace root, each
    workspace project and its registered dataset roots, then the backend's additive image roots.

    A project whose dataset registry will not decode raises.
    """
    from tcip_mcp.tools.project_tools import dataset_entry_path, read_datasets
    from tcip_mcp.workspace import project_dirs
    from tcip_store import DecodeError

    from tcip_web.state import store

    roots: list[Path] = [store.workspace]
    for project in project_dirs(store.workspace):
        # A project reached through a junction or symlink resolves outside the workspace and is
        # admitted as itself, not only through the workspace it is listed from.
        roots.append(project.resolve())
        try:
            entries = read_datasets(project)
        except DecodeError as exc:
            raise RuntimeError(
                f"the dataset registry of project {project} will not decode, so its registered "
                f"roots cannot be admitted: {exc}"
            ) from exc
        roots.extend(dataset_entry_path(project, e) for e in entries)
    roots.extend(store.image_roots)
    seen: set[str] = set()
    unique: list[Path] = []
    for root in roots:
        key = str(root)
        if key not in seen:
            seen.add(key)
            unique.append(root)
    return unique


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
    """Whether ``anchor`` is the same directory as ``root`` or sits below it, by identity; an
    ancestor that cannot be compared raises."""
    return nearest_containing_ancestor(anchor, root, tolerant=False) is not None


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


def _in_open_project(path: str | Path) -> Path:
    """``path`` located against the open project
    (:func:`~tcip_mcp.registry_paths.located`), against none when no project is open."""
    from tcip_web.state import NoProjectOpenError, store

    try:
        root = store.held().root
    except NoProjectOpenError:
        root = None
    return located(path, root)


def assert_path_allowed(path: str | Path) -> Path:
    """Locate ``path`` (:func:`_in_open_project`), ensure it sits under an allowed root, return it.

    A path that does not exist yet (a file about to be written) is judged by its nearest existing
    ancestor. An ``.imports`` or ``.removed`` staging tree is never admitted, by name
    (:func:`_excluded_by_name`). Raises :class:`ValueError` naming the roots checked on refusal,
    and on any resolution or comparison error.
    """
    try:
        resolved = _in_open_project(path)
        anchor = _existing_anchor(resolved)
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"path {path!s} cannot be examined: {exc}") from exc
    roots = allowed_roots()
    if anchor is not None and not _excluded_by_name(resolved):
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
        "or start the backend with its root in TCIP_IMAGE_ROOTS"
    )


def resolved_path(path: str) -> Path:
    """``path`` located for a route (:func:`_in_open_project`), unconfined; a path that cannot be
    located answers 400."""
    from fastapi import HTTPException

    try:
        return _in_open_project(path)
    except (OSError, RuntimeError, ValueError) as exc:
        raise HTTPException(400, f"cannot resolve {path}: {exc}") from exc


def allowed_path(path: str | Path) -> Path:
    """:func:`assert_path_allowed` for a route: its refusal answered as HTTP 403 naming it."""
    from fastapi import HTTPException

    try:
        return assert_path_allowed(path)
    except ValueError as exc:
        raise HTTPException(403, str(exc)) from exc


def allowed_file(path: str) -> Path:
    """:func:`allowed_path` of a client-supplied file path; no file there answers 404."""
    from fastapi import HTTPException

    p = allowed_path(path)
    if not p.is_file():
        raise HTTPException(404, f"not a file: {path}")
    return p


def _resolving(read: Any, path: str) -> tuple[Path, Any]:
    """The client-supplied image ``path`` (:func:`allowed_file`) and what ``read``, a reader of
    the logical image a path names, answers for it, the resolver's refusals translated: a path
    naming no logical image (a band of a grouped capture) answers 404 naming why, a grouped
    capture missing a band 409, and an ambiguous stem 400."""
    from fastapi import HTTPException

    from tcip_mcp.pipelines.data.band_groups import BandGroupIncompleteError
    from tcip_mcp.pipelines.image_utils import AmbiguousImageStemError

    p = allowed_file(path)
    try:
        return p, read(p)
    except BandGroupIncompleteError as exc:
        raise HTTPException(409, str(exc)) from exc
    except AmbiguousImageStemError as exc:
        raise HTTPException(400, str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc


def resolved_image(path: str) -> tuple[Path, Any]:
    """The client-supplied image ``path`` and the logical image it names
    (:func:`~tcip_mcp.pipelines.image_utils.resolve_image_path`, :func:`_resolving`)."""
    from tcip_mcp.pipelines.image_utils import resolve_image_path

    return _resolving(resolve_image_path, path)


def allowed_image(path: str) -> tuple[Path, int, int]:
    """The client-supplied image ``path`` and the dimensions of the logical image it names
    (:func:`~tcip_mcp.pipelines.image_utils.image_path_dimensions`, :func:`_resolving`)."""
    from tcip_mcp.pipelines.image_utils import image_path_dimensions

    p, (width, height) = _resolving(image_path_dimensions, path)
    return p, width, height


def allowed_optional(path: str | None) -> str | None:
    """:func:`allowed_path`'s resolved spelling of an optional client-supplied path, or ``None``
    when none is given."""
    return str(allowed_path(path)) if path else None


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
    if not candidate.is_relative_to(base):
        raise ValueError(f"resolved path {candidate} is outside {base}")
    return candidate
