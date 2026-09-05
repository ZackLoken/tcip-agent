"""Where a registry entry's stored path resolves, and the containment core the checkpoint and
dataset registries share when they decide whether a target sits under their own scope root.

The registry's entries-mapping convention (see ``model_registry.py``) is the value's own carrier:
relative POSIX exactly when the target lives under the registry's scope root, absolute exactly
when external.
The dataset registry (``tools/project_tools.py``) has spelled the same convention since before
this module existed; this module holds the one containment walk and the one absolute-form test
both registries build on, so they cannot silently drift onto two different notions of "under the
root".
"""

from __future__ import annotations

import os
from pathlib import Path, PurePosixPath, PureWindowsPath


def is_at_or_under(candidate: Path, root: Path) -> bool:
    """Whether ``candidate`` is ``root`` itself or sits somewhere under it, by plain path
    arithmetic (no filesystem access): the one predicate every caller deciding "is this path
    inside that tree" shares, rather than each reimplementing ``relative_to``'s own try/except.
    """
    try:
        candidate.relative_to(root)
        return True
    except ValueError:
        return False


def is_external_form(stored: str) -> bool:
    """Whether ``stored`` is an absolute path by either platform's own path grammar.

    ``Path.is_absolute()`` alone is host-grammar-dependent: a Windows drive or UNC spelling
    reads as relative under a bare ``PurePosixPath`` and would be joined under the root on
    POSIX; a POSIX root spelling reads as relative under ``PureWindowsPath`` with no drive.
    Either grammar recognizing ``stored`` as absolute is enough to call it external, so a
    pre-family absolute-under-root spelling can never be misread as designed-external on the
    wrong platform.
    """
    return PurePosixPath(stored).is_absolute() or PureWindowsPath(stored).is_absolute()


def nearest_containing_ancestor(start: Path, root: Path, *, tolerant: bool) -> Path | None:
    """The nearest of ``start`` and its parents that is the same file as ``root``, or ``None``
    when none is.

    ``tolerant=True`` (the dataset speller's own, unchanged behavior) treats an ancestor
    ``os.path.samefile`` cannot compare (an inaccessible share) as simply not a match and tries
    the next one. ``tolerant=False`` (the checkpoint speller) re-raises instead: that speller's
    stricter root gate must never fall through to an external spelling from a comparison it
    could not actually make.
    """
    for ancestor in (start, *start.parents):
        try:
            if os.path.samefile(ancestor, root):
                return ancestor
        except OSError:
            if tolerant:
                continue
            raise
    return None


class CheckpointRegistryRootUnusable(ValueError):
    """The checkpoint speller's root gate: the registry's own scope root does not exist, is not
    a directory, or an ancestor comparison against it could not be made at all. An absolute
    spelling is a positive external claim under the entries-mapping convention, so a
    mis-specified root refuses by name rather than fabricating one."""


def checkpoint_registry_path_for(checkpoint_path: str | Path, root: str | Path) -> str:
    """What a checkpoint registry entry stores for its path: relative POSIX when
    ``checkpoint_path`` resolves under ``root``, absolute otherwise.

    ``checkpoint_path`` must already name an existing file (the caller's own existence check
    runs before this is reached); ``root`` must exist as a directory, or this raises
    :class:`CheckpointRegistryRootUnusable` naming it rather than falling back to an absolute
    spelling that would read as a designed-external claim under a root that was never validly
    named. The walk starts at the checkpoint's own resolved parent directory, since the target
    is a file and only a directory is ever compared against ``root``. Spelling is decided on
    the resolved target, never the name given: a symlinked checkpoint stores its resolved
    location. The produced relative form is asserted non-empty with no ``..`` segment.
    """
    root_path = Path(root)
    if not root_path.is_dir():
        raise CheckpointRegistryRootUnusable(
            f"registry scope root {root!r} is not an existing directory; refusing to spell a "
            "checkpoint path against it rather than fall back to an absolute spelling that "
            "would read as a designed-external claim"
        )
    resolved = Path(checkpoint_path).resolve()
    try:
        ancestor = nearest_containing_ancestor(resolved.parent, root_path, tolerant=False)
    except OSError as exc:
        raise CheckpointRegistryRootUnusable(
            f"could not compare {resolved.parent} against registry scope root {root_path}: {exc}"
        ) from exc
    if ancestor is None:
        return str(resolved)
    relative = resolved.relative_to(ancestor).as_posix()
    assert relative and ".." not in PurePosixPath(relative).parts
    return relative


class RegistryPathEmpty(ValueError):
    """A registry entry names no path to resolve at all."""


class RegistryPathTraversal(ValueError):
    """A registry entry's relative path carries a ``..`` segment: the entries-mapping convention
    is that relative means internal, so this was never a legitimate spelling to resolve."""


def resolved_registry_path(root: str | Path, stored: str) -> Path:
    """The absolute path an entries-mapping registry entry's stored path value resolves to.

    ``root`` is absolutized here, so a relative process root still answers an absolute path. A
    relative ``stored`` value is joined onto the absolutized root by its POSIX parts (the form
    every writer spells); an absolute one (:func:`is_external_form`) is returned unchanged, the
    entry's own positive external claim. Raises :class:`RegistryPathEmpty` for an empty or
    missing value, and :class:`RegistryPathTraversal` for a relative value carrying a ``..``
    segment under either platform's own path grammar (:func:`is_external_form`'s own
    dual-grammar reasoning: a Windows-spelled traversal must refuse the same as a POSIX one,
    never silently resolve outside ``root`` on the platform that does not split it), never
    resolved to wherever it happens to land.
    """
    if not stored:
        raise RegistryPathEmpty("registry entry carries no path to resolve")
    if is_external_form(stored):
        return Path(stored)
    parts = PurePosixPath(stored).parts
    if ".." in parts or ".." in PureWindowsPath(stored).parts:
        raise RegistryPathTraversal(
            f"registry entry path {stored!r} carries a '..' segment, never a legitimate "
            "relative spelling under the entries-mapping convention"
        )
    return Path(root).resolve().joinpath(*parts)


__all__ = [
    "CheckpointRegistryRootUnusable",
    "RegistryPathEmpty",
    "RegistryPathTraversal",
    "checkpoint_registry_path_for",
    "is_at_or_under",
    "is_external_form",
    "nearest_containing_ancestor",
    "resolved_registry_path",
]
