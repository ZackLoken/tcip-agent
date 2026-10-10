"""How a record stores a path and where a stored path resolves, the one rule every record under a
project writes and reads paths through.

A stored path is relative POSIX exactly when the target lives under the project whose record holds
it, absolute exactly when external.
"""

from __future__ import annotations

import os
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Callable


def is_external_form(stored: str) -> bool:
    """Whether ``stored`` is an absolute path by either platform's own path grammar; either grammar
    recognizing it as absolute makes it external.
    """
    return PurePosixPath(stored).is_absolute() or PureWindowsPath(stored).is_absolute()


def nearest_containing_ancestor(start: Path, root: Path, *, tolerant: bool) -> Path | None:
    """The nearest of ``start`` and its parents that is the same file as ``root``, or ``None`` when
    none is.

    ``tolerant=True`` treats an ancestor ``os.path.samefile`` cannot compare (an inaccessible
    share) as simply not a match and tries the next one. ``tolerant=False`` re-raises instead.
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


class RelativePathWithoutProjectError(ValueError):
    """A relative path reached an entry point that holds no project to read it against."""


def located(path: str | Path, project: str | Path | None) -> Path:
    """The absolute location a caller's ``path`` names: a relative one is relative to the
    project root ``project``, never to the working directory, and an absolute one is itself.

    With no ``project``, a relative ``path`` refuses (:class:`RelativePathWithoutProjectError`)
    naming the two ways to state it: an absolute path, or a project tool holding its project.
    """
    if project is None and not Path(path).is_absolute():
        raise RelativePathWithoutProjectError(
            f"{str(path)!r} is relative and nothing here holds a project to read it against: "
            "state an absolute path, or call the project tool on its project")
    return (Path(project or "") / path).resolve()


def stored_path(path: str | Path, root: str | Path, *, tolerant: bool = True) -> str:
    """What a record stores for ``path`` (:func:`located` against ``root``): relative POSIX when
    it lies at or under ``root`` (``root`` itself stored ``"."``), absolute otherwise, so a stored
    path stores as itself.

    Containment is decided by filesystem identity over the resolved path's own ancestors
    (:func:`nearest_containing_ancestor`, with ``tolerant`` passed through), so an alias or a case
    variant of ``root`` reads as the filesystem sees it, and a path not created yet is placed by
    its nearest existing ancestor. The relative form never carries a ``..`` segment.
    """
    resolved = located(path, root)
    ancestor = nearest_containing_ancestor(resolved, Path(root), tolerant=tolerant)
    if ancestor is None:
        return str(resolved)
    return resolved.relative_to(ancestor).as_posix()


class CheckpointRegistryRootUnusableError(ValueError):
    """The registry's own scope root does not exist, is not a directory, or an ancestor comparison
    against it could not be made at all.
    """


def checkpoint_registry_path_for(checkpoint_path: str | Path, root: str | Path) -> str:
    """What a checkpoint registry entry stores for its path: relative POSIX when
    ``checkpoint_path`` resolves under ``root``, absolute otherwise.

    ``checkpoint_path`` must already name an existing file; ``root`` must exist as a directory, or
    this raises :class:`CheckpointRegistryRootUnusableError` naming it, as it does when the
    ancestor comparison :func:`stored_path` makes (with ``tolerant=False``) cannot be made.
    """
    root_path = Path(root)
    if not root_path.is_dir():
        raise CheckpointRegistryRootUnusableError(
            f"registry scope root {root!r} is not an existing directory; refusing to spell a "
            "checkpoint path against it rather than fall back to an absolute spelling that "
            "would read as a designed-external claim"
        )
    try:
        return stored_path(checkpoint_path, root_path, tolerant=False)
    except OSError as exc:
        raise CheckpointRegistryRootUnusableError(
            f"could not compare {checkpoint_path} against registry scope root {root_path}: {exc}"
        ) from exc


class RegistryPathEmptyError(ValueError):
    """A registry entry names no path to resolve at all."""


class RegistryPathTraversalError(ValueError):
    """A registry entry's relative path carries a ``..`` segment: the entries-mapping convention
    is that relative means internal, so this was never a legitimate spelling to resolve."""


def resolved_registry_path(root: str | Path, stored: str) -> Path:
    """The absolute path an entries-mapping registry entry's stored path value resolves to.

    ``root`` is absolutized here, so a relative process root still answers an absolute path. A
    relative ``stored`` value is joined onto the absolutized root by its POSIX parts; an absolute
    one (:func:`is_external_form`) is returned unchanged. Raises
    :class:`RegistryPathEmptyError` for an empty or missing value, and
    :class:`RegistryPathTraversalError` for a relative value carrying a
    ``..`` segment under either platform's own path grammar.
    """
    if not stored:
        raise RegistryPathEmptyError("registry entry carries no path to resolve")
    if is_external_form(stored):
        return Path(stored)
    parts = PurePosixPath(stored).parts
    if ".." in parts or ".." in PureWindowsPath(stored).parts:
        raise RegistryPathTraversalError(
            f"registry entry path {stored!r} carries a '..' segment, never a legitimate "
            "relative spelling under the entries-mapping convention"
        )
    return Path(root).resolve().joinpath(*parts)


PathFields = tuple[tuple[str, ...], ...]
"""The fields of one record shape that name a path, each a sequence of steps from the record's
top: a key, ``"[]"`` for every item of a list, ``"*"`` for every value of a mapping, ``"{}"``
for every key of a mapping, or ``"0"`` for the first item of a pair. A field whose value is a
mapping where a path ends, or a path where a key step continues, names no path in that record
and is left as it is."""


def within(prefix: tuple[str, ...], fields: PathFields) -> PathFields:
    """``fields`` of a shape held at ``prefix`` inside another record."""
    return tuple(prefix + field for field in fields)


def _each_path(value: Any, steps: tuple[str, ...], convert: Callable[[str], str]) -> Any:
    if value is None:
        return None
    if not steps:
        return value if isinstance(value, dict) else convert(str(value))
    step, rest = steps[0], steps[1:]
    if step == "[]":
        return [_each_path(item, rest, convert) for item in value]
    if step == "*":
        return {key: _each_path(item, rest, convert) for key, item in value.items()}
    if step == "{}":
        return {convert(key): item for key, item in value.items()}
    if step == "0":
        return [_each_path(value[0], rest, convert), *value[1:]]
    if not isinstance(value, dict) or step not in value:
        return value
    return {**value, step: _each_path(value[step], rest, convert)}


def recorded_paths(record: Any, fields: PathFields, root: str | Path) -> Any:
    """A copy of ``record`` with each path ``fields`` names as :func:`stored_path` stores it
    against ``root``. An absent or ``None`` field stays as it is."""
    for field in fields:
        record = _each_path(record, field, lambda path: stored_path(path, root))
    return record


def runtime_paths(record: Any, fields: PathFields, root: str | Path) -> Any:
    """A copy of ``record``, written by :func:`recorded_paths`, with each path ``fields`` names
    resolved against ``root`` (:func:`resolved_registry_path`)."""
    for field in fields:
        record = _each_path(record, field,
                            lambda stored: str(resolved_registry_path(root, stored)))
    return record


__all__ = [
    "CheckpointRegistryRootUnusableError",
    "PathFields",
    "RegistryPathEmptyError",
    "RegistryPathTraversalError",
    "checkpoint_registry_path_for",
    "RelativePathWithoutProjectError",
    "is_external_form",
    "located",
    "nearest_containing_ancestor",
    "recorded_paths",
    "resolved_registry_path",
    "runtime_paths",
    "stored_path",
    "within",
]
