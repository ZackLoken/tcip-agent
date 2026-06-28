"""Path resolution helpers with traversal protection.

All GUI routes that read user-supplied paths go through :func:`safe_join`
so a malicious/broken client cannot reach files outside the configured
project root.
"""

from __future__ import annotations

import os
from pathlib import Path, PurePosixPath


def allowed_image_roots() -> list[Path]:
    """Allowed roots for absolute file reads, from ``TCIP_IMAGE_ROOTS`` (os.pathsep list).

    Empty (the default) means unrestricted — appropriate for a local single-user GUI.
    Set it to lock an exposed/networked deployment down to specific dataset directories.
    """
    raw = os.environ.get("TCIP_IMAGE_ROOTS", "").strip()
    if not raw:
        return []
    return [Path(r).resolve() for r in raw.split(os.pathsep) if r.strip()]


def assert_path_allowed(path: str | Path) -> Path:
    """Resolve ``path`` and ensure it sits under an allowed root (if any are configured).

    Routes that read an absolute, client-supplied path (image serving, dimensions) call
    this so an exposed server can be restricted via ``TCIP_IMAGE_ROOTS``. Raises
    :class:`ValueError` if an allow-list is set and the path escapes it; with no allow-list
    the resolved path is returned unchanged.
    """
    resolved = Path(path).resolve()
    roots = allowed_image_roots()
    if not roots:
        return resolved
    for root in roots:
        try:
            resolved.relative_to(root)
            return resolved
        except ValueError:
            continue
    raise ValueError(f"path {resolved} is outside the allowed image roots")


def safe_join(root: Path | str, *parts: str) -> Path:
    """Join ``parts`` under ``root``, rejecting traversal and absolute paths.

    Raises :class:`ValueError` if the resolved path escapes ``root``.
    """
    base = Path(root).resolve()
    # Accept forward slashes on Windows by normalising via PurePosixPath first
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
