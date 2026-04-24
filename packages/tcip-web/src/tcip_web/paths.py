"""Path resolution helpers with traversal protection.

All GUI routes that read user-supplied paths go through :func:`safe_join`
so a malicious/broken client cannot reach files outside the configured
project root.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath


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
