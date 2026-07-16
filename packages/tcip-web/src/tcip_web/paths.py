"""Path resolution helpers with traversal protection.

All GUI routes that read user-supplied paths go through :func:`safe_join`
so a malicious/broken client cannot reach files outside the configured
project root.
"""

from __future__ import annotations

import ipaddress
import os
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse


def is_loopback_host(host: str) -> bool:
    """True if ``host`` binds only the local machine (127.0.0.0/8, ::1, localhost).

    ``0.0.0.0`` / ``::`` mean "all interfaces" and are therefore not loopback — binding
    them exposes the server to the network.
    """
    h = (host or "").strip().lower()
    if h in ("localhost", "::1"):
        return True
    if h in ("", "0.0.0.0", "::"):
        return False
    try:
        return ipaddress.ip_address(h).is_loopback
    except ValueError:
        return False


# ── Browser trust boundary (shared by app.py + the WebSocket routes) ──
# A loopback bind stays frictionless (no auth); the Origin check stops a cross-site page
# from opening a WebSocket and reading GUI state (which includes filesystem paths). A
# missing Origin means a non-browser client. Env-derived so a deliberately exposed bind
# also trusts its own host.
_BIND_HOST = os.environ.get("TCIP_WEB_HOST", "127.0.0.1")
_ALLOWED_ORIGIN_HOSTS = {"localhost", "127.0.0.1", "::1"}
if not is_loopback_host(_BIND_HOST):
    _ALLOWED_ORIGIN_HOSTS.add(_BIND_HOST)


def origin_allowed(origin: str | None) -> bool:
    """Allow same-machine browser origins; a missing Origin means a non-browser client."""
    if not origin:
        return True
    host = urlparse(origin).hostname or ""
    return is_loopback_host(host) or host in _ALLOWED_ORIGIN_HOSTS


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


def assert_project_root_allowed(project_root: str | Path) -> Path:
    """Confine a client-supplied ``project_root`` to the allowed roots.

    A route that derives a ``.tcip/state``, ``.tcip/reports``, or ``.tcip/audit.jsonl``
    path from a request's ``project_root`` should call this before touching disk, so the
    same ``TCIP_IMAGE_ROOTS`` lockdown that confines image reads also confines these
    project-scoped state readers/writers. Thin wrapper over :func:`assert_path_allowed`
    kept as its own name so call sites read as "guarding a project root" and share one
    place to diverge the policy later if needed. Raises :class:`ValueError`; callers
    convert to ``HTTPException(403)`` as elsewhere in this codebase.
    """
    return assert_path_allowed(project_root)


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
