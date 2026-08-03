"""Local-filesystem directory browsing for the dataset picker's folder browser.

The GUI runs with its backend on the same machine as the browser, so "the server's
filesystem" is the user's own. This lists directories only, and only where reads are
allowed: with ``TCIP_IMAGE_ROOTS`` unset (the default local single-user GUI) browsing is
unrestricted; set it and browsing is confined to those roots (a networked deployment).
The wider trust-boundary hardening for exposed servers (Origin / TrustedHost / token) is a
documented follow-up, not done here.
"""

from __future__ import annotations

import os
import re
import stat as statmod
import string
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query

from tcip_web.paths import allowed_image_roots, assert_path_allowed

router = APIRouter(prefix="/api/fs", tags=["fs"])

_HIDDEN_OR_SYSTEM = getattr(statmod, "FILE_ATTRIBUTE_HIDDEN", 0) | getattr(
    statmod, "FILE_ATTRIBUTE_SYSTEM", 0
)
_FOUND_RE = re.compile(r"^found\.\d{3}$", re.IGNORECASE)


def _is_noise_dir(name: str, st: os.stat_result) -> bool:
    """A folder the picker should hide: dotfiles and Windows system/hidden folders
    (``$RECYCLE.BIN``, ``System Volume Information``, ``FOUND.000``, or anything carrying
    the hidden/system attribute: the latter is a no-op on POSIX)."""
    if name.startswith((".", "$")):
        return True
    if name.lower() == "system volume information" or _FOUND_RE.match(name):
        return True
    return bool(getattr(st, "st_file_attributes", 0) & _HIDDEN_OR_SYSTEM)


def _entry(p: Path) -> dict:
    # ``is_dataset_root`` marks folders that contain an ``images/`` subdir, so the picker
    # can hint which ones are directly selectable as a dataset root.
    return {"name": p.name or str(p), "path": str(p), "is_dataset_root": (p / "images").is_dir()}


def _windows_drives() -> list[dict]:
    drives = []
    for letter in string.ascii_uppercase:
        root = f"{letter}:\\"
        if os.path.exists(root):
            drives.append({"name": root, "path": root, "is_dataset_root": False})
    return drives


def _roots_listing() -> dict:
    """Top-level view (no path given): configured roots, else drives (Windows) / '/'."""
    roots = allowed_image_roots()
    if roots:
        return {
            "path": "",
            "parent": None,
            "is_dataset_root": False,
            "entries": [_entry(r) for r in roots if r.is_dir()],
        }
    if os.name == "nt":
        return {"path": "", "parent": None, "is_dataset_root": False, "entries": _windows_drives()}
    return _list_dir(Path("/"))


def _list_dir(p: Path) -> dict:
    entries: list[dict] = []
    try:
        children = sorted(p.iterdir(), key=lambda c: c.name.lower())
    except PermissionError as exc:
        raise HTTPException(403, f"permission denied: {p}") from exc
    except OSError as exc:
        raise HTTPException(400, f"cannot list {p}: {exc}") from exc
    for child in children:
        try:
            st = child.stat()  # follows symlinks; raises on access-denied/dead links
        except OSError:
            continue  # e.g. a locked "System Volume Information" junction: just skip it
        if not statmod.S_ISDIR(st.st_mode):
            continue  # directories only
        if _is_noise_dir(child.name, st):
            continue
        entries.append(_entry(child))

    # Offer a parent link, but null it when going up would escape the allowed roots.
    parent: str | None = str(p.parent) if p.parent != p else None
    if parent is not None:
        try:
            assert_path_allowed(parent)
        except ValueError:
            parent = None

    return {
        "path": str(p),
        "parent": parent,
        "is_dataset_root": (p / "images").is_dir(),
        "has_tcip": (p / ".tcip").is_dir(),
        "entries": entries,
    }


@router.get("/list")
def list_dir(
    path: str | None = Query(None, description="Directory to list; empty = top level"),
) -> dict:
    """List sub-directories of ``path`` (or the top-level drives/roots when empty)."""
    if not path:
        return _roots_listing()
    try:
        resolved = assert_path_allowed(path)
    except ValueError as exc:
        raise HTTPException(403, str(exc)) from exc
    if not resolved.is_dir():
        raise HTTPException(404, f"not a directory: {path}")
    return _list_dir(resolved)
