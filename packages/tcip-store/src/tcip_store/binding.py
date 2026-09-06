"""Which backend a process binds, decided once at its entry point.

This lives beside both backends rather than inside either: the SQLite backend composes the
file backend for blobs, so a selector inside the file backend would point it back at the
module that imports it.
"""

from __future__ import annotations

import os

from tcip_store.file_backend import DEFAULT_LOCK_TIMEOUT_S, FileBackend
from tcip_store.sqlite_backend import SqliteBackend
from tcip_store.store import _backend, bind

BACKEND_ENV = "TCIP_STORE_BACKEND"
"""Names the backend to bind. Unset takes ``DEFAULT_BACKEND``."""

FILE_BACKEND = "file"
SQLITE_BACKEND = "sqlite"
DEFAULT_BACKEND = SQLITE_BACKEND
"""What an unset environment binds: one database per root, at ``<root>/.tcip/store.db``.

A root whose records and logs are still loose files is refused rather than read as empty, so a
layout that predates its database is conformed with ``tcip adopt-store`` before a process
on this default touches it. ``tcip export-store`` writes the files back out, which is what
``TCIP_STORE_BACKEND=file`` then reads.
"""


def bind_default(*, lock_timeout_s: float = DEFAULT_LOCK_TIMEOUT_S) -> FileBackend | SqliteBackend:
    """Bind this process's backend at its entry point and return it.

    Every entry point calls this: the MCP server, the web backend, a training subprocess, a
    test fixture. Constructing the backend is what refuses with ``BackendUnavailable`` when
    cross-process exclusion is unavailable, so a process that cannot lock stops here rather
    than writing without one. The instance is returned so an entry point that owns the
    process's lifetime can close it.

    An unrecognized value in the environment is a ``ValueError`` rather than a fallback: a
    misspelled name that quietly bound the other backend would send a process's writes
    somewhere its operator did not choose.
    """
    name = os.environ.get(BACKEND_ENV) or DEFAULT_BACKEND
    if name == FILE_BACKEND:
        backend: FileBackend | SqliteBackend = FileBackend(lock_timeout_s=lock_timeout_s)
    elif name == SQLITE_BACKEND:
        backend = SqliteBackend(lock_timeout_s=lock_timeout_s)
    else:
        raise ValueError(
            f"{BACKEND_ENV}={name!r} names no backend: set it to {FILE_BACKEND!r} or "
            f"{SQLITE_BACKEND!r}, or leave it unset for {DEFAULT_BACKEND!r}"
        )
    bind(backend)
    return backend


def is_database_backend() -> bool:
    """Whether this process's currently bound backend is the database one.

    Checks the actual bound instance, not the environment: a caller (a test, a door deciding
    whether to build a database) may bind directly rather than through :func:`bind_default`, and
    the two must never disagree about which backend is live.
    """
    return isinstance(_backend(), SqliteBackend)


__all__ = [
    "BACKEND_ENV",
    "DEFAULT_BACKEND",
    "FILE_BACKEND",
    "SQLITE_BACKEND",
    "bind_default",
    "is_database_backend",
]
