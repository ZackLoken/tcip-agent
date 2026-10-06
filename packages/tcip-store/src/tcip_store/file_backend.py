"""Files written atomically by path, and the lock a root's database is created under.

A blob is a file its owning module addresses by path. Every write takes the path's lock across
threads and processes, compares an expected version derived from the stored bytes, stages the new
bytes in a flushed temp file beside the path and renames it into place, so a failure leaves the
previous bytes.
"""

from __future__ import annotations

import hashlib
import os
import random
import tempfile
import threading
import time
from collections.abc import Callable, Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from tcip_store.errors import (
    BackendUnavailable,
    BadKey,
    NotFound,
    StoreBusy,
    VersionConflict,
)
from tcip_store.model import (
    REQUIRED, Key, Version, Versioned, canonical_path, refuse_inside_transaction,
)

_TEMP_SUFFIX = ".tmp"
_LOCK_SUFFIX = ".lock"

DATABASE_FILENAME = "store.db"
"""The database file a root's records and logs live in, under the root's ``.tcip`` directory."""

_DATABASE_SIDECARS = frozenset({f"{DATABASE_FILENAME}-wal", f"{DATABASE_FILENAME}-shm"})

DEFAULT_LOCK_TIMEOUT_S = 30.0


def creation_temp_name(destination: str, token: str) -> str:
    """The name a file is built under before it is installed at ``destination``: hidden and
    temp-suffixed."""
    return f".{destination}.{token}{_TEMP_SUFFIX}"


def require_absolute_root(root: str) -> Path:
    """The root as a path, or ``BadKey`` for a relative one."""
    directory = Path(root)
    if not directory.is_absolute():
        raise BadKey(
            f"root {root!r} is not an absolute path: a relative root resolves against "
            "whatever directory the process happens to be in"
        )
    return directory


_registry_guard = threading.Lock()
_thread_locks: dict[str, threading.RLock] = {}
_file_locks: dict[str, Any] = {}


def _filelock_classes() -> tuple[Any, Any]:
    """The lock class and its timeout error, or a refusal naming what is missing."""
    try:
        from filelock import FileLock, Timeout
    except ImportError as exc:
        raise BackendUnavailable(
            "the store needs the filelock package for cross-process exclusion and will not "
            "run without it"
        ) from exc
    return FileLock, Timeout


def _locks_for(canonical: str, lock_path: str) -> tuple[threading.RLock, Any]:
    """The one lock pair for a canonical path in this process, one ``FileLock`` instance per path
    with the library's default thread-local counting: counted same-thread re-entry, and a second
    thread blocks.
    """
    file_lock_cls, _ = _filelock_classes()
    with _registry_guard:
        thread_lock = _thread_locks.get(canonical)
        if thread_lock is None:
            thread_lock = threading.RLock()
            _thread_locks[canonical] = thread_lock
        file_lock = _file_locks.get(canonical)
        if file_lock is None:
            file_lock = file_lock_cls(lock_path)
            _file_locks[canonical] = file_lock
        return thread_lock, file_lock


def lock_file_for(path: Path | str) -> Path:
    """The lock file :func:`path_lock` holds beside the data file at ``path``.

    ``filelock`` deletes it on release under Windows and keeps it under Unix, so whatever removes
    a data file guarded here removes this file through here as well, or the directory it sits in
    never empties on Unix.
    """
    return Path(str(path) + _LOCK_SUFFIX)


@contextmanager
def path_lock(path: Path | str, *, timeout_s: float = DEFAULT_LOCK_TIMEOUT_S) -> Generator[None]:
    """Hold this process's one lock pair for a filesystem path, across threads and processes.

    The parent directory must already exist: the lock file lands beside the data file. Raises
    ``filelock``'s own ``Timeout`` when the wait runs out.
    """
    _, timeout_error = _filelock_classes()
    target = Path(path)
    lock_file = str(lock_file_for(target))
    thread_lock, file_lock = _locks_for(canonical_path(target), lock_file)
    if not thread_lock.acquire(timeout=max(0.0, timeout_s)):
        raise timeout_error(lock_file)
    try:
        file_lock.acquire(timeout=max(0.0, timeout_s))
        try:
            yield
        finally:
            file_lock.release()
    finally:
        thread_lock.release()


def database_file(root: str) -> Path:
    """Where a root's database sits, whether or not one exists."""
    return require_absolute_root(root) / ".tcip" / DATABASE_FILENAME


def database_roots(tree: Path) -> list[Path]:
    """Every root at or under the absolute ``tree`` whose database exists, sorted."""
    return sorted(db.parent.parent for db in tree.rglob(DATABASE_FILENAME)
                  if db == database_file(str(db.parent.parent)))


@contextmanager
def transition_lock(root: str, *, timeout_s: float = DEFAULT_LOCK_TIMEOUT_S) -> Generator[None]:
    """Hold the lock a root's database is created under; creating the root's ``.tcip`` directory
    is its first act."""
    db_path = database_file(root)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with path_lock(db_path, timeout_s=timeout_s):
        yield


def fsync_directory(directory: Path) -> None:
    """Flush a directory entry so a rename or a created directory survives power loss.

    POSIX only.
    """
    if os.name == "nt":
        return
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _version_of(data: bytes) -> Version:
    return Version(hashlib.sha256(data).hexdigest())


def _ensure_parent(path: Path) -> None:
    """Create ``path``'s parent directory, flushing every directory entry it created."""
    parent = path.parent
    if parent.is_dir():
        return
    created: list[Path] = []
    node = parent
    while not node.exists():
        created.append(node)
        node = node.parent
    parent.mkdir(parents=True, exist_ok=True)
    for directory in [node, *reversed(created)]:
        fsync_directory(directory)


def _stage_bytes(path: Path, data: bytes) -> str:
    """Write ``data`` to a flushed temp file beside ``path`` and return the temp file's path."""
    fd, temp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=_TEMP_SUFFIX)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        _remove_quietly(temp)
        raise
    return temp


def _apply_staged(temp: str, path: Path) -> None:
    """Make one staged temp file the file at ``path`` and flush the rename; a rename that fails
    removes its staging file."""
    try:
        retry_while_denied(lambda: os.replace(temp, path), DEFAULT_LOCK_TIMEOUT_S)
    except BaseException:
        _remove_quietly(temp)
        raise
    fsync_directory(path.parent)


def _remove_entry(path: Path) -> None:
    path.unlink(missing_ok=True)
    fsync_directory(path.parent)


def _read_bytes(path: Path) -> bytes | None:
    """The file's bytes, or None when it is absent."""

    def read() -> bytes | None:
        try:
            return path.read_bytes()
        except FileNotFoundError:
            return None

    return retry_while_denied(read, DEFAULT_LOCK_TIMEOUT_S)


@contextmanager
def _locked(path: Path) -> Generator[None]:
    """Hold ``path``'s lock (:func:`path_lock`); ``StoreBusy`` naming it when the wait runs out."""
    _, timeout_error = _filelock_classes()
    _ensure_parent(path)
    try:
        with path_lock(path):
            yield
    except timeout_error:
        raise StoreBusy(str(path), DEFAULT_LOCK_TIMEOUT_S) from None


def require_version(entry: Key | str, data: bytes | None, expect: Version) -> None:
    """Refuse (``VersionConflict`` naming ``entry``) when the stored bytes ``data``, ``None`` for
    an absent entry, are not the version ``expect`` names."""
    current = Version.ABSENT if data is None else _version_of(data)
    if current != expect:
        raise VersionConflict(entry, expect, current)


def versioned(data: bytes | None, default: Any, decode: Callable[[bytes], Any],
              absent: str) -> Versioned:
    """Stored bytes ``data`` as their decoded value (``decode``) and their version, or, for an
    absent entry (``data`` ``None``), ``default`` paired with ``Version.ABSENT``; ``NotFound``
    stating ``absent`` when no default was given."""
    if data is not None:
        return Versioned(decode(data), _version_of(data))
    if default is REQUIRED:
        raise NotFound(f"{absent}. Pass default= if absence is meaningful to this caller")
    return Versioned(default, Version.ABSENT)


def read_blob_versioned(path: Path, *, default: Any = REQUIRED) -> Versioned:
    """A file's bytes and their version, read together (:func:`versioned`)."""
    return versioned(_read_bytes(path), default, bytes, f"{path} does not exist")


def put_blob(path: Path, data: bytes, *, expect: Version | None = None) -> Version:
    """Write a file whole, atomically, and return the version derived from its bytes.

    ``expect`` compares against the stored version re-read under the path's lock: a mismatch
    raises ``VersionConflict`` with nothing written. ``Version.ABSENT`` writes only if no file
    exists; ``None`` is an unconditional write under the lock. Refuses (``TransactionMisuse``)
    inside an open transaction.
    """
    refuse_inside_transaction("put_blob")
    with _locked(path):
        if expect is not None:
            require_version(str(path), _read_bytes(path), expect)
        _apply_staged(_stage_bytes(path, data), path)
    return _version_of(data)


def delete_blob(path: Path, *, expect: Version | None = None) -> None:
    """Remove a file under its lock, comparing ``expect`` and refusing inside an open transaction
    as :func:`put_blob` does; absence is not an error."""
    refuse_inside_transaction("delete_blob")
    with _locked(path):
        if expect is not None:
            require_version(str(path), _read_bytes(path), expect)
        _remove_entry(path)


def is_bookkeeping(name: str) -> bool:
    """Whether a filename is the store's own artifact rather than data: a lock file, a temp file or
    one of the database's WAL sidecars."""
    return (
        name in _DATABASE_SIDECARS
        or name.endswith(_LOCK_SUFFIX)
        or (name.startswith(".") and name.endswith(_TEMP_SUFFIX))
    )


def _remove_quietly(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


def retry_while_denied(action: Callable[[], Any], budget_s: float) -> Any:
    """Run a filesystem action an atomic replace can transiently deny (on Windows, a rename or open
    racing another handle), retrying with jittered waits within ``budget_s``, then raising. Never
    triggers on POSIX.
    """
    deadline = time.monotonic() + budget_s
    while True:
        try:
            return action()
        except PermissionError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(random.uniform(0.005, 0.05))
