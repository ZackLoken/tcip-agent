"""The storage seam's public surface: module functions over the one backend a process binds."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from tcip_store.errors import StoreNotBoundError, TransactionMisuseError
from tcip_store.model import (
    REQUIRED,
    Key,
    LogPage,
    Version,
    Versioned,
    canonical_path,
    held_transaction,
    refuse_inside_transaction,
)
from tcip_store.sqlite_backend import SqliteBackend, Txn

if TYPE_CHECKING:
    from pydantic import JsonValue

    from tcip_store.model import ReadDefault

_bound: SqliteBackend | None = None


def bind(backend: SqliteBackend | None = None) -> SqliteBackend:
    """Bind this process's backend and return it: ``backend``, or a new :class:`SqliteBackend`.

    Construction refuses with ``BackendUnavailableError`` when cross-process exclusion is
    unavailable.
    """
    global _bound
    _bound = backend if backend is not None else SqliteBackend()
    return _bound


def unbind() -> None:
    """Drop the bound backend."""
    global _bound
    _bound = None


def _backend() -> SqliteBackend:
    if _bound is None:
        raise StoreNotBoundError(
            "no storage backend is bound: the process entry point (the MCP server, the web "
            "backend, a training subprocess, or a test fixture) must call tcip_store.bind() "
            "before any store operation"
        )
    return _bound


def release_root(root: str | Path) -> None:
    """Close every connection the bound backend holds on ``root`` or on a root under it.

    Connections close on every thread, so the tree can be moved; every other root's connections
    stay open. Waits for every operation using one of those connections to return. Refuses with
    ``TransactionMisuseError`` inside an open transaction.
    """
    refuse_inside_transaction("release_root")
    _backend().release(canonical_path(root))


def read(key: Key, *, default: ReadDefault = REQUIRED) -> JsonValue:
    """Return the record's decoded value.

    Raises ``NotFoundError`` when the record is absent and no ``default`` was given, and
    ``DecodeError`` when the record exists but will not decode, whatever ``default`` says.
    """
    return read_versioned(key, default=default).value


def read_versioned(key: Key, *, default: ReadDefault = REQUIRED) -> Versioned:
    """Return value plus version token, read together so the pair cannot straddle a write."""
    return _backend().read_versioned(key, default=default)


def exists(key: Key) -> bool:
    """Return whether the record exists, without decoding it."""
    return _backend().exists(key)


def replace(
    key: Key, value: JsonValue, *, expect: Version | None = None
) -> Version:
    """Replace one record whole, atomically, and return its new version.

    ``expect`` compares against the stored version inside the write: a mismatch raises
    ``VersionConflictError`` with nothing written. ``Version.ABSENT`` writes only if no record
    exists; ``None`` is an unconditional replace. Raises ``TransactionMisuseError`` inside an open
    transaction.
    """
    refuse_inside_transaction("replace")
    return _backend().replace(key, value, expect=expect)


def delete(key: Key, *, expect: Version | None = None) -> bool:
    """Remove the record and answer whether one was there to remove; absence is not an error.

    Refuses (``TransactionMisuseError``) inside an open transaction. ``expect``, when given, is
    compared against the stored version inside the write: a mismatch raises
    ``VersionConflictError`` with nothing removed.
    """
    refuse_inside_transaction("delete")
    return _backend().delete(key, expect=expect)


@contextmanager
def transaction(*keys: Key, timeout_s: float | None = None) -> Generator[Txn]:
    """Hold one commit over the records and logs ``keys`` name, all under one root.

    The body reads, writes, deletes and appends through the yielded :class:`Txn`; a clean exit
    commits every change at once and an exception commits none. Refuses with
    ``TransactionMisuseError``: no key, a second transaction of either kind on the same thread
    (:func:`~tcip_store.model.held_transaction`), or keys under two roots. Raises ``StoreBusyError``
    naming the first key when the write lock is not acquired in time.
    """
    if not keys:
        raise TransactionMisuseError(
            "transaction() must name at least one key"
        )
    roots = sorted({canonical_path(key.root) for key in keys})
    if len(roots) > 1:
        raise TransactionMisuseError(
            f"a transaction's keys hang off one root, and these name {len(roots)}: "
            f"{', '.join(roots)}. Take one root's keys in one transaction and the other's in "
            "another"
        )
    with (
        held_transaction(),
        _backend().transaction(keys, timeout_s=timeout_s) as txn,
    ):
        yield txn


def keys(store: str, root: str, prefix: tuple[str, ...] = ()) -> list[Key]:
    """Return every record or log key in ``store`` under ``root`` whose parts begin with ``prefix``.

    The keys are sorted.
    """
    return _backend().keys(store, root, prefix)


def stores(root: str) -> list[str]:
    """Return every store name ``root`` holds a record or a log entry under, sorted."""
    return _backend().stores(root)


def append(key: Key, record: dict[str, JsonValue]) -> None:
    """Append one entry to a log, committed before returning.

    Concurrent appenders from any process are serialized. Raises ``TransactionMisuseError``
    inside an open transaction.
    """
    refuse_inside_transaction("append")
    _backend().append(key, record)


def read_log(key: Key, *, after: str | None = None) -> LogPage:
    """Return the entries after the cursor ``after`` (from the start when None) and a new cursor."""
    return _backend().read_log(key, after=after)


def clear_log(key: Key) -> int:
    """Remove every entry from a log and report how many it held.

    Raises ``TransactionMisuseError`` inside an open transaction.
    """
    refuse_inside_transaction("clear_log")
    return _backend().clear_log(key)
