"""The storage seam's public surface: module functions bound to one backend per process.

The module functions hold the rules that must mean the same thing on every backend (kind and key
validation, the concurrency policy, the transaction-misuse rules) and delegate the storage itself
to the bound backend. :func:`close_connections` is a process-lifecycle operation over every key the
bound backend has touched.
"""

from __future__ import annotations

import hashlib
import shutil
import threading
from collections.abc import Generator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path
from typing import Any, BinaryIO, Protocol

from tcip_store.errors import (
    CapabilityUnavailable,
    ListingUnsupported,
    PolicyViolation,
    StoreNotBound,
    TransactionMisuse,
)
from tcip_store.model import (
    REQUIRED,
    Capabilities,
    Key,
    LogPage,
    Version,
    Versioned,
    canonical_path,
)
from tcip_store.registry import get_descriptor, validate_key


class Txn(Protocol):
    """A transaction's handle: reads and staged writes over exactly the keys it holds."""

    def read(self, key: Key, *, default: Any = REQUIRED) -> Any:
        """One of the transaction's keys, including this transaction's own staged write. A key the
        transaction does not hold raises ``TransactionMisuse``.
        """
        ...

    def write(self, key: Key, value: Any) -> None:
        """Stage a replace of one of the transaction's keys."""
        ...

    def delete(self, key: Key) -> None:
        """Stage a delete of one of the transaction's keys."""
        ...


class Store(Protocol):
    """What a backend implements and what the contract suite parameterizes over."""

    def read_versioned(self, key: Key, *, default: Any = REQUIRED) -> Versioned: ...

    def exists(self, key: Key) -> bool: ...

    def replace(self, key: Key, value: Any, *, expect: Version | None = None) -> Version: ...

    def delete(self, key: Key, *, expect: Version | None = None) -> None: ...

    def transaction(
        self, keys: Sequence[Key], *, timeout_s: float | None = None
    ) -> AbstractContextManager[Txn]: ...

    def keys(self, store: str, root: str, prefix: tuple[str, ...] = ()) -> list[Key]: ...

    def append(self, key: Key, record: Mapping[str, Any]) -> None: ...

    def read_log(self, key: Key, *, after: str | None = None) -> LogPage: ...

    def clear_log(self, key: Key) -> int: ...

    def read_blob_versioned(self, key: Key, *, default: Any = REQUIRED) -> Versioned: ...

    def put_blob(self, key: Key, data: bytes, *, expect: Version | None = None) -> Version: ...

    def write_blob(
        self, key: Key, *, expect: Version | None = None
    ) -> AbstractContextManager[BinaryIO]: ...

    def open_blob(self, key: Key) -> AbstractContextManager[BinaryIO]: ...

    def blob_path(self, key: Key) -> Path: ...

    def capabilities(self) -> Capabilities: ...

    def close(self) -> None: ...


_bound: Store | None = None
_open_transaction = threading.local()


def bind(backend: Store) -> None:
    """Bind this process's backend, at an entry point. A backend that cannot provide a guarantee it
    would have to declare refuses at its own construction with ``BackendUnavailable``.
    """
    global _bound
    _bound = backend


def unbind() -> None:
    """Drop the bound backend, for a test fixture tearing its backend down."""
    global _bound
    _bound = None


def _backend() -> Store:
    if _bound is None:
        raise StoreNotBound(
            "no storage backend is bound: the process entry point (the MCP server, the web "
            "backend, a training subprocess, or a test fixture) must call tcip_store.bind() "
            "before any store operation"
        )
    return _bound


def capabilities() -> Capabilities:
    """What the bound backend guarantees."""
    return _backend().capabilities()


def close_connections() -> None:
    """Close every connection the bound backend holds, on every thread.

    Refuses with ``TransactionMisuse`` inside an open transaction.

    The sqlite backend closes every connection of every root and every thread: a connection is
    opened with ``check_same_thread=False``, so closing one while another thread is inside a
    statement on it kills the process. The caller guarantees no other thread is inside the seam for
    the call's duration; the thread-local transaction guard here sees only the calling thread's own
    transaction. The file backend's own ``close`` releases nothing.
    """
    _refuse_inside_transaction("close_connections", _CLOSES_UNDER_ANOTHER_THREAD)
    _backend().close()


def _active_txn() -> Txn | None:
    return getattr(_open_transaction, "txn", None)


_ESCAPES_THE_STAGING = (
    "use txn.write or txn.delete, or name the key in transaction(...). A module-level write "
    "would escape this transaction's staging here and commit separately on a database "
    "backend, so the two backends would mean different things by one call"
)

_LOGS_ARE_NOT_TRANSACTIONAL = (
    "close the transaction first. A transaction holds records, so a log key cannot be named "
    "in one, and an append inside a transaction would join it on a database backend and roll "
    "back with the body, while append returns only once the entry has survived"
)

_CLOSES_UNDER_ANOTHER_THREAD = (
    "close the transaction first. Closing every connection while this thread holds one open "
    "inside a transaction would close the connection out from under its own commit"
)


def _refuse_inside_transaction(operation: str, reason: str = _ESCAPES_THE_STAGING) -> None:
    if _active_txn() is not None:
        raise TransactionMisuse(f"{operation} is not allowed inside an open transaction: {reason}")


def _check_policy(key: Key, expect: Version | None, operation: str) -> None:
    descriptor = get_descriptor(key.store)
    if descriptor.concurrency == "cas" and expect is None:
        raise PolicyViolation(
            f"{key.store!r} declares concurrency='cas' because more than one writer "
            f"read-modify-writes it, so an unconditional {operation} would be a lost "
            "update. Read with read_versioned and pass expect=, or name the key in "
            "transaction(...)"
        )


def read(key: Key, *, default: Any = REQUIRED) -> Any:
    """The record's decoded value.

    Raises ``NotFound`` when the record is absent and no ``default`` was given: a caller
    that treats absence as meaningful says so by passing one. Raises ``DecodeError`` when
    the record exists but will not decode, whatever ``default`` says.
    """
    return read_versioned(key, default=default).value


def read_versioned(key: Key, *, default: Any = REQUIRED) -> Versioned:
    """Value plus version token, read together so the pair cannot straddle a write.

    The token is the input to ``replace(expect=...)``, which turns a check-then-act with a
    window into a compare-and-set inside the lock.
    """
    backend = _backend()
    validate_key(key, expect_kind="record", operation="read")
    return backend.read_versioned(key, default=default)


def exists(key: Key) -> bool:
    """Whether the record or blob exists, without reading it. A create-only write is ``replace(...,
    expect=Version.ABSENT)``.
    """
    backend = _backend()
    validate_key(key, expect_kind=("record", "blob"), operation="exists")
    return backend.exists(key)


def replace(key: Key, value: Any, *, expect: Version | None = None) -> Version:
    """Replace one record whole, atomically, and return its new version.

    Acquires the key's lock for the whole call, across threads and across OS processes, whether or
    not ``expect`` is given.

    ``expect`` compares against the stored version re-read under the lock: a mismatch raises
    ``VersionConflict`` with nothing written. ``Version.ABSENT`` writes only if no record exists,
    which is how a create-once record is expressed. ``None`` is an unconditional replace under the
    lock, and is refused with ``PolicyViolation`` on a store that declares ``concurrency='cas'``.

    Raises ``TransactionMisuse`` if the calling thread holds an open transaction.
    """
    backend = _backend()
    validate_key(key, expect_kind="record", operation="replace")
    _refuse_inside_transaction("replace")
    _check_policy(key, expect, "replace")
    return backend.replace(key, value, expect=expect)


def delete(key: Key, *, expect: Version | None = None) -> None:
    """Remove the record or blob. Absence is not an error. Same locking, ``expect``,
    concurrency-policy and transaction-misuse rules as ``replace``.
    """
    backend = _backend()
    validate_key(key, expect_kind=("record", "blob"), operation="delete")
    _refuse_inside_transaction("delete")
    _check_policy(key, expect, "delete")
    backend.delete(key, expect=expect)


@contextmanager
def transaction(*keys: Key, timeout_s: float | None = None) -> Generator[Txn]:
    """Serialize a read-modify-write over exactly ``keys``, across threads and processes.

    Name every key the body will touch, up front. Locks are acquired in an order derived from the
    keys themselves, so two callers naming the same set in different orders cannot deadlock. Reads
    inside see a consistent view of those keys, including this transaction's own staged writes;
    writes are applied on a clean exit, in the order the keys were named, not the order they were
    written. An exception inside applies nothing.

    A thread holds at most one transaction; opening a second raises ``TransactionMisuse`` naming
    the multi-key form.

    Every named key hangs off one root, compared through ``canonical_path`` so two spellings of one
    directory are one root; keys from two roots raise ``TransactionMisuse`` naming them.

    On a file backend this does not promise all-or-nothing application across more than one key: a
    crash during the apply can leave a prefix of the named key order on disk, each record
    individually intact and decodable. ``capabilities().multi_key_atomic_commit`` says which.

    Raises ``StoreBusy`` naming the contended key if the locks are not acquired in time.
    """
    backend = _backend()
    if not keys:
        raise TransactionMisuse("transaction() must name at least one key")
    if _active_txn() is not None:
        raise TransactionMisuse(
            "a thread holds at most one transaction: name every key in one "
            "transaction(a, b) instead of nesting"
        )
    for key in keys:
        validate_key(key, expect_kind="record", operation="transaction")
    roots: dict[str, str] = {}
    for key in keys:
        roots.setdefault(canonical_path(key.root), key.root)
    if len(roots) > 1:
        spelled = ", ".join(repr(root) for root in sorted(roots.values()))
        raise TransactionMisuse(
            f"a transaction's keys hang off one root, and these name {len(roots)}: "
            f"{spelled}. Take one root's keys in one transaction and the other's in another, "
            "ordered so a crash between them leaves a detectably stale state"
        )
    named: list[Key] = []
    for key in keys:
        if key not in named:
            named.append(key)
    with backend.transaction(tuple(named), timeout_s=timeout_s) as txn:
        _open_transaction.txn = txn
        try:
            yield txn
        finally:
            _open_transaction.txn = None


def keys(store: str, root: str, prefix: tuple[str, ...] = ()) -> list[Key]:
    """Every key in ``store`` under ``root`` whose parts begin with ``prefix``, sorted.

    Raises ``ListingUnsupported`` naming the store when its descriptor declares no enumeration.
    Backend bookkeeping is never returned as a key.
    """
    backend = _backend()
    descriptor = get_descriptor(store)
    if not descriptor.enumerable:
        raise ListingUnsupported(
            f"store {store!r} declares no enumeration, so an empty list would be a guess "
            "rather than an answer"
        )
    return backend.keys(store, root, prefix)


def append(key: Key, record: Mapping[str, Any]) -> None:
    """Append one entry to an append-only log, durably.

    Returns only once the entry will survive a crash of this process and is readable by another
    process. Concurrent appenders from any process are serialized, so entries are never interleaved
    or lost. ``replace`` and ``delete`` against a log key raise ``WrongKind``.

    Raises ``TransactionMisuse`` if the calling thread holds an open transaction.
    """
    backend = _backend()
    validate_key(key, expect_kind="log", operation="append")
    _refuse_inside_transaction("append", _LOGS_ARE_NOT_TRANSACTIONAL)
    backend.append(key, record)


def read_log(key: Key, *, after: str | None = None) -> LogPage:
    """Entries after the cursor ``after`` (from the start when None), plus a new cursor.

    See ``LogPage`` for the torn-tail and interior-corruption reporting. The cursor is opaque: a
    byte offset under the file backend, a commit-ordered sequence number under a database.

    On the file backend this waits for any other holder of the same key: an append in flight, a
    clear, or another reader, since the key's lock is exclusive. The database backend's read waits
    on no writer at all.
    """
    backend = _backend()
    validate_key(key, expect_kind="log", operation="read_log")
    return backend.read_log(key, after=after)


def clear_log(key: Key) -> int:
    """Remove every entry from an append-only log and report how many entries it held.

    A cursor taken before this call stays comparable to one taken after it: replaying it against
    the log this call leaves behind returns exactly the entries appended since the clear, on either
    backend.

    An exception from this call on the file backend leaves either every entry where it was (the
    next append or clear_log discards the stale stage) or the entries gone with only the cursor
    watermark pending (the next append or clear_log installs it first).

    Raises ``TransactionMisuse`` if the calling thread holds an open transaction.
    """
    backend = _backend()
    validate_key(key, expect_kind="log", operation="clear_log")
    _refuse_inside_transaction("clear_log", _LOGS_ARE_NOT_TRANSACTIONAL)
    return backend.clear_log(key)


def read_blob_versioned(key: Key, *, default: Any = REQUIRED) -> Versioned:
    """A blob's bytes and its version token, read together so the pair cannot straddle a write.

    The token is the input to ``put_blob(expect=...)`` and ``write_blob(expect=...)``. ``default``
    answers absence the way ``read_versioned`` does, paired with ``Version.ABSENT``. A blob too
    large to hold in memory is read with ``open_blob`` instead, which carries no token.
    """
    backend = _backend()
    validate_key(key, expect_kind="blob", operation="read_blob_versioned")
    return backend.read_blob_versioned(key, default=default)


def put_blob(key: Key, data: bytes, *, expect: Version | None = None) -> Version:
    """Write a blob whole, atomically, and return the version derived from its bytes.

    ``expect`` compares against the stored version re-read under the same lock the write takes: a
    mismatch raises ``VersionConflict`` with nothing written, and the prior bytes stay readable.
    ``Version.ABSENT`` writes only if no blob exists. ``None`` is an unconditional write under the
    lock.
    """
    backend = _backend()
    validate_key(key, expect_kind="blob", operation="put_blob")
    _refuse_inside_transaction("put_blob")
    return backend.put_blob(key, data, expect=expect)


def write_blob(key: Key, *, expect: Version | None = None) -> AbstractContextManager[BinaryIO]:
    """A writable binary stream that becomes the blob only on a clean exit.

    For a producer that writes through a library rather than handing over bytes. An
    exception inside the block leaves the existing blob byte-identical. ``expect`` carries
    the same meaning it does on ``put_blob`` and is compared before the stream opens, so a
    conflict costs nothing the producer has written.
    """
    backend = _backend()
    validate_key(key, expect_kind="blob", operation="write_blob")
    _refuse_inside_transaction("write_blob")
    return backend.write_blob(key, expect=expect)


def put_blob_from_path(key: Key, source: Path | str, *, expect: Version | None = None) -> Version:
    """Write a blob whose bytes already sit in a file on disk, streamed through ``write_blob``
    rather than read whole into memory first.

    ``expect`` is checked under the key's lock before any byte moves, the destination is fsynced
    before the replace, and a failure partway through the copy leaves the previous bytes untouched.
    The version returned hashes the bytes as they stream through.
    """
    hasher = hashlib.sha256()
    with write_blob(key, expect=expect) as dst, open(source, "rb") as src:
        # COPY_BUFSIZE is real at runtime; the stdlib stub omits it.
        while chunk := src.read(getattr(shutil, "COPY_BUFSIZE")):
            dst.write(chunk)
            hasher.update(chunk)
    return Version(hasher.hexdigest())


def open_blob(key: Key) -> AbstractContextManager[BinaryIO]:
    """A readable binary stream for the blob."""
    backend = _backend()
    validate_key(key, expect_kind="blob", operation="open_blob")
    return backend.open_blob(key)


def blob_path(key: Key) -> Path:
    """A real filesystem path for the blob, for a library that cannot take a file object. Requires
    the bound backend to declare ``local_blob_paths`` and the store's descriptor to declare itself
    path-readable.
    """
    backend = _backend()
    descriptor = validate_key(key, expect_kind="blob", operation="blob_path")
    if not backend.capabilities().local_blob_paths:
        raise CapabilityUnavailable(
            f"the bound backend ({type(backend).__name__}) does not declare local_blob_paths: "
            "read the blob with open_blob instead of asking for a path"
        )
    if not descriptor.path_readable:
        raise CapabilityUnavailable(
            f"store {key.store!r} does not declare itself path-readable: use open_blob"
        )
    return backend.blob_path(key)


__all__ = [
    "Store",
    "Txn",
    "append",
    "bind",
    "blob_path",
    "capabilities",
    "clear_log",
    "close_connections",
    "delete",
    "exists",
    "keys",
    "open_blob",
    "put_blob",
    "put_blob_from_path",
    "read",
    "read_blob_versioned",
    "read_log",
    "read_versioned",
    "replace",
    "transaction",
    "unbind",
    "write_blob",
]
