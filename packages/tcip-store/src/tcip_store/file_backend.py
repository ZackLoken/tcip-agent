"""The filesystem backend: identity to path, atomic replace, file locks, logs, and blobs.

Byte-compatible with the layout it serves. Nothing moves on disk, and no envelope, version
field, or metadata sidecar is added, because adding one would change bytes. A record's
version is therefore derived from its content on every read rather than persisted.

Roots are resolved at use time, not at bind time: the key carries the root, so a
process that repins its platform root mid-run keeps writing where the caller says.

Records and logs are refused on a root that holds a store database: a file written beside one
is a write the database never sees and the next read never returns. Reads and blobs are
unaffected, which is what keeps every file-reading tool working on an exported root.
"""

from __future__ import annotations

import hashlib
import os
import random
import tempfile
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Protocol

from tcip_store.errors import (
    BackendUnavailable,
    BadKey,
    DecodeError,
    NotFound,
    StoreBusy,
    StoreError,
    TransactionMisuse,
    VersionConflict,
)
from tcip_store.model import (
    REQUIRED,
    Capabilities,
    Key,
    LogPage,
    Version,
    Versioned,
    canonical_order,
    canonical_path,
)
from tcip_store.registry import StoreDescriptor, get_descriptor, registered_stores

_TEMP_SUFFIX = ".tmp"
_LOCK_SUFFIX = ".lock"
_TAIL_SCAN_BYTES = 8192
"""How far back an append looks for the last entry boundary at a time, so a repair costs the
size of the tail rather than the size of the log."""

DATABASE_FILENAME = "store.db"
"""The database file a root's records live in under a database backend.

Named here, where enumeration decides what is an entry, so the backend that creates the file
and the backend that must never return it as a key cannot disagree about its name.
"""

_DATABASE_ARTIFACTS = frozenset(
    {DATABASE_FILENAME, f"{DATABASE_FILENAME}-wal", f"{DATABASE_FILENAME}-shm"}
)


def creation_temp_name(destination: str, token: str) -> str:
    """The name a file is built under before it is installed at ``destination``.

    Hidden and temp-suffixed, which is what ``_is_bookkeeping`` already recognizes, so a build
    in flight is never enumerated as an entry of whatever store owns the directory.
    """
    return f".{destination}.{token}{_TEMP_SUFFIX}"


def require_absolute_root(root: str) -> Path:
    """The root as a path, or the refusal every backend owes a relative one.

    Refused before anything resolves it, so no answer can depend on the directory the process
    happens to be in. Enumeration refuses here too rather than answering with an empty list,
    which would read as "this root holds nothing" when it means "this root names nothing".
    """
    directory = Path(root)
    if not directory.is_absolute():
        raise BadKey(
            f"root {root!r} is not an absolute path: a relative root resolves against "
            "whatever directory the process happens to be in"
        )
    return directory


class Locator(Protocol):
    """One store's identity map: where an entry of it lives under its root.

    The two methods are an inverse pair, which is what makes the mapping checkable:
    enumeration is ``parts_from`` applied over the files under a root, so it cannot drift
    from ``relative_path`` the way two independent callables would. ``parts_from`` returns
    None for a path that is not an entry of this store.

    A locator exists for the file backend only. A backend that keys on (store, root, parts)
    ignores it, which is why it is declared here and not in the shared model.
    """

    def relative_path(self, root: str, parts: tuple[str, ...]) -> PurePosixPath: ...

    def parts_from(self, relative_path: PurePosixPath) -> tuple[str, ...] | None: ...


@dataclass(frozen=True)
class RootedFileLocator:
    """Addresses an entry by its own path segments under a fixed directory of the root.

    ``prefix`` is the directory chain under the root, ``suffix`` the extension the
    last part carries on disk. A key's parts are the remaining segments, so parts
    ``("2026-03-04", "img_0001")`` under prefix ``("annotations",)`` with suffix ``".json"``
    is ``<root>/annotations/2026-03-04/img_0001.json``.

    This is the generic locator, for a store that is addressed by an explicit relative path
    and nothing more. A store with real layout rules wraps its own resolver instead, so the
    path is stated once, in the module that owns it.
    """

    prefix: tuple[str, ...] = ()
    suffix: str = ""

    def relative_path(self, root: str, parts: tuple[str, ...]) -> PurePosixPath:
        """The entry's path relative to its root."""
        if not parts:
            raise BadKey("a rooted-file key needs at least one part")
        segments = (*self.prefix, *parts[:-1], f"{parts[-1]}{self.suffix}")
        return PurePosixPath(*segments)

    def parts_from(self, relative_path: PurePosixPath) -> tuple[str, ...] | None:
        """The parts that produce ``relative_path``, or None when it is not this store's."""
        segments = relative_path.parts
        if segments[: len(self.prefix)] != self.prefix:
            return None
        rest = segments[len(self.prefix) :]
        if not rest:
            return None
        if self.suffix and not rest[-1].endswith(self.suffix):
            return None
        last = rest[-1][: len(rest[-1]) - len(self.suffix)] if self.suffix else rest[-1]
        if not last:
            return None
        return (*rest[:-1], last)


DEFAULT_LOCK_TIMEOUT_S = 30.0

_registry_guard = threading.Lock()
_thread_locks: dict[str, threading.RLock] = {}
_file_locks: dict[str, Any] = {}


def _filelock_classes() -> tuple[Any, Any]:
    """The lock class and its timeout error, or a refusal naming what is missing."""
    try:
        from filelock import FileLock, Timeout
    except ImportError as exc:
        raise BackendUnavailable(
            "the file backend needs the filelock package for cross-process exclusion and "
            "will not run without it: in-process locking alone would leave the platform's "
            "processes free to clobber each other's writes"
        ) from exc
    return FileLock, Timeout


def _locks_for(canonical: str, lock_path: str) -> tuple[threading.RLock, Any]:
    """The one lock pair for a canonical path in this process.

    One ``FileLock`` instance per path is a hard requirement, not an optimization: two
    separately constructed instances on the same path do not exclude re-entry within one
    process, so a write nested inside a transaction on the same key would deadlock against
    itself. One instance with the library's default thread-local counting gives counted
    same-thread re-entry and blocks a second thread.
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


@contextmanager
def path_lock(path: Path | str, *, timeout_s: float = DEFAULT_LOCK_TIMEOUT_S) -> Iterator[None]:
    """Hold this process's one lock pair for a filesystem path, across threads and processes.

    Anything that guards the same path this backend guards has to acquire through here, or
    the two hold separately constructed ``FileLock`` instances that do not exclude each
    other's re-entry and block until the timeout. The parent directory must already exist:
    the lock file lands beside the data file.

    Raises ``filelock``'s own ``Timeout`` when the wait runs out, which is what the backend
    turns into ``StoreBusy`` once it knows which key was contended.
    """
    _, timeout_error = _filelock_classes()
    target = Path(path)
    lock_file = str(target) + _LOCK_SUFFIX
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
    """Where a root's database sits, whether or not one exists.

    Both backends ask here: the database backend to open or build it, the file backend to find
    out whether this root's records have already moved into one. Stating it once is what keeps
    the file that must never be clobbered and the file that must never be written around from
    being two different paths.
    """
    return require_absolute_root(root) / ".tcip" / DATABASE_FILENAME


@contextmanager
def transition_lock(root: str, *, timeout_s: float = DEFAULT_LOCK_TIMEOUT_S) -> Iterator[None]:
    """Hold the one lock that decides whether a root's records live in files or in a database.

    Taken by whatever is about to publish a database (creation, adoption) and by a file-backend
    record write on a root that already has a ``.tcip`` directory, so a publication in flight
    and a write to the layout it is loading cannot interleave. Creating the directory is this
    side's first act, which is what makes it exist for the writer to find.
    """
    db_path = database_file(root)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with path_lock(db_path, timeout_s=timeout_s):
        yield


def unconformed_record_files(root: str, *, limit: int = 0) -> tuple[Path, ...]:
    """Files under ``root`` that a record or log store's locator claims and no blob's does.

    Each one is evidence the root is unconformed: its records are still files rather than rows
    in the database a database backend would read. The question is asked through the locators
    the stores themselves declare rather than through a list of filenames somebody maintains.

    A blob's file stays a file under every backend, so a file some blob store could equally own
    is not evidence that a root's records were left behind. Locator shapes collide, and a
    dataset root's ``classes.json`` is claimed by a plain ``<dir>/<name>.json`` record shape as
    readily as by the blob store that actually owns it, so counting it would refuse every real
    dataset. The cost of that reading is that a root whose only records happen to sit at the
    depth a blob store also addresses goes undetected here; what catches those is adoption's own
    accounting, which names every claimed file no plan takes in. ``limit`` stops the walk once
    that many are found, for a caller that only needs to know whether any exist.
    """
    directory = require_absolute_root(root)
    if not directory.is_dir():
        return ()
    descriptors = [get_descriptor(name) for name in registered_stores()]
    owners = [d for d in descriptors if d.kind in ("record", "log") and d.locator is not None]
    blobs = [d for d in descriptors if d.kind == "blob" and d.locator is not None]
    found: list[Path] = []
    for path in directory.rglob("*"):
        if not path.is_file() or _is_bookkeeping(path.name):
            continue
        relative = PurePosixPath(path.relative_to(directory).as_posix())
        if not _any_claims(owners, relative) or _any_claims(blobs, relative):
            continue
        found.append(path)
        if limit and len(found) >= limit:
            break
    return tuple(found)


def _any_claims(descriptors: list[StoreDescriptor], relative: PurePosixPath) -> bool:
    """Whether any of these stores' locators reads this path as one of its own entries."""
    for descriptor in descriptors:
        assert descriptor.locator is not None
        parts = descriptor.locator.parts_from(relative)
        if parts is not None and len(parts) == len(descriptor.key_fields):
            return True
    return False


def fsync_directory(directory: Path) -> None:
    """Flush a directory entry, so a rename or a created directory survives a power loss.

    POSIX only. Windows has no directory-fsync equivalent, which is why
    ``FileBackend.capabilities().durable_replace`` is false there instead of overstated.
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


@dataclass
class _Staged:
    """One key's pending change inside a transaction."""

    value: Any = None
    removed: bool = False
    temp_path: str | None = None


class FileBackend:
    """A storage backend over the local filesystem.

    Every write takes its key's lock inside the call, replaces through a temp file in the
    destination directory, and flushes when the store declares itself durable. Multi-key
    transactions stage every write, then apply in the caller's declared key order, which is
    a prefix guarantee across a crash and not atomicity; ``capabilities()`` says so.
    """

    def __init__(self, *, lock_timeout_s: float = DEFAULT_LOCK_TIMEOUT_S) -> None:
        _, timeout_error = _filelock_classes()
        self.lock_timeout_s = lock_timeout_s
        self._timeout_error = timeout_error

    def capabilities(self) -> Capabilities:
        """What this backend guarantees on the platform it is running on.

        ``durable_replace`` needs the parent directory's entry flushed after the rename, and
        Windows has no directory-fsync equivalent, so it is reported false there rather than
        claimed. ``cross_machine_exclusion`` is false unconditionally: advisory locks are
        unreliable on network mounts and this backend cannot detect the mount it is on.
        """
        return Capabilities(
            multi_key_atomic_commit=False,
            cross_machine_exclusion=False,
            durable_replace=os.name != "nt",
            durable_append=True,
            local_blob_paths=True,
        )

    # ── identity to path ────────────────────────────────────────────────────────

    def path_for(self, key: Key) -> Path:
        """Where this key's bytes live. The only place a key becomes a path."""
        descriptor = get_descriptor(key.store)
        locator = descriptor.locator
        if locator is None:
            raise StoreError(
                f"store {key.store!r} declares no locator, so the file backend cannot place "
                "it: declare one beside the store's key constructor"
            )
        directory = require_absolute_root(key.root)
        relative = locator.relative_path(key.root, key.parts)
        if relative.is_absolute() or ".." in relative.parts:
            raise BadKey(f"store {key.store!r} placed {list(key.parts)} outside its root")
        return directory.joinpath(*relative.parts)

    # ── locking ─────────────────────────────────────────────────────────────────

    @contextmanager
    def _locked(self, keys: Sequence[Key], timeout_s: float | None = None) -> Iterator[None]:
        timeout = self.lock_timeout_s if timeout_s is None else timeout_s
        requested = tuple(keys)
        items = []
        seen: set[str] = set()
        for key in canonical_order(requested):
            path = self.path_for(key)
            canonical = canonical_path(path)
            if canonical in seen:
                continue
            seen.add(canonical)
            self._ensure_parent(path, durable=get_descriptor(key.store).durable)
            items.append((key, path, canonical))
        deadline = time.monotonic() + timeout
        with ExitStack() as held:
            for key, path, _ in items:
                try:
                    held.enter_context(
                        path_lock(path, timeout_s=deadline - time.monotonic())
                    )
                except self._timeout_error:
                    raise StoreBusy(requested, key, timeout) from None
            yield

    @contextmanager
    def _conform_rail(self, keys: Sequence[Key]) -> Iterator[None]:
        """The file backend's half of the conform rail: hold each root's transition lock and
        refuse record and log writes to a conformed root, whose records live in its database.

        A write here cannot bump a database's counters, so writing a record beside a database
        that already holds it loses the write with nothing to detect it by. The check is under
        the transition lock, which creation and adoption also hold, so a publication cannot
        land between the check and the write. Blobs never reach here: they stay files under
        every backend.

        A root with no ``.tcip`` directory is passed over without taking anything: the lock
        file lands inside that directory, and creating it here would put one in every split,
        run and prediction directory the platform writes a record into. A publication creates
        it as its own first step, so once any has begun the lock is taken and the answer is
        under it.
        """
        roots: list[str] = []
        for key in keys:
            if get_descriptor(key.store).kind in ("record", "log") and key.root not in roots:
                roots.append(key.root)
        with ExitStack() as held:
            for root in roots:
                db_path = database_file(root)
                if not db_path.parent.is_dir():
                    continue
                try:
                    held.enter_context(path_lock(db_path, timeout_s=self.lock_timeout_s))
                except self._timeout_error:
                    raise StoreBusy(tuple(keys), keys[0], self.lock_timeout_s) from None
                if db_path.is_file():
                    raise StoreError(
                        f"{db_path} exists, so this root's records and logs live in the "
                        "database and a file written beside it would be lost with nothing to "
                        "detect it by. Write through the database backend, or write the files "
                        "out with python scripts/export_store.py and bind the file backend "
                        "deliberately with TCIP_STORE_BACKEND=file."
                    )
            yield

    # ── durability primitives ───────────────────────────────────────────────────

    def _fsync_file(self, handle: BinaryIO) -> None:
        """Flush one file's bytes to the device."""
        handle.flush()
        os.fsync(handle.fileno())

    def _fsync_dir(self, directory: Path) -> None:
        """Flush a directory entry, so a rename or a created directory survives a power loss."""
        fsync_directory(directory)

    def _ensure_parent(self, path: Path, *, durable: bool) -> None:
        parent = path.parent
        if parent.is_dir():
            return
        created: list[Path] = []
        node = parent
        while not node.exists():
            created.append(node)
            node = node.parent
        parent.mkdir(parents=True, exist_ok=True)
        if durable:
            for directory in [node, *reversed(created)]:
                self._fsync_dir(directory)

    def _stage_bytes(self, path: Path, data: bytes, *, durable: bool) -> str:
        """Write ``data`` to a temp file beside ``path`` and return the temp file's path."""
        fd, temp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=_TEMP_SUFFIX)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
                if durable:
                    self._fsync_file(handle)
                else:
                    handle.flush()
        except BaseException:
            _remove_quietly(temp)
            raise
        return temp

    def _apply_staged(self, temp: str, path: Path, *, durable: bool) -> None:
        """Make one staged temp file the record, and make that rename durable.

        The parent directory is flushed immediately after this rename, before the next one,
        so a crash mid-apply leaves a prefix of the applied order durable rather than an
        arbitrary subset of it. A rename that fails takes its staging file with it: the
        directories this backend writes into are enumerated by their own readers, and a
        stranded temp file accumulates there for every failed write.
        """
        try:
            _retry_while_denied(lambda: os.replace(temp, path), self.lock_timeout_s)
        except BaseException:
            _remove_quietly(temp)
            raise
        if durable:
            self._fsync_dir(path.parent)

    def _remove_entry(self, path: Path, *, durable: bool) -> None:
        path.unlink(missing_ok=True)
        if durable:
            self._fsync_dir(path.parent)

    def _read_bytes(self, path: Path) -> bytes | None:
        """The record's bytes, or None when it is absent."""

        def read() -> bytes | None:
            try:
                return path.read_bytes()
            except FileNotFoundError:
                return None

        return _retry_while_denied(read, self.lock_timeout_s)

    # ── records ─────────────────────────────────────────────────────────────────

    def read_versioned(self, key: Key, *, default: Any = REQUIRED) -> Versioned:
        descriptor = get_descriptor(key.store)
        path = self.path_for(key)
        data = self._read_bytes(path)
        if data is None:
            if default is REQUIRED:
                raise _missing_record(key)
            return Versioned(default, Version.ABSENT)
        return Versioned(_decode(descriptor, key, data), _version_of(data))

    def exists(self, key: Key) -> bool:
        return self.path_for(key).is_file()

    def replace(self, key: Key, value: Any, *, expect: Version | None = None) -> Version:
        descriptor = get_descriptor(key.store)
        path = self.path_for(key)
        with self._conform_rail([key]), self._locked([key]):
            if expect is not None:
                self._require_version(key, path, expect)
            data = _encode(descriptor, key, value)
            temp = self._stage_bytes(path, data, durable=descriptor.durable)
            self._apply_staged(temp, path, durable=descriptor.durable)
            return _version_of(data)

    def delete(self, key: Key, *, expect: Version | None = None) -> None:
        descriptor = get_descriptor(key.store)
        path = self.path_for(key)
        with self._conform_rail([key]), self._locked([key]):
            if expect is not None:
                self._require_version(key, path, expect)
            self._remove_entry(path, durable=descriptor.durable)

    def _require_version(self, key: Key, path: Path, expect: Version) -> None:
        data = self._read_bytes(path)
        current = Version.ABSENT if data is None else _version_of(data)
        if current != expect:
            raise VersionConflict(key, expect, current)

    @contextmanager
    def transaction(self, keys: Sequence[Key], *, timeout_s: float | None = None) -> Iterator["_FileTxn"]:
        named = tuple(keys)
        with self._conform_rail(named), self._locked(named, timeout_s):
            txn = _FileTxn(self, named)
            yield txn
            txn.apply()

    def keys(self, store: str, root: str, prefix: tuple[str, ...] = ()) -> list[Key]:
        """Every key of ``store`` under ``root``, as identities a caller can read back.

        A store whose layout cannot spell every key it holds (one that sanitizes a separator
        out of a filename, say) declares ``true_parts_from_entry``, and the entry's own bytes
        are what the identity comes from there. The locator still decides which files belong
        to the store; the hook only corrects what the path could not carry.
        """
        descriptor = get_descriptor(store)
        locator = descriptor.locator
        if locator is None:
            raise StoreError(f"store {store!r} declares no locator, so it cannot be enumerated")
        directory = require_absolute_root(root)
        if not directory.is_dir():
            return []
        recover = descriptor.true_parts_from_entry
        found: list[Key] = []
        for path in directory.rglob("*"):
            if not path.is_file() or _is_bookkeeping(path.name):
                continue
            parts = locator.parts_from(PurePosixPath(path.relative_to(directory).as_posix()))
            if parts is None:
                continue
            if recover is not None:
                data = self._read_bytes(path)
                recovered = None if data is None else recover(data)
                if recovered is not None:
                    parts = recovered
            if len(parts) != len(descriptor.key_fields):
                continue
            if parts[: len(prefix)] != tuple(prefix):
                continue
            found.append(Key(store, root, parts))
        return sorted(found, key=lambda k: k.parts)

    # ── logs ────────────────────────────────────────────────────────────────────

    def append(self, key: Key, record: Mapping[str, Any]) -> None:
        """Add one entry to a log, flushed before returning, and flush a first entry's
        directory entry too so the log file itself survives the same crash."""
        descriptor = get_descriptor(key.store)
        path = self.path_for(key)
        data = _encode(descriptor, key, record)
        _refuse_embedded_newline(key, data)
        with self._conform_rail([key]), self._locked([key]):
            existed = path.exists()
            self._repair_torn_tail(path)
            with open(path, "ab") as handle:
                handle.write(data + b"\n")
                self._fsync_file(handle)
            if not existed:
                self._fsync_dir(path.parent)

    def _repair_torn_tail(self, path: Path) -> None:
        """Drop a partial trailing entry left by an appender that died mid-write.

        The fragment's own append never returned, so nothing acknowledged is lost. Without
        the repair the next append would weld itself onto the fragment and turn an in-flight
        tail into interior corruption.
        """
        try:
            with open(path, "r+b") as handle:
                handle.seek(0, os.SEEK_END)
                size = handle.tell()
                if size == 0:
                    return
                handle.seek(size - 1)
                if handle.read(1) == b"\n":
                    return
                position = size
                while position > 0:
                    start = max(0, position - _TAIL_SCAN_BYTES)
                    handle.seek(start)
                    chunk = handle.read(position - start)
                    boundary = chunk.rfind(b"\n")
                    if boundary != -1:
                        handle.truncate(start + boundary + 1)
                        self._fsync_file(handle)
                        return
                    position = start
                handle.truncate(0)
                self._fsync_file(handle)
        except FileNotFoundError:
            return

    def read_log(self, key: Key, *, after: str | None = None) -> LogPage:
        descriptor = get_descriptor(key.store)
        path = self.path_for(key)
        start = int(after) if after else 0
        try:
            with open(path, "rb") as handle:
                handle.seek(start)
                data = handle.read()
        except FileNotFoundError:
            return LogPage(records=[], cursor=str(start))
        if not data:
            return LogPage(records=[], cursor=str(start))
        torn = not data.endswith(b"\n")
        lines = data.split(b"\n")
        trailing = lines.pop()
        consumed = len(data) - (len(trailing) if torn else 0)
        records: list[Mapping[str, Any]] = []
        corrupt: list[int] = []
        for position, line in enumerate(lines):
            try:
                records.append(_decode(descriptor, key, line))
            except DecodeError:
                corrupt.append(position)
        return LogPage(
            records=records,
            cursor=str(start + consumed),
            torn_tail=torn,
            corrupt=tuple(corrupt),
        )

    # ── blobs ───────────────────────────────────────────────────────────────────

    def read_blob_versioned(self, key: Key, *, default: Any = REQUIRED) -> Versioned:
        path = self.path_for(key)
        data = self._read_bytes(path)
        if data is None:
            if default is REQUIRED:
                raise NotFound(
                    f"{key.store}{list(key.parts)} has no blob under {key.root}. Pass "
                    "default= if absence is meaningful to this caller"
                )
            return Versioned(default, Version.ABSENT)
        return Versioned(data, _version_of(data))

    def put_blob(self, key: Key, data: bytes, *, expect: Version | None = None) -> Version:
        descriptor = get_descriptor(key.store)
        path = self.path_for(key)
        with self._locked([key]):
            if expect is not None:
                self._require_version(key, path, expect)
            temp = self._stage_bytes(path, data, durable=descriptor.durable)
            self._apply_staged(temp, path, durable=descriptor.durable)
        return _version_of(data)

    @contextmanager
    def write_blob(self, key: Key, *, expect: Version | None = None) -> Iterator[BinaryIO]:
        descriptor = get_descriptor(key.store)
        path = self.path_for(key)
        with self._locked([key]):
            if expect is not None:
                self._require_version(key, path, expect)
            fd, temp = tempfile.mkstemp(
                dir=str(path.parent), prefix=f".{path.name}.", suffix=_TEMP_SUFFIX
            )
            handle = os.fdopen(fd, "wb")
            try:
                yield handle
                if descriptor.durable:
                    self._fsync_file(handle)
                handle.close()
            except BaseException:
                handle.close()
                _remove_quietly(temp)
                raise
            self._apply_staged(temp, path, durable=descriptor.durable)

    @contextmanager
    def open_blob(self, key: Key) -> Iterator[BinaryIO]:
        path = self.path_for(key)
        try:
            handle = _retry_while_denied(lambda: open(path, "rb"), self.lock_timeout_s)
        except FileNotFoundError:
            raise NotFound(f"{key.store}{list(key.parts)} has no blob under {key.root}") from None
        try:
            yield handle
        finally:
            handle.close()

    def blob_path(self, key: Key) -> Path:
        return self.path_for(key)

    def close(self) -> None:
        """Release what this backend holds between calls, which is nothing.

        Every file handle is opened and closed inside the operation that needs it, so this
        exists to give both backends one lifecycle a binder can call rather than to free
        anything here.
        """


class _FileTxn:
    """The file backend's transaction handle: reads under the lock, writes staged until exit."""

    def __init__(self, backend: FileBackend, keys: tuple[Key, ...]) -> None:
        self._backend = backend
        self._keys = keys
        self._staged: dict[Key, _Staged] = {}

    def _held(self, key: Key) -> None:
        if key not in self._keys:
            raise _unheld_key(key)

    def read(self, key: Key, *, default: Any = REQUIRED) -> Any:
        self._held(key)
        staged = self._staged.get(key)
        if staged is not None:
            if staged.removed:
                if default is REQUIRED:
                    raise _deleted_in_transaction(key)
                return default
            return staged.value
        return self._backend.read_versioned(key, default=default).value

    def write(self, key: Key, value: Any) -> None:
        self._held(key)
        self._staged[key] = _Staged(value=value)

    def delete(self, key: Key) -> None:
        self._held(key)
        self._staged[key] = _Staged(removed=True)

    def apply(self) -> None:
        """Encode and stage every write, then apply in the declared key order.

        Every temp file is written before any rename, so the crash window is the rename
        sequence rather than the whole body, and every record on disk stays individually
        intact.
        """
        pending = [(key, self._staged[key]) for key in self._keys if key in self._staged]
        try:
            for key, staged in pending:
                if staged.removed:
                    continue
                descriptor = get_descriptor(key.store)
                path = self._backend.path_for(key)
                staged.temp_path = self._backend._stage_bytes(
                    path, _encode(descriptor, key, staged.value), durable=descriptor.durable
                )
        except BaseException:
            for _, staged in pending:
                if staged.temp_path is not None:
                    _remove_quietly(staged.temp_path)
            raise
        for key, staged in pending:
            descriptor = get_descriptor(key.store)
            path = self._backend.path_for(key)
            if staged.removed:
                self._backend._remove_entry(path, durable=descriptor.durable)
            else:
                assert staged.temp_path is not None
                self._backend._apply_staged(staged.temp_path, path, durable=descriptor.durable)


def _encode(descriptor: StoreDescriptor, key: Key, value: Any) -> bytes:
    """The value's bytes, or a refusal naming the entry and what would not encode.

    ``json.dumps`` names neither the store nor the key, and the canonical codec refuses a
    non-finite number and an unserializable object rather than fabricating a spelling for
    either, so the message has to say which record and which type before a caller can act
    on it.
    """
    assert descriptor.codec is not None
    try:
        return descriptor.codec.encode(value)
    except (TypeError, ValueError) as exc:
        raise StoreError(
            f"{key.store}{list(key.parts)} under {key.root} does not encode: {exc}. "
            f"The value is a {type(value).__name__}; convert what it holds to a JSON type "
            "at the writer rather than leaving the codec to spell it."
        ) from exc


def _decode(descriptor: StoreDescriptor, key: Key, data: bytes) -> Any:
    assert descriptor.codec is not None
    try:
        return descriptor.codec.decode(data)
    except Exception as exc:
        raise DecodeError(
            f"{key.store}{list(key.parts)} under {key.root} exists but does not decode: {exc}"
        ) from exc


def _missing_record(key: Key) -> NotFound:
    """The refusal a required read of an absent record raises, worded once for every backend."""
    return NotFound(
        f"{key.store}{list(key.parts)} has no record under {key.root}. Pass default= if "
        "absence is meaningful to this caller"
    )


def _unheld_key(key: Key) -> TransactionMisuse:
    """The refusal a transaction raises for a key it does not hold, on every backend."""
    return TransactionMisuse(
        f"{key.store}{list(key.parts)} is not held by this transaction: name every key the "
        "body touches in transaction(...). An unheld read inside a transaction is the lost "
        "update this layer exists to prevent"
    )


def _deleted_in_transaction(key: Key) -> NotFound:
    """The refusal a required read of a key this transaction deleted raises, on every backend."""
    return NotFound(f"{key.store}{list(key.parts)} was deleted by this transaction")


def _refuse_embedded_newline(key: Key, data: bytes) -> None:
    """Refuse an encoded log entry that is more than one line, whatever stores the entry.

    A log entry is one line: the file backend terminates it with a newline and an export
    writes database-held entries back out the same way, so an embedded newline would split one
    entry into two wherever the bytes land.
    """
    if b"\n" in data:
        raise ValueError(
            f"log store {key.store!r} encoded an entry containing a newline: a log entry is "
            "one line, so its codec must not indent or embed raw newlines"
        )




def _is_bookkeeping(name: str) -> bool:
    """Whether a filename is a storage backend's own artifact rather than an entry.

    Lock files outlive the writes that made them, a temp file is visible for the duration of a
    write or a database build, and a database and its WAL sidecars sit inside the very root
    whose entries are being enumerated; none of them may ever surface as a key. Anything else
    that is not an entry is rejected by the store's own locator instead of by pattern matching.
    """
    return (
        name in _DATABASE_ARTIFACTS
        or name.endswith(_LOCK_SUFFIX)
        or (name.startswith(".") and name.endswith(_TEMP_SUFFIX))
    )


def _remove_quietly(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


def _retry_while_denied(action: Callable[[], Any], budget_s: float) -> Any:
    """Run a filesystem action an atomic replace can transiently deny, then give up loudly.

    On Windows both sides of a replace are exposed to this: the rename is denied while any
    other handle is open on the destination, and another process opening the destination is
    denied while the rename is in flight. A virus scanner or search indexer produces the
    same denial with nothing else running. None of that is a torn read and none of it is
    corruption, so it is retried rather than reported as either, and a denial that outlasts
    the budget is raised rather than swallowed. On POSIX this never triggers.

    The budget is the backend's own lock timeout: one number for how long this layer is
    willing to wait on one key. The waits are jittered because a fixed delay lets a reader
    and a writer settle into lockstep, each waking into the other's open window.
    """
    deadline = time.monotonic() + budget_s
    while True:
        try:
            return action()
        except PermissionError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(random.uniform(0.005, 0.05))
