"""The SQLite backend: one WAL database per scope root, with blobs left as files.

Records and log entries live in ``<scope>/.tcip/store.db``; blob bytes never enter a database
and are served by a composed :class:`~tcip_store.file_backend.FileBackend`, so a dataset stays
self-contained and ``blob_path`` keeps answering with a real path.

A database only ever appears complete. Creation builds it under a unique temp name in
rollback-journal mode, commits, closes, fsyncs, and installs it with a no-clobber primitive,
all under a per-scope transition lock, so a crash leaves temp artifacts rather than a
half-built database and the file's existence is a completeness marker by construction. Every
open then verifies the journal mode, the synchronous level, the schema version and the schema
itself before a single row is read.

Exclusion is the database rather than the key: one writer at a time per scope, readers never
blocked. That is stronger for consistency and weaker for availability than a lock per path,
and ``capabilities()`` says which of the two guarantees follow.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import uuid
from collections.abc import Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO

from tcip_store.errors import (
    BackendUnavailable,
    DecodeError,
    StoreBusy,
    StoreError,
    VersionConflict,
)
from tcip_store.file_backend import (
    DATABASE_FILENAME,
    DEFAULT_LOCK_TIMEOUT_S,
    FileBackend,
    _decode,
    _deleted_in_transaction,
    _encode,
    _filelock_classes,
    _missing_record,
    _refuse_embedded_newline,
    _remove_quietly,
    _unheld_key,
    _version_of,
    creation_temp_name,
    path_lock,
    require_absolute_scope,
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
from tcip_store.registry import get_descriptor

SCHEMA_VERSION = 1
"""What ``pragma user_version`` carries. This is the backend's own table shape, not the
platform's record schemas, which stay domain work."""

SCHEMA_DDL = """
create table if not exists meta (
    key   text primary key,
    value text not null
) without rowid;

create table if not exists records (
    store      text not null,
    parts      text not null,
    value      blob not null,
    updated_at text not null,
    primary key (store, parts)
);

create table if not exists store_counters (
    store            text primary key,
    change_counter   integer not null,
    exported_counter integer
) without rowid;

create table if not exists tombstones (
    store      text not null,
    parts      text not null,
    deleted_at text not null,
    primary key (store, parts)
) without rowid;

create table if not exists log_entries (
    id          integer primary key autoincrement,
    store       text not null,
    parts       text not null,
    entry       blob not null,
    appended_at text not null
);
create index if not exists log_entries_by_key on log_entries (store, parts, id);
"""

_SYNCHRONOUS_LEVELS = {"NORMAL": 1, "FULL": 2}


def encode_parts(parts: tuple[str, ...]) -> str:
    """The one spelling a key's parts carry in the database, pinned here and nowhere else.

    ``ensure_ascii=True`` keeps a part carrying non-ASCII comparable byte for byte whatever
    the connection's text handling does, and the separators drop the whitespace ``json.dumps``
    would otherwise put between elements, so two spellings of one key cannot address two rows.
    """
    return json.dumps(list(parts), ensure_ascii=True, separators=(",", ":"))


def decode_parts(text: str) -> tuple[str, ...]:
    """The parts a stored spelling names."""
    return tuple(json.loads(text))


def database_path(scope: str) -> Path:
    """Where this scope's database lives, or ``BadKey`` when the scope is not absolute.

    The refusal comes before canonicalization: resolving a relative scope would turn a refusal
    into a cwd-dependent guess. It is the file backend's own refusal, called rather than
    restated, so both backends answer a relative scope identically.
    """
    return require_absolute_scope(scope) / ".tcip" / DATABASE_FILENAME


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _schema_of(conn: sqlite3.Connection) -> tuple[tuple[str, str, str], ...]:
    """Every schema object a database holds, with its stored sql normalized for comparison.

    SQLite rewrites the text it stores, so the expected side is produced by executing the same
    DDL rather than by restating it; whitespace is the only difference the two sides may carry.
    """
    rows = conn.execute("select type, name, sql from sqlite_master order by type, name").fetchall()
    return tuple((kind, name, " ".join((sql or "").split())) for kind, name, sql in rows)


_reference_lock = threading.Lock()
_reference: tuple[tuple[str, str, str], ...] | None = None


def reference_schema() -> tuple[tuple[str, str, str], ...]:
    """The schema the canonical DDL produces, read back out of a database that ran it."""
    global _reference
    with _reference_lock:
        if _reference is None:
            conn = sqlite3.connect(":memory:")
            try:
                conn.executescript(SCHEMA_DDL)
                _reference = _schema_of(conn)
            finally:
                conn.close()
        return _reference


def _fsync_path(path: Path) -> None:
    """Flush one file's bytes and metadata to the device.

    Opened read-write because Windows answers a flush only for a handle with write access.
    """
    fd = os.open(path, os.O_RDWR)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _install_without_clobbering(temp: Path, destination: Path) -> None:
    """Publish the built database, refusing rather than replacing an existing one.

    ``os.rename`` refuses an existing destination on Windows; on POSIX it replaces, so the hard
    link is what fails with EEXIST there and the temp name is unlinked afterwards.
    """
    if os.name == "nt":
        os.rename(temp, destination)
    else:
        os.link(temp, destination)
        os.unlink(temp)


class SqliteBackend:
    """A storage backend holding records and logs in one SQLite database per scope root.

    Blob operations are forwarded to a composed file backend, which is also what refuses to
    exist without cross-process locking, so this backend inherits that refusal rather than
    restating it.
    """

    def __init__(self, *, lock_timeout_s: float = DEFAULT_LOCK_TIMEOUT_S) -> None:
        _, timeout_error = _filelock_classes()
        self.lock_timeout_s = lock_timeout_s
        self._timeout_error = timeout_error
        self._files = FileBackend(lock_timeout_s=lock_timeout_s)
        self._connections: dict[tuple[int, int, str], sqlite3.Connection] = {}
        self._guard = threading.Lock()

    def capabilities(self) -> Capabilities:
        """What this backend guarantees on the platform it is running on.

        ``multi_key_atomic_commit`` is true because a transaction's staged writes are one SQL
        commit. ``cross_machine_exclusion`` is false for the reason the file backend's is:
        SQLite locking is unreliable on network filesystems and WAL needs shared memory it
        cannot get there. ``durable_replace`` answers for records and blobs at once, and blobs
        delegate to a rename whose directory entry Windows offers no way to flush, so the one
        answer under-claims records there rather than overstating blobs.
        """
        return Capabilities(
            multi_key_atomic_commit=True,
            cross_machine_exclusion=False,
            durable_replace=os.name != "nt",
            durable_append=True,
            local_blob_paths=True,
        )

    def close(self) -> None:
        """Close every connection this backend opened, so no handle outlives it.

        The composed file backend is closed too: it serves this backend's blobs, so leaving it
        open would be closing half of what this instance actually holds.
        """
        with self._guard:
            for conn in self._connections.values():
                conn.close()
            self._connections.clear()
        self._files.close()

    # ── database lifecycle ──────────────────────────────────────────────────────

    def _connection(
        self, scope: str, keys: tuple[Key, ...] = (), timeout_s: float | None = None
    ) -> sqlite3.Connection:
        """This process, thread and scope's connection, creating the database if it is absent.

        The pid is in the slot because a forked worker inherits the parent's handles and must
        not reuse them; the thread is in it because one connection per thread is what lets a
        write hold the database without another thread's read joining its transaction.
        """
        db_path = database_path(scope)
        slot = (os.getpid(), threading.get_ident(), canonical_path(scope))
        with self._guard:
            existing = self._connections.get(slot)
        if existing is not None:
            return existing
        if not db_path.is_file():
            self._publish(db_path, scope, keys, timeout_s)
        conn = self._open(db_path)
        with self._guard:
            self._connections[slot] = conn
        return conn

    def _publish(
        self,
        db_path: Path,
        scope: str,
        keys: tuple[Key, ...] = (),
        timeout_s: float | None = None,
    ) -> None:
        """Build a database beside its destination and install it atomically and exclusively.

        The transition lock is held across the existence re-check, the build and the install,
        so two creators serialize and the loser finds the winner's database and opens it. A
        wait that runs out is this layer's own refusal, never the lock library's timeout
        escaping raw: several processes binding one root at startup all reach here at once.
        """
        timeout = self.lock_timeout_s if timeout_s is None else timeout_s
        self._files._ensure_parent(db_path, durable=True)
        started = time.monotonic()
        try:
            held = path_lock(db_path, timeout_s=timeout)
            held.__enter__()
        except self._timeout_error:
            raise self._contended(scope, keys, time.monotonic() - started) from None
        try:
            if db_path.is_file():
                return
            temp = db_path.parent / creation_temp_name(db_path.name, uuid.uuid4().hex)
            try:
                self._build(temp)
                _fsync_path(temp)
                _install_without_clobbering(temp, db_path)
            except BaseException:
                _remove_quietly(str(temp))
                raise
            _fsync_path(db_path)
            self._files._fsync_dir(db_path.parent)
        finally:
            held.__exit__(None, None, None)

    def _contended(self, scope: str, keys: tuple[Key, ...], waited_s: float) -> StoreError:
        """The refusal for a wait this backend gave up on, naming a key when the call named one.

        A call that named no key (an enumeration) has no key to attribute the wait to, so it is
        told which scope it waited on instead of being handed a key it never asked about.
        """
        if keys:
            return StoreBusy(keys, keys[0], waited_s)
        return StoreError(
            f"waited {waited_s:.1f}s for the store database under {scope} to be created by "
            "another process and gave up; nothing was written"
        )

    def _build(self, temp: Path) -> None:
        """Apply the canonical DDL to a fresh rollback-journal database and close it.

        Never WAL: a WAL database holds committed data in a sidecar the install does not carry,
        so the file installed would be missing the schema it was just given.
        """
        conn = sqlite3.connect(str(temp), isolation_level=None)
        try:
            mode = conn.execute("pragma journal_mode = delete").fetchone()[0]
            if str(mode).lower() != "delete":
                raise BackendUnavailable(
                    f"a database being created would not take rollback-journal mode and "
                    f"reported {mode!r}, so the file installed could not hold its own commits"
                )
            conn.executescript(SCHEMA_DDL)
            conn.execute(f"pragma user_version = {SCHEMA_VERSION}")
        finally:
            conn.close()

    def _open(self, db_path: Path) -> sqlite3.Connection:
        """Open a published database and verify everything about it before it is used.

        Identity is checked before anything is set: the version and the schema are reads, and
        switching a database to WAL is a write, so a file this backend did not build is refused
        while it is still exactly as its own tool left it.
        """
        conn = sqlite3.connect(str(db_path), isolation_level=None, check_same_thread=False)
        try:
            version = conn.execute("pragma user_version").fetchone()[0]
            if version != SCHEMA_VERSION:
                raise StoreError(
                    f"{db_path} carries user_version {version}, not {SCHEMA_VERSION}: it was "
                    "written by another tool or another version of this backend, and reading "
                    "it here would be a guess about what its rows mean"
                )
            self._verify_schema(conn, db_path)

            mode = str(conn.execute("pragma journal_mode").fetchone()[0]).lower()
            if mode != "wal":
                mode = str(conn.execute("pragma journal_mode = wal").fetchone()[0]).lower()
            if mode != "wal":
                raise BackendUnavailable(
                    f"{db_path} would not take WAL journal mode and reported {mode!r}: a "
                    "rollback-journal database blocks every reader behind the writer, which is "
                    "not the exclusion this backend declares"
                )
            self._apply_synchronous(conn, "FULL")
            level = conn.execute("pragma synchronous").fetchone()[0]
            if level != _SYNCHRONOUS_LEVELS["FULL"]:
                raise BackendUnavailable(
                    f"{db_path} reports synchronous={level} after it was set to FULL, so a "
                    "committed write's durability is not what this backend would declare"
                )
        except BaseException:
            conn.close()
            raise
        return conn

    def _verify_schema(self, conn: sqlite3.Connection, db_path: Path) -> None:
        found = _schema_of(conn)
        expected = reference_schema()
        if found == expected:
            return
        found_objects = {(kind, name) for kind, name, _ in found}
        expected_objects = {(kind, name) for kind, name, _ in expected}
        missing = sorted(expected_objects - found_objects)
        extra = sorted(found_objects - expected_objects)
        changed = sorted(
            name
            for kind, name, sql in found
            if (kind, name) in expected_objects
            and sql != next(s for k, n, s in expected if (k, n) == (kind, name))
        )
        raise StoreError(
            f"{db_path} does not carry this backend's schema: missing {missing}, "
            f"unexpected {extra}, differently defined {changed}. A database this backend did "
            "not build is never half-read"
        )

    def _apply_synchronous(self, conn: sqlite3.Connection, level: str) -> None:
        """Set how hard a commit on this connection flushes, from a store's declared durability."""
        conn.execute(f"pragma synchronous = {level}")

    # ── error mapping ───────────────────────────────────────────────────────────

    def _translate(
        self, exc: sqlite3.Error, keys: tuple[Key, ...], waited_s: float
    ) -> StoreError:
        """Every sqlite3 failure as a typed refusal, so no caller sees a raw driver error.

        Contention names the first key the call itself named: exclusion is the database, so
        this backend cannot know which of the caller's keys the holder is actually writing.
        """
        text = str(exc).lower()
        contended = isinstance(exc, sqlite3.OperationalError) and (
            "locked" in text or "busy" in text
        )
        if contended and keys:
            return StoreBusy(keys, keys[0], waited_s)
        scope = keys[0].scope if keys else "?"
        return StoreError(f"the store database under {scope} refused the operation: {exc}")

    @contextmanager
    def _mapped(self, keys: tuple[Key, ...], waited_s: float = 0.0) -> Iterator[None]:
        try:
            yield
        except sqlite3.Error as exc:
            raise self._translate(exc, keys, waited_s) from exc

    def _set_busy_timeout(self, conn: sqlite3.Connection, timeout_s: float) -> None:
        conn.execute(f"pragma busy_timeout = {int(max(0.0, timeout_s) * 1000)}")

    # ── write transactions ──────────────────────────────────────────────────────

    @contextmanager
    def _write(
        self, keys: tuple[Key, ...], *, timeout_s: float | None = None
    ) -> Iterator[sqlite3.Connection]:
        """One ``begin immediate`` transaction over the single scope a call names.

        The write lock is taken up front rather than upgraded from a read, which is what keeps
        the snapshot-conflict class (which no busy timeout covers) out of reach. A store that
        relaxed durability commits at NORMAL; a transaction spanning both takes the stricter
        of the two, since one commit cannot be half flushed.
        """
        timeout = self.lock_timeout_s if timeout_s is None else timeout_s
        with self._mapped(keys, timeout):
            conn = self._connection(keys[0].scope, keys, timeout)
            self._set_busy_timeout(conn, timeout)
            self._apply_synchronous(conn, self._synchronous_for(keys))
        started = time.monotonic()
        try:
            conn.execute("begin immediate")
        except sqlite3.Error as exc:
            raise self._translate(exc, keys, time.monotonic() - started) from exc
        try:
            with self._mapped(keys, timeout):
                yield conn
                conn.execute("commit")
        except BaseException:
            if conn.in_transaction:
                conn.execute("rollback")
            raise

    def _synchronous_for(self, keys: tuple[Key, ...]) -> str:
        durable = any(get_descriptor(key.store).durable for key in keys)
        return "FULL" if durable else "NORMAL"

    def _bump(self, conn: sqlite3.Connection, store: str) -> None:
        """Count one change against a store, in the transaction that makes it.

        Counter, tombstone and export-stamp writes are bookkeeping and bump nothing, so an
        export cannot invalidate itself, and a store with no row here was never written.
        """
        conn.execute(
            "insert into store_counters (store, change_counter, exported_counter) "
            "values (?, 1, null) "
            "on conflict(store) do update set change_counter = change_counter + 1",
            (store,),
        )

    def _put(self, conn: sqlite3.Connection, key: Key, data: bytes) -> None:
        parts = encode_parts(key.parts)
        conn.execute(
            "insert into records (store, parts, value, updated_at) values (?, ?, ?, ?) "
            "on conflict(store, parts) do update set "
            "value = excluded.value, updated_at = excluded.updated_at",
            (key.store, parts, data, _now()),
        )
        conn.execute(
            "delete from tombstones where store = ? and parts = ?", (key.store, parts)
        )
        self._bump(conn, key.store)

    def _drop(self, conn: sqlite3.Connection, key: Key) -> None:
        parts = encode_parts(key.parts)
        conn.execute("delete from records where store = ? and parts = ?", (key.store, parts))
        conn.execute(
            "insert into tombstones (store, parts, deleted_at) values (?, ?, ?) "
            "on conflict(store, parts) do update set deleted_at = excluded.deleted_at",
            (key.store, parts, _now()),
        )
        self._bump(conn, key.store)

    def _stored(self, conn: sqlite3.Connection, key: Key) -> bytes | None:
        row = conn.execute(
            "select value from records where store = ? and parts = ?",
            (key.store, encode_parts(key.parts)),
        ).fetchone()
        return None if row is None else row[0]

    def _require_version(
        self, conn: sqlite3.Connection, key: Key, expect: Version
    ) -> None:
        data = self._stored(conn, key)
        current = Version.ABSENT if data is None else _version_of(data)
        if current != expect:
            raise VersionConflict(key, expect, current)

    # ── records ─────────────────────────────────────────────────────────────────

    def read_versioned(self, key: Key, *, default: Any = REQUIRED) -> Versioned:
        descriptor = get_descriptor(key.store)
        db_path = database_path(key.scope)
        data = None
        if db_path.is_file():
            with self._mapped((key,)):
                data = self._stored(self._connection(key.scope, (key,)), key)
        if data is None:
            if default is REQUIRED:
                raise _missing_record(key)
            return Versioned(default, Version.ABSENT)
        return Versioned(_decode(descriptor, key, data), _version_of(data))

    def exists(self, key: Key) -> bool:
        if get_descriptor(key.store).kind == "blob":
            return self._files.exists(key)
        db_path = database_path(key.scope)
        if not db_path.is_file():
            return False
        with self._mapped((key,)):
            return self._stored(self._connection(key.scope, (key,)), key) is not None

    def replace(self, key: Key, value: Any, *, expect: Version | None = None) -> Version:
        descriptor = get_descriptor(key.store)
        data = _encode(descriptor, key, value)
        with self._write((key,)) as conn:
            if expect is not None:
                self._require_version(conn, key, expect)
            self._put(conn, key, data)
        return _version_of(data)

    def delete(self, key: Key, *, expect: Version | None = None) -> None:
        if get_descriptor(key.store).kind == "blob":
            self._files.delete(key, expect=expect)
            return
        with self._write((key,)) as conn:
            if expect is not None:
                self._require_version(conn, key, expect)
            self._drop(conn, key)

    @contextmanager
    def transaction(
        self, keys: Sequence[Key], *, timeout_s: float | None = None
    ) -> Iterator["_SqliteTxn"]:
        named = tuple(keys)
        with self._write(named, timeout_s=timeout_s) as conn:
            txn = _SqliteTxn(self, conn, named)
            yield txn
            txn.apply()

    def keys(self, store: str, scope: str, prefix: tuple[str, ...] = ()) -> list[Key]:
        descriptor = get_descriptor(store)
        if descriptor.kind == "blob":
            return self._files.keys(store, scope, prefix)
        db_path = database_path(scope)
        if not db_path.is_file():
            return []
        with self._mapped(()):
            rows = self._connection(scope).execute(
                "select parts from records where store = ?", (store,)
            ).fetchall()
        found = [
            Key(store, scope, parts)
            for parts in (decode_parts(text) for (text,) in rows)
            if parts[: len(prefix)] == tuple(prefix)
        ]
        return sorted(found, key=lambda k: k.parts)

    # ── logs ────────────────────────────────────────────────────────────────────

    def append(self, key: Key, record: Mapping[str, Any]) -> None:
        descriptor = get_descriptor(key.store)
        data = _encode(descriptor, key, record)
        _refuse_embedded_newline(key, data)
        with self._write((key,)) as conn:
            conn.execute(
                "insert into log_entries (store, parts, entry, appended_at) values (?, ?, ?, ?)",
                (key.store, encode_parts(key.parts), data, _now()),
            )
            self._bump(conn, key.store)

    def read_log(self, key: Key, *, after: str | None = None) -> LogPage:
        """Entries committed after the cursor, in commit order.

        The cursor is the last returned row's id, which autoincrement never reuses.
        ``torn_tail`` is structurally always False: an entry is a committed row or it is not
        there, so there is no partial tail to hold back. An entry that will not decode is
        still reported through ``corrupt`` rather than skipped.
        """
        descriptor = get_descriptor(key.store)
        start = int(after) if after else 0
        db_path = database_path(key.scope)
        if not db_path.is_file():
            return LogPage(records=[], cursor=str(start))
        with self._mapped((key,)):
            rows = self._connection(key.scope, (key,)).execute(
                "select id, entry from log_entries where store = ? and parts = ? and id > ? "
                "order by id",
                (key.store, encode_parts(key.parts), start),
            ).fetchall()
        records: list[Mapping[str, Any]] = []
        corrupt: list[int] = []
        cursor = start
        for position, (row_id, entry) in enumerate(rows):
            try:
                records.append(_decode(descriptor, key, entry))
            except DecodeError:
                corrupt.append(position)
            cursor = row_id
        return LogPage(records=records, cursor=str(cursor), corrupt=tuple(corrupt))

    # ── blobs, which stay files ─────────────────────────────────────────────────

    def read_blob_versioned(self, key: Key, *, default: Any = REQUIRED) -> Versioned:
        return self._files.read_blob_versioned(key, default=default)

    def put_blob(self, key: Key, data: bytes, *, expect: Version | None = None) -> Version:
        return self._files.put_blob(key, data, expect=expect)

    def write_blob(
        self, key: Key, *, expect: Version | None = None
    ) -> AbstractContextManager[BinaryIO]:
        return self._files.write_blob(key, expect=expect)

    def open_blob(self, key: Key) -> AbstractContextManager[BinaryIO]:
        return self._files.open_blob(key)

    def blob_path(self, key: Key) -> Path:
        return self._files.blob_path(key)



class _SqliteTxn:
    """The database transaction's handle: reads inside the commit, writes staged until exit.

    Staging in Python rather than writing as the body goes is what makes this mean the same
    thing the file backend's transaction means: a read sees the transaction's own staged
    write, an unheld key is refused, and the declared key order is the order writes land in.
    """

    def __init__(
        self, backend: SqliteBackend, conn: sqlite3.Connection, keys: tuple[Key, ...]
    ) -> None:
        self._backend = backend
        self._conn = conn
        self._keys = keys
        self._staged: dict[Key, tuple[bool, Any]] = {}

    def _held(self, key: Key) -> None:
        if key not in self._keys:
            raise _unheld_key(key)

    def read(self, key: Key, *, default: Any = REQUIRED) -> Any:
        self._held(key)
        staged = self._staged.get(key)
        if staged is not None:
            removed, value = staged
            if removed:
                if default is REQUIRED:
                    raise _deleted_in_transaction(key)
                return default
            return value
        data = self._backend._stored(self._conn, key)
        if data is None:
            if default is REQUIRED:
                raise _missing_record(key)
            return default
        return _decode(get_descriptor(key.store), key, data)

    def write(self, key: Key, value: Any) -> None:
        self._held(key)
        self._staged[key] = (False, value)

    def delete(self, key: Key) -> None:
        self._held(key)
        self._staged[key] = (True, None)

    def apply(self) -> None:
        """Encode and write every staged change, in the order the keys were declared."""
        for key in self._keys:
            staged = self._staged.get(key)
            if staged is None:
                continue
            removed, value = staged
            if removed:
                self._backend._drop(self._conn, key)
            else:
                data = _encode(get_descriptor(key.store), key, value)
                self._backend._put(self._conn, key, data)
