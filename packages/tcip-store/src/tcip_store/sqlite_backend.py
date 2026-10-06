"""The store: a root's records and log entries in one WAL database, ``<root>/.tcip/store.db``.

Creation builds the database under a unique temp name in rollback-journal mode, commits, closes,
fsyncs and installs it with a no-clobber primitive, all under the root's transition lock, so the
file's existence marks a complete database. Every open verifies the schema, WAL and full
synchronous before a single row is read. One writer at a time per root;
readers are never blocked.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import uuid
from collections import Counter
from collections.abc import Callable, Generator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any

from tcip_store.errors import (
    BackendUnavailableError,
    DecodeError,
    StoreBusyError,
    StoreError,
    TransactionMisuseError,
)
from tcip_store.file_backend import (
    DEFAULT_LOCK_TIMEOUT_S,
    _ensure_parent,
    _filelock_classes,
    _remove_quietly,
    _version_of,
    creation_temp_name,
    database_file,
    fsync_directory,
    require_version,
    transition_lock,
    versioned,
)
from tcip_store.model import REQUIRED, Key, LogPage, Version, Versioned, canonical_path
from tcip_store.values import decode_value, encode_log_line, encode_record

SCHEMA_DDL = """
create table if not exists records (
    store text not null,
    parts text not null,
    value blob not null,
    primary key (store, parts)
);

create table if not exists log_entries (
    id    integer primary key autoincrement,
    store text not null,
    parts text not null,
    entry blob not null
);
create index if not exists log_entries_by_key on log_entries (store, parts, id);
"""

_FULL = 2
"""What ``pragma synchronous`` reports at FULL."""


def encode_parts(parts: tuple[str, ...]) -> str:
    """The one JSON spelling (ASCII, no whitespace) of a key's parts in the database."""
    return json.dumps(list(parts), ensure_ascii=True, separators=(",", ":"))


def decode_parts(text: str) -> tuple[str, ...]:
    """The parts a stored spelling names."""
    return tuple(json.loads(text))


def _schema_of(conn: sqlite3.Connection) -> tuple[tuple[str, str, str], ...]:
    """Every schema object a database holds, with its stored sql normalized for comparison against
    the DDL executed afresh.
    """
    rows = conn.execute("select type, name, sql from sqlite_master order by type, name").fetchall()
    return tuple((kind, name, " ".join((sql or "").split())) for kind, name, sql in rows)


_reference_lock = threading.Lock()
_reference: tuple[tuple[str, str, str], ...] | None = None


def reference_schema() -> tuple[tuple[str, str, str], ...]:
    """The schema the DDL produces, read back out of a database that ran it."""
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
    """Flush one file's bytes and metadata to the device, through a read-write handle (Windows
    flushes only a handle with write access).
    """
    fd = os.open(path, os.O_RDWR)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _install_without_clobbering(temp: Path, destination: Path) -> None:
    """Publish the built database, refusing rather than replacing an existing one (``os.rename`` on
    Windows, a hard link then an unlink on POSIX).
    """
    if os.name == "nt":
        os.rename(temp, destination)
    else:
        os.link(temp, destination)
        os.unlink(temp)


def verify_identity(conn: sqlite3.Connection, db_path: Path) -> None:
    """Refuse (``StoreError``) a database that is not this module's, before a single row is read:
    a file that is not a SQLite database, or one whose schema is not this module's DDL."""
    try:
        found = _schema_of(conn)
    except sqlite3.DatabaseError as exc:
        raise StoreError(
            f"{db_path} is not a SQLite database: {exc}. Something else holds the name this "
            "store's database has."
        ) from exc
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
        f"{db_path} does not carry this store's schema: missing {missing}, "
        f"unexpected {extra}, differently defined {changed}. A database this store did not "
        "build is never half-read"
    )


def _set_wal(conn: sqlite3.Connection) -> str:
    """Put one connection's database into WAL, returning the mode it reports afterwards. Only safe
    under the transition lock: changing the journal mode does not wait on a busy database.
    """
    return str(conn.execute("pragma journal_mode = wal").fetchone()[0]).lower()


def _refuse_rollback_journal(db_path: Path, mode: str) -> None:
    if mode != "wal":
        raise BackendUnavailableError(
            f"{db_path} would not take WAL journal mode and reported {mode!r}: a "
            "rollback-journal database blocks every reader behind the writer"
        )


def open_verified(db_path: Path, root: str,
                  timeout_s: float = DEFAULT_LOCK_TIMEOUT_S) -> sqlite3.Connection:
    """Open a published database, verify it, and leave it in WAL at full synchronous; a database
    not already in WAL is converted under ``root``'s transition lock."""
    conn = sqlite3.connect(str(db_path), isolation_level=None, check_same_thread=False)
    try:
        verify_identity(conn, db_path)
        mode = str(conn.execute("pragma journal_mode").fetchone()[0]).lower()
        if mode != "wal":
            with transition_lock(root, timeout_s=timeout_s):
                mode = _set_wal(conn)
        _refuse_rollback_journal(db_path, mode)
        conn.execute("pragma synchronous = FULL")
        level = conn.execute("pragma synchronous").fetchone()[0]
        if level != _FULL:
            raise BackendUnavailableError(
                f"{db_path} reports synchronous={level} after it was set to FULL, so a "
                "committed write's durability is not what this store declares"
            )
    except BaseException:
        conn.close()
        raise
    return conn


def copy_database(source: Path, destination: Path) -> None:
    """Write a consistent copy of the database at ``source`` to ``destination`` through SQLite's
    own backup, whatever another connection is committing meanwhile."""
    src = sqlite3.connect(str(source))
    try:
        dst = sqlite3.connect(str(destination))
        try:
            src.backup(dst)
        finally:
            dst.close()
    finally:
        src.close()


def _encoded(key: Key, encode: Callable[[Any], bytes], value: Any) -> bytes:
    """``value`` through ``encode``, or a ``StoreError`` naming the entry and what would not
    encode."""
    try:
        return encode(value)
    except (TypeError, ValueError) as exc:
        raise StoreError(
            f"{key.store}{list(key.parts)} under {key.root} does not encode: {exc}. "
            f"The value is a {type(value).__name__}; convert what it holds to a JSON type "
            "at the writer."
        ) from exc


def _decoded(key: Key, data: bytes) -> Any:
    """A stored entry's value, or ``DecodeError`` naming the entry."""
    try:
        return decode_value(data)
    except ValueError as exc:
        raise DecodeError(
            f"{key.store}{list(key.parts)} under {key.root} exists but does not decode: {exc}"
        ) from exc


def _versioned(key: Key, data: bytes | None, default: Any) -> Versioned:
    """A stored record as :func:`~tcip_store.file_backend.versioned` answers it, its bytes
    decoded through :func:`_decoded`."""
    return versioned(data, default, lambda stored: _decoded(key, stored),
                     f"{key.store}{list(key.parts)} has no record under {key.root}")


class SqliteBackend:
    """The store over one SQLite database per root, one connection per process, thread and root."""

    def __init__(self, *, lock_timeout_s: float = DEFAULT_LOCK_TIMEOUT_S) -> None:
        _, timeout_error = _filelock_classes()
        self.lock_timeout_s = lock_timeout_s
        self._timeout_error = timeout_error
        self._connections: dict[tuple[int, int, str], sqlite3.Connection] = {}
        self._guard = threading.Lock()
        # How many operations are using each connection right now; a release waits on it.
        self._busy: Counter[tuple[int, int, str]] = Counter()
        self._idle = threading.Condition(self._guard)

    def close(self) -> None:
        """Close every connection this backend opened."""
        with self._guard:
            for conn in self._connections.values():
                conn.close()
            self._connections.clear()

    def release(self, root: str) -> None:
        """Close every connection this backend holds on the canonical ``root`` or on a root under
        it, on every thread, once every operation using one of them has returned; every other
        root's connections stay open and in use."""
        prefix = os.path.join(root, "")

        def under(slot: tuple[int, int, str]) -> bool:
            return slot[2] == root or slot[2].startswith(prefix)

        with self._idle:
            self._idle.wait_for(lambda: not any(under(slot) for slot in self._busy))
            for slot in [s for s in self._connections if under(s)]:
                self._connections.pop(slot).close()

    # ── database lifecycle ──────────────────────────────────────────────────────

    @contextmanager
    def _connection(
        self, root: str, keys: tuple[Key, ...] = (), timeout_s: float | None = None,
        *, create: bool,
    ) -> Generator[sqlite3.Connection | None]:
        """This process, thread and root's connection, held in use (so :meth:`release` waits for
        it) until the block returns. With no database at ``root``, a write (``create``) builds one
        and a read is answered with None.

        One connection per (pid, thread, root): a forked worker never reuses the parent's handles,
        and a write on one thread never joins another thread's read.
        """
        db_path = database_file(root)
        slot = (os.getpid(), threading.get_ident(), canonical_path(root))
        with self._guard:
            conn = self._connections.get(slot)
            if conn is not None:
                self._busy[slot] += 1
        if conn is None:
            if not db_path.is_file():
                if not create:
                    yield None
                    return
                self._publish(db_path, root, keys, timeout_s)
            conn = open_verified(db_path, root, self.lock_timeout_s)
            with self._guard:
                self._connections[slot] = conn
                self._busy[slot] += 1
        try:
            yield conn
        finally:
            with self._idle:
                self._busy[slot] -= 1
                if not self._busy[slot]:
                    del self._busy[slot]
                self._idle.notify_all()

    def _publish(self, db_path: Path, root: str, keys: tuple[Key, ...],
                 timeout_s: float | None) -> None:
        """Build a database beside its destination and install it atomically and exclusively.

        The transition lock is held across the existence re-check, the build and the install, so
        two creators serialize and the loser opens the winner's database.
        """
        timeout = self.lock_timeout_s if timeout_s is None else timeout_s
        _ensure_parent(db_path)
        started = time.monotonic()
        try:
            held = transition_lock(root, timeout_s=timeout)
            held.__enter__()
        except self._timeout_error:
            raise StoreBusyError(keys[0], time.monotonic() - started) from None
        try:
            if db_path.is_file():
                return
            temp = db_path.parent / creation_temp_name(db_path.name, uuid.uuid4().hex)
            try:
                self._build(temp)
                _fsync_path(temp)
                _install_without_clobbering(temp, db_path)
                conn = sqlite3.connect(str(db_path), isolation_level=None)
                try:
                    _refuse_rollback_journal(db_path, _set_wal(conn))
                finally:
                    conn.close()
            except BaseException:
                _remove_quietly(str(temp))
                raise
            _fsync_path(db_path)
            fsync_directory(db_path.parent)
        finally:
            held.__exit__(None, None, None)

    def _build(self, temp: Path) -> None:
        """Apply the DDL to a fresh rollback-journal (not WAL) database and close it."""
        conn = sqlite3.connect(str(temp), isolation_level=None)
        try:
            mode = conn.execute("pragma journal_mode = delete").fetchone()[0]
            if str(mode).lower() != "delete":
                raise BackendUnavailableError(
                    f"a database being created would not take rollback-journal mode and "
                    f"reported {mode!r}, so the file installed could not hold its own commits"
                )
            conn.executescript(SCHEMA_DDL)
        finally:
            conn.close()

    # ── error mapping ───────────────────────────────────────────────────────────

    @contextmanager
    def _mapped(self, keys: tuple[Key, ...]) -> Generator[None]:
        """Turn a driver error raised inside into this layer's own refusal: contention names the
        first key the call named."""
        started = time.monotonic()
        try:
            yield
        except sqlite3.Error as exc:
            text = str(exc).lower()
            contended = isinstance(exc, sqlite3.OperationalError) and (
                "locked" in text or "busy" in text
            )
            if contended and keys:
                raise StoreBusyError(keys[0], time.monotonic() - started) from exc
            root = keys[0].root if keys else "?"
            raise StoreError(
                f"the store database under {root} refused the operation: {exc}"
            ) from exc

    # ── writes ──────────────────────────────────────────────────────────────────

    @contextmanager
    def _write(
        self, keys: tuple[Key, ...], *, timeout_s: float | None = None
    ) -> Generator[sqlite3.Connection]:
        """One ``begin immediate`` transaction over the single root a call names, the write lock
        taken up front rather than upgraded from a read."""
        timeout = self.lock_timeout_s if timeout_s is None else timeout_s
        with ExitStack() as held:
            with self._mapped(keys):
                conn = held.enter_context(
                    self._connection(keys[0].root, keys, timeout, create=True))
                assert conn is not None
                conn.execute(f"pragma busy_timeout = {int(max(0.0, timeout) * 1000)}")
            with self._mapped(keys):
                conn.execute("begin immediate")
            try:
                with self._mapped(keys):
                    yield conn
                    conn.execute("commit")
            except BaseException:
                if conn.in_transaction:
                    conn.execute("rollback")
                raise

    def _put(self, conn: sqlite3.Connection, key: Key, data: bytes) -> None:
        conn.execute(
            "insert into records (store, parts, value) values (?, ?, ?) "
            "on conflict(store, parts) do update set value = excluded.value",
            (key.store, encode_parts(key.parts), data),
        )

    def _drop(self, conn: sqlite3.Connection, key: Key) -> bool:
        return conn.execute("delete from records where store = ? and parts = ?",
                            (key.store, encode_parts(key.parts))).rowcount > 0

    def _append(self, conn: sqlite3.Connection, key: Key, data: bytes) -> None:
        conn.execute("insert into log_entries (store, parts, entry) values (?, ?, ?)",
                     (key.store, encode_parts(key.parts), data))

    def _stored(self, conn: sqlite3.Connection, key: Key) -> bytes | None:
        row = conn.execute(
            "select value from records where store = ? and parts = ?",
            (key.store, encode_parts(key.parts)),
        ).fetchone()
        return None if row is None else row[0]

    # ── records ─────────────────────────────────────────────────────────────────

    def read_versioned(self, key: Key, *, default: Any = REQUIRED) -> Versioned:
        data = None
        with self._mapped((key,)), self._connection(key.root, create=False) as conn:
            if conn is not None:
                data = self._stored(conn, key)
        return _versioned(key, data, default)

    def exists(self, key: Key) -> bool:
        with self._mapped((key,)), self._connection(key.root, create=False) as conn:
            return conn is not None and self._stored(conn, key) is not None

    def replace(self, key: Key, value: Any, *, expect: Version | None = None) -> Version:
        data = _encoded(key, encode_record, value)
        with self._write((key,)) as conn:
            if expect is not None:
                require_version(key, self._stored(conn, key), expect)
            self._put(conn, key, data)
        return _version_of(data)

    def delete(self, key: Key, *, expect: Version | None = None) -> bool:
        with self._write((key,)) as conn:
            if expect is not None:
                require_version(key, self._stored(conn, key), expect)
            return self._drop(conn, key)

    @contextmanager
    def transaction(
        self, keys: Sequence[Key], *, timeout_s: float | None = None
    ) -> Generator[Txn]:
        with self._write(tuple(keys), timeout_s=timeout_s) as conn:
            yield Txn(self, conn, tuple(keys))

    def keys(self, store: str, root: str, prefix: tuple[str, ...] = ()) -> list[Key]:
        with self._mapped(()), self._connection(root, create=False) as conn:
            if conn is None:
                return []
            rows = conn.execute(
                "select parts from records where store = ? "
                "union select parts from log_entries where store = ?", (store, store)
            ).fetchall()
        found = [
            Key(store, root, parts)
            for parts in (decode_parts(text) for (text,) in rows)
            if parts[: len(prefix)] == tuple(prefix)
        ]
        return sorted(found, key=lambda k: k.parts)

    def stores(self, root: str) -> list[str]:
        """Every store name ``root``'s database holds a record or a log entry under, sorted."""
        with self._mapped(()), self._connection(root, create=False) as conn:
            if conn is None:
                return []
            rows = conn.execute(
                "select store from records union select store from log_entries order by 1"
            ).fetchall()
        return [store for (store,) in rows]

    # ── logs ────────────────────────────────────────────────────────────────────

    def append(self, key: Key, record: Mapping[str, Any]) -> None:
        data = _encoded(key, encode_log_line, record)
        with self._write((key,)) as conn:
            self._append(conn, key, data)

    def read_log(self, key: Key, *, after: str | None = None) -> LogPage:
        """Entries committed after the cursor, in commit order. The cursor is the last returned
        row's id, which autoincrement never reuses; an entry that will not decode is reported
        through ``corrupt``."""
        start = int(after) if after else 0
        with self._mapped((key,)), self._connection(key.root, create=False) as conn:
            if conn is None:
                return LogPage(records=[], cursor=str(start))
            rows = conn.execute(
                "select id, entry from log_entries where store = ? and parts = ? and id > ? "
                "order by id",
                (key.store, encode_parts(key.parts), start),
            ).fetchall()
        records: list[Mapping[str, Any]] = []
        corrupt: list[int] = []
        cursor = start
        for position, (row_id, entry) in enumerate(rows):
            try:
                records.append(_decoded(key, entry))
            except DecodeError:
                corrupt.append(position)
            cursor = row_id
        return LogPage(records=records, cursor=str(cursor), corrupt=tuple(corrupt))

    def clear_log(self, key: Key) -> int:
        """Delete every committed entry of this log, returning how many there were."""
        with self._write((key,)) as conn:
            return conn.execute(
                "delete from log_entries where store = ? and parts = ?",
                (key.store, encode_parts(key.parts)),
            ).rowcount


class Txn:
    """A transaction's handle over the keys it names: reads and writes inside one commit, a read
    seeing this transaction's own writes. A key the transaction does not name is refused."""

    def __init__(self, backend: SqliteBackend, conn: sqlite3.Connection,
                 keys: tuple[Key, ...]) -> None:
        self._backend = backend
        self._conn = conn
        self._keys = keys

    def _held(self, key: Key) -> None:
        if key not in self._keys:
            raise TransactionMisuseError(
                f"{key.store}{list(key.parts)} is not held by this transaction: name every key "
                "the body touches in transaction(...)"
            )

    def read_versioned(self, key: Key, *, default: Any = REQUIRED) -> Versioned:
        """One of the transaction's records and its version; absence answers ``default`` at
        ``Version.ABSENT`` or raises ``NotFoundError``."""
        self._held(key)
        return _versioned(key, self._backend._stored(self._conn, key), default)

    def read(self, key: Key, *, default: Any = REQUIRED) -> Any:
        """One of the transaction's records (:meth:`read_versioned`'s value)."""
        return self.read_versioned(key, default=default).value

    def write(self, key: Key, value: Any) -> Version:
        """Replace one of the transaction's records whole, returning its new version."""
        self._held(key)
        data = _encoded(key, encode_record, value)
        self._backend._put(self._conn, key, data)
        return _version_of(data)

    def delete(self, key: Key) -> None:
        """Remove one of the transaction's records."""
        self._held(key)
        self._backend._drop(self._conn, key)

    def append(self, key: Key, record: Mapping[str, Any]) -> None:
        """Append one entry to one of the transaction's logs."""
        self._held(key)
        self._backend._append(self._conn, key, _encoded(key, encode_log_line, record))
