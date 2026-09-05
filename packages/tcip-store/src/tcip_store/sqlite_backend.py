"""The SQLite backend: one WAL database per root, with blobs left as files.

Records and log entries live in ``<root>/.tcip/store.db``; blob bytes never enter a database
and are served by a composed :class:`~tcip_store.file_backend.FileBackend`, so a dataset stays
self-contained and ``blob_path`` keeps answering with a real path.

A database only ever appears complete. Creation builds it under a unique temp name in
rollback-journal mode, commits, closes, fsyncs, and installs it with a no-clobber primitive,
all under a per-root transition lock, so a crash leaves temp artifacts rather than a
half-built database and the file's existence is a completeness marker by construction. Every
open then verifies the journal mode, the synchronous level, the schema version and the schema
itself before a single row is read.

Exclusion is the database rather than the key: one writer at a time per root, readers never
blocked. That is stronger for consistency and weaker for availability than a lock per path,
and ``capabilities()`` says which of the two guarantees follow.

A root whose records are still files is refused rather than answered about: an empty database
beside a populated layout would report every one of those entries as absent. Moving them in is
``scripts/adopt_store.py``, and writing them back out again is :mod:`tcip_store.export`.

Every operation says which stores it is serving, and their layouts are what the rail reasons
about: which files under the root would be those stores' own. A directory serves whatever
stores a caller roots there, so an operation may span layouts, and each layout is checked on
its first use by each connection.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import uuid
from collections.abc import Generator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO

from tcip_store.errors import (
    BackendUnavailable,
    DecodeError,
    SchemaVersionRefused,
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
    database_file,
    transition_lock,
)
from tcip_store.layout_claims import (
    claim_of,
    claimed_files,
    contested_claimants,
    layouts_in_play,
    layouts_of,
    unconformed_files,
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
from tcip_store.registry import claim_generation, get_descriptor

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

_HELD_STORES = """
select store from records
union select store from log_entries
union select store from tombstones
union select store from store_counters where exported_counter is not null
"""


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


def database_path(root: str) -> Path:
    """Where this root's database lives, or ``BadKey`` when the root is not absolute.

    The refusal comes before canonicalization: resolving a relative root would turn a refusal
    into a cwd-dependent guess. Both the path and the refusal are the file backend's, called
    rather than restated, so the file both backends reason about is one file.
    """
    return database_file(root)


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


def verify_identity(conn: sqlite3.Connection, db_path: Path) -> None:
    """Refuse a database that is not this backend's, before a single row is read.

    The version and the schema are both reads, so this runs before anything is set on the
    connection: a file another tool wrote is refused while it is still exactly as that tool
    left it. Export, adoption and the read-only gate all come through here, so one database is
    never accepted by one of them and refused by another.

    A file at this path that is not a SQLite database at all refuses as a typed store error
    like every other database this backend will not read, rather than as the driver's own
    exception: the gates that read counters here answer for a root they could not read, and a
    raw driver error would leave them tracebacking instead of reporting.
    """
    try:
        version = conn.execute("pragma user_version").fetchone()[0]
    except sqlite3.DatabaseError as exc:
        raise StoreError(
            f"{db_path} is not a SQLite database: {exc}. Something else holds the name this "
            "backend's database has, so whether this root's state is in a database cannot be "
            "answered at all."
        ) from exc
    if version != SCHEMA_VERSION:
        raise StoreError(
            f"{db_path} carries user_version {version}, not {SCHEMA_VERSION}: it was written "
            "by another tool or another version of this backend, and reading it here would be "
            "a guess about what its rows mean"
        )
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


def set_wal(conn: sqlite3.Connection) -> str:
    """Put one connection's database into WAL, returning the mode it reports afterwards.

    Kept apart from :func:`open_verified` because the caller has to decide where this runs.
    SQLite takes an exclusive lock to change journal mode and does not consult the busy handler
    while doing it, so this returns SQLITE_BUSY at once whenever another connection has the
    database open, however long a busy timeout says to wait. It is only safe under the
    transition lock, which is the one place no second connection exists.
    """
    return str(conn.execute("pragma journal_mode = wal").fetchone()[0]).lower()


def convert_to_wal(db_path: Path) -> None:
    """Put a freshly installed database into WAL, before anything else can open it.

    Neither publication path can build a WAL database directly, because WAL keeps committed data
    in a sidecar the atomic install does not carry, so the installed file starts in rollback mode
    and something has to convert it. The one safe moment is this one: still under the transition
    lock, with the database installed and no other connection against it. Left to each opener
    instead, the conversion is a race SQLite refuses without ever consulting a busy timeout, so a
    thread arriving while another holds the database open fails at once rather than waiting.

    Every path that installs a database calls this, the backend's own publication and adoption
    alike. A database installed without it reads as unusable to the next opener that has no root
    to lock on, which is what an adopted root's second pass and its export both are.
    """
    conn = sqlite3.connect(str(db_path), isolation_level=None)
    try:
        mode = set_wal(conn)
        if mode != "wal":
            raise BackendUnavailable(
                f"{db_path} would not take WAL journal mode at publication and reported "
                f"{mode!r}: a rollback-journal database blocks every reader behind the "
                "writer, which is not the exclusion this backend declares"
            )
    finally:
        conn.close()


def open_verified(db_path: Path, root: str | None = None,
                  timeout_s: float = DEFAULT_LOCK_TIMEOUT_S) -> sqlite3.Connection:
    """Open a published database, verify it, and leave it in WAL at full synchronous.

    Identity is checked before anything is set, since switching a database to WAL is a write.

    A database this backend published is already in WAL, set once under the transition lock, so
    the usual open only reads the mode back. One that is not (a database adopted from files, or
    a publication whose transition did not complete) is converted here, and ``root`` is what
    makes that safe: the conversion runs under the same transition lock publication uses, so
    openers serialize on it instead of racing each other. Without a root there is no lock to
    take, and a non-WAL database is refused rather than converted by a racing pragma, since
    that pragma fails the moment a second connection exists and no timeout changes it.
    """
    conn = sqlite3.connect(str(db_path), isolation_level=None, check_same_thread=False)
    try:
        verify_identity(conn, db_path)
        mode = str(conn.execute("pragma journal_mode").fetchone()[0]).lower()
        if mode != "wal" and root is not None:
            with transition_lock(root, timeout_s=timeout_s):
                mode = set_wal(conn)
        if mode != "wal":
            raise BackendUnavailable(
                f"{db_path} would not take WAL journal mode and reported {mode!r}: a "
                "rollback-journal database blocks every reader behind the writer, which is "
                "not the exclusion this backend declares"
            )
        conn.execute("pragma synchronous = FULL")
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


def open_read_only(db_path: Path) -> sqlite3.Connection:
    """Open a published database for reading only, verified, without changing a byte of it.

    What a gate reading counters beside a live writer needs: a read-only handle cannot set the
    journal mode, so this deliberately does not, and it is the one open that leaves a database
    a caller does not own exactly as it found it.
    """
    if not db_path.is_file():
        raise StoreError(f"{db_path} does not exist, so there are no store counters to read")
    conn = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True, isolation_level=None)
    try:
        verify_identity(conn, db_path)
    except BaseException:
        conn.close()
        raise
    return conn


class SqliteBackend:
    """A storage backend holding records and logs in one SQLite database per root.

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
        self._checked: dict[tuple[int, int, str], tuple[set[str], int]] = {}
        self._marked: dict[str, set[str]] = {}
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
            self._checked.clear()
            self._marked.clear()
        self._files.close()

    # ── database lifecycle ──────────────────────────────────────────────────────

    def _connection(
        self,
        root: str,
        stores: Sequence[str],
        keys: tuple[Key, ...] = (),
        timeout_s: float | None = None,
    ) -> sqlite3.Connection:
        """This process, thread and root's connection, creating the database if it is absent.

        The pid is in the slot because a forked worker inherits the parent's handles and must
        not reuse them; the thread is in it because one connection per thread is what lets a
        write hold the database without another thread's read joining its transaction.

        ``stores`` is what the operation is serving, and the layouts they hang off are what
        the root is checked against, each on its first use by this connection. Reusing a
        connection is what makes the check cheap, never what lets it be skipped.
        """
        layouts = layouts_of(stores)
        db_path = database_path(root)
        slot = (os.getpid(), threading.get_ident(), canonical_path(root))
        with self._guard:
            existing = self._connections.get(slot)
        if existing is not None:
            self._verify_layouts(existing, root, layouts, slot)
            return existing
        if not db_path.is_file():
            for layout in layouts:
                self._refuse_unconformed(root, layout)
            self._publish(db_path, root, layouts, keys, timeout_s)
        conn = open_verified(db_path, root, self.lock_timeout_s)
        try:
            self._verify_layouts(conn, root, layouts, slot)
        except BaseException:
            conn.close()
            raise
        with self._guard:
            self._connections[slot] = conn
        return conn

    def require_conformed(self, root: str, stores: Sequence[str]) -> None:
        """The database backend's half of the conform rail: refuse an unconformed root, whose
        records are still files, before this backend answers about it.

        A root holding files that a claim of this operation's layout matches, and no database,
        has state this backend cannot see: creating an empty database beside those files would
        answer every read with absence, which for a confirmed negative means an annotated image
        training as empty. The answer is to move the files in deliberately, not to start beside
        them. A root holding no such files is fresh and proceeds.

        A database present is not on its own an answer either. Export legitimately writes
        claimed paths beside it, so the check becomes per store: a claimed file refuses only
        when the database has never held the store that claims it, which no export can produce
        and a store whose files arrived since adoption can.

        Nothing is remembered across the answers that can go stale. A root with no database is
        exactly the root the file backend does not refuse writes to and the one an archive
        restores into, so a "fresh" answer from a moment ago says nothing about the root now;
        the walk runs again under the transition lock at the one moment a database can come
        into being, once per layout on each connection that serves that layout, and again for
        every layout once a declared claim has joined the catalogue.
        """
        layouts = layouts_of(stores)
        if not database_path(root).is_file():
            for layout in layouts:
                self._refuse_unconformed(root, layout)
            return
        self._connection(root, stores)

    def _serving(
        self, root: str, stores: Sequence[str], keys: tuple[Key, ...] = ()
    ) -> sqlite3.Connection | None:
        """The checked connection for these stores, or None when this root holds no database.

        A read never creates a database: the root may be one nothing has written yet, and a
        reader is not the caller that decides a directory becomes a database's home.
        """
        layouts = layouts_of(stores)
        if not database_path(root).is_file():
            for layout in layouts:
                self._refuse_unconformed(root, layout)
            return None
        return self._connection(root, stores, keys)

    def _verify_layouts(
        self,
        conn: sqlite3.Connection,
        root: str,
        layouts: tuple[str, ...],
        slot: tuple[int, int, str],
    ) -> None:
        """Check this root against every layout this connection has not checked it against yet.

        A connection carries the set of layouts it has verified, so the walk is first use per
        layout rather than once per connection: a directory that serves two kinds of root is
        ordinary, and a connection opened for the first kind must still answer for the second
        before it serves it. Reuse can never skip the check, only repeat work already done.

        A claim set that has grown empties the set, because every layout's answer was derived
        from the claims in force when it was given.
        """
        with self._guard:
            verified, generation = self._checked.get(slot, (set(), claim_generation()))
        current = claim_generation()
        if current != generation:
            verified = set()
        for layout in layouts:
            if layout in verified:
                continue
            self._refuse_never_held_files(conn, root, layout)
            verified = verified | {layout}
        with self._guard:
            self._checked[slot] = (verified, current)

    def _refuse_unconformed(self, root: str, layout: str) -> None:
        """Refuse a root with no database that still holds files of this layout's stores."""
        unconformed = unconformed_files(root, layout, limit=5)
        if unconformed:
            listed = ", ".join(str(path) for path in unconformed)
            raise StoreError(
                f"{root} holds record or log files but no {DATABASE_FILENAME}, so its state "
                f"is still in the file layout: {listed}. Move it in with "
                "python scripts/adopt_store.py before this backend touches the root; an empty "
                "database beside those files would read every one of them as absent."
            )

    def _refuse_never_held_files(
        self, conn: sqlite3.Connection, root: str, layout: str
    ) -> None:
        """Refuse a claimed file beside a database that has never held the store claiming it.

        With a database present a claimed file is ordinarily that store's own export, so the
        accounting is per store and comes from the database itself: rows, tombstones or an
        export stamp all say the store has been held here. A file whose every claimant is
        unknown to this database is state that arrived outside it.

        A directory serves whatever stores a caller roots there, and two layouts' templates can
        describe one path, so the contenders are gathered across the layouts this root actually
        serves rather than only the one being checked. Contenders that disagree, one holding
        markers and another never held, are a file whose owner cannot be told from the database
        at all: that is refused naming all of them rather than resolved by picking, the same way
        the planner refuses a tie rather than attributing one store's document to another.
        """
        claimed = claimed_files(root, layout)
        if not claimed:
            return
        held = {store for (store,) in conn.execute(_HELD_STORES)}
        in_play = layouts_in_play(sorted(held), (layout,))
        stranded: list[tuple[Path, tuple[str, ...]]] = []
        ambiguous: list[tuple[Path, tuple[str, ...]]] = []
        for item in claimed:
            across = contested_claimants(root, item.path, in_play)
            dispositions = {store in held for store in across}
            if dispositions == {False}:
                stranded.append((item.path, across))
            elif len(dispositions) > 1:
                ambiguous.append((item.path, across))
        if ambiguous:
            raise self._ambiguous_claim(root, ambiguous)
        if stranded:
            listed = ", ".join(f"{path} ({'/'.join(stores)})" for path, stores in stranded[:5])
            raise StoreError(
                f"{root} holds a database that has never held the stores claiming these files, "
                f"so their state is still in the file layout beside it: {listed}. Take them in "
                "with python scripts/adopt_store.py, which loads exactly the stores this "
                "database has no record of; reading past them would answer every one of them "
                "with absence."
            )

    def _ambiguous_claim(
        self, root: str, ambiguous: list[tuple[Path, tuple[str, ...]]]
    ) -> StoreError:
        """The refusal for a file whose claimants this database cannot be read to agree about."""
        listed = ", ".join(f"{path} ({', '.join(stores)})" for path, stores in ambiguous[:5])
        return StoreError(
            f"{root} holds files more than one store could own, and this database holds state "
            f"for some of those stores and none for the others: {listed}. Which store each "
            "file belongs to cannot be told from the database, and taking it in under the wrong "
            "one would count another store's document as this one's. Move the file to a root "
            "only one of the stores hangs off, or export and conform the root deliberately."
        )

    def _guard_first_marker(self, conn: sqlite3.Connection, keys: tuple[Key, ...]) -> None:
        """Refuse a store's first write here while a file that store claims already exists.

        A store with no rows, no tombstones and no export counter has never been held in this
        database, so the per-store accounting reads any file claiming it as state left behind.
        The moment this write lands, that store holds markers and the same file starts reading
        as its own export instead, which is how a file that predates the store's arrival would
        stop being visible to the accounting. The walk is this store's own claim rather than a
        whole layout's, and it runs only while the store is unmarked, so it costs one targeted
        walk per store per database.

        Adoption sits below this: its supplement case exists to mint a never-held store's
        markers from exactly these files, under the transition lock, so it writes its rows
        through its own transaction rather than through this door.
        """
        marked = self._marked.setdefault(canonical_path(keys[0].root), set())
        unknown = sorted({key.store for key in keys} - marked)
        if not unknown:
            return
        placeholders = ", ".join("?" for _ in unknown)
        held = {
            store
            for (store,) in conn.execute(
                f"select store from ({_HELD_STORES}) where store in ({placeholders})", unknown
            )
        }
        marked |= held
        for store in unknown:
            if store in held:
                continue
            claim = claim_of(store)
            first = claimed_files(
                keys[0].root, claim.layout, claims={store: claim}, limit=1
            )
            if first:
                raise StoreError(
                    f"{first[0].path} is a file {store} owns, and this database has never held "
                    f"{store}, so this would be its first write here and the file would then "
                    "read as its own export rather than as state nothing took in. Take the file "
                    "in with python scripts/adopt_store.py, or remove it if it is not this "
                    "store's."
                )

    def _publish(
        self,
        db_path: Path,
        root: str,
        layouts: tuple[str, ...],
        keys: tuple[Key, ...] = (),
        timeout_s: float | None = None,
    ) -> None:
        """Build a database beside its destination and install it atomically and exclusively.

        The transition lock is held across the existence re-check, the build and the install,
        so two creators serialize and the loser finds the winner's database and opens it. A
        wait that runs out is this layer's own refusal, never the lock library's timeout
        escaping raw: several processes binding one root at startup all reach here at once.

        Every layout the operation serves is checked again here, under the lock, and not only
        by the caller: creation is the one moment a root's records stop being files, so a file
        that appeared since the caller looked (a restored archive, a file-backend write that
        got in first) has to be seen now or it never will be.
        """
        timeout = self.lock_timeout_s if timeout_s is None else timeout_s
        self._files._ensure_parent(db_path, durable=True)
        started = time.monotonic()
        try:
            held = transition_lock(root, timeout_s=timeout)
            held.__enter__()
        except self._timeout_error:
            raise self._contended(root, keys, time.monotonic() - started) from None
        try:
            if db_path.is_file():
                return
            for layout in layouts:
                self._refuse_unconformed(root, layout)
            temp = db_path.parent / creation_temp_name(db_path.name, uuid.uuid4().hex)
            try:
                self._build(temp)
                _fsync_path(temp)
                _install_without_clobbering(temp, db_path)
                convert_to_wal(db_path)
            except BaseException:
                _remove_quietly(str(temp))
                raise
            _fsync_path(db_path)
            self._files._fsync_dir(db_path.parent)
        finally:
            held.__exit__(None, None, None)

    def _contended(self, root: str, keys: tuple[Key, ...], waited_s: float) -> StoreError:
        """The refusal for a wait this backend gave up on, naming a key when the call named one.

        A call that named no key (an enumeration) has no key to attribute the wait to, so it is
        told which root it waited on instead of being handed a key it never asked about.
        """
        if keys:
            return StoreBusy(keys, keys[0], waited_s)
        return StoreError(
            f"waited {waited_s:.1f}s for the store database under {root} to be created by "
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
        root = keys[0].root if keys else "?"
        return StoreError(f"the store database under {root} refused the operation: {exc}")

    @contextmanager
    def _mapped(self, keys: tuple[Key, ...], waited_s: float | None = None) -> Generator[None]:
        """Turn a driver error raised inside into this layer's own refusal.

        The wait it reports is measured here rather than taken from the caller's configured
        timeout. A refusal that names the timeout instead reads as a lock held for the full
        wait, which sent two sessions looking for contention behind a failure that took
        milliseconds. A caller that already knows the true elapsed time passes it.
        """
        started = time.monotonic()
        try:
            yield
        except sqlite3.Error as exc:
            elapsed = time.monotonic() - started if waited_s is None else waited_s
            raise self._translate(exc, keys, elapsed) from exc

    def _set_busy_timeout(self, conn: sqlite3.Connection, timeout_s: float) -> None:
        conn.execute(f"pragma busy_timeout = {int(max(0.0, timeout_s) * 1000)}")

    # ── write transactions ──────────────────────────────────────────────────────

    @contextmanager
    def _write(
        self, keys: tuple[Key, ...], *, timeout_s: float | None = None
    ) -> Generator[sqlite3.Connection]:
        """One ``begin immediate`` transaction over the single root a call names.

        The write lock is taken up front rather than upgraded from a read, which is what keeps
        the snapshot-conflict class (which no busy timeout covers) out of reach. A store that
        relaxed durability commits at NORMAL; a transaction spanning both takes the stricter
        of the two, since one commit cannot be half flushed.

        A store's first marker in this database is guarded inside the transaction, so a write
        that would make a file predating it read as that store's own export is refused with
        nothing written.
        """
        timeout = self.lock_timeout_s if timeout_s is None else timeout_s
        with self._mapped(keys):
            conn = self._connection(
                keys[0].root, tuple(key.store for key in keys), keys, timeout
            )
            self._set_busy_timeout(conn, timeout)
            self._apply_synchronous(conn, self._synchronous_for(keys))
        started = time.monotonic()
        try:
            conn.execute("begin immediate")
        except sqlite3.Error as exc:
            raise self._translate(exc, keys, time.monotonic() - started) from exc
        try:
            with self._mapped(keys):
                self._guard_first_marker(conn, keys)
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
        data = None
        with self._mapped((key,)):
            conn = self._serving(key.root, (key.store,), (key,))
            if conn is not None:
                data = self._stored(conn, key)
        if data is None:
            if default is REQUIRED:
                raise _missing_record(key)
            return Versioned(default, Version.ABSENT)
        return Versioned(_decode(descriptor, key, data), _version_of(data))

    def exists(self, key: Key) -> bool:
        if get_descriptor(key.store).kind == "blob":
            return self._files.exists(key)
        with self._mapped((key,)):
            conn = self._serving(key.root, (key.store,), (key,))
            return conn is not None and self._stored(conn, key) is not None

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
    ) -> Generator["_SqliteTxn"]:
        named = tuple(keys)
        with self._write(named, timeout_s=timeout_s) as conn:
            txn = _SqliteTxn(self, conn, named)
            yield txn
            txn.apply()

    def keys(self, store: str, root: str, prefix: tuple[str, ...] = ()) -> list[Key]:
        descriptor = get_descriptor(store)
        if descriptor.kind == "blob":
            return self._files.keys(store, root, prefix)
        with self._mapped(()):
            conn = self._serving(root, (store,))
            if conn is None:
                return []
            rows = conn.execute(
                "select parts from records where store = ?", (store,)
            ).fetchall()
        found = [
            Key(store, root, parts)
            for parts in (decode_parts(text) for (text,) in rows)
            if parts[: len(prefix)] == tuple(prefix)
        ]
        return sorted(found, key=lambda k: k.parts)

    # ── logs ────────────────────────────────────────────────────────────────────

    def append(self, key: Key, record: Mapping[str, Any]) -> None:
        """Add one entry, clearing any tombstone a prior ``clear_log`` on this key left.

        A log this key names again is alive, so a pending "this key's exported file should
        be deleted" tombstone from an earlier clear is now wrong: the next export must write
        this fresh entry out as a normal file, not delete it, the same way ``_put`` already
        clears a record's own tombstone the moment it is written again.
        """
        descriptor = get_descriptor(key.store)
        data = _encode(descriptor, key, record)
        _refuse_embedded_newline(key, data)
        with self._write((key,)) as conn:
            parts = encode_parts(key.parts)
            conn.execute(
                "insert into log_entries (store, parts, entry, appended_at) values (?, ?, ?, ?)",
                (key.store, parts, data, _now()),
            )
            conn.execute(
                "delete from tombstones where store = ? and parts = ?", (key.store, parts)
            )
            self._bump(conn, key.store)

    def read_log(self, key: Key, *, after: str | None = None) -> LogPage:
        """Entries committed after the cursor, in commit order.

        The cursor is the last returned row's id, which autoincrement never reuses.
        ``torn_tail`` is structurally always False: an entry is a committed row or it is not
        there, so there is no partial tail to hold back. An entry that will not decode is
        still reported through ``corrupt`` rather than skipped, and one that decodes but
        carries a schema_version this reader does not know is reported through
        ``version_refused`` instead.
        """
        descriptor = get_descriptor(key.store)
        start = int(after) if after else 0
        with self._mapped((key,)):
            conn = self._serving(key.root, (key.store,), (key,))
            if conn is None:
                return LogPage(records=[], cursor=str(start))
            rows = conn.execute(
                "select id, entry from log_entries where store = ? and parts = ? and id > ? "
                "order by id",
                (key.store, encode_parts(key.parts), start),
            ).fetchall()
        records: list[Mapping[str, Any]] = []
        corrupt: list[int] = []
        version_refused: list[int] = []
        cursor = start
        for position, (row_id, entry) in enumerate(rows):
            try:
                records.append(_decode(descriptor, key, entry))
            except SchemaVersionRefused:
                version_refused.append(position)
            except DecodeError:
                corrupt.append(position)
            cursor = row_id
        return LogPage(
            records=records,
            cursor=str(cursor),
            corrupt=tuple(corrupt),
            version_refused=tuple(version_refused),
        )

    def clear_log(self, key: Key) -> int:
        """Delete every committed row for this log, returning how many there were.

        Tombstones the key when it held any rows, the same record ``_drop`` leaves for a
        deleted record: ``export._materialize``'s delete pass is driven by tombstones alone,
        so without one here a later export would neither rewrite this log's exported file
        (nothing left in ``log_entries`` to write) nor delete it (nothing in ``tombstones`` to
        act on), leaving stale bytes on disk that pass for current. Bumps the change counter
        only when a row was actually removed, so clearing a log nothing ever wrote does not
        manufacture a ``store_counters`` row that ``stale_stores`` would then call behind.
        """
        with self._write((key,)) as conn:
            parts = encode_parts(key.parts)
            count = conn.execute(
                "select count(*) from log_entries where store = ? and parts = ?",
                (key.store, parts),
            ).fetchone()[0]
            conn.execute(
                "delete from log_entries where store = ? and parts = ?", (key.store, parts)
            )
            if count:
                conn.execute(
                    "insert into tombstones (store, parts, deleted_at) values (?, ?, ?) "
                    "on conflict(store, parts) do update set deleted_at = excluded.deleted_at",
                    (key.store, parts, _now()),
                )
                self._bump(conn, key.store)
        return int(count)

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
