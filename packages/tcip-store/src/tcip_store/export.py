"""Writing one root's database back out as the file layout, and saying when it is stale.

Every tool that reads TCIP's state as files rather than through the seam (the data-state
doctor, an archive, an auditor tailing a log) keeps working under a database backend because
this module puts the bytes back where each store's locator says they belong. It is the one
sanctioned writer of database-owned record and log files, and it writes through the file
backend's staging and durable-directory helpers, below the public write API where the
file-backend conform rail sits, so the rail never fires on an export by construction.

The counters are the other half: a store's ``change_counter`` moves on every write and its
``exported_counter`` only here, so a reader of the files can ask whether what it is about to
read is what the database currently holds, instead of assuming.
"""

from __future__ import annotations

import os
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from tcip_store.errors import BadKey, StoreError, UnknownStore
from tcip_store.file_backend import (
    DATABASE_FILENAME,
    FileBackend,
    database_file,
    require_absolute_root,
)
from tcip_store.registry import StoreDescriptor, get_descriptor
from tcip_store.sqlite_backend import decode_parts, encode_parts, open_read_only, open_verified


@dataclass(frozen=True)
class StoreExport:
    """What one store's export wrote, deleted, and whether the stamp landed."""

    store: str
    change_counter: int
    records_written: int
    logs_written: int
    deleted: tuple[str, ...]
    stamped: bool


@dataclass(frozen=True)
class RootExport:
    """One root's whole export, per store."""

    root: str
    database: Path
    stores: tuple[StoreExport, ...]

    @property
    def raced(self) -> tuple[str, ...]:
        """Stores whose counter moved while the files were being written, so no stamp landed.

        Not a failure: the files this export wrote are a consistent snapshot of the moment it
        read them. It means the database has moved on since, so the export is run again.
        """
        return tuple(export.store for export in self.stores if not export.stamped)


@dataclass(frozen=True)
class StoreState:
    """One store's write and export counters, as the database currently holds them."""

    store: str
    change_counter: int
    exported_counter: int | None

    @property
    def exported(self) -> bool:
        """Whether every write this store has taken is in the files."""
        return self.exported_counter == self.change_counter


@dataclass(frozen=True)
class _Collected:
    """One store's rows, grouped log entries and tombstones, read from one snapshot."""

    store: str
    descriptor: StoreDescriptor
    change_counter: int
    records: tuple[tuple[tuple[str, ...], bytes], ...]
    logs: tuple[tuple[tuple[str, ...], tuple[bytes, ...]], ...]
    tombstones: tuple[tuple[str, ...], ...]


def read_store_states(db_path: Path) -> dict[str, StoreState]:
    """Every store the database has counted a write against, opened read-only.

    A store with no row here was never written in this database, so it has nothing to export
    and a reader of its files is current by definition.
    """
    conn = open_read_only(db_path)
    try:
        rows = conn.execute(
            "select store, change_counter, exported_counter from store_counters"
        ).fetchall()
    finally:
        conn.close()
    return {store: StoreState(store, changed, exported) for store, changed, exported in rows}


def stale_stores(db_path: Path, stores: tuple[str, ...] | None = None) -> tuple[str, ...]:
    """The named stores (or every written store) whose files are behind the database, sorted.

    ``stores`` names what a particular reader reads; None asks about everything the database
    holds, which is what a bundle carrying the whole tree needs.
    """
    states = read_store_states(db_path)
    considered = states.values() if stores is None else [states[s] for s in stores if s in states]
    return tuple(sorted(state.store for state in considered if not state.exported))


def export_root(
    root: str,
    *,
    backend: FileBackend | None = None,
    report: Callable[[str], None] = print,
) -> RootExport:
    """Write one root's database-held records and logs back out as files.

    The order is fixed and each step exists for a failure it prevents. Everything is collected
    from one read snapshot, so no store's files straddle a concurrent write. Every logical
    key's target path is computed and compared before a single file changes, so a key that
    would land on another store's file is caught rather than acted on. Records and log files
    are written through the file backend's own staging, so an export's bytes reach disk exactly
    the way a file-backend write would. Deletions are driven by tombstones alone and never by
    enumerating a directory, because locator shapes collide and enumeration would sweep away a
    neighbouring store's file. The stamps come last, one short write transaction per store,
    each landing only if that store has not moved since it was read.
    """
    directory = require_absolute_root(root)
    db_path = database_file(root)
    if not db_path.is_file():
        raise StoreError(
            f"{root} holds no {DATABASE_FILENAME}, so its records are already files and there "
            "is nothing to export"
        )
    files = backend if backend is not None else FileBackend()
    conn = open_verified(db_path)
    try:
        conn.execute("begin")
        try:
            collected = _collect(conn, root)
            _refuse_colliding_targets(collected, directory, root)
            written = _materialize(collected, root, directory, files, report)
            conn.execute("commit")
        except BaseException:
            if conn.in_transaction:
                conn.execute("rollback")
            raise
        stores = tuple(_stamp(conn, item, written[item.store]) for item in collected)
    finally:
        conn.close()
    return RootExport(root=root, database=db_path, stores=stores)


def _collect(conn: sqlite3.Connection, root: str) -> tuple[_Collected, ...]:
    """Every written store's rows, log entries and tombstones, plus the counter they sit at."""
    counters = conn.execute(
        "select store, change_counter from store_counters order by store"
    ).fetchall()
    collected: list[_Collected] = []
    for store, change_counter in counters:
        try:
            descriptor = get_descriptor(store)
        except UnknownStore as exc:
            raise StoreError(
                f"{root} holds rows for store {store!r}, which nothing has registered, so "
                "there is no locator to write them back out with. Import "
                "tcip_mcp.store_catalogue, which imports the module that declares it, before "
                "exporting."
            ) from exc
        records = tuple(
            (decode_parts(parts), value)
            for parts, value in conn.execute(
                "select parts, value from records where store = ? order by parts", (store,)
            )
        )
        tombstones = tuple(
            decode_parts(parts)
            for (parts,) in conn.execute(
                "select parts from tombstones where store = ? order by parts", (store,)
            )
        )
        collected.append(
            _Collected(
                store=store,
                descriptor=descriptor,
                change_counter=change_counter,
                records=records,
                logs=_grouped_log_entries(conn, store),
                tombstones=tombstones,
            )
        )
    return tuple(collected)


def _grouped_log_entries(
    conn: sqlite3.Connection, store: str
) -> tuple[tuple[tuple[str, ...], tuple[bytes, ...]], ...]:
    """One store's log entries per log key, in the order they were committed."""
    grouped: dict[str, list[bytes]] = {}
    for parts, entry in conn.execute(
        "select parts, entry from log_entries where store = ? order by parts, id", (store,)
    ):
        grouped.setdefault(parts, []).append(entry)
    return tuple((decode_parts(parts), tuple(entries)) for parts, entries in grouped.items())


def _target(
    descriptor: StoreDescriptor, root: str, directory: Path, parts: tuple[str, ...]
) -> Path:
    """Where one logical key's bytes belong under the root."""
    locator = descriptor.locator
    if locator is None:
        raise StoreError(
            f"store {descriptor.name!r} declares no locator, so its rows cannot be written out"
        )
    relative = locator.relative_path(root, parts)
    if relative.is_absolute() or ".." in relative.parts:
        raise BadKey(f"store {descriptor.name!r} placed {list(parts)} outside its root")
    return directory.joinpath(*PurePosixPath(relative).parts)


def _refuse_colliding_targets(
    collected: tuple[_Collected, ...], directory: Path, root: str
) -> None:
    """Refuse before any file changes if two logical keys name one file.

    Thirteen stores share the ``.tcip/state`` json shape, so a mis-parted key maps cleanly onto
    another store's file. Caught here it costs an operator a message; acted on it would have
    one store's bytes overwrite another's, or a tombstone delete a live record.
    """
    claimed: dict[str, tuple[str, tuple[str, ...], str]] = {}
    for item in collected:
        logical = [(parts, "record") for parts, _ in item.records]
        logical += [(parts, "log") for parts, _ in item.logs]
        logical += [(parts, "tombstone") for parts in item.tombstones]
        for parts, sort in logical:
            path = _target(item.descriptor, root, directory, parts)
            marker = os.path.normcase(str(path))
            previous = claimed.get(marker)
            if previous is not None:
                first_store, first_parts, first_sort = previous
                raise StoreError(
                    f"exporting {root} would write {path} twice: {first_store}"
                    f"{list(first_parts)} ({first_sort}) and {item.store}{list(parts)} "
                    f"({sort}) both land there. Nothing was written. One of the two keys is "
                    "mis-parted; fix it in the database before exporting."
                )
            claimed[marker] = (item.store, parts, sort)


def _materialize(
    collected: tuple[_Collected, ...],
    root: str,
    directory: Path,
    files: FileBackend,
    report: Callable[[str], None],
) -> dict[str, tuple[int, int, tuple[str, ...]]]:
    """Write every record and log file, delete every tombstoned path, and say what was deleted."""
    written: dict[str, tuple[int, int, tuple[str, ...]]] = {}
    for item in collected:
        durable = item.descriptor.durable
        for parts, value in item.records:
            _write_file(files, _target(item.descriptor, root, directory, parts), value, durable)
        for parts, entries in item.logs:
            data = b"".join(entry + b"\n" for entry in entries)
            _write_file(files, _target(item.descriptor, root, directory, parts), data, durable)
        deleted: list[str] = []
        for parts in item.tombstones:
            path = _target(item.descriptor, root, directory, parts)
            if path.is_file():
                files._remove_entry(path, durable=durable)
                deleted.append(str(path))
                report(f"deleted {path} ({item.store}{list(parts)} was deleted in the database)")
        written[item.store] = (len(item.records), len(item.logs), tuple(deleted))
    return written


def _write_file(files: FileBackend, path: Path, data: bytes, durable: bool) -> None:
    """Put exactly these bytes at this path, the way a file-backend write would."""
    files._ensure_parent(path, durable=durable)
    temp = files._stage_bytes(path, data, durable=durable)
    files._apply_staged(temp, path, durable=durable)


def _stamp(
    conn: sqlite3.Connection, item: _Collected, written: tuple[int, int, tuple[str, ...]]
) -> StoreExport:
    """Record that this store's files are current, unless it moved while they were written.

    Its own short write transaction, never the read snapshot the files came from: a snapshot
    upgraded to a write raises a conflict class no busy timeout covers. The stamp and the
    tombstone prune are bookkeeping and bump no counter, so an export cannot invalidate itself.
    """
    records_written, logs_written, deleted = written
    conn.execute("begin immediate")
    try:
        row = conn.execute(
            "select change_counter from store_counters where store = ?", (item.store,)
        ).fetchone()
        stamped = row is not None and row[0] == item.change_counter
        if stamped:
            conn.execute(
                "update store_counters set exported_counter = ? where store = ?",
                (item.change_counter, item.store),
            )
            for parts in item.tombstones:
                conn.execute(
                    "delete from tombstones where store = ? and parts = ?",
                    (item.store, encode_parts(parts)),
                )
        conn.execute("commit" if stamped else "rollback")
    except BaseException:
        if conn.in_transaction:
            conn.execute("rollback")
        raise
    return StoreExport(
        store=item.store,
        change_counter=item.change_counter,
        records_written=records_written,
        logs_written=logs_written,
        deleted=deleted,
        stamped=stamped,
    )


__all__ = [
    "RootExport",
    "StoreExport",
    "StoreState",
    "export_root",
    "read_store_states",
    "stale_stores",
]
