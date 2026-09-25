"""Moving a root's existing record and log files into a database, atomically or not at all.

Adoption reads every file a store's locator claims (:mod:`tcip_store.layout_claims`), decodes it
through that store's own codec, and publishes a database holding exactly those entries, stamped as
already exported. A file that will not decode, or one whose owning store cannot be told apart from
another's, refuses the whole root before anything is written. The per-root transition lock is held
from before the preflight through the install, and the built database is installed with a
no-clobber primitive. A root that already holds a database is supplemented only with the stores it
has never held.
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from tcip_store.errors import DecodeError, SchemaVersionRefused, StoreError
from tcip_store.file_backend import (
    _remove_quietly,
    creation_temp_name,
    database_file,
    fsync_directory,
    require_absolute_root,
    transition_lock,
)
from tcip_store.layout_claims import (
    ClaimedFile,
    claimed_files,
    contested_claimants,
    layouts_in_play,
)
from tcip_store.registry import get_descriptor
from tcip_store.schema_version import check_schema_version
from tcip_store.sqlite_backend import (
    SCHEMA_DDL,
    SCHEMA_VERSION,
    _fsync_path,
    _HELD_STORES,
    _install_without_clobbering,
    encode_parts,
    convert_to_wal,
    open_verified,
)


@dataclass(frozen=True)
class PlanEntry:
    """One file adoption will read, and the identity it will hold in the database."""

    store: str
    parts: tuple[str, ...]
    path: Path


@dataclass(frozen=True)
class AdoptionPlan:
    """Every entry one root contributes, and the files its layout claims but no store owns."""

    root: str
    layout: str
    entries: tuple[PlanEntry, ...]
    claimed: tuple[Path, ...]


@dataclass(frozen=True)
class AdoptionResult:
    """What one root's adoption loaded, per store."""

    root: str
    database: Path
    records: dict[str, int]
    log_entries: dict[str, int]


def plan_root(root: str, layout: str) -> AdoptionPlan:
    """Which store owns each record or log file under ``root``, or a refusal naming the tie.

    Among the stores claiming one file, the one whose template says the most about it wins, and two
    saying equally much refuses. A file whose bytes state their own key (a shard whose filename had
    a separator sanitized out of it) is held under the key the bytes state.
    """
    directory = require_absolute_root(root)
    if not directory.is_dir():
        return AdoptionPlan(root=root, layout=layout, entries=(), claimed=())
    claimed = claimed_files(root, layout)
    entries: list[PlanEntry] = []
    for item in claimed:
        winners = _winners(item)
        if not winners:
            continue
        if len(winners) > 1:
            named = ", ".join(sorted(winners))
            raise StoreError(
                f"{item.path} is claimed equally by {named}, so adopting it would attribute one "
                "store's document to another. Nothing was written. State which store owns it "
                "in the layout claims before running this again."
            )
        name = winners[0]
        parts = _parts_under(name, directory, item.path)
        if parts is None:
            continue
        entries.append(PlanEntry(store=name, parts=_true_parts(name, item.path, parts), path=item.path))
    return AdoptionPlan(
        root=root,
        layout=layout,
        entries=tuple(entries),
        claimed=tuple(item.path for item in claimed),
    )


def _refuse_ambiguous(
    root: str, layout: str, pending: tuple[PlanEntry, ...], held: set[str]
) -> None:
    """Refuse to take in a file claimed by two stores when the database holds state for one of them
    and none for the other.
    """
    in_play = layouts_in_play(sorted(held), (layout,))
    ambiguous = []
    for entry in pending:
        across = contested_claimants(root, entry.path, in_play)
        if len({store in held for store in across}) > 1:
            ambiguous.append(f"{entry.path} ({', '.join(across)})")
    if ambiguous:
        raise StoreError(
            f"these files under {root} could belong to more than one store, and the database "
            f"holds state for some of those stores and none for the others: {', '.join(ambiguous)}. "
            "Nothing was written. Move each file to a root only one of its stores hangs off, "
            "or export the database and conform the roots deliberately."
        )


def _winners(item: ClaimedFile) -> tuple[str, ...]:
    """The stores whose claim says the most about this file, which is one unless it is a tie."""
    if not item.claimants:
        return ()
    best = max(claimant.specificity for claimant in item.claimants)
    return tuple(
        sorted(claimant.store for claimant in item.claimants if claimant.specificity == best)
    )


def _parts_under(store: str, directory: Path, path: Path) -> tuple[str, ...] | None:
    """The key this store's own locator reads the file's path as, or None when it reads none."""
    descriptor = get_descriptor(store)
    locator = descriptor.locator
    if locator is None:
        return None
    relative = PurePosixPath(path.relative_to(directory).as_posix())
    parts = locator.parts_from(relative)
    if parts is None or len(parts) != len(descriptor.key_fields):
        return None
    return parts


def _true_parts(store: str, path: Path, parts: tuple[str, ...]) -> tuple[str, ...]:
    """The key the entry's own bytes state, when its layout cannot spell every key it holds."""
    recover = get_descriptor(store).true_parts_from_entry
    if recover is None:
        return parts
    recovered = recover(path.read_bytes())
    return parts if recovered is None else recovered


def unaccounted_files(plans: tuple[AdoptionPlan, ...]) -> tuple[Path, ...]:
    """Record or log files some locator claims that no plan adopts, across every root planned. A
    file that belongs to a neighboring root shows up here until that root is planned too.
    """
    adopted = {os.path.normcase(str(entry.path)) for plan in plans for entry in plan.entries}
    left: dict[str, Path] = {}
    for plan in plans:
        for path in plan.claimed:
            marker = os.path.normcase(str(path))
            if marker not in adopted:
                left[marker] = path
    return tuple(sorted(left.values()))


def adopt_root(
    root: str,
    layout: str,
    *,
    report: Callable[[str], None] = print,
) -> AdoptionResult:
    """Move this root's record and log files into its database, exclusively and atomically.

    The transition lock is taken before anything is read and held through the install. Every
    adopted file is re-checked against the size and content hash the preflight saw, immediately
    before the install, and a stale load is refused. The published file and the directory entry
    naming it are both flushed.

    A root that already holds a database is supplemented rather than rebuilt: only the stores that
    database has never held are loaded, in one transaction.
    """
    db_path = database_file(root)
    with transition_lock(root):
        if db_path.is_file():
            return _supplement(root, layout, db_path, report)
        plan = plan_root(root, layout)
        loaded = _preflight(plan)
        report(f"{root}: {len(plan.entries)} file(s) decode, building {db_path}")
        temp = db_path.parent / creation_temp_name(db_path.name, uuid.uuid4().hex)
        try:
            result = _build(temp, plan, loaded)
            _fsync_path(temp)
            _revalidate(loaded)
            _install_without_clobbering(temp, db_path)
            convert_to_wal(db_path)
        except BaseException:
            _remove_quietly(str(temp))
            raise
        _fsync_path(db_path)
        fsync_directory(db_path.parent)
    return AdoptionResult(
        root=root, database=db_path, records=result[0], log_entries=result[1]
    )


def _supplement(
    root: str, layout: str, db_path: Path, report: Callable[[str], None]
) -> AdoptionResult:
    """Load the files of the stores this root's database has never held, and nothing else.

    Held is what the database itself says: rows, a tombstone, or an export stamp. A file more than
    one store could own, where the database holds state for some of them and none for the others,
    is excluded, and refuses outright if it is one of the files this run would take in.
    """
    conn = open_verified(db_path)
    try:
        held = {store for (store,) in conn.execute(_HELD_STORES)}
        plan = plan_root(root, layout)
        pending = tuple(entry for entry in plan.entries if entry.store not in held)
        _refuse_ambiguous(root, layout, pending, held)
        plan = AdoptionPlan(root=root, layout=layout, entries=pending, claimed=plan.claimed)
        loaded = _preflight(plan)
        if not pending:
            report(f"{root}: already holds every store's state, nothing to take in")
            return AdoptionResult(root=root, database=db_path, records={}, log_entries={})
        report(f"{root}: {len(pending)} file(s) decode, loading into {db_path}")
        conn.execute("begin immediate")
        try:
            records, log_entries = _insert_entries(conn, loaded)
            _write_counters(conn, records, log_entries)
            _revalidate(loaded)
            conn.execute("commit")
        except BaseException:
            if conn.in_transaction:
                conn.execute("rollback")
            raise
    finally:
        conn.close()
    _fsync_path(db_path)
    return AdoptionResult(root=root, database=db_path, records=records, log_entries=log_entries)


@dataclass(frozen=True)
class _Loaded:
    """One adopted file's bytes as the preflight read them, with what proves they have not moved."""

    entry: PlanEntry
    data: bytes
    size: int
    digest: str


def _preflight(plan: AdoptionPlan) -> tuple[_Loaded, ...]:
    """Read and decode every file the plan adopts, refusing the whole root on the first that will
    not decode.

    A record decodes whole; a log decodes line by line. Runs the schema_version check, reporting an
    unsupported version as a plan refusal naming the version.
    """
    loaded: list[_Loaded] = []
    for entry in plan.entries:
        descriptor = get_descriptor(entry.store)
        assert descriptor.codec is not None
        data = entry.path.read_bytes()
        try:
            if descriptor.kind == "log":
                for line in _log_lines(data):
                    check_schema_version(descriptor, descriptor.codec.decode(line))
            else:
                check_schema_version(descriptor, descriptor.codec.decode(data))
        except SchemaVersionRefused as exc:
            raise DecodeError(
                f"{entry.path} is {entry.store}{list(entry.parts)} but carries a schema_version "
                f"this reader does not accept: {exc}. Nothing was written. Adopting it would "
                "put a document into the database that no reader will accept back out."
            ) from exc
        except Exception as exc:
            raise DecodeError(
                f"{entry.path} is {entry.store}{list(entry.parts)} but does not decode: {exc}. "
                f"Nothing was written. Adopting it would put bytes in the database that no "
                "reader can get back out."
            ) from exc
        loaded.append(
            _Loaded(
                entry=entry,
                data=data,
                size=len(data),
                digest=hashlib.sha256(data).hexdigest(),
            )
        )
    return tuple(loaded)


def preflight_decode(plans: tuple[AdoptionPlan, ...]) -> None:
    """Decode-check every entry across ``plans``, refusing on the first that will not decode: the
    decode :func:`adopt_root` runs, without building a database.
    """
    for plan in plans:
        _preflight(plan)


def _log_lines(data: bytes) -> tuple[bytes, ...]:
    """One log file's entries, without the terminating newlines and without an empty tail."""
    return tuple(line for line in data.split(b"\n") if line)


def _revalidate(loaded: tuple[_Loaded, ...]) -> None:
    """Refuse to publish a load that no longer matches the files it came from."""
    for item in loaded:
        current = item.entry.path.read_bytes()
        if len(current) != item.size or hashlib.sha256(current).hexdigest() != item.digest:
            raise StoreError(
                f"{item.entry.path} changed while this root was being adopted, so the database "
                "built from it is already stale. Nothing was installed; run adoption again."
            )


def _build(
    temp: Path, plan: AdoptionPlan, loaded: tuple[_Loaded, ...]
) -> tuple[dict[str, int], dict[str, int]]:
    """Load every entry into a fresh rollback-journal (not WAL) database and close it."""
    conn = sqlite3.connect(str(temp), isolation_level=None)
    try:
        mode = conn.execute("pragma journal_mode = delete").fetchone()[0]
        if str(mode).lower() != "delete":
            raise StoreError(
                f"the database being built for {plan.root} would not take rollback-journal "
                f"mode and reported {mode!r}, so the file installed could not hold its commits"
            )
        conn.executescript(SCHEMA_DDL)
        conn.execute(f"pragma user_version = {SCHEMA_VERSION}")
        conn.execute("begin immediate")
        records, log_entries = _insert_entries(conn, loaded)
        _write_counters(conn, records, log_entries)
        conn.execute("commit")
    finally:
        conn.close()
    return records, log_entries


def _insert_entries(
    conn: sqlite3.Connection, loaded: tuple[_Loaded, ...]
) -> tuple[dict[str, int], dict[str, int]]:
    """Put every read file's bytes in as rows, and say how many landed per store."""
    stamped = datetime.now(timezone.utc).isoformat()
    records: dict[str, int] = {}
    log_entries: dict[str, int] = {}
    for item in loaded:
        entry = item.entry
        parts = encode_parts(entry.parts)
        if get_descriptor(entry.store).kind == "log":
            lines = _log_lines(item.data)
            for line in lines:
                conn.execute(
                    "insert into log_entries (store, parts, entry, appended_at) "
                    "values (?, ?, ?, ?)",
                    (entry.store, parts, line, stamped),
                )
            log_entries[entry.store] = log_entries.get(entry.store, 0) + len(lines)
        else:
            conn.execute(
                "insert into records (store, parts, value, updated_at) values (?, ?, ?, ?)",
                (entry.store, parts, item.data, stamped),
            )
            records[entry.store] = records.get(entry.store, 0) + 1
    return records, log_entries


def _write_counters(
    conn: sqlite3.Connection, records: dict[str, int], log_entries: dict[str, int]
) -> None:
    """Stamp each loaded store as already exported: its files are what it was just built from."""
    for store in sorted(set(records) | set(log_entries)):
        counted = records.get(store, 0) + log_entries.get(store, 0)
        conn.execute(
            "insert into store_counters (store, change_counter, exported_counter) "
            "values (?, ?, ?) on conflict(store) do update set "
            "change_counter = excluded.change_counter, exported_counter = excluded.exported_counter",
            (store, counted, counted),
        )


__all__ = [
    "AdoptionPlan",
    "AdoptionResult",
    "PlanEntry",
    "adopt_root",
    "plan_root",
    "preflight_decode",
    "unaccounted_files",
]
