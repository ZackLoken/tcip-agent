"""Moving a scope's existing record and log files into a database, atomically or not at all.

The file layout came first, so the first database a scope ever sees must be built from what is
already on disk rather than beside it. Adoption reads every file a store's locator claims,
decodes it through that store's own codec, and publishes a database holding exactly those
entries, stamped as already exported so a file-reading tool is current the moment adoption
returns.

Two rules make it safe to run against live state. It refuses before it writes: a file that
will not decode, or one whose owning store cannot be told apart from another's, stops the whole
scope. And it publishes exclusively: the per-scope transition lock is held from before the
preflight through the install, and the built database is installed with a no-clobber primitive,
so a crash leaves temp artifacts rather than a half-loaded database.

Which entries a store owns in a scope is the one thing a locator cannot answer on its own,
because locator shapes collide: thirteen stores place a single json document under
``.tcip/state``. The caller supplies that inventory as :class:`StoreSource` patterns, stating
per store which parts are constants and which vary.
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from tcip_store.errors import DecodeError, StoreError
from tcip_store.file_backend import (
    _is_bookkeeping,
    _remove_quietly,
    claimed_record_files,
    creation_temp_name,
    database_file,
    fsync_directory,
    require_absolute_scope,
    transition_lock,
)
from tcip_store.registry import StoreDescriptor, get_descriptor
from tcip_store.sqlite_backend import (
    SCHEMA_DDL,
    SCHEMA_VERSION,
    _fsync_path,
    _install_without_clobbering,
    encode_parts,
)


@dataclass(frozen=True)
class PartPattern:
    """What one part of a store's key looks like across every entry the store holds.

    A part is either a constant the store's key constructor spells (``literal``), a varying
    value with a fixed opening the constructor puts there (``starts_with``), or free. The
    three are ordered by how much they say, which is how a file two stores' locators both
    claim is attributed to the store that says more about it.
    """

    literal: str | None = None
    starts_with: str = ""

    def matches(self, part: str) -> bool:
        """Whether this part could belong to the store this pattern describes."""
        if self.literal is not None:
            return part == self.literal
        return part.startswith(self.starts_with)

    @property
    def specificity(self) -> int:
        """How much the pattern constrains, for choosing between two stores claiming one file."""
        if self.literal is not None:
            return 2
        return 1 if self.starts_with else 0


ANY = PartPattern()
"""A part whose value varies with no fixed opening."""


def literal(text: str) -> PartPattern:
    """A part that is the same constant in every entry of the store."""
    return PartPattern(literal=text)


def starting_with(text: str) -> PartPattern:
    """A part that varies but always opens with ``text``."""
    return PartPattern(starts_with=text)


@dataclass(frozen=True)
class StoreSource:
    """Where one store's entries are found: the kind of scope it hangs off and its key shape.

    ``layout`` names the kind of directory the store's scope is, since a locator's relative
    path is only meaningful under the scope kind it was written for: the same
    ``<dir>/<name>.json`` shape addresses a split's stem list under a split directory and an
    evaluation's results under a run directory.
    """

    layout: str
    parts: tuple[PartPattern, ...]


@dataclass(frozen=True)
class PlanEntry:
    """One file adoption will read, and the identity it will hold in the database."""

    store: str
    parts: tuple[str, ...]
    path: Path


@dataclass(frozen=True)
class AdoptionPlan:
    """Every entry one scope contributes, and the files its layout claims but no store owns."""

    scope: str
    layout: str
    entries: tuple[PlanEntry, ...]
    claimed: tuple[Path, ...]


@dataclass(frozen=True)
class AdoptionResult:
    """What one scope's adoption loaded, per store."""

    scope: str
    database: Path
    records: dict[str, int]
    log_entries: dict[str, int]


def plan_scope(scope: str, layout: str, sources: Mapping[str, StoreSource]) -> AdoptionPlan:
    """Which store owns each record or log file under ``scope``, or a refusal naming the tie.

    Every store declared for this layout offers its locator's reading of a file; the store
    whose patterns say the most about the recovered parts wins, and two stores saying equally
    much is a refusal rather than a coin toss. A file whose bytes state their own key (a shard
    whose filename had a separator sanitized out of it) is held under the key the bytes state,
    through the same recovery hook enumeration uses, so adoption and ``keys`` cannot disagree.
    """
    root = require_absolute_scope(scope)
    here = {name: source for name, source in sources.items() if source.layout == layout}
    entries: list[PlanEntry] = []
    if not root.is_dir():
        return AdoptionPlan(scope=scope, layout=layout, entries=(), claimed=())
    for path in sorted(root.rglob("*")):
        if not path.is_file() or _is_bookkeeping(path.name):
            continue
        relative = PurePosixPath(path.relative_to(root).as_posix())
        best: list[tuple[str, tuple[str, ...]]] = []
        best_score = -1
        for name, source in here.items():
            descriptor = get_descriptor(name)
            parts = _claimed_parts(descriptor, source, relative)
            if parts is None:
                continue
            score = sum(pattern.specificity for pattern in source.parts)
            if score > best_score:
                best_score, best = score, [(name, parts)]
            elif score == best_score:
                best.append((name, parts))
        if not best:
            continue
        if len(best) > 1:
            named = ", ".join(sorted(name for name, _ in best))
            raise StoreError(
                f"{path} is claimed equally by {named}, so adopting it would attribute one "
                "store's document to another. Nothing was written. State which store owns it "
                "in the adoption inventory before running this again."
            )
        name, parts = best[0]
        entries.append(PlanEntry(store=name, parts=_true_parts(name, path, parts), path=path))
    return AdoptionPlan(
        scope=scope,
        layout=layout,
        entries=tuple(entries),
        claimed=claimed_record_files(scope),
    )


def _claimed_parts(
    descriptor: StoreDescriptor, source: StoreSource, relative: PurePosixPath
) -> tuple[str, ...] | None:
    """The parts this store would hold the file under, or None when the file is not its."""
    locator = descriptor.locator
    if locator is None:
        return None
    parts = locator.parts_from(relative)
    if parts is None or len(parts) != len(descriptor.key_fields):
        return None
    if len(parts) != len(source.parts):
        return None
    if not all(pattern.matches(part) for pattern, part in zip(source.parts, parts, strict=True)):
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
    """Record or log files some locator claims that no plan adopts, across every scope planned.

    A file left behind would read as absent under a database backend while still sitting on
    disk, which for a confirmed negative means an annotated image training as empty. A file
    that belongs to a neighbouring scope shows up here until that scope is planned too.
    """
    adopted = {os.path.normcase(str(entry.path)) for plan in plans for entry in plan.entries}
    left: dict[str, Path] = {}
    for plan in plans:
        for path in plan.claimed:
            marker = os.path.normcase(str(path))
            if marker not in adopted:
                left[marker] = path
    return tuple(sorted(left.values()))


def adopt_scope(
    scope: str,
    layout: str,
    sources: Mapping[str, StoreSource],
    *,
    report: Callable[[str], None] = print,
) -> AdoptionResult:
    """Build this scope's database from the files it already holds, exclusively and atomically.

    The transition lock is taken before anything is read and held through the install, so no
    write can land in the layout between the decode and the publication. Every adopted file is
    re-checked against the size and content hash the preflight saw, immediately before the
    install, so a load that went stale under a writer that got in first is refused rather than
    published. The published file and the directory entry naming it are both flushed, so the
    database a crash leaves behind is either absent or complete.
    """
    db_path = database_file(scope)
    with transition_lock(scope):
        if db_path.is_file():
            raise StoreError(
                f"{db_path} already exists, so this scope's records are already in a database "
                "and adopting again would build one from files it no longer owns"
            )
        plan = plan_scope(scope, layout, sources)
        loaded = _preflight(plan)
        report(f"{scope}: {len(plan.entries)} file(s) decode, building {db_path}")
        temp = db_path.parent / creation_temp_name(db_path.name, uuid.uuid4().hex)
        try:
            result = _build(temp, plan, loaded)
            _fsync_path(temp)
            _revalidate(loaded)
            _install_without_clobbering(temp, db_path)
        except BaseException:
            _remove_quietly(str(temp))
            raise
        _fsync_path(db_path)
        fsync_directory(db_path.parent)
    return AdoptionResult(
        scope=scope, database=db_path, records=result[0], log_entries=result[1]
    )


@dataclass(frozen=True)
class _Loaded:
    """One adopted file's bytes as the preflight read them, with what proves they have not moved."""

    entry: PlanEntry
    data: bytes
    size: int
    digest: str


def _preflight(plan: AdoptionPlan) -> tuple[_Loaded, ...]:
    """Read and decode every file the plan adopts, refusing the whole scope on the first that
    will not decode.

    A record decodes whole; a log decodes line by line, because one unreadable line in an
    append-only file is the entry that would silently vanish from the database.
    """
    loaded: list[_Loaded] = []
    for entry in plan.entries:
        descriptor = get_descriptor(entry.store)
        assert descriptor.codec is not None
        data = entry.path.read_bytes()
        try:
            if descriptor.kind == "log":
                for line in _log_lines(data):
                    descriptor.codec.decode(line)
            else:
                descriptor.codec.decode(data)
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


def _log_lines(data: bytes) -> tuple[bytes, ...]:
    """One log file's entries, without the terminating newlines and without an empty tail."""
    return tuple(line for line in data.split(b"\n") if line)


def _revalidate(loaded: tuple[_Loaded, ...]) -> None:
    """Refuse to publish a load that no longer matches the files it came from."""
    for item in loaded:
        current = item.entry.path.read_bytes()
        if len(current) != item.size or hashlib.sha256(current).hexdigest() != item.digest:
            raise StoreError(
                f"{item.entry.path} changed while this scope was being adopted, so the database "
                "built from it is already stale. Nothing was installed; run adoption again."
            )


def _build(
    temp: Path, plan: AdoptionPlan, loaded: tuple[_Loaded, ...]
) -> tuple[dict[str, int], dict[str, int]]:
    """Load every entry into a fresh rollback-journal database and close it.

    Never WAL while building: a WAL database holds its commits in a sidecar the install does
    not carry, so the file published would be missing the rows it was just given.
    """
    stamped = datetime.now(timezone.utc).isoformat()
    records: dict[str, int] = {}
    log_entries: dict[str, int] = {}
    conn = sqlite3.connect(str(temp), isolation_level=None)
    try:
        mode = conn.execute("pragma journal_mode = delete").fetchone()[0]
        if str(mode).lower() != "delete":
            raise StoreError(
                f"the database being built for {plan.scope} would not take rollback-journal "
                f"mode and reported {mode!r}, so the file installed could not hold its commits"
            )
        conn.executescript(SCHEMA_DDL)
        conn.execute(f"pragma user_version = {SCHEMA_VERSION}")
        conn.execute("begin immediate")
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
        for store in sorted(set(records) | set(log_entries)):
            counted = records.get(store, 0) + log_entries.get(store, 0)
            conn.execute(
                "insert into store_counters (store, change_counter, exported_counter) "
                "values (?, ?, ?)",
                (store, counted, counted),
            )
        conn.execute("commit")
    finally:
        conn.close()
    return records, log_entries


__all__ = [
    "ANY",
    "AdoptionPlan",
    "AdoptionResult",
    "PartPattern",
    "PlanEntry",
    "StoreSource",
    "adopt_scope",
    "literal",
    "plan_scope",
    "starting_with",
    "unaccounted_files",
]
