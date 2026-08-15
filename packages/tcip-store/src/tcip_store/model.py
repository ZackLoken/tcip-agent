"""Identity and value types the storage seam speaks, identical on every backend.

Nothing here names a filesystem path: mapping an identity onto storage is a backend's
private job, so a record can move from files to a database without a consumer changing.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar


@dataclass(frozen=True)
class Key:
    """The identity of one record, log, or blob: which store, which root, which entry.

    ``store`` names a registered store. ``scope`` is the root that store's descriptor says
    the entry hangs off, as an opaque string: a dataset root for stores that travel with the
    data, a platform or project root for platform state, a sweep root for HPO trial state.
    ``parts`` is the identity inside the store, ordered coarse to fine, so a prefix of it is
    a meaningful scan.

    Each store's owning module publishes a named constructor beside its existing path
    resolver, so a key's shape is stated once next to the store's declaration and importing
    the constructor is what guarantees the store is registered. A hand-built key that passes
    the descriptor's validation is indistinguishable from constructor output; the store
    layer validates shape, not provenance.
    """

    store: str
    scope: str
    parts: tuple[str, ...] = ()


@dataclass(frozen=True)
class Version:
    """What the caller believes is currently stored, as an opaque token.

    Obtained from a read and only ever echoed back into ``replace(expect=...)`` or
    ``delete(expect=...)``. ``Version.ABSENT`` asserts that no entry exists yet, which is
    how a create-only write is expressed. Tokens are derived from the stored content on
    every backend, so a byte-identical rewrite leaves a held token valid.
    """

    token: str

    ABSENT: ClassVar["Version"]


Version.ABSENT = Version("")


@dataclass(frozen=True)
class Versioned:
    """A value and its version, read together so the pair cannot straddle a concurrent write."""

    value: Any
    version: Version


@dataclass(frozen=True)
class LogPage:
    """Entries read from a log, the cursor to resume from, and what was not returned.

    ``torn_tail`` is true when the last bytes in the log are a partial entry left by an
    in-flight appender: those bytes are excluded and ``cursor`` does not advance past them,
    so a later read picks the entry up once it is complete. ``corrupt`` holds the positions
    of undecodable entries that are not the tail, counted over every entry encountered in
    this page including the undecodable ones, so entry 1 of (good, bad, good) is reported
    while ``records`` holds the two that decoded. An undecodable entry is reported rather
    than skipped: on a measurement platform, a metrics stream that drops a row and one that
    says it dropped a row are different things.
    """

    records: list[Mapping[str, Any]]
    cursor: str
    torn_tail: bool = False
    corrupt: tuple[int, ...] = ()


@dataclass(frozen=True)
class Capabilities:
    """What the bound backend actually guarantees, so a caller refuses rather than degrades.

    ``multi_key_atomic_commit``: a transaction's staged writes land all-or-nothing.
    ``cross_machine_exclusion``: the lock excludes writers on another machine.
    ``durable_replace``: a returned durable write survives a power loss, rename included.
    ``durable_append``: an append that returned survives a power loss.
    ``local_blob_paths``: ``blob_path`` is answerable without a download.
    """

    multi_key_atomic_commit: bool
    cross_machine_exclusion: bool
    durable_replace: bool
    durable_append: bool
    local_blob_paths: bool


class _Required:
    """The sentinel distinguishing "no default given" from a default of ``None``."""

    def __repr__(self) -> str:
        return "REQUIRED"


REQUIRED: Any = _Required()
"""Passed as ``default`` to mean the entry is required: absence raises ``NotFound``."""


def canonical_order(keys: tuple[Key, ...]) -> tuple[Key, ...]:
    """Order keys deterministically, so two callers naming the same set cannot deadlock.

    The order is derived from the keys themselves, not from the order they were named, and
    it is the order locks are acquired in. It is not the order a transaction applies its
    writes in: that is the declared order, which callers depend on.
    """
    return tuple(sorted(keys, key=lambda k: (k.store, k.scope, k.parts)))
