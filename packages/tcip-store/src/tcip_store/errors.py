"""Every refusal the storage seam raises.

A typed refusal is the whole point: the layer under it never degrades quietly. Absence and
corruption are different errors, a lost update is an error rather than a silence, and a
guarantee the bound backend cannot provide is refused instead of claimed.
"""

from __future__ import annotations

from tcip_store.model import Key, Version


class StoreError(Exception):
    """Base for every refusal this layer raises."""


class StoreNotBound(StoreError):
    """No backend is bound in this process."""


class UnknownStore(StoreError):
    """The key names a store nothing has registered."""


class WrongKind(StoreError):
    """The operation does not apply to the store's declared kind."""


class BadKey(StoreError):
    """The key does not match the store's declared shape."""


class NotFound(StoreError):
    """A required read found no entry."""


class DecodeError(StoreError):
    """The entry exists but its bytes do not decode, or a backend's own bookkeeping file
    holds bytes it cannot make sense of.

    Distinct from ``NotFound`` on purpose: an unreadable measurement record must never
    present as an absent one. The file backend also raises it out of ``append`` and
    ``clear_log``, never only out of a read, when a clear-base watermark file it needs to
    settle does not hold a decimal integer.
    """


class SchemaVersionRefused(StoreError):
    """A document's ``schema_version`` is outside what this reader's descriptor accepts.

    Deliberately not a ``DecodeError`` subclass: the bytes decoded perfectly well, and an
    unsupported version is a policy fact about a document from a newer writer, never
    corruption. A softener written to catch ``DecodeError`` must not absorb this by accident.
    """


class PolicyViolation(StoreError):
    """The write form is not one the store's concurrency policy allows."""


class ListingUnsupported(StoreError):
    """The store's descriptor declares no enumeration, so ``keys`` has no answer.

    Raised rather than returning an empty list, which would read as "none".
    """


class CapabilityUnavailable(StoreError):
    """The call requires a guarantee the bound backend does not declare."""


class BackendUnavailable(StoreError):
    """The backend cannot provide a guarantee it would have to declare, so it refuses to exist."""


class TransactionMisuse(StoreError):
    """A transaction was used in a way that means different things on different backends."""


class VersionConflict(StoreError):
    """The stored version is not the one the caller expected, so nothing was written."""

    def __init__(self, key: Key, expected: Version, actual: Version) -> None:
        super().__init__(
            f"{key.store}{list(key.parts)} changed since it was read: expected version "
            f"{expected.token or '(absent)'}, found {actual.token or '(absent)'}. "
            "Re-read the record and reapply the change; nothing was written."
        )
        self.key = key
        self.expected = expected
        self.actual = actual


class StoreBusy(StoreError):
    """A lock was not acquired within the timeout, so nothing was written.

    The file backend cannot name the process holding the lock: its lock carries no owner
    identity. It names how long it waited and, in ``blocked_on``, a key the refused call
    itself named, never the holder's. The file backend names the key whose lock it was
    waiting on; a backend excluding writers more coarsely than one key at a time names the
    first key the call named, since it cannot know which of them the contender holds.
    """

    def __init__(self, keys: tuple[Key, ...], blocked_on: Key, waited_s: float) -> None:
        requested = ", ".join(f"{k.store}{list(k.parts)}" for k in keys)
        super().__init__(
            f"waited {waited_s:.1f}s for {blocked_on.store}{list(blocked_on.parts)} and gave up; "
            f"nothing was written. Keys requested: {requested}."
        )
        self.keys = keys
        self.blocked_on = blocked_on
        self.waited_s = waited_s
