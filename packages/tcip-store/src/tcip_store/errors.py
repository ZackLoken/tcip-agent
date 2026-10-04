"""Every refusal the storage seam raises."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tcip_store.model import Key, Version


class StoreError(Exception):
    """Base for every refusal this layer raises."""


class StoreNotBound(StoreError):
    """No backend is bound in this process."""


class BadKey(StoreError):
    """The key names no root or carries an empty part."""


class NotFound(StoreError):
    """A required read found no entry."""


class DecodeError(StoreError):
    """The entry exists but its bytes do not decode. Distinct from ``NotFound``."""


class BackendUnavailable(StoreError):
    """The backend cannot provide a guarantee it would have to declare, so it refuses to exist."""


class TransactionMisuse(StoreError):
    """A transaction was nested, named no key, spanned two roots, or was bypassed by a write
    inside it."""


class VersionConflict(StoreError):
    """The stored version is not the one the caller expected, so nothing was written."""

    def __init__(self, entry: Key | str, expected: Version, actual: Version) -> None:
        named = entry if isinstance(entry, str) else f"{entry.store}{list(entry.parts)}"
        super().__init__(
            f"{named} changed since it was read: expected version "
            f"{expected.token or '(absent)'}, found {actual.token or '(absent)'}. "
            "Re-read it and reapply the change; nothing was written."
        )
        self.entry = entry
        self.expected = expected
        self.actual = actual


class StoreBusy(StoreError):
    """A lock was not acquired within the timeout, so nothing was written. ``blocked_on`` names
    the first key or file the refused call named."""

    def __init__(self, blocked_on: Key | str, waited_s: float) -> None:
        named = (blocked_on if isinstance(blocked_on, str)
                 else f"{blocked_on.store}{list(blocked_on.parts)}")
        super().__init__(f"waited {waited_s:.1f}s for {named} and gave up; nothing was written.")
        self.blocked_on = blocked_on
        self.waited_s = waited_s
