"""Identity and value types the storage seam speaks.

A key's root is the one string that holds a directory path, and ``canonical_path`` decides when
two spellings of it name one directory.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from tcip_store.errors import BadKeyError, TransactionMisuseError

_open_transaction = threading.local()


def refuse_inside_transaction(operation: str) -> None:
    """Refuse (``TransactionMisuseError``) ``operation`` while this thread holds a transaction,
    since it would commit apart from it: a record or log write, or a file write."""
    if getattr(_open_transaction, "held", False):
        raise TransactionMisuseError(
            f"{operation} is not allowed inside an open transaction: use the transaction's own "
            "operations on the keys it names, name every key in one transaction(a, b) rather "
            "than nesting, and write a file only once the transaction has committed"
        )


@contextmanager
def held_transaction() -> Generator[None]:
    """Mark this thread as holding a transaction for the body. Refuses
    (``TransactionMisuseError``) a second one on the same thread."""
    refuse_inside_transaction("a second transaction")
    _open_transaction.held = True
    try:
        yield
    finally:
        _open_transaction.held = False


@dataclass(frozen=True)
class Key:
    """The identity of one record or log: which store, which root, which entry.

    ``root`` is the directory whose database holds the entry. ``parts`` is the identity inside the
    store, ordered coarse to fine, so a prefix of it is a meaningful scan. Refuses (``BadKeyError``)
    a key with no root, or with a store or a part that is not a non-empty string.
    """

    store: str
    root: str
    parts: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.store, str) or not self.store:
            raise BadKeyError(f"a key's store must be a non-empty string; got {self.store!r}")
        if not self.root:
            raise BadKeyError(
                f"{self.store!r} key carries no root: name the root the entry hangs off")
        for part in self.parts:
            if not isinstance(part, str) or not part:
                raise BadKeyError(
                    f"{self.store!r} key part must be a non-empty string; got {part!r}")


@dataclass(frozen=True)
class Version:
    """What the caller believes is currently stored, as an opaque token derived from the stored
    bytes. ``Version.ABSENT`` asserts that no entry exists yet, which is how a create-only write is
    expressed."""

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
    """Entries read from a log, the cursor to resume from, and the positions of the entries in
    this page that would not decode, which ``records`` leaves out."""

    records: list[Mapping[str, Any]]
    cursor: str
    corrupt: tuple[int, ...] = ()


class _Required:
    """The sentinel distinguishing "no default given" from a default of ``None``."""

    def __repr__(self) -> str:
        return "REQUIRED"


REQUIRED: Any = _Required()
"""Passed as ``default`` to mean the entry is required: absence raises ``NotFoundError``."""


def canonical_path(path: str | Path) -> str:
    """One spelling for a filesystem path, so two spellings of one directory compare equal.

    Resolution collapses the relative segments and the links; the case rule is the platform's own.
    A relative path resolves against the process's current directory, so a caller that must refuse
    one refuses before canonicalizing.
    """
    return os.path.normcase(str(Path(path).resolve()))
