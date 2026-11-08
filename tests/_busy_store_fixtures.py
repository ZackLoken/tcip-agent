"""A project's store held by another connection, read through a backend that gives up quickly.

The lock is the one a second process holding the database would take: a plain connection in
exclusive locking mode with its transaction open.
"""

from __future__ import annotations

import re
import sqlite3
from contextlib import contextmanager
from typing import TYPE_CHECKING

import tcip_store
from tcip_store.file_backend import database_file
from tcip_store.sqlite_backend import SqliteBackend

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

SHORT_WAIT_S = 0.3
"""A lock wait well under the driver's own connect default of 5 s, so the two are told apart."""

GAVE_UP_WITHIN_S = 3.0
"""Above what a short wait was measured to take here (0.8 s for 0.3 s on a rollback-journal
database) and below the driver's 5 s default."""


@contextmanager
def held_by_another_connection(
    root: str | Path,
) -> Iterator[sqlite3.Connection]:
    """Hold ``root``'s database from a plain connection in exclusive locking mode.

    The connection takes its lock with ``begin exclusive`` and gives it up when it closes, on
    leaving the block unless it was closed before.
    """
    holder = sqlite3.connect(
        str(database_file(str(root))),
        isolation_level=None,
        check_same_thread=False,
    )
    try:
        holder.execute("pragma locking_mode = exclusive")
        holder.execute("begin exclusive")
        yield holder
    finally:
        holder.close()


@contextmanager
def short_wait_backend() -> Iterator[SqliteBackend]:
    """Bind a backend that waits :data:`SHORT_WAIT_S` on a lock, and close it on leaving.

    No other backend is bound again on leaving, so a block holds one test's own operations.
    """
    backend = tcip_store.bind(SqliteBackend(lock_timeout_s=SHORT_WAIT_S))
    try:
        yield backend
    finally:
        backend.close()


def seconds_given_up_after(sentence: str) -> float:
    """Return the seconds the store's busy ``sentence`` says the operation gave up after."""
    found = re.search(r"gave up after (\d+(?:\.\d+)?)s", sentence)
    assert found is not None, sentence
    return float(found.group(1))
