"""A project's store held by another connection, read through a backend that gives up quickly.

The lock is the one a second process holding the database would take: a plain connection in
exclusive locking mode with its transaction open.
"""

from __future__ import annotations

import re
from contextlib import contextmanager
from typing import TYPE_CHECKING

import tcip_store
from tests._record_damage_fixtures import connected
from tests._store_worker import make_backend

if TYPE_CHECKING:
    import sqlite3
    from collections.abc import Iterator
    from pathlib import Path

    from tcip_store.sqlite_backend import SqliteBackend

SHORT_WAIT_S = 0.3
"""A lock wait well under the driver's own connect default of 5 s, so the two are told apart."""

GAVE_UP_WITHIN_S = 3.0
"""Above what a 0.3 s wait took (0.8 s) on a rollback-journal database, measured on the Windows
machine the store contract tests were written on, and below the driver's 5 s default."""


@contextmanager
def held_by_another_connection(
    root: str | Path,
) -> Iterator[sqlite3.Connection]:
    """Hold ``root``'s database from a plain connection in exclusive locking mode.

    The connection takes its lock with ``begin exclusive`` and gives it up when it closes, on
    leaving the block unless it was closed before.
    """
    holder = connected(root, check_same_thread=False)
    try:
        holder.execute("pragma locking_mode = exclusive")
        holder.execute("begin exclusive")
        yield holder
    finally:
        holder.close()


@contextmanager
def waiting(lock_timeout_s: float) -> Iterator[SqliteBackend]:
    """Bind a backend that waits ``lock_timeout_s`` on a lock, yield it, and close it on leaving.

    The backend is the store contract's (:func:`tests._store_worker.make_backend`). Binding
    another backend afterward is the caller's.
    """
    backend = tcip_store.bind(make_backend(lock_timeout_s=lock_timeout_s))
    try:
        yield backend
    finally:
        backend.close()


@contextmanager
def locked_by_another_holder(root: str | Path) -> Iterator[None]:
    """Hold ``root``'s database from another connection while the bound backend waits briefly.

    The bound backend's connections on ``root`` are released first, then a backend waiting
    :data:`SHORT_WAIT_S` is bound (:func:`waiting`) and the database held
    (:func:`held_by_another_connection`). On leaving, the holder and the short-wait backend are
    closed and a new default backend is bound.
    """
    tcip_store.release_root(root)
    try:
        with waiting(SHORT_WAIT_S), held_by_another_connection(root):
            yield
    finally:
        tcip_store.bind()


def seconds_given_up_after(sentence: str) -> float:
    """Return the seconds the store's busy ``sentence`` says the operation gave up after."""
    found = re.search(r"gave up after (\d+(?:\.\d+)?)s", sentence)
    assert found is not None, sentence
    return float(found.group(1))
