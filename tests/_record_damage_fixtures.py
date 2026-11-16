"""Corrupting a stored record or log entry, or a root's whole database.

Bytes that will not decode, or that decode and are refused, are put behind a key in the database,
or over the database file. The module also opens a root's database directly in autocommit
through :func:`connected`.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from typing import TYPE_CHECKING

import pytest

import tcip_store as ts
from tcip_store.file_backend import _DATABASE_SIDECARS, database_file
from tcip_store.sqlite_backend import encode_parts

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

NON_DATABASE = b"not a database, only bytes under its name" * 100
"""Bytes that fail the store's identity check."""

REFUSED_DOCUMENT = b'{"frame": "not a label document"}'
"""Bytes that decode as JSON and that the label-document reader refuses."""

UNDECODABLE = b"{not valid json"
"""Bytes that do not decode as JSON."""


@contextmanager
def unopenable_database(root: str | Path) -> Iterator[Path]:
    """Put :data:`NON_DATABASE` where ``root``'s database is, yield that path, and remove it after.

    The store's connections on ``root`` are released first and again on leaving, so a read inside
    the block meets the identity check rather than a held connection. The database's log and
    shared-memory index are removed where the bytes go in, so the block stands for a database that
    will not read with nothing left to recover it from, never a database replaced under a log that
    stayed. ``root``'s ``.tcip`` directory is made when it holds no database yet. After the block
    no database is there, and the next write builds one.
    """
    ts.release_root(root)
    path = database_file(str(root))
    path.parent.mkdir(parents=True, exist_ok=True)
    for name in _DATABASE_SIDECARS:
        path.with_name(name).unlink(missing_ok=True)
    path.write_bytes(NON_DATABASE)
    try:
        yield path
    finally:
        ts.release_root(root)
        path.unlink()


def connected(
    root: str | Path, *, check_same_thread: bool = True
) -> sqlite3.Connection:
    """Open ``root``'s database directly, in autocommit mode, and return the connection.

    ``check_same_thread`` is :func:`sqlite3.connect`'s own parameter: pass ``False`` for a
    connection that will be closed from a thread other than the one that opened it.
    """
    return sqlite3.connect(
        str(database_file(str(root))),
        isolation_level=None,
        check_same_thread=check_same_thread,
    )


def released_into_rollback_journal(root: str | Path) -> None:
    """Release ``root``'s store connections and put its database in rollback-journal mode.

    A WAL database refuses another connection's exclusive lock while the store holds a
    connection on it; a rollback-journal database takes it. Asserts the database answers
    ``delete`` for the mode.
    """
    ts.release_root(root)
    conn = connected(root)
    try:
        mode = conn.execute("pragma journal_mode = delete").fetchone()[0]
    finally:
        conn.close()
    assert mode == "delete"


def damage_record(key: ts.Key, data: bytes) -> None:
    """Overwrite the stored value of the record at ``key`` with ``data`` in its root's database."""
    conn = connected(key.root)
    try:
        conn.execute(
            "update records set value = ? where store = ? and parts = ?",
            (data, key.store, encode_parts(key.parts)),
        )
    finally:
        conn.close()


def made_undecodable(key: ts.Key) -> str:
    """Overwrite the record at ``key`` with :data:`UNDECODABLE`; return the store's sentence.

    The sentence is the ``DecodeError`` a direct read of ``key`` raises, which this asserts it does.
    """
    damage_record(key, UNDECODABLE)
    with pytest.raises(ts.DecodeError) as refused:
        ts.read(key)
    return str(refused.value)


def stripped_of_its_engine(key: ts.Key) -> str:
    """Store the record at ``key`` without its ``engine``; return the refusal reading it earns."""
    stored = dict(ts.read(key))
    del stored["engine"]
    ts.replace(key, stored)
    return "engine: Field required"


def damage_log_entry(key: ts.Key, position: int, data: bytes) -> None:
    """Overwrite the entry at ``position`` in the log at ``key`` with ``data``.

    Positions count from the log's first entry, and a negative one from its last, as a list's
    do. Asserts the log holds an entry at ``position``.
    """
    conn = connected(key.root)
    try:
        ids = [
            row[0]
            for row in conn.execute(
                "select id from log_entries where store = ? and parts = ? order by id",
                (key.store, encode_parts(key.parts)),
            )
        ]
        assert -len(ids) <= position < len(ids), (
            f"{key} holds {len(ids)} entries, none at position {position}"
        )
        conn.execute(
            "update log_entries set entry = ? where id = ?",
            (data, ids[position]),
        )
    finally:
        conn.close()


def damage_last_log_entry(key: ts.Key) -> str:
    """Overwrite the last entry of the log at ``key`` with bytes that will not decode.

    Returns the log's cursor past that entry, read before the damage. Asserts the log holds an
    entry.
    """
    past = ts.read_log(key).cursor
    damage_log_entry(key, -1, UNDECODABLE)
    return past
