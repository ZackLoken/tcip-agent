"""Corrupting a stored record or log entry, or a root's whole database.

Bytes that will not decode, or that decode and are refused, are put behind a key in the database,
or over the database file.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from typing import TYPE_CHECKING

import tcip_store as ts
from tcip_store.file_backend import database_file
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
    the block meets the identity check rather than a held connection. ``root``'s ``.tcip``
    directory is made when it holds no database yet. After the block no database is there, and the
    next write builds one.
    """
    ts.release_root(root)
    path = database_file(str(root))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(NON_DATABASE)
    try:
        yield path
    finally:
        ts.release_root(root)
        path.unlink()


def _connected(key: ts.Key) -> sqlite3.Connection:
    """Open the database of ``key``'s root directly, in autocommit mode."""
    return sqlite3.connect(
        str(database_file(str(key.root))), isolation_level=None
    )


def damage_record(key: ts.Key, data: bytes) -> None:
    """Overwrite the stored value of the record at ``key`` with ``data`` in its root's database."""
    conn = _connected(key)
    try:
        conn.execute(
            "update records set value = ? where store = ? and parts = ?",
            (data, key.store, encode_parts(key.parts)),
        )
    finally:
        conn.close()


def damage_log_entry(key: ts.Key, position: int, data: bytes) -> None:
    """Overwrite the entry at ``position`` in the log at ``key`` with ``data``.

    Positions count from the log's first entry, and a negative one from its last, as a list's
    do. Asserts the log holds an entry at ``position``.
    """
    conn = _connected(key)
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
