"""Corrupting a stored record or log entry.

Bytes that will not decode are put behind its key in the database.
"""

from __future__ import annotations

import sqlite3

import tcip_store as ts
from tcip_store.file_backend import database_file
from tcip_store.sqlite_backend import encode_parts


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


def damage_last_log_entry(key: ts.Key) -> str:
    """Overwrite the last entry of the log at ``key`` with bytes that will not decode.

    Returns the log's cursor past that entry, read before the damage. Asserts the log holds an
    entry.
    """
    past = ts.read_log(key).cursor
    conn = _connected(key)
    try:
        damaged = conn.execute(
            "update log_entries set entry = ? where id = ?",
            (b"{not valid json", int(past)),
        )
    finally:
        conn.close()
    assert damaged.rowcount == 1, f"{key} holds no entry to damage"
    return past
