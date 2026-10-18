"""Corrupting a stored record.

Bytes that will not decode are put behind its key in the database.
"""

from __future__ import annotations

import sqlite3

import tcip_store as ts
from tcip_store.file_backend import database_file
from tcip_store.sqlite_backend import encode_parts


def damage_record(key: ts.Key, data: bytes) -> None:
    """Overwrite the stored value of the record at ``key`` with ``data`` in its root's database."""
    conn = sqlite3.connect(
        str(database_file(str(key.root))), isolation_level=None
    )
    try:
        conn.execute(
            "update records set value = ? where store = ? and parts = ?",
            (data, key.store, encode_parts(key.parts)),
        )
    finally:
        conn.close()
