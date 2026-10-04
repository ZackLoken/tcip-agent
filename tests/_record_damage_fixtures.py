"""Corrupting a record already written: bytes that will not decode put behind its key in the
database."""

from __future__ import annotations

import sqlite3

import tcip_store as ts
from tcip_store.file_backend import database_file
from tcip_store.sqlite_backend import encode_parts


def damage_record(key: ts.Key, data: bytes) -> None:
    """Put ``data`` behind a record already written at ``key``, corrupting it in place."""
    conn = sqlite3.connect(str(database_file(str(key.root))), isolation_level=None)
    try:
        conn.execute(
            "update records set value = ? where store = ? and parts = ?",
            (data, key.store, encode_parts(key.parts)),
        )
    finally:
        conn.close()
