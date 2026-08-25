"""Corrupting a record already on record, the shape no writer through the seam can produce.

Several suites need a record that will not decode on whichever backend is bound, to prove a
reader treats corruption as its own kind of failure rather than as absence. ``replace`` always
encodes through a store's own codec, so producing that case means writing bytes behind an
already-written key on the exact path the bound backend reads records from.
"""

from __future__ import annotations

import os
import sqlite3

import tcip_store as ts
from tcip_store.binding import BACKEND_ENV, DEFAULT_BACKEND, FILE_BACKEND, SQLITE_BACKEND
from tcip_store.sqlite_backend import database_path, encode_parts
from tcip_store.store import _backend


def damage_record(key: ts.Key, data: bytes) -> None:
    """Put ``data`` behind a record already written at ``key``, on whichever backend is bound.

    A record must already exist at ``key``; this corrupts the bytes behind it in place, so an
    undecodable-record case is genuine on whichever backend the suite runs rather than reporting
    absence on one and corruption on the other.
    """
    name = os.environ.get(BACKEND_ENV) or DEFAULT_BACKEND
    if name == FILE_BACKEND:
        _backend().path_for(key).write_bytes(data)
        return
    if name != SQLITE_BACKEND:
        raise ValueError(f"no bytes-corruption path for backend {name!r}")
    conn = sqlite3.connect(str(database_path(str(key.root))), isolation_level=None)
    try:
        conn.execute(
            "update records set value = ? where store = ? and parts = ?",
            (data, key.store, encode_parts(key.parts)),
        )
    finally:
        conn.close()
