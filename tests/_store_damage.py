"""Corrupting a record's stored bytes in place, on whichever backend is bound.

The one implementation ``test_project_tools.py``, ``test_repair_classified_predictions.py`` and
``test_deliver_per_image_counts_bucket_regime.py`` each need for a test proving a caller's
behaviour under a stamp or record that will not decode.
"""

from __future__ import annotations

import os

import tcip_store
from tcip_store.binding import BACKEND_ENV, DEFAULT_BACKEND, FILE_BACKEND, SQLITE_BACKEND


def damage_record(key: tcip_store.Key, data: bytes) -> None:
    """Put ``data`` behind a record, wherever the bound backend keeps it.

    A record must already exist at the key; this corrupts the bytes behind it in place, on the
    same path the bound backend actually reads, so the case is genuine on both backends rather
    than reporting absence on one and corruption on the other.
    """
    from tcip_store.store import _backend

    name = os.environ.get(BACKEND_ENV) or DEFAULT_BACKEND
    if name == FILE_BACKEND:
        _backend().path_for(key).write_bytes(data)
        return
    if name != SQLITE_BACKEND:
        raise ValueError(f"no bytes-corruption path for backend {name!r}")
    import sqlite3

    from tcip_store.sqlite_backend import database_path, encode_parts

    conn = sqlite3.connect(str(database_path(str(key.root))), isolation_level=None)
    try:
        conn.execute(
            "update records set value = ? where store = ? and parts = ?",
            (data, key.store, encode_parts(key.parts)),
        )
    finally:
        conn.close()
