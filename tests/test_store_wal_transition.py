"""The WAL transition happens once, at publication, not as a race between openers.

SQLite takes an exclusive lock to change journal mode and does not consult the busy handler
while doing it, so a pragma that converts a database refuses the instant a second connection
holds it open, however long a busy timeout says to wait. Leaving that conversion to each opener
therefore makes concurrent first use fail rather than serialize.

These assert on the state between publication and first open, which is the only place the two
arrangements differ: once anything has opened the database, both leave it in WAL.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import pytest

from tcip_store.layout_claims import layouts_of
from tcip_store.sqlite_backend import SqliteBackend, database_path, open_verified


def _journal_mode(db: Path) -> str:
    """The journal mode a database reports, read without changing it."""
    conn = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)
    try:
        return str(conn.execute("pragma journal_mode").fetchone()[0]).lower()
    finally:
        conn.close()


def _store() -> str:
    """A real registered store, so these tests declare no layout of their own."""
    from tcip_mcp.tools import meta_tools

    return meta_tools.RETROSPECTIVE_STORE


def _publish_only(backend: SqliteBackend, root: str) -> Path:
    """Publish a root's database and hand it back without opening it for use."""
    db_path = database_path(root)
    backend._publish(db_path, root, layouts_of((_store(),)))  # noqa: SLF001 - publication is the subject
    return db_path


def _key(root: str, name: str = "only"):
    """A record key in a real registered store."""
    from tcip_store.model import Key

    return Key(_store(), root, (name,))


def test_publication_leaves_the_database_in_wal_before_anything_opens_it(tmp_path: Path) -> None:
    """The installed file is converted while the transition lock still excludes every other opener.

    The build cannot emit WAL directly, since WAL keeps committed data in a sidecar the atomic
    install does not carry, so the file arrives in rollback mode and something has to convert it.
    Doing it here is the only moment no second connection can exist. Left to the openers, the
    database sits in rollback mode until one of them wins the conversion.
    """
    backend = SqliteBackend()
    try:
        db_path = _publish_only(backend, str(tmp_path))
        assert _journal_mode(db_path) == "wal"
    finally:
        backend.close()


def test_an_opener_arriving_beside_another_connection_still_gets_a_usable_database(
    tmp_path: Path,
) -> None:
    """The flake, made deterministic: a second connection is open when an opener arrives.

    This is what sixteen barrier-released threads produce by chance. If the opener has to convert
    the journal mode, SQLite refuses it outright for as long as the other connection lives, and no
    busy timeout applies. If publication already converted it, the opener reads the mode and
    proceeds.
    """
    backend = SqliteBackend()
    root = str(tmp_path)
    try:
        db_path = _publish_only(backend, root)
        bystander = sqlite3.connect(str(db_path), isolation_level=None)
        try:
            bystander.execute("select count(*) from records").fetchone()
            conn = open_verified(db_path, root, 30.0)
            try:
                mode = str(conn.execute("pragma journal_mode").fetchone()[0]).lower()
                assert mode == "wal"
            finally:
                conn.close()
        finally:
            bystander.close()
    finally:
        backend.close()


def test_a_refusal_reports_the_wait_it_measured_rather_than_a_configured_timeout(
    tmp_path: Path,
) -> None:
    """A refusal naming the configured timeout reads as a lock held for the whole wait.

    That is what sent two sessions looking for contention behind a failure that took
    milliseconds, so the elapsed value a refusal carries has to be one this layer measured.
    """
    backend = SqliteBackend(lock_timeout_s=30.0)
    key = _key(str(tmp_path))
    try:
        with pytest.raises(Exception) as caught:  # noqa: PT011 - the refusal type is the subject
            with backend._mapped((key,)):  # noqa: SLF001 - the measurement is the subject
                time.sleep(0.25)
                raise sqlite3.OperationalError("database is locked")
        waited = getattr(caught.value, "waited_s", None)
        assert waited is not None, f"refusal carried no wait: {caught.value!r}"
        assert 0.15 <= waited < 5.0, f"reported {waited}s for a wait that lasted about 0.25s"
    finally:
        backend.close()


if __name__ == "__main__":
    pytest.main([__file__])
