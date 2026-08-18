"""What the export writes, and what it refuses to write.

The database backend is only allowed to be the default because the file layout survives it, so
the subject here is fidelity: the same writes made through the file backend and through the
database plus an export must leave the same files, byte for byte, including the file a delete
removes. Everything else guards a way that fidelity could be lost quietly: a key mis-parted
onto another store's file, a log welded back together in the wrong order, or a stamp claiming
files are current after the database moved on underneath them.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path

import pytest

import tcip_store as ts
from tcip_store import export as store_export
from tcip_store.file_backend import FileBackend, RootedFileLocator, _is_bookkeeping
from tcip_store.sqlite_backend import SqliteBackend, database_path, encode_parts
from tests._store_worker import LOG, LWW, register_contract_stores, document_claim

register_contract_stores()

STATE_ALPHA = "export_state_alpha"
STATE_BETA = "export_state_beta"
_SHARED_SHAPE = RootedFileLocator(prefix=("documents",), suffix=".json")
"""One locator handed to two stores, so their files are indistinguishable by shape.

That is the property the export has to survive, and it is why the export cannot trust a
directory: thirteen shipped stores place a single document under ``.tcip/state`` the same way.
This fixture keeps the shape and puts it in a directory of the suite's own.
"""

_declared = False


def _register_export_stores() -> None:
    """Two stores whose files are indistinguishable by shape, declared once per process."""
    global _declared
    if _declared:
        return
    _declared = True
    for name in (STATE_ALPHA, STATE_BETA):
        ts.register_store(
            ts.StoreDescriptor(
                name=name,
                kind="record",
                key_fields=("document",),
                codec=ts.RECORD_JSON,
                concurrency="last_writer_wins",
                locator=_SHARED_SHAPE,
                claim=document_claim(),
            )
        )


_register_export_stores()


@contextmanager
def bound(backend):
    """Bind one backend for a block, so a case can write the same values through each in turn."""
    ts.bind(backend)
    try:
        yield backend
    finally:
        ts.unbind()
        backend.close()


@pytest.fixture
def database(tmp_path):
    """A root written through the database backend, which the export writes back out."""
    root = tmp_path / "written_as_rows"
    root.mkdir()
    backend = SqliteBackend()
    ts.bind(backend)
    try:
        yield root
    finally:
        ts.unbind()
        backend.close()


def _key(store: str, root: Path, *parts: str) -> ts.Key:
    return ts.Key(store, str(root), parts)


def _entries(root: Path) -> dict[str, bytes]:
    """Every file under a root that is an entry rather than a backend's own bookkeeping."""
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file() and not _is_bookkeeping(path.name)
    }


def _write_the_value_set(root: Path) -> None:
    """The same writes on either backend: two stores sharing a shape, a nested key, and a log."""
    ts.replace(_key(STATE_ALPHA, root, "image_status"), {"a_1.jpg": "negative", "ü": "complete"})
    ts.replace(_key(STATE_BETA, root, "gui"), {"active_tab": "annotate"})
    ts.replace(_key(LWW, root, "kept"), {"n": 1})
    ts.replace(_key(LWW, root, "removed"), {"n": 2})
    for epoch in (1, 2, 3):
        ts.append(_key(LOG, root, "metrics"), {"epoch": epoch, "loss": 0.5 / epoch, "note": "ü"})


def test_the_files_an_export_writes_are_the_files_the_file_backend_would_have_written(tmp_path):
    """The whole produced file set, not one file at a time.

    Comparing per file would pass an export that wrote every expected file correctly and one
    extra beside them, which is exactly what a delete driven by directory enumeration would
    leave behind.
    """
    files = tmp_path / "written_as_files"
    rows = tmp_path / "written_as_rows"
    files.mkdir()
    rows.mkdir()

    with bound(FileBackend()):
        _write_the_value_set(files)
        ts.delete(_key(LWW, files, "removed"))

    with bound(SqliteBackend()):
        _write_the_value_set(rows)
        store_export.export_root(str(rows), report=lambda line: None)
        ts.delete(_key(LWW, rows, "removed"))
        exported = store_export.export_root(str(rows), report=lambda line: None)

    assert _entries(rows) == _entries(files)
    assert not exported.raced


def test_a_log_is_reassembled_one_entry_per_line_in_the_order_it_was_appended(tmp_path):
    """A metrics stream read as files is read line by line, so the order and the newline are the
    content, not a formatting choice."""
    files = tmp_path / "written_as_files"
    rows = tmp_path / "written_as_rows"
    files.mkdir()
    rows.mkdir()

    with bound(FileBackend()):
        for epoch in (1, 2, 3):
            ts.append(_key(LOG, files, "metrics"), {"epoch": epoch})

    with bound(SqliteBackend()):
        for epoch in (1, 2, 3):
            ts.append(_key(LOG, rows, "metrics"), {"epoch": epoch})
        store_export.export_root(str(rows), report=lambda line: None)

    written = (rows / "logs" / "metrics.jsonl").read_bytes()
    assert written == (files / "logs" / "metrics.jsonl").read_bytes()
    assert written.decode().splitlines() == [
        '{"epoch": 1}', '{"epoch": 2}', '{"epoch": 3}'
    ]


def test_two_keys_that_would_land_on_one_file_refuse_the_export_before_it_writes_anything(
    database
):
    """Two stores share a locator shape, so a mis-parted key maps cleanly onto another store's
    document. Acted on, one store's bytes would silently become the other's."""
    ts.replace(_key(STATE_ALPHA, database, "shared"), {"owner": "alpha"})
    ts.replace(_key(STATE_BETA, database, "shared"), {"owner": "beta"})

    with pytest.raises(ts.StoreError) as raised:
        store_export.export_root(str(database), report=lambda line: None)

    message = str(raised.value)
    assert STATE_ALPHA in message and STATE_BETA in message and "shared" in message
    assert not (database / "documents" / "shared.json").exists()


def test_an_export_whose_keys_are_distinct_writes_both_stores_files(database):
    """The refusal above is about one file claimed twice, not about two stores sharing a shape:
    the shape is what thirteen shipped stores do."""
    ts.replace(_key(STATE_ALPHA, database, "image_status"), {"owner": "alpha"})
    ts.replace(_key(STATE_BETA, database, "gui"), {"owner": "beta"})

    exported = store_export.export_root(str(database), report=lambda line: None)

    documents = database / "documents"
    assert ts.RECORD_JSON.decode((documents / "image_status.json").read_bytes()) == {"owner": "alpha"}
    assert ts.RECORD_JSON.decode((documents / "gui.json").read_bytes()) == {"owner": "beta"}
    assert not exported.raced


def test_a_store_written_again_while_its_files_were_being_written_is_reported_rather_than_stamped(
    database, monkeypatch
):
    """The stamp says the files are the database, so it may only land if the database has not
    moved. Reporting the race is what makes the rerun the operator's call rather than a guess."""
    key = _key(LWW, database, "moving")
    ts.replace(key, {"n": 1})
    original = store_export._materialize

    def write_again(*args, **kwargs):
        result = original(*args, **kwargs)
        ts.replace(key, {"n": 2})
        return result

    monkeypatch.setattr(store_export, "_materialize", write_again)
    exported = store_export.export_root(str(database), report=lambda line: None)

    assert exported.raced == (LWW,)
    states = store_export.read_store_states(database_path(str(database)))
    assert not states[LWW].exported


def test_a_store_that_was_not_written_during_the_export_is_stamped_current(database):
    """The partner of the raced case: an export nothing raced stamps, and the staleness gate
    every file reader consults then answers current."""
    ts.replace(_key(LWW, database, "settled"), {"n": 1})

    exported = store_export.export_root(str(database), report=lambda line: None)

    assert exported.raced == ()
    assert store_export.stale_stores(database_path(str(database))) == ()
    assert store_export.read_store_states(database_path(str(database)))[LWW].exported


def test_a_store_written_after_its_export_reads_stale_until_it_is_exported_again(database):
    """What the doctor and the archive ask before they read files."""
    key = _key(LWW, database, "moving")
    ts.replace(key, {"n": 1})
    store_export.export_root(str(database), report=lambda line: None)
    ts.replace(key, {"n": 2})

    db_path = database_path(str(database))
    assert store_export.stale_stores(db_path) == (LWW,)

    store_export.export_root(str(database), report=lambda line: None)
    assert store_export.stale_stores(db_path) == ()


def test_a_store_the_database_never_recorded_a_write_against_reads_current(database):
    """A legitimately empty store must never leave a file reader permanently invalid."""
    ts.replace(_key(LWW, database, "only-one"), {"n": 1})
    store_export.export_root(str(database), report=lambda line: None)

    states = store_export.read_store_states(database_path(str(database)))
    assert STATE_ALPHA not in states
    assert store_export.stale_stores(database_path(str(database)), (STATE_ALPHA,)) == ()


def test_a_deleted_record_takes_its_file_and_its_tombstone_with_it(database):
    """The tombstone is what tells a later export the file is meant to be gone; once the file is
    gone and the stamp has landed it has done its work and is pruned."""
    key = _key(LWW, database, "removed")
    ts.replace(key, {"n": 1})
    store_export.export_root(str(database), report=lambda line: None)
    path = database / "lww" / "removed.json"
    assert path.is_file()

    ts.delete(key)
    deleted: list[str] = []
    store_export.export_root(str(database), report=deleted.append)

    assert not path.exists()
    assert any(str(path) in line for line in deleted)
    conn = sqlite3.connect(str(database_path(str(database))), isolation_level=None)
    try:
        rows = conn.execute(
            "select count(*) from tombstones where store = ? and parts = ?",
            (LWW, encode_parts(("removed",))),
        ).fetchone()
    finally:
        conn.close()
    assert rows[0] == 0


def test_exporting_a_root_that_never_had_a_database_refuses_rather_than_reporting_nothing(
    tmp_path
):
    """"Nothing to export" and "this root's records are files already" are different answers,
    and only one of them is true here."""
    with pytest.raises(ts.StoreError) as raised:
        store_export.export_root(str(tmp_path), report=lambda line: None)

    assert "store.db" in str(raised.value)
