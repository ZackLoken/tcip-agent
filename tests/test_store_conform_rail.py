"""The conform rail: what each backend refuses about a root the other one's layout holds.

A root is conformed once its records live in a store database; until then they are files. A
file-backend write cannot bump a database's counters and a database write leaves the files
where they were, so a root written through both backends loses writes with nothing to detect
it by. Each backend therefore refuses the half of that it can see: the database backend refuses
an unconformed root, and the file backend refuses record and log writes to a conformed one. Both
refusals ship with the legitimate call they must still admit.

The stores here are the suite's own, so what is under test is the rail's mechanics: which paths
ask, what a database that has never held a store says about a file claiming it, and what a claim
registered mid-process does to a connection already open. The shipped claim table meeting real
project directories is ``test_store_conform_rail_layouts.py``.
"""

from __future__ import annotations

from contextlib import contextmanager

import pytest

import tcip_store as ts
from tcip_store.file_backend import FileBackend, RootedFileLocator
from tcip_store.sqlite_backend import SqliteBackend, database_path
from tests._store_worker import BLOB, LOG, LWW, directory_claim, register_contract_stores

register_contract_stores()

LATE_ARRIVAL = "rail_late_arrival"
"""A store whose files can appear beside a database that has never held it."""

UNCLAIMED = "rail_unclaimed"
"""A store that states no claim, which is what the database backend has to refuse."""

_declared = False


def _register_rail_stores() -> None:
    global _declared
    if _declared:
        return
    _declared = True
    ts.register_store(
        ts.StoreDescriptor(
            name=LATE_ARRIVAL,
            kind="record",
            key_fields=("name",),
            codec=ts.RECORD_JSON,
            concurrency="last_writer_wins",
            locator=RootedFileLocator(prefix=("late",), suffix=".json"),
            claim=directory_claim("late", ".json"),
        )
    )
    ts.register_store(
        ts.StoreDescriptor(
            name=UNCLAIMED,
            kind="record",
            key_fields=("name",),
            codec=ts.RECORD_JSON,
            concurrency="last_writer_wins",
            locator=RootedFileLocator(prefix=("unclaimed",), suffix=".json"),
        )
    )


_register_rail_stores()


@contextmanager
def bound(backend):
    """Bind one backend for a block, since these cases are about handing a root between two.

    The suite's own backend is put back on the way out rather than dropped: a call after the
    block still has to reach a store, and a process with nothing bound is the state the seam
    refuses outright.
    """
    from tcip_store.store import _backend

    previous = _backend()
    ts.bind(backend)
    try:
        yield backend
    finally:
        ts.bind(previous)
        backend.close()


def export_files(root) -> None:
    """Write this root's rows back out as the files every reader outside the seam sees."""
    from tcip_store.export import export_root

    export_root(str(root), report=lambda line: None)


def _key(store: str, root, *parts: str) -> ts.Key:
    return ts.Key(store, str(root), parts)


# ── the database backend refuses an unconformed root ───────────────────────────


def test_a_root_whose_records_are_still_files_is_refused_rather_than_read_as_empty(tmp_path):
    """An empty database beside a populated layout answers every read with absence, and absence
    is what a confirmed negative looks like to a trainer."""
    with bound(FileBackend()):
        ts.replace(_key(LWW, tmp_path, "already_here"), {"n": 1})

    with bound(SqliteBackend()):
        with pytest.raises(ts.StoreError) as raised:
            ts.read(_key(LWW, tmp_path, "already_here"), default=None)

    message = str(raised.value)
    assert "scripts/adopt_store.py" in message
    assert "already_here.json" in message


def test_a_write_to_a_root_whose_records_are_still_files_is_refused_before_it_lands(tmp_path):
    """The refusal covers the write door too: a database created beside the files would hold
    the new value and leave the old one on disk for the next file reader."""
    with bound(FileBackend()):
        ts.replace(_key(LWW, tmp_path, "already_here"), {"n": 1})

    with bound(SqliteBackend()):
        with pytest.raises(ts.StoreError) as raised:
            ts.replace(_key(LWW, tmp_path, "another"), {"n": 2})

    assert "scripts/adopt_store.py" in str(raised.value)
    assert not (tmp_path / ".tcip" / "store.db").exists()


def test_a_root_with_no_record_files_creates_its_database_and_proceeds(tmp_path):
    """The rail is about state that would be lost, so a root that holds none is ordinary work."""
    with bound(SqliteBackend()):
        ts.replace(_key(LWW, tmp_path, "fresh"), {"n": 1})
        assert ts.read(_key(LWW, tmp_path, "fresh")) == {"n": 1}

    assert (tmp_path / ".tcip" / "store.db").is_file()


def test_a_root_holding_only_blob_files_is_not_a_root_holding_records(tmp_path):
    """Blobs stay files under every backend, so an imagery or label tree is not state the
    database would be missing."""
    with bound(FileBackend()):
        ts.put_blob(_key(BLOB, tmp_path, "picture"), b"\x89PNG", expect=ts.Version.ABSENT)

    with bound(SqliteBackend()):
        ts.replace(_key(LWW, tmp_path, "fresh"), {"n": 1})
        assert ts.read(_key(LWW, tmp_path, "fresh")) == {"n": 1}
        assert ts.read_blob_versioned(_key(BLOB, tmp_path, "picture")).value == b"\x89PNG"


def test_an_exported_root_holds_both_and_is_not_refused(tmp_path):
    """Once the records are in the database, the files an export writes beside it are the
    export's own output and must not read as a root that was never adopted."""
    with bound(SqliteBackend()):
        ts.replace(_key(LWW, tmp_path, "fresh"), {"n": 1})
        export_files(tmp_path)
    assert (tmp_path / "lww" / "fresh.json").is_file()

    with bound(SqliteBackend()) as backend:
        backend.require_conformed(str(tmp_path), (LWW,))
        assert ts.read(_key(LWW, tmp_path, "fresh")) == {"n": 1}


def test_a_file_written_after_this_backend_last_looked_still_refuses(tmp_path):
    """The root was empty when this backend first answered about it, and a file backend wrote
    into it afterwards, which is exactly what the file backend does not refuse while no database
    exists. An answer remembered from the first look would create a database over that file and
    read it as absent."""
    with bound(SqliteBackend()) as database:
        assert ts.read(_key(LWW, tmp_path, "not_yet"), default=None) is None

        with bound(FileBackend()):
            ts.replace(_key(LWW, tmp_path, "arrived_late"), {"n": 1})

        ts.bind(database)
        with pytest.raises(ts.StoreError) as raised:
            ts.replace(_key(LWW, tmp_path, "not_yet"), {"n": 2})

    assert "scripts/adopt_store.py" in str(raised.value)
    assert "arrived_late.json" in str(raised.value)
    assert not (tmp_path / ".tcip" / "store.db").exists()


def test_a_restored_archive_reads_back_at_once_with_no_hand_adoption(tmp_path, monkeypatch):
    """An import extracts a project's files into a fresh directory and, bound to the database
    backend, adopts them into a database itself: the root is usable at once, with no operator
    scripts/adopt_store.py run between the two doors and no window where a confirmed negative
    would otherwise read as absent.

    The two phases take separate platform roots. The archive and the import are audited calls
    that record under the root their process is pinned to, so one platform root written through
    both backends would trip the conform rail on the setup rather than on the case.
    """
    from tcip_mcp import dataset_layout
    from tcip_mcp.tools.project_tools import archive_project, import_project

    source = tmp_path / "source"
    (source / "images" / "2026-03-04").mkdir(parents=True)
    (source / "images" / "2026-03-04" / "a_1.jpg").write_bytes(b"\xff\xd8\xff")
    negative = {"bud/2026-03-04": {"a_1.jpg": {"status": "negative", "by": "user:ü"}}}
    restored = tmp_path / "restored"
    platform_files = tmp_path / "platform_files"
    platform_database = tmp_path / "platform_database"
    platform_files.mkdir()
    platform_database.mkdir()

    monkeypatch.setenv("TCIP_STATE_ROOT", str(platform_files))
    with bound(FileBackend()):
        ts.replace(dataset_layout.image_status_key(source), negative, expect=ts.Version.ABSENT)
        assert "error" not in archive_project(str(source), str(tmp_path / "bundle.zip"))

    monkeypatch.setenv("TCIP_STATE_ROOT", str(platform_database))
    with bound(SqliteBackend()):
        # The order a long-lived process reaches a destination in: it answers about the root,
        # then the bundle lands in it, then something reads.
        restored.mkdir()
        assert ts.read(dataset_layout.image_status_key(restored), default=None) is None
        imported = import_project(str(tmp_path / "bundle.zip"), str(restored))
        assert "error" not in imported
        assert imported["database_built"] is True

        assert ts.read(dataset_layout.image_status_key(restored)) == negative
    assert (restored / ".tcip" / "store.db").is_file()


def test_a_file_that_is_not_a_database_at_the_database_path_refuses_by_name(tmp_path):
    """Something else holding the name means this backend cannot say whether the root's state
    is in a database, and a driver error escaping untyped would come out of the gates that read
    counters here as a traceback rather than a report."""
    from tcip_store.sqlite_backend import open_read_only

    (tmp_path / ".tcip").mkdir()
    (tmp_path / ".tcip" / "store.db").write_bytes(b"not a database, just some bytes\n")

    with pytest.raises(ts.StoreError) as raised:
        open_read_only(tmp_path / ".tcip" / "store.db")

    assert "not a SQLite database" in str(raised.value)


# ── the file backend refuses records and logs on a conformed root ──────────────


def test_a_record_write_beside_a_database_that_owns_the_root_is_refused(tmp_path):
    """The write would land as a file the database never sees, and the next database read would
    answer with what the database holds, so one of the two writes is simply gone."""
    with bound(SqliteBackend()):
        ts.replace(_key(LWW, tmp_path, "owned"), {"n": 1})

    with bound(FileBackend()):
        with pytest.raises(ts.StoreError) as raised:
            ts.replace(_key(LWW, tmp_path, "owned"), {"n": 2})

    message = str(raised.value)
    assert "scripts/export_store.py" in message
    assert "TCIP_STORE_BACKEND=file" in message
    assert not (tmp_path / "lww" / "owned.json").exists()


def test_an_append_beside_a_database_that_owns_the_root_is_refused(tmp_path):
    """A log entry written to a file the database does not know about is an audit line the
    trail loses."""
    with bound(SqliteBackend()):
        ts.append(_key(LOG, tmp_path, "trail"), {"event": "one"})

    with bound(FileBackend()):
        with pytest.raises(ts.StoreError) as raised:
            ts.append(_key(LOG, tmp_path, "trail"), {"event": "two"})

    assert "scripts/export_store.py" in str(raised.value)


def test_reads_and_blobs_stay_open_on_a_root_a_database_owns(tmp_path):
    """The doctor and every other file reader work through the file backend on an exported
    root, and blobs are files under both backends, so neither may be refused here."""
    with bound(SqliteBackend()):
        ts.replace(_key(LWW, tmp_path, "owned"), {"n": 1})
        export_files(tmp_path)

    with bound(FileBackend()):
        assert ts.read(_key(LWW, tmp_path, "owned")) == {"n": 1}
        ts.put_blob(_key(BLOB, tmp_path, "picture"), b"\x89PNG", expect=ts.Version.ABSENT)
        assert ts.read_blob_versioned(_key(BLOB, tmp_path, "picture")).value == b"\x89PNG"
        ts.delete(_key(BLOB, tmp_path, "picture"))


def test_a_root_with_no_database_takes_file_writes_as_it_always_did(tmp_path):
    """The refusal is about a database that exists, and most roots do not have one."""
    with bound(FileBackend()):
        ts.replace(_key(LWW, tmp_path, "plain"), {"n": 1})
        assert ts.read(_key(LWW, tmp_path, "plain")) == {"n": 1}
        ts.append(_key(LOG, tmp_path, "trail"), {"event": "one"})
        assert ts.read_log(_key(LOG, tmp_path, "trail")).records == [{"event": "one"}]


# ── every path asks, in every state a root can be in ─────────────────────────


def test_a_root_exported_and_then_reopened_by_a_fresh_process_is_admitted(tmp_path):
    """A restart meets its own export: every claimed file beside the database belongs to a store
    the database holds, which is what separates an export from state left behind."""
    with bound(SqliteBackend()):
        ts.replace(_key(LWW, tmp_path, "kept"), {"n": 1})
        ts.append(_key(LOG, tmp_path, "trail"), {"event": "one"})
        export_files(tmp_path)
    assert (tmp_path / "lww" / "kept.json").is_file()

    with bound(SqliteBackend()):
        assert ts.read(_key(LWW, tmp_path, "kept")) == {"n": 1}
        assert ts.read_log(_key(LOG, tmp_path, "trail")).records == [{"event": "one"}]


# ── the layouts an operation serves, and what each connection has checked ────

_EVERY_PATH = [
    "append", "exists", "keys", "read_log", "read_versioned", "replace", "transaction"
]


def _refuses_on_every_path(root) -> dict[str, str]:
    """Run one operation of each rail path and say which refused, so a gap is named not counted."""
    outcomes: dict[str, str] = {}
    calls = {
        "read_versioned": lambda: ts.read(_key(LWW, root, "x"), default=None),
        "exists": lambda: ts.exists(_key(LWW, root, "x")),
        "keys": lambda: ts.keys(LWW, str(root)),
        "read_log": lambda: ts.read_log(_key(LOG, root, "trail")),
        "replace": lambda: ts.replace(_key(LWW, root, "x"), {"n": 1}),
        "append": lambda: ts.append(_key(LOG, root, "trail"), {"event": "one"}),
        "transaction": lambda: _write_in_transaction(root),
    }
    for name, call in calls.items():
        try:
            call()
        except ts.StoreError as exc:
            outcomes[name] = str(exc)
    return outcomes


def _write_in_transaction(root) -> None:
    key = _key(LWW, root, "x")
    with ts.transaction(key) as txn:
        txn.write(key, {"n": 1})


def test_every_rail_path_refuses_a_root_whose_records_are_still_files(tmp_path):
    """A rail on the write door alone lets a read answer absence for a confirmed negative that
    is sitting right there, so every path this backend answers on asks first."""
    with bound(FileBackend()):
        ts.replace(_key(LWW, tmp_path, "already_here"), {"n": 1})

    with bound(SqliteBackend()):
        refused = _refuses_on_every_path(tmp_path)

    assert sorted(refused) == _EVERY_PATH
    assert all("scripts/adopt_store.py" in message for message in refused.values())
    assert not database_path(str(tmp_path)).exists()


def test_every_rail_path_refuses_a_file_of_a_store_the_database_never_held(tmp_path):
    """With a database present the accounting is per store, and a store the database has never
    held is the one whose files an export cannot explain."""
    with bound(SqliteBackend()):
        ts.replace(_key(LWW, tmp_path, "owned"), {"n": 1})
    (tmp_path / "late").mkdir()
    (tmp_path / "late" / "arrived.json").write_text('{"n": 2}', encoding="utf-8")

    with bound(SqliteBackend()):
        refused = _refuses_on_every_path(tmp_path)

    assert sorted(refused) == _EVERY_PATH
    assert all(LATE_ARRIVAL in message for message in refused.values())


def test_every_rail_path_refuses_on_a_cached_connection_once_the_claim_set_has_grown(tmp_path):
    """A claim can arrive after a connection is open, and a connection that kept serving absence
    for the store it now covers is the silent invisibility this rail exists for. The re-check is
    per operation, so it belongs on every path rather than only on the next open."""
    grown = "rail_grown_after_the_connection"
    (tmp_path / "grown").mkdir()
    (tmp_path / "grown" / "arrived.json").write_text('{"n": 2}', encoding="utf-8")

    with bound(SqliteBackend()):
        ts.replace(_key(LWW, tmp_path, "owned"), {"n": 1})
        assert ts.read(_key(LWW, tmp_path, "owned")) == {"n": 1}

        ts.register_store(
            ts.StoreDescriptor(
                name=grown,
                kind="record",
                key_fields=("name",),
                codec=ts.RECORD_JSON,
                concurrency="last_writer_wins",
                locator=RootedFileLocator(prefix=("grown",), suffix=".json"),
                claim=directory_claim("grown", ".json"),
            )
        )
        refused = _refuses_on_every_path(tmp_path)

    assert sorted(refused) == _EVERY_PATH
    assert all(grown in message for message in refused.values())


def test_a_store_that_states_no_claim_is_refused_rather_than_placed_by_guess(tmp_path):
    """Where a store's files live is what the rail reasons about, so a store that never says is
    one this backend cannot answer for at all."""
    with bound(SqliteBackend()):
        with pytest.raises(ts.StoreError) as raised:
            ts.replace(_key(UNCLAIMED, tmp_path, "x"), {"n": 1})

    assert UNCLAIMED in str(raised.value)
    assert "claim=" in str(raised.value)
