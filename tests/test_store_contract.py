"""The storage seam's contract on both backends, through the ``store`` fixture; a case about one
backend's own mechanics skips the other. Isolation cases run across real OS processes, and also
against a weakened backend (``TCIP_STORE_CONTRACT_UNLOCKED=1`` on files,
``TCIP_STORE_CONTRACT_IGNORES_EXPECT=1`` on the database) where they fail."""

from __future__ import annotations

import json
import os
import re
import sqlite3
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

import pytest

import tcip_store as ts
from tcip_annotation import json_io, verdicts
from tcip_mcp import (
    audit,
    dataset_layout,
    delivery,
    model_registry,
    project_record,
    project_status,
    traits,
    web_client,
    workspace,
)
from tcip_mcp.project_paths import project_state_dir
from tcip_mcp.pipelines.delivery_events_schema import DeliveryEventRecord
from tcip_mcp.pipelines.data import band_groups, selection
from tcip_mcp.pipelines.postprocessing import plant_mapping
from tcip_mcp.pipelines.training import hpo
from tcip_mcp.tools import (
    inference_tools,
    meta_tools,
    project_tools,
    proposal_tools,
)
from tcip_store.file_backend import (
    DATABASE_FILENAME,
    FileBackend,
    RootedFileLocator,
    creation_temp_name,
)
from tcip_store.sqlite_backend import SqliteBackend, database_path, encode_parts
from tcip_web.routes import canvas
from tests._store_worker import (
    BACKEND_ENV,
    BLOB,
    CAS,
    FILE,
    LOG,
    LWW,
    NESTED,
    OPAQUE,
    RELAXED,
    SEALED_BLOB,
    SQLITE,
    STATE_FILES,
    STRICT,
    make_backend,
    register_contract_stores,
    wait_for,
)

register_contract_stores()

_WORKER = Path(__file__).with_name("_store_worker.py")


@dataclass
class Harness:
    """The bound backend, the root its keys hang off, and how to reach bytes behind it."""

    backend: FileBackend | SqliteBackend
    root: Path
    name: str = FILE
    procs: list[subprocess.Popen] = field(default_factory=list)

    def key(self, store: str, *parts: str) -> ts.Key:
        return ts.Key(store, str(self.root), tuple(parts))

    def path(self, key: ts.Key) -> Path:
        """Where the file backend puts a key. Only the file-backend cases use this."""
        return self.backend.path_for(key)

    def connect(self) -> sqlite3.Connection:
        """A connection of this test's own to the root's database, for reaching behind the seam."""
        return sqlite3.connect(str(database_path(str(self.root))), isolation_level=None)

    def damage_record(self, key: ts.Key, data: bytes) -> None:
        """Put bytes that will not decode behind a record, where this backend reads it from."""
        if self.name == FILE:
            self.path(key).write_bytes(data)
            return
        conn = self.connect()
        try:
            conn.execute(
                "update records set value = ? where store = ? and parts = ?",
                (data, key.store, encode_parts(key.parts)),
            )
        finally:
            conn.close()

    def damage_log_entry(self, key: ts.Key, position: int, data: bytes) -> None:
        """Replace one committed log entry's bytes with bytes that will not decode."""
        conn = self.connect()
        try:
            ids = [
                row[0]
                for row in conn.execute(
                    "select id from log_entries where store = ? and parts = ? order by id",
                    (key.store, encode_parts(key.parts)),
                )
            ]
            conn.execute("update log_entries set entry = ? where id = ?", (data, ids[position]))
        finally:
            conn.close()

    def spawn(self, *args: object) -> subprocess.Popen:
        proc = subprocess.Popen(
            [sys.executable, str(_WORKER), *[str(a) for a in args]], env=_child_env()
        )
        self.procs.append(proc)
        return proc


def _child_env() -> dict[str, str]:
    env = dict(os.environ)
    package_src = str(Path(ts.__file__).resolve().parents[1])
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = package_src + (os.pathsep + existing if existing else "")
    return env


def only_on(store: "Harness", backend: str, why: str) -> None:
    """Leave a case to the backend whose own mechanics it is about, saying which and why."""
    if store.name != backend:
        pytest.skip(f"this case is about the {backend} backend: {why}")


@pytest.fixture(params=[FILE, SQLITE])
def store(request, tmp_path, monkeypatch):
    monkeypatch.setenv(BACKEND_ENV, request.param)
    backend = make_backend()
    ts.bind(backend)
    harness = Harness(backend=backend, root=tmp_path, name=request.param)
    try:
        yield harness
    finally:
        for proc in harness.procs:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=30)
        ts.unbind()
        backend.close()


# ── process lifecycle ───────────────────────────────────────────────────────────


def test_release_root_releases_that_root_so_it_can_be_renamed_and_keeps_every_other(store):
    """``release_root`` releases what a read of one root opened, on one thread, so that root can
    be renamed, and leaves another root's handle open."""
    key = store.key(LWW, "before-close")
    other_root = store.root.parent / f"{store.root.name}-other"
    other_root.mkdir()
    other = ts.Key(key.store, str(other_root), key.parts)
    ts.replace(key, {"n": 1})
    ts.replace(other, {"n": 2})
    assert ts.read(key) == {"n": 1} and ts.read(other) == {"n": 2}

    moved = store.root.parent / f"{store.root.name}-moved"
    denies_an_open_root = store.name == SQLITE and os.name == "nt"
    if denies_an_open_root:
        with pytest.raises(OSError):
            os.rename(str(store.root), str(moved))

    ts.release_root(store.root)

    os.rename(str(store.root), str(moved))
    moved_key = ts.Key(key.store, str(moved), key.parts)
    assert ts.read(moved_key) == {"n": 1}
    if denies_an_open_root:
        with pytest.raises(OSError):
            os.rename(str(other_root), str(other_root.parent / f"{other_root.name}-moved"))
    assert ts.read(other) == {"n": 2}


def test_release_root_waits_for_an_operation_in_flight_on_that_root(store):
    """A release issued while another thread is inside a transaction on the root returns only
    once that transaction has committed, and the commit lands."""
    import threading

    if store.name != SQLITE:
        pytest.skip("the file backend holds no handles for a release to wait on")
    key = store.key(LWW, "in-flight")
    ts.replace(key, {"n": 1})
    inside, finish = threading.Event(), threading.Event()
    order: list[str] = []

    def hold() -> None:
        with ts.transaction(key) as txn:
            inside.set()
            finish.wait(30)
            txn.write(key, {"n": 2})
        order.append("committed")

    def release() -> None:
        ts.release_root(store.root)
        order.append("released")

    worker = threading.Thread(target=hold)
    worker.start()
    assert inside.wait(30)
    releaser = threading.Thread(target=release)
    releaser.start()
    releaser.join(0.5)
    assert releaser.is_alive(), "the release returned while a transaction on the root was open"
    finish.set()
    worker.join(30)
    releaser.join(30)
    assert order == ["committed", "released"]
    assert ts.read(key) == {"n": 2}


# ── atomicity and durability ────────────────────────────────────────────────────


def test_replace_round_trips_and_the_version_tracks_the_content(store):
    key = store.key(LWW, "round-trip")
    first = ts.replace(key, {"n": 1})
    assert ts.read(key) == {"n": 1}
    assert ts.read_versioned(key).version == first

    second = ts.replace(key, {"n": 2})
    assert second != first
    assert ts.read(key) == {"n": 2}


def test_a_concurrent_rewriter_is_never_observed_half_written(store):
    key = store.key(LWW, "large")
    ts.replace(key, {"tag": "a", "pad": "a" * 20000})
    writer = store.spawn("rewrite", store.root, "large", 60, 20000)

    seen = set()
    while writer.poll() is None:
        value = ts.read(key)
        assert value["pad"] == value["tag"] * 20000
        seen.add(value["tag"])
        time.sleep(0.001)
    assert writer.wait(timeout=60) == 0
    assert seen <= {"a", "b"}


def test_a_failed_encode_leaves_the_previous_value_and_no_artifact(store):
    key = store.key(STRICT, "encodable")
    ts.replace(key, {"n": 1})

    with pytest.raises(ts.StoreError):
        ts.replace(key, {"n": float("nan")})

    assert ts.read(key) == {"n": 1}
    assert ts.keys(STRICT, str(store.root)) == [key]


def test_a_failed_blob_write_leaves_the_previous_bytes_untouched(store):
    key = store.key(BLOB, "checkpoint")
    ts.put_blob(key, b"first")

    with pytest.raises(RuntimeError):
        with ts.write_blob(key) as handle:
            handle.write(b"second")
            raise RuntimeError("the producer failed part way")

    with ts.open_blob(key) as handle:
        assert handle.read() == b"first"
    with pytest.raises(ts.WrongKind):
        ts.append(key, {"entry": 1})


# ── isolation across real OS processes ──────────────────────────────────────────


def test_a_stale_writer_cannot_clobber_a_committed_transaction(store):
    """A compare-and-set writer that read before another process's transaction committed."""
    key = store.key(CAS, "contended")
    ts.replace(key, {"who": "seed"}, expect=ts.Version.ABSENT)
    ready_reader = store.root / "reader.ready"
    ready_holder = store.root / "holder.ready"
    go = store.root / "go"
    result = store.root / "result.json"

    writer = store.spawn(
        "write-after", store.root, CAS, "contended", "cas", "outsider",
        ready_reader, go, result,
    )
    wait_for(ready_reader)
    holder = store.spawn("hold-transaction", store.root, CAS, "contended", "insider", 1.5, ready_holder)
    wait_for(ready_holder)
    go.write_text("go", encoding="utf-8")

    assert holder.wait(timeout=60) == 0
    assert writer.wait(timeout=60) == 0
    outcome = json.loads(result.read_text(encoding="utf-8"))
    assert outcome["outcome"] == "VersionConflict"
    assert outcome["waited_s"] >= 1.0
    assert ts.read(key) == {"who": "insider"}


def test_an_unconditional_writer_waits_for_a_transaction_and_then_wins(store):
    """The admitted form of the same race, on a store that declares last-writer-wins."""
    key = store.key(LWW, "contended")
    ts.replace(key, {"who": "seed"})
    ready_writer = store.root / "writer.ready"
    ready_holder = store.root / "holder.ready"
    go = store.root / "go"
    result = store.root / "result.json"

    writer = store.spawn(
        "write-after", store.root, LWW, "contended", "unconditional", "outsider",
        ready_writer, go, result,
    )
    wait_for(ready_writer)
    holder = store.spawn("hold-transaction", store.root, LWW, "contended", "insider", 1.5, ready_holder)
    wait_for(ready_holder)
    go.write_text("go", encoding="utf-8")

    assert holder.wait(timeout=60) == 0
    assert writer.wait(timeout=60) == 0
    outcome = json.loads(result.read_text(encoding="utf-8"))
    assert outcome["outcome"] == "written"
    assert outcome["waited_s"] >= 1.0
    assert ts.read(key) == {"who": "outsider"}


def test_transactions_serialize_a_read_modify_write_across_processes(store):
    key = store.key(CAS, "counter")
    workers = [store.spawn("increment", store.root, "counter", 12) for _ in range(4)]
    for worker in workers:
        assert worker.wait(timeout=120) == 0

    assert ts.read(key) == {"n": 48}


def test_two_processes_naming_the_same_keys_in_opposite_orders_both_finish(store):
    forward = store.spawn("two-keys", store.root, "alpha", "beta", 25)
    backward = store.spawn("two-keys", store.root, "beta", "alpha", 25)
    assert forward.wait(timeout=60) == 0
    assert backward.wait(timeout=60) == 0


def test_two_processes_writing_from_one_version_produce_one_winner_and_one_conflict(store):
    key = store.key(CAS, "compare-and-set")
    ts.replace(key, {"who": "seed"}, expect=ts.Version.ABSENT)
    go = store.root / "go"
    contenders = []
    for name in ("first", "second"):
        ready = store.root / f"{name}.ready"
        result = store.root / f"{name}.json"
        proc = store.spawn(
            "write-after", store.root, CAS, "compare-and-set", "cas", name, ready, go, result
        )
        wait_for(ready)
        contenders.append((proc, result))
    go.write_text("go", encoding="utf-8")

    outcomes = []
    for proc, result in contenders:
        assert proc.wait(timeout=60) == 0
        outcomes.append(json.loads(result.read_text(encoding="utf-8"))["outcome"])
    assert sorted(outcomes) == ["VersionConflict", "written"]
    assert ts.read(key)["who"] in ("first", "second")


def test_create_only_writes_once(store):
    key = store.key(LWW, "created-once")
    ts.replace(key, {"n": 1}, expect=ts.Version.ABSENT)

    with pytest.raises(ts.VersionConflict):
        ts.replace(key, {"n": 2}, expect=ts.Version.ABSENT)
    assert ts.read(key) == {"n": 1}


def test_an_identical_rewrite_leaves_a_held_token_valid(store):
    key = store.key(CAS, "unchanged")
    held = ts.replace(key, {"n": 1}, expect=ts.Version.ABSENT)

    rewritten = ts.replace(key, {"n": 1}, expect=held)
    assert rewritten == held
    assert ts.replace(key, {"n": 2}, expect=held) != held


def test_a_cas_record_delete_from_a_current_token_lands_and_a_stale_one_is_refused(store):
    key = store.key(CAS, "conditional-delete")
    held = ts.replace(key, {"n": 1}, expect=ts.Version.ABSENT)
    moved = ts.replace(key, {"n": 2}, expect=held)

    with pytest.raises(ts.VersionConflict) as raised:
        ts.delete(key, expect=held)
    assert raised.value.actual == moved
    assert ts.read(key) == {"n": 2}

    ts.delete(key, expect=moved)
    assert ts.read(key, default=None) is None


def test_a_transaction_applies_in_the_declared_order_and_a_crash_leaves_a_prefix(store):
    """The order comes from the declaration, not from the order the body wrote the keys."""
    only_on(store, FILE, "a crash mid-apply is what a backend applying key by key leaves, the "
                         "worker that pauses between two applies binds a file backend of its "
                         "own that the fixture's parameter cannot reach, and inside one atomic "
                         "commit the applied order is unobservable because no reader ever sees "
                         "a state between the two writes")
    first = store.key(LWW, "alpha")
    second = store.key(LWW, "beta")
    marker = store.root / "applied.marker"
    proc = store.spawn("pause-mid-apply", store.root, "alpha", "beta", marker, 30)
    wait_for(marker, timeout_s=60)
    proc.kill()
    proc.wait(timeout=30)

    assert ts.read(first) == {"who": "alpha"}
    assert ts.read(second, default=None) is None


def test_a_transaction_killed_before_its_commit_applies_nothing_and_wedges_no_successor(store):
    """The all-or-nothing half of the crash contract, and the lock a killed writer must not keep.

    A database commits the whole transaction or none of it, so there is no prefix to find. The
    successor write is the other half: a holder killed mid-transaction that left the database
    claimed would strand every later writer, which is worse than the lost work.
    """
    only_on(store, SQLITE, "all-or-nothing across a crash is exactly what the file backend "
                           "does not promise, and the case above pins what it does instead")
    first = store.key(LWW, "alpha")
    second = store.key(LWW, "beta")
    marker = store.root / "staged.marker"
    proc = store.spawn("pause-before-commit", store.root, "alpha", "beta", marker, 30)
    wait_for(marker, timeout_s=60)
    proc.kill()
    proc.wait(timeout=30)

    assert ts.read(first, default=None) is None
    assert ts.read(second, default=None) is None

    with ts.transaction(first, second, timeout_s=10) as txn:
        txn.write(first, {"who": "successor"})
        txn.write(second, {"who": "successor"})
    assert ts.read(first) == ts.read(second) == {"who": "successor"}


# ── append durability ───────────────────────────────────────────────────────────


def test_concurrent_appenders_lose_no_entry_and_interleave_none(store):
    key = store.key(LOG, "audit")
    workers = [store.spawn("append", store.root, "audit", f"proc{n}", 25) for n in range(3)]
    for worker in workers:
        assert worker.wait(timeout=120) == 0

    page = ts.read_log(key)
    assert len(page.records) == 75
    assert not page.torn_tail and page.corrupt == ()
    for tag in ("proc0", "proc1", "proc2"):
        assert sorted(r["i"] for r in page.records if r["tag"] == tag) == list(range(25))


def test_an_append_is_readable_by_another_process_once_it_returns(store):
    key = store.key(LOG, "handoff")
    for i in range(3):
        ts.append(key, {"i": i})
    result = store.root / "count.json"

    reader = store.spawn("count-log", store.root, "handoff", result)
    assert reader.wait(timeout=60) == 0
    assert json.loads(result.read_text(encoding="utf-8")) == {"records": 3, "torn_tail": False}


def test_a_killed_appender_leaves_every_acknowledged_entry_in_order(store):
    key = store.key(LOG, "killed")
    ready = store.root / "appending.ready"
    proc = store.spawn("append-until-killed", store.root, "killed", ready)
    wait_for(ready, timeout_s=60)
    time.sleep(0.3)
    proc.kill()
    proc.wait(timeout=30)

    page = ts.read_log(key)
    assert page.corrupt == ()
    assert [r["i"] for r in page.records] == list(range(len(page.records)))
    assert len(page.records) >= 1


def test_a_cursor_resumes_with_no_gap_and_no_repeat(store):
    key = store.key(LOG, "streamed")
    for i in range(3):
        ts.append(key, {"i": i})
    first = ts.read_log(key)
    assert [r["i"] for r in first.records] == [0, 1, 2]

    assert ts.read_log(key, after=first.cursor).records == []
    for i in range(3, 5):
        ts.append(key, {"i": i})
    second = ts.read_log(key, after=first.cursor)
    assert [r["i"] for r in second.records] == [3, 4]
    assert [r["i"] for r in ts.read_log(key, after=second.cursor).records] == []


# ── clearing a log ───────────────────────────────────────────────────────────────


def test_clear_log_removes_every_entry_and_reports_how_many_there_were(store):
    key = store.key(LOG, "history")
    for i in range(4):
        ts.append(key, {"i": i})

    removed = ts.clear_log(key)

    assert removed == 4
    assert ts.read_log(key).records == []
    ts.append(key, {"i": "after"})
    assert [r["i"] for r in ts.read_log(key).records] == ["after"]


def test_clear_log_on_a_log_with_no_entries_removes_nothing(store):
    key = store.key(LOG, "never-appended")

    assert ts.clear_log(key) == 0
    assert ts.read_log(key).records == []


def test_clear_log_on_an_empty_log_leaves_no_stale_store_behind(store):
    """Clearing a log nothing ever wrote must not manufacture a store_counters row: a store
    with no row there was never written, and one appearing here would read as behind the
    database it was never ahead of in the first place."""
    only_on(store, SQLITE, "store_counters and stale_stores are the database backend's own "
                            "bookkeeping; the file backend keeps no counter to leave untouched")
    from tcip_store.export import stale_stores

    key = store.key(LOG, "never-appended")

    assert ts.clear_log(key) == 0
    assert stale_stores(database_path(str(store.root)), (LOG,)) == ()


def test_clearing_a_log_tombstones_it_so_a_later_export_removes_the_stale_copy(store):
    """A log's own clear has to leave the same trail a record's delete already does: without
    a tombstone, export_root's delete pass never learns this key's exported file is stale."""
    only_on(store, SQLITE, "export_root and stale_stores reconcile a database backend's "
                            "exported loose copies; the file backend keeps no such copies")
    from tcip_store.export import export_root, stale_stores

    key = store.key(LOG, "history")
    exported_path = store.root / "logs" / "history.jsonl"
    ts.append(key, {"i": 0})
    export_root(str(store.root), report=lambda _line: None)
    assert exported_path.is_file()
    assert stale_stores(database_path(str(store.root)), (LOG,)) == ()

    ts.clear_log(key)
    export_root(str(store.root), report=lambda _line: None)

    assert not exported_path.is_file()
    assert stale_stores(database_path(str(store.root)), (LOG,)) == ()


def test_appending_to_a_cleared_log_clears_its_stale_tombstone(store):
    """A log's tombstone from an earlier clear must not survive a fresh append to the same
    key: a live entry and a pending delete for the one target would collide in export_root,
    which reads that combination as two mis-parted keys rather than one key written again."""
    only_on(store, SQLITE, "tombstones are the database backend's own export bookkeeping; "
                            "the file backend deletes and rewrites the log file directly")
    from tcip_store.export import export_root

    key = store.key(LOG, "reopened")
    ts.append(key, {"i": 0})
    export_root(str(store.root), report=lambda _line: None)
    ts.clear_log(key)
    ts.append(key, {"i": "again"})

    export_root(str(store.root), report=lambda _line: None)

    exported_path = store.root / "logs" / "reopened.jsonl"
    lines = exported_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["i"] == "again"


def test_a_cursor_taken_before_a_clear_reads_only_what_was_appended_after(store):
    """A cursor held across a clear_log is not a corruption risk on either backend: replaying
    it returns exactly the entries appended since the clear, never the cleared ones and never
    a misreading of the new bytes as damaged."""
    key = store.key(LOG, "cleared-and-resumed")
    for i in range(3):
        ts.append(key, {"i": i})
    before = ts.read_log(key)
    assert [r["i"] for r in before.records] == [0, 1, 2]

    ts.clear_log(key)
    ts.append(key, {"i": "after"})

    resumed = ts.read_log(key, after=before.cursor)
    assert [r["i"] for r in resumed.records] == ["after"]
    assert resumed.corrupt == ()
    assert not resumed.torn_tail


_CLEAR_LOG_RACE = (
    "clear_log stages a pending watermark, then removes the log file, then installs the "
    "watermark onto the marker: three file changes on the file backend that a reader or a "
    "crash can land between; the database backend has no such sequence to race"
)


def test_a_reader_interleaved_between_staging_and_the_unlink_reads_consistently(
    store, monkeypatch,
):
    """A reader that lands at the point clear_log is about to remove the log file (the
    file backend's own lock is thread-reentrant, so a same-thread call from inside that
    window runs rather than blocking) must not replay the cleared entries to the cursor
    that was current before the clear. This is the window between the pending watermark's
    own write and the unlink that commits the clear: the log file is still present with
    every old entry, the marker is still whatever it held before this clear, and the
    reader's own read of the marker, never the pending file, is what keeps it from seeing
    anything that has not committed yet."""
    only_on(store, FILE, _CLEAR_LOG_RACE)
    backend = store.backend
    key = store.key(LOG, "interleaved-clear")
    for i in range(3):
        ts.append(key, {"i": i})
    before = ts.read_log(key)
    path = store.path(key)
    old_size = path.stat().st_size

    real_remove = backend._remove_entry
    interleaved: dict[str, ts.LogPage] = {}
    pending_bytes = b""
    marker_exists = True

    def read_then_remove(path, *, durable):
        nonlocal pending_bytes, marker_exists
        interleaved["page"] = ts.read_log(key, after=before.cursor)
        pending_bytes = backend._clear_base_pending_path(path).read_bytes()
        marker_exists = backend._clear_base_path(path).exists()
        real_remove(path, durable=durable)

    monkeypatch.setattr(backend, "_remove_entry", read_then_remove)

    removed = ts.clear_log(key)

    assert removed == 3
    page = interleaved["page"]
    assert page.records == []
    assert page.cursor == before.cursor
    assert int(pending_bytes) == old_size
    assert not marker_exists


def test_a_reader_interleaved_between_the_unlink_and_the_watermark_install_reads_consistently(
    store, monkeypatch,
):
    """The other half of the same window: a reader landing after the unlink has committed
    but before the pending watermark is installed onto the marker. The log file is gone,
    the marker is still whatever it held before this clear, and the pending file already
    holds the value the marker is about to become; the reader answers from the marker
    alone, exactly as it does the instant a real crash lands here."""
    only_on(store, FILE, _CLEAR_LOG_RACE)
    backend = store.backend
    key = store.key(LOG, "interleaved-clear-install")
    for i in range(3):
        ts.append(key, {"i": i})
    before = ts.read_log(key)
    path = store.path(key)
    old_size = path.stat().st_size

    real_install = backend._install_pending_clear_base
    interleaved: dict[str, ts.LogPage] = {}
    pending_bytes = b""
    marker_exists = True

    def read_then_install(path, *, durable):
        nonlocal pending_bytes, marker_exists
        interleaved["page"] = ts.read_log(key, after=before.cursor)
        pending_bytes = backend._clear_base_pending_path(path).read_bytes()
        marker_exists = backend._clear_base_path(path).exists()
        real_install(path, durable=durable)

    monkeypatch.setattr(backend, "_install_pending_clear_base", read_then_install)

    removed = ts.clear_log(key)

    assert removed == 3
    page = interleaved["page"]
    assert page.records == []
    assert page.cursor == before.cursor
    assert int(pending_bytes) == old_size
    assert not marker_exists


def test_a_crash_after_the_unlink_replays_no_entry(store, monkeypatch):
    """A process that dies right after the unlink commits, before the pending watermark is
    installed onto the marker, must leave a state a later, fully independent read_log call
    reads correctly for the pre-clear end cursor: nothing new, never every cleared entry
    replayed. The pending file stays on disk holding the value the next writer will
    install, and the marker stays exactly where the crash found it."""
    only_on(store, FILE, _CLEAR_LOG_RACE)
    backend = store.backend
    key = store.key(LOG, "crash-mid-clear")
    for i in range(3):
        ts.append(key, {"i": i})
    before = ts.read_log(key)
    path = store.path(key)
    old_size = path.stat().st_size

    class _SimulatedCrash(Exception):
        pass

    real_remove = backend._remove_entry

    def crash_after_remove(*args, **kwargs):
        real_remove(*args, **kwargs)
        raise _SimulatedCrash

    monkeypatch.setattr(backend, "_remove_entry", crash_after_remove)

    with pytest.raises(_SimulatedCrash):
        ts.clear_log(key)

    resumed = ts.read_log(key, after=before.cursor)
    assert resumed.records == []
    assert resumed.cursor == before.cursor
    assert int(backend._clear_base_pending_path(path).read_bytes()) == old_size
    assert not backend._clear_base_path(path).exists()


def test_a_crash_after_the_unlink_then_new_entries_replays_only_the_new_ones(
    store, monkeypatch,
):
    """One step past test_a_crash_after_the_unlink_replays_no_entry: the identical crash
    (file removed, watermark pending), then fresh entries appended until the recreated
    file's own length passes the pre-crash cursor's byte offset. The first of those appends
    settles the pending watermark before writing anything, so a replay from the pre-crash
    cursor lands exactly on the boundary between the cleared entries and the new ones,
    never inside an entry a stale, unadvanced base would have made it seek into."""
    only_on(store, FILE, _CLEAR_LOG_RACE)
    backend = store.backend
    key = store.key(LOG, "crash-mid-clear-then-new-entries")
    for i in range(3):
        ts.append(key, {"i": i})
    before = ts.read_log(key)
    old_end = int(before.cursor)

    class _SimulatedCrash(Exception):
        pass

    real_remove = backend._remove_entry

    def crash_after_remove(*args, **kwargs):
        real_remove(*args, **kwargs)
        raise _SimulatedCrash

    monkeypatch.setattr(backend, "_remove_entry", crash_after_remove)

    with pytest.raises(_SimulatedCrash):
        ts.clear_log(key)

    path = store.path(key)
    labels: list[str] = []
    for i in range(200):
        label = f"new-{i}"
        ts.append(key, {"i": label})
        labels.append(label)
        size = path.stat().st_size
        assert size != old_end
        if size > old_end:
            break
    else:
        pytest.fail("could not grow the recreated log past the pre-crash cursor's byte offset")

    resumed = ts.read_log(key, after=before.cursor)

    assert [r["i"] for r in resumed.records] == labels
    assert resumed.corrupt == ()


def test_a_crash_after_staging_the_pending_watermark_abandons_it_on_the_next_append(
    store, monkeypatch,
):
    """A crash between writing the pending watermark and the unlink that would have made it
    real: the log file is untouched, the marker is untouched, and the pending file is the
    only sign anything happened. The clear never took effect, so the next append discards
    the stale stage rather than mistaking it for a completed clear."""
    only_on(store, FILE, _CLEAR_LOG_RACE)
    backend = store.backend
    key = store.key(LOG, "crash-after-staging")
    for i in range(3):
        ts.append(key, {"i": i})
    path = store.path(key)

    class _SimulatedCrash(Exception):
        pass

    real_write_pending = backend._write_pending_clear_base

    def crash_after_write(*args, **kwargs):
        real_write_pending(*args, **kwargs)
        raise _SimulatedCrash

    monkeypatch.setattr(backend, "_write_pending_clear_base", crash_after_write)

    with pytest.raises(_SimulatedCrash):
        ts.clear_log(key)

    assert path.is_file()
    assert [r["i"] for r in ts.read_log(key).records] == [0, 1, 2]
    assert backend._clear_base_pending_path(path).exists()
    assert not backend._clear_base_path(path).exists()

    ts.append(key, {"i": "after"})

    assert not backend._clear_base_pending_path(path).exists()
    page = ts.read_log(key)
    assert [r["i"] for r in page.records] == [0, 1, 2, "after"]
    assert page.corrupt == ()


def test_a_second_clear_log_after_a_crash_mid_staging_clears_normally(store, monkeypatch):
    """The crash-after-staging window above, met by a clear_log call instead of an append:
    the log is still present, so settling discards the stale pending stage unread and
    this call clears the log exactly as an uninterrupted clear would, staging and
    installing a watermark of its own."""
    only_on(store, FILE, _CLEAR_LOG_RACE)
    backend = store.backend
    key = store.key(LOG, "crash-after-staging-then-clear")
    for i in range(3):
        ts.append(key, {"i": i})
    path = store.path(key)
    old_size = path.stat().st_size

    class _SimulatedCrash(Exception):
        pass

    real_write_pending = backend._write_pending_clear_base

    def crash_after_write(*args, **kwargs):
        real_write_pending(*args, **kwargs)
        raise _SimulatedCrash

    monkeypatch.setattr(backend, "_write_pending_clear_base", crash_after_write)

    with pytest.raises(_SimulatedCrash):
        ts.clear_log(key)

    monkeypatch.setattr(backend, "_write_pending_clear_base", real_write_pending)

    removed = ts.clear_log(key)

    assert removed == 3
    assert not path.exists()
    assert backend._read_clear_base(path) == old_size


def test_settling_through_append_after_a_crash_installs_the_pending_watermark(
    store, monkeypatch,
):
    """The crash test above, carried one step further: instead of reading the log next,
    append to it. The marker's own crash-recovery path runs first, installing the pending
    watermark before the new entry is written, so the log holds exactly the one new entry
    and a replay from the pre-crash cursor returns it rather than a fragment of stale
    bytes read against an unadvanced base. The marker assertion below is the guard: the
    append settles the pending watermark onto the marker. The pending-file-gone assertion
    beside it documents the state rather than guarding anything, since nothing writes a
    pending file once the append has settled it."""
    only_on(store, FILE, _CLEAR_LOG_RACE)
    backend = store.backend
    key = store.key(LOG, "settle-through-append")
    for i in range(3):
        ts.append(key, {"i": i})
    before = ts.read_log(key)
    path = store.path(key)
    old_size = path.stat().st_size

    class _SimulatedCrash(Exception):
        pass

    real_remove = backend._remove_entry

    def crash_after_remove(*args, **kwargs):
        real_remove(*args, **kwargs)
        raise _SimulatedCrash

    monkeypatch.setattr(backend, "_remove_entry", crash_after_remove)

    with pytest.raises(_SimulatedCrash):
        ts.clear_log(key)

    ts.append(key, {"i": "after"})

    assert backend._read_clear_base(path) == old_size
    assert not backend._clear_base_pending_path(path).exists()

    full = ts.read_log(key)
    assert [r["i"] for r in full.records] == ["after"]

    resumed = ts.read_log(key, after=before.cursor)
    assert [r["i"] for r in resumed.records] == ["after"]


def test_a_second_clear_log_after_a_crash_settles_and_returns_zero(store, monkeypatch):
    """The base must never advance twice for one clear: a second clear_log call after the
    crash above settles the pending watermark itself (there is no file left to clear) and
    reports nothing removed, rather than computing a fresh base from bytes that are no
    longer there."""
    only_on(store, FILE, _CLEAR_LOG_RACE)
    backend = store.backend
    key = store.key(LOG, "second-clear-after-crash")
    for i in range(3):
        ts.append(key, {"i": i})
    path = store.path(key)
    old_size = path.stat().st_size

    class _SimulatedCrash(Exception):
        pass

    real_remove = backend._remove_entry

    def crash_after_remove(*args, **kwargs):
        real_remove(*args, **kwargs)
        raise _SimulatedCrash

    monkeypatch.setattr(backend, "_remove_entry", crash_after_remove)

    with pytest.raises(_SimulatedCrash):
        ts.clear_log(key)

    monkeypatch.setattr(backend, "_remove_entry", real_remove)

    removed = ts.clear_log(key)

    assert removed == 0
    assert backend._read_clear_base(path) == old_size
    assert not backend._clear_base_pending_path(path).exists()


def test_a_pending_file_holding_non_integer_bytes_refuses_settling(store):
    """A pending file nothing on this platform writes, beside a log that is absent: append
    refuses rather than guessing at the watermark it would install, naming the pending
    file in the refusal, while read_log keeps answering from the marker, which the
    pending file's presence or content never changes."""
    only_on(store, FILE, _CLEAR_LOG_RACE)
    backend = store.backend
    key = store.key(LOG, "garbled-pending")
    ts.append(key, {"i": 0})
    path = store.path(key)
    pending_path = backend._clear_base_pending_path(path)
    path.unlink()
    pending_path.write_bytes(b"not-a-number")

    with pytest.raises(ts.DecodeError, match=re.escape(str(pending_path))):
        ts.append(key, {"i": 1})

    page = ts.read_log(key)
    assert page.records == []


def test_a_pending_file_beside_a_present_log_is_discarded_unread_on_append(store):
    """A pending file left behind by a crash between staging and the unlink finds the log
    still present: the clear it staged never committed, so append discards it without
    trying to parse it, whatever garbage it holds, and the marker is untouched."""
    only_on(store, FILE, _CLEAR_LOG_RACE)
    backend = store.backend
    key = store.key(LOG, "garbled-pending-log-present")
    for i in range(2):
        ts.append(key, {"i": i})
    path = store.path(key)
    marker_path = backend._clear_base_path(path)
    pending_path = backend._clear_base_pending_path(path)
    pending_path.write_bytes(b"not-a-number")
    assert not marker_path.exists()

    ts.append(key, {"i": "after"})

    assert not pending_path.exists()
    assert not marker_path.exists()
    page = ts.read_log(key)
    assert [r["i"] for r in page.records] == [0, 1, "after"]
    assert page.corrupt == ()


def test_a_pending_file_is_invisible_to_enumeration_and_reported_as_bookkeeping(store):
    """A pending clear-base watermark must never surface as a log entry: not to the
    backend's own enumeration, not to adoption's plan. Both already exclude it through
    this store's own ``.jsonl`` suffix match, a fact independent of the file backend's
    bookkeeping predicate. What that predicate actually guards is ``account_for``, which
    walks every file unconditionally rather than matching a store's own shape: without
    the predicate naming the pending suffix, it would report the file as unaccounted
    instead of sorting it into bookkeeping."""
    only_on(store, FILE, _CLEAR_LOG_RACE)
    from tcip_mcp.tools.bundle import account_for
    from tcip_store.adoption import plan_root, unaccounted_files
    from tests._store_worker import CONTRACT_LAYOUT

    backend = store.backend
    key = store.key(LOG, "pending-visible")
    ts.append(key, {"i": 0})
    path = store.path(key)
    pending_path = backend._clear_base_pending_path(path)
    pending_path.write_bytes(b"3")

    assert backend.keys(LOG, str(store.root)) == [key]

    plan = plan_root(str(store.root), CONTRACT_LAYOUT)
    assert pending_path not in plan.claimed
    assert unaccounted_files((plan,)) == ()

    accounting = account_for(store.root)
    assert pending_path in accounting.bookkeeping
    assert pending_path not in accounting.unaccounted


_LOG_FILE_MECHANICS = (
    "a torn tail is bytes left in a file by an appender that died mid-write, reached here "
    "through the path the file backend places the log at"
)


def test_a_torn_tail_is_held_back_and_an_interior_corruption_is_reported(store):
    only_on(store, FILE, _LOG_FILE_MECHANICS)
    key = store.key(LOG, "damaged")
    for i in range(3):
        ts.append(key, {"i": i})
    path = store.path(key)

    lines = path.read_bytes().split(b"\n")
    lines[1] = b'{"i": bro'
    path.write_bytes(b"\n".join(lines))
    interior = ts.read_log(key)
    assert [r["i"] for r in interior.records] == [0, 2]
    assert interior.corrupt == (1,)
    assert not interior.torn_tail

    with open(path, "ab") as handle:
        handle.write(b'{"i": 3, "par')
    torn = ts.read_log(key)
    assert [r["i"] for r in torn.records] == [0, 2]
    assert torn.torn_tail
    assert int(torn.cursor) == len(b"\n".join(lines))


def test_an_appender_repairs_a_torn_tail_before_adding_its_own_entry(store):
    only_on(store, FILE, _LOG_FILE_MECHANICS)
    key = store.key(LOG, "repaired")
    ts.append(key, {"i": 0})
    path = store.path(key)
    with open(path, "ab") as handle:
        handle.write(b'{"i": 1, "pad": "' + b"x" * 20000)

    ts.append(key, {"i": 2})

    page = ts.read_log(key)
    assert [r["i"] for r in page.records] == [0, 2]
    assert not page.torn_tail and page.corrupt == ()


def test_an_appender_repairs_a_log_that_is_nothing_but_a_fragment(store):
    only_on(store, FILE, _LOG_FILE_MECHANICS)
    key = store.key(LOG, "all-fragment")
    ts.append(key, {"i": 0})
    path = store.path(key)
    path.write_bytes(b'{"i": 0, "unfin')

    ts.append(key, {"i": 1})

    page = ts.read_log(key)
    assert [r["i"] for r in page.records] == [1]
    assert not page.torn_tail and page.corrupt == ()


def test_clear_log_repairs_a_torn_tail_before_counting_and_removing(store):
    only_on(store, FILE, _LOG_FILE_MECHANICS)
    key = store.key(LOG, "damaged")
    for i in range(3):
        ts.append(key, {"i": i})
    path = store.path(key)
    with open(path, "ab") as handle:
        handle.write(b'{"i": 3, "par')

    removed = ts.clear_log(key)

    assert removed == 3
    ts.append(key, {"i": "after"})
    assert [r["i"] for r in ts.read_log(key).records] == ["after"]


def test_a_committed_entry_is_never_a_torn_tail_and_a_damaged_one_is_still_reported(store):
    """What replaces the torn-tail cases where an entry is a row rather than a line of a file.

    An entry is committed or it is not there, so no read can catch a partial one and
    ``torn_tail`` is structurally False. ``corrupt`` still has to answer, because the tuning
    route branches on both fields and a metrics stream that drops a row and one that says it
    dropped a row are different things.
    """
    only_on(store, SQLITE, "there is no partial row for a reader to catch, which is the fact "
                           "this pins; the cases above pin the file backend's torn tail")
    key = store.key(LOG, "damaged")
    for i in range(3):
        ts.append(key, {"i": i})
    assert not ts.read_log(key).torn_tail

    store.damage_log_entry(key, 1, b'{"i": bro')
    page = ts.read_log(key)
    assert [r["i"] for r in page.records] == [0, 2]
    assert page.corrupt == (1,)
    assert not page.torn_tail

    ts.append(key, {"i": 3})
    resumed = ts.read_log(key, after=page.cursor)
    assert [r["i"] for r in resumed.records] == [3]
    assert not resumed.torn_tail and resumed.corrupt == ()


# ── refusals, each with the call it must still admit ────────────────────────────


def test_an_unregistered_store_refuses_and_names_what_is_registered(store):
    with pytest.raises(ts.UnknownStore) as raised:
        ts.read(ts.Key("no_such_store", str(store.root), ("x",)))
    assert "Import the module that declares it" in str(raised.value)
    assert LWW in str(raised.value)

    assert ts.read(store.key(LWW, "x"), default={"ok": True}) == {"ok": True}


def test_every_operation_refuses_before_a_backend_is_bound(store):
    key = store.key(LWW, "unbound")
    ts.unbind()

    with pytest.raises(ts.StoreNotBound) as raised:
        ts.read(key, default=None)
    assert "entry point" in str(raised.value)
    with pytest.raises(ts.StoreNotBound):
        ts.replace(key, {"n": 1})

    ts.bind(store.backend)
    ts.replace(key, {"n": 1})
    assert ts.read(key) == {"n": 1}


def test_the_registry_refuses_a_declaration_that_would_be_ignored_or_shadowed():
    with pytest.raises(ValueError) as duplicate:
        ts.register_store(
            ts.StoreDescriptor(name=LWW, kind="record", key_fields=("name",),
                               codec=ts.RECORD_JSON, concurrency="last_writer_wins",
                               locator=RootedFileLocator(prefix=("shadow",), suffix=".json"))
        )
    assert "already registered" in str(duplicate.value)

    with pytest.raises(ValueError) as unpoliced:
        ts.register_store(
            ts.StoreDescriptor(name="contract_unpoliced", kind="record", key_fields=("name",),
                               codec=ts.RECORD_JSON,
                               locator=RootedFileLocator(prefix=("unpoliced",), suffix=".json"))
        )
    assert "concurrency=" in str(unpoliced.value)

    with pytest.raises(ValueError) as relaxed_log:
        ts.register_store(
            ts.StoreDescriptor(name="contract_relaxed_log", kind="log", key_fields=("name",),
                               codec=ts.LOG_JSON, durable=False,
                               locator=RootedFileLocator(prefix=("relaxed",), suffix=".jsonl"))
        )
    assert "durability" in str(relaxed_log.value)

    declared = ts.register_store(
        ts.StoreDescriptor(
            name="contract_late_declaration",
            kind="record",
            key_fields=("name",),
            codec=ts.RECORD_JSON,
            concurrency="last_writer_wins",
            locator=RootedFileLocator(prefix=("late",), suffix=".json"),
        )
    )
    assert declared.declared_in == __name__
    assert "contract_late_declaration" in ts.registered_stores()


def test_the_registry_refuses_a_record_that_declares_no_locator_and_admits_one_that_does():
    """A locator is the store's own statement of the file it owns, so a record without one
    can be written but never written back out as the file the tools reading the layout
    expect. Blobs are unaffected: their bytes are files wherever they are stored."""
    with pytest.raises(ValueError) as unplaceable:
        ts.register_store(
            ts.StoreDescriptor(
                name="contract_unplaceable",
                kind="record",
                key_fields=("name",),
                codec=ts.RECORD_JSON,
                concurrency="last_writer_wins",
            )
        )
    assert "locator" in str(unplaceable.value)
    assert "contract_unplaceable" not in ts.registered_stores()

    declared = ts.register_store(
        ts.StoreDescriptor(
            name="contract_placeable",
            kind="record",
            key_fields=("name",),
            codec=ts.RECORD_JSON,
            concurrency="last_writer_wins",
            locator=RootedFileLocator(prefix=("placeable",), suffix=".json"),
        )
    )
    assert declared.locator is not None
    assert "contract_placeable" in ts.registered_stores()


def test_the_registry_refuses_a_bespoke_json_spelling_and_admits_a_stated_exemption():
    """A store cannot quietly pick its own spelling: the check is in ``register_store``, so a
    module nothing has imported is caught too, and an exemption is written down where a
    reader of the declaration finds it."""
    from tcip_store.registry import _JsonCodec

    bespoke = _JsonCodec(indent=4, ensure_ascii=True, default=str, allow_nan=True,
                         sort_keys=True, trailing_newline=False)

    with pytest.raises(ValueError) as refused:
        ts.register_store(
            ts.StoreDescriptor(name="contract_bespoke_codec", kind="record",
                               key_fields=("name",), codec=bespoke,
                               concurrency="last_writer_wins",
                               locator=RootedFileLocator(prefix=("bespoke",), suffix=".json"))
        )
    assert "RECORD_JSON" in str(refused.value)
    assert "contract_bespoke_codec" not in ts.registered_stores()

    declared = ts.register_store(
        ts.StoreDescriptor(
            name="contract_stated_exemption",
            kind="record",
            key_fields=("name",),
            codec=bespoke,
            codec_exemption="the scaffolding this store exists for is not JSON",
            concurrency="last_writer_wins",
            locator=RootedFileLocator(prefix=("exempt",), suffix=".txt"),
        )
    )
    assert declared.codec_exemption


def test_an_operation_refuses_a_store_of_the_wrong_kind(store):
    record = store.key(LWW, "record")
    log = store.key(LOG, "log")

    with pytest.raises(ts.WrongKind) as replacing_a_log:
        ts.replace(log, {"n": 1})
    assert "append / read_log" in str(replacing_a_log.value)
    with pytest.raises(ts.WrongKind) as appending_to_a_record:
        ts.append(record, {"n": 1})
    assert "replace" in str(appending_to_a_record.value)
    with pytest.raises(ts.WrongKind) as clearing_a_record:
        ts.clear_log(record)
    assert "replace" in str(clearing_a_record.value)

    ts.replace(record, {"n": 1})
    ts.append(log, {"n": 1})
    assert ts.read(record) == {"n": 1}
    assert len(ts.read_log(log).records) == 1


def test_a_key_of_the_wrong_arity_refuses_and_names_the_key_fields(store):
    with pytest.raises(ts.BadKey) as raised:
        ts.read(ts.Key(NESTED, str(store.root), ("only-one",)))
    assert "['group', 'name']" in str(raised.value)

    ts.replace(store.key(NESTED, "group", "name"), {"n": 1})
    assert ts.read(store.key(NESTED, "group", "name")) == {"n": 1}


def test_absence_and_corruption_are_different_answers(store):
    """Backend-general, and it stays that way: an unreadable measurement record presenting as an
    absent one is the failure the no-silent-fallback invariant rests on, so every backend
    answers it, with the damage reaching whatever that backend actually reads."""
    key = store.key(LWW, "sometimes-there")
    with pytest.raises(ts.NotFound) as raised:
        ts.read(key)
    assert "default=" in str(raised.value)
    assert ts.read(key, default={"fallback": True}) == {"fallback": True}

    ts.replace(key, {"n": 1})
    assert ts.read(key) == {"n": 1}
    store.damage_record(key, b"{not json at all")
    with pytest.raises(ts.DecodeError):
        ts.read(key, default={"fallback": True})


def test_a_relative_root_is_refused_before_it_resolves_against_a_working_directory(store):
    """Every backend has to refuse first and canonicalize second, or the refusal turns into a
    guess about which directory the process happened to be started in.

    Enumeration refuses alongside the reads and writes, on both backends. An empty list back
    from a root that names nothing reads as "this root holds no entries", which is the
    silent-fallback shape: a caller asking whether any verdict exists would be told no.
    """
    relative_root = "not/an/absolute/root"
    relative = ts.Key(LWW, relative_root, ("x",))

    with pytest.raises(ts.BadKey) as reading:
        ts.read(relative, default=None)
    assert "absolute" in str(reading.value)
    with pytest.raises(ts.BadKey):
        ts.replace(relative, {"n": 1})
    with pytest.raises(ts.BadKey):
        ts.exists(relative)
    with pytest.raises(ts.BadKey) as enumerating:
        ts.keys(LWW, relative_root)
    assert "absolute" in str(enumerating.value)

    absolute = store.key(LWW, "absolute")
    ts.replace(absolute, {"n": 1})
    assert ts.read(absolute) == {"n": 1}
    assert ts.keys(LWW, str(store.root)) == [absolute]


def test_a_key_part_carrying_non_ascii_or_a_separator_round_trips(store):
    """Parts are opaque identity, not path segments a backend may reinterpret."""
    key = store.key(NESTED, "grüne/reihe", "ü_2")
    held = ts.replace(key, {"note": "ü"})

    assert ts.read(key) == {"note": "ü"}
    assert ts.read_versioned(key).version == held
    assert ts.read(store.key(NESTED, "grüne", "reihe"), default=None) is None


def test_a_committed_record_is_readable_by_another_process(store):
    key = store.key(LWW, "handed-over")
    ts.replace(key, {"n": 7})
    result = store.root / "handed-over.json"

    reader = store.spawn("read-record", store.root, LWW, "handed-over", result)
    assert reader.wait(timeout=60) == 0
    assert json.loads(result.read_text(encoding="utf-8")) == {"value": {"n": 7}}


def test_a_transaction_refuses_every_form_that_would_escape_it(store):
    first = store.key(LWW, "alpha")
    second = store.key(LWW, "beta")
    unheld = store.key(LWW, "gamma")

    with ts.transaction(first, second) as txn:
        with pytest.raises(ts.TransactionMisuse) as nested:
            with ts.transaction(first):
                pass
        assert "transaction(a, b)" in str(nested.value)
        with pytest.raises(ts.TransactionMisuse) as outside_write:
            ts.replace(first, {"n": 1})
        assert "txn.write" in str(outside_write.value)
        with pytest.raises(ts.TransactionMisuse):
            txn.read(unheld)
        txn.write(first, {"n": 1})
        txn.write(second, {"n": 2})

    assert ts.read(first) == {"n": 1}
    assert ts.read(second) == {"n": 2}


def test_a_transaction_refuses_two_roots_and_admits_two_spellings_of_one(store):
    """One transaction is one root's exclusion: a backend holding a database per root would
    have to commit the second root's write somewhere the first root's transaction cannot
    reach, so the seam refuses before any backend infers a root from the first key. Two
    spellings of one directory are one root, or the refusal would reject work that is fine."""
    elsewhere = store.root / "elsewhere"
    elsewhere.mkdir()
    here = store.key(LWW, "here")
    there = ts.Key(LWW, str(elsewhere), ("there",))

    with pytest.raises(ts.TransactionMisuse) as raised:
        with ts.transaction(here, there):
            pass
    assert repr(str(store.root)) in str(raised.value)
    assert repr(str(elsewhere)) in str(raised.value)
    assert ts.read(here, default=None) is None

    detour = store.root / "detour"
    detour.mkdir()
    spelled_around = ts.Key(LWW, str(detour / ".."), ("beta",))
    with ts.transaction(store.key(LWW, "alpha"), spelled_around) as txn:
        txn.write(store.key(LWW, "alpha"), {"n": 1})
        txn.write(spelled_around, {"n": 2})

    assert ts.read(store.key(LWW, "alpha")) == {"n": 1}
    assert ts.read(store.key(LWW, "beta")) == {"n": 2}


def test_an_append_inside_a_transaction_is_refused_and_the_same_append_outside_it_lands(store):
    """An append returns only once its entry has survived, which a transaction that rolls
    back would take away again, and a log key cannot be named in a transaction to begin
    with."""
    record = store.key(LWW, "under-transaction")
    log = store.key(LOG, "appended-to")

    with ts.transaction(record) as txn:
        with pytest.raises(ts.TransactionMisuse) as raised:
            ts.append(log, {"i": "inside"})
        assert "close the transaction first" in str(raised.value)
        with pytest.raises(ts.TransactionMisuse) as clearing:
            ts.clear_log(log)
        assert "close the transaction first" in str(clearing.value)
        txn.write(record, {"n": 1})

    ts.append(log, {"i": "outside"})
    assert [entry["i"] for entry in ts.read_log(log).records] == ["outside"]
    assert ts.read(record) == {"n": 1}


def test_a_transaction_over_blob_keys_commits_them_together_and_refuses_mixing_kinds(store):
    """A transaction may name blobs as it names records: it reads each as bytes, stages each
    write, and commits every one or, on a raise inside it, none. A set mixing a record and a blob
    refuses before anything is held."""
    first, second = store.key(BLOB, "first"), store.key(BLOB, "second")
    ts.put_blob(second, b"kept")

    with pytest.raises(RuntimeError):
        with ts.transaction(first, second) as txn:
            txn.write(first, b"staged")
            raise RuntimeError("the caller found a conflict")
    assert not ts.exists(first)
    assert ts.read_blob_versioned(second).value == b"kept"

    with ts.transaction(first, second) as txn:
        assert txn.read(first, default=None) is None
        assert txn.read(second) == b"kept"
        txn.write(first, b"one")
        txn.write(second, b"two")
    assert ts.read_blob_versioned(first).value == b"one"
    assert ts.read_blob_versioned(second).value == b"two"

    with pytest.raises(ts.TransactionMisuse) as mixed:
        with ts.transaction(store.key(LWW, "record"), first):
            pass
    assert "all records or all blobs" in str(mixed.value)


def test_a_backend_refuses_to_exist_without_cross_process_locking(store, monkeypatch):
    monkeypatch.setitem(sys.modules, "filelock", None)
    with pytest.raises(ts.BackendUnavailable) as raised:
        FileBackend()
    assert "filelock" in str(raised.value)

    monkeypatch.undo()
    assert FileBackend().capabilities().cross_machine_exclusion is False


def test_a_blob_path_is_refused_unless_both_the_backend_and_the_store_declare_it(store):
    open_blob_key = store.key(BLOB, "readable")
    sealed = store.key(SEALED_BLOB, "not-readable")
    ts.put_blob(open_blob_key, b"bytes")
    ts.put_blob(sealed, b"bytes")

    with pytest.raises(ts.CapabilityUnavailable) as by_store:
        ts.blob_path(sealed)
    assert SEALED_BLOB in str(by_store.value)

    class _NoLocalPaths(FileBackend):
        def capabilities(self):
            return ts.Capabilities(
                multi_key_atomic_commit=False,
                cross_machine_exclusion=False,
                durable_replace=False,
                durable_append=True,
                local_blob_paths=False,
            )

    ts.bind(_NoLocalPaths())
    with pytest.raises(ts.CapabilityUnavailable) as by_backend:
        ts.blob_path(open_blob_key)
    assert "local_blob_paths" in str(by_backend.value)

    ts.bind(store.backend)
    assert ts.blob_path(open_blob_key).read_bytes() == b"bytes"


def test_an_unenumerable_store_refuses_rather_than_answering_none(store):
    with pytest.raises(ts.ListingUnsupported) as raised:
        ts.keys(OPAQUE, str(store.root))
    assert OPAQUE in str(raised.value)

    assert ts.keys(LWW, str(store.root)) == []
    ts.replace(store.key(NESTED, "g1", "a"), {"n": 1})
    ts.replace(store.key(NESTED, "g1", "b"), {"n": 2})
    ts.replace(store.key(NESTED, "g2", "c"), {"n": 3})
    assert ts.keys(NESTED, str(store.root), ("g1",)) == [
        store.key(NESTED, "g1", "a"),
        store.key(NESTED, "g1", "b"),
    ]


def test_a_contended_key_is_named_and_released_when_its_holder_dies(store):
    key = store.key(LWW, "orphanable")
    ready = store.root / "holder.ready"
    holder = store.spawn("hold-lock", store.root, "orphanable", ready)
    wait_for(ready, timeout_s=60)

    with pytest.raises(ts.StoreBusy) as raised:
        with ts.transaction(key, timeout_s=0.3):
            pass
    assert "orphanable" in str(raised.value)
    assert raised.value.blocked_on == key
    assert ts.read(key, default=None) is None

    holder.kill()
    holder.wait(timeout=30)
    with ts.transaction(key, timeout_s=10) as txn:
        txn.write(key, {"who": "successor"})
    assert ts.read(key) == {"who": "successor"}


def test_an_unconditional_write_is_refused_where_the_policy_says_compare_and_set(store):
    with pytest.raises(ts.PolicyViolation) as raised:
        ts.replace(store.key(CAS, "guarded"), {"n": 1})
    assert "read_versioned" in str(raised.value)
    with pytest.raises(ts.PolicyViolation):
        ts.delete(store.key(CAS, "guarded"))

    ts.replace(store.key(CAS, "guarded"), {"n": 1}, expect=ts.Version.ABSENT)
    ts.replace(store.key(LWW, "open"), {"n": 1})
    assert ts.read(store.key(CAS, "guarded")) == {"n": 1}
    assert ts.read(store.key(LWW, "open")) == {"n": 1}


def test_a_version_read_before_a_transaction_committed_is_refused_afterwards(store):
    key = store.key(CAS, "moved-under-us")
    ts.replace(key, {"who": "seed"}, expect=ts.Version.ABSENT)
    stale = ts.read_versioned(key).version
    ready = store.root / "holder.ready"

    holder = store.spawn("hold-transaction", store.root, CAS, "moved-under-us", "insider", 0.1, ready)
    assert holder.wait(timeout=60) == 0

    with pytest.raises(ts.VersionConflict) as raised:
        ts.replace(key, {"who": "outsider"}, expect=stale)
    assert raised.value.actual == ts.read_versioned(key).version
    assert ts.read(key) == {"who": "insider"}


def test_a_transaction_reads_its_own_staged_write_and_shows_it_to_nobody_else(store):
    key = store.key(LWW, "staged")
    ts.replace(key, {"n": 1})
    result = store.root / "seen.json"

    with ts.transaction(key) as txn:
        txn.write(key, {"n": 2})
        assert txn.read(key) == {"n": 2}
        reader = store.spawn("read-record", store.root, LWW, "staged", result)
        assert reader.wait(timeout=60) == 0
        assert json.loads(result.read_text(encoding="utf-8")) == {"value": {"n": 1}}

    assert ts.read(key) == {"n": 2}


# ── file backend annex ──────────────────────────────────────────────────────────

_FILE_LAYOUT = (
    "placement on disk and the flushes that make it durable are the file backend's own "
    "mechanics, and a backend keying on (store, root, parts) has no path to check"
)


def test_a_rooted_locator_inverts_its_own_placement(store):
    only_on(store, FILE, _FILE_LAYOUT)
    locator = RootedFileLocator(prefix=("annotations",), suffix=".json")
    parts = ("2026-03-04", "img_0001")
    relative = locator.relative_path(str(store.root), parts)

    assert relative == PurePosixPath("annotations/2026-03-04/img_0001.json")
    assert locator.parts_from(relative) == parts
    assert locator.parts_from(PurePosixPath("elsewhere/img_0001.json")) is None
    assert locator.parts_from(PurePosixPath("annotations/img_0001.txt")) is None

    key = store.key(NESTED, "2026-03-04", "img_0001")
    ts.replace(key, {"n": 1})
    assert store.path(key) == store.root / "nested" / "2026-03-04" / "img_0001.json"


def test_a_codec_round_trips_the_value_it_encoded():
    """Decoding what a codec wrote returns what went in, for both JSON kinds and for text."""
    payload = {"b": 1, "a": "ü", "nested": {"ratio": 0.5}, "absent": None}

    assert ts.RECORD_JSON.decode(ts.RECORD_JSON.encode(payload)) == payload
    assert ts.LOG_JSON.decode(ts.LOG_JSON.encode(payload)) == payload
    assert ts.text_codec().encode("port 8765") == b"port 8765"
    assert ts.text_codec(trailing_newline=True).encode("ü") == "ü\n".encode("utf-8")


class _RecordingBackend(FileBackend):
    """Records the durability calls a write makes, in the order it makes them."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.events: list[tuple[str, str]] = []

    def _fsync_file(self, handle):
        self.events.append(("fsync_file", ""))
        super()._fsync_file(handle)

    def _fsync_dir(self, directory):
        self.events.append(("fsync_dir", Path(directory).name))
        super()._fsync_dir(directory)

    def _apply_staged(self, temp, path, *, durable):
        self.events.append(("replace", path.name))
        super()._apply_staged(temp, path, durable=durable)


def test_a_relaxed_store_flushes_nothing_and_a_durable_one_flushes_both(store):
    only_on(store, FILE, _FILE_LAYOUT)
    backend = _RecordingBackend()
    ts.bind(backend)
    relaxed = store.key(RELAXED, "heartbeat")
    durable = store.key(LWW, "durable")
    ts.replace(relaxed, {"n": 0})
    ts.replace(durable, {"n": 0})

    backend.events.clear()
    ts.replace(relaxed, {"n": 1})
    assert [event for event, _ in backend.events] == ["replace"]

    backend.events.clear()
    ts.replace(durable, {"n": 1})
    assert [event for event, _ in backend.events] == ["fsync_file", "replace", "fsync_dir"]
    assert ts.capabilities().durable_replace is (os.name != "nt")

    backend.events.clear()
    ts.replace(store.key(NESTED, "fresh-group", "record"), {"n": 1})
    flushed = [name for event, name in backend.events if event == "fsync_dir"]
    assert flushed[:3] == [store.root.name, "nested", "fresh-group"]


def test_a_transaction_flushes_each_parent_before_the_next_replace(store):
    only_on(store, FILE, _FILE_LAYOUT)
    backend = _RecordingBackend()
    ts.bind(backend)
    first = store.key(NESTED, "group", "alpha")
    second = store.key(LWW, "beta")

    with ts.transaction(first, second) as txn:
        txn.write(second, {"n": 2})
        txn.write(first, {"n": 1})
    applied = [event for event in backend.events if event[0] in ("replace", "fsync_dir")]

    assert applied[-4:] == [
        ("replace", "alpha.json"),
        ("fsync_dir", "group"),
        ("replace", "beta.json"),
        ("fsync_dir", "lww"),
    ]


def test_lock_files_and_staged_temp_files_are_never_keys(store):
    only_on(store, FILE, _FILE_LAYOUT)
    key = store.key(LWW, "enumerated")
    ts.replace(key, {"n": 1})
    directory = store.path(key).parent

    (directory / "enumerated.json.lock").write_bytes(b"")
    (directory / ".enumerated.json.abc123.tmp").write_bytes(b"{}")
    (directory / "tmpabc123.tmp").write_bytes(b"{}")

    assert ts.keys(LWW, str(store.root)) == [key]


# ── database backend annex ──────────────────────────────────────────────────────

_DATABASE_MECHANICS = (
    "the subject is what one database per root does, and the file backend has no database "
    "for any of it to be true of"
)


def test_the_database_backend_declares_exactly_these_guarantees(store):
    """The database backend's declared capabilities, exactly."""
    only_on(store, SQLITE, _DATABASE_MECHANICS)
    assert ts.capabilities() == ts.Capabilities(
        multi_key_atomic_commit=True,
        cross_machine_exclusion=False,
        durable_replace=os.name != "nt",
        durable_append=True,
        local_blob_paths=True,
    )


def test_a_database_and_its_sidecars_are_never_keys_of_the_store_they_sit_inside(store):
    """Enumerating a store whose entries share ``.tcip/`` with the database returns none of the
    database's own files or its build temp."""
    only_on(store, SQLITE, _DATABASE_MECHANICS)
    ts.replace(store.key(LWW, "opens-the-database"), {"n": 1})
    entry = store.key(STATE_FILES, "note.txt")
    ts.put_blob(entry, b"an entry of a store whose files share the database's directory")

    tcip_dir = store.root / ".tcip"
    (tcip_dir / creation_temp_name(DATABASE_FILENAME, "abc123")).write_bytes(b"")
    present = {path.name for path in tcip_dir.iterdir()}
    assert {DATABASE_FILENAME, f"{DATABASE_FILENAME}-wal", f"{DATABASE_FILENAME}-shm"} <= present

    assert ts.keys(STATE_FILES, str(store.root)) == [entry]


def test_two_spellings_of_one_root_address_one_database(store):
    """Two spellings of one root share one connection slot and one database."""
    only_on(store, SQLITE, _DATABASE_MECHANICS)
    detour = store.root / "detour"
    detour.mkdir()
    spelled_around = ts.Key(LWW, str(detour / ".."), ("shared",))

    ts.replace(store.key(LWW, "shared"), {"n": 1})
    assert ts.read(spelled_around) == {"n": 1}
    ts.replace(spelled_around, {"n": 2}, expect=ts.read_versioned(spelled_around).version)
    assert ts.read(store.key(LWW, "shared")) == {"n": 2}

    slots = list(store.backend._connections)
    assert len(slots) == 1
    assert slots[0][2] == ts.canonical_path(str(store.root))
    assert slots[0][2] == ts.canonical_path(str(detour / ".."))
    assert list(store.root.rglob(DATABASE_FILENAME)) == [
        store.root / ".tcip" / DATABASE_FILENAME
    ]


def test_traits_shares_the_state_database_rather_than_gaining_its_own(store):
    """Writing a ``traits`` record alongside a sibling ``STATE``-rooted store
    (``delivery_events``) creates exactly one database under the project's shared state root,
    never a second, nested one under ``traits`` itself.
    """
    only_on(store, SQLITE, _DATABASE_MECHANICS)
    ts.replace(traits.trait_key(store.root, "bud_opening"), {"revisions": []},
               expect=ts.Version.ABSENT)
    ts.replace(delivery.delivery_event_key(store.root, "e1"), {"event_id": "e1"})

    databases = sorted(store.root.rglob(DATABASE_FILENAME))
    assert databases == [store.root / ".tcip" / "state" / ".tcip" / DATABASE_FILENAME]
    assert not (store.root / ".tcip" / "state" / "traits" / ".tcip").exists()


def test_a_key_part_carrying_non_ascii_or_a_separator_is_stored_under_one_spelling(store):
    """A key's parts are stored as ASCII-escaped compact JSON."""
    only_on(store, SQLITE, _DATABASE_MECHANICS)
    key = store.key(NESTED, "grüne/reihe", "ü_2")
    ts.replace(key, {"n": 1})

    conn = store.connect()
    try:
        stored = [
            row[0]
            for row in conn.execute("select parts from records where store = ?", (NESTED,))
        ]
    finally:
        conn.close()
    assert stored == ['["gr\\u00fcne/reihe","\\u00fc_2"]']
    assert ts.keys(NESTED, str(store.root)) == [key]


def test_a_cursor_returns_every_entry_once_while_appenders_are_still_writing(store):
    """Reading a log by cursor while two processes append returns every entry exactly once."""
    only_on(store, SQLITE, _DATABASE_MECHANICS)
    key = store.key(LOG, "streamed-live")
    workers = [store.spawn("append", store.root, "streamed-live", f"proc{n}", 20) for n in range(2)]

    seen: list[tuple[str, int]] = []
    cursor = None
    deadline = time.monotonic() + 120
    while len(seen) < 40:
        assert time.monotonic() < deadline, f"only {len(seen)} of 40 entries arrived"
        page = ts.read_log(key, after=cursor)
        seen.extend((r["tag"], r["i"]) for r in page.records)
        assert not page.torn_tail and page.corrupt == ()
        cursor = page.cursor
    for worker in workers:
        assert worker.wait(timeout=120) == 0

    assert ts.read_log(key, after=cursor).records == []
    assert len(set(seen)) == 40
    for tag in ("proc0", "proc1"):
        assert sorted(i for name, i in seen if name == tag) == list(range(20))


class _RecordingSyncBackend(SqliteBackend):
    """Records the synchronous level each write sets on its connection, in the order it sets it."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.levels: list[str] = []

    def _apply_synchronous(self, conn, level):
        self.levels.append(level)
        super()._apply_synchronous(conn, level)


def test_a_write_commits_at_the_synchronous_level_its_store_declares(store):
    """A write to a relaxed store commits at the relaxed synchronous level, a durable one at the
    durable level."""
    only_on(store, SQLITE, _DATABASE_MECHANICS)
    backend = _RecordingSyncBackend()
    ts.bind(backend)
    try:
        relaxed = store.key(RELAXED, "heartbeat")
        durable = store.key(LWW, "durable")
        ts.replace(durable, {"n": 0})

        def synchronous() -> int:
            with backend._connection(str(store.root), (LWW,)) as conn:
                return conn.execute("pragma synchronous").fetchone()[0]

        backend.levels.clear()
        ts.replace(relaxed, {"n": 1})
        assert backend.levels == ["NORMAL"]
        assert synchronous() == 1

        backend.levels.clear()
        ts.replace(durable, {"n": 1})
        assert backend.levels == ["FULL"]
        assert synchronous() == 2
    finally:
        backend.close()
        ts.bind(store.backend)


# ── the platform's own stores: same codec, same path ────────────────────────────

REPORT_UNDER_TEST = "20260304T120000Z_missing_tool_a1b2"
RETROSPECTIVE_UNDER_TEST = "project_under_test"
PROPOSAL_STEM = "a_1"
TRAIT_UNDER_TEST = "trait_under_test"
DELIVERY_KIND_UNDER_TEST = "state_crossing_dates"
EVENT_ID_UNDER_TEST = "a1b2c3d4e5f60718"
ACKNOWLEDGMENT_UNDER_TEST = "9e8d7c6b5a403122"
EXPERIMENT = "exp_042"
IMAGE_DATE = "2026-03-04"
IMAGE_STEM = "a_1"
IMAGE_EXT = ".JPG"

SUBJECT_REGISTRY_BYTES = (
    '{\n'
    '  "bud": {\n'
    '    "description": "a männlich flower",\n'
    '    "attributes": {}\n'
    '  }\n'
    '}\n'
).encode("utf-8")
"""The exact bytes of the documents whose owning module encodes them."""

DATASET_IDENTITY_BYTES = (
    '{\n'
    '  "dataset_id": "a1",\n'
    '  "crop": "currant",\n'
    '  "created": "2026-03-04T12:00:00+00:00"\n'
    '}\n'
).encode("utf-8")
BAND_GROUP_MANIFEST_BYTES = (
    '{\n'
    '  "bands": {\n'
    '    "Green": "cap_ü_G.tif",\n'
    '    "Red": "cap_ü_R.tif"\n'
    '  },\n'
    '  "source": "embedded-metadata",\n'
    '  "central_wavelength_nm": {\n'
    '    "Green": 560.0,\n'
    '    "Red": 650.0\n'
    '  }\n'
    '}\n'
).encode("utf-8")
IMAGE_BYTES = b"\xff\xd8\xff\xe0not a real frame, only bytes handed to the store\x00"
LABEL_BYTES = '{"annotations": [{"subject": "bud", "bbox": [1.0, 2.0, 3.5, 4.5]}]}'.encode("utf-8")


def _generic_label_dir(root: Path) -> Path:
    """A label tree no layout resolver describes, the shape a materialized split writes."""
    return root / "labels"


def _band_group_dir(root: Path) -> Path:
    return root / "images"


def _split_dir(root: Path) -> Path:
    """A split output directory, the shape a caller asks a partition to be written into."""
    return root / "splits"


@dataclass(frozen=True)
class Registered:
    """One registered store: a ``golden`` value of the shape it holds, the key it is written
    under (``key_of``), the path under the root its file lands at (``relative``), and the root
    its keys hang off (``root_of``)."""

    golden: object
    key_of: Callable[[Path], ts.Key]
    relative: str
    root_of: Callable[[Path], Path] = lambda root: root


def _construct_via_scratch_backend(build: Callable[[Path], dict]) -> dict:
    """Call ``build`` with a file backend bound and a scratch directory, both only for the call,
    and return what it built."""
    from tcip_store.file_backend import FileBackend

    ts.bind(FileBackend())
    try:
        with tempfile.TemporaryDirectory() as scratch:
            return build(Path(scratch))
    finally:
        ts.unbind()


def _real_selection() -> dict:
    """The record ``selection.write_selection`` writes, into a scratch project."""
    from tcip_mcp.pipelines.data.selection import (
        ClassScope, Sample, Selection, selection_key, write_selection,
    )

    def build(scratch: Path) -> dict:
        write_selection(scratch, Selection(
            samples=(
                Sample(member="a_1", source=str(scratch / "ü/images/2026-03-04/a_1.jpg"),
                       ground_truth=str(scratch / "ü/annotations/2026-03-04/a_1.json"),
                       group="a", side="train", ground_truth_digest="7f3a1b9c2d4e5f60"),
            ),
            scope=ClassScope(subject="bud", id_map={"bud": 0}), seed=42,
            group_by="stem", dataset_fingerprint="7ac1",
        ), project=scratch)
        return ts.read(selection_key(scratch))

    return _construct_via_scratch_backend(build)


REGISTERED = {
    "subject_registry": Registered(
        SUBJECT_REGISTRY_BYTES, dataset_layout.subject_registry_key, "subjects.json"),
    "dataset_identity": Registered(
        DATASET_IDENTITY_BYTES,
        dataset_layout.dataset_identity_key, "dataset.json"),
    "review_verdicts": Registered(
        {"proposal": 0, "action": "accepted", "by": "user:ü", "at": "2026-03-04T12:00:00+00:00"},
        lambda root: verdicts.verdict_key(project_state_dir(root), "predictions", "a_1.jpg"),
        ".tcip/state/review/predictions/a_1.jpg.jsonl", root_of=project_state_dir),
    "canvas_meta": Registered(
        {"tab": "annotate", "image": "ü.jpg"},
        lambda root: canvas.canvas_meta_key(str(root)), ".tcip/state/canvas_live.json"),
    "canvas_geometry": Registered(
        {"image_path": "ü.jpg", "tab": "annotate", "shapes": []},
        lambda root: canvas.canvas_geometry_key(str(root)), ".tcip/state/canvas_shapes.json"),
    "model_registry": Registered(
        [{"name": "detector_v1", "sha256": "0" * 64, "metrics": {"val_map50": 0.61}}],
        model_registry.registry_index_key, ".tcip/models/registry.json"),
    "workspace_last_opened": Registered(
        "3f9a1c7e5b20\n", workspace.last_opened_key, ".tcip/state/last_opened.txt"),
    "gui_snapshot": Registered(
        {"active_tab": "annotate", "dataset": {"subject": "büsch"}},
        web_client.gui_snapshot_key, ".tcip/state/gui.json"),
    "project_status": Registered(
        {"last_activity": "2026-03-04T12:00:00+00:00", "reports_since_last_retrospective": 2},
        lambda root: project_status.project_status_key(root), ".tcip/state/project_status.json"),
    "project_record": Registered(
        {"id": "3f9a1c7e5b20", "display_name": "Nördliche Reihe", "site": "north orchard"},
        project_record.project_record_key, ".tcip/project.json"),
    "friction_reports": Registered(
        {"timestamp": "2026-03-04T12:00:00+00:00", "category": "missing_tool", "detail": "ü",
         "context": {}, "user_disagreement": False},
        lambda root: meta_tools.friction_report_key(str(root), REPORT_UNDER_TEST),
        f".tcip/reports/{REPORT_UNDER_TEST}.json"),
    "retrospectives": Registered(
        f"# {RETROSPECTIVE_UNDER_TEST}\n\n## Retrospective: 2026-03-04T12:00:00+00:00\n\nü\n",
        lambda root: meta_tools.retrospective_key(str(root), RETROSPECTIVE_UNDER_TEST),
        f".tcip/retrospectives/{RETROSPECTIVE_UNDER_TEST}.md"),
    "proposal_staging": Registered(
        {"engine": "sam", "candidates": [{"candidate_id": 1, "score": 0.9, "note": "ü"}]},
        lambda root: proposal_tools.proposal_staging_key(root, "2026-03-04", PROPOSAL_STEM),
        f".tcip/state/proposals/2026-03-04/{PROPOSAL_STEM}.json"),
    "backend_port": Registered(
        "8765", web_client.backend_port_key, ".tcip/state/web_port.txt"),
    "raster_pass_progress": Registered(
        {"schema_version": 1,
         "raster_identity": {"width": 100, "height": 100, "num_channels": 3, "dtype": "uint8",
                             "pixel_checksum": "ab12", "seed": 0, "window_size": 1024,
                             "max_windows": 8, "pixel_fraction": 1.0, "band_interpretations": None,
                             "geotransform": None},
         "checkpoint_sha256": "0" * 64, "experiment_id": None, "assessment_id": None,
         "tile_batch_size": 96, "require_masks": False,
         "execution": {"conf": 0.42, "max_dets": 1000, "tile_size": 512, "overlap": 0.2,
                       "tile_resize": None, "postprocess": "nms", "merge_type": "NMS",
                       "match_metric": "IOU", "cross_tile_nms": 0.5, "sahi_version": "0.11",
                       "sources": {"conf": "explicit", "tile_size": "explicit"}}},
        lambda root: ts.Key(inference_tools.RASTER_PASS_PROGRESS_STORE, str(root),
                            ("ab12cd34ef567890", "identity")),
        ".tcip/raster_pass_progress/ab12cd34ef567890/identity.json"),
    "traits": Registered(
        # One trait's record as the proposing and confirming producers leave it; the
        # producer-agreement module holds this golden's keys to what those producers write.
        {"revisions": [{
            "number": 1,
            "entry": {
                "name": TRAIT_UNDER_TEST, "delivers": ["measure_one"], "positive_value": "büsch",
                "milestone_fractions": [0.5], "milestone_on": "positive_fraction",
                "majority_milestone": "",
                "phenology_prefix": "", "majority_label": "", "count_objective": "",
                "localization": "", "localization_tolerance": "half_class_avg_size",
                "localization_tolerance_frac": 0.5, "count_bias_tolerance_frac": None,
                "count_error_tolerance": None, "classifier_agreement_floor": None,
                "ordinal_agreement_floor": None, "regression_skill_floor": None,
                "regression_criterion": "",
                "scale_tolerance_frac": None, "holdout_match_quality_floor": None, "notes": "ü",
                "operationalizations": {DELIVERY_KIND_UNDER_TEST: {
                    "statement": "the date each büsch reached the measured state",
                    "mechanism": "the calibrated state classifier over isolated buds",
                    "measured_subject": "bud", "delivered_phenotypes": ["measure_one"],
                    "delivered_value_keys": []}}},
            "entry_sha256": "7f3a1b9c2d4e5f60",
            "rationale": "the breeder described the state directly", "relayed_note": "",
            "proposed_at": "2026-03-04T12:00:00+00:00",
            "confirmed_by": "user:ü", "confirmed_at": "2026-03-04T12:30:00+00:00",
            "withdrawn_by": None, "withdrawn_at": None}]},
        lambda root: traits.trait_key(root, TRAIT_UNDER_TEST),
        f".tcip/state/traits/{TRAIT_UNDER_TEST}.json", root_of=project_state_dir),
    "annotation_records": Registered(
        LABEL_BYTES, lambda root: json_io.annotation_record_key(_generic_label_dir(root), "a_1"),
        "labels/a_1.json", root_of=_generic_label_dir),
    "annotation_stats": Registered(
        {"sessions": [{"user": "ü", "images_annotated": 1, "total_annotations": 3,
                       "total_time_seconds": 42.5}]},
        lambda root: web_client.annotation_stats_key(str(root)), ".tcip/state/annotation_stats.json"),
    "band_group_manifest": Registered(
        BAND_GROUP_MANIFEST_BYTES,
        lambda root: band_groups.band_group_manifest_key(_band_group_dir(root), "cap_ü"),
        f"images/cap_ü{band_groups.MANIFEST_EXT}", root_of=_band_group_dir),
    "ray_dashboard": Registered(
        {"url": "http://127.0.0.1:8265", "pid": 4242},
        hpo.ray_dashboard_key, ".tcip/state/ray_dashboard.json"),
    "dataset_registry": Registered(
        [{"id": "a1", "path": "dü", "crop": "currant", "fingerprint": "9f2c"}],
        project_tools.dataset_registry_key, ".tcip/datasets.json"),
    "selection": Registered(
        _real_selection(),
        lambda root: selection.selection_key(_split_dir(root)),
        "splits/selection.json", root_of=_split_dir),
    "audit_log": Registered(
        {"timestamp": "2026-03-04T12:00:00+00:00", "tool": "save_label_document",
         "arguments": {"image_path": "images/2026-03-04/a_1.JPG", "n_annotations": 3},
         "status": "ok"},
        lambda root: audit.audit_log_key(root), ".tcip/audit.jsonl"),
    "imagery": Registered(
        IMAGE_BYTES,
        lambda root: dataset_layout.image_key(root, IMAGE_DATE, IMAGE_STEM, IMAGE_EXT),
        f"images/{IMAGE_DATE}/{IMAGE_STEM}{IMAGE_EXT}"),
    "plant_mapping": Registered(
        {"2026-03-04": [{"image_path": "images/2026-03-04/a_1.JPG", "stem": "a_1",
                         "date_folder": "2026-03-04", "plot_name": "plot_ü",
                         "accession_name": "ü", "source": "sequence", "distance_m": 1.25}]},
        lambda root: plant_mapping.plant_mapping_key(root, "valley"),
        ".tcip/state/plant_mappings/valley.json", root_of=project_state_dir),
    "plant_registries": Registered(
        {"name": "valley-plants", "crop": "currant", "site": "north orchard",
         "csvs": [{"path": "dü/plants.csv", "sha256": "0" * 64, "n_plants": 2}],
         "n_plants": 2, "digest": "0" * 64,
         "registered_at": "2026-03-04T12:00:00+00:00"},
        lambda root: plant_mapping.plant_registry_key(root, "valley-plants"),
        ".tcip/state/plant_registries/valley-plants.json", root_of=project_state_dir),
    # one completed delivery, with the gate's finding for the bucket it shipped
    "delivery_events": Registered(
        {"event_id": EVENT_ID_UNDER_TEST, "trait": TRAIT_UNDER_TEST, "trait_revision": 1,
         "trait_revision_sha256": "7f3a1b9c2d4e5f60",
         "delivery_kind": DELIVERY_KIND_UNDER_TEST, "door": "deliver_phenology_milestones",
         "output_path": "büsch_phenology.csv", "output_sha256": "0" * 64,
         "producer": {"checkpoint_sha256": "0" * 64, "experiment_id": EXPERIMENT},
         "buckets": [{"path": "predictions/live/2026-03-04", "date": "2026-03-04",
                      "assessment_id": "assessment-ü", "validated": True, "reason": None}],
         "scale_assessment_id": None, "validated": True, "acknowledgment": None,
         "population": ["plot_ü"], "require_all_dates_complete": True,
         "plant_mapping": {
             "name": "valley", "dataset_id": "ds-1",
             "dataset_root": "dü", "built_at": "2026-03-04T12:00:00+00:00",
             "record_sha256": "0" * 64, "nn_tolerance_m": {"value": 3.0, "source": "stated"},
             "capture_identity": {"2026-03-04": "0" * 16}, "captures_unverified": [],
             "plant_csvs_unverified": [], "dates_delivered": ["2026-03-04"],
             "images_unattributed": 0, "images_unattributed_scope": "delivered_dates",
             "plant_attribution": "image"},
         "produced_at": "2026-03-04T12:00:00+00:00"},
        lambda root: delivery.delivery_event_key(root, EVENT_ID_UNDER_TEST),
        f".tcip/state/delivery_events/{EVENT_ID_UNDER_TEST}.json", root_of=project_state_dir),
    "delivery_acknowledgments": Registered(
        {"acknowledgment_id": ACKNOWLEDGMENT_UNDER_TEST, "acknowledged_by": "user:breeder",
         "reason": "a look before the büsch assessment", "result_sha256": "0" * 64,
         "recorded_at": "2026-03-04T12:00:00+00:00"},
        lambda root: delivery._acknowledgment_key(root, ACKNOWLEDGMENT_UNDER_TEST),
        f".tcip/state/delivery_acknowledgments/{ACKNOWLEDGMENT_UNDER_TEST}.json",
        root_of=project_state_dir),
}

CODEC_EXEMPT = {
    "backend_port": "the value is the port text itself",
    "retrospectives": "the value is the markdown document itself",
    "workspace_last_opened": "the value is the last-opened project's id",
}
"""Every registered record or log whose codec is not its kind's canonical one, with why."""


def is_test_scaffolding(name: str) -> bool:
    """Whether the registered store ``name`` was declared by a test module."""
    return ts.get_descriptor(name).declared_in.startswith(("tests", "test_"))


def test_every_registered_store_has_a_byte_and_path_identity_case():
    """Every platform store has a case in ``REGISTERED``, and nothing else does."""
    declared = {name for name in ts.registered_stores() if not is_test_scaffolding(name)}
    assert declared == set(REGISTERED)


def test_the_delivery_events_golden_sample_validates_against_its_declared_shape():
    """The registered delivery-event golden validates as a ``DeliveryEventRecord``."""
    DeliveryEventRecord.model_validate(REGISTERED["delivery_events"].golden)


def test_every_json_store_encodes_through_the_one_codec_its_kind_declares():
    """Every non-test JSON store's codec is the very ``RECORD_JSON`` or ``LOG_JSON`` instance its
    kind declares, or it is named in ``CODEC_EXEMPT``."""
    off_canon = {}
    for name in ts.registered_stores():
        descriptor = ts.get_descriptor(name)
        if is_test_scaffolding(name) or descriptor.kind == "blob":
            continue
        expected = ts.RECORD_JSON if descriptor.kind == "record" else ts.LOG_JSON
        if descriptor.codec is not expected and name not in CODEC_EXEMPT:
            off_canon[name] = type(descriptor.codec).__name__

    assert off_canon == {}
    assert set(CODEC_EXEMPT) <= set(ts.registered_stores())


def test_the_canonical_record_codec_writes_the_bytes_this_test_spells_out():
    """The record and log codecs' exact bytes."""
    golden = {"b": {"nested": True}, "a": "ü", "ratio": 0.5, "absent": None}

    assert ts.RECORD_JSON.encode(golden) == (
        b'{\n'
        b'  "b": {\n'
        b'    "nested": true\n'
        b'  },\n'
        b'  "a": "\xc3\xbc",\n'
        b'  "ratio": 0.5,\n'
        b'  "absent": null\n'
        b'}\n'
    )
    assert ts.LOG_JSON.encode({"epoch": 1, "loss": 0.5, "note": "ü"}) == (
        b'{"epoch": 1, "loss": 0.5, "note": "\xc3\xbc"}'
    )


def test_a_text_store_refuses_a_value_that_is_not_text_and_accepts_one_that_is(store, monkeypatch):
    """A text store refuses a non-text value and stores text."""
    key = web_client.backend_port_key(store.root)

    with pytest.raises(ts.StoreError):
        ts.replace(key, 8765)

    ts.replace(key, "8765")
    assert ts.read(key) == "8765"


def test_a_value_the_codec_cannot_spell_names_the_store_the_key_and_the_type(store):
    """The refusal names the store, the key and the value's type; the converted value stores."""
    key = store.key(LWW, "convertible")
    where = Path("a/b")

    with pytest.raises(ts.StoreError) as raised:
        ts.replace(key, {"where": where})
    message = str(raised.value)
    assert LWW in message
    assert "convertible" in message
    assert type(where).__name__ in message

    ts.replace(key, {"where": Path("a/b").as_posix()})
    assert ts.read(key) == {"where": "a/b"}


def test_a_non_finite_number_is_refused_rather_than_written_as_a_word_json_has_no_type_for(store):
    """A NaN is refused; its ``stored_number`` form stores."""
    key = store.key(LWW, "measurement")

    with pytest.raises(ts.StoreError):
        ts.replace(key, {"score": float("nan")})

    ts.replace(key, ts.stored_number("score", float("nan")))
    assert ts.read(key) == {"score": None, "score_state": "nan"}


@pytest.mark.parametrize("name", sorted(REGISTERED))
def test_a_registered_store_lands_where_its_locator_says_with_the_bytes_its_codec_produces(
    store, tmp_path, monkeypatch, name
):
    """Nothing between a store's codec and the disk adds, translates or wraps a byte."""
    only_on(store, FILE, "the subject is the bytes each store's locator puts on disk, which is "
                         "what an export writes back out rather than what a database holds")
    case = REGISTERED[name]
    descriptor = ts.get_descriptor(name)
    seam_root = tmp_path / "seam"

    key = case.key_of(seam_root)
    if descriptor.kind == "log":
        ts.append(key, case.golden)
        expected = descriptor.codec.encode(case.golden) + b"\n"
    elif descriptor.kind == "blob":
        ts.put_blob(key, case.golden, expect=ts.Version.ABSENT)
        expected = case.golden
    else:
        ts.replace(key, case.golden, expect=ts.Version.ABSENT)
        expected = descriptor.codec.encode(case.golden)

    landed = store.backend.path_for(key)
    assert landed.read_bytes() == expected
    assert landed.relative_to(seam_root).as_posix() == case.relative
    relative_to_root = landed.relative_to(case.root_of(seam_root)).as_posix()
    assert descriptor.locator.parts_from(PurePosixPath(relative_to_root)) == key.parts


def test_a_shard_path_spells_its_key_back_whatever_separators_its_names_carry(tmp_path):
    """An image name carrying a separator is one filename and a bucket carrying separators one
    directory, and the path reads back as the very key that placed it, while two names folding
    alike keep distinct keys."""
    state_dir = project_state_dir(tmp_path)
    key = verdicts.verdict_key(state_dir, "predictions/live/2026-03-04", "a/b.jpg")
    locator = ts.get_descriptor(verdicts.REVIEW_VERDICTS_STORE).locator

    placed = locator.relative_path(str(state_dir), key.parts)
    assert len(placed.parts) == 3  # review/<one bucket dir>/<one shard file>
    assert locator.parts_from(placed) == key.parts
    assert verdicts.verdict_key(state_dir, "predictions_live_2026-03-04", "a_b.jpg") != key


def test_enumerating_review_verdicts_answers_identities_that_read_back(store):
    """``keys`` answers the shard's own key, which reads the shard back, though its names carry
    separators."""
    state_dir = project_state_dir(store.root)
    key = verdicts.verdict_key(state_dir, "predictions/live/2026-03-04", "a/b.jpg")
    decided = verdicts.Verdict(proposal=0, action="accepted", by="user:ü",
                               at="2026-03-04T12:00:00+00:00")
    verdicts.record_verdicts(key, [decided])

    assert ts.keys(verdicts.REVIEW_VERDICTS_STORE, str(state_dir)) == [key]
    assert verdicts.read_verdicts(key) == [decided]


def test_a_cleared_log_enumerates_as_absent(store):
    """A log that holds entries is enumerated once, however many it holds, and one cleared of
    them is not enumerated at all."""
    state_dir = project_state_dir(store.root)
    kept = verdicts.verdict_key(state_dir, "predictions/live/2026-03-04", "a.jpg")
    cleared = verdicts.verdict_key(state_dir, "predictions/live/2026-03-04", "b.jpg")
    decided = verdicts.Verdict(proposal=0, action="rejected", by="user:ü",
                               at="2026-03-04T12:00:00+00:00")
    verdicts.record_verdicts(kept, [decided, decided])
    verdicts.record_verdicts(cleared, [decided])

    ts.clear_log(cleared)

    assert ts.keys(verdicts.REVIEW_VERDICTS_STORE, str(state_dir)) == [kept]


# ── conditional blob writes ─────────────────────────────────────────────────────


def test_a_blob_read_carries_the_token_its_own_bytes_produce(store):
    key = store.key(BLOB, "tokened")
    assert ts.read_blob_versioned(key, default=b"").version == ts.Version.ABSENT

    written = ts.put_blob(key, b"first")
    stored = ts.read_blob_versioned(key)
    assert stored.value == b"first"
    assert stored.version == written
    with pytest.raises(ts.NotFound) as absent:
        ts.read_blob_versioned(store.key(BLOB, "never-written"))
    assert "default=" in str(absent.value)


def test_a_blob_write_from_a_current_token_lands_and_a_stale_one_is_refused(store):
    key = store.key(BLOB, "compare-and-set")
    held = ts.put_blob(key, b"first", expect=ts.Version.ABSENT)
    moved = ts.put_blob(key, b"second", expect=held)

    with pytest.raises(ts.VersionConflict) as raised:
        ts.put_blob(key, b"third", expect=held)
    assert raised.value.actual == moved
    with ts.open_blob(key) as handle:
        assert handle.read() == b"second"

    assert ts.put_blob(key, b"fourth", expect=moved) != moved
    with ts.open_blob(key) as handle:
        assert handle.read() == b"fourth"


def test_a_create_only_blob_write_refuses_an_existing_blob_and_keeps_its_bytes(store):
    key = store.key(BLOB, "captured-once")
    ts.put_blob(key, b"pristine", expect=ts.Version.ABSENT)

    with pytest.raises(ts.VersionConflict):
        ts.put_blob(key, b"overwritten", expect=ts.Version.ABSENT)
    with ts.open_blob(key) as handle:
        assert handle.read() == b"pristine"

    ts.put_blob(key, b"unconditional")
    with ts.open_blob(key) as handle:
        assert handle.read() == b"unconditional"


def test_a_blob_delete_from_a_current_token_lands_and_a_stale_one_is_refused(store):
    key = store.key(BLOB, "conditional-delete")
    held = ts.put_blob(key, b"first", expect=ts.Version.ABSENT)
    moved = ts.put_blob(key, b"second", expect=held)

    with pytest.raises(ts.VersionConflict) as raised:
        ts.delete(key, expect=held)
    assert raised.value.actual == moved
    with ts.open_blob(key) as handle:
        assert handle.read() == b"second"

    ts.delete(key, expect=moved)
    assert not ts.exists(key)


def test_a_streamed_blob_write_refuses_a_stale_token_before_the_producer_writes(store):
    key = store.key(BLOB, "streamed")
    held = ts.put_blob(key, b"first", expect=ts.Version.ABSENT)
    ts.put_blob(key, b"second", expect=held)

    with pytest.raises(ts.VersionConflict):
        with ts.write_blob(key, expect=held) as handle:
            handle.write(b"never reached")
    with ts.open_blob(key) as handle:
        assert handle.read() == b"second"

    with ts.write_blob(key, expect=ts.read_blob_versioned(key).version) as handle:
        handle.write(b"third")
    with ts.open_blob(key) as handle:
        assert handle.read() == b"third"


def test_put_blob_from_path_streams_a_source_files_bytes_and_returns_their_version(store):
    key = store.key(BLOB, "from-path")
    source = store.root / "source.bin"
    source.write_bytes(b"capture bytes" * 10000)

    version = ts.put_blob_from_path(key, source, expect=ts.Version.ABSENT)

    with ts.open_blob(key) as handle:
        assert handle.read() == source.read_bytes()
    assert ts.read_blob_versioned(key).version == version
    comparison = store.key(BLOB, "put-blob-comparison")
    assert ts.put_blob(comparison, source.read_bytes()) == version


def test_put_blob_from_path_refuses_a_stale_expect_and_leaves_the_previous_bytes(store):
    key = store.key(BLOB, "from-path-conflict")
    held = ts.put_blob(key, b"first", expect=ts.Version.ABSENT)
    ts.put_blob(key, b"second", expect=held)
    source = store.root / "third-attempt.bin"
    source.write_bytes(b"third attempt")

    with pytest.raises(ts.VersionConflict):
        ts.put_blob_from_path(key, source, expect=held)
    with ts.open_blob(key) as handle:
        assert handle.read() == b"second"


def test_put_blob_from_path_refuses_a_missing_source_before_any_byte_moves(store):
    """A missing source path raises and leaves whatever the destination held untouched."""
    key = store.key(BLOB, "from-path-missing-source")
    held = ts.put_blob(key, b"prior", expect=ts.Version.ABSENT)
    source = store.root / "does-not-exist.bin"

    with pytest.raises(FileNotFoundError):
        ts.put_blob_from_path(key, source, expect=held)
    with ts.open_blob(key) as handle:
        assert handle.read() == b"prior"


def test_two_processes_writing_a_blob_from_one_token_produce_one_winner_and_one_conflict(store):
    key = store.key(BLOB, "contested")
    ts.put_blob(key, b"seed", expect=ts.Version.ABSENT)
    go = store.root / "blob.go"
    contenders = []
    for name in ("first", "second"):
        ready = store.root / f"blob-{name}.ready"
        result = store.root / f"blob-{name}.json"
        proc = store.spawn(
            "write-blob-after", store.root, "contested", "cas", name, ready, go, result
        )
        wait_for(ready)
        contenders.append((proc, result))
    go.write_text("go", encoding="utf-8")

    outcomes = []
    for proc, result in contenders:
        assert proc.wait(timeout=60) == 0
        outcomes.append(json.loads(result.read_text(encoding="utf-8"))["outcome"])
    assert sorted(outcomes) == ["VersionConflict", "written"]
    with ts.open_blob(key) as handle:
        assert handle.read() in (b"first", b"second")


def test_the_generic_key_a_dated_write_uses_lands_where_dataset_layout_computes_the_path(store):
    """The generic annotation key and ``dataset_layout.annotation_path`` name one file."""
    only_on(store, FILE, "the agreement asserted here is between the store's locator and "
                         "dataset_layout's own path arithmetic, which path_for exposes only on "
                         "the file backend")
    layout_path = dataset_layout.annotation_path(store.root, "2026-03-04", "a_1")
    generic_key = json_io.annotation_record_key(
        dataset_layout.annotation_dir(store.root, "2026-03-04"), "a_1"
    )
    assert store.backend.path_for(generic_key) == layout_path

    version = ts.put_blob(generic_key, b"{}", expect=ts.Version.ABSENT)
    assert layout_path.read_bytes() == b"{}"
    assert ts.read_blob_versioned(generic_key).version == version
