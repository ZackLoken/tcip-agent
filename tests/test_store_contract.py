"""The storage seam's contract, through the ``store`` fixture. Isolation cases run across real OS
processes, and also against a weakened backend (``TCIP_STORE_CONTRACT_IGNORES_EXPECT=1``, or
``TCIP_STORE_CONTRACT_COMMITS_EACH_WRITE=1`` for the crash case) where they fail."""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import pytest

import tcip_store as ts
from tcip_annotation import verdicts
from tcip_mcp import delivery, traits
from tcip_mcp.project_paths import project_state_dir
from tcip_store.file_backend import DATABASE_FILENAME, database_file
from tcip_store.sqlite_backend import SqliteBackend, encode_parts, open_verified
from tests._store_worker import (
    CAS,
    LOG,
    LWW,
    NESTED,
    STRICT,
    blob_path,
    make_backend,
    wait_for,
)

_WORKER = Path(__file__).with_name("_store_worker.py")


@dataclass
class Harness:
    """The bound backend and the root its keys hang off."""

    backend: SqliteBackend
    root: Path
    procs: list[subprocess.Popen] = field(default_factory=list)

    def key(self, store: str, *parts: str) -> ts.Key:
        return ts.Key(store, str(self.root), tuple(parts))

    def connect(self) -> sqlite3.Connection:
        """A connection of this test's own to the root's database, for reaching behind the seam."""
        return sqlite3.connect(str(database_file(str(self.root))), isolation_level=None)

    def damage_record(self, key: ts.Key, data: bytes) -> None:
        """Put bytes that will not decode behind a record."""
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


@pytest.fixture
def store(tmp_path):
    backend = make_backend()
    ts.bind(backend)
    harness = Harness(backend=backend, root=tmp_path)
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
    denies_an_open_root = os.name == "nt"
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


# ── the database's own creation ─────────────────────────────────────────────────


def _journal_mode(db: Path) -> str:
    """The journal mode a database reports, read without changing it."""
    conn = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)
    try:
        return str(conn.execute("pragma journal_mode").fetchone()[0]).lower()
    finally:
        conn.close()


def test_the_first_write_publishes_a_database_already_in_wal(store):
    """The installed file is converted while the transition lock still excludes every other
    opener, so it is in WAL before anything opens it for use."""
    ts.replace(store.key(LWW, "first"), {"n": 1})
    ts.release_root(store.root)

    assert _journal_mode(database_file(str(store.root))) == "wal"


def test_an_opener_arriving_beside_another_connection_still_gets_a_usable_database(store):
    """A second connection is open when an opener arrives: since publication already converted
    the journal mode, the opener reads it and proceeds rather than being refused."""
    ts.replace(store.key(LWW, "first"), {"n": 1})
    ts.release_root(store.root)
    db_path = database_file(str(store.root))
    bystander = sqlite3.connect(str(db_path), isolation_level=None)
    try:
        bystander.execute("select count(*) from records").fetchone()
        conn = open_verified(db_path, str(store.root), 30.0)
        try:
            assert str(conn.execute("pragma journal_mode").fetchone()[0]).lower() == "wal"
        finally:
            conn.close()
    finally:
        bystander.close()


def test_a_refusal_reports_the_wait_it_measured_rather_than_a_configured_timeout(store):
    """The elapsed value a contention refusal carries is one this layer measured."""
    key = store.key(LWW, "measured")
    with pytest.raises(ts.StoreBusy) as caught:
        with store.backend._mapped((key,)):  # noqa: SLF001 - the measurement is the subject
            time.sleep(0.25)
            raise sqlite3.OperationalError("database is locked")
    assert 0.15 <= caught.value.waited_s < 5.0


def test_every_connection_commits_at_full_synchronous(store):
    """A committed write survives a power loss: the connection a write commits on is at FULL."""
    ts.replace(store.key(LWW, "durable"), {"n": 1})
    with store.backend._connection(str(store.root), create=False) as conn:  # noqa: SLF001
        assert conn.execute("pragma synchronous").fetchone()[0] == 2


def test_a_read_of_a_root_with_no_database_answers_absence_and_creates_none(store):
    """Reading never creates a database: a root nothing has written answers absence."""
    key = store.key(LWW, "never-written")

    assert ts.read(key, default=None) is None
    assert ts.keys(LWW, str(store.root)) == []
    assert ts.read_log(store.key(LOG, "never-appended")).records == []
    assert not database_file(str(store.root)).exists()


# ── atomicity ───────────────────────────────────────────────────────────────────


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
    """The unconditional form of the same race: it waits for the holder, then lands."""
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


def test_a_record_delete_from_a_current_token_lands_and_a_stale_one_is_refused(store):
    key = store.key(CAS, "conditional-delete")
    held = ts.replace(key, {"n": 1}, expect=ts.Version.ABSENT)
    moved = ts.replace(key, {"n": 2}, expect=held)

    with pytest.raises(ts.VersionConflict) as raised:
        ts.delete(key, expect=held)
    assert raised.value.actual == moved
    assert ts.read(key) == {"n": 2}

    ts.delete(key, expect=moved)
    assert ts.read(key, default=None) is None


def test_a_crash_mid_transaction_leaves_the_old_records_and_wedges_no_successor(store):
    """A process killed after its transaction's first write executed, before the commit, leaves
    both records as they were, and the lock it held does not strand the next writer."""
    first = store.key(LWW, "alpha")
    second = store.key(LWW, "beta")
    with ts.transaction(first, second) as txn:
        txn.write(first, {"who": "old"})
        txn.write(second, {"who": "old"})
    marker = store.root / "staged.marker"
    proc = store.spawn("pause-before-commit", store.root, "alpha", "beta", marker, 30)
    wait_for(marker, timeout_s=60)
    proc.kill()
    proc.wait(timeout=30)

    assert ts.read(first) == ts.read(second) == {"who": "old"}

    with ts.transaction(first, second, timeout_s=10) as txn:
        txn.write(first, {"who": "successor"})
        txn.write(second, {"who": "successor"})
    assert ts.read(first) == ts.read(second) == {"who": "successor"}


# ── one transaction over records and logs ───────────────────────────────────────


def test_a_transaction_over_a_record_and_a_log_commits_both_or_neither(store):
    """A raise inside the body leaves the record and the log as they were; a clean exit lands
    the write and every append together."""
    record = store.key(LWW, "decided")
    log = store.key(LOG, "decisions")
    ts.replace(record, {"n": 0})

    with pytest.raises(RuntimeError):
        with ts.transaction(record, log) as txn:
            txn.write(record, {"n": 1})
            txn.append(log, {"i": 1})
            raise RuntimeError("the caller found a conflict")
    assert ts.read(record) == {"n": 0}
    assert ts.read_log(log).records == []

    with ts.transaction(record, log) as txn:
        txn.write(record, {"n": 1})
        txn.append(log, {"i": 1})
        txn.append(log, {"i": 2})
    assert ts.read(record) == {"n": 1}
    assert [entry["i"] for entry in ts.read_log(log).records] == [1, 2]


def test_a_verdict_that_will_not_encode_lands_none_of_the_save_s_verdicts(store):
    """``record_verdicts`` appends a save's verdicts in one commit: one that will not encode,
    after another already appended, leaves the shard holding neither."""
    from datetime import datetime, timezone

    key = verdicts.verdict_key(project_state_dir(store.root), "predictions/live", "a_1.jpg")
    good = verdicts.Verdict(proposal=0, action="accepted", by="user:ü",
                            at="2026-03-04T12:00:00+00:00")
    unencodable = verdicts.Verdict(proposal=1, action="rejected", by="user:ü",
                                   at=datetime(2026, 3, 4, tzinfo=timezone.utc))  # type: ignore[arg-type]

    with pytest.raises(ts.StoreError):
        verdicts.record_verdicts(key, [good, unencodable])
    assert verdicts.read_verdicts(key) == []

    verdicts.record_verdicts(key, [good])
    assert verdicts.read_verdicts(key) == [good]


def test_a_module_level_write_inside_a_transaction_is_refused_and_lands_outside_it(store):
    """A write that is not the transaction's own would commit apart from it, so it refuses and
    names the transaction's own operations; the same append outside the transaction lands."""
    record = store.key(LWW, "under-transaction")
    log = store.key(LOG, "appended-to")

    with ts.transaction(record) as txn:
        with pytest.raises(ts.TransactionMisuse) as raised:
            ts.append(log, {"i": "inside"})
        assert "transaction's own" in str(raised.value)
        with pytest.raises(ts.TransactionMisuse):
            ts.clear_log(log)
        txn.write(record, {"n": 1})

    ts.append(log, {"i": "outside"})
    assert [entry["i"] for entry in ts.read_log(log).records] == ["outside"]
    assert ts.read(record) == {"n": 1}


# ── append durability ───────────────────────────────────────────────────────────


def test_concurrent_appenders_lose_no_entry_and_interleave_none(store):
    key = store.key(LOG, "audit")
    workers = [store.spawn("append", store.root, "audit", f"proc{n}", 25) for n in range(3)]
    for worker in workers:
        assert worker.wait(timeout=120) == 0

    page = ts.read_log(key)
    assert len(page.records) == 75
    assert page.corrupt == ()
    for tag in ("proc0", "proc1", "proc2"):
        assert sorted(r["i"] for r in page.records if r["tag"] == tag) == list(range(25))


def test_an_append_is_readable_by_another_process_once_it_returns(store):
    key = store.key(LOG, "handoff")
    for i in range(3):
        ts.append(key, {"i": i})
    result = store.root / "count.json"

    reader = store.spawn("count-log", store.root, "handoff", result)
    assert reader.wait(timeout=60) == 0
    assert json.loads(result.read_text(encoding="utf-8")) == {"records": 3}


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


def test_a_cursor_returns_every_entry_once_while_appenders_are_still_writing(store):
    """Reading a log by cursor while two processes append returns every entry exactly once."""
    key = store.key(LOG, "streamed-live")
    workers = [store.spawn("append", store.root, "streamed-live", f"proc{n}", 20) for n in range(2)]

    seen: list[tuple[str, int]] = []
    cursor = None
    deadline = time.monotonic() + 120
    while len(seen) < 40:
        assert time.monotonic() < deadline, f"only {len(seen)} of 40 entries arrived"
        page = ts.read_log(key, after=cursor)
        seen.extend((r["tag"], r["i"]) for r in page.records)
        assert page.corrupt == ()
        cursor = page.cursor
    for worker in workers:
        assert worker.wait(timeout=120) == 0

    assert ts.read_log(key, after=cursor).records == []
    assert len(set(seen)) == 40
    for tag in ("proc0", "proc1"):
        assert sorted(i for name, i in seen if name == tag) == list(range(20))


def test_a_damaged_entry_is_reported_and_the_rest_still_read(store):
    """``corrupt`` names a damaged entry's position; the others decode and the cursor moves on."""
    key = store.key(LOG, "damaged")
    for i in range(3):
        ts.append(key, {"i": i})

    store.damage_log_entry(key, 1, b'{"i": bro')
    page = ts.read_log(key)
    assert [r["i"] for r in page.records] == [0, 2]
    assert page.corrupt == (1,)

    ts.append(key, {"i": 3})
    resumed = ts.read_log(key, after=page.cursor)
    assert [r["i"] for r in resumed.records] == [3]
    assert resumed.corrupt == ()


# ── clearing a log ──────────────────────────────────────────────────────────────


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


def test_a_cursor_taken_before_a_clear_reads_only_what_was_appended_after(store):
    """Replaying a cursor held across a clear returns exactly the entries appended since it."""
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


# ── refusals, each with the call it must still admit ────────────────────────────


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


def test_a_key_with_no_root_or_an_empty_store_or_part_refuses_and_a_whole_one_is_admitted(store):
    with pytest.raises(ts.BadKey):
        ts.Key(LWW, "", ("x",))
    with pytest.raises(ts.BadKey):
        ts.Key("", str(store.root), ("a",))
    with pytest.raises(ts.BadKey):
        ts.Key(NESTED, str(store.root), ("group", ""))

    ts.replace(store.key(NESTED, "group", "name"), {"n": 1})
    assert ts.read(store.key(NESTED, "group", "name")) == {"n": 1}


def test_absence_and_corruption_are_different_answers(store):
    """A record that will not decode raises ``DecodeError`` whatever ``default`` says; an absent
    one answers the default."""
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
    """The store refuses first and canonicalizes second, or the refusal turns into a guess about
    which directory the process happened to be started in; enumeration refuses alongside."""
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


def test_a_key_part_carrying_non_ascii_or_a_separator_is_stored_under_one_spelling(store):
    """A key's parts are stored as ASCII-escaped compact JSON."""
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


def test_keys_answer_every_record_and_log_of_a_store_under_a_prefix(store):
    assert ts.keys(NESTED, str(store.root)) == []
    ts.replace(store.key(NESTED, "g1", "a"), {"n": 1})
    ts.replace(store.key(NESTED, "g1", "b"), {"n": 2})
    ts.replace(store.key(NESTED, "g2", "c"), {"n": 3})
    ts.append(store.key(LOG, "g1"), {"n": 4})

    assert ts.keys(NESTED, str(store.root), ("g1",)) == [
        store.key(NESTED, "g1", "a"),
        store.key(NESTED, "g1", "b"),
    ]
    assert ts.keys(LOG, str(store.root)) == [store.key(LOG, "g1")]
    assert ts.stores(str(store.root)) == sorted([NESTED, LOG])


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
        assert "transaction's own" in str(outside_write.value)
        with pytest.raises(ts.TransactionMisuse):
            txn.read(unheld)
        txn.write(first, {"n": 1})
        txn.write(second, {"n": 2})

    assert ts.read(first) == {"n": 1}
    assert ts.read(second) == {"n": 2}


def test_a_transaction_refuses_two_roots_and_admits_two_spellings_of_one(store):
    """One transaction is one root's database, so keys from two roots refuse before anything is
    held; two spellings of one directory are one root."""
    elsewhere = store.root / "elsewhere"
    elsewhere.mkdir()
    here = store.key(LWW, "here")
    there = ts.Key(LWW, str(elsewhere), ("there",))

    with pytest.raises(ts.TransactionMisuse) as raised:
        with ts.transaction(here, there):
            pass
    assert ts.canonical_path(store.root) in str(raised.value)
    assert ts.canonical_path(elsewhere) in str(raised.value)
    assert ts.read(here, default=None) is None

    detour = store.root / "detour"
    detour.mkdir()
    spelled_around = ts.Key(LWW, str(detour / ".."), ("beta",))
    with ts.transaction(store.key(LWW, "alpha"), spelled_around) as txn:
        txn.write(store.key(LWW, "alpha"), {"n": 1})
        txn.write(spelled_around, {"n": 2})

    assert ts.read(store.key(LWW, "alpha")) == {"n": 1}
    assert ts.read(store.key(LWW, "beta")) == {"n": 2}


def test_two_spellings_of_one_root_address_one_database(store):
    """Two spellings of one root share one connection slot and one database."""
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
    """Writing a ``traits`` record beside a ``delivery_events`` record creates exactly one
    database under the project's state root, never a second, nested one."""
    ts.replace(traits.trait_key(store.root, "bud_opening"), {"revisions": []},
               expect=ts.Version.ABSENT)
    ts.replace(delivery.delivery_event_key(store.root, "e1"), {"event_id": "e1"})

    databases = sorted(store.root.rglob(DATABASE_FILENAME))
    assert databases == [store.root / ".tcip" / "state" / ".tcip" / DATABASE_FILENAME]


def test_a_backend_refuses_to_exist_without_cross_process_locking(store, monkeypatch):
    monkeypatch.setitem(sys.modules, "filelock", None)
    with pytest.raises(ts.BackendUnavailable) as raised:
        SqliteBackend()
    assert "filelock" in str(raised.value)

    monkeypatch.undo()
    SqliteBackend().close()


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


def test_a_transaction_reads_its_own_write_and_shows_it_to_nobody_else(store):
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


# ── the two spellings ───────────────────────────────────────────────────────────


def test_the_record_and_log_spellings_write_the_bytes_this_test_spells_out():
    golden = {"b": {"nested": True}, "a": "ü", "ratio": 0.5, "absent": None}

    assert ts.encode_record(golden) == (
        b'{\n'
        b'  "b": {\n'
        b'    "nested": true\n'
        b'  },\n'
        b'  "a": "\xc3\xbc",\n'
        b'  "ratio": 0.5,\n'
        b'  "absent": null\n'
        b'}\n'
    )
    assert ts.encode_log_line({"epoch": 1, "loss": 0.5, "note": "ü"}) == (
        b'{"epoch": 1, "loss": 0.5, "note": "\xc3\xbc"}'
    )
    assert ts.decode_value(ts.encode_record(golden)) == golden
    assert ts.encode_record("a text value") == b'"a text value"\n'


def test_a_value_the_spelling_cannot_carry_names_the_store_the_key_and_the_type(store):
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


def test_a_value_with_a_non_string_key_is_refused_rather_than_merged_with_its_spelling(store):
    """JSON names an object's keys with strings, so ``1`` and ``"1"`` would land as one key and
    one of the two values would vanish; both writers refuse, and the string-keyed value stores."""
    record = store.key(LWW, "keyed")
    log = store.key(LOG, "keyed")
    mixed: dict = {1: "number", "1": "string"}

    with pytest.raises(ts.StoreError):
        ts.replace(record, mixed)
    with pytest.raises(ts.StoreError):
        ts.append(log, mixed)
    assert ts.read(record, default=None) is None
    assert ts.read_log(log).records == []

    ts.replace(record, {"1": "string"})
    assert ts.read(record) == {"1": "string"}


# ── one transaction at a time, over records and logs or over files ─────────────


def test_a_file_write_inside_a_record_transaction_refuses_and_lands_outside_it(store):
    """A file write or file transaction inside a record transaction would commit apart from it,
    so each refuses and the record transaction rolls back whole; the same write outside lands."""
    record = store.key(LWW, "beside-a-file")
    path = blob_path(store.root, "beside-a-record")
    ts.replace(record, {"n": 0})

    with pytest.raises(ts.TransactionMisuse):
        with ts.transaction(record) as txn:
            txn.write(record, {"n": 1})
            ts.put_blob(path, b"new")
    with pytest.raises(ts.TransactionMisuse):
        with ts.transaction(record):
            with ts.blob_transaction(path):
                pass
    assert ts.read(record) == {"n": 0}
    assert not path.exists()

    ts.put_blob(path, b"new")
    assert path.read_bytes() == b"new"


def test_a_record_write_inside_a_file_transaction_refuses_and_lands_outside_it(store):
    """A record write or record transaction inside a file transaction refuses, and the file
    transaction applies nothing; the same write outside lands."""
    record = store.key(LWW, "beside-a-file-transaction")
    path = blob_path(store.root, "under-a-file-transaction")

    with pytest.raises(ts.TransactionMisuse):
        with ts.blob_transaction(path) as txn:
            txn.write(path, b"new")
            ts.replace(record, {"n": 1})
    with pytest.raises(ts.TransactionMisuse):
        with ts.blob_transaction(path):
            with ts.transaction(record):
                pass
    assert not path.exists()
    assert ts.read(record, default=None) is None

    ts.replace(record, {"n": 1})
    assert ts.read(record) == {"n": 1}


def test_a_file_that_is_not_a_database_at_the_database_s_path_refuses_naming_it(store):
    """Something else holding the database's name refuses by name before a row is read."""
    path = database_file(str(store.root))
    path.parent.mkdir(parents=True)
    path.write_bytes(b"not a database, only bytes under its name" * 100)

    with pytest.raises(ts.StoreError) as raised:
        ts.read(store.key(LWW, "anything"), default=None)
    assert "not a SQLite database" in str(raised.value)
    assert str(path) in str(raised.value)


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


# ── files written by path ───────────────────────────────────────────────────────


def test_a_blob_read_carries_the_token_its_own_bytes_produce(store):
    path = blob_path(store.root, "tokened")
    assert ts.read_blob_versioned(path, default=b"").version == ts.Version.ABSENT

    written = ts.put_blob(path, b"first")
    stored = ts.read_blob_versioned(path)
    assert stored.value == b"first"
    assert stored.version == written
    with pytest.raises(ts.NotFound) as absent:
        ts.read_blob_versioned(blob_path(store.root, "never-written"))
    assert "default=" in str(absent.value)


def test_a_blob_write_from_a_current_token_lands_and_a_stale_one_is_refused(store):
    path = blob_path(store.root, "compare-and-set")
    held = ts.put_blob(path, b"first", expect=ts.Version.ABSENT)
    moved = ts.put_blob(path, b"second", expect=held)

    with pytest.raises(ts.VersionConflict) as raised:
        ts.put_blob(path, b"third", expect=held)
    assert raised.value.actual == moved
    assert path.read_bytes() == b"second"

    assert ts.put_blob(path, b"fourth", expect=moved) != moved
    assert path.read_bytes() == b"fourth"


def test_a_create_only_blob_write_refuses_an_existing_blob_and_keeps_its_bytes(store):
    path = blob_path(store.root, "captured-once")
    ts.put_blob(path, b"pristine", expect=ts.Version.ABSENT)

    with pytest.raises(ts.VersionConflict):
        ts.put_blob(path, b"overwritten", expect=ts.Version.ABSENT)
    assert path.read_bytes() == b"pristine"

    ts.put_blob(path, b"unconditional")
    assert path.read_bytes() == b"unconditional"


def test_a_blob_delete_from_a_current_token_lands_and_a_stale_one_is_refused(store):
    path = blob_path(store.root, "conditional-delete")
    held = ts.put_blob(path, b"first", expect=ts.Version.ABSENT)
    moved = ts.put_blob(path, b"second", expect=held)

    with pytest.raises(ts.VersionConflict) as raised:
        ts.delete_blob(path, expect=held)
    assert raised.value.actual == moved
    assert path.read_bytes() == b"second"

    ts.delete_blob(path, expect=moved)
    assert not path.exists()


def test_a_blob_transaction_writes_every_file_or_none(store):
    """A blob transaction reads each file as bytes, stages each write, and lands every one or,
    on a raise inside it, none."""
    first, second = blob_path(store.root, "first"), blob_path(store.root, "second")
    ts.put_blob(second, b"kept")

    with pytest.raises(RuntimeError):
        with ts.blob_transaction(first, second) as txn:
            txn.write(first, b"staged")
            raise RuntimeError("the caller found a conflict")
    assert not first.exists()
    assert second.read_bytes() == b"kept"

    with ts.blob_transaction(first, second) as txn:
        assert txn.read(first, default=None) is None
        assert txn.read(second) == b"kept"
        with pytest.raises(ts.TransactionMisuse):
            txn.read(blob_path(store.root, "unheld"))
        txn.write(first, b"one")
        txn.write(second, b"two")
    assert first.read_bytes() == b"one"
    assert second.read_bytes() == b"two"


def test_two_processes_writing_a_blob_from_one_token_produce_one_winner_and_one_conflict(store):
    path = blob_path(store.root, "contested")
    ts.put_blob(path, b"seed", expect=ts.Version.ABSENT)
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
    assert path.read_bytes() in (b"first", b"second")
