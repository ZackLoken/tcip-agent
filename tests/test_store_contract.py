"""The storage seam's contract: what every backend must do, whatever it stores bytes in.

Most of these run against real OS processes rather than threads, because the guarantee under
test is exactly the one threads cannot check: two interpreters sharing a lock registry would
pass a same-process test against a backend with no cross-process exclusion at all.

The isolation cases are the ones that pin the defect this layer exists to remove, so they
are also run against a weakened backend and observed failing there: on the file backend one
whose writes skip the lock (``TCIP_STORE_CONTRACT_UNLOCKED=1``), on the database backend one
that never compares ``expect`` (``TCIP_STORE_CONTRACT_IGNORES_EXPECT=1``).

The ``store`` fixture runs every case it serves against both backends. A handful of cases are
about one backend's own mechanics rather than the contract, and each says at its top which
backend it is about and why; everything else must pass unchanged on both.
"""

from __future__ import annotations

import json
import os
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
from tcip_annotation import format_io, json_io, review_engine
from tcip_mcp import (
    audit,
    dataset_layout,
    experiments,
    model_registry,
    operationalization,
    project_record,
    project_status,
    traits,
    web_client,
    workspace,
)
from tcip_mcp.pipelines import image_utils, model_build, resolution
from tcip_mcp.pipelines.delivery_events_schema import DeliveryEventRecord
from tcip_mcp.pipelines.data import band_groups, splits
from tcip_mcp.pipelines.feedback import materialize
from tcip_mcp.pipelines.postprocessing import plant_mapping
from tcip_mcp.pipelines.training import eval_runners, generic_trainer, hpo
from tcip_mcp.tools import (
    data_tools,
    inference_tools,
    meta_tools,
    project_tools,
    training_tools,
    proposal_tools,
)
from tcip_store.file_backend import (
    DATABASE_FILENAME,
    FileBackend,
    RootedFileLocator,
    creation_temp_name,
)
from tcip_store.sqlite_backend import SqliteBackend, database_path, encode_parts
from tcip_web import agent_learning_capture
from tcip_web import jobstore
from tcip_web import state as web_state
from tcip_web.routes import canvas, sessions
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
        """Put bytes that will not decode behind a record, wherever this backend keeps them.

        The corruption has to reach the same place the backend reads from, or the case would
        report absence on one backend and corruption on the other for the same setup.
        """
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
        txn.write(record, {"n": 1})

    ts.append(log, {"i": "outside"})
    assert [entry["i"] for entry in ts.read_log(log).records] == ["outside"]
    assert ts.read(record) == {"n": 1}


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
    """A capability is what a caller refuses on rather than degrades against, so a false
    declaration cannot be allowed to ship quietly."""
    only_on(store, SQLITE, _DATABASE_MECHANICS)
    assert ts.capabilities() == ts.Capabilities(
        multi_key_atomic_commit=True,
        cross_machine_exclusion=False,
        durable_replace=os.name != "nt",
        durable_append=True,
        local_blob_paths=True,
    )


def test_a_database_and_its_sidecars_are_never_keys_of_the_store_they_sit_inside(store):
    """The database backend keeps its files in ``.tcip/``, which is a directory a store's own
    entries live in, so enumeration has to tell one from the other by name.

    The build temp is asked for by the same helper the creation path names it with, so this
    checks the name that is actually produced rather than a pattern restated here.
    """
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
    """Root strings arrive from callers that do not canonicalize them the same way, and a
    second database for the same directory would silently split one store in two.

    The connection slot is what this asserts on. Reading the same value back through both
    spellings does not pin the backend: the filesystem resolves the detour on its own, so that
    much passes even with canonicalization reduced to identity. One slot for two spellings is
    only true if the backend canonicalized them itself.
    """
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


def test_trait_specs_shares_the_state_database_rather_than_gaining_its_own(store):
    """The positive proof the trait-spec re-root exists to establish: writing a ``trait_specs``
    record alongside a sibling ``STATE``-rooted store (``trait_operationalizations``) creates
    exactly one database under the project's shared state root, never a second, nested one under
    ``trait_specs`` itself the way the store's self-rooted predecessor did.
    """
    only_on(store, SQLITE, _DATABASE_MECHANICS)
    spec_key = traits.trait_spec_key(traits.trait_specs_dir(store.root), "catkin")
    op_key = operationalization.operationalization_key(
        operationalization.operationalizations_scope(store.root), "catkin", "phenology")

    ts.replace(spec_key, {"name": "catkin", "delivers": ["catkin_05per_date"]},
               expect=ts.Version.ABSENT)
    ts.replace(op_key, {"trait": "catkin", "delivery_kind": "phenology"}, expect=ts.Version.ABSENT)

    databases = sorted(store.root.rglob(DATABASE_FILENAME))
    assert databases == [store.root / ".tcip" / "state" / ".tcip" / DATABASE_FILENAME]
    assert not (store.root / ".tcip" / "state" / "trait_specs" / ".tcip").exists()


def test_a_key_part_carrying_non_ascii_or_a_separator_is_stored_under_one_spelling(store):
    """The parts column's spelling, pinned here rather than left to whatever json.dumps
    defaults to on the day: two spellings of one key would address two rows."""
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
    """A metrics tail reads a log other processes are appending to, so the cursor has to be
    exact under concurrency: a gap loses an epoch and a repeat double-counts one."""
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
    """The three live-state stores declare relaxed durability, and that declaration has to keep
    meaning something once a commit rather than an fsync is what makes a write durable."""
    only_on(store, SQLITE, _DATABASE_MECHANICS)
    backend = _RecordingSyncBackend()
    ts.bind(backend)
    try:
        relaxed = store.key(RELAXED, "heartbeat")
        durable = store.key(LWW, "durable")
        ts.replace(durable, {"n": 0})
        conn = backend._connection(str(store.root), (LWW,))

        backend.levels.clear()
        ts.replace(relaxed, {"n": 1})
        assert backend.levels == ["NORMAL"]
        assert conn.execute("pragma synchronous").fetchone()[0] == 1

        backend.levels.clear()
        ts.replace(durable, {"n": 1})
        assert backend.levels == ["FULL"]
        assert conn.execute("pragma synchronous").fetchone()[0] == 2
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
EXPERIMENT = "exp_042"
STUDY = "hpo_1a2b3c4d"
TRIAL_DIR = "trial_00000"
LOCK_IDENTITY = "d41d8cd98f00b204"
SWEEP_IDENTITY = "7f3a1b9c2d4e5f60"
SNAPSHOT_CONTENT = "ab12cd34"
SNAPSHOT_FILENAME = "my_model.py"
IMAGE_DATE = "2026-03-04"
IMAGE_STEM = "a_1"
IMAGE_EXT = ".JPG"

CHECKPOINT_BYTES = b"PK\x03\x04not a real archive, only bytes handed to the store\x00\xff"

CLASS_REGISTRY_BYTES = (
    '{\n'
    '  "catkin": {\n'
    '    "description": "a männlich flower",\n'
    '    "attributes": {}\n'
    '  }\n'
    '}\n'
).encode("utf-8")
"""The documents a human or a tool reads as files hold bytes at the seam and are encoded by the
module that owns them, so their spelling is written out here rather than reached for through a
codec the descriptor no longer carries. What each writer produces is pinned separately, against
these same spellings, by ``test_document_store_bytes``."""

DATASET_IDENTITY_BYTES = (
    '{\n'
    '  "dataset_id": "a1",\n'
    '  "crop": "hazelnut",\n'
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
SNAPSHOT_BYTES = "def build():\n    return 'ü'\n".encode("utf-8")
IMAGE_BYTES = b"\xff\xd8\xff\xe0not a real frame, only bytes handed to the store\x00"
LABEL_BYTES = '{"annotations": [{"subject": "catkin", "bbox": [1.0, 2.0, 3.5, 4.5]}]}'.encode("utf-8")
COCO_BYTES = '{"images": [{"file_name": "ü_2.jpg"}], "annotations": []}'.encode("utf-8")


def _review_state_dir(root: Path) -> Path:
    return root / ".tcip" / "state"


def _generic_label_dir(root: Path) -> Path:
    """A label tree no layout resolver describes, the shape a materialized split writes."""
    return root / "labels"


def _band_group_dir(root: Path) -> Path:
    return root / "images"


def _split_dir(root: Path) -> Path:
    """A split output directory, the shape a caller asks a partition to be written into."""
    return root / "splits"


def _curated_dir(root: Path) -> Path:
    """A curated dataset's output directory, wherever the caller asked it to be materialized."""
    return root / "curated"


def _coco_dir(root: Path) -> Path:
    """A directory holding an assembled COCO view, which no layout resolver describes."""
    return root / "export"


def _plant_mapping_dir(root: Path) -> Path:
    return root / ".tcip" / "state"


def _stamp_bucket(root: Path) -> Path:
    """The prediction bucket a run's provenance stamps sit in, from the layout's own resolver."""
    return dataset_layout.prediction_dir(root, "live", "2026-03-04")


def _trait_spec_key(root: Path) -> ts.Key:
    return traits.trait_spec_key(traits.trait_specs_dir(root), TRAIT_UNDER_TEST)


def _trait_specs_root(root: Path) -> Path:
    """The shared ``.tcip/state`` root the store's key actually hangs off, mirroring
    ``_trait_spec_statements_root``'s own shape: the state directory, not the specs directory
    (``trait_specs_dir``) the locator's fixed prefix places records under."""
    return traits.trait_specs_dir(root).parent


def _trait_spec_statement_key(root: Path) -> ts.Key:
    return traits.trait_spec_statement_key(traits.trait_spec_statements_scope(root), TRAIT_UNDER_TEST)


def _trait_spec_statements_root(root: Path) -> Path:
    return traits.trait_spec_statements_scope(root)


def _operationalization_key(root: Path) -> ts.Key:
    return operationalization.operationalization_key(
        operationalization.operationalizations_scope(root),
        TRAIT_UNDER_TEST,
        DELIVERY_KIND_UNDER_TEST,
    )


def _operationalizations_root(root: Path) -> Path:
    return operationalization.operationalizations_scope(root)


def _pin_platform_root(root: Path, monkeypatch) -> None:
    monkeypatch.setenv("TCIP_STATE_ROOT", str(root))


def _pin_workspace(root: Path, monkeypatch) -> None:
    monkeypatch.setenv("TCIP_WORKSPACE", str(root))


@dataclass(frozen=True)
class Registered:
    """One registered store: a value of the shape it holds, and how the seam addresses it.

    ``golden`` is written through the seam and the file's bytes are compared against the
    store's own codec applied to it. That is not a restatement of the codec: it is the proof
    that nothing between the codec and the disk adds a byte-order mark, translates a newline
    or wraps an envelope. The canonical spelling itself is pinned once, centrally, by
    ``test_the_canonical_record_codec_writes_the_bytes_this_test_spells_out``.

    This suite no longer reproduces each store's pre-seam writer. That equivalence was the
    instrument for migrating writers behind the seam, and there are no pre-seam writers left
    to compare against; each owner module's own tests are where its written content is
    asserted now. What remains here is placement and encoding, for every store at once.

    ``relative`` is the path under the root the store is given. ``root_of`` is the root the
    store's own keys hang off, which for the review shards and the experiment members is a
    directory below that. ``pin`` sets the environment a store's key constructor resolves
    against, for the stores addressed relative to the platform root or the workspace.
    """

    golden: object
    key_of: Callable[[Path], ts.Key]
    relative: str
    pin: Callable[[Path, object], None] | None = None
    root_of: Callable[[Path], Path] = lambda root: root


def _construct_via_scratch_backend(build: Callable[[], dict]) -> dict:
    """Call ``build`` with a throwaway file backend bound only for the call: these fixtures are
    built at import time, before any test's own ``store`` fixture has bound one."""
    from tcip_store.file_backend import FileBackend

    ts.bind(FileBackend())
    try:
        return build()
    finally:
        ts.unbind()


def _real_split_manifest() -> dict:
    """The shape ``data_tools.compose_split_manifest`` writes today, called for real into a
    throwaway directory so this golden cannot drift from the writer silently again."""
    return _construct_via_scratch_backend(lambda: data_tools.compose_split_manifest(
        Path(tempfile.mkdtemp()), seed=42, group_by="stem_prefix", dataset_fingerprint="7ac1",
        subject="bud", attribute=None, id_map={"bud": 0},
        members={"2026-03-04": {"labels_root": "ü/annotations", "images_root": "ü/images",
                                 "dataset_hash": "9f2c",
                                 "label_digests": {"a_1": "7f3a1b9c2d4e5f60"}}},
        splits={"train": ["a_1"], "val": [], "calibration": []},
        admission_counts={"a_1": 1}, calibration_foreground_groups_by_date={},
        realized_ratios={"train": 1.0, "val": 0.0, "calibration": 0.0},
    ))


def _real_cal_holdout_lock() -> dict:
    """The shape ``splits.resolve_locked_cal_holdout_split`` writes today, drawn for real over a
    throwaway directory rather than hand-typed."""
    return _construct_via_scratch_backend(lambda: splits.resolve_locked_cal_holdout_split(
        ["a_1", "b_2", "c_3", "d_4"], identity_hash=LOCK_IDENTITY,
        scope_root=Path(tempfile.mkdtemp())))


def _real_selection_disjointness() -> dict:
    """The full shape ``resolution.resolver_selection_disjointness`` returns for a foreign
    checkpoint with no split manifest named, read back through the same
    ``operating_point._selection_disjointness`` path a live calibration takes, so this fixture
    cannot silently drift to a partial shape the resolver never produces again."""
    from tcip_mcp.pipelines.operating_point import _selection_disjointness

    raw = _selection_disjointness(None, set(), set())
    return resolution.resolver_selection_disjointness(
        {"gate_evidence": {"selection_disjointness": raw}}, "operating_point")


def _real_job_registry_summary() -> list[dict]:
    """The shape ``routes.inference._summary`` writes for one persisted job, called for real."""
    from tcip_web.routes.inference import InferenceJob, _summary

    job = InferenceJob(
        job_id="j1", checkpoint_path="model_best.pt", images_dir="images/2026-03-04",
        output_dir="predictions/live/2026-03-04", conf=0.5, iou=0.5,
        slice_hw=(512, 512), overlap=0.2, status="completed", platform_root="C:/orchards/valley",
    )
    return [_summary(job)]


REGISTERED = {
    "image_status": Registered(
        {"catkin/2026-03-04": {"a_1.jpg": "negative", "ü_2.jpg": "complete"}},
        dataset_layout.image_status_key, ".tcip/state/image_status.json"),
    "image_status_digest": Registered(
        {"catkin/2026-03-04": {"a_1.jpg": "9f2c"}},
        dataset_layout.image_status_digest_key, ".tcip/state/image_status_digest.json"),
    "view_coverage": Registered(
        {"catkin/2026-03-04": {"a_1.jpg": {"grid": {"rows": 3, "cols": 3},
                                           "cells_served_at_native": ["r1c1"],
                                           "cells_seen_at_scale": {"r1c1": 1.0}}}},
        dataset_layout.view_coverage_key, ".tcip/state/view_coverage.json"),
    "coverage_grid_zoom": Registered(
        {"catkin": {"zoom": 1.5, "set_by": "user:ü", "set_at": "2026-03-04T00:00:00+00:00"}},
        dataset_layout.coverage_grid_zoom_key, ".tcip/state/coverage_grid_zoom.json"),
    "region_completeness": Registered(
        {"catkin/orthö": {"grid": {"rows": 2, "cols": 2}, "cells_complete": ["r1c1"],
                          "stem": "orthö"}},
        dataset_layout.region_completeness_key, ".tcip/state/region_completeness.json"),
    "region_completeness_digest": Registered(
        {"catkin/orthö": {"r1c1": "3ab9"}},
        dataset_layout.region_completeness_digest_key,
        ".tcip/state/region_completeness_digest.json"),
    "class_registry": Registered(
        CLASS_REGISTRY_BYTES, dataset_layout.class_registry_key, "classes.json"),
    "dataset_identity": Registered(
        DATASET_IDENTITY_BYTES,
        dataset_layout.dataset_identity_key, "dataset.json", pin=_pin_platform_root),
    "review_verdicts": Registered(
        {"image": "a_1.jpg", "reviewed_by": "ü", "verdict": "accepted"},
        lambda root: review_engine.review_verdict_key(
            _review_state_dir(root), "predictions", "a_1.jpg"),
        ".tcip/state/review/predictions/a_1.jpg.json", root_of=_review_state_dir),
    "canvas_meta": Registered(
        {"tab": "annotate", "image": "ü.jpg"},
        lambda root: canvas.canvas_meta_key(str(root)), ".tcip/state/canvas_live.json"),
    "canvas_geometry": Registered(
        {"image_path": "ü.jpg", "tab": "annotate", "shapes": []},
        lambda root: canvas.canvas_geometry_key(str(root)), ".tcip/state/canvas_shapes.json"),
    "model_registry": Registered(
        [{"name": "detector_v1", "sha256": "0" * 64, "metrics": {"val_map50": 0.61}}],
        model_registry.registry_index_key, ".tcip/models/registry.json"),
    "job_registry": Registered(
        _real_job_registry_summary(),
        lambda root: jobstore.job_registry_key("inference_jobs"),
        ".tcip/state/inference_jobs.json", pin=_pin_platform_root),
    "workspace_active_project": Registered(
        "hazelnut_catkin_valley\n", lambda root: workspace.active_project_key(), ".active",
        pin=_pin_workspace),
    "gui_snapshot": Registered(
        {"active_tab": "annotate", "dataset": {"subject": "büsch"}},
        lambda root: web_state.gui_snapshot_key(root), ".tcip/state/gui.json"),
    "project_status": Registered(
        {"last_activity": "2026-03-04T12:00:00+00:00", "reports_since_last_retrospective": 2},
        lambda root: project_status.project_status_key(root), ".tcip/state/project_status.json"),
    "project_record": Registered(
        {"site": "north orchard"}, project_record.project_record_key, ".tcip/project.json"),
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
        "8765", lambda root: web_client.backend_port_key(), ".tcip/state/web_port.txt",
        pin=_pin_workspace),
    "canvas_open_binding": Registered(
        {"generation": 1, "root": "C:/orchards/catkin", "project_name": "hazelnut_catkin_valley",
         "issued_at": "2026-03-04T12:00:00+00:00"},
        lambda root: web_client.canvas_open_binding_key(),
        ".tcip/state/canvas_open_binding.json", pin=_pin_workspace),
    "experiment_config": Registered(
        {"model_source": {"builder": "my_module:build"}, "training": {"epochs": 3}},
        lambda root: experiments.config_key(EXPERIMENT),
        f".tcip/experiments/{EXPERIMENT}/config.json", pin=_pin_platform_root,
        root_of=lambda root: Path(experiments.experiments_scope())),
    "experiment_status": Registered(
        {"state": "created", "created": "2026-03-04T12:00:00+00:00"},
        lambda root: experiments.status_key(EXPERIMENT),
        f".tcip/experiments/{EXPERIMENT}/status.json", pin=_pin_platform_root,
        root_of=lambda root: Path(experiments.experiments_scope())),
    "experiment_lineage": Registered(
        {"data_source": "ds", "model_weights": "model_best.pt"},
        lambda root: experiments.lineage_key(EXPERIMENT),
        f".tcip/experiments/{EXPERIMENT}/lineage.json", pin=_pin_platform_root,
        root_of=lambda root: Path(experiments.experiments_scope())),
    "experiment_artifacts": Registered(
        {"weights": {"path": "model_best.pt", "recorded": "2026-03-04T12:00:00+00:00"}},
        lambda root: experiments.artifacts_key(EXPERIMENT),
        f".tcip/experiments/{EXPERIMENT}/artifacts.json", pin=_pin_platform_root,
        root_of=lambda root: Path(experiments.experiments_scope())),
    "experiment_env": Registered(
        {"env": {"python": "3.12"}, "seed": 42}, lambda root: experiments.env_key(EXPERIMENT),
        f".tcip/experiments/{EXPERIMENT}/env.json", pin=_pin_platform_root,
        root_of=lambda root: Path(experiments.experiments_scope())),
    "experiment_split": Registered(
        # every key split_construction.persist_split_manifest always writes, not only the three
        # a drawn run happens to vary
        {"train": ["img_001"], "val": ["img_002"], "seed": 42, "dataset_hash": "9f2c1b0a4d6e8f31",
         "dataset_id": "a1", "dataset_fingerprint": "7ac1", "group_by": "stem_prefix",
         "date": "2026-03-04"},
        lambda root: experiments.split_key(EXPERIMENT),
        f".tcip/experiments/{EXPERIMENT}/split.json", pin=_pin_platform_root,
        root_of=lambda root: Path(experiments.experiments_scope())),
    "experiment_metrics": Registered(
        {"epoch": 1, "timestamp": "2026-03-04T12:00:00+00:00", "loss": 0.5},
        lambda root: experiments.metrics_key(EXPERIMENT),
        f".tcip/experiments/{EXPERIMENT}/metrics.jsonl", pin=_pin_platform_root,
        root_of=lambda root: Path(experiments.experiments_scope())),
    "operating_point_sidecar": Registered(
        {"schema_version": 2, "trait": "catkin_50per_date", "dataset_hash": "9f2c1b0a4d6e8f31",
         "operating_point": {"conf": {"name": "conf", "value": 0.42, "source": "derived",
                                      "validated_against": "held_out_annotations"}},
         "id_map": {"catkin": 0}, "validated": True, "shippable_issues": [],
         "checkpoint": "ü_best", "raster_path": None},
        lambda root: resolution.sidecar_key(_stamp_bucket(root), "operating_point"),
        "predictions/live/2026-03-04/operating_point.json", root_of=_stamp_bucket),
    "classifier_operating_point_sidecar": Registered(
        {"schema_version": 2,
         "operating_point": {"classifier": {"validated_against": "held_out_annotations",
                                            "value": "elongiert"}},
         "validated": True, "failures": [], "gate_evidence": {"kappa": 0.81}},
        lambda root: resolution.sidecar_key(_stamp_bucket(root), "classifier_operating_point"),
        "predictions/live/2026-03-04/classifier_operating_point.json", root_of=_stamp_bucket),
    "ordinal_operating_point_sidecar": Registered(
        {"schema_version": 2,
         "operating_point": {"ordinal": {"validated_against": "held_out_annotations",
                                         "criterion": "quadratic_weighted_kappa"}},
         "validated": True, "failures": [], "gate_evidence": {"qwk": 0.77},
         "trait": "ü_ordinal_trait"},
        lambda root: resolution.sidecar_key(_stamp_bucket(root), "ordinal_operating_point"),
        "predictions/live/2026-03-04/ordinal_operating_point.json", root_of=_stamp_bucket),
    "regression_operating_point_sidecar": Registered(
        {"schema_version": 2,
         "operating_point": {"regression": {"validated_against": "held_out_annotations",
                                            "criterion": "r_squared"}},
         "validated": False, "failures": ["insufficient_holdout"],
         "gate_evidence": {"score": None, "score_state": "nan"}, "trait": "ü_regression_trait"},
        lambda root: resolution.sidecar_key(_stamp_bucket(root), "regression_operating_point"),
        "predictions/live/2026-03-04/regression_operating_point.json", root_of=_stamp_bucket),
    "resolve_scale_sidecar": Registered(
        # every key scale_tools.calibrate_physical_scale's stamp literal writes, "unit" as the
        # writer spells it rather than "units"
        {"schema_version": 2,
         "operating_point": {"scale": {
             "name": "scale_mm_per_px", "value": 0.271, "unit": "mm", "source": "derived",
             "derived_from": "mean of 3 'reference_object' reference object(s), the calibration "
                              "half of the locked reference split",
             "requires_validation": True, "validation_kind": "physical",
             "validated_against": "physical_measurement",
             "capture_scoped": True, "capture_id": "2026-03-04_handheld"}},
         "validated": True, "validated_by": None, "failures": [],
         "gate_evidence": {"calibration_implied_scales": {"a_1": 0.27}},
         "trait": "büsch_length", "reference_subject": "reference_object",
         "reference_csv": "references/scale_reference.csv",
         "produced_at": "2026-03-04T12:00:00+00:00"},
        lambda root: resolution.sidecar_key(_stamp_bucket(root), "resolve_scale"),
        "predictions/live/2026-03-04/resolve_scale.json", root_of=_stamp_bucket),
    "raster_pass_progress": Registered(
        {"schema_version": 1,
         "raster_identity": {"width": 100, "height": 100, "num_channels": 3, "dtype": "uint8",
                             "pixel_checksum": "ab12", "seed": 0, "window_size": 1024,
                             "max_windows": 8, "pixel_fraction": 1.0, "band_interpretations": None,
                             "geotransform": None},
         "checkpoint_sha256": "0" * 64, "trait": "büsch_count", "experiment_id": None,
         "tile_batch_size": 96,
         "operating_point": {"conf": 0.42, "cross_tile_nms": 0.5, "max_dets": None,
                             "tile_size": 512, "overlap": 0.2, "tile_resize": None,
                             "postprocess": "nms", "require_masks": False}},
        lambda root: inference_tools._raster_pass_key(_stamp_bucket(root), "identity"),
        "predictions/live/2026-03-04/.tcip/raster_pass_progress/identity.json",
        root_of=_stamp_bucket),
    "trait_specs": Registered(
        {"name": TRAIT_UNDER_TEST, "delivers": ["measure_one"], "notes": "ü"},
        _trait_spec_key, f".tcip/state/trait_specs/{TRAIT_UNDER_TEST}.json",
        root_of=_trait_specs_root),
    "annotation_records": Registered(
        LABEL_BYTES, lambda root: json_io.annotation_record_key(_generic_label_dir(root), "a_1"),
        "labels/a_1.json", root_of=_generic_label_dir),
    "label_baselines": Registered(
        LABEL_BYTES,
        lambda root: review_engine.label_baseline_key(_generic_label_dir(root), "a_1"),
        "labels/.original/a_1.json", root_of=_generic_label_dir),
    "annotation_stats": Registered(
        {"sessions": [{"user": "ü", "images_annotated": 1, "total_annotations": 3,
                       "total_time_seconds": 42.5}]},
        lambda root: sessions.annotation_stats_key(str(root)), ".tcip/state/annotation_stats.json"),
    "band_group_manifest": Registered(
        BAND_GROUP_MANIFEST_BYTES,
        lambda root: band_groups.band_group_manifest_key(_band_group_dir(root), "cap_ü"),
        f"images/cap_ü{band_groups.MANIFEST_EXT}", root_of=_band_group_dir),
    "flat_image": Registered(
        IMAGE_BYTES,
        lambda root: image_utils.flat_image_key(_band_group_dir(root), "cap_ü.jpg"),
        "images/cap_ü.jpg", root_of=_band_group_dir),
    "run_checkpoint": Registered(
        CHECKPOINT_BYTES, lambda root: generic_trainer.checkpoint_key(root, "model_best"),
        "model_best.pt"),
    "model_snapshot_manifest": Registered(
        {"builder": "my_module:build",
         "files": [{"file": f"{SNAPSHOT_CONTENT}/{SNAPSHOT_FILENAME}", "sha256": "0" * 64,
                    "bytes": 27}],
         "missing": [], "snapshot_errors": [], "seed": 7, "notes": "ü"},
        lambda root: model_build.snapshot_manifest_key(root / EXPERIMENT),
        f"{EXPERIMENT}/model_src/manifest.json"),
    "model_snapshot_file": Registered(
        SNAPSHOT_BYTES,
        lambda root: model_build.snapshot_file_key(root, SNAPSHOT_CONTENT, SNAPSHOT_FILENAME),
        f"model_src/{SNAPSHOT_CONTENT}/{SNAPSHOT_FILENAME}"),
    "evaluation_results": Registered(
        {"map50": 0.5, "tp": 3, "eval_regime": "full-frame-single-pass", "trait": "messgröße",
         "loss": None, "loss_state": "positive_infinity"},
        eval_runners.evaluation_results_key, "test_results.json"),
    "cal_holdout_split_lock": Registered(
        _real_cal_holdout_lock(),
        lambda root: splits.cal_holdout_lock_key(LOCK_IDENTITY, scope_root=root),
        f".tcip/artifacts/cal_holdout_split_{LOCK_IDENTITY}.json"),
    "hpo_sweep_manifest": Registered(
        {"study_name": STUDY, "status": "running", "n_trials": 2,
         "param_space": {"lr": [0.1, 0.01]},
         "base_config": {"model_source": {"builder": "x:y"}, "data": {}, "training": {}},
         "grace_period": 5, "reduction_factor": 3, "max_concurrent": 1,
         "warm_start": False, "baseline_params": None, "resources_per_trial": None,
         "cancel_requested": False},
        lambda root: training_tools.sweep_manifest_key(STUDY),
        f".tcip/hpo/{STUDY}/manifest.json", pin=_pin_platform_root,
        root_of=lambda root: training_tools.hpo_root()),
    "hpo_study_result": Registered(
        {"best_params": {"lr": 0.01}, "best_value": 0.25, "n_trials": 2,
         "all_trials": [
             {"params": {"lr": 0.1}, "value": None,
              "value_state": "positive_infinity", "state": "ERROR",
              "error": "RuntimeError: worker exploded before reporting"},
             {"params": None, "value": None, "iterations": None, "state": "ERROR",
              "error": "the trial never answered Ray: its actor died during start"},
         ]},
        lambda root: training_tools.study_result_key(STUDY), f".tcip/hpo/{STUDY}.json",
        pin=_pin_platform_root, root_of=lambda root: training_tools.hpo_root()),
    "hpo_trial_config": Registered(
        {"training": {"batch_size": 4}, "trial_params": {"lr": 0.01}, "unconsumed_params": []},
        lambda root: training_tools.trial_config_key(training_tools.sweep_dir(STUDY), TRIAL_DIR),
        f".tcip/hpo/{STUDY}/{TRIAL_DIR}/resolved_config.json", pin=_pin_platform_root,
        root_of=lambda root: training_tools.sweep_dir(STUDY)),
    "ray_dashboard": Registered(
        {"url": "http://127.0.0.1:8265", "pid": 4242, "started_at": "2026-01-01T00:00:00+00:00"},
        lambda root: hpo.ray_dashboard_key(), ".tcip/state/ray_dashboard.json",
        pin=_pin_platform_root),
    "dataset_registry": Registered(
        [{"id": "a1", "path": "dü", "crop": "hazelnut", "fingerprint": "9f2c"}],
        project_tools.dataset_registry_key, ".tcip/datasets.json"),
    "split_manifest": Registered(
        _real_split_manifest(),
        lambda root: data_tools.split_manifest_key(_split_dir(root)),
        "splits/split_manifest.json", root_of=_split_dir),
    "curated_manifest": Registered(
        {"created": "2026-03-04T00:00:00+00:00", "subject": "catkin", "subjects": ["catkin"],
         "images": [{"image": "ü_2.jpg", "status": "hard_negative", "n_boxes": 0,
                     "rejected_count": 1, "label": "labels/ü_2.json"}]},
        lambda root: materialize.curated_manifest_key(_curated_dir(root)),
        "curated/curated_manifest.json", root_of=_curated_dir),
    "coco_documents": Registered(
        COCO_BYTES, lambda root: format_io.coco_document_key(_coco_dir(root), "instances"),
        "export/instances.json", root_of=_coco_dir),
    "audit_log": Registered(
        {"timestamp": "2026-03-04T12:00:00+00:00", "tool": "gui_save_labels",
         "arguments": {"image_path": "images/2026-03-04/a_1.JPG", "n_annotations": 3},
         "status": "ok", "source": "gui"},
        lambda root: audit.audit_log_key(root), ".tcip/audit.jsonl"),
    "learning_capture": Registered(
        {"ts": "2026-03-04T12:00:00+00:00", "session_id": "s_1", "reason": "clear",
         "active_project": "grüne_reihe", "note": "session ended"},
        lambda root: agent_learning_capture.learning_capture_key(root),
        ".tcip/learning_capture.jsonl"),
    "hpo_trial_metrics": Registered(
        {"epoch": 1, "val_loss": 0.25, "selection": 0.25},
        lambda root: training_tools.trial_metrics_key(training_tools.sweep_dir(STUDY), TRIAL_DIR),
        f".tcip/hpo/{STUDY}/{TRIAL_DIR}/metrics.jsonl", pin=_pin_platform_root,
        root_of=lambda root: training_tools.sweep_dir(STUDY)),
    "run_launch_config": Registered(
        {"model_source": {"builder": "my_module:build"}, "data": {"subject": "büsch"},
         "device": "cpu"},
        lambda root: training_tools.launch_config_key(root), "launch_config.json"),
    "confidence_sweep": Registered(
        {"schema_version": 2, "trait": "messgröße", "dataset_hash": "d41d8cd98f00b204",
         "checkpoint_sha256": "0" * 64,
         "gate_evidence": [{"conf": 0.1, "f1": 0.4}, {"conf": 0.2, "f1": 0.6}]},
        lambda root: inference_tools.calibration_curve_key(SWEEP_IDENTITY),
        f".tcip/artifacts/operating_point_sweep_{SWEEP_IDENTITY}.json", pin=_pin_platform_root),
    "imagery": Registered(
        IMAGE_BYTES,
        lambda root: dataset_layout.image_key(root, IMAGE_DATE, IMAGE_STEM, IMAGE_EXT),
        f"images/{IMAGE_DATE}/{IMAGE_STEM}{IMAGE_EXT}"),
    "plant_mapping": Registered(
        {"2026-03-04": [{"image_path": "images/2026-03-04/a_1.JPG", "stem": "a_1",
                         "date_folder": "2026-03-04", "plot_name": "plot_ü",
                         "accession_name": "ü", "source": "sequence", "distance_m": 1.25}]},
        lambda root: plant_mapping.plant_mapping_key(root, "valley"),
        ".tcip/state/plant_mappings/valley.json", root_of=_plant_mapping_dir),
    "plant_registries": Registered(
        {"name": "valley-plants", "crop": "hazelnut", "site": "north orchard",
         "csvs": [{"path": "dü/plants.csv", "sha256": "0" * 64, "n_plants": 2}],
         "n_plants": 2, "digest": "0" * 64, "registered_by": "agent:register_plant_registry",
         "registered_at": "2026-03-04T12:00:00+00:00"},
        lambda root: plant_mapping.plant_registry_key(root, "valley-plants"),
        ".tcip/state/plant_registries/valley-plants.json", root_of=_plant_mapping_dir),
    "delivery_supersessions": Registered(
        {"superseded_event_id": EVENT_ID_UNDER_TEST, "output_sha256": "0" * 64,
         "replacement_event_id": None, "reason": "a mis-stated crop was corrected upstream",
         "superseded_by": "agent:supersede_delivery",
         "superseded_at": "2026-03-04T12:00:00+00:00"},
        lambda root: resolution.delivery_supersession_key(
            resolution.delivery_events_scope(root), EVENT_ID_UNDER_TEST),
        f".tcip/state/delivery_supersessions/{EVENT_ID_UNDER_TEST}.json",
        root_of=lambda root: resolution.delivery_events_scope(root)),
    # the experiment record's validation member
    "experiment_validations": Registered(
        # every field _VALIDATION_FIELDS requires, train_disjointness/selection_disjointness
        # included, the latter the full shape resolver_selection_disjointness actually returns
        {"document": "operating_point", "trait": "catkin_50per_date",
         "claim": {"operating_point": {"conf": {"value": 0.42}}},
         "validated_against": "held_out_annotations", "checkpoint_sha256": "0" * 64,
         "producing_experiment_id": EXPERIMENT,
         "reference_identity": {"calibration_dataset_hash": "9f2c1b0a4d6e8f31"},
         "covered_buckets": {"predictions/live/2026-03-04": "7f3a1b9c2d4e5f60"},
         "dataset_root": "dü", "recorded_at": "2026-03-04T12:00:00+00:00",
         "train_disjointness": {"checked": True, "group_check": None},
         "selection_disjointness": _real_selection_disjointness()},
        lambda root: experiments.validations_key(EXPERIMENT),
        f".tcip/experiments/{EXPERIMENT}/validations.jsonl", pin=_pin_platform_root,
        root_of=lambda root: Path(experiments.experiments_scope())),
    # what one trait's delivered number means, for one delivery kind
    "trait_operationalizations": Registered(
        {"trait": TRAIT_UNDER_TEST, "delivery_kind": DELIVERY_KIND_UNDER_TEST,
         "statement": "the date each büsch reached the measured state",
         "mechanism": "the calibrated state classifier over isolated catkins",
         "measured_subject": "catkin", "delivered_phenotypes": ["measure_one"],
         "delivered_value_keys": [], "stated_by": "state_trait_operationalization",
         "stated_at": "2026-03-04T12:00:00+00:00", "relayed_note": "",
         "confirmed_by": "user:ü", "confirmed_at": "2026-03-04T12:30:00+00:00",
         "identity_from_request": True, "confirmed_fields": {"milestone_on": "positive_fraction"}},
        _operationalization_key,
        f".tcip/state/trait_operationalizations/{TRAIT_UNDER_TEST}/{DELIVERY_KIND_UNDER_TEST}.json",
        root_of=_operationalizations_root),
    # what a trait spec means and why the agent chose it, proposed and confirmed separately from
    # the spec record itself
    "trait_spec_statements": Registered(
        {"trait": TRAIT_UNDER_TEST,
         "statement_fields": {"delivers": ["measure_one"], "positive_class_name": "büsch"},
         "rationale": "the breeder described the state directly", "stated_by": "author_trait_spec",
         "stated_at": "2026-03-04T12:00:00+00:00", "relayed_note": "",
         "confirmed_by": "user:ü", "confirmed_at": "2026-03-04T12:30:00+00:00",
         "identity_from_request": True,
         "record_seen": "7f3a1b9c2d4e5f60"},
        _trait_spec_statement_key, f".tcip/state/trait_spec_statements/{TRAIT_UNDER_TEST}.json",
        root_of=_trait_spec_statements_root),
    # one completed delivery, carrying the real per-bucket StampBinding evidence it shipped under
    "delivery_events": Registered(
        {"event_id": EVENT_ID_UNDER_TEST, "trait": TRAIT_UNDER_TEST,
         "delivery_kind": DELIVERY_KIND_UNDER_TEST, "door": "deliver_phenology_milestones",
         "output_path": "büsch_phenology.csv", "output_sha256": "0" * 64,
         "measurement_documents": ["operating_point", "classifier_operating_point"],
         "scale_document": None,
         "acknowledged_by": None, "acknowledgement_reason": None,
         "plant_mapping": {
             "name": "valley", "project_root": "P:/valley", "dataset_id": "ds-1",
             "dataset_root": "dü", "built_at": "2026-03-04T12:00:00+00:00",
             "record_sha256": "0" * 64, "nn_tolerance_m": {"value": 3.0, "source": "stated"},
             "capture_identity": {"2026-03-04": "0" * 16}, "captures_unverified": [],
             "plant_csvs_unverified": [], "dates_delivered": ["2026-03-04"],
             "images_unattributed": 0, "images_unattributed_scope": "delivered_dates",
             "plant_attribution": "image"},
         "documents": {"predictions/live/2026-03-04": {
             "ok": True, "claimed": True, "experiment_id": EXPERIMENT,
             "producing_experiment_id": EXPERIMENT, "checkpoint_sha256": "0" * 64,
             "record_digest": "7f3a1b9c2d4e5f60", "note": ""}},
         "produced_at": "2026-03-04T12:00:00+00:00"},
        lambda root: resolution.delivery_event_key(
            resolution.delivery_events_scope(root), EVENT_ID_UNDER_TEST),
        f".tcip/state/delivery_events/{EVENT_ID_UNDER_TEST}.json",
        root_of=lambda root: resolution.delivery_events_scope(root)),
}

CODEC_EXEMPT = {
    "backend_port": "the value is the port text itself",
    "retrospectives": "the value is the markdown document itself",
    "workspace_active_project": "the value is the active project's name",
}
"""Every registered record or log whose codec is deliberately not the canonical constant.

Naming one here is what makes an exemption a decision somebody made rather than a default
somebody got: a store that picks its own spelling fails the identity check below until its
reason is written down.
"""


def test_every_registered_store_has_a_byte_and_path_identity_case():
    """A store registered without one would be a placement nothing has checked.

    Stores a test declares for its own scaffolding are not the platform's, and are told
    apart by the module that declared them rather than by a naming convention.
    """
    declared = {
        name for name in ts.registered_stores()
        if not ts.get_descriptor(name).declared_in.startswith(("tests", "test_"))
    }
    assert declared == set(REGISTERED)


def test_the_delivery_events_golden_sample_validates_against_its_declared_shape():
    """The registered golden here and ``DeliveryEventRecord`` (``delivery_events_schema.py``) are
    two statements of the same shape; a golden the model refuses would mean the two had already
    drifted apart without either side's own tests catching it."""
    DeliveryEventRecord.model_validate(REGISTERED["delivery_events"].golden)


def test_every_json_store_encodes_through_the_one_codec_its_kind_declares():
    """One record spelling and one log spelling across the platform, or a named reason.

    Identity rather than equality: a second instance carrying the same fields is still a
    second implementation, and it is the one that would drift.
    """
    off_canon = {}
    for name in ts.registered_stores():
        descriptor = ts.get_descriptor(name)
        if descriptor.declared_in.startswith(("tests", "test_")) or descriptor.kind == "blob":
            continue
        expected = ts.RECORD_JSON if descriptor.kind == "record" else ts.LOG_JSON
        if descriptor.codec is not expected and name not in CODEC_EXEMPT:
            off_canon[name] = type(descriptor.codec).__name__

    assert off_canon == {}
    assert set(CODEC_EXEMPT) <= set(ts.registered_stores())


def test_the_canonical_record_codec_writes_the_bytes_this_test_spells_out():
    """The one place the record spelling is pinned, so changing it shows up in a diff here
    rather than rippling silently through every store that carries it."""
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
    """Calling str() on whatever arrived would fabricate a value out of its repr, the way a
    JSON default does, so a text store takes text and the caller formats the rest."""
    _pin_workspace(store.root, monkeypatch)
    key = web_client.backend_port_key()

    with pytest.raises(ts.StoreError):
        ts.replace(key, 8765)

    ts.replace(key, "8765")
    assert ts.read(key) == "8765"


def test_a_value_the_codec_cannot_spell_names_the_store_the_key_and_the_type(store):
    """The refusal has to say which entry and what about it, since json.dumps names neither,
    and the same store takes the explicitly converted value."""
    key = store.key(LWW, "convertible")

    with pytest.raises(ts.StoreError) as raised:
        ts.replace(key, {"where": Path("a/b")})
    message = str(raised.value)
    assert LWW in message and "convertible" in message and "PosixPath" in message or "Path" in message

    ts.replace(key, {"where": Path("a/b").as_posix()})
    assert ts.read(key) == {"where": "a/b"}


def test_a_non_finite_number_is_refused_rather_than_written_as_a_word_json_has_no_type_for(store):
    """NaN and Infinity are not JSON and no strict parser reads them, so a producer states
    the non-finite value instead of the codec inventing a spelling for it."""
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

    if case.pin is not None:
        case.pin(seam_root, monkeypatch)
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


def test_a_sanitized_shard_name_places_the_file_the_review_engine_places_it_at(tmp_path):
    """An image key carrying a separator is one filename, a bucket key carrying separators is one
    directory, and the keys recoverable from that path place the very same file."""
    state_dir = _review_state_dir(tmp_path)
    engine = review_engine.ReviewEngine(state_dir, current_user="ü")
    key = review_engine.review_verdict_key(state_dir, "predictions/live/2026-03-04", "a/b.jpg")
    locator = ts.get_descriptor(review_engine.REVIEW_VERDICTS_STORE).locator

    placed = locator.relative_path(str(state_dir), key.parts)
    assert Path(state_dir, *placed.parts) == engine._shard_path(*key.parts)
    assert len(placed.parts) == 3  # review/<one bucket dir>/<one shard file>
    recovered = locator.parts_from(placed)
    assert recovered != key.parts
    assert locator.relative_path(str(state_dir), recovered) == placed


def test_a_verdict_with_no_prediction_bucket_places_its_shard_under_the_review_dir(tmp_path):
    """A ground-truth-only review names no bucket, and its shard sits directly under ``review/``
    rather than in a directory standing in for one."""
    state_dir = _review_state_dir(tmp_path)
    key = review_engine.review_verdict_key(state_dir, review_engine.NO_BUCKET, "a_1.jpg")
    locator = ts.get_descriptor(review_engine.REVIEW_VERDICTS_STORE).locator

    placed = locator.relative_path(str(state_dir), key.parts)
    assert placed == PurePosixPath("review/a_1.jpg.json")
    assert locator.parts_from(placed) == key.parts


def test_enumerating_review_verdicts_answers_with_identities_that_read_back(store):
    """``keys`` returns identities, not whatever a path happened to be able to spell.

    The shard filename sanitizes a separator out of the image name and folds a bucket key into
    one directory, so the path alone recovers a key that names a different entry. Both backends
    answer with the key the payload states, and the proof that it is an identity rather than a
    label is that reading it back works.
    """
    state_dir = _review_state_dir(store.root)
    bucket, image = "predictions/live/2026-03-04", "a/b.jpg"
    key = review_engine.review_verdict_key(state_dir, bucket, image)
    ts.replace(
        key,
        {"bucket": bucket, "img_name": image, "state": {"img_status": "completed"}},
        expect=ts.Version.ABSENT,
    )

    found = ts.keys(review_engine.REVIEW_VERDICTS_STORE, str(state_dir))
    assert found == [key]
    assert ts.read(found[0])["img_name"] == image


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
    """A source path that isn't there refuses before the destination is touched at all: the
    version check ahead of it passes (there is nothing stale about it), and only then does the
    missing source itself raise, leaving whatever the destination held untouched.

    Coverage, not a fix: fsync-before-replace on a successful write is held by inspection of
    ``put_blob_from_path``'s one implementation (``store.py``), never exercised by a test, since
    there is no portable way to observe an fsync from outside the process that issued it.
    """
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
    """``write_annotations`` addresses a real dataset's label through the store's own generic,
    directory-rooted key, never a layout-specific one; ``dataset_layout``'s readers find the same
    file by plain path arithmetic instead. The two must name one file, or a document the store
    places would sit somewhere its own layout's readers never look.
    """
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
