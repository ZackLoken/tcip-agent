"""The storage seam's contract: what every backend must do, whatever it stores bytes in.

Most of these run against real OS processes rather than threads, because the guarantee under
test is exactly the one threads cannot check: two interpreters sharing a lock registry would
pass a same-process test against a backend with no cross-process exclusion at all.

The isolation cases are the ones that pin the defect this layer exists to remove, so they
are also run against a backend whose writes skip the lock (set
``TCIP_STORE_CONTRACT_UNLOCKED=1``) and observed failing there. Everything below the file
annex is backend-independent: a second backend joins the ``store`` fixture's params and must
pass it unchanged.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

import pytest

import tcip_store as ts
from tcip_annotation import json_io, review_engine
from tcip_annotation.state import Annotation, BBox
from tcip_mcp import class_registry, dataset_layout, experiments, model_registry, workspace
from tcip_mcp.tools import project_tools
from tcip_mcp.utils.atomic_io import atomic_write_json
from tcip_store.file_backend import FileBackend, RootedFileLocator
from tcip_web import jobstore
from tcip_web.routes import canvas
from tests._store_worker import (
    BLOB,
    CAS,
    LOG,
    LWW,
    NESTED,
    OPAQUE,
    RELAXED,
    SEALED_BLOB,
    STRICT,
    make_backend,
    register_contract_stores,
    wait_for,
)

register_contract_stores()

_WORKER = Path(__file__).with_name("_store_worker.py")


@dataclass
class Harness:
    """The bound backend, the scope its keys hang off, and how to reach bytes behind it."""

    backend: FileBackend
    root: Path
    procs: list[subprocess.Popen] = field(default_factory=list)

    def key(self, store: str, *parts: str) -> ts.Key:
        return ts.Key(store, str(self.root), tuple(parts))

    def path(self, key: ts.Key) -> Path:
        """Where the file backend puts a key. The file annex uses this; nothing above does."""
        return self.backend.path_for(key)

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


@pytest.fixture(params=["file"])
def store(request, tmp_path):
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

    with pytest.raises(ValueError):
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
    first = store.key(LWW, "alpha")
    second = store.key(LWW, "beta")
    marker = store.root / "applied.marker"
    proc = store.spawn("pause-mid-apply", store.root, "alpha", "beta", marker, 30)
    wait_for(marker, timeout_s=60)
    proc.kill()
    proc.wait(timeout=30)

    assert ts.read(first) == {"who": "alpha"}
    assert ts.read(second, default=None) is None


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


def test_a_torn_tail_is_held_back_and_an_interior_corruption_is_reported(store):
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
    key = store.key(LOG, "all-fragment")
    ts.append(key, {"i": 0})
    path = store.path(key)
    path.write_bytes(b'{"i": 0, "unfin')

    ts.append(key, {"i": 1})

    page = ts.read_log(key)
    assert [r["i"] for r in page.records] == [1]
    assert not page.torn_tail and page.corrupt == ()


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
        ts.register_store(ts.StoreDescriptor(name=LWW, kind="record", key_fields=("name",)))
    assert "already registered" in str(duplicate.value)

    with pytest.raises(ValueError) as unpoliced:
        ts.register_store(
            ts.StoreDescriptor(name="contract_unpoliced", kind="record", key_fields=("name",),
                               codec=ts.json_codec())
        )
    assert "concurrency=" in str(unpoliced.value)

    with pytest.raises(ValueError) as relaxed_log:
        ts.register_store(
            ts.StoreDescriptor(name="contract_relaxed_log", kind="log", key_fields=("name",),
                               codec=ts.json_codec(indent=None), durable=False)
        )
    assert "durability" in str(relaxed_log.value)

    declared = ts.register_store(
        ts.StoreDescriptor(
            name="contract_late_declaration",
            kind="record",
            key_fields=("name",),
            codec=ts.json_codec(),
            concurrency="last_writer_wins",
            locator=RootedFileLocator(prefix=("late",), suffix=".json"),
        )
    )
    assert declared.declared_in == __name__
    assert "contract_late_declaration" in ts.registered_stores()


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
    key = store.key(LWW, "sometimes-there")
    with pytest.raises(ts.NotFound) as raised:
        ts.read(key)
    assert "default=" in str(raised.value)
    assert ts.read(key, default={"fallback": True}) == {"fallback": True}

    ts.replace(key, {"n": 1})
    assert ts.read(key) == {"n": 1}
    store.path(key).write_bytes(b"{not json at all")
    with pytest.raises(ts.DecodeError):
        ts.read(key, default={"fallback": True})


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


def test_a_rooted_locator_inverts_its_own_placement(store):
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


def test_a_codec_writes_the_exact_bytes_its_spelling_declares():
    payload = {"b": 1, "a": "ü"}

    assert ts.json_codec().encode(payload) == json.dumps(payload, indent=2, default=str).encode()
    assert ts.json_codec(indent=1, ensure_ascii=False, default=None, allow_nan=False).encode(
        payload
    ) == json.dumps(payload, ensure_ascii=False, indent=1, allow_nan=False).encode()
    assert ts.json_codec(indent=None, ensure_ascii=False, separators=(",", ":")).encode(
        payload
    ) == json.dumps(payload, separators=(",", ":"), ensure_ascii=False, default=str).encode()
    assert ts.json_codec(default=None, trailing_newline=True).encode(payload) == (
        json.dumps(payload, indent=2) + "\n"
    ).encode()
    assert ts.text_codec().encode("port 8765") == b"port 8765"


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
    key = store.key(LWW, "enumerated")
    ts.replace(key, {"n": 1})
    directory = store.path(key).parent

    (directory / "enumerated.json.lock").write_bytes(b"")
    (directory / ".enumerated.json.abc123.tmp").write_bytes(b"{}")
    (directory / "tmpabc123.tmp").write_bytes(b"{}")

    assert ts.keys(LWW, str(store.root)) == [key]


# ── the platform's own stores: same bytes, same path ────────────────────────────


def _write_status(root: Path) -> Path:
    path = dataset_layout.image_status_path(root)
    atomic_write_json(path, {"catkin/2026-03-04": {"a_1.jpg": "negative", "ü_2.jpg": "complete"}})
    return path


def _write_status_digest(root: Path) -> Path:
    path = dataset_layout.image_status_digest_path(root)
    atomic_write_json(path, {"catkin/2026-03-04": {"a_1.jpg": "9f2c"}})
    return path


def _write_view_coverage(root: Path) -> Path:
    path = dataset_layout.view_coverage_path(root)
    atomic_write_json(path, {"catkin/2026-03-04": {"a_1.jpg": {
        "grid": {"rows": 3, "cols": 3}, "cells_served_at_native": ["r1c1"],
        "cells_swept": ["r1c1"]}}})
    return path


def _write_region_completeness(root: Path) -> Path:
    path = dataset_layout.region_completeness_path(root)
    atomic_write_json(path, {"catkin/orthö": {"grid": {"rows": 2, "cols": 2},
                                              "cells_complete": ["r1c1"], "stem": "orthö"}})
    return path


def _write_region_completeness_digest(root: Path) -> Path:
    path = dataset_layout.region_completeness_digest_path(root)
    atomic_write_json(path, {"catkin/orthö": {"r1c1": "3ab9"}})
    return path


def _write_class_registry(root: Path) -> Path:
    path = dataset_layout.classes_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    registry = class_registry.registry_from_dict({"catkin": {"description": "a männlich flower"}})
    class_registry.write_registry(path, registry)
    return path


def _write_dataset_identity(root: Path) -> Path:
    (root / "images").mkdir(parents=True, exist_ok=True)
    project_tools.register_dataset(str(root), crop="hazelnut")
    return dataset_layout.dataset_identity_path(root)


def _write_labels(root: Path) -> Path:
    path = dataset_layout.annotation_path(root, "2026-03-04", "a_1")
    json_io.write_annotations(
        path, [Annotation(subject="catkin", geometry=BBox(1.0, 2.0, 3.5, 4.5),
                          attributes={"elongation": "elongiert"}, created_by="ü")], 640, 480)
    return path


def _write_predictions(root: Path) -> Path:
    path = dataset_layout.prediction_path(root, "live", "2026-03-04", "a_1")
    json_io.write_annotations(
        path, [Annotation(subject="catkin", geometry=BBox(1.0, 2.0, 3.5, 4.5), score=0.75)],
        640, 480)
    return path


def _review_state_dir(root: Path) -> Path:
    return root / ".tcip" / "state"


def _write_review_verdict(root: Path) -> Path:
    engine = review_engine.ReviewEngine(_review_state_dir(root), current_user="ü")
    engine.mark_image_reviewed("a_1.jpg")
    return engine._shard_path("a_1.jpg")


def _write_canvas_meta(root: Path) -> Path:
    path = canvas.meta_path(str(root))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas._write_json_no_fsync(path, {"schema_version": 1, "tab": "annotate", "image": "ü.jpg"})
    return path


def _write_canvas_geometry(root: Path) -> Path:
    path = canvas.shapes_path(str(root))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas._write_json_no_fsync(path, {"image_path": "ü.jpg", "tab": "annotate", "shapes": []})
    return path


def _write_model_registry(root: Path) -> Path:
    checkpoint = root / "weights.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"stand-in for checkpoint bytes")
    model_registry.ModelRegistry(str(root)).register_model(
        "hazelnut_catkin_detector_v1", str(checkpoint), {"epochs": 3}, kind="tcip_module")
    return model_registry.registry_index_path(root)


def _write_job_registry(root: Path) -> Path:
    jobstore.persist("inference_jobs", [{"id": "j1", "status": "completed"}])
    return jobstore._state_path("inference_jobs")


def _write_active_marker(root: Path) -> Path:
    workspace.set_active_project("hazelnut_catkin_valley")
    return workspace.active_marker_path()


def _created_experiment(root: Path) -> Path:
    if not experiments._exp_dir(EXPERIMENT).exists():
        experiments.create_experiment(EXPERIMENT, {"model": "faster_rcnn"}, data_source="ds")
    return experiments._exp_dir(EXPERIMENT)


def _write_experiment_member(root: Path, member: str) -> Path:
    return _created_experiment(root) / f"{member}.json"


def _write_experiment_env(root: Path) -> Path:
    path = _created_experiment(root) / "env.json"
    atomic_write_json(path, {"env": {"python": "3.12"}, "seed": 42})
    return path


def _write_experiment_metrics(root: Path) -> Path:
    directory = _created_experiment(root)
    experiments.log_metrics(EXPERIMENT, 1, {"loss": 0.5})
    return directory / "metrics.jsonl"


def _pin_platform_root(root: Path, monkeypatch) -> None:
    monkeypatch.setenv("TCIP_PROJECT_ROOT", str(root))


def _pin_workspace(root: Path, monkeypatch) -> None:
    monkeypatch.setenv("TCIP_WORKSPACE", str(root))


EXPERIMENT = "exp_042"


@dataclass(frozen=True)
class Registered:
    """One registered store: how today's writer produces it, and how the seam addresses it.

    ``relative`` is the path under the root the writer is given. ``scope_of`` is the root the
    store's own keys hang off, which for the review shards and the experiment members is a
    directory below that.

    ``newline_translated`` marks a writer that opens its file in text mode, so on Windows
    the bytes it produces carry CRLF where the seam's byte write produces LF. The comparison
    then holds line for line rather than byte for byte, and says so, rather than a codec
    emitting different bytes on different platforms.
    """

    write_today: Callable[[Path], Path]
    key_of: Callable[[Path], ts.Key]
    relative: str
    pin: Callable[[Path, object], None] | None = None
    scope_of: Callable[[Path], Path] = lambda root: root
    newline_translated: bool = False


REGISTERED = {
    "image_status": Registered(
        _write_status, dataset_layout.image_status_key, ".tcip/state/image_status.json"),
    "image_status_digest": Registered(
        _write_status_digest, dataset_layout.image_status_digest_key,
        ".tcip/state/image_status_digest.json"),
    "view_coverage": Registered(
        _write_view_coverage, dataset_layout.view_coverage_key, ".tcip/state/view_coverage.json"),
    "region_completeness": Registered(
        _write_region_completeness, dataset_layout.region_completeness_key,
        ".tcip/state/region_completeness.json"),
    "region_completeness_digest": Registered(
        _write_region_completeness_digest, dataset_layout.region_completeness_digest_key,
        ".tcip/state/region_completeness_digest.json"),
    "class_registry": Registered(
        _write_class_registry, dataset_layout.class_registry_key, "classes.json"),
    "dataset_identity": Registered(
        _write_dataset_identity, dataset_layout.dataset_identity_key, "dataset.json",
        pin=_pin_platform_root),
    "labels": Registered(
        _write_labels, lambda root: dataset_layout.label_key(root, "2026-03-04", "a_1"),
        "annotations/2026-03-04/a_1.json", newline_translated=True),
    "predictions": Registered(
        _write_predictions,
        lambda root: dataset_layout.prediction_key(root, "live", "2026-03-04", "a_1"),
        "predictions/live/2026-03-04/a_1.json", newline_translated=True),
    "review_verdicts": Registered(
        _write_review_verdict,
        lambda root: review_engine.review_verdict_key(_review_state_dir(root), "a_1.jpg"),
        ".tcip/state/review/a_1.jpg.json", scope_of=_review_state_dir,
        newline_translated=True),
    "canvas_meta": Registered(
        _write_canvas_meta, lambda root: canvas.canvas_meta_key(str(root)),
        ".tcip/state/canvas_live.json", newline_translated=True),
    "canvas_geometry": Registered(
        _write_canvas_geometry, lambda root: canvas.canvas_geometry_key(str(root)),
        ".tcip/state/canvas_shapes.json", newline_translated=True),
    "model_registry": Registered(
        _write_model_registry, model_registry.registry_index_key, ".tcip/models/registry.json"),
    "job_registry": Registered(
        _write_job_registry, lambda root: jobstore.job_registry_key("inference_jobs"),
        ".tcip/state/inference_jobs.json", pin=_pin_platform_root),
    "workspace_active_project": Registered(
        _write_active_marker, lambda root: workspace.active_project_key(), ".active",
        pin=_pin_workspace),
    "experiment_config": Registered(
        lambda root: _write_experiment_member(root, "config"), lambda root: experiments.config_key(EXPERIMENT),
        f".tcip/experiments/{EXPERIMENT}/config.json", pin=_pin_platform_root,
        scope_of=lambda root: Path(experiments.experiments_scope())),
    "experiment_status": Registered(
        lambda root: _write_experiment_member(root, "status"), lambda root: experiments.status_key(EXPERIMENT),
        f".tcip/experiments/{EXPERIMENT}/status.json", pin=_pin_platform_root,
        scope_of=lambda root: Path(experiments.experiments_scope())),
    "experiment_lineage": Registered(
        lambda root: _write_experiment_member(root, "lineage"), lambda root: experiments.lineage_key(EXPERIMENT),
        f".tcip/experiments/{EXPERIMENT}/lineage.json", pin=_pin_platform_root,
        scope_of=lambda root: Path(experiments.experiments_scope())),
    "experiment_artifacts": Registered(
        lambda root: _write_experiment_member(root, "artifacts"), lambda root: experiments.artifacts_key(EXPERIMENT),
        f".tcip/experiments/{EXPERIMENT}/artifacts.json", pin=_pin_platform_root,
        scope_of=lambda root: Path(experiments.experiments_scope())),
    "experiment_env": Registered(
        _write_experiment_env, lambda root: experiments.env_key(EXPERIMENT),
        f".tcip/experiments/{EXPERIMENT}/env.json", pin=_pin_platform_root,
        scope_of=lambda root: Path(experiments.experiments_scope())),
    "experiment_metrics": Registered(
        _write_experiment_metrics, lambda root: experiments.metrics_key(EXPERIMENT),
        f".tcip/experiments/{EXPERIMENT}/metrics.jsonl", pin=_pin_platform_root,
        scope_of=lambda root: Path(experiments.experiments_scope()), newline_translated=True),
}


def _line_endings(raw: bytes, case: Registered) -> bytes:
    """What the seam must produce for bytes today's writer produced."""
    return raw.replace(b"\r\n", b"\n") if case.newline_translated else raw


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


@pytest.mark.parametrize("name", sorted(REGISTERED))
def test_a_registered_store_writes_the_bytes_and_the_path_its_writer_does(
    store, tmp_path, monkeypatch, name
):
    """Today's writer and the seam produce the same file, byte for byte, in the same place."""
    case = REGISTERED[name]
    descriptor = ts.get_descriptor(name)
    today_root, seam_root = tmp_path / "today", tmp_path / "seam"

    if case.pin is not None:
        case.pin(today_root, monkeypatch)
    written = case.write_today(today_root)
    raw = written.read_bytes()
    assert written.relative_to(today_root).as_posix() == case.relative

    if case.pin is not None:
        case.pin(seam_root, monkeypatch)
    key = case.key_of(seam_root)
    if descriptor.kind == "log":
        ts.append(key, descriptor.codec.decode(raw.rstrip(b"\n")))
    else:
        ts.replace(key, descriptor.codec.decode(raw), expect=ts.Version.ABSENT)

    landed = store.backend.path_for(key)
    assert landed.read_bytes() == _line_endings(raw, case)
    assert landed.relative_to(seam_root).as_posix() == case.relative
    relative_to_scope = landed.relative_to(case.scope_of(seam_root)).as_posix()
    assert descriptor.locator.parts_from(PurePosixPath(relative_to_scope)) == key.parts


def test_a_sanitized_shard_name_places_the_file_the_review_engine_places_it_at(tmp_path):
    """An image key carrying a separator is one filename, and the key recoverable from that
    filename places the very same file."""
    state_dir = _review_state_dir(tmp_path)
    engine = review_engine.ReviewEngine(state_dir, current_user="ü")
    key = review_engine.review_verdict_key(state_dir, "a/b.jpg")
    locator = ts.get_descriptor(review_engine.REVIEW_VERDICTS_STORE).locator

    placed = locator.relative_path(str(state_dir), key.parts)
    assert Path(state_dir, *placed.parts) == engine._shard_path("a/b.jpg")
    recovered = locator.parts_from(placed)
    assert recovered != key.parts
    assert locator.relative_path(str(state_dir), recovered) == placed
