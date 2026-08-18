"""Store declarations and the child-process bodies the storage-contract suite drives.

The contract's isolation and durability cases are only real across OS processes: threads in
one interpreter share a lock registry, so a same-process check would pass against a backend
that has no cross-process exclusion at all. This module is both the child script those cases
spawn (``python tests/_store_worker.py <command> ...``) and the parent's source of store
declarations, so both sides address exactly the same stores.

``TCIP_STORE_CONTRACT_BACKEND`` names which backend both sides construct, so a spawned child
writes through the same one the parametrized fixture bound in the parent.

Two weakened variants exist, each selected by its own environment variable, each what a set of
cases is observed failing against, and neither reachable from any shipped code path:
``TCIP_STORE_CONTRACT_UNLOCKED=1`` swaps in a file backend whose writes skip the key's lock,
and ``TCIP_STORE_CONTRACT_IGNORES_EXPECT=1`` swaps in a database backend that never compares
``expect``.
"""

from __future__ import annotations

import json
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import tcip_store as ts
from tcip_store.file_backend import FileBackend, RootedFileLocator
from tcip_store.layout_claims import ANY, Claim, Constant, PartPattern, Patterned
from tcip_store.sqlite_backend import SqliteBackend

CONTRACT_LAYOUT = "contract_root"
"""The kind of root these stores hang off, which no shipped store shares.

A store outside the platform's own claim table states where its entries live in its
descriptor, and the database backend refuses to serve one that does not, so these say it
here the same way a shipped store's row says it in the claim table.
"""

SECOND_LAYOUT = "another_root_kind"
"""A second kind of root, for the case where one directory serves two of them.

Two layouts' templates can describe one path, which is what the accounting has to notice
rather than pick between, so the suite needs a second layout to put a store under.
"""

OVERLAP_DIR = "overlap"
"""The directory the two overlapping stores below both place their documents in."""

OVERLAP_SERVED = "contract_overlap_second_layout"
OVERLAP_STRANDED = "contract_overlap_contract_layout"

CAS = "contract_cas"
LWW = "contract_lww"
RELAXED = "contract_relaxed"
NESTED = "contract_nested"
OPAQUE = "contract_opaque"
STRICT = "contract_strict"
LOG = "contract_log"
BLOB = "contract_blob"
SEALED_BLOB = "contract_sealed_blob"
STATE_FILES = "contract_state_files"

BACKEND_ENV = "TCIP_STORE_CONTRACT_BACKEND"
FILE, SQLITE = "file", "sqlite"

_registered = False


def directory_claim(directory: str, suffix: str) -> Claim:
    """A claim over ``<directory>/<name><suffix>``, the shape most of these stores carry."""
    return Claim(CONTRACT_LAYOUT, ((Constant(directory), Patterned(ANY, tail=suffix)),))


DOCUMENT_DIR = "documents"
"""The directory the suite's own single-document stores share.

Several stores over one file shape is the property those cases are about, which is the shape
thirteen shipped stores share under ``.tcip/state``. The directory is the suite's own so that
sharing a shape with each other does not also mean sharing a path with a shipped store, which
would make every one of their files a file two layouts claim.
"""


def document_claim(pattern: PartPattern = ANY) -> Claim:
    """A claim over ``documents/<document>.json``, the shape these stores share with each other."""
    return Claim(CONTRACT_LAYOUT, ((Constant(DOCUMENT_DIR), Patterned(pattern, tail=".json")),))


def overlap_claim(layout: str) -> Claim:
    """A claim over ``overlap/<document>.json``, spelled identically under two layouts."""
    return Claim(layout, ((Constant(OVERLAP_DIR), Patterned(ANY, tail=".json")),))


def register_contract_stores() -> None:
    """Declare the stores the contract suite addresses, once per process.

    Each store isolates one declaration the contract makes: the two concurrency policies,
    relaxed durability, a two-part key, a store with no enumeration, a store the encode
    refusals are exercised against, a log, two blob stores that differ only in whether they
    hand out a path, and one whose entries sit in the very directory a database backend keeps
    its own files in.
    """
    global _registered
    if _registered:
        return
    _registered = True
    ts.register_store(
        ts.StoreDescriptor(
            name=CAS,
            kind="record",
            key_fields=("name",),
            codec=ts.RECORD_JSON,
            concurrency="cas",
            enumerable=True,
            locator=RootedFileLocator(prefix=("cas",), suffix=".json"),
            claim=directory_claim("cas", ".json"),
        )
    )
    ts.register_store(
        ts.StoreDescriptor(
            name=LWW,
            kind="record",
            key_fields=("name",),
            codec=ts.RECORD_JSON,
            concurrency="last_writer_wins",
            enumerable=True,
            locator=RootedFileLocator(prefix=("lww",), suffix=".json"),
            claim=directory_claim("lww", ".json"),
        )
    )
    ts.register_store(
        ts.StoreDescriptor(
            name=RELAXED,
            kind="record",
            key_fields=("name",),
            codec=ts.RECORD_JSON,
            concurrency="last_writer_wins",
            durable=False,
            locator=RootedFileLocator(prefix=("relaxed",), suffix=".json"),
            claim=directory_claim("relaxed", ".json"),
        )
    )
    ts.register_store(
        ts.StoreDescriptor(
            name=NESTED,
            kind="record",
            key_fields=("group", "name"),
            codec=ts.RECORD_JSON,
            concurrency="last_writer_wins",
            enumerable=True,
            locator=RootedFileLocator(prefix=("nested",), suffix=".json"),
            claim=Claim(
                CONTRACT_LAYOUT,
                ((Constant("nested"), Patterned(ANY), Patterned(ANY, tail=".json")),),
            ),
        )
    )
    ts.register_store(
        ts.StoreDescriptor(
            name=OPAQUE,
            kind="record",
            key_fields=("name",),
            codec=ts.RECORD_JSON,
            concurrency="last_writer_wins",
            locator=RootedFileLocator(prefix=("opaque",), suffix=".json"),
            claim=directory_claim("opaque", ".json"),
        )
    )
    ts.register_store(
        ts.StoreDescriptor(
            name=STRICT,
            kind="record",
            key_fields=("name",),
            codec=ts.RECORD_JSON,
            concurrency="last_writer_wins",
            enumerable=True,
            locator=RootedFileLocator(prefix=("strict",), suffix=".json"),
            claim=directory_claim("strict", ".json"),
        )
    )
    ts.register_store(
        ts.StoreDescriptor(
            name=LOG,
            kind="log",
            key_fields=("name",),
            codec=ts.LOG_JSON,
            locator=RootedFileLocator(prefix=("logs",), suffix=".jsonl"),
            claim=directory_claim("logs", ".jsonl"),
        )
    )
    ts.register_store(
        ts.StoreDescriptor(
            name=BLOB,
            kind="blob",
            key_fields=("name",),
            path_readable=True,
            locator=RootedFileLocator(prefix=("blobs",), suffix=".bin"),
        )
    )
    ts.register_store(
        ts.StoreDescriptor(
            name=SEALED_BLOB,
            kind="blob",
            key_fields=("name",),
            locator=RootedFileLocator(prefix=("sealed",), suffix=".bin"),
        )
    )
    for name, layout in ((OVERLAP_SERVED, SECOND_LAYOUT), (OVERLAP_STRANDED, CONTRACT_LAYOUT)):
        ts.register_store(
            ts.StoreDescriptor(
                name=name,
                kind="record",
                key_fields=("document",),
                codec=ts.RECORD_JSON,
                concurrency="last_writer_wins",
                enumerable=True,
                locator=RootedFileLocator(prefix=(OVERLAP_DIR,), suffix=".json"),
                claim=overlap_claim(layout),
            )
        )
    ts.register_store(
        ts.StoreDescriptor(
            name=STATE_FILES,
            kind="blob",
            key_fields=("name",),
            enumerable=True,
            locator=RootedFileLocator(prefix=(".tcip",)),
        )
    )


class UnlockedWriteBackend(FileBackend):
    """A file backend whose writes take no lock, for observing what the lock is holding up.

    Everything else is the shipped path: the same staging, the same replace, the same
    version comparison. Only the mutual exclusion is gone, which is the difference between
    a write that waits for a concurrent transaction and one that lands inside its window.
    """

    @contextmanager
    def _locked(self, keys, timeout_s=None):
        for key in keys:
            self._ensure_parent(self.path_for(key), durable=True)
        yield


class PausingApplyBackend(FileBackend):
    """A file backend that pauses after each staged replace lands, so a kill hits mid-apply.

    ``pause_marker`` is written after every applied key, which is what lets the parent wait
    until the first replace is on disk before killing this process.
    """

    def __init__(self, *, pause_marker: Path, pause_s: float, **kwargs) -> None:
        super().__init__(**kwargs)
        self.pause_marker = pause_marker
        self.pause_s = pause_s

    def _apply_staged(self, temp, path, *, durable):
        super()._apply_staged(temp, path, durable=durable)
        self.pause_marker.write_text("applied", encoding="utf-8")
        time.sleep(self.pause_s)


class IgnoredExpectBackend(SqliteBackend):
    """A database backend whose writes never compare ``expect``, for observing what it holds up.

    Everything else is the shipped path: the same transaction, the same rows, the same version
    derived from the stored bytes. Only the comparison is gone, which is the difference between
    a stale writer refused and one that lands on top of a committed write.
    """

    def _require_version(self, conn, key, expect):
        return


class PausingCommitBackend(SqliteBackend):
    """A database backend that pauses after the first staged write has executed, so a kill lands
    with real uncommitted rows in the transaction rather than before any of them.

    ``pause_marker`` is written once the first insert has run, which is what lets the parent wait
    until there is something for a rollback to take back before killing this process. With
    ``TCIP_STORE_CONTRACT_COMMITS_EACH_WRITE=1`` each staged write is committed on its own
    instead, which is the shape a kill cannot take back and what the crash case is observed
    failing against.
    """

    def __init__(self, *, pause_marker: Path, pause_s: float, **kwargs) -> None:
        super().__init__(**kwargs)
        self.pause_marker = pause_marker
        self.pause_s = pause_s
        self._paused = False

    def _put(self, conn, key, data):
        super()._put(conn, key, data)
        if os.environ.get("TCIP_STORE_CONTRACT_COMMITS_EACH_WRITE") == "1":
            conn.execute("commit")
            conn.execute("begin immediate")
        if not self._paused:
            self._paused = True
            self.pause_marker.write_text("staged", encoding="utf-8")
            time.sleep(self.pause_s)


def make_backend(**kwargs) -> FileBackend | SqliteBackend:
    """The backend this process writes through, weakened when the environment says so."""
    if os.environ.get(BACKEND_ENV) == SQLITE:
        if os.environ.get("TCIP_STORE_CONTRACT_IGNORES_EXPECT") == "1":
            return IgnoredExpectBackend(**kwargs)
        return SqliteBackend(**kwargs)
    if os.environ.get("TCIP_STORE_CONTRACT_UNLOCKED") == "1":
        return UnlockedWriteBackend(**kwargs)
    return FileBackend(**kwargs)


def wait_for(path: Path, timeout_s: float = 30.0) -> None:
    """Block until another process creates ``path``."""
    deadline = time.monotonic() + timeout_s
    while not path.exists():
        if time.monotonic() > deadline:
            raise TimeoutError(f"{path} never appeared")
        time.sleep(0.01)


def _record(store: str, root: str, *parts: str) -> ts.Key:
    return ts.Key(store, root, tuple(parts))


def _cmd_rewrite(root: str, name: str, rounds: str, padding: str) -> None:
    key = _record(LWW, root, name)
    values = [{"tag": tag, "pad": tag * int(padding)} for tag in ("a", "b")]
    for i in range(int(rounds)):
        ts.replace(key, values[i % 2])


def _cmd_hold_transaction(root: str, store: str, name: str, value: str, hold_s: str, ready: str) -> None:
    key = _record(store, root, name)
    with ts.transaction(key) as txn:
        txn.write(key, {"who": value})
        Path(ready).write_text("held", encoding="utf-8")
        time.sleep(float(hold_s))


def _cmd_write_after(root: str, store: str, name: str, mode: str, value: str, ready: str, go: str, result: str) -> None:
    key = _record(store, root, name)
    expect = ts.read_versioned(key).version if mode == "cas" else None
    Path(ready).write_text("read", encoding="utf-8")
    wait_for(Path(go))
    started = time.monotonic()
    outcome = "written"
    try:
        ts.replace(key, {"who": value}, expect=expect)
    except ts.VersionConflict:
        outcome = "VersionConflict"
    except ts.StoreBusy:
        outcome = "StoreBusy"
    Path(result).write_text(
        json.dumps({"outcome": outcome, "waited_s": time.monotonic() - started}), encoding="utf-8"
    )


def _cmd_increment(root: str, name: str, rounds: str) -> None:
    key = _record(CAS, root, name)
    for _ in range(int(rounds)):
        with ts.transaction(key) as txn:
            current = txn.read(key, default={"n": 0})
            txn.write(key, {"n": current["n"] + 1})


def _cmd_two_keys(root: str, first: str, second: str, rounds: str) -> None:
    keys = (_record(LWW, root, first), _record(LWW, root, second))
    for i in range(int(rounds)):
        with ts.transaction(*keys) as txn:
            for key in keys:
                txn.write(key, {"round": i})


def _cmd_append(root: str, name: str, tag: str, count: str) -> None:
    key = _record(LOG, root, name)
    for i in range(int(count)):
        ts.append(key, {"tag": tag, "i": i})


def _cmd_append_until_killed(root: str, name: str, ready: str) -> None:
    key = _record(LOG, root, name)
    i = 0
    while True:
        ts.append(key, {"i": i})
        if i == 0:
            Path(ready).write_text("appending", encoding="utf-8")
        i += 1


def _cmd_read_record(root: str, store: str, name: str, result: str) -> None:
    value = ts.read(_record(store, root, name), default=None)
    Path(result).write_text(json.dumps({"value": value}), encoding="utf-8")


def _cmd_count_log(root: str, name: str, result: str) -> None:
    page = ts.read_log(_record(LOG, root, name))
    Path(result).write_text(
        json.dumps({"records": len(page.records), "torn_tail": page.torn_tail}), encoding="utf-8"
    )


def _cmd_hold_lock(root: str, name: str, ready: str) -> None:
    key = _record(LWW, root, name)
    with ts.transaction(key) as txn:
        txn.write(key, {"who": "holder"})
        Path(ready).write_text("held", encoding="utf-8")
        time.sleep(300)


def _cmd_write_blob_after(root: str, name: str, mode: str, value: str, ready: str, go: str, result: str) -> None:
    key = _record(BLOB, root, name)
    expect = ts.read_blob_versioned(key, default=b"").version if mode == "cas" else None
    Path(ready).write_text("read", encoding="utf-8")
    wait_for(Path(go))
    outcome = "written"
    try:
        ts.put_blob(key, value.encode("utf-8"), expect=expect)
    except ts.VersionConflict:
        outcome = "VersionConflict"
    except ts.StoreBusy:
        outcome = "StoreBusy"
    Path(result).write_text(json.dumps({"outcome": outcome}), encoding="utf-8")


def _cmd_pause_mid_apply(root: str, first: str, second: str, marker: str, pause_s: str) -> None:
    ts.bind(PausingApplyBackend(pause_marker=Path(marker), pause_s=float(pause_s)))
    keys = (_record(LWW, root, first), _record(LWW, root, second))
    with ts.transaction(*keys) as txn:
        for key in reversed(keys):
            txn.write(key, {"who": key.parts[-1]})


def _cmd_pause_before_commit(root: str, first: str, second: str, marker: str, pause_s: str) -> None:
    """Write both keys through a transaction that pauses once the first row is in it, so a kill
    lands on an open transaction holding uncommitted rows rather than on an empty one."""
    ts.bind(PausingCommitBackend(pause_marker=Path(marker), pause_s=float(pause_s)))
    keys = (_record(LWW, root, first), _record(LWW, root, second))
    with ts.transaction(*keys) as txn:
        for key in keys:
            txn.write(key, {"who": key.parts[-1]})


_COMMANDS = {
    "rewrite": _cmd_rewrite,
    "hold-transaction": _cmd_hold_transaction,
    "write-after": _cmd_write_after,
    "increment": _cmd_increment,
    "two-keys": _cmd_two_keys,
    "append": _cmd_append,
    "append-until-killed": _cmd_append_until_killed,
    "read-record": _cmd_read_record,
    "count-log": _cmd_count_log,
    "hold-lock": _cmd_hold_lock,
    "pause-mid-apply": _cmd_pause_mid_apply,
    "pause-before-commit": _cmd_pause_before_commit,
    "write-blob-after": _cmd_write_blob_after,
}


def main(argv: list[str]) -> int:
    register_contract_stores()
    ts.bind(make_backend())
    command, *args = argv
    _COMMANDS[command](*args)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
