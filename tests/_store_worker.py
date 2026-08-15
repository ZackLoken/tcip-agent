"""Store declarations and the child-process bodies the storage-contract suite drives.

The contract's isolation and durability cases are only real across OS processes: threads in
one interpreter share a lock registry, so a same-process check would pass against a backend
that has no cross-process exclusion at all. This module is both the child script those cases
spawn (``python tests/_store_worker.py <command> ...``) and the parent's source of store
declarations, so both sides address exactly the same stores.

``TCIP_STORE_CONTRACT_UNLOCKED=1`` in the environment swaps in a backend whose writes skip
the key's lock. That variant is what the isolation cases are observed failing against; it is
test scaffolding and no shipped code path can reach it.
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

CAS = "contract_cas"
LWW = "contract_lww"
RELAXED = "contract_relaxed"
NESTED = "contract_nested"
OPAQUE = "contract_opaque"
STRICT = "contract_strict"
LOG = "contract_log"
BLOB = "contract_blob"
SEALED_BLOB = "contract_sealed_blob"

_registered = False


def register_contract_stores() -> None:
    """Declare the stores the contract suite addresses, once per process.

    Each store isolates one declaration the contract makes: the two concurrency policies,
    relaxed durability, a two-part key, a store with no enumeration, a codec that refuses
    some values, a log, and two blob stores that differ only in whether they hand out a
    path.
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
            codec=ts.json_codec(),
            concurrency="cas",
            enumerable=True,
            locator=RootedFileLocator(prefix=("cas",), suffix=".json"),
        )
    )
    ts.register_store(
        ts.StoreDescriptor(
            name=LWW,
            kind="record",
            key_fields=("name",),
            codec=ts.json_codec(),
            concurrency="last_writer_wins",
            enumerable=True,
            locator=RootedFileLocator(prefix=("lww",), suffix=".json"),
        )
    )
    ts.register_store(
        ts.StoreDescriptor(
            name=RELAXED,
            kind="record",
            key_fields=("name",),
            codec=ts.json_codec(),
            concurrency="last_writer_wins",
            durable=False,
            locator=RootedFileLocator(prefix=("relaxed",), suffix=".json"),
        )
    )
    ts.register_store(
        ts.StoreDescriptor(
            name=NESTED,
            kind="record",
            key_fields=("group", "name"),
            codec=ts.json_codec(),
            concurrency="last_writer_wins",
            enumerable=True,
            locator=RootedFileLocator(prefix=("nested",), suffix=".json"),
        )
    )
    ts.register_store(
        ts.StoreDescriptor(
            name=OPAQUE,
            kind="record",
            key_fields=("name",),
            codec=ts.json_codec(),
            concurrency="last_writer_wins",
            locator=RootedFileLocator(prefix=("opaque",), suffix=".json"),
        )
    )
    ts.register_store(
        ts.StoreDescriptor(
            name=STRICT,
            kind="record",
            key_fields=("name",),
            codec=ts.json_codec(allow_nan=False, default=None),
            concurrency="last_writer_wins",
            enumerable=True,
            locator=RootedFileLocator(prefix=("strict",), suffix=".json"),
        )
    )
    ts.register_store(
        ts.StoreDescriptor(
            name=LOG,
            kind="log",
            key_fields=("name",),
            codec=ts.json_codec(indent=None),
            locator=RootedFileLocator(prefix=("logs",), suffix=".jsonl"),
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


def make_backend(**kwargs) -> FileBackend:
    """The backend this process writes through, weakened when the environment says so."""
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


def _cmd_pause_mid_apply(root: str, first: str, second: str, marker: str, pause_s: str) -> None:
    ts.bind(PausingApplyBackend(pause_marker=Path(marker), pause_s=float(pause_s)))
    keys = (_record(LWW, root, first), _record(LWW, root, second))
    with ts.transaction(*keys) as txn:
        for key in reversed(keys):
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
}


def main(argv: list[str]) -> int:
    register_contract_stores()
    ts.bind(make_backend())
    command, *args = argv
    _COMMANDS[command](*args)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
