"""Store names and the child-process bodies the storage-contract suite drives.

The contract's isolation and durability cases are only real across OS processes: threads in
one interpreter share a lock registry, so a same-process check would pass against a store that
has no cross-process exclusion at all. This module is both the child script those cases spawn
(``python tests/_store_worker.py <command> ...``) and the parent's source of the store names, so
both sides address exactly the same records.

Two weakened variants exist, each selected by its own environment variable, each what a set of
cases is observed failing against, and neither reachable from any shipped code path:
``TCIP_STORE_CONTRACT_IGNORES_EXPECT=1`` swaps in a backend that never compares ``expect``, and
``TCIP_STORE_CONTRACT_COMMITS_EACH_WRITE=1`` makes :class:`PausingCommitBackend` commit each write
of a transaction on its own.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import tcip_store as ts
from tcip_store.sqlite_backend import SqliteBackend

CAS = "contract_cas"
LWW = "contract_lww"
NESTED = "contract_nested"
STRICT = "contract_strict"
LOG = "contract_log"


def blob_path(root: str | Path, name: str) -> Path:
    """The file the suite's blob cases write ``name`` to under ``root``."""
    return Path(root, "blobs", f"{name}.bin")


class IgnoredExpectBackend(SqliteBackend):
    """A backend whose writes never compare ``expect``, for observing what it holds up.

    Everything else is the shipped path: the same transaction, the same rows, the same version
    derived from the stored bytes. Only the comparison is gone, which is the difference between
    a stale writer refused and one that lands on top of a committed write.
    """

    def replace(self, key, value, *, expect=None):
        return super().replace(key, value)

    def delete(self, key, *, expect=None):
        super().delete(key)


class PausingCommitBackend(SqliteBackend):
    """A backend that pauses after the first write of a transaction has executed, so a kill lands
    with real uncommitted rows in the transaction rather than before any of them.

    ``pause_marker`` is written once the first write has run, which is what lets the parent wait
    until there is something for a rollback to take back before killing this process. With
    ``TCIP_STORE_CONTRACT_COMMITS_EACH_WRITE=1`` each write is committed on its own instead, which
    is the shape a kill cannot take back and what the crash case is observed failing against.
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


def make_backend(**kwargs) -> SqliteBackend:
    """The backend this process writes through, weakened when the environment says so."""
    if os.environ.get("TCIP_STORE_CONTRACT_IGNORES_EXPECT") == "1":
        return IgnoredExpectBackend(**kwargs)
    return SqliteBackend(**kwargs)


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


def _cmd_hold_transaction(
    root: str, store: str, name: str, value: str, hold_s: str, ready: str
) -> None:
    key = _record(store, root, name)
    with ts.transaction(key) as txn:
        txn.write(key, {"who": value})
        Path(ready).write_text("held", encoding="utf-8")
        time.sleep(float(hold_s))


def _cmd_write_after(
    root: str, store: str, name: str, mode: str, value: str, ready: str, go: str, result: str
) -> None:
    key = _record(store, root, name)
    expect = ts.read_versioned(key).version if mode == "cas" else None
    Path(ready).write_text("read", encoding="utf-8")
    wait_for(Path(go))
    started = time.monotonic()
    outcome = "written"
    try:
        ts.replace(key, {"who": value}, expect=expect)
    except ts.VersionConflictError:
        outcome = "VersionConflictError"
    except ts.StoreBusyError:
        outcome = "StoreBusyError"
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
    Path(result).write_text(json.dumps({"records": len(page.records)}), encoding="utf-8")


def _cmd_hold_lock(root: str, name: str, ready: str) -> None:
    key = _record(LWW, root, name)
    with ts.transaction(key) as txn:
        txn.write(key, {"who": "holder"})
        Path(ready).write_text("held", encoding="utf-8")
        time.sleep(300)


def _cmd_write_blob_after(
    root: str, name: str, mode: str, value: str, ready: str, go: str, result: str
) -> None:
    path = blob_path(root, name)
    expect = ts.read_blob_versioned(path, default=b"").version if mode == "cas" else None
    Path(ready).write_text("read", encoding="utf-8")
    wait_for(Path(go))
    outcome = "written"
    try:
        ts.put_blob(path, value.encode("utf-8"), expect=expect)
    except ts.VersionConflictError:
        outcome = "VersionConflictError"
    except ts.StoreBusyError:
        outcome = "StoreBusyError"
    Path(result).write_text(json.dumps({"outcome": outcome}), encoding="utf-8")


def _cmd_pause_before_commit(root: str, first: str, second: str, marker: str, pause_s: str) -> None:
    """Write both keys through a transaction that pauses once the first row is in it, so a kill
    lands on an open transaction holding uncommitted rows rather than on an empty one."""
    ts.bind(PausingCommitBackend(pause_marker=Path(marker), pause_s=float(pause_s)))
    keys = (_record(LWW, root, first), _record(LWW, root, second))
    with ts.transaction(*keys) as txn:
        for key in keys:
            txn.write(key, {"who": "crashed"})


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
    "pause-before-commit": _cmd_pause_before_commit,
    "write-blob-after": _cmd_write_blob_after,
}


def main(argv: list[str]) -> int:
    ts.bind(make_backend())
    command, *args = argv
    _COMMANDS[command](*args)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
