"""Store names and the storage-contract cases' child-process commands.

Run as ``python tests/_store_worker.py <command> ...`` it executes one command against a store;
imported, it provides the store names its commands write under.

Two weakened variants are each selected by its own environment variable:
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
from typing import TYPE_CHECKING

import tcip_store as ts
from tcip_store.sqlite_backend import SqliteBackend

if TYPE_CHECKING:
    import sqlite3

    from pydantic import JsonValue

CAS = "contract_cas"
LWW = "contract_lww"
NESTED = "contract_nested"
STRICT = "contract_strict"
LOG = "contract_log"


def blob_path(root: str | Path, name: str) -> Path:
    """Return ``<root>/blobs/<name>.bin``."""
    return Path(root, "blobs", f"{name}.bin")


class IgnoredExpectBackend(SqliteBackend):
    """A ``SqliteBackend`` whose ``replace`` and ``delete`` never compare ``expect``.

    Its other behavior is ``SqliteBackend``'s.
    """

    def replace(
        self,
        key: ts.Key,
        value: JsonValue,
        *,
        expect: ts.Version | None = None,
    ) -> ts.Version:
        return super().replace(key, value)

    def delete(self, key: ts.Key, *, expect: ts.Version | None = None) -> bool:
        return super().delete(key)


class PausingCommitBackend(SqliteBackend):
    """A ``SqliteBackend`` that pauses after its first write.

    After its first ``_put`` it writes ``pause_marker`` and sleeps ``pause_s`` seconds. With
    ``TCIP_STORE_CONTRACT_COMMITS_EACH_WRITE=1`` each ``_put`` is followed by a commit and a new
    ``begin immediate``.
    """

    def __init__(
        self, *, pause_marker: Path, pause_s: float, **kwargs: float
    ) -> None:
        super().__init__(**kwargs)
        self.pause_marker = pause_marker
        self.pause_s = pause_s
        self._paused = False

    def _put(self, conn: sqlite3.Connection, key: ts.Key, data: bytes) -> None:
        super()._put(conn, key, data)
        if os.environ.get("TCIP_STORE_CONTRACT_COMMITS_EACH_WRITE") == "1":
            conn.execute("commit")
            conn.execute("begin immediate")
        if not self._paused:
            self._paused = True
            self.pause_marker.write_text("staged", encoding="utf-8")
            time.sleep(self.pause_s)


def make_backend(**kwargs: float) -> SqliteBackend:
    """Return an ``IgnoredExpectBackend`` when the environment asks, else a ``SqliteBackend``.

    The environment asks with ``TCIP_STORE_CONTRACT_IGNORES_EXPECT=1``; ``kwargs`` are passed on.
    """
    if os.environ.get("TCIP_STORE_CONTRACT_IGNORES_EXPECT") == "1":
        return IgnoredExpectBackend(**kwargs)
    return SqliteBackend(**kwargs)


def wait_for(path: Path, timeout_s: float = 30.0) -> None:
    """Block until ``path`` exists, raising ``TimeoutError`` after ``timeout_s`` seconds."""
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
    root: str,
    store: str,
    name: str,
    mode: str,
    value: str,
    ready: str,
    go: str,
    result: str,
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
        json.dumps(
            {"outcome": outcome, "waited_s": time.monotonic() - started}
        ),
        encoding="utf-8",
    )


def _cmd_increment(root: str, name: str, rounds: str) -> None:
    key = _record(CAS, root, name)
    for _ in range(int(rounds)):
        with ts.transaction(key) as txn:
            current = txn.read(key, default={"n": 0})
            assert isinstance(current, dict)
            n = current["n"]
            assert isinstance(n, int)
            txn.write(key, {"n": n + 1})


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
        json.dumps({"records": len(page.records)}), encoding="utf-8"
    )


def _cmd_hold_lock(root: str, name: str, ready: str) -> None:
    key = _record(LWW, root, name)
    with ts.transaction(key) as txn:
        txn.write(key, {"who": "holder"})
        Path(ready).write_text("held", encoding="utf-8")
        time.sleep(300)


def _cmd_write_blob_after(
    root: str,
    name: str,
    mode: str,
    value: str,
    ready: str,
    go: str,
    result: str,
) -> None:
    path = blob_path(root, name)
    expect = (
        ts.read_blob_versioned(path, default=b"").version
        if mode == "cas"
        else None
    )
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


def _cmd_pause_before_commit(
    root: str, first: str, second: str, marker: str, pause_s: str
) -> None:
    """Bind a :class:`PausingCommitBackend` and write both keys in one transaction."""
    ts.bind(
        PausingCommitBackend(pause_marker=Path(marker), pause_s=float(pause_s))
    )
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
