"""Atomic, lock-guarded filesystem writes for ``.tcip/`` state.

Generalizes the ``tempfile.mkstemp`` + ``os.replace`` pattern already used in
``tcip_annotation.json_io``. ``os.replace`` is atomic on POSIX and Windows, so a
reader never observes a half-written file (no more corrupt JSON on crash / torn read).

For read-modify-write sequences (lineage / artifacts / registry index) wrap the whole
read→modify→write in :func:`file_transaction` to also prevent lost updates. It holds the
storage layer's own lock pair for the path (``tcip_store.file_backend.path_lock``): a
per-path ``threading.RLock`` for the threads of one process (parallel HPO trials, web
request handlers) and one ``filelock.FileLock`` instance for the MCP server, the web
backend and the training subprocess. Going through that registry rather than building a
second ``FileLock`` is what keeps a path guarded here and a record written through the
storage layer in one lock domain instead of two that ignore each other.

Example::

    with file_transaction(path):
        data = read_json(path, default={})
        data["k"] = v
        atomic_write_json(path, data)
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from tcip_store.file_backend import path_lock

_LOCK_TIMEOUT_S = 30.0


def atomic_write_bytes(path: str | os.PathLike, data: bytes) -> None:
    """Write ``data`` to ``path`` atomically (temp file in the same dir + os.replace)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        _replace_with_retry(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def _replace_with_retry(src: str, dst: str | os.PathLike, *, attempts: int = 5, delay: float = 0.05) -> None:
    """``os.replace`` with a short retry on transient ``PermissionError``.

    On Windows a virus scanner / search indexer can momentarily hold the destination,
    making ``os.replace`` raise ``PermissionError`` even though nothing in this process
    has it open. On POSIX this never triggers, so it's a no-op there.
    """
    import time

    for attempt in range(attempts):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(delay)


def atomic_write_text(path: str | os.PathLike, text: str, *, encoding: str = "utf-8") -> None:
    """Write ``text`` to ``path`` atomically."""
    atomic_write_bytes(path, text.encode(encoding))


def atomic_write_json(path: str | os.PathLike, obj: Any, *, indent: int = 2) -> None:
    """Serialize ``obj`` to JSON and write it to ``path`` atomically."""
    atomic_write_text(path, json.dumps(obj, indent=indent, default=str))


def read_json(path: str | os.PathLike, default: Any = None) -> Any:
    """Read JSON from ``path``; return ``default`` if missing or unparseable."""
    path = Path(path)
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return default


@contextmanager
def file_transaction(path: str | os.PathLike) -> Iterator[None]:
    """Serialize a read-modify-write on ``path`` across threads and processes.

    The lock comes from the storage layer's per-path registry, so a record written through
    that layer while this transaction is open waits for it instead of taking a second lock
    on the same file and blocking until the timeout.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path_lock(path, timeout_s=_LOCK_TIMEOUT_S):
        yield


def append_jsonl(path: str | os.PathLike, obj: Any) -> None:
    """Append one JSON line under the transaction lock with an fsync.

    Append-only logs (``audit.jsonl``, ``metrics.jsonl``) can't use os.replace, so this
    serializes concurrent appenders and flushes to disk to avoid interleaved/lost lines.
    """
    path = Path(path)
    line = json.dumps(obj, default=str) + "\n"
    with file_transaction(path):
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())
