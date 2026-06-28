"""Atomic, lock-guarded filesystem writes for ``.tcip/`` state.

Generalizes the ``tempfile.mkstemp`` + ``os.replace`` pattern already used in
``tcip_annotation.label_io``. ``os.replace`` is atomic on POSIX and Windows, so a
reader never observes a half-written file (no more corrupt JSON on crash / torn read).

For read-modify-write sequences (lineage / artifacts / registry index) wrap the whole
read→modify→write in :func:`file_transaction` to also prevent **lost updates**:

  - within a process: a per-path ``threading.RLock`` serializes threads (parallel HPO
    trials, web request handlers);
  - across processes: a ``filelock.FileLock`` advisory lock serializes the MCP server
    and the web backend. If ``filelock`` is unavailable the cross-process layer is
    skipped and only in-process safety is provided.

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
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

try:  # filelock ships with torch; treated as optional for cross-process locking.
    from filelock import FileLock

    _HAS_FILELOCK = True
except Exception:  # pragma: no cover - exercised only in stripped environments
    _HAS_FILELOCK = False

# Per-path in-process locks (reentrant so accidental re-entry can't self-deadlock).
_locks_guard = threading.Lock()
_path_locks: dict[str, threading.RLock] = {}

_LOCK_TIMEOUT_S = 30.0


def _lock_for(path: Path) -> threading.RLock:
    key = str(path.resolve())
    with _locks_guard:
        lk = _path_locks.get(key)
        if lk is None:
            lk = threading.RLock()
            _path_locks[key] = lk
        return lk


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
    """Serialize a read-modify-write on ``path`` across threads and processes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    in_proc = _lock_for(path)
    in_proc.acquire()
    cross = None
    try:
        if _HAS_FILELOCK:
            cross = FileLock(str(path) + ".lock")
            cross.acquire(timeout=_LOCK_TIMEOUT_S)
        yield
    finally:
        if cross is not None:
            try:
                cross.release()
            except Exception:  # pragma: no cover
                pass
        in_proc.release()


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
