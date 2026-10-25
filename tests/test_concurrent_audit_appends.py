"""Concurrent appenders to one audit log, in threads and in separate processes.

Every append returns, and each is in the log exactly once.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
from collections import Counter
from typing import TYPE_CHECKING

import pytest

import tcip_store as ts
from tcip_mcp.audit import audit_log_key, record_event_or_raise
from tcip_store.sqlite_backend import SqliteBackend
from tests._audit_fixtures import audit_rows

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


def _generous_lock_timeout_s() -> float:
    """Return four times the store's default lock timeout, in seconds."""
    from tcip_store.file_backend import DEFAULT_LOCK_TIMEOUT_S

    return 4 * DEFAULT_LOCK_TIMEOUT_S


@pytest.fixture
def unstarved_backend() -> Iterator[SqliteBackend]:
    """Bind a sqlite backend at four times the default lock timeout, yield it, then close it."""
    backend = ts.bind(SqliteBackend(lock_timeout_s=_generous_lock_timeout_s()))
    yield backend
    backend.close()


def test_concurrent_threads_no_torn_or_lost_lines(
    tmp_path: Path, unstarved_backend: SqliteBackend
) -> None:
    n_threads, per_thread = 16, 40
    appended: list[tuple[int, int]] = []
    refusals: list[str] = []
    record = threading.Lock()

    def worker(tag: int) -> None:
        for i in range(per_thread):
            try:
                ts.append(audit_log_key(tmp_path), {"tag": tag, "i": i})
            except Exception as exc:
                with record:
                    refusals.append(
                        f"thread {tag} entry {i}: {type(exc).__name__}: {exc}"
                    )
                return
            with record:
                appended.append((tag, i))

    threads = [
        threading.Thread(target=worker, args=(t,)) for t in range(n_threads)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not refusals, "an append raised instead of returning: " + "; ".join(
        refusals
    )
    assert len(appended) == n_threads * per_thread

    page = ts.read_log(audit_log_key(tmp_path))
    assert page.corrupt == ()
    assert Counter((r["tag"], r["i"]) for r in page.records) == Counter(
        appended
    )


_PROCESS_APPENDER = """\
import json, sys
import tcip_store as ts
from tcip_store.sqlite_backend import SqliteBackend
from tcip_mcp.audit import audit_log_key

root, tag, timeout_s, n = sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4])
ts.bind(SqliteBackend(lock_timeout_s=timeout_s))
returned = []
try:
    for i in range(n):
        ts.append(audit_log_key(root), {'tag': tag, 'i': i})
        returned.append(i)
finally:
    print(json.dumps(returned))
"""


def test_concurrent_processes_no_torn_or_lost_lines(
    tmp_path: Path, unstarved_backend: SqliteBackend
) -> None:
    """Appends from two separate processes all land, none torn or lost.

    Each process returns cleanly, the log holds no corrupt line, and its records are exactly the
    appends both processes report.
    """
    per_process = 50
    timeout_s = _generous_lock_timeout_s()
    procs = [
        subprocess.Popen(
            [
                sys.executable,
                "-c",
                _PROCESS_APPENDER,
                str(tmp_path),
                f"proc{n}",
                str(timeout_s),
                str(per_process),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for n in range(2)
    ]
    appended: Counter[tuple[str, int]] = Counter()
    for n, proc in enumerate(procs):
        out, err = proc.communicate(timeout=timeout_s)
        assert proc.returncode == 0, (
            f"appender proc{n} raised instead of returning: {err.strip()}"
        )
        appended.update((f"proc{n}", i) for i in json.loads(out))
    assert sum(appended.values()) == 2 * per_process

    page = ts.read_log(audit_log_key(tmp_path))
    assert page.corrupt == ()
    assert Counter((r["tag"], r["i"]) for r in page.records) == appended


def test_an_ordinary_event_lands_in_the_log_its_scope_names(
    tmp_path: Path,
) -> None:
    """An ordinary event lands as one whole record in the log of the root its scope names."""
    record_event_or_raise(
        "save_label_document",
        {"n_annotations": 3},
        actor="user:breeder",
        scope=str(tmp_path),
    )

    rows = audit_rows(tmp_path)
    assert [r["tool"] for r in rows] == ["save_label_document"]
    assert rows[0]["arguments"] == {"n_annotations": 3}
    assert rows[0]["actor"] == "user:breeder"
