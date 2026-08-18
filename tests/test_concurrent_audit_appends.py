"""Concurrency invariant for the embedded agent terminal (chat-popup-design.md §6):
two tcip-mcp instances (the operator's own terminal agent + the in-app Claude Code
session, each spawning its own MCP server) may append to one audit log concurrently.
The append-only log must not tear or lose lines.

What the store promises is that every append which returned is in the log, once, not that no
append times out under saturation. So each appender here records what returned and reports what
raised, and the log is compared against exactly that record: a refusal is then named as the
refusal it was instead of reading as a lost line.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
from collections import Counter

import pytest

import tcip_store as ts
from tcip_mcp.audit import audit_log_key, record_event


def _generous_lock_timeout_s() -> float:
    """The lock wait these tests bind their backend with.

    Four times the seam's own default, so a shared runner's slow disk cannot starve one
    appender out of its turn, while a real deadlock is still bounded rather than endless. The
    claim under test is completeness, not latency.
    """
    from tcip_store.file_backend import DEFAULT_LOCK_TIMEOUT_S

    return 4 * DEFAULT_LOCK_TIMEOUT_S


@pytest.fixture
def unstarved_backend():
    """Bind this test's backend with the generous lock wait, and close it on the way out."""
    from tcip_store.binding import bind_default

    backend = bind_default(lock_timeout_s=_generous_lock_timeout_s())
    yield backend
    backend.close()


def test_concurrent_threads_no_torn_or_lost_lines(tmp_path, unstarved_backend):
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
                    refusals.append(f"thread {tag} entry {i}: {type(exc).__name__}: {exc}")
                return
            with record:
                appended.append((tag, i))

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not refusals, "an append raised instead of returning: " + "; ".join(refusals)
    assert len(appended) == n_threads * per_thread

    page = ts.read_log(audit_log_key(tmp_path))
    assert page.corrupt == () and page.torn_tail is False
    assert Counter((r["tag"], r["i"]) for r in page.records) == Counter(appended)


_PROCESS_APPENDER = """\
import json, sys
import tcip_store as ts
from tcip_store.binding import bind_default
from tcip_mcp.audit import audit_log_key

root, tag, timeout_s, n = sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4])
bind_default(lock_timeout_s=timeout_s)
returned = []
try:
    for i in range(n):
        ts.append(audit_log_key(root), {'tag': tag, 'i': i})
        returned.append(i)
finally:
    print(json.dumps(returned))
"""


def test_concurrent_processes_no_torn_or_lost_lines(tmp_path, unstarved_backend):
    """Two separate processes (mirroring the terminal + sidecar MCP instances)."""
    per_process = 50
    timeout_s = _generous_lock_timeout_s()
    procs = [
        subprocess.Popen(
            [sys.executable, "-c", _PROCESS_APPENDER, str(tmp_path), f"proc{n}",
             str(timeout_s), str(per_process)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
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
    assert page.corrupt == () and page.torn_tail is False
    assert Counter((r["tag"], r["i"]) for r in page.records) == appended


def test_an_ordinary_event_lands_in_the_log_its_scope_names(tmp_path):
    """The rail admits the plain call: one event, recorded whole, in the named root's log."""
    record_event("gui_save_labels", {"n_annotations": 3}, source="gui", scope=str(tmp_path))

    page = ts.read_log(audit_log_key(tmp_path))
    assert [r["tool"] for r in page.records] == ["gui_save_labels"]
    assert page.records[0]["arguments"] == {"n_annotations": 3}
    assert page.records[0]["source"] == "gui"
