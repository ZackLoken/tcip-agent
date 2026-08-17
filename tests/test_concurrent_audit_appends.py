"""Concurrency invariant for the embedded agent terminal (chat-popup-design.md §6):
two tcip-mcp instances (the operator's own terminal agent + the in-app Claude Code
session, each spawning its own MCP server) may append to one audit log concurrently.
The append-only log must not tear or lose lines.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
from collections import Counter

import tcip_store as ts
from tcip_mcp.audit import audit_log_key, record_event


def test_concurrent_threads_no_torn_or_lost_lines(tmp_path):
    n_threads, per_thread = 16, 40

    def worker(tag: int) -> None:
        for i in range(per_thread):
            ts.append(audit_log_key(tmp_path), {"tag": tag, "i": i})

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    lines = (tmp_path / ".tcip" / "audit.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == n_threads * per_thread
    parsed = [json.loads(ln) for ln in lines]  # every line is intact JSON
    counts = Counter(p["tag"] for p in parsed)
    assert all(counts[t] == per_thread for t in range(n_threads))


def test_concurrent_processes_no_torn_or_lost_lines(tmp_path):
    """Two separate processes (mirroring the terminal + sidecar MCP instances)."""
    worker = (
        "import sys; import tcip_store as ts; "
        "from tcip_store.binding import bind_default; "
        "from tcip_mcp.audit import audit_log_key; "
        "bind_default(); "
        "root, tag = sys.argv[1], sys.argv[2]; "
        "[ts.append(audit_log_key(root), {'tag': tag, 'i': i}) for i in range(50)]"
    )
    procs = [
        subprocess.Popen([sys.executable, "-c", worker, str(tmp_path), f"proc{n}"])
        for n in range(2)
    ]
    for p in procs:
        assert p.wait(timeout=60) == 0

    lines = (tmp_path / ".tcip" / "audit.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 100
    parsed = [json.loads(ln) for ln in lines]
    counts = Counter(p["tag"] for p in parsed)
    assert counts["proc0"] == 50 and counts["proc1"] == 50


def test_an_ordinary_event_lands_in_the_log_its_scope_names(tmp_path):
    """The rail admits the plain call: one event, recorded whole, in the named root's log."""
    record_event("gui_save_labels", {"n_annotations": 3}, source="gui", scope=str(tmp_path))

    page = ts.read_log(audit_log_key(tmp_path))
    assert [r["tool"] for r in page.records] == ["gui_save_labels"]
    assert page.records[0]["arguments"] == {"n_annotations": 3}
    assert page.records[0]["source"] == "gui"
