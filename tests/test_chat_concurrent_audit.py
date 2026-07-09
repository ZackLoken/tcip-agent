"""Concurrency invariant for the chat feature (chat-popup-design.md §6): two tcip-mcp
instances (the terminal agent + the chat sidecar) may append to ``.tcip/audit.jsonl``
concurrently. The append-only JSONL primitive must not tear or lose lines.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
from collections import Counter

from tcip_mcp.utils.atomic_io import append_jsonl


def test_concurrent_threads_no_torn_or_lost_lines(tmp_path):
    path = tmp_path / "audit.jsonl"
    n_threads, per_thread = 16, 40

    def worker(tag: int) -> None:
        for i in range(per_thread):
            append_jsonl(path, {"tag": tag, "i": i})

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == n_threads * per_thread
    parsed = [json.loads(ln) for ln in lines]  # every line is intact JSON
    counts = Counter(p["tag"] for p in parsed)
    assert all(counts[t] == per_thread for t in range(n_threads))


def test_concurrent_processes_no_torn_or_lost_lines(tmp_path):
    """Two separate processes (mirroring the terminal + sidecar MCP instances)."""
    path = tmp_path / "audit.jsonl"
    worker = (
        "import sys; from tcip_mcp.utils.atomic_io import append_jsonl; "
        "p, tag = sys.argv[1], sys.argv[2]; "
        "[append_jsonl(p, {'tag': tag, 'i': i}) for i in range(50)]"
    )
    procs = [
        subprocess.Popen([sys.executable, "-c", worker, str(path), f"proc{n}"]) for n in range(2)
    ]
    for p in procs:
        assert p.wait(timeout=60) == 0

    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 100
    parsed = [json.loads(ln) for ln in lines]
    counts = Counter(p["tag"] for p in parsed)
    assert counts["proc0"] == 50 and counts["proc1"] == 50
