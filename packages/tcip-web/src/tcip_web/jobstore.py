"""Persistence + memory-cap helpers for the web's async job registries.

The inference/HPO routes keep an in-memory dict of background jobs that previously grew
unbounded (a memory leak) and vanished on restart. These helpers atomically persist job
summaries to ``.tcip/state/<name>.json`` and evict the oldest *terminal* jobs so the live
registry stays bounded. (Running jobs can't survive a restart — their threads are gone —
so the persisted file is a record, not a resumable state.)
"""

from __future__ import annotations

from pathlib import Path

from tcip_mcp.utils.atomic_io import atomic_write_json

_STATE_DIR = Path(".tcip") / "state"
MAX_JOBS = 100
_TERMINAL = {"completed", "failed", "cancelled", "interrupted"}


def persist(name: str, summaries: list[dict]) -> None:
    """Atomically write job summaries to ``.tcip/state/<name>.json`` (best-effort)."""
    try:
        atomic_write_json(_STATE_DIR / f"{name}.json", summaries)
    except Exception:  # pragma: no cover - persistence is best-effort
        pass


def evict_terminal(jobs: dict, max_jobs: int = MAX_JOBS) -> None:
    """Drop the oldest terminal jobs in place once the registry exceeds ``max_jobs``.

    Relies on dict insertion order (oldest first); running/pending jobs are never evicted.
    """
    overflow = len(jobs) - max_jobs
    if overflow <= 0:
        return
    evictable = [jid for jid, job in jobs.items() if getattr(job, "status", "") in _TERMINAL]
    for jid in evictable[:overflow]:
        jobs.pop(jid, None)
