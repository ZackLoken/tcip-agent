"""Persistence + memory-cap helpers for the web's async job registries.

The inference/HPO routes keep an in-memory dict of background jobs that previously grew
unbounded (a memory leak) and vanished on restart. These helpers atomically persist job
summaries to ``.tcip/state/<name>.json`` and evict the oldest *terminal* jobs so the live
registry stays bounded. (Running jobs can't survive a restart, their threads are gone,
so the persisted file is a record, not a resumable state.)
"""

from __future__ import annotations

from pathlib import Path

from tcip_mcp.project_paths import resolve_state
from tcip_mcp.utils.atomic_io import atomic_write_json, read_json


def _state_path(name: str) -> Path:
    # Resolved at use time against the pinned platform root: a bare CWD-relative path would
    # scatter job records by launch dir (and let tests pollute the repo's real .tcip/).
    return resolve_state(Path(".tcip") / "state" / f"{name}.json")


MAX_JOBS = 100
TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled", "interrupted"})


def persist(name: str, summaries: list[dict]) -> None:
    """Atomically write job summaries to ``.tcip/state/<name>.json`` (best-effort)."""
    try:
        atomic_write_json(_state_path(name), summaries)
    except Exception:  # pragma: no cover - persistence is best-effort
        pass


def load(name: str) -> list[dict]:
    """Read persisted job summaries from ``.tcip/state/<name>.json``.

    Returns ``[]`` when the file is missing/unparseable or doesn't hold a list, a
    restart with no prior state (or a corrupt file) starts clean rather than raising.
    """
    data = read_json(_state_path(name), default=[])
    return data if isinstance(data, list) else []


def evict_terminal(jobs: dict, max_jobs: int = MAX_JOBS) -> None:
    """Drop the oldest terminal jobs in place once the registry exceeds ``max_jobs``.

    Relies on dict insertion order (oldest first); running/pending jobs are never evicted.
    """
    overflow = len(jobs) - max_jobs
    if overflow <= 0:
        return
    evictable = [
        jid for jid, job in jobs.items() if getattr(job, "status", "") in TERMINAL_STATUSES
    ]
    for jid in evictable[:overflow]:
        jobs.pop(jid, None)
