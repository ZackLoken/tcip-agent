"""The web's in-memory async job registry and its memory cap.

A job carries the root it launched under (``platform_root``). :func:`evict_terminal` evicts the
oldest terminal jobs, first within one root's share, then across every root this process holds.
Nothing here persists: a job lives for as long as this process does.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

from tcip_mcp.experiments import FINAL_STATES, TERMINAL_STATES

MAX_JOBS = 100

JOB_STATES = ("pending", "running", *FINAL_STATES)
"""Every status a job holds: ``pending`` and ``running``, then the final state it ends in
(``experiments.FINAL_STATES``)."""


def evict_terminal(jobs: dict, root: str | None, max_jobs: int = MAX_JOBS) -> None:
    """Drop the oldest terminal jobs, in place, once ``jobs`` overflows ``max_jobs``: once for
    ``root``'s own share, then once for the whole dict across every root this process holds.

    ``root`` scopes the first pass: a job's own ``platform_root`` for a per-root registry, or
    ``None`` for a registry with no root concept of its own, matching every entry, so the two
    passes coincide.

    The second pass evicts the oldest terminal job in the whole dict regardless of which root it
    belongs to, so a root whose own share never overflowed can still lose an entry.

    Relies on dict insertion order (oldest first); running/pending jobs are never evicted.
    """
    scoped = [jid for jid, job in jobs.items() if getattr(job, "platform_root", None) == root]
    overflow = len(scoped) - max_jobs
    if overflow > 0:
        evictable = [jid for jid in scoped if getattr(jobs[jid], "status", "") in TERMINAL_STATES]
        for jid in evictable[:overflow]:
            jobs.pop(jid, None)

    total_overflow = len(jobs) - max_jobs
    if total_overflow > 0:
        evictable_any = [
            jid for jid, job in jobs.items() if getattr(job, "status", "") in TERMINAL_STATES
        ]
        for jid in evictable_any[:total_overflow]:
            jobs.pop(jid, None)


class JobRegistry:
    """The dict-plus-lock live-job registry: register, find-or-register, get and list."""

    def __init__(self, *, id_field: str = "job_id") -> None:
        """``id_field`` names the attribute :meth:`find_or_register` keys a new job by."""
        self.jobs: dict[str, Any] = {}
        self.lock = threading.Lock()
        self._id_field = id_field

    def register(self, job_id: str, job: Any, *, job_root: str | None) -> None:
        """Add ``job`` under ``job_id`` and evict overflow (:func:`evict_terminal`'s own two
        passes). ``job_root`` is the job's own ``platform_root`` for a per-root registry, or
        ``None`` for one with no root concept."""
        with self.lock:
            self.jobs[job_id] = job
            evict_terminal(self.jobs, job_root)

    def find_or_register(
        self, match: Callable[[Any], bool], make: Callable[[], Any], *, job_root: str | None = None,
    ) -> tuple[Any, bool]:
        """Under one lock acquisition: the first live job ``match`` accepts, or ``make()``'s new
        job once inserted and evicted the way :meth:`register` does. Returns ``(job, created)``,
        ``created`` true only when ``make()`` ran, so the caller knows whether to start work for
        it.
        """
        with self.lock:
            for job in self.jobs.values():
                if match(job):
                    return job, False
            job = make()
            self.jobs[getattr(job, self._id_field)] = job
            evict_terminal(self.jobs, job_root)
        return job, True

    def get(self, job_id: str) -> Any:
        """A job by id, from any root this process holds."""
        with self.lock:
            return self.jobs.get(job_id)

    def list(self, root: str | None = None) -> list[Any]:
        """Every live job, or only ``root``'s own share when given."""
        with self.lock:
            if root is None:
                return list(self.jobs.values())
            return [j for j in self.jobs.values() if getattr(j, "platform_root", None) == root]
