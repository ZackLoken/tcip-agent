"""The web's in-memory async job registry and its memory cap.

A job carries the project it launched for (``project``). :func:`evict_terminal` holds the whole
registry, every project's jobs together, at :data:`MAX_JOBS`. Nothing here persists: a job lives
for as long as this process does.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

from tcip_mcp.experiments import FINAL_STATES

MAX_JOBS = 100

JOB_STATES = ("pending", "running", *FINAL_STATES)
"""Every status a job holds: ``pending`` and ``running``, then the final state it ends in
(``experiments.FINAL_STATES``)."""


def live(job: Any) -> bool:
    """Whether ``job`` has not yet ended: its status is ``pending`` or ``running``."""
    return job.status not in FINAL_STATES


def evict_terminal(jobs: dict, max_jobs: int = MAX_JOBS) -> None:
    """Drop the oldest jobs of any project that have ended (:func:`live`), in place, until
    ``jobs`` holds at most ``max_jobs`` or every job left is live. Relies on dict insertion order
    (oldest first).
    """
    overflow = len(jobs) - max_jobs
    if overflow > 0:
        evictable = [jid for jid, job in jobs.items() if not live(job)]
        for jid in evictable[:overflow]:
            del jobs[jid]


class JobRegistry:
    """The dict-plus-lock live-job registry: register, find-or-register, get and list."""

    def __init__(self, *, id_field: str = "job_id") -> None:
        """``id_field`` names the attribute :meth:`find_or_register` keys a new job by."""
        self.jobs: dict[str, Any] = {}
        self.lock = threading.Lock()
        self._id_field = id_field

    def _insert(self, job_id: str, job: Any) -> None:
        """Add ``job`` under ``job_id`` and evict overflow (:func:`evict_terminal`); called under
        the lock."""
        self.jobs[job_id] = job
        evict_terminal(self.jobs)

    def register(self, job_id: str, job: Any) -> None:
        """Add ``job`` under ``job_id`` (:meth:`_insert`)."""
        with self.lock:
            self._insert(job_id, job)

    def find_or_register(
        self, match: Callable[[Any], bool], make: Callable[[], Any],
    ) -> tuple[Any, bool]:
        """Under one lock acquisition: the first live job ``match`` accepts, or ``make()``'s new
        job once inserted (:meth:`_insert`). Returns ``(job, created)``, ``created`` true only
        when ``make()`` ran, so the caller knows whether to start work for it.
        """
        with self.lock:
            for job in self.jobs.values():
                if match(job):
                    return job, False
            job = make()
            self._insert(getattr(job, self._id_field), job)
        return job, True

    def get(self, job_id: str) -> Any:
        """A job by id, from any project this process holds."""
        with self.lock:
            return self.jobs.get(job_id)

    def list(self, project: str) -> list[Any]:
        """Every job this process holds that runs for ``project``."""
        with self.lock:
            return [j for j in self.jobs.values() if j.project == project]
