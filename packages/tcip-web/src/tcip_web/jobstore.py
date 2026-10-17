"""The web's in-memory async job registry and its memory cap.

A job carries the project it launched for (``project``). :func:`evict_terminal` holds the whole
registry, every project's jobs together, at :data:`MAX_JOBS`. Nothing here persists: a job lives
for as long as this process does.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any, Literal

from tcip_mcp.experiments import FINAL_STATES, FinalState

MAX_JOBS = 100

JobStatus = Literal["pending", "running", FinalState]
"""Every status a job holds: ``pending`` and ``running``, then the final state it ends in
(``experiments.FinalState``)."""


def live(job: Any) -> bool:
    """Return whether the job has not yet ended.

    A job that has not ended has status ``pending`` or ``running``.
    """
    return job.status not in FINAL_STATES


def evict_terminal(jobs: dict, max_jobs: int = MAX_JOBS) -> None:
    """Drop the oldest ended jobs of any project from ``jobs``, in place.

    Ended means not :func:`live`. Drops until ``jobs`` holds at most ``max_jobs`` or every job
    left is live. Relies on dict insertion order (oldest first).
    """
    overflow = len(jobs) - max_jobs
    if overflow > 0:
        evictable = [jid for jid, job in jobs.items() if not live(job)]
        for jid in evictable[:overflow]:
            del jobs[jid]


class JobRegistry:
    """The dict-plus-lock live-job registry: register, find-or-register, get and list."""

    def __init__(self, *, id_field: str = "job_id") -> None:
        """Initialize an empty registry keyed by ``id_field``."""
        self.jobs: dict[str, Any] = {}
        self.lock = threading.Lock()
        self._id_field = id_field

    def _insert(self, job_id: str, job: Any) -> None:
        """Add ``job`` under ``job_id`` and evict overflow (:func:`evict_terminal`).

        Called under the lock.
        """
        self.jobs[job_id] = job
        evict_terminal(self.jobs)

    def register(self, job_id: str, job: Any) -> None:
        """Add ``job`` under ``job_id`` (:meth:`_insert`)."""
        with self.lock:
            self._insert(job_id, job)

    def find_or_register(
        self,
        match: Callable[[Any], bool],
        make: Callable[[], Any],
    ) -> tuple[Any, bool]:
        """Return the first job ``match`` accepts, or insert and return ``make()``'s new job.

        Both happen under one lock acquisition; the new job is inserted through
        :meth:`_insert`. Returns ``(job, created)``, ``created`` true only when ``make()`` ran.
        """
        with self.lock:
            for job in self.jobs.values():
                if match(job):
                    return job, False
            job = make()
            self._insert(getattr(job, self._id_field), job)
        return job, True

    def get(self, job_id: str) -> Any:
        """Return a job by id, from any project this process holds."""
        with self.lock:
            return self.jobs.get(job_id)

    def list(self, project: str) -> list[Any]:
        """Return every held job that runs for ``project``."""
        with self.lock:
            return [j for j in self.jobs.values() if j.project == project]
