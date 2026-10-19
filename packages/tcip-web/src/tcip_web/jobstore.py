"""The web's in-memory async job registry and its memory cap.

A job carries the project it launched for (``project``). :func:`evict_terminal` holds the whole
registry, every project's jobs together, at :data:`MAX_JOBS`. Nothing here persists: a job lives
for as long as this process does.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Literal, Protocol

from tcip_mcp.experiments import FINAL_STATES, FinalState

MAX_JOBS = 100

JobStatus = Literal["pending", "running", FinalState]
"""Every status a job holds: ``pending`` and ``running``, then the final state it ends in
(``experiments.FinalState``)."""


class Job(Protocol):
    """A live job as the registry reads it: its id and the status it holds."""

    job_id: str
    status: JobStatus


class ProjectJob(Job, Protocol):
    """A job that runs for one project."""

    project: str


def live(job: Job) -> bool:
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


class JobRegistry[J: Job]:
    """The dict-plus-lock live-job registry: register, find-or-register, get and list."""

    def __init__(self) -> None:
        """Initialize an empty registry keyed by each job's ``job_id``."""
        self.jobs: dict[str, J] = {}
        self.lock = threading.Lock()

    def _insert(self, job: J) -> None:
        """Add ``job`` under its ``job_id`` and evict overflow (:func:`evict_terminal`).

        Called under the lock.
        """
        self.jobs[job.job_id] = job
        evict_terminal(self.jobs)

    def register(self, job: J) -> None:
        """Add ``job`` under its ``job_id`` (:meth:`_insert`)."""
        with self.lock:
            self._insert(job)

    def find_or_register(
        self,
        match: Callable[[J], bool],
        make: Callable[[], J],
    ) -> tuple[J, bool]:
        """Return the first job ``match`` accepts, or insert and return ``make()``'s new job.

        Both happen under one lock acquisition; the new job is inserted through
        :meth:`_insert`. Returns ``(job, created)``, ``created`` true only when ``make()`` ran.
        """
        with self.lock:
            for job in self.jobs.values():
                if match(job):
                    return job, False
            job = make()
            self._insert(job)
        return job, True

    def get(self, job_id: str) -> J | None:
        """Return a job by id, from any project this process holds."""
        with self.lock:
            return self.jobs.get(job_id)

    def list[P: ProjectJob](self: JobRegistry[P], project: str) -> list[P]:
        """Return every held job that runs for ``project``."""
        with self.lock:
            return [j for j in self.jobs.values() if j.project == project]
