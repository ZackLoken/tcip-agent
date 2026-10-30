"""The web's in-memory async job registry and its memory cap.

A job carries the project it launched for (``project``). :func:`evict_terminal` holds the whole
registry, every project's jobs together, at :data:`MAX_JOBS`. Nothing here persists: a job lives
for as long as this process does.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path
from typing import ClassVar, Literal, Self

from pydantic import BaseModel, PrivateAttr, computed_field

from tcip_mcp.dataset_layout import parse_capture_dir
from tcip_mcp.experiments import FINAL_STATES, FinalState

MAX_JOBS = 100

JobStatus = Literal["pending", "running", FinalState]
"""Every status a job holds: ``pending`` and ``running``, then the final state it ends in
(``experiments.FinalState``)."""


class Job(BaseModel):
    """A job as the registry holds it: its id, the status it holds and the error it ended on.

    The thread it runs on is private state, absent from every dump of it.
    """

    job_id: str
    status: JobStatus = "pending"
    error: str | None = None
    _thread: threading.Thread | None = PrivateAttr(default=None)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def live(self) -> bool:
        """Whether the job has not yet ended: its status is ``pending`` or ``running``."""
        return self.status not in FINAL_STATES

    @property
    def thread(self) -> threading.Thread | None:
        """The thread the job runs on, ``None`` before :meth:`start` starts one."""
        return self._thread

    def start(self, target: Callable[[Self], object]) -> None:
        """Run ``target`` over this job on its own daemon thread."""
        self._thread = threading.Thread(
            target=target, args=(self,), daemon=True
        )
        self._thread.start()


class ProjectJob(Job):
    """A job running one checkpoint over one capture directory, for the ``project`` it serves.

    ``kind`` names what the job does, for a sentence about it. The capture's dataset root and
    name are read from ``images_dir`` (:func:`~tcip_mcp.dataset_layout.parse_capture_dir`).
    """

    kind: ClassVar[str]
    project: str
    checkpoint_path: str
    images_dir: str

    @computed_field  # type: ignore[prop-decorator]
    @property
    def dataset_root(self) -> Path:
        """The dataset root of the capture ``images_dir``."""
        return parse_capture_dir(self.images_dir)[0]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def capture(self) -> str:
        """The name of the capture ``images_dir`` is."""
        return parse_capture_dir(self.images_dir)[1]


def evict_terminal(jobs: dict, max_jobs: int = MAX_JOBS) -> None:
    """Drop the oldest ended jobs of any project from ``jobs``, in place.

    Ended means not :attr:`Job.live`. Drops until ``jobs`` holds at most ``max_jobs`` or every job
    left is live. Relies on dict insertion order (oldest first).
    """
    overflow = len(jobs) - max_jobs
    if overflow > 0:
        evictable = [jid for jid, job in jobs.items() if not job.live]
        for jid in evictable[:overflow]:
            del jobs[jid]


class JobRegistry[J: Job]:
    """The dict-plus-lock live-job registry: launch, find-or-launch, get, list and signal.

    Every job a route answers is a snapshot (``model_copy``) the registry hands out; the live
    object goes only to the worker it starts and to an act it runs for a caller (:meth:`signal`).
    """

    def __init__(self) -> None:
        """Initialize an empty registry keyed by each job's ``job_id``."""
        self.jobs: dict[str, J] = {}
        self.lock = threading.Lock()

    def _insert(self, job: J) -> J:
        """Add ``job`` under its ``job_id``, evict overflow (:func:`evict_terminal`), snapshot it.

        Called under the lock.
        """
        self.jobs[job.job_id] = job
        evict_terminal(self.jobs)
        return job.model_copy()

    def launch(self, job: J, worker: Callable[[J], object]) -> J:
        """Add ``job`` and run ``worker`` over it on its own thread (:meth:`Job.start`).

        Returns its snapshot as it stood before the thread started.
        """
        with self.lock:
            snapshot = self._insert(job)
        job.start(worker)
        return snapshot

    def find_or_launch(
        self,
        match: Callable[[J], bool],
        make: Callable[[], J],
        worker: Callable[[J], object],
    ) -> tuple[J, bool]:
        """Return a snapshot of the first job ``match`` accepts, or launch ``make()``'s new job.

        The match and the insert happen under one lock acquisition; a new job is run as
        :meth:`launch` runs it. Returns ``(snapshot, created)``, ``created`` true only when
        ``make()`` ran.
        """
        with self.lock:
            for job in self.jobs.values():
                if match(job):
                    return job.model_copy(), False
            job = make()
            snapshot = self._insert(job)
        job.start(worker)
        return snapshot, True

    def signal(self, job_id: str, act: Callable[[J], object]) -> J | None:
        """Run ``act`` over the live job ``job_id`` names and return its snapshot after.

        ``None``, with nothing run, for an id naming no job.
        """
        with self.lock:
            job = self.jobs.get(job_id)
            if job is None:
                return None
            act(job)
            return job.model_copy()

    def get(self, job_id: str) -> J | None:
        """Return a snapshot of a job by id, from any project this process holds."""
        with self.lock:
            job = self.jobs.get(job_id)
            return None if job is None else job.model_copy()

    def list[P: ProjectJob](
        self: JobRegistry[P], project: str, *, unended: bool = False
    ) -> list[P]:
        """Return a snapshot of every held job that runs for ``project``.

        With ``unended``, only the jobs that have not ended (:attr:`Job.live`).
        """
        with self.lock:
            return [
                j.model_copy()
                for j in self.jobs.values()
                if j.project == project and (j.live or not unended)
            ]
