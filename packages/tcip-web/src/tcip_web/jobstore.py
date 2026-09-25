"""Persistence and memory-cap helpers for the web's async job registries.

One registry document per job kind per platform root: a job carries the root it launched under
(``platform_root``), and :func:`persist_grouped` writes each root's own jobs to that root's own
key. :func:`evict_terminal` evicts the oldest terminal jobs, first within one root's share, then
across every root this process holds. A persisted job is a record, not a resumable state.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from typing import Any, Literal, get_args

from tcip_store import DecodeError, Key, read, replace

from tcip_mcp.web_client import current_root, job_registry_key

logger = logging.getLogger(__name__)


MAX_JOBS = 100

JobStatus = Literal["pending", "running", "completed", "failed", "canceled", "interrupted"]
"""Every status ``routes.inference.InferenceJob.status`` can hold."""

TERMINAL_STATUSES: frozenset[str] = frozenset({"completed", "failed", "canceled", "interrupted"})


def persist_grouped(name: str, summaries: list[dict]) -> None:
    """Atomically write ``summaries`` to one registry document per root, grouped by each
    summary's own ``platform_root``; a root with no summaries here keeps whatever its document
    already holds."""
    groups: dict[str, list[dict]] = {}
    for s in summaries:
        groups.setdefault(s["platform_root"], []).append(s)
    for root, group in groups.items():
        persist_to(job_registry_key(name, root=root), group)


def persist_to(key: Key, summaries: list[dict]) -> None:
    """Atomically write job summaries to a registry key the caller already resolved; a failure is
    logged with the registry it belongs to, never raised.
    """
    try:
        replace(key, summaries)
    except Exception:
        logger.exception("Could not persist the %s job registry", key.parts[0])


def load(name: str) -> list[dict]:
    """Read persisted job summaries from ``.tcip/state/<name>.json``.

    Returns ``[]`` when the file is missing/undecodable or doesn't hold a list.
    """
    try:
        data = read(job_registry_key(name), default=[])
    except DecodeError:
        return []
    return data if isinstance(data, list) else []


def find_job(jobs: dict, job_id: str):
    """A job by id, from any root this process holds."""
    return jobs.get(job_id)


def evict_terminal(jobs: dict, root: str | None, max_jobs: int = MAX_JOBS) -> None:
    """Drop the oldest terminal jobs, in place, once ``jobs`` overflows ``max_jobs``: once for
    ``root``'s own share, then once for the whole dict across every root this process holds.

    ``root`` scopes the first pass: a job's own ``platform_root`` for a per-root registry, or
    ``None`` for a registry with no root concept of its own, matching every entry, so the two
    passes coincide.

    The second pass evicts the oldest terminal job in the whole dict regardless of which root it
    belongs to, so a root whose own share never overflowed can still lose an entry, and its next
    persisted file is one entry shorter.

    Relies on dict insertion order (oldest first); running/pending jobs are never evicted.
    """
    scoped = [jid for jid, job in jobs.items() if getattr(job, "platform_root", None) == root]
    overflow = len(scoped) - max_jobs
    if overflow > 0:
        evictable = [jid for jid in scoped if getattr(jobs[jid], "status", "") in TERMINAL_STATUSES]
        for jid in evictable[:overflow]:
            jobs.pop(jid, None)

    total_overflow = len(jobs) - max_jobs
    if total_overflow > 0:
        evictable_any = [
            jid for jid, job in jobs.items() if getattr(job, "status", "") in TERMINAL_STATUSES
        ]
        for jid in evictable_any[:total_overflow]:
            jobs.pop(jid, None)


def rehydrated_status(summary: dict) -> JobStatus:
    """A persisted job's status as a rehydrate reads it back: a terminal status as recorded, and
    ``pending`` or ``running`` as ``"interrupted"``, since its worker thread is gone. Raises
    ``ValueError`` naming a status outside :data:`JobStatus`."""
    status = summary["status"]
    if status in TERMINAL_STATUSES:
        return status
    if status in ("pending", "running"):
        return "interrupted"
    raise ValueError(
        f"a persisted job records status {status!r}, not one of {get_args(JobStatus)}")


class JobRegistry:
    """The dict-plus-lock live-job registry: register, get, list, persist and rehydrate, with the
    job codec (``to_summary``/``from_summary``) fixed at construction.
    """

    def __init__(
        self,
        name: str | None = None,
        *,
        to_summary: Callable[[Any], dict] | None = None,
        from_summary: Callable[[dict], Any] | None = None,
        id_field: str = "job_id",
    ) -> None:
        """``name`` is the persisted registry (one of
        :data:`tcip_mcp.web_client.JOB_REGISTRY_DOCUMENTS`) this registry reads and writes through
        :func:`persist_grouped`/:func:`load`; ``None`` for a registry with no root concept of its
        own that persists nothing, which needs neither codec below and so refuses neither.

        ``to_summary`` turns a live job into its persisted summary dict; ``from_summary`` turns a
        persisted summary back into a job instance for :meth:`rehydrate`. A named registry must
        supply both, or construction refuses. ``id_field`` names the summary's own id key.
        """
        if name is not None and (to_summary is None or from_summary is None):
            raise ValueError(
                f"JobRegistry({name!r}) persists and must be constructed with both "
                "to_summary and from_summary"
            )
        self.name = name
        self.jobs: dict[str, Any] = {}
        self.lock = threading.Lock()
        self._to_summary = to_summary
        self._from_summary = from_summary
        self._id_field = id_field

    def register(self, job_id: str, job: Any, *, job_root: str | None) -> None:
        """Add ``job`` under ``job_id``, evict overflow (:func:`evict_terminal`'s own two passes),
        then persist every live job if this registry does. ``job_root`` is the job's own
        ``platform_root`` for a per-root registry, or ``None`` for one with no root concept,
        matching :func:`evict_terminal`'s own contract.
        """
        with self.lock:
            self.jobs[job_id] = job
            evict_terminal(self.jobs, job_root)
        self.persist()

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
        self.persist()
        return job, True

    def get(self, job_id: str) -> Any:
        """A job by id, from any root this process holds: see :func:`find_job`."""
        with self.lock:
            return find_job(self.jobs, job_id)

    def list(self, root: str | None = None) -> list[Any]:
        """Every live job, or only ``root``'s own share when given."""
        with self.lock:
            if root is None:
                return list(self.jobs.values())
            return [j for j in self.jobs.values() if getattr(j, "platform_root", None) == root]

    def persist(self) -> None:
        """Write every live job's own summary, grouped by root, through the codec given at
        construction; a no-op for an unpersisted registry (``name`` is ``None``)."""
        if self.name is None:
            return
        assert self._to_summary is not None, "a named registry always has one, refused otherwise"
        with self.lock:
            summaries = [self._to_summary(j) for j in self.jobs.values()]
        persist_grouped(self.name, summaries)

    def rehydrate(self) -> None:
        """Merge this root's persisted summaries, not already live, into memory via the
        ``from_summary`` codec given at construction, then bound the dict the same way
        :meth:`register` does. A no-op for an unpersisted registry."""
        if self.name is None:
            return
        assert self._from_summary is not None, "a named registry always has one, refused otherwise"
        root = current_root()
        with self.lock:
            for s in load(self.name):
                if s[self._id_field] not in self.jobs:
                    self.jobs[s[self._id_field]] = self._from_summary(s)
            evict_terminal(self.jobs, root)
