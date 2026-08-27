"""Persistence + memory-cap helpers for the web's async job registries.

The inference/HPO/review-priority-queue routes keep an in-memory dict of background jobs;
left unbounded, this dict leaks memory, and every job vanishes on restart. These helpers
atomically persist job summaries to ``.tcip/state/<name>.json`` and evict the oldest
*terminal* jobs so the live registry stays bounded. (Running jobs can't survive a restart,
their threads are gone, so the persisted file is a record, not a resumable state.)

One registry document per job kind per platform root: a job carries the root it launched
under (``platform_root``, set on the request thread at launch), and :func:`persist_grouped`
writes each root's own jobs to that root's own key. :func:`evict_terminal` bounds one root's
own share of the in-memory dict first, then the dict as a whole across every root this
process holds, so the process's memory stays bounded whatever roots it has adopted rather
than growing by ``max_jobs`` for every root that ever registers a job. The whole-dict pass
can evict another root's own oldest terminal job even when that root's own share never
overflowed, and the persist that follows a registration writes that root's file one entry
shorter: one project's activity can shorten another project's persisted registry of
terminal jobs.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

from tcip_store import (
    RECORD_JSON,
    DecodeError,
    Key,
    StoreDescriptor,
    read,
    register_store,
    replace,
)
from tcip_store.file_backend import RootedFileLocator

from tcip_mcp.project_paths import project_root

logger = logging.getLogger(__name__)

_REGISTRY_DOC = RootedFileLocator(prefix=(".tcip", "state"), suffix=".json")
"""One registry document per job kind, one per platform root."""

JOB_REGISTRY_STORE = "job_registry"
register_store(
    StoreDescriptor(
        name=JOB_REGISTRY_STORE,
        kind="record",
        key_fields=("registry",),
        codec=RECORD_JSON,
        concurrency="last_writer_wins",
        locator=_REGISTRY_DOC,
    )
)


def current_root() -> str:
    """This process's platform-state root, resolved: the value a job's own ``platform_root``
    field carries and :func:`job_registry_key`'s default group."""
    return str(project_root().resolve())


def job_registry_key(name: str, *, root: str | Path | None = None) -> Key:
    """One job registry's persisted summaries, under ``root`` (default: :func:`current_root`).

    ``last_writer_wins``: one root's own group of a registry is written whole from the live
    jobs that carry it, so the file is a snapshot of that root's state rather than a document
    writers merge into.
    """
    resolved = str(Path(root).resolve()) if root is not None else current_root()
    return Key(JOB_REGISTRY_STORE, resolved, (name,))


MAX_JOBS = 100

JobStatus = Literal["pending", "running", "completed", "failed", "cancelled", "interrupted"]
"""Every status ``routes.inference.InferenceJob.status`` can hold."""

TERMINAL_STATUSES: frozenset[str] = frozenset({"completed", "failed", "cancelled", "interrupted"})


def persist_grouped(name: str, summaries: list[dict]) -> None:
    """Atomically write ``summaries`` to one registry document per root.

    Groups by each summary's own ``platform_root`` and writes each group under that root's
    own key, so a live registry holding jobs launched under more than one root (this process
    adopted another project in between) never overwrites one project's persisted history with
    a snapshot that includes another's, and never mislabels a root that overflowed out of this
    snapshot as empty: a root with no summaries here keeps whatever its file already holds. A
    summary carrying no ``platform_root`` (a registry document written before this field
    existed) groups under this process's own current root, the file it already lived under.
    """
    groups: dict[str, list[dict]] = {}
    for s in summaries:
        root = s.get("platform_root") or current_root()
        groups.setdefault(root, []).append(s)
    for root, group in groups.items():
        persist_to(job_registry_key(name, root=root), group)


def persist_to(key: Key, summaries: list[dict]) -> None:
    """Atomically write job summaries to a registry key the caller already resolved.

    :func:`persist_grouped`'s own writer, once per root it grouped a registry's live jobs
    into, so the write lands under each job's own launch root rather than under whatever the
    environment names by the time the write happens.

    A failure here loses the GUI's history of this registry across a restart, not the jobs
    themselves, so it is logged with the registry it belongs to rather than raised into the
    route that was reporting a job's progress.
    """
    try:
        replace(key, summaries)
    except Exception:
        logger.exception("Could not persist the %s job registry", key.parts[0])


def load(name: str) -> list[dict]:
    """Read persisted job summaries from ``.tcip/state/<name>.json``.

    Returns ``[]`` when the file is missing/undecodable or doesn't hold a list. Absence and
    corruption are folded together on purpose here, unlike stores where they differ: this
    file is a record of jobs whose threads are already gone, so a restart that cannot read it
    starts clean rather than refusing to start.
    """
    try:
        data = read(job_registry_key(name), default=[])
    except DecodeError:
        return []
    return data if isinstance(data, list) else []


def find_job(jobs: dict, job_id: str):
    """A job by id, from any root this process holds.

    A job id is unique per process and the breeder that launched it holds the id, so a repin
    to another project must not make an in-flight job unreachable by it. The one by-id lookup
    every per-root registry (inference, HPO, the review priority queue) shares, so their
    by-id routes agree without each re-deriving whether a lookup is root-scoped; a caller's
    list route stays scoped by filtering on its own job's ``platform_root`` instead, which
    this function is never the right tool for.
    """
    return jobs.get(job_id)


def evict_terminal(jobs: dict, root: str | None, max_jobs: int = MAX_JOBS) -> None:
    """Drop the oldest terminal jobs, in place, once ``jobs`` overflows ``max_jobs``: once for
    ``root``'s own share, then once for the whole dict across every root this process holds, so
    the process's memory stays bounded whatever roots it has adopted rather than growing by
    ``max_jobs`` for every root that ever registers a job.

    ``root`` scopes the first pass: pass a job's own ``platform_root`` for a per-root registry
    (inference, HPO, the review priority queue). A registry with no root concept of its own
    (nothing in ``jobs`` carries a ``platform_root``) passes ``None``, matching every entry, so
    the two passes coincide and it keeps its original single-collection behaviour.

    The second pass evicts the oldest terminal job in the whole dict regardless of which root
    it belongs to, so a root whose own share never overflowed can still lose an entry here; the
    persist that follows a registration writes that root's file one entry shorter, so one
    project's activity can shorten another project's persisted registry of terminal jobs.

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
