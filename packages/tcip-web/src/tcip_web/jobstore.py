"""Persistence + memory-cap helpers for the web's async job registries.

The inference/HPO/review-priority-queue routes keep an in-memory dict of background jobs;
left unbounded, this dict leaks memory, and every job vanishes on restart. These helpers
atomically persist job summaries to ``.tcip/state/<name>.json`` and evict the oldest
*terminal* jobs so the live registry stays bounded. (Running jobs can't survive a restart,
their threads are gone, so the persisted file is a record, not a resumable state.)

One registry document per job kind per platform root: a job carries the root it launched
under (``platform_root``, set on the request thread at launch), :func:`persist_grouped`
writes each root's own jobs to that root's own key, and :func:`evict_terminal` bounds one
root's own share of the in-memory dict, so one project's history cannot push another
project's jobs out of memory or overwrite its persisted file.
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

    The form a background worker takes: it is handed its key when it is spawned and writes
    through this, so the write lands under the root its launch resolved rather than under
    whatever the environment names by the time the worker gets there.

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


def evict_terminal(jobs: dict, root: str | None, max_jobs: int = MAX_JOBS) -> None:
    """Drop the oldest terminal jobs of one root, in place, once that root's own share of
    ``jobs`` exceeds ``max_jobs``, so one project's history cannot push another project's
    jobs out of memory.

    ``root`` scopes which of ``jobs`` count towards ``max_jobs`` and are eligible for
    eviction: pass a job's own ``platform_root`` for a per-root registry (inference, HPO,
    the review priority queue). A registry with no root concept of its own (nothing in
    ``jobs`` carries a ``platform_root``) passes ``None``, matching every entry and bounding
    the whole dict as one collection, its original behaviour.

    Relies on dict insertion order (oldest first, within that root's own entries);
    running/pending jobs are never evicted.
    """
    scoped = [jid for jid, job in jobs.items() if getattr(job, "platform_root", None) == root]
    overflow = len(scoped) - max_jobs
    if overflow <= 0:
        return
    evictable = [
        jid for jid in scoped if getattr(jobs[jid], "status", "") in TERMINAL_STATUSES
    ]
    for jid in evictable[:overflow]:
        jobs.pop(jid, None)
