"""Persistence + memory-cap helpers for the web's async job registries.

The inference/HPO routes keep an in-memory dict of background jobs; left unbounded, this dict
leaks memory, and every job vanishes on restart. These helpers atomically persist job
summaries to ``.tcip/state/<name>.json`` and evict the oldest *terminal* jobs so the live
registry stays bounded. (Running jobs can't survive a restart, their threads are gone,
so the persisted file is a record, not a resumable state.)
"""

from __future__ import annotations

import logging
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
"""One registry document per job kind, under the platform state root."""

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


def job_registry_key(name: str) -> Key:
    """One job registry's persisted summaries.

    ``last_writer_wins``: the route owns the live registry in memory and persists the whole
    list from it, so the file is a snapshot of one process's state rather than a document
    writers merge into.
    """
    return Key(JOB_REGISTRY_STORE, str(project_root().resolve()), (name,))


MAX_JOBS = 100

JobStatus = Literal["pending", "running", "completed", "failed", "cancelled", "interrupted"]
"""Every status ``routes.inference.InferenceJob.status`` can hold."""

TERMINAL_STATUSES: frozenset[str] = frozenset({"completed", "failed", "cancelled", "interrupted"})


def persist(name: str, summaries: list[dict]) -> None:
    """Atomically write job summaries to ``.tcip/state/<name>.json``.

    Resolves the registry key against this process's root right here, so it is the form for a
    caller running where that root is the one it means, which is any request thread.
    """
    persist_to(job_registry_key(name), summaries)


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
