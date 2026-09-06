"""Persistence + memory-cap helpers for the web's async job registries.

The inference/HPO/review-priority-queue routes, and images.py's overview builds, each keep an
in-memory dict of background jobs; left unbounded, a dict like this leaks memory, and (for the
three that persist) every job vanishes on restart. These helpers atomically persist job
summaries to ``.tcip/state/<name>.json`` and evict the oldest *terminal* jobs so a live
registry stays bounded, whether or not it persists. (Running jobs can't survive a restart,
their threads are gone, so a persisted file is a record, not a resumable state; images.py's
registry carries no root concept of its own and persists nothing.)

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
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

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

from tcip_mcp.project_paths import platform_state_root

logger = logging.getLogger(__name__)

_REGISTRY_DOC = RootedFileLocator(prefix=(".tcip", "state"), suffix=".json")
"""One registry document per job kind, one per platform root."""

INFERENCE_JOBS = "inference_jobs"
REVIEW_PRIORITY_JOBS = "review_priority_jobs"
HPO_SWEEPS = "hpo_sweeps"

JOB_REGISTRY_DOCUMENTS: tuple[str, ...] = (INFERENCE_JOBS, REVIEW_PRIORITY_JOBS, HPO_SWEEPS)
"""Every document name a job registry persists under ``.tcip/state/<name>.json``.

The one spelling of each name: routes/inference.py, routes/review.py and routes/tuning.py each
hold their own registry constant from here rather than typing the string again. tcip-store
cannot import tcip-web, so the ``job_registry`` claim in ``tcip_store.layout_claims`` cannot
enumerate this tuple itself; a test asserts every name here matches one of that claim's own
templates, holding the agreement from this side.
"""

JOB_REGISTRY_STORE = "job_registry"
register_store(
    StoreDescriptor(
        name=JOB_REGISTRY_STORE,
        kind="record",
        key_fields=("registry",),
        frozen=True,
        cannot_carry_field="a top-level JSON array of entries, with no object to hold the field; "
                            "a future bump wraps this into {schema_version, entries}",
        codec=RECORD_JSON,
        concurrency="last_writer_wins",
        locator=_REGISTRY_DOC,
    )
)


def current_root() -> str:
    """This process's platform-state root, resolved: the value a job's own ``platform_root``
    field carries and :func:`job_registry_key`'s default group."""
    return str(platform_state_root().resolve())


def job_registry_key(name: str, *, root: str | Path | None = None) -> Key:
    """One job registry's persisted summaries, under ``root`` (default: :func:`current_root`).

    ``last_writer_wins``: one root's own group of a registry is written whole from the live
    jobs that carry it, so the file is a snapshot of that root's state rather than a document
    writers merge into.
    """
    resolved = str(Path(root).resolve()) if root is not None else current_root()
    return Key(JOB_REGISTRY_STORE, resolved, (name,))


def require_platform_root(summary: dict, *, name: str, root: str) -> str:
    """``summary['platform_root']``, refusing rather than substituting ``root`` when it is
    missing.

    A summary carrying no ``platform_root`` predates the field, so ``root`` (the caller's
    current process root, or the registry document's own root at load time) would stand in as a
    guess for the summary's actual launch root, not a derived fact. No operator door stamps the
    missing field onto an existing summary; a persisted registry holding one keeps refusing here
    until it is corrected.
    """
    value = summary.get("platform_root")
    if value:
        return value
    job_id = summary.get("job_id") or summary.get("sweep_id") or "<unknown>"
    raise ValueError(
        f"{name} summary {job_id!r} under {root} carries no platform_root; no operator door "
        "stamps the missing field onto an existing summary, so this platform root's persisted "
        "registries stay unreadable here until they are corrected"
    )


MAX_JOBS = 100

JobStatus = Literal["pending", "running", "completed", "failed", "cancelled", "interrupted"]
"""Every status ``routes.inference.InferenceJob.status`` can hold."""

TERMINAL_STATUSES: frozenset[str] = frozenset({"completed", "failed", "cancelled", "interrupted"})


def persist_grouped(
    name: str, summaries: list[dict], *, refused_roots: frozenset[str] = frozenset(),
) -> None:
    """Atomically write ``summaries`` to one registry document per root.

    Groups by each summary's own ``platform_root`` and writes each group under that root's
    own key, so a live registry holding jobs launched under more than one root (this process
    adopted another project in between) never overwrites one project's persisted history with
    a snapshot that includes another's, and never mislabels a root that overflowed out of this
    snapshot as empty: a root with no summaries here keeps whatever its file already holds. A
    summary carrying no ``platform_root`` refuses by name (:func:`require_platform_root`)
    rather than being guessed onto this process's own current root.

    ``refused_roots`` names a root whose stored document this process could not fully
    rehydrate (:meth:`JobRegistry.rehydrate`, on a summary with no ``platform_root``): its own
    group is skipped rather than written, so the untouched document survives on disk, and this
    call raises naming the fact once every other root's group has landed, rather than silently
    dropping the caller's own new work for that root or letting it overwrite the document with
    only the summaries that did rehydrate.
    """
    this_root = current_root()
    groups: dict[str, list[dict]] = {}
    for s in summaries:
        root = require_platform_root(s, name=name, root=this_root)
        groups.setdefault(root, []).append(s)
    blocked = sorted(root for root in groups if root in refused_roots)
    for root, group in groups.items():
        if root in refused_roots:
            continue
        persist_to(job_registry_key(name, root=root), group)
    if blocked:
        raise ValueError(
            f"{name} document(s) under {', '.join(blocked)} were not fully rehydrated this "
            "process (a stored summary carries no platform_root) and this write would destroy "
            "the summaries it never loaded; no operator door stamps the missing field onto an "
            "existing summary, so each blocked root's document must be corrected before this "
            "write can proceed"
        )


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


_startup_refusals: list[dict[str, str]] = []
_startup_refusals_lock = threading.Lock()


def record_startup_refusal(registry: str, error: str) -> None:
    """Record that ``registry``'s rehydrate refused this process's startup or a repin, for
    :func:`startup_refusals` to report until this process restarts against a conformed
    document. One call per refusal: a later repin that refuses again appends its own entry."""
    with _startup_refusals_lock:
        _startup_refusals.append({"registry": registry, "error": error})


def startup_refusals() -> list[dict[str, str]]:
    """Every job-registry rehydrate this process has refused, each naming the conform script
    in its own error text (:func:`require_platform_root`'s message). Reported by the workspace
    status route so a refusal the lifespan's own per-registry try only logged is still visible
    to the breeder rather than silently leaving that registry's history stuck."""
    with _startup_refusals_lock:
        return list(_startup_refusals)


def rehydrated_status(summary: dict) -> JobStatus:
    """A persisted job's status as a rehydrate should read it back.

    Any status this process does not consider terminal is surfaced as ``"interrupted"``: the
    worker thread behind it is gone after a restart or a repin, so it can never reach one of
    :data:`TERMINAL_STATUSES` on its own. A summary with no recorded status at all reads the
    same way, since there is nothing else it could mean.
    """
    status = summary.get("status", "interrupted")
    return status if status in TERMINAL_STATUSES else "interrupted"


class JobRegistry:
    """The dict-plus-lock live-job registry shape restated around each of inference.py's,
    review.py's priority queue's, tuning's own job dataclass, and images.py's overview builds
    (which adopts it unnamed, with no root concept): register, get, list, persist (where a
    registry persists) and rehydrate, sharing this module's own
    :func:`evict_terminal`/:func:`find_job`/:func:`persist_grouped`/:func:`load` underneath.

    Holds no knowledge of a job's own shape: a caller's dataclass and worker logic stays in its
    own module; only the dict, the lock and the four operations move here. The job-shape codec
    (``to_summary``/``from_summary``) is fixed once at construction rather than passed to each
    call, so a named registry's :meth:`register` can never skip its own persist by a caller
    simply omitting the argument at one call site.
    """

    def __init__(
        self,
        name: str | None = None,
        *,
        to_summary: Callable[[Any], dict] | None = None,
        from_summary: Callable[[dict, str], Any] | None = None,
        id_field: str = "job_id",
    ) -> None:
        """``name`` is the persisted registry (one of :data:`JOB_REGISTRY_DOCUMENTS`) this
        registry reads and writes through :func:`persist_grouped`/:func:`load`; ``None`` for a
        registry with no root concept of its own that persists nothing (images.py's overview
        builds), which needs neither codec below and so refuses neither.

        ``to_summary`` turns a live job into its persisted summary dict; ``from_summary`` turns
        a persisted summary plus the current root back into a job instance for
        :meth:`rehydrate`. A named registry must supply both here: a registry that persists but
        was constructed with no codec would otherwise skip every :meth:`register`'s persist (or
        every :meth:`rehydrate`) silently, exactly the gap this refusal closes. ``id_field``
        names the summary's own id key (``job_id`` for inference and the review priority queue,
        ``sweep_id`` for HPO).
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
        # A root whose stored document this instance's rehydrate could not fully load; persist
        # refuses to write over it until a restart rehydrates a conformed document (see below).
        self._refused_roots: set[str] = set()

    def register(self, job_id: str, job: Any, *, job_root: str | None) -> None:
        """Add ``job`` under ``job_id``, evict overflow (:func:`evict_terminal`'s own two
        passes: ``job_root``'s own share, then the whole dict), then persist every live job if
        this registry does. ``job_root`` is the job's own ``platform_root`` for a per-root
        registry, or ``None`` for one with no root concept, matching :func:`evict_terminal`'s
        own contract -- distinct from :meth:`list`'s ``root``, where ``None`` means every root
        this process holds rather than an exact match against ``None``."""
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
        it. The scan-then-insert compound a caller needs when two concurrent requests naming the
        same underlying work (one raster's overview build, one path) must not both pass the scan
        and both start a second job: the check and the insert must run under one lock
        acquisition, so :meth:`register` (which takes its own lock separately) is not the right
        call for this."""
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
        construction; a no-op for an unpersisted registry (``name`` is ``None``).

        Raises, writing nothing for that root, when a root about to be written is in
        :attr:`_refused_roots`: see :func:`persist_grouped`.
        """
        if self.name is None:
            return
        assert self._to_summary is not None, "a named registry always has one, refused otherwise"
        with self.lock:
            summaries = [self._to_summary(j) for j in self.jobs.values()]
        persist_grouped(self.name, summaries, refused_roots=frozenset(self._refused_roots))

    def rehydrate(self) -> None:
        """Merge this root's persisted summaries, not already live, into memory via the
        ``from_summary`` codec given at construction, then bound the dict the same way
        :meth:`register` does. A no-op for an unpersisted registry, which has nothing to
        rehydrate from.

        A summary with no ``platform_root`` (:func:`require_platform_root`, inside
        ``from_summary``) marks this root refused on this instance before the exception
        propagates, so a later :meth:`persist` never overwrites that root's document with only
        the summaries that did load here.
        """
        if self.name is None:
            return
        assert self._from_summary is not None, "a named registry always has one, refused otherwise"
        root = current_root()
        with self.lock:
            try:
                for s in load(self.name):
                    jid = s.get(self._id_field)
                    if not jid or jid in self.jobs:
                        continue
                    self.jobs[jid] = self._from_summary(s, root)
            except ValueError:
                self._refused_roots.add(root)
                raise
            evict_terminal(self.jobs, root)
