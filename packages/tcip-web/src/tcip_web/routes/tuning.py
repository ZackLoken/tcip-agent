"""HPO / Tuning routes: relaunch + cancel + list + per-trial visibility.

Sweeps reach this surface two ways: launched here over HTTP (tracked in memory for as long
as this process lives), or launched by the agent straight through the ``run_hyperparameter_search`` MCP tool,
which this process never sees. The durable source for both is the ``manifest.json`` that
``run_hyperparameter_search`` stamps under the sweep's own directory when the sweep starts, so the listing
below reads disk and overlays the in-memory jobs, the same live-plus-historical merge the
Training routes do for runs.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from tcip_web import jobstore
from tcip_web.routes._body_common import EmptyBodyPayload
from tcip_web.routes._metrics_common import metrics_response

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/tuning", tags=["tuning"])

_TRIAL_DIR_PREFIX = "trial_"

HPO_REGISTRY = jobstore.HPO_SWEEPS
"""The job registry this module persists its sweeps to."""


def _current_root() -> str:
    from tcip_web import jobstore
    return jobstore.current_root()


@dataclass
class HPOJob:
    sweep_id: str
    status: str = "pending"
    error: Optional[str] = None
    result: dict[str, Any] = field(default_factory=dict)
    # The platform root this sweep launched under, resolved on the request thread.
    platform_root: str = field(default_factory=_current_root)
    # A relaunch's source sweep name, in memory only; the persisted document gains nothing.
    relaunched_from: Optional[str] = None


def _manifest_fields(manifest: dict) -> dict:
    """The config-picker projection of a sweep manifest: search shape, relaunchability,
    whether a cancel has been requested, and which sweep (if any) it was relaunched from.
    Shared by :func:`_manifest_summary` (a disk-only sweep's own manifest) and :func:`_summary`
    (a live sweep's row, read from the manifest under its own launch root), so a sweep reads
    the same fields whichever way it is listed.

    ``base_config`` is the relaunchable marker (:func:`training_tools.run_hyperparameter_search` writes it
    whenever it creates a manifest); its absence is reported with a reason rather than a
    reconstructed config. ``cancel_requested`` is the manifest's own field, set by
    ``cancel_hyperparameter_search`` and never derived from a side file this route cannot see across roots.
    ``relaunched_from`` projects as ``None`` for a manifest predating the field, the same as
    for one that genuinely was not a relaunch: a relaunch of an older sweep must still work
    (see :func:`_missing_relaunch_fields`, which does not require this key). ``split_draws``
    projects the same way, ``None`` for a manifest predating it (read as 1, no draws, by
    ``run_hyperparameter_search``'s own default): the header line only renders it above 1. ``redraws_within_manifest``
    reads the recorded ``base_config``'s own ``data.split.redraw_within_manifest``, ``False``
    for a manifest carrying none (predating the flag, or a config that never set it): the
    header's draws line names the bound manifest as what each draw redraws inside only then.

    An empty ``manifest`` (no manifest exists yet, or one predating a caller's own launch) is
    never relaunchable, but carries no reason either: the pre-manifest window and every refused
    relaunch never mint one, and "this sweep's record holds no base config" would misname
    absence itself as a recorded fact. A caller with words for that case has them in
    ``job.error`` instead.
    """
    if not manifest:
        return {
            "n_trials": None, "search_alg": None, "scheduler": None, "param_space_keys": [],
            "relaunchable": False, "reason": None, "cancel_requested": False,
            "relaunched_from": None, "split_draws": None, "redraws_within_manifest": False,
        }
    relaunchable = "base_config" in manifest
    base_config = manifest.get("base_config") or {}
    base_split = (base_config.get("data") or {}).get("split") or {}
    return {
        "n_trials": manifest.get("n_trials"),
        "search_alg": manifest.get("search_alg"),
        "scheduler": manifest.get("scheduler"),
        "param_space_keys": sorted((manifest.get("param_space") or {}).keys()),
        "relaunchable": relaunchable,
        "reason": None if relaunchable else "this sweep's record holds no base config",
        "cancel_requested": bool(manifest.get("cancel_requested")),
        "relaunched_from": manifest.get("relaunched_from"),
        "split_draws": manifest.get("split_draws"),
        "redraws_within_manifest": bool(base_split.get("redraw_within_manifest")),
    }


def _persisted_summary(job: HPOJob) -> dict:
    """The registry's own persisted fields for one job, with no manifest read: what
    :class:`~tcip_web.jobstore.JobRegistry` writes to ``.tcip/state/hpo_sweeps.json`` under its
    own lock, and reads back on rehydrate. The manifest projection belongs to a listing row
    (:func:`_summary`), never to the persisted document: a frozen record must not gain fields
    over time, and a store read has no business running while the registry's lock is held.
    """
    return {"sweep_id": job.sweep_id, "status": job.status,
            "error": job.error, "has_result": bool(job.result),
            "platform_root": job.platform_root}


def _driver_live(sweep_id: str) -> bool:
    """Whether this process's own worker thread for ``sweep_id`` is still running: the one
    signal :func:`~tcip_mcp.tools.training_tools.sweep_state` trusts over the manifest's own
    heartbeat, since a live thread proves the driver is running this instant."""
    with _lock:
        thread = _workers.get(sweep_id)
        return thread is not None and thread.is_alive()


def _job_sweep_state(job: HPOJob, manifest: dict) -> str:
    """A live-registry job's derived liveness. The registry's own record wins once ``job.status``
    is itself a recorded done state, so a job the registry knows completed never reads
    ``interrupted`` off a manifest whose terminal write ``run_hyperparameter_search`` tolerates failing;
    otherwise the manifest is the source when one exists, else the registry's own record, the
    only source of truth for a job that never got as far as writing a manifest (a relaunch
    ``run_hyperparameter_search`` refused at preflight, or any job that failed before its first manifest write).
    Either way the rule is :func:`~tcip_mcp.tools.training_tools.sweep_state`'s own: a recorded
    done state is trusted, a live worker thread reads ``running``, anything else reads
    ``interrupted``."""
    from tcip_mcp.experiments import _RECORDED_AS_DONE
    from tcip_mcp.tools.training_tools import TCIP_HEARTBEAT_STALE_SECONDS, sweep_state

    if job.status in _RECORDED_AS_DONE:
        source = {"status": job.status}
    else:
        source = manifest if manifest else {"status": job.status}
    return sweep_state(source, stale_seconds=TCIP_HEARTBEAT_STALE_SECONDS,
                       driver_live=_driver_live(job.sweep_id))


def _summary(job: HPOJob) -> dict:
    """One live sweep's listing row: the registry's own persisted fields plus the manifest
    projection, read fresh (outside the registry's own lock) for whichever caller is listing.
    ``status`` is the derived liveness (:func:`_job_sweep_state`), never the registry's own
    persisted ``job.status`` verbatim: the persisted document keeps that (see
    :func:`_persisted_summary`), and gains nothing from this derivation.

    ``has_manifest`` reports whether ``run_hyperparameter_search`` has written this sweep's first manifest yet,
    the fact a caller keys a pre-manifest render on rather than a 404 the relaunch route never
    produces (it registers the job before it answers). ``relaunched_from`` falls back to the
    job's own in-memory record (set at relaunch) only in that same pre-manifest window; once a
    manifest exists, its own recorded field is authoritative and wins.
    """
    raw_manifest = _read_manifest(job.sweep_id, root=job.platform_root)
    manifest = raw_manifest or {}
    has_manifest = raw_manifest is not None
    status = _job_sweep_state(job, manifest)
    fields = _manifest_fields(manifest)
    if not has_manifest and job.relaunched_from:
        fields["relaunched_from"] = job.relaunched_from
    return {**_persisted_summary(job), **fields, "status": status, "has_manifest": has_manifest}


def _from_summary(s: dict, root: str) -> HPOJob:
    return HPOJob(
        sweep_id=s["sweep_id"], status=jobstore.rehydrated_status(s), error=s.get("error"),
        platform_root=jobstore.require_platform_root(s, name=HPO_REGISTRY, root=root),
    )


_registry = jobstore.JobRegistry(
    HPO_REGISTRY, to_summary=_persisted_summary, from_summary=_from_summary, id_field="sweep_id",
)
"""The dict-plus-lock live registry for this route's own sweeps (see ``jobstore.JobRegistry``),
the shared home review.py's priority queue and inference.py's jobs adopt too. ``_lock`` below is
this registry's own lock, bound under its historical name since callers (tests among them, and
this module's own ``_workers`` guard) already reach into it directly."""

_lock = _registry.lock
_workers: dict[str, threading.Thread] = {}
"""Every sweep worker this process has spawned and not yet seen finish, by sweep id. Guarded
by ``_lock``, the same lock ``_registry`` takes for its own dict; the two are unrelated state
sharing one mutex, predating the registry adoption, not a stated invariant between them."""


def wait_for_workers(*, timeout_s: float) -> tuple[str, ...]:
    """Join this module's sweep workers and return the sweeps still running when time ran out.

    A worker writes through the process's storage backend for as long as it runs, so a caller
    that is about to close that backend, or otherwise must not outrace a sweep, waits here
    instead of sleeping. ``timeout_s`` is the caller's own bound and has no default: how long a
    shutdown or a test may block is the caller's decision, not this module's.
    """
    with _lock:
        pending = list(_workers.items())
    deadline = time.monotonic() + timeout_s
    for _, thread in pending:
        thread.join(max(0.0, deadline - time.monotonic()))
    with _lock:
        for sweep_id, thread in pending:
            if not thread.is_alive():
                _workers.pop(sweep_id, None)
    return tuple(sweep_id for sweep_id, thread in pending if thread.is_alive())


def _manifest_summary(manifest: dict) -> dict:
    """A sweep read off disk, in the same shape as an in-memory job's summary. No process here
    can vouch for its driver, so ``status`` derives with ``driver_live=False``: a manifest whose
    heartbeat has gone stale reads ``interrupted`` rather than the ``running`` it still says."""
    from tcip_mcp.tools.training_tools import TCIP_HEARTBEAT_STALE_SECONDS, sweep_state

    return {"sweep_id": manifest.get("study_name", ""),
            "status": sweep_state(manifest, stale_seconds=TCIP_HEARTBEAT_STALE_SECONDS, driver_live=False),
            "error": manifest.get("error"),
            "has_result": bool(manifest.get("result")),
            "external": True,
            "has_manifest": True,
            **_manifest_fields(manifest)}


def _sweeps_dir() -> Path:
    from tcip_mcp.tools.training_tools import hpo_root

    return hpo_root()


def _read_manifest(sweep_id: str, *, root: Path | str | None = None) -> dict | None:
    """The sweep's manifest, or ``None`` if no sweep by that name is on disk under ``root``
    (default: the current platform root).

    ``sweep_id`` is untrusted, and the store's own key constructor is what refuses one that
    would address a record outside the HPO store. A manifest that will not decode is reported
    and then answered as absent: the listing serves every other sweep rather than failing
    whole, and the sweep with the unreadable manifest is not presented as running.
    """
    from tcip_store import BadKey, DecodeError, store

    from tcip_mcp.tools.training_tools import sweep_manifest_key

    try:
        key = sweep_manifest_key(sweep_id, root=root)
    except BadKey as exc:
        raise HTTPException(400, f"invalid sweep_id: {sweep_id}") from exc
    try:
        manifest = store.read(key, default=None)
    except DecodeError:
        logger.warning("the manifest for sweep %s does not decode", sweep_id, exc_info=True)
        return None
    return manifest if isinstance(manifest, dict) else None


def _terminal_response(job: HPOJob) -> dict:
    """The live-registry response for a job already in a terminal status.

    A job this process ran to completion itself carries its full result in memory; a job a
    restart rehydrated carries none (the live registry persists status, not trial results), so
    its result is read off the disk manifest instead, exactly as the disk-only branch of
    :func:`get_sweep` would serve it.
    """
    result = job.result
    if not result:
        manifest = _read_manifest(job.sweep_id, root=job.platform_root)
        if manifest is not None:
            result = manifest.get("result") or {}
    return {"sweep_id": job.sweep_id, "status": job.status, "error": job.error, "result": result}


def _sweep_launch_root(sweep_id: str) -> Optional[str]:
    """The root this sweep's own live registry entry says it launched under, or ``None`` when
    the registry has forgotten it (never launched here, or launched before a restart), the
    only case a bare disk listing can resolve under the current root instead.

    The one lookup shared by everywhere a sweep's own files (its trial directory, its
    TensorBoard link farm) must be addressed under the root the sweep actually belongs to.
    """
    job = _registry.get(sweep_id)
    return job.platform_root if job is not None else None


def _sweep_root(sweep_id: str) -> Path:
    """The directory a sweep's trials live in, once a manifest proves the sweep exists.

    A sweep this process's own live registry still remembers is addressed under the root it
    launched under (:func:`_sweep_launch_root`), so a running or finished sweep stays
    reachable through the sweep detail, trial view and TensorBoard routes across a repin to
    another project. A sweep the registry has forgotten (never launched here, or launched
    before a restart) is addressed under the current root instead, the only root a bare disk
    listing can mean.

    The sweep's own resolved location is what this is, not a path recorded inside the
    manifest: an absolute path in a file is not a path this process should follow.
    """
    from tcip_mcp.tools.training_tools import sweep_dir

    root = _sweep_launch_root(sweep_id)
    if _read_manifest(sweep_id, root=root) is None:
        raise HTTPException(404, f"sweep not found: {sweep_id}")
    return sweep_dir(sweep_id, root=root)


def _disk_sweeps(exclude: frozenset[str] = frozenset()) -> list[dict]:
    """Every sweep with a manifest under the HPO root, in directory-name order, skipping a
    directory named in ``exclude``: a live sweep's manifest is already read fresh by
    :func:`_summary`, so a listing that also read it here would only discard the second read."""
    root = _sweeps_dir()
    if not root.is_dir():
        return []
    found: list[dict] = []
    for d in sorted(root.iterdir()):
        if not d.is_dir() or d.name in exclude:
            continue
        manifest = _read_manifest(d.name)
        if isinstance(manifest, dict) and manifest.get("study_name"):
            found.append(_manifest_summary(manifest))
    return found


def _persist() -> None:
    """Write the live registry, grouped by each sweep's own launch root.

    A sweep's summary carries the root it launched under, resolved once at launch, so this
    reaches the right root's file even from a background worker after this process has since
    adopted another project.
    """
    _registry.persist()


def rehydrate_for_current_root() -> None:
    """Merge this root's persisted sweeps, not already live, into memory via :func:`_from_summary`.

    Called at startup and again after this process repins to another root. Worker threads
    behind a persisted non-terminal sweep are gone, so it is surfaced as ``interrupted``.
    Trial results aren't persisted, so a rehydrated sweep has no result. Merges by sweep id
    rather than requiring an empty registry first, so it never displaces a sweep still live
    from another root. Bounds the dict afterwards the same way launching a sweep does, so
    adopting N roots without ever launching one here still keeps this process's memory
    bounded rather than growing by ``MAX_JOBS`` for every root adopted.
    """
    _registry.rehydrate()


class RelaunchSweepPayload(BaseModel):
    study_name: str


@dataclass
class _RelaunchSpec:
    """Every ``run_hyperparameter_search`` argument a relaunchable manifest holds, read once so the worker
    replays exactly what the manifest recorded rather than trusting untyped dict access at
    the call site."""

    base_config: dict[str, Any]
    param_space: Optional[dict[str, Any]]
    n_trials: int
    search_alg: str
    scheduler: str
    grace_period: int
    reduction_factor: int
    max_concurrent: int
    warm_start: bool
    baseline_params: Optional[dict[str, Any]]
    resources_per_trial: Optional[dict[str, Any]]
    split_draws: int = 1
    split_draw_seeds: Optional[list[int]] = None


_RELAUNCH_FIELDS: tuple[str, ...] = (
    "base_config", "param_space", "n_trials", "search_alg", "scheduler", "grace_period",
    "reduction_factor", "max_concurrent", "warm_start", "baseline_params", "resources_per_trial",
)
"""Every ``run_hyperparameter_search`` argument a relaunch replays. ``run_hyperparameter_search`` writes every one of these as a
key whenever it creates a manifest (a value of ``None`` is a recorded choice, not an absence),
so only a manifest without the field, or one truncated some other way, names anything as
missing here."""


def _missing_relaunch_fields(manifest: dict) -> list[str]:
    """Every field in :data:`_RELAUNCH_FIELDS` that ``manifest`` does not carry as a key."""
    return sorted(f for f in _RELAUNCH_FIELDS if f not in manifest)


def _relaunch_spec(manifest: dict) -> _RelaunchSpec:
    """``manifest``'s own ``run_hyperparameter_search`` arguments, read directly rather than defaulted: a caller
    checks :func:`_missing_relaunch_fields` first, so a key absent here would be a programming
    error, never a silently substituted value that was never the sweep's own.

    ``split_draws``/``split_draw_seeds`` are the one exception, read with ``run_hyperparameter_search``'s own
    defaults (1, ``None``) rather than required: a manifest without the field carries neither
    key, and must still relaunch."""
    return _RelaunchSpec(
        base_config=manifest["base_config"],
        param_space=manifest["param_space"],
        n_trials=manifest["n_trials"],
        search_alg=manifest["search_alg"],
        scheduler=manifest["scheduler"],
        grace_period=manifest["grace_period"],
        reduction_factor=manifest["reduction_factor"],
        max_concurrent=manifest["max_concurrent"],
        warm_start=bool(manifest["warm_start"]),
        baseline_params=manifest["baseline_params"],
        resources_per_trial=manifest["resources_per_trial"],
        split_draws=manifest.get("split_draws", 1),
        split_draw_seeds=manifest.get("split_draw_seeds"),
    )


def _worker(job: HPOJob, spec: _RelaunchSpec, output_dir: str, relaunched_from: str) -> None:
    """Run one relaunched sweep to completion off the request thread.

    ``job.platform_root``, resolved on the request thread at launch, is what
    :func:`_persist` groups this sweep's summary under, so this thread's writes land under
    the root that launch named rather than under whatever the environment names later. A
    returned ``{"error", "issues"}`` (a relaunch whose data paths moved, say) is a failed job
    carrying the issues, and a returned ``{"status": "cancelled", ...}`` a cancelled one,
    rather than either reading as a completed job with no useful result. ``relaunched_from``
    is the source study's own name, recorded on this sweep's manifest so a listing can show it.
    Discards ``job.sweep_id``'s own pre-manifest launch mark on every exit, ``run_hyperparameter_search`` never
    reached included, so a caller that failed before or inside that call leaves no mark behind.
    """
    from tcip_mcp.tools.training_tools import discard_sweep_launching

    try:
        job.status = "running"
        _persist()
        from tcip_mcp.tools.training_tools import run_hyperparameter_search

        res = run_hyperparameter_search(
            base_config=spec.base_config,
            param_space=spec.param_space,
            n_trials=spec.n_trials,
            output_dir=output_dir,
            search_alg=spec.search_alg,
            scheduler=spec.scheduler,
            grace_period=spec.grace_period,
            reduction_factor=spec.reduction_factor,
            warm_start=spec.warm_start,
            baseline_params=spec.baseline_params,
            max_concurrent=spec.max_concurrent,
            resources_per_trial=spec.resources_per_trial,
            study_name=job.sweep_id,
            auto_tensorboard=False,
            split_draws=spec.split_draws,
            split_draw_seeds=spec.split_draw_seeds,
            relaunched_from=relaunched_from,
        )
        if isinstance(res, dict) and res.get("status") == "cancelled":
            # Checked before the "error" key below: a cancelled result carries its own reason
            # under that same key, and must still read cancelled, not failed, because of it.
            job.status = "cancelled"
            job.error = res.get("error")
            job.result = res
        elif isinstance(res, dict) and "error" in res:
            job.status = "failed"
            job.error = res["error"]
            job.result = res
        else:
            job.result = res if isinstance(res, dict) else {"raw": res}
            job.status = "completed"
    except Exception as exc:
        logger.exception("HPO sweep %s failed", job.sweep_id)
        job.status = "failed"
        job.error = str(exc)
    finally:
        discard_sweep_launching(job.sweep_id)
        _persist()


@router.post("/sweeps")
def relaunch_sweep(payload: RelaunchSweepPayload) -> dict:
    """Relaunch a sweep from its own recorded manifest: no config, param space or path is
    ever submitted by the browser. The source manifest is read under this sweep's own launch
    root (:func:`_sweep_launch_root`, the same resolution the cancel route uses), so a sweep
    launched under a root this process has since repinned away from is still relaunchable.
    ``output_dir`` is always this request thread's own ``hpo_root()``, never a path the
    manifest carries (an absolute path in a file is not a path this process should follow).
    ``relaunched_from`` is recorded from the source manifest's own ``study_name``, never the
    request body's echoed string, so the two names it might resolve to the same record under
    (case, say) never disagree on this sweep's manifest.
    Marks the new sweep id as launching, on this request thread, before the worker starts, so
    a cancel that arrives before ``run_hyperparameter_search`` writes its own first manifest still reaches it."""
    from tcip_mcp.tools.training_tools import hpo_root, mark_sweep_launching

    manifest = _read_manifest(payload.study_name, root=_sweep_launch_root(payload.study_name))
    if manifest is None:
        raise HTTPException(404, f"sweep not found: {payload.study_name}")
    if "base_config" not in manifest:
        raise HTTPException(409, "this sweep's record holds no base config")
    missing = _missing_relaunch_fields(manifest)
    if missing:
        raise HTTPException(409, f"this sweep's record is missing {missing}: cannot relaunch")

    output_dir = str(hpo_root())
    spec = _relaunch_spec(manifest)

    job = HPOJob(sweep_id=f"hpo-{uuid.uuid4().hex[:8]}", relaunched_from=manifest["study_name"])
    _registry.register(job.sweep_id, job, job_root=job.platform_root)
    mark_sweep_launching(job.sweep_id, output_dir)
    t = threading.Thread(
        target=_worker, args=(job, spec, output_dir, manifest["study_name"]), daemon=True)
    with _lock:
        for sweep_id in [sid for sid, done in _workers.items() if not done.is_alive()]:
            _workers.pop(sweep_id, None)
        _workers[job.sweep_id] = t
    t.start()
    return {"status": "launched", "sweep_id": job.sweep_id}


@router.post("/sweeps/{sweep_id}/cancel")
def cancel_sweep_route(sweep_id: str, payload: EmptyBodyPayload) -> dict:
    """Request cooperative cancellation of a running sweep, wrapping ``cancel_hyperparameter_search``."""
    from tcip_mcp.tools.training_tools import cancel_hyperparameter_search

    result = cancel_hyperparameter_search(sweep_id, root=_sweep_launch_root(sweep_id))
    if result.get("error"):
        raise HTTPException(404, result["error"])
    return result


@router.get("/sweeps")
def list_sweeps() -> dict:
    """Sweeps this process launched under its current root, backfilled with the ones on disk.

    A sweep the agent launched exists only as a manifest, and a sweep launched here has a
    manifest too, so a live entry wins by sweep_id and each sweep is listed once. The disk
    listing (``hpo_root()``) is already scoped to the current root, so only the live half
    needs its own root filter.
    """
    from tcip_web import jobstore

    live = [_summary(j) for j in _registry.list(jobstore.current_root())]
    live_ids = frozenset(s["sweep_id"] for s in live)
    return {"sweeps": live + _disk_sweeps(exclude=live_ids)}


@router.get("/sweeps/{sweep_id}")
def get_sweep(sweep_id: str) -> dict:
    """One sweep by id: a live entry (whichever root it launched under) wins, else its
    manifest under the current root. ``status`` is the derived liveness in both branches (see
    :func:`_job_sweep_state`/:func:`_manifest_summary`), not either source's own recorded
    value; ``relaunched_from`` is a top-level field on both branches too, the manifest's own
    when one exists, else (live branch only, pre-manifest) the job's own in-memory record.
    ``has_manifest`` is always true on the disk branch (a disk sweep has one by definition) and
    reflects whether ``run_hyperparameter_search`` has written one yet on the live branch. The
    disk branch is :func:`training_tools.read_sweep_from_disk`'s own result, so it also carries
    ``trials``, absent from the live branch (a different question, answered by the trials route)."""
    from tcip_store import BadKey
    from tcip_web import jobstore
    from tcip_mcp.tools.training_tools import enrich_with_study_result, read_sweep_from_disk

    j = _registry.get(sweep_id)
    if j is not None:
        raw_manifest = _read_manifest(j.sweep_id, root=j.platform_root)
        manifest = raw_manifest or {}
        has_manifest = raw_manifest is not None
        status = _job_sweep_state(j, manifest)
        response = (
            _terminal_response(j) if j.status in jobstore.TERMINAL_STATUSES
            else {"sweep_id": j.sweep_id, "status": j.status, "error": j.error, "result": j.result}
        )
        response["status"] = status
        response["has_manifest"] = has_manifest
        response["relaunched_from"] = (
            manifest.get("relaunched_from") if has_manifest else j.relaunched_from
        )
        return enrich_with_study_result(response, sweep_id, root=j.platform_root)
    try:
        disk = read_sweep_from_disk(sweep_id)
    except BadKey as exc:
        raise HTTPException(400, f"invalid sweep_id: {sweep_id}") from exc
    if disk is None:
        raise HTTPException(404, f"sweep not found: {sweep_id}")
    disk = enrich_with_study_result(disk, sweep_id)
    return {**disk, "external": True}


@router.get("/sweeps/{sweep_id}/trials")
def list_trials(sweep_id: str) -> dict:
    """The trial directories a sweep has produced so far, from the shared disk reader (see
    :func:`training_tools.read_sweep_from_disk`), under whichever root the sweep actually
    launched under.

    Ray names its own per-trial directories after the trainable, so only the
    ``trial_<id>`` dirs the platform writes are listed here.
    """
    from tcip_store import BadKey
    from tcip_mcp.tools.training_tools import read_sweep_from_disk

    try:
        disk = read_sweep_from_disk(sweep_id, root=_sweep_launch_root(sweep_id))
    except BadKey as exc:
        raise HTTPException(400, f"invalid sweep_id: {sweep_id}") from exc
    if disk is None:
        raise HTTPException(404, f"sweep not found: {sweep_id}")
    return {"sweep_id": sweep_id, "trials": disk["trials"]}


@router.get("/sweeps/{sweep_id}/trials/{trial_id}/metrics")
def get_trial_metrics(sweep_id: str, trial_id: str) -> dict:
    """Every metrics row one trial has written.

    ``exists`` reports whether the log holds anything: rows, an entry still being appended, or
    bytes that will not decode. A trial that has logged nothing and a trial with no log at all
    are the same answer to the caller, which is what the Tuning view asks.
    """
    from tcip_store import BadKey, read_log

    from tcip_mcp.tools.training_tools import log_holds_anything, trial_metrics_key

    root = _sweep_root(sweep_id)
    try:
        page = read_log(trial_metrics_key(root, f"{_TRIAL_DIR_PREFIX}{trial_id}"))
    except BadKey as exc:
        raise HTTPException(400, f"invalid trial_id: {trial_id}") from exc
    if page.corrupt:
        logger.warning("trial %s has %d metrics rows that do not decode",
                       trial_id, len(page.corrupt))
    if page.version_refused:
        logger.warning("trial %s has %d metrics rows at a schema_version this reader does "
                       "not accept", trial_id, len(page.version_refused))
    rows = [dict(row) for row in page.records]
    return metrics_response(rows, exists=log_holds_anything(page))


@router.get("/ray-dashboard")
def get_ray_dashboard() -> dict:
    """The live Ray dashboard's URL, or ``null`` when no cluster is up.

    Ray is one cluster per process and an agent-launched sweep initializes it inside the
    MCP server, so the URL comes off the state file that process wrote, not from anything
    in this one. Not scoped to a sweep for the same reason: there is one cluster, shared.
    """
    from tcip_mcp.pipelines.training.hpo import read_ray_dashboard

    state = read_ray_dashboard()
    return {"url": state["url"] if state else None}


def _trial_tb_key(sweep_id: str, trial_id: str) -> str:
    """The ``tensorboard_manager`` process key for one trial's TensorBoard."""
    return f"sweep_{sweep_id}_trial_{trial_id}"


def _link_dir(link: Path, target: Path) -> None:
    """Point ``link`` at ``target`` as a directory, however this machine allows it.

    A symlink needs ``SeCreateSymbolicLinkPrivilege`` on Windows (developer mode, or admin) and
    plenty of machines this platform runs on won't have either; a junction is the unprivileged
    NTFS equivalent for directories and is the fallback everywhere ``os.symlink`` refuses.
    """
    import os
    import subprocess

    try:
        os.symlink(target, link, target_is_directory=True)
        return
    except OSError:
        pass
    result = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(link), str(target)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise OSError(f"could not link {link} -> {target}: {result.stderr.strip()}")


def _trial_view_dir(sweep_id: str, *, root: Optional[str] = None) -> Path:
    """Where this sweep's clean-named trial links live, apart from the real trial dirs.

    ``root`` (the sweep's own launch root, from :func:`_sweep_launch_root`) wins when given,
    so the link farm always lands beside the sweep's own trial directories rather than inside
    whichever project this process is currently pinned to; omitted, this falls back to the
    current platform root, the historical behaviour for a sweep the registry has forgotten.
    """
    from tcip_mcp.project_paths import resolve_state

    if root is not None:
        return Path(root) / ".tcip" / "state" / "tensorboard_views" / sweep_id
    return resolve_state(Path(".tcip") / "state" / "tensorboard_views" / sweep_id)


def _ensure_trial_view(sweep_id: str, sweep_root: Path, *, root: Optional[str] = None) -> Path:
    """A directory where every trial with a tensorboard dir today is linked under its bare
    ``trial_<id>`` name, so TensorBoard's own per-run picker shows that instead of
    ``trial_<id>\\tensorboard`` -- the leaf-directory name TensorBoard would otherwise read off
    ``sweep_root`` directly, since each trial nests its event files one level down.

    Only adds links; never removes one, so a run open in a browser tab never has its link pulled
    out from under it. Existing links are left alone -- a trial's own tensorboard dir, once
    created, is never moved -- and TensorBoard's own ``--reload_interval`` picks up a link added
    after it already started, the same as it would a new subdirectory.
    """
    view = _trial_view_dir(sweep_id, root=root)
    view.mkdir(parents=True, exist_ok=True)
    for trial_dir in sorted(sweep_root.iterdir()):
        if not trial_dir.is_dir() or not trial_dir.name.startswith(_TRIAL_DIR_PREFIX):
            continue
        tb_dir = trial_dir / "tensorboard"
        if not tb_dir.is_dir():
            continue
        link = view / trial_dir.name
        if link.exists():
            continue
        try:
            _link_dir(link, tb_dir)
        except OSError:
            logger.warning("could not link %s for the sweep TensorBoard view", trial_dir.name, exc_info=True)
    return view


@router.post("/sweeps/{sweep_id}/tensorboard")
def launch_sweep_tensorboard(sweep_id: str, payload: EmptyBodyPayload) -> dict:
    """Start (or reuse) a TensorBoard over the whole sweep, one run per trial.

    Rooted at a clean-named link farm (see ``_ensure_trial_view``) rather than the sweep
    directory itself, so TensorBoard's own run picker reads ``trial_<id>`` instead of the
    nested ``trial_<id>\\tensorboard`` its default directory-name-as-run-name behavior would
    otherwise show, and a breeder can toggle trials on and off there rather than reading all of
    them, which stops being legible somewhere well short of a hundred-trial sweep.
    """
    from tcip_mcp.pipelines.training.tensorboard_manager import launch_tensorboard

    root = _sweep_launch_root(sweep_id)
    view = _ensure_trial_view(sweep_id, _sweep_root(sweep_id), root=root)
    return launch_tensorboard(str(view), run_id=f"sweep_{sweep_id}")


@router.post("/sweeps/{sweep_id}/trials/{trial_id}/tensorboard")
def launch_trial_tensorboard(sweep_id: str, trial_id: str, payload: EmptyBodyPayload) -> dict:
    """Start (or reuse) a TensorBoard over a single trial's own log directory."""
    from tcip_mcp.pipelines.training.tensorboard_manager import launch_tensorboard
    from tcip_web.paths import safe_join

    root = _sweep_root(sweep_id)
    try:
        logdir = safe_join(root, f"{_TRIAL_DIR_PREFIX}{trial_id}", "tensorboard")
    except ValueError as exc:
        raise HTTPException(400, f"invalid trial_id: {trial_id}") from exc
    return launch_tensorboard(str(logdir), run_id=_trial_tb_key(sweep_id, trial_id))


@router.post("/sweeps/{sweep_id}/trials/{trial_id}/tensorboard/stop")
def stop_trial_tensorboard(sweep_id: str, trial_id: str, payload: EmptyBodyPayload) -> dict:
    """Stop the TensorBoard serving one trial.

    TensorBoards share a bounded port range, so a session that opens one per trial and
    closes none exhausts it; the GUI stops a trial's TensorBoard when it moves off it.
    """
    from tcip_mcp.pipelines.training.tensorboard_manager import stop_tensorboard

    return stop_tensorboard(run_id=_trial_tb_key(sweep_id, trial_id))
