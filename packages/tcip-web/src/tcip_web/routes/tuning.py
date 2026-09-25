"""HPO / Tuning routes: relaunch + cancel + list + per-trial visibility.

Sweeps reach this surface two ways: launched here over HTTP (tracked in memory for as long as this
process lives), or launched through the ``run_hyperparameter_search`` MCP tool, which this process
never sees. The durable source for both is the ``manifest.json`` that ``run_hyperparameter_search``
stamps under the sweep's own directory when the sweep starts, so the listing below reads disk and
overlays the in-memory jobs.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from tcip_mcp.web_client import HPO_SWEEPS, current_root

if TYPE_CHECKING:
    from tcip_mcp.tools.training_tools import SeedAxisRefusal
from tcip_web import jobstore
from tcip_web.routes._body_common import EmptyBodyPayload
from tcip_web.routes._metrics_common import metrics_response

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/tuning", tags=["tuning"])

_TRIAL_DIR_PREFIX = "trial_"


def _current_root() -> str:
    return current_root()


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
    """The config-picker projection of a sweep manifest: search shape, relaunchability, whether a
    cancel has been requested, and which sweep (if any) it was relaunched from.

    ``relaunchable`` and ``reason`` are :func:`_relaunch_refusal`'s own absence and presence,
    ``reason`` never joined with the refusal's ``remedy``. ``redraw_within_selection`` is the
    recorded ``base_config``'s own ``data.split.redraw_within_selection``.

    An empty ``manifest`` (no manifest exists yet) is never relaunchable and carries no reason.
    """
    if not manifest:
        return {
            "n_trials": None, "search_alg": None, "scheduler": None, "param_space_keys": [],
            "relaunchable": False, "reason": None, "cancel_requested": False,
            "relaunched_from": None, "split_draws": None, "redraw_within_selection": False,
        }
    refusal = _relaunch_refusal(manifest)
    base_split = manifest["base_config"].get("data", {}).get("split", {})
    return {
        "n_trials": manifest["n_trials"],
        "search_alg": manifest["search_alg"],
        "scheduler": manifest["scheduler"],
        "param_space_keys": sorted(manifest["param_space"] or {}),
        "relaunchable": refusal is None,
        "reason": refusal.reason if refusal is not None else None,
        "cancel_requested": manifest["cancel_requested"],
        "relaunched_from": manifest["relaunched_from"],
        "split_draws": manifest["split_draws"],
        "redraw_within_selection": bool(base_split.get("redraw_within_selection")),
    }


def _persisted_summary(job: HPOJob) -> dict:
    """The registry's own persisted fields for one job, with no manifest read: what
    :class:`~tcip_web.jobstore.JobRegistry` writes to ``.tcip/state/hpo_sweeps.json`` under its own
    lock, and reads back on rehydrate.
    """
    return {"sweep_id": job.sweep_id, "status": job.status,
            "error": job.error,
            "platform_root": job.platform_root}


def _driver_live(sweep_id: str) -> bool:
    """Whether this process's own worker thread for ``sweep_id`` is still running."""
    with _lock:
        thread = _workers.get(sweep_id)
        return thread is not None and thread.is_alive()


def _job_sweep_state(job: HPOJob, manifest: dict) -> str:
    """A live-registry job's derived liveness. The registry's own record wins once ``job.status``
    is itself a recorded done state; otherwise the manifest is the source when one exists, else the
    registry's own record. Either way the rule is
    :func:`~tcip_mcp.tools.training_tools.sweep_state`'s own: a recorded done state is trusted, a
    live worker thread reads ``running``, anything else reads ``interrupted``.
    """
    from tcip_mcp.experiments import _RECORDED_AS_DONE
    from tcip_mcp.tools.training_tools import TCIP_HEARTBEAT_STALE_SECONDS, sweep_state

    if job.status in _RECORDED_AS_DONE:
        source = {"status": job.status, "heartbeat": None}
    else:
        source = manifest if manifest else {"status": job.status, "heartbeat": None}
    return sweep_state(source, stale_seconds=TCIP_HEARTBEAT_STALE_SECONDS,
                       driver_live=_driver_live(job.sweep_id))


def _summary(job: HPOJob) -> dict:
    """One live sweep's listing row: the registry's own persisted fields plus the manifest
    projection, read fresh (outside the registry's own lock). ``status`` is the derived liveness
    (:func:`_job_sweep_state`).

    ``has_manifest`` reports whether ``run_hyperparameter_search`` has written this sweep's first
    manifest yet. ``relaunched_from`` falls back to the job's own in-memory record (set at
    relaunch) only before a manifest exists; once one exists, its own recorded field wins.
    """
    raw_manifest = _read_manifest(job.sweep_id, root=job.platform_root)
    manifest = raw_manifest or {}
    has_manifest = raw_manifest is not None
    status = _job_sweep_state(job, manifest)
    fields = _manifest_fields(manifest)
    if not has_manifest and job.relaunched_from:
        fields["relaunched_from"] = job.relaunched_from
    return {**_persisted_summary(job), **fields, "status": status, "has_manifest": has_manifest}


def _from_summary(s: dict) -> HPOJob:
    return HPOJob(
        sweep_id=s["sweep_id"], status=jobstore.rehydrated_status(s), error=s["error"],
        platform_root=s["platform_root"],
    )


_registry = jobstore.JobRegistry(
    HPO_SWEEPS, to_summary=_persisted_summary, from_summary=_from_summary, id_field="sweep_id",
)
"""The dict-plus-lock live registry for this route's own sweeps (see ``jobstore.JobRegistry``),
the shared home review.py's priority queue and inference.py's jobs adopt too. ``_lock`` below is
this registry's own lock, kept under this name since callers (tests among them, and
this module's own ``_workers`` guard) already reach into it directly."""

_lock = _registry.lock
_workers: dict[str, threading.Thread] = {}
"""Every sweep worker this process has spawned and not yet seen finish, by sweep id. Guarded
by ``_lock``, the same lock ``_registry`` takes for its own dict; the two are unrelated state
sharing one mutex, not a stated invariant between them."""


def wait_for_workers(*, timeout_s: float) -> tuple[str, ...]:
    """Join this module's sweep workers and return the sweeps still running when time ran out.
    ``timeout_s`` is required.
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
            "external": True,
            "has_manifest": True,
            **_manifest_fields(manifest)}


def _sweeps_dir() -> Path:
    from tcip_mcp.tools.training_tools import hpo_root

    return hpo_root()


def _read_manifest(sweep_id: str, *, root: Path | str | None = None) -> dict | None:
    """The sweep's manifest, or ``None`` if no sweep by that name is on disk under ``root``
    (default: the current platform root).

    ``sweep_id`` is untrusted, and the store's own key constructor refuses one that would address a
    record outside the HPO store. A manifest that will not decode is reported and then answered as
    absent.
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

    A job this process ran to completion itself carries its full result in memory; a job a restart
    rehydrated carries none, so its result is read off the disk manifest instead.
    """
    result = job.result
    if not result:
        manifest = _read_manifest(job.sweep_id, root=job.platform_root)
        if manifest is not None:
            result = manifest.get("result") or {}
    return {"sweep_id": job.sweep_id, "status": job.status, "error": job.error, "result": result}


def _sweep_launch_root(sweep_id: str) -> Optional[str]:
    """The root this sweep's own live registry entry says it launched under, or ``None`` when the
    registry has forgotten it (never launched here, or launched before a restart).
    """
    job = _registry.get(sweep_id)
    return job.platform_root if job is not None else None


def _sweep_root(sweep_id: str) -> Path:
    """The directory a sweep's trials live in, once a manifest proves the sweep exists: under its
    launch root when the live registry remembers it (:func:`_sweep_launch_root`), else under the
    current root. Never a path recorded inside the manifest.
    """
    from tcip_mcp.tools.training_tools import sweep_dir

    root = _sweep_launch_root(sweep_id)
    if _read_manifest(sweep_id, root=root) is None:
        raise HTTPException(404, f"sweep not found: {sweep_id}")
    return sweep_dir(sweep_id, root=root)


def _disk_sweeps(exclude: frozenset[str] = frozenset()) -> list[dict]:
    """Every sweep with a manifest under the HPO root, in directory-name order, skipping a
    directory named in ``exclude``.
    """
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
    """Write the live registry, grouped by each sweep's own launch root."""
    _registry.persist()


def rehydrate_for_current_root() -> None:
    """Merge this root's persisted sweeps, not already live, into memory via :func:`_from_summary`.

    A persisted non-terminal sweep is surfaced as ``interrupted``, with no result. Merges by sweep
    id, so it never displaces a sweep still live from another root, and bounds the dict afterwards
    the same way launching a sweep does.
    """
    _registry.rehydrate()


class RelaunchSweepPayload(BaseModel):
    study_name: str


@dataclass
class _RelaunchSpec:
    """Every ``run_hyperparameter_search`` argument a manifest holds, read once so the worker
    replays exactly what the manifest recorded.
    """

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
    split_draws: int
    split_draw_seeds: Optional[list[int]]
    search_seed: int
    trial_budget: Optional[int]


def _relaunch_refusal(manifest: dict) -> SeedAxisRefusal | None:
    """The refusal a relaunch of ``manifest`` meets, or ``None``: a caller-supplied
    ``data.split.seed`` axis at one draw (:func:`training_tools.caller_split_seed_refusal`).
    """
    from tcip_mcp.tools.training_tools import caller_split_seed_refusal

    return caller_split_seed_refusal(manifest["param_space"], manifest["split_draws"])


def _relaunch_spec(manifest: dict) -> _RelaunchSpec:
    """``manifest``'s own ``run_hyperparameter_search`` arguments, read as recorded."""
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
        split_draws=manifest["split_draws"],
        split_draw_seeds=manifest["split_draw_seeds"],
        search_seed=manifest["search_seed"],
        trial_budget=manifest["trial_budget"],
    )


def _worker(job: HPOJob, spec: _RelaunchSpec, output_dir: str, relaunched_from: str) -> None:
    """Run one relaunched sweep to completion off the request thread.

    ``job.platform_root``, resolved on the request thread at launch, is what :func:`_persist`
    groups this sweep's summary under. A returned ``{"error", "issues"}`` is a failed job carrying
    the issues, and a returned ``{"status": "canceled", ...}`` a canceled one.
    ``relaunched_from`` is the source study's own name, recorded on this sweep's manifest. Discards
    ``job.sweep_id``'s own pre-manifest launch mark on every exit.
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
            search_seed=spec.search_seed,
            trial_budget=spec.trial_budget,
            relaunched_from=relaunched_from,
        )
        if isinstance(res, dict) and res.get("status") == "canceled":
            # Checked before the "error" key below: a canceled result carries its own reason
            # under that same key, and must still read canceled, not failed, because of it.
            job.status = "canceled"
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
    """Relaunch a sweep from its own recorded manifest: no config, param space or path is ever
    submitted by the browser. The source manifest is read under this sweep's own launch root
    (:func:`_sweep_launch_root`). ``output_dir`` is always this request thread's own
    ``hpo_root()``, never a path the manifest carries. ``relaunched_from`` is recorded from the
    source manifest's own ``study_name``. Marks the new sweep id as launching, on this request
    thread, before the worker starts, so a cancel that arrives before ``run_hyperparameter_search``
    writes its own first manifest still reaches it.
    """
    from tcip_mcp.tools.training_tools import hpo_root, mark_sweep_launching

    manifest = _read_manifest(payload.study_name, root=_sweep_launch_root(payload.study_name))
    if manifest is None:
        raise HTTPException(404, f"sweep not found: {payload.study_name}")
    refusal = _relaunch_refusal(manifest)
    if refusal is not None:
        detail = refusal.reason if refusal.remedy is None else f"{refusal.reason} {refusal.remedy}"
        raise HTTPException(409, detail)

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
    live = [_summary(j) for j in _registry.list(current_root())]
    live_ids = frozenset(s["sweep_id"] for s in live)
    return {"sweeps": live + _disk_sweeps(exclude=live_ids)}


@router.get("/sweeps/{sweep_id}")
def get_sweep(sweep_id: str) -> dict:
    """One sweep by id: a live entry (whichever root it launched under) wins, else its manifest
    under the current root. ``status`` is the derived liveness in both branches (see
    :func:`_job_sweep_state`/:func:`_manifest_summary`); ``relaunched_from`` is a top-level field
    on both branches too, the manifest's own when one exists, else (live branch only, pre-manifest)
    the job's own in-memory record. ``has_manifest`` is always true on the disk branch and reflects
    whether ``run_hyperparameter_search`` has written one yet on the live branch. The disk branch
    is :func:`training_tools.read_sweep_from_disk`'s own result, so it also carries ``trials``,
    absent from the live branch.
    """
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
            manifest["relaunched_from"] if has_manifest else j.relaunched_from
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
    """The ``trial_<id>`` directories a sweep has produced so far, from the shared disk reader (see
    :func:`training_tools.read_sweep_from_disk`), under whichever root the sweep actually launched
    under.
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
    bytes that will not decode. A trial that has logged nothing and a trial with no log at all are
    the same answer.
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
    """The live Ray dashboard's URL, or ``null`` when no cluster is up, read off the state file the
    MCP server process wrote; not scoped to a sweep.
    """
    from tcip_mcp.pipelines.training.hpo import read_ray_dashboard

    state = read_ray_dashboard()
    return {"url": state["url"] if state else None}


def _trial_tb_key(sweep_id: str, trial_id: str) -> str:
    """The ``tensorboard_manager`` process key for one trial's TensorBoard."""
    return f"sweep_{sweep_id}_trial_{trial_id}"


def _link_dir(link: Path, target: Path) -> None:
    """Link ``link`` to ``target`` as a directory: a symlink, else a junction where ``os.symlink``
    refuses.
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
    """Where this sweep's clean-named trial links live, apart from the real trial dirs: under
    ``root`` (the sweep's own launch root) when given, else the current platform root.
    """
    from tcip_mcp.project_paths import resolve_state

    if root is not None:
        return Path(root) / ".tcip" / "state" / "tensorboard_views" / sweep_id
    return resolve_state(Path(".tcip") / "state" / "tensorboard_views" / sweep_id)


def _ensure_trial_view(sweep_id: str, sweep_root: Path, *, root: Optional[str] = None) -> Path:
    """A directory where every trial with a tensorboard dir today is linked under its bare
    ``trial_<id>`` name. Only adds links; never removes one.
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
    """Start (or reuse) a TensorBoard over the whole sweep, one run per trial, rooted at the
    :func:`_ensure_trial_view` link farm.
    """
    from tcip_mcp.pipelines.training.tensorboard_manager import launch_tensorboard

    root = _sweep_launch_root(sweep_id)
    view = _ensure_trial_view(sweep_id, _sweep_root(sweep_id), root=root)
    return launch_tensorboard(str(view), key=f"sweep_{sweep_id}")


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
    return launch_tensorboard(str(logdir), key=_trial_tb_key(sweep_id, trial_id))


@router.post("/sweeps/{sweep_id}/trials/{trial_id}/tensorboard/stop")
def stop_trial_tensorboard(sweep_id: str, trial_id: str, payload: EmptyBodyPayload) -> dict:
    """Stop the TensorBoard serving one trial."""
    from tcip_mcp.pipelines.training.tensorboard_manager import stop_tensorboard

    return stop_tensorboard(key=_trial_tb_key(sweep_id, trial_id))
