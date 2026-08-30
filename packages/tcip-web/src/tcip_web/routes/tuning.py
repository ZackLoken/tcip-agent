"""HPO / Tuning routes: launch + list + per-trial visibility.

Sweeps reach this surface two ways: launched here over HTTP (tracked in memory for as long
as this process lives), or launched by the agent straight through the ``run_hpo`` MCP tool,
which this process never sees. The durable source for both is the ``manifest.json`` that
``run_hpo`` stamps under the sweep's own directory when the sweep starts, so the listing
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


def _summary(job: HPOJob) -> dict:
    return {"sweep_id": job.sweep_id, "status": job.status,
            "error": job.error, "has_result": bool(job.result),
            "platform_root": job.platform_root}


def _from_summary(s: dict, root: str) -> HPOJob:
    return HPOJob(
        sweep_id=s["sweep_id"], status=jobstore.rehydrated_status(s), error=s.get("error"),
        platform_root=s.get("platform_root") or root,
    )


_registry = jobstore.JobRegistry(
    HPO_REGISTRY, to_summary=_summary, from_summary=_from_summary, id_field="sweep_id",
)
"""The dict-plus-lock live registry for this route's own sweeps (see ``jobstore.JobRegistry``),
the shared home review.py's priority queue and inference.py's jobs adopt too. ``_sweeps``/
``_lock`` below are this registry's own dict and lock, bound under their historical names since
callers (tests among them) already reach into them directly."""

_sweeps: dict[str, HPOJob] = _registry.jobs
_lock = _registry.lock
_workers: dict[str, threading.Thread] = {}
"""Every sweep worker this process has spawned and not yet seen finish, by sweep id."""


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


def _summary(job: HPOJob) -> dict:
    return {"sweep_id": job.sweep_id, "status": job.status,
            "error": job.error, "has_result": bool(job.result),
            "platform_root": job.platform_root}


def _manifest_summary(manifest: dict) -> dict:
    """A sweep read off disk, in the same shape as an in-memory job's summary."""
    return {"sweep_id": manifest.get("study_name", ""),
            "status": manifest.get("status", "unknown"),
            "error": manifest.get("error"),
            "has_result": bool(manifest.get("result")),
            "external": True}


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


_STUDY_RESULT_FIELDS = ("all_trials", "search_alg", "scheduler", "warm_start", "baseline_params")
"""The study result's own fields, absent from the manifest's completion projection."""


def _read_study_result(sweep_id: str, *, root: Path | str | None = None) -> dict | None:
    """A finished sweep's full study-result record, or ``None`` when it has none: an old sweep
    from before this record existed, a failed sweep (never written), or one whose record won't
    decode. ``sweep_id`` is untrusted; a name that would address a record outside the HPO store
    is answered the same way, absent rather than raised."""
    from tcip_store import BadKey, DecodeError, store

    from tcip_mcp.tools.training_tools import study_result_key

    try:
        key = study_result_key(sweep_id, root=root)
    except BadKey:
        return None
    try:
        result = store.read(key, default=None)
    except DecodeError:
        logger.warning("the study result for sweep %s does not decode", sweep_id, exc_info=True)
        return None
    return result if isinstance(result, dict) else None


def _enrich_with_study_result(response: dict, sweep_id: str, *, root: Optional[str]) -> dict:
    """Layer the study result's own fields onto ``response["result"]`` for a completed sweep,
    read through the store and never fabricated: a sweep whose study result is absent (or
    already carries these fields, the common case for a sweep this process just ran) is served
    exactly as it already was."""
    if response.get("status") != "completed":
        return response
    result = response.get("result") or {}
    if "all_trials" in result:
        return response
    study_result = _read_study_result(sweep_id, root=root)
    if study_result is None:
        return response
    response["result"] = {
        **result,
        **{k: study_result[k] for k in _STUDY_RESULT_FIELDS if k in study_result},
    }
    return response


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


def _disk_sweeps() -> list[dict]:
    """Every sweep with a manifest under the HPO root, in directory-name order."""
    root = _sweeps_dir()
    if not root.is_dir():
        return []
    found: list[dict] = []
    for d in sorted(root.iterdir()):
        if not d.is_dir():
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


class LaunchHPOPayload(BaseModel):
    base_config: dict[str, Any]
    param_space: Optional[dict[str, Any]] = None
    n_trials: int = 5
    output_dir: str = ""
    search_alg: str = "random"
    scheduler: str = "asha"


def _worker(job: HPOJob, payload: LaunchHPOPayload, output_dir: str) -> None:
    """Run one sweep to completion off the request thread.

    ``job.platform_root``, resolved on the request thread at launch, is what
    :func:`_persist` groups this sweep's summary under, so this thread's writes land under
    the root that launch named rather than under whatever the environment names later.
    """
    try:
        job.status = "running"
        _persist()
        from tcip_mcp.tools.training_tools import run_hpo

        res = run_hpo(
            base_config=payload.base_config,
            param_space=payload.param_space,
            n_trials=payload.n_trials,
            output_dir=output_dir,
            search_alg=payload.search_alg,
            scheduler=payload.scheduler,
            study_name=job.sweep_id,
            auto_tensorboard=False,
        )
        job.result = res if isinstance(res, dict) else {"raw": res}
        job.status = "completed"
    except Exception as exc:
        logger.exception("HPO sweep %s failed", job.sweep_id)
        job.status = "failed"
        job.error = str(exc)
    finally:
        _persist()


@router.post("/launch")
def launch_hpo(payload: LaunchHPOPayload) -> dict:
    from tcip_mcp.tools.training_tools import hpo_root

    from tcip_web.paths import assert_path_allowed

    # Confine the client-supplied output dir when the server is locked down (no-op otherwise).
    # Data paths nested inside base_config are a documented follow-up (not top-level here).
    if payload.output_dir:
        try:
            assert_path_allowed(payload.output_dir)
        except ValueError as exc:
            raise HTTPException(403, str(exc)) from exc

    output_dir = payload.output_dir or str(hpo_root())

    job = HPOJob(sweep_id=f"hpo-{uuid.uuid4().hex[:8]}")
    _registry.register(job.sweep_id, job, job_root=job.platform_root)
    t = threading.Thread(target=_worker, args=(job, payload, output_dir), daemon=True)
    with _lock:
        for sweep_id in [sid for sid, done in _workers.items() if not done.is_alive()]:
            _workers.pop(sweep_id, None)
        _workers[job.sweep_id] = t
    t.start()
    return {"status": "launched", "sweep_id": job.sweep_id}


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
    live_ids = {s["sweep_id"] for s in live}
    return {"sweeps": live + [d for d in _disk_sweeps() if d["sweep_id"] not in live_ids]}


@router.get("/sweeps/{sweep_id}")
def get_sweep(sweep_id: str) -> dict:
    """One sweep by id: a live entry (whichever root it launched under) wins, else its
    manifest under the current root."""
    from tcip_web import jobstore
    j = _registry.get(sweep_id)
    if j is not None:
        response = (
            _terminal_response(j) if j.status in jobstore.TERMINAL_STATUSES
            else {"sweep_id": j.sweep_id, "status": j.status, "error": j.error, "result": j.result}
        )
        return _enrich_with_study_result(response, sweep_id, root=j.platform_root)
    manifest = _read_manifest(sweep_id)
    if manifest is None:
        raise HTTPException(404, f"sweep not found: {sweep_id}")
    response = {
        "sweep_id": manifest.get("study_name", sweep_id),
        "status": manifest.get("status", "unknown"),
        "error": manifest.get("error"),
        "result": manifest.get("result") or {},
        "manifest": manifest,
        "external": True,
    }
    return _enrich_with_study_result(response, sweep_id, root=None)


def _log_holds_anything(page) -> bool:
    """Whether a metrics log holds anything at all: rows, a torn tail, undecodable bytes, or
    entries at a schema_version this reader does not accept."""
    return bool(page.records or page.torn_tail or page.corrupt or page.version_refused)


@router.get("/sweeps/{sweep_id}/trials")
def list_trials(sweep_id: str) -> dict:
    """The trial directories a sweep has produced so far.

    Ray names its own per-trial directories after the trainable, so only the
    ``trial_<id>`` dirs the platform writes are listed here.
    """
    from tcip_store import DecodeError, read_log, store

    from tcip_mcp.tools.training_tools import trial_config_key, trial_metrics_key

    root = _sweep_root(sweep_id)
    trials: list[dict] = []
    for d in sorted(root.iterdir()):
        if not d.is_dir() or not d.name.startswith(_TRIAL_DIR_PREFIX):
            continue
        try:
            resolved = store.read(trial_config_key(root, d.name), default={})
        except DecodeError:
            # A trial whose own record is unreadable is still listed, with no params to show.
            logger.warning("the resolved config for %s does not decode", d.name, exc_info=True)
            resolved = {}
        if not isinstance(resolved, dict):
            resolved = {}
        trials.append({
            "trial_id": d.name[len(_TRIAL_DIR_PREFIX):],
            "has_metrics": _log_holds_anything(read_log(trial_metrics_key(root, d.name))),
            "params": resolved.get("trial_params") or {},
            "unconsumed_params": resolved.get("unconsumed_params") or [],
        })
    return {"sweep_id": sweep_id, "trials": trials}


@router.get("/sweeps/{sweep_id}/trials/{trial_id}/metrics")
def get_trial_metrics(sweep_id: str, trial_id: str) -> dict:
    """Every metrics row one trial has written.

    ``exists`` reports whether the log holds anything: rows, an entry still being appended, or
    bytes that will not decode. A trial that has logged nothing and a trial with no log at all
    are the same answer to the caller, which is what the Tuning view asks.
    """
    from tcip_store import BadKey, read_log

    from tcip_mcp.tools.training_tools import trial_metrics_key

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
    return metrics_response(rows, exists=_log_holds_anything(page))


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
