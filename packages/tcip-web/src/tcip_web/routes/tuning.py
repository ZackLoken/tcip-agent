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
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from tcip_web.routes._metrics_common import read_metrics_file

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/tuning", tags=["tuning"])

_TRIAL_DIR_PREFIX = "trial_"


@dataclass
class HPOJob:
    sweep_id: str
    status: str = "pending"
    error: Optional[str] = None
    result: dict[str, Any] = field(default_factory=dict)
    thread: Optional[threading.Thread] = field(default=None, repr=False)


_sweeps: dict[str, HPOJob] = {}
_lock = threading.Lock()


def _summary(job: HPOJob) -> dict:
    return {"sweep_id": job.sweep_id, "status": job.status,
            "error": job.error, "has_result": bool(job.result)}


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


def _read_manifest(sweep_id: str) -> dict | None:
    """The sweep's manifest, or ``None`` if no sweep by that name is on disk.

    ``sweep_id`` is untrusted, and the store's own key constructor is what refuses one that
    would address a record outside the HPO store. A manifest that will not decode is reported
    and then answered as absent: the listing serves every other sweep rather than failing
    whole, and the sweep with the unreadable manifest is not presented as running.
    """
    from tcip_store import BadKey, DecodeError, store

    from tcip_mcp.tools.training_tools import sweep_manifest_key

    try:
        key = sweep_manifest_key(sweep_id)
    except BadKey as exc:
        raise HTTPException(400, f"invalid sweep_id: {sweep_id}") from exc
    try:
        manifest = store.read(key, default=None)
    except DecodeError:
        logger.warning("the manifest for sweep %s does not decode", sweep_id, exc_info=True)
        return None
    return manifest if isinstance(manifest, dict) else None


def _sweep_root(sweep_id: str) -> Path:
    """The directory a sweep's trials live in, once a manifest proves the sweep exists.

    The sweep's own resolved location is what this is, not a path recorded inside the
    manifest: an absolute path in a file is not a path this process should follow.
    """
    from tcip_mcp.tools.training_tools import sweep_dir

    if _read_manifest(sweep_id) is None:
        raise HTTPException(404, f"sweep not found: {sweep_id}")
    return sweep_dir(sweep_id)


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
    from tcip_web import jobstore
    with _lock:
        summaries = [_summary(j) for j in _sweeps.values()]
    jobstore.persist("hpo_sweeps", summaries)


def rehydrate() -> None:
    """Seed the sweep registry from the last persisted summaries after a restart.

    Worker threads are gone, so a persisted non-terminal sweep is dead: surfaced as
    ``interrupted``. Trial results aren't persisted, so a rehydrated sweep has no result.
    """
    from tcip_web import jobstore

    with _lock:
        if _sweeps:
            return
        for s in jobstore.load("hpo_sweeps"):
            sid = s.get("sweep_id")
            if not sid:
                continue
            status = s.get("status", "interrupted")
            if status not in jobstore.TERMINAL_STATUSES:
                status = "interrupted"
            _sweeps[sid] = HPOJob(sweep_id=sid, status=status, error=s.get("error"))


class LaunchHPOPayload(BaseModel):
    base_config: dict[str, Any]
    param_space: Optional[dict[str, Any]] = None
    n_trials: int = 5
    output_dir: str = ""
    search_alg: str = "random"
    scheduler: str = "asha"


def _worker(job: HPOJob, payload: LaunchHPOPayload) -> None:
    try:
        job.status = "running"
        _persist()
        from tcip_mcp.tools.training_tools import run_hpo

        res = run_hpo(
            base_config=payload.base_config,
            param_space=payload.param_space,
            n_trials=payload.n_trials,
            output_dir=payload.output_dir,
            search_alg=payload.search_alg,
            scheduler=payload.scheduler,
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
    from tcip_web import jobstore
    from tcip_web.paths import assert_path_allowed

    # Confine the client-supplied output dir when the server is locked down (no-op otherwise).
    # Data paths nested inside base_config are a documented follow-up (not top-level here).
    if payload.output_dir:
        try:
            assert_path_allowed(payload.output_dir)
        except ValueError as exc:
            raise HTTPException(403, str(exc)) from exc

    job = HPOJob(sweep_id=f"hpo-{uuid.uuid4().hex[:8]}")
    with _lock:
        _sweeps[job.sweep_id] = job
        jobstore.evict_terminal(_sweeps)  # bound the registry (drop oldest terminal sweeps)
    _persist()
    t = threading.Thread(target=_worker, args=(job, payload), daemon=True)
    job.thread = t
    t.start()
    return {"status": "launched", "sweep_id": job.sweep_id}


@router.get("/sweeps")
def list_sweeps() -> dict:
    """Sweeps this process launched, backfilled with the ones on disk.

    A sweep the agent launched exists only as a manifest, and a sweep launched here has a
    manifest too, so a live entry wins by sweep_id and each sweep is listed once.
    """
    with _lock:
        live = [_summary(j) for j in _sweeps.values()]
    live_ids = {s["sweep_id"] for s in live}
    return {"sweeps": live + [d for d in _disk_sweeps() if d["sweep_id"] not in live_ids]}


@router.get("/sweeps/{sweep_id}")
def get_sweep(sweep_id: str) -> dict:
    with _lock:
        j = _sweeps.get(sweep_id)
    if j is not None:
        return {
            "sweep_id": j.sweep_id,
            "status": j.status,
            "error": j.error,
            "result": j.result,
        }
    manifest = _read_manifest(sweep_id)
    if manifest is None:
        raise HTTPException(404, f"sweep not found: {sweep_id}")
    return {
        "sweep_id": manifest.get("study_name", sweep_id),
        "status": manifest.get("status", "unknown"),
        "error": manifest.get("error"),
        "result": manifest.get("result") or {},
        "manifest": manifest,
        "external": True,
    }


@router.get("/sweeps/{sweep_id}/trials")
def list_trials(sweep_id: str) -> dict:
    """The trial directories a sweep has produced so far.

    Ray names its own per-trial directories after the trainable, so only the
    ``trial_<id>`` dirs the platform writes are listed here.
    """
    from tcip_store import DecodeError, store

    from tcip_mcp.tools.training_tools import trial_config_key

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
            "has_metrics": (d / "metrics.jsonl").is_file(),
            "params": resolved.get("trial_params") or {},
            "unconsumed_params": resolved.get("unconsumed_params") or [],
        })
    return {"sweep_id": sweep_id, "trials": trials}


@router.get("/sweeps/{sweep_id}/trials/{trial_id}/metrics")
def get_trial_metrics(sweep_id: str, trial_id: str) -> dict:
    """Every metrics row one trial has written."""
    from tcip_web.paths import safe_join

    root = _sweep_root(sweep_id)
    try:
        path = safe_join(root, f"{_TRIAL_DIR_PREFIX}{trial_id}", "metrics.jsonl")
    except ValueError as exc:
        raise HTTPException(400, f"invalid trial_id: {trial_id}") from exc
    return read_metrics_file(path)


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


def _trial_view_dir(sweep_id: str) -> Path:
    """Where this sweep's clean-named trial links live, apart from the real trial dirs."""
    from tcip_mcp.project_paths import resolve_state

    return resolve_state(Path(".tcip") / "state" / "tensorboard_views" / sweep_id)


def _ensure_trial_view(sweep_id: str, sweep_root: Path) -> Path:
    """A directory where every trial with a tensorboard dir today is linked under its bare
    ``trial_<id>`` name, so TensorBoard's own per-run picker shows that instead of
    ``trial_<id>\\tensorboard`` -- the leaf-directory name TensorBoard would otherwise read off
    ``sweep_root`` directly, since each trial nests its event files one level down.

    Only adds links; never removes one, so a run open in a browser tab never has its link pulled
    out from under it. Existing links are left alone -- a trial's own tensorboard dir, once
    created, is never moved -- and TensorBoard's own ``--reload_interval`` picks up a link added
    after it already started, the same as it would a new subdirectory.
    """
    view = _trial_view_dir(sweep_id)
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
def launch_sweep_tensorboard(sweep_id: str) -> dict:
    """Start (or reuse) a TensorBoard over the whole sweep, one run per trial.

    Rooted at a clean-named link farm (see ``_ensure_trial_view``) rather than the sweep
    directory itself, so TensorBoard's own run picker reads ``trial_<id>`` instead of the
    nested ``trial_<id>\\tensorboard`` its default directory-name-as-run-name behavior would
    otherwise show, and a breeder can toggle trials on and off there rather than reading all of
    them, which stops being legible somewhere well short of a hundred-trial sweep.
    """
    from tcip_mcp.pipelines.training.tensorboard_manager import launch_tensorboard

    view = _ensure_trial_view(sweep_id, _sweep_root(sweep_id))
    return launch_tensorboard(str(view), run_id=f"sweep_{sweep_id}")


@router.post("/sweeps/{sweep_id}/trials/{trial_id}/tensorboard")
def launch_trial_tensorboard(sweep_id: str, trial_id: str) -> dict:
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
def stop_trial_tensorboard(sweep_id: str, trial_id: str) -> dict:
    """Stop the TensorBoard serving one trial.

    TensorBoards share a bounded port range, so a session that opens one per trial and
    closes none exhausts it; the GUI stops a trial's TensorBoard when it moves off it.
    """
    from tcip_mcp.pipelines.training.tensorboard_manager import stop_tensorboard

    return stop_tensorboard(run_id=_trial_tb_key(sweep_id, trial_id))
