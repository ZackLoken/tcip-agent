"""HPO / Tuning routes: relaunch, cancel, list and per-trial visibility, each read off the sweep's
own directory under ``.tcip/hpo`` (``training_tools.sweep_record`` and ``read_sweep``).
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from tcip_mcp.experiments import TRIAL_DIR_PREFIX
from tcip_mcp.identity import actor

from tcip_web.routes._body_common import EmptyBodyPayload, PersonPayload
from tcip_web.routes._metrics_common import metrics_response
from tcip_web.state import store

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/tuning", tags=["tuning"])


_launches: dict[tuple[Path, str], threading.Thread] = {}
"""The thread running each sweep this process relaunched, by its project and sweep name."""
_lock = threading.Lock()


def wait_for_workers(*, timeout_s: float) -> tuple[str, ...]:
    """Join the threads of the sweeps this process relaunched and return the sweeps still running
    when time ran out. ``timeout_s`` is required.
    """
    with _lock:
        pending = list(_launches.items())
    deadline = time.monotonic() + timeout_s
    for _, thread in pending:
        thread.join(max(0.0, deadline - time.monotonic()))
    return tuple(sweep_id for (_, sweep_id), thread in pending if thread.is_alive())


def _external(sweep_id: str) -> bool:
    """Whether ``sweep_id`` is a sweep of the open project this process did not launch."""
    with _lock:
        return (store.open_root(), sweep_id) not in _launches


def _sweep_or_404(sweep_id: str):
    """The sweep ``sweep_id`` names in the open project (``training_tools.sweep_observation``),
    404 when no sweep by that name is on disk there and 400 for an id that is not a single
    directory name."""
    from tcip_store import BadKey

    from tcip_mcp.tools.training_tools import sweep_observation

    try:
        sweep = sweep_observation(sweep_id, project=store.open_root())
    except BadKey as exc:
        raise HTTPException(400, f"invalid sweep_id: {sweep_id}") from exc
    if sweep is None:
        raise HTTPException(404, f"sweep not found: {sweep_id}")
    return sweep


def _sweep_fields(observation) -> dict:
    """The listing projection of an observed sweep's own record (``training_tools.sweep_record``):
    its status and error, its search shape, whether a cancel was requested, the sweep it
    relaunched, the base config's ``data.split.redraw_within_selection``, and whether this process
    did not launch it."""
    from tcip_mcp.tools.training_tools import sweep_record

    sweep = sweep_record(observation)
    sweep_input = sweep["input"]
    base_split = (sweep_input["base_config"].get("data") or {}).get("split") or {}
    return {
        "sweep_id": sweep["sweep_id"], "status": sweep["status"], "error": sweep["error"],
        "n_trials": sweep_input["n_trials"], "search_alg": sweep_input["search_alg"],
        "scheduler": sweep_input["scheduler"],
        "param_space_keys": sorted(sweep_input["param_space"] or {}),
        "cancel_requested": sweep["cancel_requested"],
        "relaunched_from": sweep_input["relaunched_from"],
        "split_draws": sweep_input["split_draws"],
        "redraw_within_selection": bool(base_split.get("redraw_within_selection")),
        "external": _external(sweep["sweep_id"]),
    }


class RelaunchSweepPayload(BaseModel):
    study_name: str
    user: str


def _run(opened) -> None:
    """Run one opened sweep (``training_tools.run_sweep``) to its final status off the request
    thread."""
    from tcip_mcp.tools.training_tools import run_sweep

    try:
        run_sweep(opened, auto_tensorboard=False)
    except Exception:
        logger.exception("HPO sweep %s failed", opened.directory.name)


@router.post("/sweeps")
def relaunch_sweep(payload: RelaunchSweepPayload) -> dict:
    """Relaunch a sweep of the open project from its own recorded input
    (``training_tools.reopen_sweep``), by the person ``user`` names: no config, param space or
    path is ever submitted by the browser. A source no sweep records answers 404, a refusal 422,
    and an opened sweep whose audit line could not be written 409, before its thread starts."""
    from tcip_mcp.audit import AuditEntryNotWritten
    from tcip_mcp.tools.training_tools import reopen_sweep
    from tcip_web.routes.audit_gap import audit_gap_409

    person = actor(payload.user)
    source = _sweep_or_404(payload.study_name)
    project = store.open_root()
    try:
        opened = reopen_sweep(project, source, actor=person)
    except AuditEntryNotWritten as exc:
        raise audit_gap_409(exc, exc.arguments) from exc
    if isinstance(opened, dict):
        raise HTTPException(422, detail=opened)
    thread = threading.Thread(target=_run, args=(opened,), daemon=True)
    with _lock:
        _launches[(project, opened.directory.name)] = thread
    thread.start()
    return {"status": "launched", "sweep_id": opened.directory.name}


@router.post("/sweeps/{sweep_id}/cancel")
def cancel_sweep_route(sweep_id: str, payload: PersonPayload) -> dict:
    """Request cooperative cancellation of a sweep of the open project
    (``cancel_hyperparameter_search``), by the person ``user`` names; its refusal answers 404."""
    from tcip_mcp.tools.training_tools import cancel_hyperparameter_search

    result = cancel_hyperparameter_search(store.open_root(), sweep_id,
                                          actor=actor(payload.user))
    if result.get("error"):
        raise HTTPException(404, result["error"])
    return result


@router.get("/sweeps")
def list_sweeps() -> dict:
    """Every sweep directory under the open project holding its input, in directory-name order,
    each by :func:`_sweep_fields`."""
    from tcip_mcp.experiments import SWEEP_FILE, observe, sweeps_dir

    root = sweeps_dir(store.open_root())
    if not root.is_dir():
        return {"sweeps": []}
    return {"sweeps": [_sweep_fields(observe(d, SWEEP_FILE))
                       for d in sorted(root.iterdir()) if (d / SWEEP_FILE).is_file()]}


@router.get("/sweeps/{sweep_id}")
def get_sweep(sweep_id: str) -> dict:
    """One sweep of the open project read whole (``training_tools.read_sweep``), with whether
    this process did not launch it."""
    from tcip_mcp.tools.training_tools import read_sweep

    return {**read_sweep(_sweep_or_404(sweep_id)), "external": _external(sweep_id)}


def _sweep_root(sweep_id: str) -> Path:
    """The directory a sweep's trials live in (:func:`_sweep_or_404`)."""
    return _sweep_or_404(sweep_id).directory


@router.get("/sweeps/{sweep_id}/trials")
def list_trials(sweep_id: str) -> dict:
    """The trial run directories a sweep has produced so far (``training_tools.read_sweep``)."""
    from tcip_mcp.tools.training_tools import read_sweep

    return {"sweep_id": sweep_id, "trials": read_sweep(_sweep_or_404(sweep_id))["trials"]}


@router.get("/sweeps/{sweep_id}/trials/{trial_id}/metrics")
def get_trial_metrics(sweep_id: str, trial_id: str) -> dict:
    """Every metrics row one trial has written (``experiments.observe`` of its directory), 404 for
    a trial the sweep has no run directory for. ``exists`` reports whether its log holds anything,
    a line still being appended included."""
    from tcip_mcp.experiments import RUN_FILE, observe, read_rows
    from tcip_web.paths import safe_join

    try:
        trial_dir = safe_join(_sweep_root(sweep_id), f"{TRIAL_DIR_PREFIX}{trial_id}")
    except ValueError as exc:
        raise HTTPException(400, f"invalid trial_id: {trial_id}") from exc
    if not (trial_dir / RUN_FILE).is_file():
        raise HTTPException(404, f"trial not found: {trial_id}")
    log = observe(trial_dir).metrics_log
    return metrics_response(read_rows(log)[0], exists=log.stat().st_size > 0)


@router.get("/ray-dashboard")
def get_ray_dashboard() -> dict:
    """The live Ray dashboard's URL for the open project, or ``null`` when no cluster is up, read
    off the state file the process that started it wrote; not scoped to a sweep.
    """
    from tcip_mcp.pipelines.training.hpo import read_ray_dashboard

    state = read_ray_dashboard(store.open_root())
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


def _ensure_trial_view(sweep_id: str, sweep_root: Path, *, project: Path) -> Path:
    """A directory under ``project``'s state, apart from the real trial dirs, where every trial
    with a tensorboard dir today is linked under its bare ``trial_<id>`` name. Only adds links;
    never removes one.
    """
    from tcip_mcp.project_paths import project_state_dir

    view = project_state_dir(project) / "tensorboard_views" / sweep_id
    view.mkdir(parents=True, exist_ok=True)
    for trial_dir in sorted(sweep_root.iterdir()):
        if not trial_dir.is_dir() or not trial_dir.name.startswith(TRIAL_DIR_PREFIX):
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

    view = _ensure_trial_view(sweep_id, _sweep_root(sweep_id), project=store.open_root())
    return launch_tensorboard(str(view), key=f"sweep_{sweep_id}")


@router.post("/sweeps/{sweep_id}/trials/{trial_id}/tensorboard")
def launch_trial_tensorboard(sweep_id: str, trial_id: str, payload: EmptyBodyPayload) -> dict:
    """Start (or reuse) a TensorBoard over a single trial's own log directory."""
    from tcip_mcp.pipelines.training.tensorboard_manager import launch_tensorboard
    from tcip_web.paths import safe_join

    root = _sweep_root(sweep_id)
    try:
        logdir = safe_join(root, f"{TRIAL_DIR_PREFIX}{trial_id}", "tensorboard")
    except ValueError as exc:
        raise HTTPException(400, f"invalid trial_id: {trial_id}") from exc
    return launch_tensorboard(str(logdir), key=_trial_tb_key(sweep_id, trial_id))


@router.post("/sweeps/{sweep_id}/trials/{trial_id}/tensorboard/stop")
def stop_trial_tensorboard(sweep_id: str, trial_id: str, payload: EmptyBodyPayload) -> dict:
    """Stop the TensorBoard serving one trial."""
    from tcip_mcp.pipelines.training.tensorboard_manager import stop_tensorboard

    return stop_tensorboard(key=_trial_tb_key(sweep_id, trial_id))
