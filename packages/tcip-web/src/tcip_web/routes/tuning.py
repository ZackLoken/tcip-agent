"""HPO / Tuning routes: relaunch, cancel, list and per-trial visibility, each read off the sweep's
own directory under ``.tcip/hpo`` (``training_tools.sweep_record`` and ``read_sweep``).
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from tcip_mcp.experiments import TRIAL_DIR_PREFIX
from tcip_mcp.web_client import current_root

from tcip_web.routes._body_common import EmptyBodyPayload
from tcip_web.routes._metrics_common import metrics_response

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/tuning", tags=["tuning"])


@dataclass(frozen=True)
class _Launch:
    """A sweep this process relaunched: the root it launched under and the thread running it."""

    platform_root: str
    thread: threading.Thread


_launches: dict[str, _Launch] = {}
_lock = threading.Lock()


def wait_for_workers(*, timeout_s: float) -> tuple[str, ...]:
    """Join the threads of the sweeps this process relaunched and return the sweeps still running
    when time ran out. ``timeout_s`` is required.
    """
    with _lock:
        pending = list(_launches.items())
    deadline = time.monotonic() + timeout_s
    for _, launch in pending:
        launch.thread.join(max(0.0, deadline - time.monotonic()))
    return tuple(sweep_id for sweep_id, launch in pending if launch.thread.is_alive())


def _launch_root(sweep_id: str) -> Optional[str]:
    """The root a sweep this process relaunched launched under, or ``None`` (the current root)
    for one it did not launch."""
    with _lock:
        launch = _launches.get(sweep_id)
    return launch.platform_root if launch is not None else None


def _external(sweep_id: str) -> bool:
    """Whether ``sweep_id`` is a sweep this process did not launch."""
    with _lock:
        return sweep_id not in _launches


def _sweep_or_404(sweep_id: str):
    """The sweep ``sweep_id`` names, observed under its launch root (:func:`_launch_root`,
    ``training_tools.sweep_observation``), 404 when no sweep by that name is on disk there and 400
    for an id that is not a single directory name."""
    from tcip_store import BadKey

    from tcip_mcp.tools.training_tools import sweep_observation

    try:
        sweep = sweep_observation(sweep_id, root=_launch_root(sweep_id))
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
    """Relaunch a sweep from its own recorded input: no config, param space or path is ever
    submitted by the browser. The source is read under its launch root (:func:`_launch_root`);
    the new sweep's input is written (``training_tools.open_sweep``), meeting every refusal the
    source's launch passed (422 with the refusal), before its thread starts. Recorded as
    ``launched_by: {"launcher": "gui"}``."""
    from tcip_mcp.tools.training_tools import declare_launcher, open_sweep

    source = _sweep_or_404(payload.study_name).record
    root = current_root()
    with declare_launcher("gui"):
        opened = open_sweep(
            source["base_config"], source["param_space"], n_trials=source["n_trials"],
            search_alg=source["search_alg"], scheduler=source["scheduler"],
            grace_period=source["grace_period"], reduction_factor=source["reduction_factor"],
            warm_start=source["warm_start"], baseline_params=source["baseline_params"],
            max_concurrent=source["max_concurrent"],
            resources_per_trial=source["resources_per_trial"],
            study_name=f"hpo-{uuid.uuid4().hex[:8]}", split_draws=source["split_draws"],
            split_draw_seeds=source["split_draw_seeds"], search_seed=source["search_seed"],
            trial_budget=source["trial_budget"], relaunched_from=payload.study_name)
    if isinstance(opened, dict):
        raise HTTPException(422, detail=opened)
    thread = threading.Thread(target=_run, args=(opened,), daemon=True)
    with _lock:
        _launches[opened.directory.name] = _Launch(platform_root=root, thread=thread)
    thread.start()
    return {"status": "launched", "sweep_id": opened.directory.name}


@router.post("/sweeps/{sweep_id}/cancel")
def cancel_sweep_route(sweep_id: str, payload: EmptyBodyPayload) -> dict:
    """Request cooperative cancellation of a sweep (``cancel_hyperparameter_search``) under its
    launch root; its refusal answers 404."""
    from tcip_mcp.tools.training_tools import cancel_hyperparameter_search

    result = cancel_hyperparameter_search(sweep_id, root=_launch_root(sweep_id))
    if result.get("error"):
        raise HTTPException(404, result["error"])
    return result


@router.get("/sweeps")
def list_sweeps() -> dict:
    """Every sweep directory under the current root holding its input, in directory-name order,
    each by :func:`_sweep_fields`."""
    from tcip_mcp.experiments import SWEEP_FILE, observe, sweeps_dir

    root = sweeps_dir()
    if not root.is_dir():
        return {"sweeps": []}
    return {"sweeps": [_sweep_fields(observe(d, SWEEP_FILE))
                       for d in sorted(root.iterdir()) if (d / SWEEP_FILE).is_file()]}


@router.get("/sweeps/{sweep_id}")
def get_sweep(sweep_id: str) -> dict:
    """One sweep read whole (``training_tools.read_sweep``) under its launch root
    (:func:`_launch_root`), with whether this process did not launch it."""
    from tcip_mcp.tools.training_tools import read_sweep

    return {**read_sweep(_sweep_or_404(sweep_id)), "external": _external(sweep_id)}


def _sweep_root(sweep_id: str) -> Path:
    """The directory a sweep's trials live in under its launch root (:func:`_sweep_or_404`)."""
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
    from tcip_mcp.project_paths import project_state_dir

    return project_state_dir(root) / "tensorboard_views" / sweep_id


def _ensure_trial_view(sweep_id: str, sweep_root: Path, *, root: Optional[str] = None) -> Path:
    """A directory where every trial with a tensorboard dir today is linked under its bare
    ``trial_<id>`` name. Only adds links; never removes one.
    """
    view = _trial_view_dir(sweep_id, root=root)
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

    view = _ensure_trial_view(sweep_id, _sweep_root(sweep_id), root=_launch_root(sweep_id))
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
