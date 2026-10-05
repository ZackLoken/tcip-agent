"""Training routes: launch or relaunch a run or sweep, list runs and sweeps, live metrics stream."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from tcip_store.errors import BadKey

from tcip_mcp.experiments import RunRow, TrainingDetail, TrainingListing, training_listing
from tcip_mcp.identity import actor
from tcip_web.routes._body_common import EmptyBodyPayload, PersonPayload
from tcip_web.state import store

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/training", tags=["training"])


@router.get("/configs/{experiment_id}/splits")
def list_split_choices_route(experiment_id: str) -> dict:
    """Every choice this config's own "Data" control offers a relaunch: the data section its
    launch stated, and every selection directory this project's own bound runs or the dataset's
    own splits directory hold, compatibility-checked as the launch itself would check them."""
    from tcip_mcp.tools.training_tools import list_split_choices

    result = list_split_choices(store.open_root(), experiment_id)
    if result.get("error"):
        raise HTTPException(404, result["error"])
    return result


class RelaunchPayload(BaseModel):
    relaunched_from: str
    selection_dir: str | None = None
    user: str


@router.post("/runs")
def relaunch_route(payload: RelaunchPayload) -> dict:
    """Start a new run or sweep from what a run or sweep of this project recorded, the one
    ``relaunched_from`` names, by the person ``user`` names: no config, param space or path is
    ever submitted by the browser. A run relaunches through ``launch_training`` with its launch
    config; a sweep through ``training_tools.launch_sweep``, off the request thread in its own
    worker process, the way a run's body runs.

    An optional ``selection_dir`` names a partition the browser picked instead of a run's own
    "As recorded" data section: the launch config then carries ``data.split`` replaced wholesale
    by ``{"selection_dir": chosen}``. A source naming nothing answers 404, a sweep source with a
    ``selection_dir`` 422, a refusal 422, and a launch whose audit line could not be written 409.
    """
    from tcip_mcp.audit import AuditEntryNotWritten
    from tcip_mcp.experiments import SWEEP_FILE, named_directory, observe
    from tcip_mcp.tools.training_tools import (
        candidate_config_with_selection, launch_sweep, launch_training,
    )
    from tcip_web.routes.audit_gap import audit_gap_409

    person = actor(payload.user)
    project = store.open_root()
    source = named_directory(payload.relaunched_from, project=project)
    if source is None:
        raise HTTPException(404, f"no run or sweep named {payload.relaunched_from}")
    try:
        if (source / SWEEP_FILE).is_file():
            if payload.selection_dir:
                raise HTTPException(422, "a sweep relaunches from its own recorded input")
            result = launch_sweep(project, source, actor=person)
        else:
            config = observe(source).record["config"]
            if payload.selection_dir:
                config = candidate_config_with_selection(config, payload.selection_dir)
            result = launch_training(project, config, relaunched_from=source.name, actor=person)
    except AuditEntryNotWritten as exc:
        raise audit_gap_409(exc, exc.arguments) from exc
    if result.get("error"):
        raise HTTPException(422, detail=result)
    return result


@router.get("/runs")
def list_runs_route() -> TrainingListing:
    """Every run and sweep of the open project (``experiments.training_listing``)."""
    return training_listing(store.open_root())


@router.get("/runs/{experiment_id}")
def get_run(experiment_id: str) -> TrainingDetail:
    """One run, trial or sweep of the open project (``training_tools.training_detail``), 404 for
    an id naming none."""
    from tcip_mcp.tools.training_tools import training_detail

    detail = training_detail(store.open_root(), experiment_id)
    if detail is None:
        raise HTTPException(404, f"Run not found: {experiment_id}")
    return detail


@router.post("/runs/{experiment_id}/tensorboard")
def launch_run_tensorboard(experiment_id: str, payload: EmptyBodyPayload) -> dict:
    """Start (or reuse) a TensorBoard over the board (``experiments.board_of``) of the run, trial
    or sweep ``experiment_id`` names, keyed by that directory.

    A board holding no event file refuses with ``no_logs: True``, carrying the final status's
    error when one is written; a board with events whose run or sweep ended in error refuses
    with that error alone.
    """
    from tcip_mcp.experiments import board_of, named_directory, observe
    from tcip_mcp.pipelines.training.tensorboard_manager import launch_tensorboard

    directory = named_directory(experiment_id, project=store.open_root())
    if directory is None:
        raise HTTPException(404, f"Run not found: {experiment_id}")
    board = board_of(directory)
    error = observe(directory).error
    if not any(board.rglob("events.out.tfevents*")):
        raise HTTPException(
            404, {"error": error or f"run produced no logs: {experiment_id}", "no_logs": True})
    if error:
        raise HTTPException(404, error)
    return launch_tensorboard(str(board))


@router.post("/runs/{experiment_id}/cancel")
def cancel_run_route(experiment_id: str, payload: PersonPayload) -> dict:
    """Request graceful cancellation of the run, trial or sweep ``experiment_id`` names
    (``cancel_training``), by the person ``user`` names; its refusal answers 404."""
    from tcip_mcp.tools.training_tools import cancel_training

    result = cancel_training(store.open_root(), experiment_id, actor=actor(payload.user))
    if result.get("error"):
        raise HTTPException(404, result["error"])
    return result


class ExperimentComparePayload(BaseModel):
    experiment_ids: list[str]


@router.post("/compare")
def compare_runs_route(payload: ExperimentComparePayload) -> dict:
    from tcip_mcp.experiments import compare_experiments

    return compare_experiments(payload.experiment_ids, project=store.open_root())


class CompareBestPayload(BaseModel):
    experiment_ids: list[str]
    metric: str
    higher_is_better: bool | None = None
    include_unverified: bool = False


@router.post("/compare/best")
def compare_best_route(payload: CompareBestPayload) -> dict:
    """Rank the marked comparison's own registered checkpoints by one metric
    (:func:`~tcip_mcp.tools.model_tools.ranked_registered_model`).

    A registry index that will not decode answers 409 naming why; the ranking's own error dicts
    map to 422 with the whole dict as ``detail``. The answer is projected to name, experiment id,
    stamped metrics, source, the metric ranked by, the direction used and its source, and the
    exclusions.
    """
    from tcip_store import DecodeError

    from tcip_mcp.tools.model_tools import ranked_registered_model

    try:
        result = ranked_registered_model(
            store.open_root(), payload.metric, higher_is_better=payload.higher_is_better,
            include_unverified=payload.include_unverified, experiment_ids=payload.experiment_ids,
            tag=None)
    except DecodeError as exc:
        raise HTTPException(409, f"registry unreadable: {exc}") from exc
    if "error" in result:
        raise HTTPException(422, detail=result)

    return {
        "name": result["name"],
        "experiment_id": result["experiment_id"],
        "metrics": result["metrics"],
        "metrics_source": result["metrics_source"],
        "ranking_basis": result["ranking_basis"],
        "higher_is_better": result["higher_is_better"],
        "direction_source": result["direction_source"],
        "excluded_unverified": result["excluded_unverified"],
    }


@router.get("/metric-directions")
def metric_directions_route() -> dict:
    """Every metric name evaluation.py declares a ranking direction for, with that direction."""
    from tcip_mcp.pipelines.training.evaluation import HIGHER_IS_BETTER_BY_METRIC

    return {"higher_is_better": dict(HIGHER_IS_BETTER_BY_METRIC)}


# ── WebSocket live metrics ──────────────────────────────────────────────


class TrainingMetricFrame(BaseModel):
    """One epoch's whole row (``experiments.epoch_rows``), pushed each time a row of that epoch
    is appended."""

    type: Literal["metric"]
    experiment_id: str
    row: dict


class TrainingStatusFrame(BaseModel):
    """The terminal frame: ``status`` carries the run's row for a run this process can still
    identify, ``error`` is set instead when it cannot."""

    type: Literal["status"]
    experiment_id: str
    status: RunRow | None
    error: str | None


async def _stream_metrics(
    ws: WebSocket, project: Path, experiment_id: str, poll_seconds: float = 1.0
) -> None:
    """Push a run's metrics, a sweep's trial included, to the browser as they are appended: each
    epoch a new row touches is sent again whole (``experiments.epoch_rows`` over every row read).

    Each tick observes the run once (``experiments.observe``) and reads its log from a byte-offset
    cursor (``experiments.read_rows``), so it reads only what was appended since the last tick and
    a row still being written is replayed once it is complete. Rows are read after the
    observation, so a run the observation found in a terminal state has every row sent before its
    one status frame (``experiments.run_summary`` over the rows sent), which ends the stream. An id naming no run
    directory under ``project`` ends the stream with one status frame naming it, and one that
    is not a single directory name raises ``BadKey``. Every read runs off the event loop.
    """
    from tcip_mcp.experiments import (
        EPOCH_KEY, TERMINAL_STATES, epoch_rows, find_observation, launch_declarations, observe,
        read_rows, run_name, run_summary,
    )

    observation = await asyncio.to_thread(find_observation, run_name(experiment_id),
                                          project=project)
    if observation is None:
        await ws.send_json(TrainingStatusFrame(
            type="status", experiment_id=experiment_id, status=None,
            error=f"Run not found: {experiment_id}").model_dump())
        return
    cursor = 0
    sent: list[dict] = []

    while True:
        rows, cursor = await asyncio.to_thread(read_rows, observation.metrics_log, after=cursor)
        sent += rows
        touched = [row.get(EPOCH_KEY) for row in rows]
        for row in epoch_rows(sent):
            if row.get(EPOCH_KEY) in touched:
                frame = TrainingMetricFrame(type="metric", experiment_id=experiment_id, row=row)
                await ws.send_json(frame.model_dump())
        if observation.state in TERMINAL_STATES:
            launches = await asyncio.to_thread(launch_declarations, project)
            status = await asyncio.to_thread(run_summary, observation, sent,
                                             launches.get(experiment_id))
            await ws.send_json(TrainingStatusFrame(
                type="status", experiment_id=experiment_id, status=status,
                error=None).model_dump())
            break
        await asyncio.sleep(poll_seconds)
        observation = await asyncio.to_thread(observe, observation.directory)


@router.websocket("/runs/{experiment_id}/stream")
async def training_stream_ws(websocket: WebSocket, experiment_id: str) -> None:
    """Tail ``experiment_id``'s metrics log, a run of the open project, and push new rows to the
    browser; closes with 1008 while no project is open."""
    project = store.project_root
    if project is None:
        await websocket.close(code=1008, reason="no project is open")
        return
    await websocket.accept()
    try:
        await _stream_metrics(websocket, project, experiment_id)
    except BadKey as exc:
        await websocket.close(code=1008, reason=str(exc))
    except WebSocketDisconnect:
        pass
    except Exception:
        logger.exception("training stream failed")
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
