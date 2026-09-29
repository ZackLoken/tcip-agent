"""Training routes: launchable configs, launch/relaunch, list runs, live metrics stream."""

from __future__ import annotations

import asyncio
import logging
from typing import Literal

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from tcip_store.errors import BadKey

from tcip_web.paths import assert_project_root_allowed
from tcip_web.routes._body_common import EmptyBodyPayload

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/training", tags=["training"])


@router.get("/configs")
def list_configs_route() -> dict:
    """Every experiment in this project a run can be started or relaunched from."""
    from tcip_mcp.tools.training_tools import list_launchable_configs

    return {"configs": list_launchable_configs()}


@router.get("/configs/{experiment_id}/splits")
def list_split_choices_route(experiment_id: str) -> dict:
    """Every choice this config's own "Data" control offers a relaunch: the data section its
    launch stated, and every selection directory this project's own bound runs or the dataset's
    own splits directory hold, compatibility-checked as the launch itself would check them."""
    from tcip_mcp.tools.training_tools import list_split_choices

    result = list_split_choices(experiment_id)
    if result.get("error"):
        raise HTTPException(404, result["error"])
    return result


class RelaunchConfigPayload(BaseModel):
    experiment_id: str
    selection_dir: str | None = None


@router.post("/runs")
def relaunch_config_route(payload: RelaunchConfigPayload) -> dict:
    """Start a new run from the config a run of this project was launched with, as a fresh run id
    with the picked one as parent: no config, param space or path is ever submitted by the
    browser.

    An optional ``selection_dir`` names a partition the browser picked instead of the launch's own
    "As recorded" data section: checked against this same config's own
    :func:`~tcip_mcp.tools.training_tools.list_split_choices` listing (an enabled offer or 409)
    through ``tcip_store.canonical_path``, so a symlinked or differently cased spelling of an
    offered directory is admitted. The launch config then carries ``data.split`` replaced
    wholesale by ``{"selection_dir": chosen}``.

    The launch is wrapped in ``declare_launcher("gui")``, so the run's ``run.json`` records
    ``launched_by: {"launcher": "gui"}``, whatever client posted here.
    """
    from tcip_mcp.tools.training_tools import (
        candidate_config_with_selection, declare_launcher, launch_training, list_split_choices,
        stated_config,
    )
    from tcip_store import canonical_path

    config = stated_config(payload.experiment_id)
    if config is None:
        raise HTTPException(404, f"no launchable config named {payload.experiment_id}")

    if payload.selection_dir:
        choices = list_split_choices(payload.experiment_id)
        enabled = {canonical_path(s["selection_dir"])
                   for s in choices.get("selections", []) if s.get("enabled")}
        if canonical_path(payload.selection_dir) not in enabled:
            raise HTTPException(
                409, f"{payload.selection_dir!r} is not an offered partition for "
                     f"{payload.experiment_id}",
            )
        config = candidate_config_with_selection(config, payload.selection_dir)
    try:
        with declare_launcher("gui"):
            result = launch_training(config, parent_experiment=payload.experiment_id)
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc
    if result.get("error"):
        raise HTTPException(422, detail=result)
    return result


@router.get("/runs")
def list_runs_route() -> dict:
    """Every training run directory of the project (``training_tools._all_training_runs``)."""
    from tcip_mcp.tools.training_tools import _all_training_runs

    return {"runs": _all_training_runs()}


@router.get("/runs/{experiment_id}")
def get_run(experiment_id: str) -> dict:
    from tcip_mcp.tools.training_tools import monitor_training

    return monitor_training(experiment_id)


@router.post("/runs/{experiment_id}/tensorboard")
def launch_run_tensorboard(experiment_id: str, payload: EmptyBodyPayload) -> dict:
    """Start (or reuse) a TensorBoard serving this run's log directory, keyed by that directory.

    A run with no recorded output directory, or whose output directory's ``tensorboard``
    subdirectory holds no event file, refuses with ``no_logs: True``. A run whose own status
    already carries an error is checked for events first: with none it reads as no-logs (with that
    reason attached); with real event files it keeps the plain refusal.
    """
    from pathlib import Path

    from tcip_mcp.pipelines.training.tensorboard_manager import launch_tensorboard
    from tcip_mcp.tools.training_tools import monitor_training

    status = monitor_training(experiment_id)
    if "status" not in status:
        raise HTTPException(404, status.get("error") or f"Run not found: {experiment_id}")
    output_dir = status.get("output_dir")
    tb_dir = Path(f"{output_dir}/tensorboard") if output_dir else None
    has_events = bool(tb_dir is not None and tb_dir.is_dir()
                       and any(tb_dir.glob("events.out.tfevents*")))
    error = status.get("error")
    if error and has_events:
        raise HTTPException(404, error)
    if not has_events:
        raise HTTPException(
            404,
            {"error": error or f"run produced no logs: {experiment_id}", "no_logs": True},
        )
    return launch_tensorboard(f"{output_dir}/tensorboard")


@router.post("/runs/{experiment_id}/cancel")
def cancel_run_route(experiment_id: str, payload: EmptyBodyPayload) -> dict:
    """Request graceful cancellation of a running run (stops at the next batch boundary).

    Wraps the ``cancel_training`` MCP tool: the trainer still writes ``model_final.pt``
    so partial progress is recoverable. Status flips to 'canceled' asynchronously, unless the
    run's divergence verdict lands first, in which case it ends 'failed' instead.
    """
    from tcip_mcp.tools.training_tools import cancel_training

    result = cancel_training(experiment_id)
    if result.get("error"):
        raise HTTPException(404, result["error"])
    return result


class ExperimentComparePayload(BaseModel):
    experiment_ids: list[str]


@router.post("/compare")
def compare_runs_route(payload: ExperimentComparePayload) -> dict:
    from tcip_mcp.experiments import compare_experiments

    return compare_experiments(payload.experiment_ids)


class CompareBestPayload(BaseModel):
    experiment_ids: list[str]
    metric: str
    higher_is_better: bool | None = None
    include_unverified: bool = False


@router.post("/compare/best")
def compare_best_route(payload: CompareBestPayload) -> dict:
    """Rank the marked comparison's own registered checkpoints by one metric.

    Narrowed to the marked experiments. A project with no registered checkpoint
    (``model_registry.registered_entries``) answers 404; a registry index that will not decode or
    is not the entries mapping answers 409 naming why. The tool's own error dicts map to 422 with
    the whole dict as ``detail``. The answer is projected to name, experiment id, stamped
    metrics, source, the direction used and its source, and the exclusions.
    """
    from tcip_store.errors import DecodeError, SchemaVersionRefused

    from tcip_mcp.model_registry import RegistryVersionRefused, registered_entries
    from tcip_mcp.project_paths import platform_state_root
    from tcip_mcp.tools.model_tools import rank_registered_models

    try:
        entries = registered_entries(platform_state_root())
    except (DecodeError, RegistryVersionRefused, SchemaVersionRefused) as exc:
        raise HTTPException(409, f"registry unreadable: {exc}") from exc
    if not entries:
        raise HTTPException(404, "no model registry in this project")

    result = rank_registered_models(
        metric=payload.metric, higher_is_better=payload.higher_is_better,
        include_unverified=payload.include_unverified, experiment_ids=payload.experiment_ids,
    )
    if "error" in result:
        raise HTTPException(422, detail=result)

    return {
        "name": result["name"],
        "experiment_id": result["experiment_id"],
        "metrics": result["metrics"],
        "metrics_source": result["metrics_source"],
        "higher_is_better": result["higher_is_better"],
        "direction_source": result["direction_source"],
        "excluded_unverified": result["excluded_unverified"],
    }


@router.get("/metric-directions")
def metric_directions_route() -> dict:
    """Every metric name evaluation.py declares a ranking direction for, a plain read with no audit
    line and no registry touch.
    """
    from tcip_mcp.pipelines.training.evaluation import HIGHER_IS_BETTER_BY_METRIC

    return {"higher_is_better": dict(HIGHER_IS_BETTER_BY_METRIC)}


# ── WebSocket live metrics ──────────────────────────────────────────────


class TrainingMetricFrame(BaseModel):
    """One metrics-log row, pushed as it is appended."""

    type: Literal["metric"]
    experiment_id: str
    row: dict


class TrainingStatusFrame(BaseModel):
    """The terminal frame: ``status`` carries the run's ``experiments.run_summary`` row for a run
    this process can still identify, ``error`` is set instead when it cannot."""

    type: Literal["status"]
    experiment_id: str
    status: dict | None
    error: str | None


async def _stream_metrics(
    ws: WebSocket, project_root: str, experiment_id: str, poll_seconds: float = 1.0
) -> None:
    """Push every row of a run's ``metrics.jsonl`` to the browser as it is appended.

    Each tick observes the run once (``experiments.observe``) and reads its log from a byte-offset
    cursor (``experiments.read_rows``), so it reads only what was appended since the last tick and
    a row still being written is replayed once it is complete. Rows are read after the
    observation, so a run the observation found in a terminal state has every row sent before its
    one status frame (``experiments.run_summary`` over the rows sent), which ends the stream. An id naming no run
    directory under ``project_root`` ends the stream with one status frame naming it, and one that
    is not a single directory name raises ``BadKey``. Every read runs off the event loop.
    """
    from tcip_mcp.experiments import (
        TERMINAL_STATES, find_observation, observe, read_rows, run_name, run_summary,
    )

    observation = await asyncio.to_thread(find_observation, run_name(experiment_id),
                                          root=project_root)
    if observation is None:
        await ws.send_json(TrainingStatusFrame(
            type="status", experiment_id=experiment_id, status=None,
            error=f"Run not found: {experiment_id}").model_dump())
        return
    cursor = 0
    sent: list[dict] = []

    while True:
        rows, cursor = await asyncio.to_thread(read_rows, observation.metrics_log, after=cursor)
        for row in rows:
            frame = TrainingMetricFrame(type="metric", experiment_id=experiment_id, row=row)
            await ws.send_json(frame.model_dump())
        sent += rows
        if observation.state in TERMINAL_STATES:
            status = await asyncio.to_thread(run_summary, observation, sent)
            await ws.send_json(TrainingStatusFrame(
                type="status", experiment_id=experiment_id, status=status,
                error=None).model_dump())
            break
        await asyncio.sleep(poll_seconds)
        observation = await asyncio.to_thread(observe, observation.directory)


@router.websocket("/runs/{experiment_id}/stream")
async def training_stream_ws(websocket: WebSocket, experiment_id: str, project_root: str) -> None:
    """Tail ``experiment_id``'s metrics log and push new rows to the browser."""
    try:
        assert_project_root_allowed(project_root)
    except ValueError as exc:
        await websocket.close(code=1008, reason=str(exc))
        return
    await websocket.accept()
    try:
        await _stream_metrics(websocket, project_root, experiment_id)
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
