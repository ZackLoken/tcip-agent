"""Training routes: validate config, launch, list runs, live metrics stream."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, Literal

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from tcip_store.errors import BadKey

if TYPE_CHECKING:
    from tcip_store import Key

from tcip_web.paths import assert_path_allowed, assert_project_root_allowed
from tcip_web.trust_boundary import origin_allowed
from tcip_web.routes._body_common import EmptyBodyPayload

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/training", tags=["training"])


def _metrics_key(project_root: str, run_id: str) -> "Key | None":
    """The metrics log of the experiment that claims ``run_id``, under ``project_root``.

    ``None`` when no record claims the run: the experiment may not exist yet on a run just
    launched, and the caller then has nothing to serve rather than an empty log to assert.
    A relaunch mints an experiment id that is not the run id, so the id is resolved from the
    records themselves rather than assumed; ``run_id`` never becomes a path component here.
    An id no record could ever carry (a path separator, an empty or dot name) raises
    ``BadKey`` instead of resolving to "no record": absence is an answer, malformedness is a
    refusal.
    """
    from tcip_mcp.experiments import metrics_key, resolve_experiment_for_run, status_key

    # Shape check through the key constructor, the one place the id rule lives; the key is
    # discarded because resolution below decides which record actually claims the run.
    status_key(run_id, root=project_root)
    experiment_id = resolve_experiment_for_run(run_id, root=project_root)
    if experiment_id is None:
        return None
    return metrics_key(experiment_id, root=project_root)


class ConfigPayload(BaseModel):
    config: dict[str, Any]


@router.post("/validate")
def preflight_config_route(payload: ConfigPayload) -> dict:
    from tcip_mcp.tools.training_tools import preflight_config

    return preflight_config(payload.config)


class LaunchPayload(BaseModel):
    config: dict[str, Any]
    # Empty defers to launch_training's own default (the project's experiment store), so a
    # client that names no directory can never land a run in the server process's cwd.
    output_dir: str = ""


@router.post("/launch")
def launch_training_route(payload: LaunchPayload) -> dict:
    from tcip_mcp.tools.training_tools import launch_training

    # Confine the client-supplied output dir when the server is locked down (no-op otherwise):
    # mirrors tuning.py's launch_hpo. This does not confine config.model_source.builder, which this
    # same route importlib-imports and calls (model_build.py): that is a separate code-execution
    # trust boundary path confinement can't close; don't read this guard as closing it.
    if payload.output_dir:
        try:
            assert_path_allowed(payload.output_dir)
        except ValueError as exc:
            raise HTTPException(403, str(exc)) from exc

    try:
        return launch_training(payload.config, payload.output_dir)
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc


@router.get("/runs")
def list_runs_route() -> dict:
    """Every training run the platform can currently account for.

    A pass-through to ``list_training_runs``, which already merges this process's live runs
    with every launched run's own record on disk (surviving a restart with no second
    persistence file) and excludes HPO trials (they stay in the Tuning view).
    """
    from tcip_mcp.tools.training_tools import list_training_runs

    return list_training_runs()


@router.get("/runs/{run_id}")
def get_run(run_id: str) -> dict:
    from tcip_mcp.tools.training_tools import check_training_status

    return check_training_status(run_id)


@router.post("/runs/{run_id}/tensorboard")
def launch_run_tensorboard(run_id: str, payload: EmptyBodyPayload) -> dict:
    """Start (or reuse) a TensorBoard serving this run's log directory.

    ``tensorboard_manager`` tracks its children in module-level process state, so a TensorBoard
    started by the agent's own process is not one this process can hand the browser a URL for.
    This route is how a TensorBoard exists from the GUI's side, whichever process trained the run.
    """
    from tcip_mcp.pipelines.training.tensorboard_manager import launch_tensorboard
    from tcip_mcp.tools.training_tools import check_training_status

    status = check_training_status(run_id)
    if status.get("error"):
        raise HTTPException(404, status["error"])
    output_dir = status.get("output_dir")
    if not output_dir:
        raise HTTPException(404, f"run has no output directory: {run_id}")
    return launch_tensorboard(f"{output_dir}/tensorboard", run_id=run_id)


@router.post("/runs/{run_id}/cancel")
def cancel_run_route(run_id: str, payload: EmptyBodyPayload) -> dict:
    """Request graceful cancellation of a running run (stops at the next batch boundary).

    Wraps the ``cancel_training`` MCP tool: the trainer still writes ``model_final.pt``
    so partial progress is recoverable. Status flips to 'cancelled' asynchronously, unless the
    run's divergence verdict lands first, in which case it ends 'failed' instead.
    """
    from tcip_mcp.tools.training_tools import cancel_training

    result = cancel_training(run_id)
    if result.get("error"):
        raise HTTPException(404, result["error"])
    return result


class ExperimentComparePayload(BaseModel):
    experiment_ids: list[str]


@router.post("/compare")
def compare_runs_route(payload: ExperimentComparePayload) -> dict:
    from tcip_mcp.tools.experiment_tools import compare_experiments

    return compare_experiments(payload.experiment_ids)


# ── WebSocket live metrics ──────────────────────────────────────────────


class TrainingMetricFrame(BaseModel):
    """One metrics-log row, pushed as it is appended."""

    type: Literal["metric"]
    run_id: str
    row: dict


class TrainingStatusFrame(BaseModel):
    """The terminal frame: ``status`` carries ``check_training_status``'s report whole for a
    run this process can still identify, ``error`` is set instead when it cannot."""

    type: Literal["status"]
    run_id: str
    status: dict | None
    error: str | None


async def _stream_metrics(
    ws: WebSocket, project_root: str, run_id: str, poll_seconds: float = 1.0
) -> None:
    """Push every row of a run's metrics log to the browser as it is appended.

    The cursor is the log's own resume token, so each tick reads only what was appended
    since the last one and an entry still being written is replayed once it is complete.
    The run's record is re-resolved until it exists, since a stream can be opened before the
    launch has created it.
    """
    from tcip_store import read_log

    key = None
    cursor: str | None = None

    while True:
        if key is None:
            key = _metrics_key(project_root, run_id)
        rows: list[dict] = []
        if key is not None:
            page = read_log(key, after=cursor)
            cursor = page.cursor
            rows = [dict(row) for row in page.records]
        for row in rows:
            frame = TrainingMetricFrame(type="metric", run_id=run_id, row=row)
            await ws.send_json(frame.model_dump())

        # Has the run finished (or gone away)? ``error`` with no ``status`` key => unknown run;
        # a cancelled run never reaches completed/failed, so either case ends the stream.
        try:
            from tcip_mcp.tools.training_tools import check_training_status
            from tcip_web import jobstore

            status = check_training_status(run_id)
            if status.get("error") or status.get("status") in jobstore.TERMINAL_STATUSES:
                # A row can land between the read above and this terminal observation; drain
                # it now so the status frame never precedes the row it terminates on.
                if key is not None:
                    final_page = read_log(key, after=cursor)
                    cursor = final_page.cursor
                    for row in (dict(r) for r in final_page.records):
                        frame = TrainingMetricFrame(type="metric", run_id=run_id, row=row)
                        await ws.send_json(frame.model_dump())
                if "status" in status:
                    status_frame = TrainingStatusFrame(
                        type="status", run_id=run_id, status=status, error=None
                    )
                else:
                    status_frame = TrainingStatusFrame(
                        type="status", run_id=run_id, status=None, error=status.get("error")
                    )
                await ws.send_json(status_frame.model_dump())
                break
        except Exception:
            logger.exception("check_training_status failed in stream")

        await asyncio.sleep(poll_seconds)


@router.websocket("/runs/{run_id}/stream")
async def training_stream_ws(websocket: WebSocket, run_id: str, project_root: str) -> None:
    """Tail ``run_id``'s metrics log and push new rows to the browser."""
    if not origin_allowed(websocket.headers.get("origin"), websocket.scope):
        await websocket.close(code=1008, reason="origin not allowed")
        return
    try:
        assert_project_root_allowed(project_root)
    except ValueError as exc:
        await websocket.close(code=1008, reason=str(exc))
        return
    await websocket.accept()
    try:
        await _stream_metrics(websocket, project_root, run_id)
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
