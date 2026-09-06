"""Training routes: launchable configs, launch/relaunch, list runs, live metrics stream."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Literal

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from tcip_store.errors import BadKey

if TYPE_CHECKING:
    from tcip_store import Key

from tcip_web.paths import assert_project_root_allowed
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


@router.get("/configs")
def list_configs_route() -> dict:
    """Every experiment in this project a run can be started or relaunched from."""
    from tcip_mcp.tools.training_tools import list_launchable_configs

    return {"configs": list_launchable_configs()}


@router.get("/configs/{experiment_id}/splits")
def list_split_choices_route(experiment_id: str) -> dict:
    """Every choice this config's own "Data" control offers a relaunch: its stored data
    section as recorded, and every split manifest directory this project's own bound runs or
    the dataset's own splits directory hold, compatibility-checked as the launch itself would
    check them."""
    from tcip_mcp.tools.training_tools import list_split_choices

    result = list_split_choices(experiment_id)
    if result.get("error"):
        raise HTTPException(404, result["error"])
    return result


class RelaunchConfigPayload(BaseModel):
    experiment_id: str
    split_manifest_dir: str | None = None


@router.post("/runs")
def relaunch_config_route(payload: RelaunchConfigPayload) -> dict:
    """Start a run from a config already recorded in this project: no config, param space or
    path is ever submitted by the browser. A pristine config launches as its own first run; a
    run's config launches as a new experiment id with the picked one as parent.

    An optional ``split_manifest_dir`` names a partition the browser picked instead of the
    snapshot's own "As recorded" data section: checked against this same config's own
    :func:`~tcip_mcp.tools.training_tools.list_split_choices` listing (an enabled offer or 409)
    through :func:`~tcip_mcp.tools.training_tools.split_dir_identity`, so a symlinked or
    differently cased spelling of an offered directory is admitted, not just an exact string
    match; the path itself is never resolved as one the server follows. The launch config then
    carries ``data.split`` replaced wholesale by ``{"manifest_dir": chosen}`` with any
    ``data.val_images_dir`` removed; ``auto_train_val`` clears the previous binding's own stamps
    on its way to a fresh one.

    The launch is wrapped in ``declare_launcher("gui")``, so the run's status record stamps
    ``launched_by: {"launcher": "gui"}``: the fact this route started it, true of whatever client
    actually posted here (a browser, a script, another agent), since nothing here tells them
    apart; that distinction is a later authentication concern, not this route's.
    """
    from tcip_mcp.experiments import config_key, read_member, status_key
    from tcip_mcp.pipelines.model_build import MODEL_SOURCE_KEY
    from tcip_mcp.tools.training_tools import (
        candidate_config_with_manifest, declare_launcher, launch_training, list_split_choices,
        split_dir_identity,
    )

    config = read_member(config_key(payload.experiment_id), None)
    if not isinstance(config, dict) or not config.get(MODEL_SOURCE_KEY):
        raise HTTPException(404, f"no launchable config named {payload.experiment_id}")

    status = read_member(status_key(payload.experiment_id), {})
    if status.get("state") == "created" and status.get("run_id"):
        raise HTTPException(409, f"experiment {payload.experiment_id} already has a run "
                                 "attached to it")

    config = {**config, "experiment_id": payload.experiment_id}
    if payload.split_manifest_dir:
        choices = list_split_choices(payload.experiment_id)
        enabled = {split_dir_identity(m["manifest_dir"])
                   for m in choices.get("manifests", []) if m.get("enabled")}
        if split_dir_identity(payload.split_manifest_dir) not in enabled:
            raise HTTPException(
                409, f"{payload.split_manifest_dir!r} is not an offered partition for "
                     f"{payload.experiment_id}",
            )
        config = candidate_config_with_manifest(config, payload.split_manifest_dir)
    try:
        with declare_launcher("gui"):
            result = launch_training(config)
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc
    if result.get("error"):
        raise HTTPException(422, detail=result)
    return result


@router.get("/runs")
def list_runs_route() -> dict:
    """Every training run the platform can currently account for.

    A pass-through to ``_all_training_runs``, which merges this process's live runs with every
    launched run's own record on disk (surviving a restart with no second persistence file) and
    excludes HPO trials (they stay in the Tuning view); the same rows
    ``list_experiments(launched_only=True)`` returns to the agent.
    """
    from tcip_mcp.tools.training_tools import _all_training_runs

    return {"runs": _all_training_runs(read_progress=True)}


@router.get("/runs/{run_id}")
def get_run(run_id: str) -> dict:
    from tcip_mcp.tools.training_tools import monitor_training

    return monitor_training(run_id)


@router.post("/runs/{run_id}/tensorboard")
def launch_run_tensorboard(run_id: str, payload: EmptyBodyPayload) -> dict:
    """Start (or reuse) a TensorBoard serving this run's log directory.

    ``tensorboard_manager`` tracks its children in module-level process state, so a TensorBoard
    started by the agent's own process is not one this process can hand the browser a URL for.
    This route is how a TensorBoard exists from the GUI's side, whichever process trained the run.

    A run with no recorded output directory (it failed before writing one) or whose output
    directory's ``tensorboard`` subdirectory holds no event file (it crashed before
    ``SummaryWriter`` ever wrote one) never had anything to log to; that refusal carries
    ``no_logs: True`` so the GUI can say the run produced no logs instead of starting a
    TensorBoard against a directory nothing populated and offering a retry that would do the
    same. A run whose own status already carries an error is checked for events first rather
    than refused on the error alone, so a crash that recorded a reason still reads as no-logs
    (with that reason attached) when it produced none; an error paired with real event files
    keeps the plain refusal, no retry warranted against a directory that has something to serve.
    """
    from pathlib import Path

    from tcip_mcp.pipelines.training.tensorboard_manager import launch_tensorboard
    from tcip_mcp.tools.training_tools import monitor_training

    status = monitor_training(run_id)
    if "status" not in status:
        raise HTTPException(404, status.get("error") or f"Run not found: {run_id}")
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
            {"error": error or f"run produced no logs: {run_id}", "no_logs": True},
        )
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


class CompareBestPayload(BaseModel):
    experiment_ids: list[str]
    metric: str
    higher_is_better: bool | None = None
    include_unverified: bool = False


@router.post("/compare/best")
def compare_best_route(payload: CompareBestPayload) -> dict:
    """Rank the marked comparison's own registered checkpoints by one metric.

    Wraps the platform's one best-model derivation (``rank_registered_models``), narrowed to the
    marked experiments before anything is derived. Reads the registry index first so a project
    with none never takes the tool's own directory-creating construction: only the reader's own
    empty answer (no index at all) is a 404; a document that exists but will not decode
    (``DecodeError``), that this reader does not recognize (``RegistryVersionRefused``), or whose
    ``schema_version`` sits above the ceiling this store knows (``SchemaVersionRefused``, a
    sibling of ``RegistryVersionRefused`` under ``StoreError``, not folded into it) is not
    a project with no models, and answers 409 naming why, the same wording
    ``compare_experiments``'s own ``registry_error`` carries. The tool's own error dicts (a
    required metric, an undeclared direction, no carrier) map to 422 with the whole dict as
    ``detail``, and the pre-``metrics_source`` refusal (a registry entry predating the field)
    maps to 409 with the registry's own message. The answer is projected to name, experiment id,
    stamped metrics, source, the direction used and its source, and the exclusions; no checkpoint
    path, config or file size leaves this route.
    """
    from tcip_store.errors import DecodeError, SchemaVersionRefused

    from tcip_mcp.model_registry import RegistryEntryPredatesMetricsSource, RegistryVersionRefused, read_registry_index
    from tcip_mcp.project_paths import platform_state_root
    from tcip_mcp.tools.model_tools import rank_registered_models

    try:
        entries = read_registry_index(platform_state_root())
    except (DecodeError, RegistryVersionRefused, SchemaVersionRefused) as exc:
        raise HTTPException(409, f"registry unreadable: {exc}") from exc
    if not entries:
        raise HTTPException(404, "no model registry in this project")

    try:
        result = rank_registered_models(
            metric=payload.metric, higher_is_better=payload.higher_is_better,
            include_unverified=payload.include_unverified, experiment_ids=payload.experiment_ids,
        )
    except RegistryEntryPredatesMetricsSource as exc:
        raise HTTPException(409, str(exc)) from exc
    if "error" in result:
        raise HTTPException(422, detail=result)

    return {
        "name": result["name"],
        "experiment_id": result.get("experiment_id"),
        "metrics": result["metrics"],
        "metrics_source": result["metrics_source"],
        "higher_is_better": result["higher_is_better"],
        "direction_source": result["direction_source"],
        "excluded_unverified": result["excluded_unverified"],
    }


@router.get("/metric-directions")
def metric_directions_route() -> dict:
    """Every metric name evaluation.py declares a ranking direction for, a plain read with no
    audit line and no registry touch: the comparison's own metric chooser groups its stamped
    keys by this table on mount, instead of eliciting it through the audited rank tool."""
    from tcip_mcp.pipelines.training.evaluation import HIGHER_IS_BETTER_BY_METRIC

    return {"higher_is_better": dict(HIGHER_IS_BETTER_BY_METRIC)}


# ── WebSocket live metrics ──────────────────────────────────────────────


class TrainingMetricFrame(BaseModel):
    """One metrics-log row, pushed as it is appended."""

    type: Literal["metric"]
    run_id: str
    row: dict


class TrainingStatusFrame(BaseModel):
    """The terminal frame: ``status`` carries ``monitor_training``'s report whole for a
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
    launch has created it. Both reads run off the event loop: a file-backend read can wait
    on a training subprocess's own append, and that wait must stall this socket's own
    coroutine rather than every request and socket the backend serves.
    """
    from tcip_store import read_log

    key = None
    cursor: str | None = None

    while True:
        if key is None:
            key = _metrics_key(project_root, run_id)
        rows: list[dict] = []
        if key is not None:
            page = await asyncio.to_thread(read_log, key, after=cursor)
            cursor = page.cursor
            rows = [dict(row) for row in page.records]
        for row in rows:
            frame = TrainingMetricFrame(type="metric", run_id=run_id, row=row)
            await ws.send_json(frame.model_dump())

        # Has the run finished (or gone away)? ``error`` with no ``status`` key => unknown run;
        # a cancelled run never reaches completed/failed, so either case ends the stream.
        try:
            from tcip_mcp.tools.training_tools import monitor_training
            from tcip_web import jobstore

            status = monitor_training(run_id)
            if status.get("error") or status.get("status") in jobstore.TERMINAL_STATUSES:
                # A row can land between the read above and this terminal observation; drain
                # it now so the status frame never precedes the row it terminates on.
                if key is not None:
                    final_page = await asyncio.to_thread(read_log, key, after=cursor)
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
            logger.exception("monitor_training failed in stream")

        await asyncio.sleep(poll_seconds)


@router.websocket("/runs/{run_id}/stream")
async def training_stream_ws(websocket: WebSocket, run_id: str, project_root: str) -> None:
    """Tail ``run_id``'s metrics log and push new rows to the browser."""
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
