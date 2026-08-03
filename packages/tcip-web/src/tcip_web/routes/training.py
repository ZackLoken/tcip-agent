"""Training routes: validate config, launch, list runs, live metrics stream."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from tcip_web.paths import assert_path_allowed, assert_project_root_allowed, origin_allowed, safe_join
from tcip_web.routes._metrics_common import read_metrics_file

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/training", tags=["training"])

_TERMINAL_STATES = {"completed", "failed", "cancelled", "interrupted"}

# A reconstructed non-terminal run is only "interrupted" if its heartbeat is stale; a fresh
# heartbeat means a training process (possibly the MCP agent) is still updating it. Generous
# default so a slow epoch between per-epoch heartbeats doesn't flap the status.
_HEARTBEAT_STALE_SECONDS = float(os.environ.get("TCIP_HEARTBEAT_STALE_SECONDS", "600"))


def _heartbeat_fresh(hb_iso: str | None) -> bool:
    """True if ``hb_iso`` (ISO-8601) is within the staleness window: i.e. a process is
    still actively updating this run. Missing/unparseable → not fresh (treat as dead)."""
    if not hb_iso:
        return False
    try:
        hb = datetime.fromisoformat(hb_iso)
    except (ValueError, TypeError):
        return False
    if hb.tzinfo is None:
        hb = hb.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - hb).total_seconds() <= _HEARTBEAT_STALE_SECONDS


def _historical_training_runs() -> list[dict]:
    """Reconstruct past training runs from the immutable ``.tcip/experiments/`` records.

    Every ``launch_training`` (standalone drift-retrain, GUI, or agent) writes an
    experiment, so this recovers runs the in-memory registry lost on restart, with no
    second persistence file. Only genuine *training* experiments are included (a
    ``model_source`` in the config); review-feedback / ad-hoc experiments are skipped, and
    HPO trials never create experiments so they can't appear here. A non-terminal state
    on a run that isn't live means the process died -> surfaced as ``interrupted``.

    Delegates the actual state/current_epoch reconstruction to
    ``tcip_mcp.experiments.reconstruct_run_status``, the single implementation of
    experiment-directory-to-``run_id`` resolution. A fresh-id relaunch's experiment directory
    name does not necessarily equal the real ``run_id``; a directory whose ``status.json`` never
    stamped ``run_id`` still resolves correctly through the shared resolver's exact-match
    strategy: defaulting to ``d.name`` here means the lookup is for the directory's own name,
    which trivially matches itself, and ``current_epoch`` is populated from ``metrics.jsonl`` via
    that same resolver. The inline fallback below is reached only when
    ``reconstruct_run_status`` itself returns ``None`` or resolves to a *different* directory than
    the one being iterated (a malformed/unreadable ``status.json``, or a genuine identity
    anomaly), a narrower, degenerate case, not the common case of a directory whose ``status.json``
    never stamped ``run_id``.
    """
    from tcip_mcp.experiments import experiments_dir, reconstruct_run_status
    from tcip_mcp.utils.atomic_io import read_json

    exp_root = experiments_dir()
    if not exp_root.exists():
        return []
    runs: list[dict] = []
    for d in sorted(exp_root.iterdir()):
        if not d.is_dir():
            continue
        config = read_json(d / "config.json", default={})
        if not isinstance(config, dict) or not config.get("model_source"):
            continue  # not a training experiment (e.g. review-feedback lineage)
        status = read_json(d / "status.json", default={})
        run_id = status.get("run_id", d.name)  # the real run_id when stamped, else the
                                                # experiment_id itself as the fallback
        reconstructed = reconstruct_run_status(run_id, stale_seconds=_HEARTBEAT_STALE_SECONDS)
        if reconstructed is not None and reconstructed["experiment_id"] == d.name:
            runs.append({
                "run_id": run_id,
                "status": reconstructed["status"],
                "current_epoch": reconstructed["current_epoch"],
                "best_metric": reconstructed["best_metric"],
                "external": True,  # reconstructed → not managed by this web process
            })
            continue
        # Fallback: reconstruct_run_status found nothing (or something else) for this directory's
        # own run_id: a malformed/unreadable status.json, not the common case of a status.json
        # that never stamped run_id (that one resolves correctly above via the exact-match
        # strategy on the defaulted d.name).
        state = status.get("state", "unknown")
        if state not in _TERMINAL_STATES:
            state = "running" if _heartbeat_fresh(status.get("heartbeat")) else "interrupted"
        runs.append({
            "run_id": run_id,
            "status": state,
            "current_epoch": None,
            "best_metric": None,
            "external": True,
        })
    return runs


def _metrics_path(project_root: str, run_id: str) -> Path:
    """``<project_root>/.tcip/experiments/<run_id>/metrics.jsonl`` with traversal guarded.

    ``run_id`` is an untrusted path component, so it is joined via ``safe_join`` (which
    rejects ``..`` / absolute paths), raises ``ValueError`` on an attempted escape.
    """
    return safe_join(Path(project_root) / ".tcip" / "experiments", run_id, "metrics.jsonl")


class ConfigPayload(BaseModel):
    config: dict[str, Any]


@router.post("/validate")
def preflight_config_route(payload: ConfigPayload) -> dict:
    from tcip_mcp.tools.training_tools import preflight_config

    return preflight_config(payload.config)


class LaunchPayload(BaseModel):
    config: dict[str, Any]
    output_dir: str


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
    """Live in-memory training runs, backfilled with past runs from experiment records.

    ``list_training_runs`` already excludes HPO trials (they stay in the Tuning view);
    historical runs come from ``.tcip/experiments/`` so they survive a restart without a
    separate persistence file. Live runs win over a historical entry by run_id.
    """
    from tcip_mcp.tools.training_tools import list_training_runs

    live = list_training_runs().get("runs", [])
    live_ids = {r.get("run_id") for r in live}
    historical = [h for h in _historical_training_runs() if h.get("run_id") not in live_ids]
    return {"runs": list(live) + historical}


@router.get("/runs/{run_id}")
def get_run(run_id: str) -> dict:
    from tcip_mcp.tools.training_tools import check_training_status

    return check_training_status(run_id)


@router.post("/runs/{run_id}/tensorboard")
def launch_run_tensorboard(run_id: str) -> dict:
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
def cancel_run_route(run_id: str) -> dict:
    """Request graceful cancellation of a running run (stops at the next batch boundary).

    Wraps the ``cancel_training`` MCP tool: the trainer still writes ``model_final.pt``
    so partial progress is recoverable. Status flips to 'cancelled' asynchronously.
    """
    from tcip_mcp.tools.training_tools import cancel_training

    result = cancel_training(run_id)
    if result.get("error"):
        raise HTTPException(404, result["error"])
    return result


@router.get("/runs/{run_id}/metrics")
def get_run_metrics(project_root: str, run_id: str) -> dict:
    """Read the full metrics.jsonl for a run."""
    try:
        assert_project_root_allowed(project_root)
    except ValueError as exc:
        raise HTTPException(403, str(exc)) from exc
    try:
        metrics_path = _metrics_path(project_root, run_id)
    except ValueError:
        raise HTTPException(400, f"invalid run_id: {run_id}") from None
    return read_metrics_file(metrics_path)


class ExperimentComparePayload(BaseModel):
    experiment_ids: list[str]


@router.post("/compare")
def compare_runs_route(payload: ExperimentComparePayload) -> dict:
    from tcip_mcp.tools.experiment_tools import compare_experiments

    return compare_experiments(payload.experiment_ids)


# ── WebSocket live metrics ──────────────────────────────────────────────


def _read_metrics_after(path: Path, offset: int) -> tuple[list[dict], int]:
    """Read metrics.jsonl rows written since byte ``offset``, seeking there instead of
    re-parsing from the start every poll tick. Returns the byte offset to resume from next
    time; a trailing incomplete line (a write still in flight) is left unread and replayed
    once it's complete."""
    if not path.exists():
        return [], offset
    rows: list[dict] = []
    with path.open("rb") as f:
        f.seek(offset)
        chunk = f.read()
    new_offset = offset
    for raw_line in chunk.splitlines(keepends=True):
        if not raw_line.endswith(b"\n"):
            break
        new_offset += len(raw_line)
        line = raw_line.decode("utf-8").strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    return rows, new_offset


async def _stream_metrics(
    ws: WebSocket, project_root: str, run_id: str, poll_seconds: float = 1.0
) -> None:
    try:
        metrics_path = _metrics_path(project_root, run_id)
    except ValueError:
        await ws.send_json({"type": "error", "error": f"invalid run_id: {run_id}"})
        return
    cursor = 0

    while True:
        rows, cursor = _read_metrics_after(metrics_path, cursor)
        for row in rows:
            await ws.send_json({"type": "metric", "run_id": run_id, "row": row})

        # Has the run finished (or gone away)?
        try:
            from tcip_mcp.tools.training_tools import check_training_status
            from tcip_web import jobstore

            status = check_training_status(run_id)
            # ``error`` => unknown run (e.g. streamed after a restart); a cancelled run
            # never reaches completed/failed. Either way, terminate: the prior check
            # keyed only on completed/failed/"not_found" and spun forever on both.
            if status.get("error") or status.get("status") in jobstore.TERMINAL_STATUSES:
                await ws.send_json({"type": "status", "run_id": run_id, "status": status})
                break
        except Exception:
            logger.exception("check_training_status failed in stream")

        await asyncio.sleep(poll_seconds)


@router.websocket("/runs/{run_id}/stream")
async def training_stream_ws(websocket: WebSocket, run_id: str, project_root: str) -> None:
    """Tail metrics.jsonl for ``run_id`` and push new rows to the browser."""
    if not origin_allowed(websocket.headers.get("origin")):
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
    except WebSocketDisconnect:
        pass
    except Exception:
        logger.exception("training stream failed")
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
