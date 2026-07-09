"""Training routes: validate config, launch, list runs, live metrics stream."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from tcip_web.paths import safe_join

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/training", tags=["training"])

_TERMINAL_STATES = {"completed", "failed", "cancelled", "interrupted"}


def _historical_training_runs() -> list[dict]:
    """Reconstruct past training runs from the immutable ``.tcip/experiments/`` records.

    Every ``launch_training`` (standalone drift-retrain, GUI, or agent) writes an
    experiment, so this recovers runs the in-memory registry lost on restart — with no
    second persistence file. Only genuine *training* experiments are included (a
    ``model_spec`` in the config); review-feedback / ad-hoc experiments are skipped, and
    HPO trials never create experiments so they can't appear here. A non-terminal state
    on a run that isn't live means the process died -> surfaced as ``interrupted``.
    """
    from tcip_mcp.experiments import EXPERIMENTS_DIR
    from tcip_mcp.utils.atomic_io import read_json

    if not EXPERIMENTS_DIR.exists():
        return []
    runs: list[dict] = []
    for d in sorted(EXPERIMENTS_DIR.iterdir()):
        if not d.is_dir():
            continue
        config = read_json(d / "config.json", default={})
        if not isinstance(config, dict) or not (config.get("model_spec") or config.get("model")):
            continue  # not a training experiment (e.g. review-feedback lineage)
        state = read_json(d / "status.json", default={}).get("state", "unknown")
        if state not in _TERMINAL_STATES:
            state = "interrupted"  # historical run, process gone
        runs.append({
            "run_id": d.name,
            "status": state,
            "current_epoch": None,
            "best_metric": None,
        })
    return runs


def _metrics_path(project_root: str, run_id: str) -> Path:
    """``<project_root>/.tcip/experiments/<run_id>/metrics.jsonl`` with traversal guarded.

    ``run_id`` is an untrusted path component, so it is joined via ``safe_join`` (which
    rejects ``..`` / absolute paths) — raises ``ValueError`` on an attempted escape.
    """
    return safe_join(Path(project_root) / ".tcip" / "experiments", run_id, "metrics.jsonl")


class ConfigPayload(BaseModel):
    config: dict[str, Any]


@router.post("/validate")
def validate_config_route(payload: ConfigPayload) -> dict:
    from tcip_mcp.tools.training_tools import validate_config

    return validate_config(payload.config)


class LaunchPayload(BaseModel):
    config: dict[str, Any]
    output_dir: str


@router.post("/launch")
def launch_training_route(payload: LaunchPayload) -> dict:
    from tcip_mcp.tools.training_tools import launch_training

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


@router.post("/runs/{run_id}/cancel")
def cancel_run_route(run_id: str) -> dict:
    """Request graceful cancellation of a running run (stops at the next batch boundary).

    Wraps the ``cancel_training`` MCP tool — the trainer still writes ``model_final.pt``
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
        metrics_path = _metrics_path(project_root, run_id)
    except ValueError:
        raise HTTPException(400, f"invalid run_id: {run_id}") from None
    if not metrics_path.exists():
        return {"metrics": [], "exists": False}
    rows: list[dict] = []
    with metrics_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return {"metrics": rows, "exists": True}


class ExperimentComparePayload(BaseModel):
    experiment_ids: list[str]


@router.post("/compare")
def compare_runs_route(payload: ExperimentComparePayload) -> dict:
    from tcip_mcp.tools.experiment_tools import compare_experiments

    return compare_experiments(payload.experiment_ids)


class RegisterModelPayload(BaseModel):
    project_path: str
    model_name: str
    checkpoint_path: str
    tag: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None


@router.post("/register_model")
def register_model_route(payload: RegisterModelPayload) -> dict:
    from tcip_mcp.tools.model_tools import register_model

    return register_model(
        project_path=payload.project_path,
        model_name=payload.model_name,
        checkpoint_path=payload.checkpoint_path,
        tag=payload.tag,
        metadata=payload.metadata,
    )


# ── WebSocket live metrics ──────────────────────────────────────────────


def _read_metrics_after(path: Path, after_line: int) -> tuple[list[dict], int]:
    """Read lines of a metrics.jsonl after a given line index."""
    if not path.exists():
        return [], after_line
    rows: list[dict] = []
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < after_line:
                count = i + 1
                continue
            line = line.strip()
            count = i + 1
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows, count


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
            # never reaches completed/failed. Either way, terminate — the prior check
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
