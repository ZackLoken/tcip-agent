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
    from tcip_mcp.tools.training_tools import list_training_runs

    return list_training_runs()


@router.get("/runs/{run_id}")
def get_run(run_id: str) -> dict:
    from tcip_mcp.tools.training_tools import check_training_status

    return check_training_status(run_id)


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

        # Has the run finished?
        try:
            from tcip_mcp.tools.training_tools import check_training_status

            status = check_training_status(run_id)
            if status.get("status") in ("completed", "failed", "not_found"):
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
