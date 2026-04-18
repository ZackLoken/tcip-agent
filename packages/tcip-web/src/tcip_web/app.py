"""FastAPI application — REST API for MCP tools + WebSocket for training progress."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

logger = logging.getLogger(__name__)

app = FastAPI(title="TCIP Pipeline", version="0.1.0")

# Serve static files (web frontend)
STATIC_DIR = Path(__file__).parent.parent.parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ── Health ──


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


# ── Dataset tools ──


@app.get("/api/dataset")
def load_dataset(folder_path: str = ".") -> dict:
    from tcip_mcp.tools.data_tools import load_dataset as _load
    return _load(folder_path)


@app.get("/api/dataset/validate")
def validate_data(folder_path: str = ".") -> dict:
    from tcip_mcp.tools.data_tools import validate_data_quality as _validate
    return _validate(folder_path)


# ── Project tools ──


@app.post("/api/project/init")
def init_project(project_path: str = ".") -> dict:
    from tcip_mcp.tools.project_tools import init_project as _init
    return _init(project_path)


@app.get("/api/project/status")
def project_status(project_path: str = ".") -> dict:
    from tcip_mcp.tools.project_tools import get_project_status as _status
    return _status(project_path)


# ── Experiment tools ──


@app.get("/api/experiments")
def list_experiments() -> list:
    from tcip_mcp.experiments import list_experiments as _list
    return _list()


@app.get("/api/experiments/{experiment_id}")
def get_experiment(experiment_id: str) -> dict:
    from tcip_mcp.experiments import get_experiment as _get
    return _get(experiment_id)


@app.post("/api/experiments")
def create_experiment(experiment_id: str, config: dict) -> dict:
    from tcip_mcp.experiments import create_experiment as _create
    return _create(experiment_id, config)


@app.get("/api/experiments/compare")
def compare_experiments(ids: str) -> dict:
    """Compare experiments. Pass comma-separated IDs."""
    from tcip_mcp.experiments import compare_experiments as _compare
    return _compare(ids.split(","))


# ── Training tools ──


@app.post("/api/training/validate")
def validate_config(config: dict) -> dict:
    from tcip_mcp.tools.training_tools import validate_config as _validate
    return _validate(config)


@app.post("/api/training/launch")
def launch_training(config: dict, output_dir: str = "./runs") -> dict:
    from tcip_mcp.tools.training_tools import launch_training as _launch
    return _launch(config, output_dir)


@app.get("/api/training/{run_id}")
def training_status(run_id: str) -> dict:
    from tcip_mcp.tools.training_tools import check_training_status as _status
    return _status(run_id)


@app.get("/api/training")
def list_runs() -> dict:
    from tcip_mcp.tools.training_tools import list_training_runs as _list
    return _list()


# ── Model tools ──


@app.get("/api/models/available")
def available_models() -> dict:
    from tcip_mcp.tools.model_tools import list_available_models as _list
    return _list()


@app.get("/api/models/registered")
def registered_models(project_path: str = ".") -> dict:
    from tcip_mcp.tools.model_tools import list_registered_models as _list
    return _list(project_path)


# ── Pipeline tools ──


@app.get("/api/components")
def list_components() -> dict:
    from tcip_mcp.tools.pipeline_tools import list_components as _list
    return _list()


@app.post("/api/pipeline/run")
def run_pipeline(spec: dict, work_dir: str = "./pipeline_runs") -> dict:
    from tcip_mcp.tools.pipeline_tools import run_pipeline as _run
    return _run(spec, work_dir)


# ── WebSocket for training progress ──

_training_watchers: list[WebSocket] = []


@app.websocket("/ws/training/{run_id}")
async def training_ws(websocket: WebSocket, run_id: str):
    """WebSocket endpoint for streaming training metrics updates."""
    await websocket.accept()
    _training_watchers.append(websocket)

    try:
        from tcip_mcp.tools.training_tools import check_training_status

        # Poll training status every 2 seconds and push updates
        last_epoch = -1
        while True:
            status = check_training_status(run_id)
            current_epoch = status.get("epoch", -1)

            if current_epoch > last_epoch:
                await websocket.send_json(status)
                last_epoch = current_epoch

            if status.get("status") in ("completed", "failed", "not_found"):
                await websocket.send_json(status)
                break

            await asyncio.sleep(2)

    except WebSocketDisconnect:
        pass
    finally:
        if websocket in _training_watchers:
            _training_watchers.remove(websocket)


# ── Frontend ──


@app.get("/")
def index():
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "TCIP Pipeline API", "docs": "/docs"}
