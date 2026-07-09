"""FastAPI application — REST API for MCP tools + WebSocket for GUI state sync.

This backend is the single source of truth for live GUI state across the
TCIP tabs (Annotate / Review / Training / Tuning / Inference / Results).
Claude agent and browser clients both connect through here.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    """Rehydrate persisted job/sweep/run registries so GUI history survives a restart.

    Their worker threads don't survive, so rehydrated non-terminal entries surface as
    'interrupted' — a record, not a resumable job (see ``jobstore``).
    """
    try:
        from tcip_web.routes import inference, training, tuning

        inference.rehydrate()
        tuning.rehydrate()
        training.rehydrate()
    except Exception:  # pragma: no cover - rehydrate is best-effort
        logger.exception("job registry rehydrate failed")
    yield


app = FastAPI(title="TCIP Pipeline", version="0.1.0", lifespan=_lifespan)

# CORS is not enabled by default — browser is expected to hit the same origin
# via the Vite dev proxy. If we ever serve the frontend elsewhere, add
# fastapi.middleware.cors.CORSMiddleware here.

# ── Tab routes ──
from tcip_web.routes import register_all as _register_routes  # noqa: E402  (needs `app`)
_register_routes(app)

# ── State snapshot + WS ──
from tcip_web.state import store as _gui_store  # noqa: E402  (needs `app`)
_state_watchers: set[WebSocket] = set()


async def _broadcast_state_snapshot(payload: dict[str, Any]) -> None:
    """Push the new state (with its version) to every connected browser."""
    msg = {"type": "state_snapshot", "state": payload["state"], "version": payload["version"]}
    dead: list[WebSocket] = []
    for ws in list(_state_watchers):
        try:
            await ws.send_json(msg)
        except Exception:
            dead.append(ws)
    for ws in dead:
        _state_watchers.discard(ws)


_gui_store.subscribe(_broadcast_state_snapshot)


@app.get("/api/state")
def get_state() -> dict:
    return _gui_store.snapshot()


@app.websocket("/ws/state")
async def state_ws(websocket: WebSocket) -> None:
    """Receive live GuiState deltas. Replays the current snapshot on connect."""
    await websocket.accept()
    _state_watchers.add(websocket)
    try:
        await websocket.send_json(
            {"type": "state_snapshot", "state": _gui_store.snapshot(), "version": _gui_store.version}
        )
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        _state_watchers.discard(websocket)


# ── Panel event hub (replaces .tcip/events/ file bridge) ──

VALID_PANELS = {"annotate", "review", "training", "tuning", "inference", "results"}

# Recent events per panel, kept in memory for replay on reconnect.
_recent_events: dict[str, deque[dict[str, Any]]] = defaultdict(lambda: deque(maxlen=64))
# Open WebSocket subscribers per panel.
_panel_subscribers: dict[str, set[WebSocket]] = defaultdict(set)


class PanelEvent(BaseModel):
    panel: str | None = None
    event_type: str
    data: dict[str, Any] = {}


async def _broadcast_to_panel(panel: str, event: dict[str, Any]) -> None:
    """Fan out an event to every WebSocket currently subscribed to a panel."""
    dead: list[WebSocket] = []
    for ws in list(_panel_subscribers.get(panel, ())):
        try:
            await ws.send_json(event)
        except Exception:
            dead.append(ws)
    for ws in dead:
        _panel_subscribers[panel].discard(ws)

# Serve static files (web frontend)
STATIC_DIR = Path(__file__).parent.parent.parent / "static"
if STATIC_DIR.exists():
    # The built frontend references absolute /assets/... paths (Vite's default
    # base="/"), so mount that subdirectory at /assets. Keep /static available
    # for any ad-hoc static resources.
    assets_dir = STATIC_DIR / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")
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


# ── Panel events: POST endpoint (MCP tools) + WS subscription (browsers) ──


@app.post("/api/events/{panel}")
async def post_panel_event(panel: str, event: PanelEvent):
    """Accept an event pushed from an MCP tool and broadcast to subscribers.

    Replaces the legacy ``.tcip/events/`` file-bridge. The payload schema
    matches the old file format: ``{panel, event_type, data}``.
    """
    if panel not in VALID_PANELS:
        return {"error": f"unknown panel: {panel}", "valid": sorted(VALID_PANELS)}
    payload = {
        "panel": panel,
        "event_type": event.event_type,
        "data": event.data,
    }
    _recent_events[panel].append(payload)
    await _broadcast_to_panel(panel, payload)
    return {"status": "ok", "panel": panel, "event_type": event.event_type}


@app.get("/api/events/{panel}/recent")
def get_recent_panel_events(panel: str, limit: int = 16):
    """Return the last N events for a panel (useful on browser reconnect)."""
    if panel not in VALID_PANELS:
        return {"error": f"unknown panel: {panel}", "valid": sorted(VALID_PANELS)}
    events = list(_recent_events.get(panel, ()))
    return {"panel": panel, "events": events[-limit:]}


@app.websocket("/ws/panel/{panel}")
async def panel_ws(websocket: WebSocket, panel: str):
    """Stream panel events to a browser client."""
    if panel not in VALID_PANELS:
        await websocket.close(code=1008, reason=f"unknown panel: {panel}")
        return
    await websocket.accept()
    _panel_subscribers[panel].add(websocket)
    # Replay recent events so late-joining browsers see the current state
    for event in list(_recent_events.get(panel, ())):
        try:
            await websocket.send_json(event)
        except Exception:
            break
    try:
        while True:
            # Keep the socket open; clients are read-only subscribers here.
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        _panel_subscribers[panel].discard(websocket)
