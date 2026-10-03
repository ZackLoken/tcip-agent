"""FastAPI application: the GUI state snapshot and its WebSocket, the panel-event hub, the built
frontend and the health probe; every domain route is mounted from ``tcip_web.routes``."""

from __future__ import annotations

import asyncio
import itertools
import logging
import uuid
from collections import defaultdict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from tcip_store.binding import bind_default

from tcip_mcp.audit import now_iso
from tcip_mcp.buckets import NotABucket
from tcip_mcp.identity import NoActor
from tcip_mcp.web_client import PANEL_EVENT_ANNOTATE_FOCUS, VALID_PANELS
from tcip_web.trust_boundary import TrustBoundaryMiddleware

logger = logging.getLogger(__name__)


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    """Open the project the last-opened pointer of the workspace the backend was configured with
    names, size GDAL's block cache and warm the agent terminal, before the app serves. Refuses a
    backend configured with no workspace (``StateStore.workspace``)."""
    from tcip_web.routes.projects import open_last_opened

    await open_last_opened()
    # Size GDAL's block cache once per process, at the entry point, never at source construction.
    from tcip_mcp.pipelines.raster_source import configure_gdal_cache

    configure_gdal_cache()
    # Warm the cold first-spawn cost (tcip_mcp import + PowerShell/.NET) off the request path.
    try:
        from tcip_web import terminal

        terminal.prewarm()
    except Exception:  # pragma: no cover - prewarm is best-effort
        logger.exception("agent terminal prewarm failed to start")
    yield
    # Kill live agent terminals off the loop: terminate can block seconds, and stalling the loop
    # would break in-flight WebSocket close handshakes.
    try:
        from tcip_web.routes import terminal as terminal_routes

        await asyncio.to_thread(terminal_routes.shutdown_all)
    except Exception:  # pragma: no cover - shutdown cleanup is best-effort
        logger.exception("agent terminal shutdown failed")


# At import, not in the lifespan: a route may be exercised against this app without one running,
# and a route that reaches a store with no backend bound would refuse rather than write.
bind_default()


app = FastAPI(title="TCIP Pipeline", version="0.1.0", lifespan=_lifespan)

# CORS is not enabled by default: the browser hits the same origin via the Vite dev proxy.
# Serving the frontend elsewhere would add fastapi.middleware.cors.CORSMiddleware here.

# A connection must arrive through this machine and name a loopback Host; the middleware also
# applies the Origin policy before a route runs (trust_boundary).
app.add_middleware(TrustBoundaryMiddleware)

# Compress JSON/text responses above ~1KB. A label document's payload scales with polygon count
# (dense images ship high-hundreds-of-KB to multi-MB uncompressed JSON).
app.add_middleware(GZipMiddleware, minimum_size=1000)

# ── Tab routes ──
from tcip_web.routes import register_all as _register_routes  # noqa: E402  (needs `app`)
_register_routes(app)

# ── State snapshot + WS ──
from tcip_web.state import (  # noqa: E402  (needs `app`)
    GuiMutationInvalid, NoProjectOpen, ProjectNotOpen, store as _gui_store,
)
_state_watchers: set[WebSocket] = set()


async def _bad_request_handler(_request: Request, exc: Exception) -> JSONResponse:
    """An invalid GUI mutation, a directory read as a bucket it is not, or an act naming no
    person: 400 with the reason."""
    return JSONResponse(status_code=400, content={"detail": str(exc)})


app.add_exception_handler(GuiMutationInvalid, _bad_request_handler)
app.add_exception_handler(NotABucket, _bad_request_handler)
app.add_exception_handler(NoActor, _bad_request_handler)


@app.exception_handler(NoProjectOpen)
async def _no_project_open_handler(_request: Request, exc: NoProjectOpen) -> JSONResponse:
    """Every route that acts on the open project answers 409 while none is open."""
    return JSONResponse(status_code=409, content={"detail": str(exc)})


@app.exception_handler(ProjectNotOpen)
async def _project_not_open_handler(_request: Request, exc: ProjectNotOpen) -> JSONResponse:
    """A request built for a project the backend does not have open answers 409 naming the one it
    has open."""
    return JSONResponse(status_code=409, content={"detail": {
        "error": str(exc), "open_project_id": exc.open_project_id}})


# This process's launch identity, minted once at import: rides every snapshot envelope so a
# restarted backend's lower-numbered first snapshot is accepted across a client's wsVersion guard.
SERVER_EPOCH = uuid.uuid4().hex


def state_snapshot_message(state: dict[str, Any], version: int) -> dict[str, Any]:
    """The state-snapshot envelope: state, version, the open project (``{id, path}``, or null
    when none is open) and this process's epoch."""
    root = _gui_store.project_root
    return {
        "type": "state_snapshot",
        "state": state,
        "version": version,
        "project": None if root is None else {"id": _gui_store.project_id, "path": str(root)},
        "epoch": SERVER_EPOCH,
    }


async def _broadcast_state_snapshot(payload: dict[str, Any]) -> None:
    """Push the new state, version and open project to every connected browser."""
    msg = state_snapshot_message(payload["state"], payload["version"])
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


class ActiveTabPayload(BaseModel):
    active_tab: str


@app.post("/api/state/tab")
async def set_active_tab(payload: ActiveTabPayload) -> dict:
    """Record which tab the browser is actually showing, so ``gui.json`` tracks what the human
    sees.
    """
    await _gui_store.mutate({"active_tab": payload.active_tab})
    return {"status": "ok", "active_tab": payload.active_tab}


@app.websocket("/ws/state")
async def state_ws(websocket: WebSocket) -> None:
    """Push live GuiState snapshots to the browser; replays the current snapshot on connect.
    One-directional: the client never sends a payload over this socket, and an inbound frame,
    if one ever arrived, is read and discarded, only to detect disconnect."""
    await websocket.accept()
    _state_watchers.add(websocket)
    try:
        await websocket.send_json(
            state_snapshot_message(_gui_store.snapshot(), _gui_store.version)
        )
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        _state_watchers.discard(websocket)


# ── Panel event hub ──

# Per-event identity; the timestamp keeps ids distinct across a restart, where the counter resets
# while browsers keep their dismissals.
_event_counter = itertools.count(1)
# Open WebSocket subscribers per panel.
_panel_subscribers: dict[str, set[WebSocket]] = defaultdict(set)


class PanelEvent(BaseModel):
    panel: str | None = None
    event_type: str
    data: dict[str, Any] = {}
    # The id of the project the sender acts on.
    project_id: str


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

def _static_dir_candidates() -> list[Path]:
    """The built frontend's install-layout candidates, in preference order: the packaged copy
    inside an installed wheel, then the src-layout checkout.
    """
    return [
        Path(__file__).parent / "static",
        Path(__file__).parent.parent.parent / "static",
    ]


def _find_static_dir() -> Path:
    """Locate the built frontend across install layouts.

    Prefers a copy packaged inside the installed package (``tcip_web/static/``, how a
    wheel should ship it) and falls back to the src-layout checkout
    (``packages/tcip-web/static/``). Returns the src-layout path if neither is built yet.
    """
    candidates = _static_dir_candidates()
    for c in candidates:
        if (c / "index.html").exists():
            return c
    return candidates[-1]


# Serve static files (web frontend)
STATIC_DIR = _find_static_dir()
if STATIC_DIR.exists():
    # The built frontend references absolute /assets/... paths (Vite's default base="/"), so
    # mount that subdirectory at /assets; /static stays for ad-hoc static resources.
    assets_dir = STATIC_DIR / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# The paths this app serves the built frontend and its health probe through, so the generated
# dev proxy never claims them and Vite keeps serving its own root, modules and HMR.
FRONTEND_SERVING_PATHS = frozenset({"/", "/health"})


# ── Health ──


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


# ── Frontend ──


@app.get("/")
def index():
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    # Fail loudly instead of silently serving a stub: the API works, but the GUI bundle
    # hasn't been built. Tell the operator exactly how to build it.
    return JSONResponse(
        status_code=503,
        content={
            "error": "GUI not built",
            "detail": (
                "The frontend bundle is missing. Build it, then reload: "
                "cd packages/tcip-web/frontend && npm install && npm run build"
            ),
            "api_docs": "/docs",
        },
    )


# ── Panel events: POST endpoint (MCP tools) + WS subscription (browsers) ──


@app.post("/api/events/{panel}")
async def post_panel_event(panel: str, event: PanelEvent, request: Request):
    """Accept an event pushed from an MCP tool and broadcast to subscribers.

    Payload shape: ``{panel, event_type, data, project_id}``. An event naming any project but the
    open one answers 409 with ``open_project_id`` (``StateStore.admit``) and is neither retained
    nor broadcast. The broadcast
    and replay payload, not this route's response, carries every agent identity field the sender
    declared in its headers (``agent_identity.HEADERS``, each ``None`` when not sent). Declared,
    not verified: any sender can set the headers.
    """
    from tcip_mcp import agent_identity

    if panel not in VALID_PANELS:
        return {"error": f"unknown panel: {panel}", "valid": sorted(VALID_PANELS)}
    _gui_store.admit(event.project_id)
    payload = {
        "panel": panel,
        "event_type": event.event_type,
        "data": event.data,
        "event_id": f"{now_iso()}#{next(_event_counter)}",
        **agent_identity.fields_from_headers(request.headers),
    }
    _gui_store.retain_event(panel, payload)
    # Agent focus events also update the advisory GuiState slice, so gui.json reflects where the
    # agent pointed the human: the browser applies the event locally and never syncs these back.
    if event.event_type == PANEL_EVENT_ANNOTATE_FOCUS:
        mutation: dict[str, Any] = {"active_tab": "annotate"}
        if "mode" in event.data:
            mutation["mode"] = event.data["mode"]
        if "active_subject" in event.data:
            mutation["active_subject"] = event.data["active_subject"]
        await _gui_store.mutate(mutation)
    await _broadcast_to_panel(panel, payload)
    return {"status": "ok", "panel": panel, "event_type": event.event_type}


@app.websocket("/ws/panel/{panel}")
async def panel_ws(websocket: WebSocket, panel: str):
    """Stream panel events to a browser client."""
    if panel not in VALID_PANELS:
        await websocket.close(code=1008, reason=f"unknown panel: {panel}")
        return
    await websocket.accept()
    _panel_subscribers[panel].add(websocket)
    # Replay the open project's retained events so a late-joining browser sees the current state.
    for event in _gui_store.retained_events(panel):
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
